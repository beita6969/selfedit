"""Weight-surgery engine for Qwen3.8-27B (Qwen3_5ForConditionalGeneration) inside a running vLLM instance.

Edits are point edits of real parameters, named with HF checkpoint names, applied in place to the tensors
the vLLM forward pass actually reads (fused / TP-sharded layouts resolved), reverted bit-exactly.
See README.md for the API, op list, fused-layout map and caveats.

Driver side: Surgeon (wraps vllm.LLM).  Worker side: module-level _w_* functions, sent with
LLM.collective_rpc (callables are cloudpickled by reference, so this directory must be on PYTHONPATH in the
worker processes; Surgeon sets PYTHONPATH and VLLM_ALLOW_INSECURE_SERIALIZATION=1 before creating the engine).
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
import shutil
import time
import warnings
from dataclasses import dataclass, field

import numpy as np

ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))
HF_PREFIX = "model.language_model."
V_PREFIX = "language_model.model."
VIS_HF = "model.visual."         # vision tower (multimodal=True only): HF names ...
VIS_V = "visual."                # ... and vLLM names (Qwen3VLForConditionalGeneration.hf_to_vllm_mapper)
IMAGE_LIMIT = 16                 # default images per prompt (ERQA has items with 16)
CHUNK_ELEMS = 1 << 23            # elements per GPU work chunk (bounds transient memory to ~100 MB)


# ======================================================================================================
# Config and fused / sharded layout map (pure python, shared by driver and workers)
# ======================================================================================================
class TextConfig:
    def __init__(self, model_dir):
        raw = json.load(open(os.path.join(model_dir, "config.json")))
        t = raw.get("text_config", raw)
        self.raw = raw
        self.H, self.L, self.I, self.V = t["hidden_size"], t["num_hidden_layers"], t["intermediate_size"], t["vocab_size"]
        self.layer_types = list(t["layer_types"])
        self.nq, self.nkv, self.hd = t["num_attention_heads"], t["num_key_value_heads"], t["head_dim"]
        self.gate = bool(t.get("attn_output_gate", True))
        self.nk, self.nv = t["linear_num_key_heads"], t["linear_num_value_heads"]
        self.dk, self.dv = t["linear_key_head_dim"], t["linear_value_head_dim"]
        self.kconv = t["linear_conv_kernel_dim"]
        self.Kd, self.Vd = self.nk * self.dk, self.nv * self.dv
        self.qrows = self.nq * self.hd * (2 if self.gate else 1)       # q_proj rows (q and output gate)


class VisionConfig:
    """Qwen3-VL vision tower: `depth` pre-norm ViT blocks (norm1 -> attn.qkv/proj, norm2 -> mlp.linear_fc1/fc2, LayerNorm
    with bias), then the patch merger (norm -> linear_fc1 -> GELU -> linear_fc2) mapping 2x2 merged patches into the
    language model's embedding space (out_hidden_size)."""

    def __init__(self, model_dir):
        v = json.load(open(os.path.join(model_dir, "config.json")))["vision_config"]
        self.depth, self.H, self.I, self.nh = v["depth"], v["hidden_size"], v["intermediate_size"], v["num_heads"]
        self.out, self.merge = v["out_hidden_size"], v["spatial_merge_size"]
        self.M = self.H * self.merge ** 2                       # merger width (2x2 patches concatenated)
        self.in_ch, self.tps, self.ps = v.get("in_channels", 3), v["temporal_patch_size"], v["patch_size"]
        self.npos = v["num_position_embeddings"]
        self.hd = self.H // self.nh
        self.deepstack = list(v.get("deepstack_visual_indexes") or [])


# ------------------------------------------------------------------------------------------------------
# Other model families (cross-family port).  Qwen3.8 keeps TextConfig / Layout above unchanged; the families below are
# selected from config.json "model_type" and map onto the parameters vLLM 0.33 builds for them:
#   gemma4: Gemma4ForConditionalGeneration (vllm/model_executor/models/gemma4_mm.py) -> Gemma4ForCausalLM (gemma4.py)
#   olmo3:  Olmo3ForCausalLM -> Transformers modeling backend (registry.py:763), fused by its fusers
# ------------------------------------------------------------------------------------------------------
def model_family(model_dir):
    raw = json.load(open(os.path.join(model_dir, "config.json")))
    mt = raw.get("model_type")
    if mt == "gemma4" or raw.get("text_config", {}).get("model_type") == "gemma4_text":
        return "gemma4"
    if mt == "olmo3":
        return "olmo3"
    return "qwen3_5"


def pad_vocab(v, pad=64):
    """vllm/model_executor/layers/vocab_parallel_embedding.py pad_vocab_size (DEFAULT_VOCAB_PADDING_SIZE = 64)."""
    return ((v + pad - 1) // pad) * pad


class Gemma4Config:
    """Gemma 4 dense text stack.  Sliding layers: head_dim / num_key_value_heads; full layers: global_head_dim /
    num_global_key_value_heads, and with attention_k_eq_v the full layers have no v_proj (V = K before v_norm).
    (transformers configuration_gemma4.py __post_init__ per_layer_config; vLLM gemma4.py:578-580, 86-127.)"""
    family = "gemma4"

    def __init__(self, model_dir):
        raw = json.load(open(os.path.join(model_dir, "config.json")))
        t = raw.get("text_config", raw)
        self.raw = raw
        self.H, self.L, self.I, self.V = t["hidden_size"], t["num_hidden_layers"], t["intermediate_size"], t["vocab_size"]
        self.layer_types = list(t["layer_types"])
        self.nq = t["num_attention_heads"]
        self.nkv, self.hd = t["num_key_value_heads"], t["head_dim"]
        self.k_eq_v = bool(t.get("attention_k_eq_v", False))
        self.hd_full = t.get("global_head_dim", 512)
        nkg = t.get("num_global_key_value_heads")
        self.nkv_full = nkg if (self.k_eq_v and nkg is not None) else self.nkv
        self.softcap = t.get("final_logit_softcapping")
        self.tie = bool(t.get("tie_word_embeddings", raw.get("tie_word_embeddings", True)))
        unsupported = dict(num_kv_shared_layers=t.get("num_kv_shared_layers", 0) or 0,
                           use_double_wide_mlp=bool(t.get("use_double_wide_mlp", False)),
                           num_experts=t.get("num_experts"), enable_moe_block=bool(t.get("enable_moe_block", False)),
                           hidden_size_per_layer_input=t.get("hidden_size_per_layer_input", 0) or 0)
        if any(unsupported.values()):
            raise NotImplementedError(f"Gemma 4 variant not mapped: {unsupported}")

    def attn(self, L):
        """(num_kv_heads, head_dim, k_eq_v) of layer L."""
        if self.layer_types[L] == "full_attention":
            return self.nkv_full, self.hd_full, self.k_eq_v
        return self.nkv, self.hd, False


class Olmo3Config:
    """OLMo 3 (post-norm: no pre-norms; q_norm / k_norm over the whole q / k projection width)."""
    family = "olmo3"

    def __init__(self, model_dir):
        raw = json.load(open(os.path.join(model_dir, "config.json")))
        t = raw.get("text_config", raw)
        self.raw = raw
        self.H, self.L, self.I, self.V = t["hidden_size"], t["num_hidden_layers"], t["intermediate_size"], t["vocab_size"]
        self.layer_types = list(t["layer_types"])
        self.nq, self.nkv = t["num_attention_heads"], t["num_key_value_heads"]
        self.hd = t.get("head_dim") or self.H // self.nq
        self.tie = bool(t.get("tie_word_embeddings", False))
        if self.tie:
            raise NotImplementedError("tied OLMo 3 embeddings not mapped")

    def attn(self, L):
        return self.nkv, self.hd, False


def text_config(model_dir):
    fam = model_family(model_dir)
    return {"gemma4": Gemma4Config, "olmo3": Olmo3Config}.get(fam, TextConfig)(model_dir)


class Layout:
    """HF tensor name -> where its values live in the vLLM model.

    spec = {hf, vname, kind, layer, shape (HF), shape2 (logical 1D/2D shape used for selections),
            dim (sharded logical dim: 0 rows, 1 cols, None replicated), segs [(hf_off, n, local_off)]}
    Along `dim`, HF indices [hf_off + r*n/T, hf_off + (r+1)*n/T) live on TP rank r at local indices
    [local_off, local_off + n/T) of parameter `vname`.  Other dims are identical.
    """

    def __init__(self, cfg: TextConfig, tp: int, hf_names, vision: VisionConfig | None = None, vision_tp=None):
        c, T = cfg, tp
        assert c.nkv % T == 0 and c.nk % T == 0 and c.nv % T == 0 and c.nq % T == 0, f"unsupported TP size {T}"
        self.cfg, self.tp = cfg, tp
        # vision=None (default): language model only, exactly as before. With a VisionConfig the vision tower's tensors
        # are mapped too; vision_tp = TP degree of the vision tower (1 when vLLM runs it data-parallel/replicated).
        self.vision, self.vtp = vision, (tp if vision_tp is None else vision_tp)
        if vision is not None:
            assert self.vtp in (1, tp) and vision.nh % self.vtp == 0, f"unsupported vision TP {self.vtp}"
        self.names = [n for n in hf_names if n.startswith(HF_PREFIX) or n == "lm_head.weight"]
        if vision is not None:
            self.names += [n for n in hf_names if n.startswith(VIS_HF)]
        self.specs = {n: self._spec(n) for n in self.names}
        # expected local shape of every vLLM parameter
        self.vshape, self.vdim = {}, {}
        for s in self.specs.values():
            vn, shp = s["vname"], list(s["shape"])
            self.vdim[vn] = s["dim"]
            if s["dim"] is None:
                self.vshape[vn] = shp
                continue
            ext = max(lo + n // T for _, n, lo in s["segs"])
            cur = self.vshape.get(vn)
            if cur is None:
                cur = shp[:]
                cur[s["dim"]] = ext
            else:
                cur[s["dim"]] = max(cur[s["dim"]], ext)
            self.vshape[vn] = cur

    def canon(self, name: str) -> str:
        n = name
        if self.vision is not None:
            for p in (VIS_HF, VIS_V):
                if n.startswith(p):
                    full = VIS_HF + n[len(p):]
                    if full in self.specs:
                        return full
                    if full + ".weight" in self.specs:
                        return full + ".weight"
                    raise KeyError(f"unknown vision parameter {name!r}")
        for p in (HF_PREFIX, V_PREFIX, "model."):
            if n.startswith(p):
                n = n[len(p):]
                break
        if n.startswith("language_model."):
            n = n[len("language_model."):]
        if n.startswith("lm_head"):
            return "lm_head.weight"
        if any(f in n for f in ("in_proj_qkvz", "in_proj_ba", "qkv_proj", "gate_up_proj")):
            raise KeyError(f"{name}: use HF names (in_proj_qkv/in_proj_z/in_proj_b/in_proj_a, q/k/v_proj, gate/up_proj)")
        full = HF_PREFIX + n
        if full in self.specs:
            return full
        if full + ".weight" in self.specs:
            return full + ".weight"
        raise KeyError(f"unknown HF parameter {name!r}")

    def _spec(self, hf):
        c, T = self.cfg, self.tp
        if hf.startswith(VIS_HF):
            return self._vspec(hf)
        if hf == "lm_head.weight":
            return dict(hf=hf, vname="language_model.lm_head.weight", kind="lm_head", layer=None,
                        shape=[c.V, c.H], shape2=[c.V, c.H], dim=0, segs=[(0, c.V, 0)])
        n = hf[len(HF_PREFIX):]
        if n == "embed_tokens.weight":
            return dict(hf=hf, vname=V_PREFIX + n, kind="embed_tokens", layer=None,
                        shape=[c.V, c.H], shape2=[c.V, c.H], dim=0, segs=[(0, c.V, 0)])
        if n == "norm.weight":
            return dict(hf=hf, vname=V_PREFIX + n, kind="final_norm", layer=None,
                        shape=[c.H], shape2=[c.H], dim=None, segs=[])
        m = re.fullmatch(r"layers\.(\d+)\.(.+)", n)
        if not m:
            raise KeyError(hf)
        i, rest = int(m[1]), m[2]
        b = f"{V_PREFIX}layers.{i}."
        Kd, Vd, H, I = c.Kd, c.Vd, c.H, c.I
        qkv_segs = [(0, Kd, 0), (Kd, Kd, Kd // T), (2 * Kd, Vd, 2 * Kd // T)]
        tab = {
            "input_layernorm.weight": ("input_layernorm.weight", [H], None, []),
            "post_attention_layernorm.weight": ("post_attention_layernorm.weight", [H], None, []),
            "mlp.gate_proj.weight": ("mlp.gate_up_proj.weight", [I, H], 0, [(0, I, 0)]),
            "mlp.up_proj.weight": ("mlp.gate_up_proj.weight", [I, H], 0, [(0, I, I // T)]),
            "mlp.down_proj.weight": ("mlp.down_proj.weight", [H, I], 1, [(0, I, 0)]),
            # Gated DeltaNet
            "linear_attn.in_proj_qkv.weight": ("linear_attn.in_proj_qkvz.weight", [2 * Kd + Vd, H], 0, qkv_segs),
            "linear_attn.in_proj_z.weight": ("linear_attn.in_proj_qkvz.weight", [Vd, H], 0, [(0, Vd, (2 * Kd + Vd) // T)]),
            "linear_attn.in_proj_b.weight": ("linear_attn.in_proj_ba.weight", [c.nv, H], 0, [(0, c.nv, 0)]),
            "linear_attn.in_proj_a.weight": ("linear_attn.in_proj_ba.weight", [c.nv, H], 0, [(0, c.nv, c.nv // T)]),
            "linear_attn.conv1d.weight": ("linear_attn.conv1d.weight", [2 * Kd + Vd, 1, c.kconv], 0, qkv_segs),
            "linear_attn.A_log": ("linear_attn.A_log", [c.nv], 0, [(0, c.nv, 0)]),
            "linear_attn.dt_bias": ("linear_attn.dt_bias", [c.nv], 0, [(0, c.nv, 0)]),
            "linear_attn.norm.weight": ("linear_attn.norm.weight", [c.dv], None, []),
            "linear_attn.out_proj.weight": ("linear_attn.out_proj.weight", [H, Vd], 1, [(0, Vd, 0)]),
            # gated full attention
            "self_attn.q_proj.weight": ("self_attn.qkv_proj.weight", [c.qrows, H], 0, [(0, c.qrows, 0)]),
            "self_attn.k_proj.weight": ("self_attn.qkv_proj.weight", [c.nkv * c.hd, H], 0,
                                        [(0, c.nkv * c.hd, c.qrows // T)]),
            "self_attn.v_proj.weight": ("self_attn.qkv_proj.weight", [c.nkv * c.hd, H], 0,
                                        [(0, c.nkv * c.hd, (c.qrows + c.nkv * c.hd) // T)]),
            "self_attn.o_proj.weight": ("self_attn.o_proj.weight", [H, c.nq * c.hd], 1, [(0, c.nq * c.hd, 0)]),
            "self_attn.q_norm.weight": ("self_attn.q_norm.weight", [c.hd], None, []),
            "self_attn.k_norm.weight": ("self_attn.k_norm.weight", [c.hd], None, []),
        }
        if rest not in tab:
            raise KeyError(hf)
        vsuf, shape, dim, segs = tab[rest]
        for _, nn_, _ in segs:
            assert nn_ % T == 0, (hf, nn_, T)
        shape2 = [shape[0], shape[2]] if len(shape) == 3 else shape
        kind = rest.replace(".weight", "").split(".")[-1]
        return dict(hf=hf, vname=b + vsuf, kind=kind, layer=i, shape=shape, shape2=shape2, dim=dim, segs=segs)

    def _vspec(self, hf):
        """Vision tower tensor.  vLLM keeps the HF names (prefix model.visual. -> visual.); attn.qkv is a
        QKVParallelLinear (q, k, v thirds sharded by heads), attn.proj / mlp.linear_fc2 / merger.linear_fc2 are
        RowParallelLinear (cols sharded, bias replicated), mlp.linear_fc1 / merger.linear_fc1 ColumnParallelLinear;
        norms, patch_embed (Conv3d) and pos_embed are replicated.  With vision_tp == 1 everything is replicated."""
        v, T = self.vision, self.tp
        if v is None:
            raise KeyError(hf)
        H, I, M, O = v.H, v.I, v.M, v.out
        n = hf[len(VIS_HF):]
        qkv = [(0, H, 0), (H, H, H // T), (2 * H, H, 2 * H // T)]
        m = re.fullmatch(r"blocks\.(\d+)\.(.+)", n)
        if m:
            i, rest = int(m[1]), m[2]
            if not 0 <= i < v.depth:
                raise KeyError(hf)
            tab = {"norm1.weight": ([H], None, [], "v_norm"), "norm1.bias": ([H], None, [], "v_norm_b"),
                   "norm2.weight": ([H], None, [], "v_norm"), "norm2.bias": ([H], None, [], "v_norm_b"),
                   "attn.qkv.weight": ([3 * H, H], 0, qkv, "v_qkv"), "attn.qkv.bias": ([3 * H], 0, qkv, "v_qkv_b"),
                   "attn.proj.weight": ([H, H], 1, [(0, H, 0)], "v_proj"), "attn.proj.bias": ([H], None, [], "v_proj_b"),
                   "mlp.linear_fc1.weight": ([I, H], 0, [(0, I, 0)], "v_fc1"),
                   "mlp.linear_fc1.bias": ([I], 0, [(0, I, 0)], "v_fc1_b"),
                   "mlp.linear_fc2.weight": ([H, I], 1, [(0, I, 0)], "v_fc2"),
                   "mlp.linear_fc2.bias": ([H], None, [], "v_fc2_b")}
        else:
            i, rest = None, n
            tab = {"merger.norm.weight": ([H], None, [], "m_norm"), "merger.norm.bias": ([H], None, [], "m_norm_b"),
                   "merger.linear_fc1.weight": ([M, M], 0, [(0, M, 0)], "m_fc1"),
                   "merger.linear_fc1.bias": ([M], 0, [(0, M, 0)], "m_fc1_b"),
                   "merger.linear_fc2.weight": ([O, M], 1, [(0, M, 0)], "m_fc2"),
                   "merger.linear_fc2.bias": ([O], None, [], "m_fc2_b"),
                   "patch_embed.proj.weight": ([H, v.in_ch, v.tps, v.ps, v.ps], None, [], "patch_embed"),
                   "patch_embed.proj.bias": ([H], None, [], "patch_embed_b"),
                   "pos_embed.weight": ([v.npos, H], None, [], "pos_embed")}
        if rest not in tab:
            raise KeyError(hf)
        shape, dim, segs, kind = tab[rest]
        if self.vtp == 1 and T > 1:
            dim, segs = None, []
        for _, nn_, _ in segs:
            assert nn_ % T == 0, (hf, nn_, T)
        shape2 = [shape[0], int(np.prod(shape[1:]))] if len(shape) > 2 else shape
        return dict(hf=hf, vname=VIS_V + n, kind=kind, layer=i, shape=shape, shape2=shape2, dim=dim, segs=segs,
                    vision=True)

    def _vhead_index(self, spec, heads, part):
        v, k = self.vision, spec["kind"]
        heads = [heads] if isinstance(heads, (int, np.integer)) else list(heads)
        for h in heads:
            if not 0 <= h < v.nh:
                raise IndexError(f"head {h} out of range [0,{v.nh}) for {spec['hf']}")

        def blocks(off):
            return np.concatenate([np.arange(off + h * v.hd, off + (h + 1) * v.hd) for h in heads])

        if k in ("v_qkv", "v_qkv_b"):
            if part not in ("q", "k", "v"):
                raise ValueError(f"{spec['hf']}: head selection needs part='q'|'k'|'v'")
            return 0, blocks({"q": 0, "k": v.H, "v": 2 * v.H}[part])
        if k == "v_proj":
            return 1, blocks(0)
        raise ValueError(f"{spec['hf']} ({k}) has no head structure")

    # ---------- semantic selections ----------
    def head_index(self, spec, heads, part=None):
        """HF indices (axis, np.array) of whole heads for this tensor kind."""
        if spec.get("vision"):
            return self._vhead_index(spec, heads, part)
        c, k = self.cfg, spec["kind"]
        heads = [heads] if isinstance(heads, (int, np.integer)) else list(heads)

        def blocks(off, size, nmax):
            for h in heads:
                if not 0 <= h < nmax:
                    raise IndexError(f"head {h} out of range [0,{nmax}) for {spec['hf']}")
            return np.concatenate([np.arange(off + h * size, off + (h + 1) * size) for h in heads])

        if k in ("A_log", "dt_bias", "in_proj_a", "in_proj_b"):
            return 0, blocks(0, 1, c.nv)
        if k == "in_proj_z":
            return 0, blocks(0, c.dv, c.nv)
        if k in ("in_proj_qkv", "conv1d"):
            if part == "q":
                return 0, blocks(0, c.dk, c.nk)
            if part == "k":
                return 0, blocks(c.Kd, c.dk, c.nk)
            if part == "v":
                return 0, blocks(2 * c.Kd, c.dv, c.nv)
            raise ValueError(f"{spec['hf']}: head selection needs part='q'|'k'|'v'")
        if k == "out_proj":
            return 1, blocks(0, c.dv, c.nv)
        if k == "q_proj":
            w = c.hd * (2 if c.gate else 1)
            if part in (None, "both"):
                return 0, blocks(0, w, c.nq)
            if part == "q":
                return 0, blocks(0, w, c.nq).reshape(len(heads), w)[:, :c.hd].reshape(-1)
            if part == "gate":
                assert c.gate
                return 0, blocks(0, w, c.nq).reshape(len(heads), w)[:, c.hd:].reshape(-1)
            raise ValueError("q_proj part must be 'q', 'gate' or None")
        if k in ("k_proj", "v_proj"):
            return 0, blocks(0, c.hd, c.nkv)
        if k == "o_proj":
            return 1, blocks(0, c.hd, c.nq)
        raise ValueError(f"{spec['hf']} ({k}) has no head structure")

    def neuron_index(self, spec, neurons):
        k = spec["kind"]
        a = np.atleast_1d(np.asarray(neurons, dtype=np.int64))
        if k in ("gate_proj", "up_proj"):
            return 0, a
        if k == "down_proj":
            return 1, a
        if k in ("v_fc1", "v_fc1_b", "m_fc1", "m_fc1_b"):      # vision MLP / merger hidden units
            return 0, a
        if k in ("v_fc2", "m_fc2"):
            return 1, a
        raise ValueError(f"{spec['hf']}: neuron selection only for mlp gate/up/down_proj")

    # ---------- cross-family hooks (Qwen: the behaviour Surgeon had inline; inert for existing edits) ----------
    family = "qwen3_5"
    buffers = ()           # vLLM buffer names that hold checkpoint tensors (mapped like parameters)
    aliases = {}           # vLLM parameter name -> name of the parameter it is tied to (same storage)

    def block_writer(self, L, part):
        """Short HF name of the matrix whose output is the residual write of block `part` of layer L."""
        if part == "attn":
            return (f"layers.{L}.linear_attn.out_proj.weight" if self.cfg.layer_types[L] == "linear_attention"
                    else f"layers.{L}.self_attn.o_proj.weight")
        if part == "mlp":
            return f"layers.{L}.mlp.down_proj.weight"
        raise ValueError("zero_block parts: 'attn', 'mlp'")

    def resid_gain(self, spec, side):
        """Residual-space direction u -> parameter-space direction normalize(g * u) (see Surgeon.expand, space='residual').
        Returns (short HF name of the norm gain g, form) where form is 'w' (output = rms_norm(x) * w) or '1+w'
        (Qwen3.5 RMSNorm: rms_norm(x) * (1 + w)), or (None, None) when the parameter reads / writes the raw stream."""
        k, L = spec["kind"], spec["layer"]
        if side == "out" and k in ("o_proj", "out_proj", "down_proj"):
            return None, None
        if side == "in" and k in ("q_proj", "k_proj", "v_proj", "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b"):
            return f"layers.{L}.input_layernorm.weight", "1+w"
        if side == "in" and k in ("gate_proj", "up_proj"):
            return f"layers.{L}.post_attention_layernorm.weight", "1+w"
        if side == "in" and k == "lm_head":
            return "norm.weight", "1+w"
        if side == "in" and k == "embed_tokens":
            return None, None
        raise ValueError(f"{spec['hf']}: side '{side}' does not touch the residual stream")

    def expected_gaps(self, vname, rank):
        """Local index ranges of vname on `rank` that hold no checkpoint value (vocab padding rows)."""
        return []


def _seg_m(sg, T):
    """Per-rank chunk of a segment: (hf_off, n, local_off) -> n/T; (hf_off, n, local_off, m) -> m (padded vocab: rank r
    holds HF [hf_off + r m, min(hf_off + n, hf_off + (r+1) m)) and the remaining local rows are padding)."""
    return sg[3] if len(sg) > 3 else sg[1] // T


class _FamilyLayout(Layout):
    """Common part of the non-Qwen layouts (language model only)."""
    hf_prefix = v_prefix = ""

    def __init__(self, cfg, tp, hf_names):
        T = tp
        self.cfg, self.tp = cfg, tp
        self.vision, self.vtp = None, tp
        for L in range(cfg.L):
            nkv = cfg.attn(L)[0]
            assert cfg.nq % T == 0 and nkv % T == 0 and cfg.I % T == 0, f"unsupported TP size {T}"
        self.names = [n for n in hf_names if self._is_lm(n)]
        self.specs = {n: self._spec(n) for n in self.names}
        self.vshape, self.vdim = {}, {}
        for s in self.specs.values():
            vn, shp = s["vname"], list(s["shape"])
            if vn in self.vdim and self.vdim[vn] != s["dim"]:
                raise AssertionError(f"{vn}: inconsistent sharded dim")
            self.vdim[vn] = s["dim"]
            if s["dim"] is None:
                self.vshape[vn] = shp
                continue
            ext = max(sg[2] + _seg_m(sg, T) for sg in s["segs"] + s.get("rsegs", []))
            cur = self.vshape.get(vn)
            if cur is None:
                cur = shp[:]
                cur[s["dim"]] = ext
            else:
                cur[s["dim"]] = max(cur[s["dim"]], ext)
            self.vshape[vn] = cur
        self.buffers = tuple(sorted(s["vname"] for s in self.specs.values() if s.get("buffer")))

    def _is_lm(self, n):
        return n.startswith(self.hf_prefix + "layers.") or n in (self.hf_prefix + "embed_tokens.weight",
                                                                 self.hf_prefix + "norm.weight", "lm_head.weight")

    def canon(self, name: str) -> str:
        n = name
        for p in sorted({self.hf_prefix, self.v_prefix, "model.language_model.", "language_model.model.", "model."},
                        key=len, reverse=True):
            if p and n.startswith(p):
                n = n[len(p):]
                break
        if n.startswith("language_model."):
            n = n[len("language_model."):]
        if any(f in n for f in ("qkv_proj", "gate_up_proj")):
            raise KeyError(f"{name}: use HF names (q/k/v_proj, gate/up_proj)")
        if n.startswith("lm_head"):
            if "lm_head.weight" in self.specs:
                return "lm_head.weight"
            raise KeyError(f"{name}: lm_head is tied to embed_tokens in this model; edit {self.hf_prefix}embed_tokens.weight")
        full = self.hf_prefix + n
        if full in self.specs:
            return full
        if full + ".weight" in self.specs:
            return full + ".weight"
        raise KeyError(f"unknown HF parameter {name!r}")

    def _vocab_spec(self, hf, vname, kind):
        c, T = self.cfg, self.tp
        m = pad_vocab(c.V) // T          # VocabParallelEmbedding: padded vocab split evenly, rank r -> [r m, (r+1) m)
        return dict(hf=hf, vname=vname, kind=kind, layer=None, shape=[c.V, c.H], shape2=[c.V, c.H], dim=0,
                    segs=[(0, c.V, 0, m)])

    def _layer_spec(self, hf, i, rest, tab):
        if rest not in tab:
            raise KeyError(hf)
        vsuf, shape, dim, segs = tab[rest]
        for sg in segs:
            assert sg[1] % self.tp == 0, (hf, sg, self.tp)
        kind = rest.replace(".weight", "").split(".")[-1]
        sp = dict(hf=hf, vname=f"{self.v_prefix}layers.{i}.{vsuf}", kind=kind, layer=i, shape=shape, shape2=shape,
                  dim=dim, segs=segs)
        if rest.startswith("self_attn."):
            sp["nkv"], sp["hd"] = self.cfg.attn(i)[:2]
        return sp

    def head_index(self, spec, heads, part=None):
        c, k = self.cfg, spec["kind"]
        if part not in (None, "both"):
            raise ValueError(f"{spec['hf']}: no head parts in this model (q/k/v are separate tensors)")
        heads = [heads] if isinstance(heads, (int, np.integer)) else list(heads)

        def blocks(size, nmax):
            for h in heads:
                if not 0 <= h < nmax:
                    raise IndexError(f"head {h} out of range [0,{nmax}) for {spec['hf']}")
            return np.concatenate([np.arange(h * size, (h + 1) * size) for h in heads])

        hd, nkv = spec.get("hd"), spec.get("nkv")
        if k == "q_proj":
            return 0, blocks(hd, c.nq)
        if k in ("k_proj", "v_proj"):
            return 0, blocks(hd, nkv)
        if k == "o_proj":
            return 1, blocks(hd, c.nq)
        if k == "q_norm" and spec["shape"][0] == c.nq * hd:      # OLMo 3: q_norm spans all heads
            return 0, blocks(hd, c.nq)
        if k == "k_norm" and spec["shape"][0] == nkv * hd:
            return 0, blocks(hd, nkv)
        raise ValueError(f"{spec['hf']} ({k}) has no head structure")

    def block_writer(self, L, part):
        if not 0 <= int(L) < self.cfg.L:
            raise IndexError(f"layer {L} out of range [0,{self.cfg.L})")
        if part == "attn":
            return f"layers.{L}.self_attn.o_proj.weight"
        if part == "mlp":
            return f"layers.{L}.mlp.down_proj.weight"
        raise ValueError("zero_block parts: 'attn', 'mlp'")

    def expected_gaps(self, vname, rank):
        out = []
        for s in self.specs.values():
            if s["vname"] == vname and s["dim"] is not None:
                for sg in s["segs"]:
                    if len(sg) > 3:
                        ho, n, lo, m = sg
                        cnt = max(0, min(ho + rank * m + m, ho + n) - (ho + rank * m))
                        if cnt < m:
                            out.append((lo + cnt, lo + m))
        return out


_GEMMA_VIS_WRITER = re.compile(r"model\.vision_tower\.encoder\.layers\.(\d+)\.post_(?:attention|feedforward)_layernorm\.weight")


class Gemma4Layout(_FamilyLayout):
    """Gemma 4 dense (31B) under vLLM 0.33, language model only.

    HF model.language_model.X -> vLLM language_model.model.X (gemma4_mm.py:1066-1079); per layer
    q/k/v_proj -> self_attn.qkv_proj (QKVParallelLinear, local [q | k | v], linear.py:1101-1115), with the full-attention
    k_eq_v layers loading k_proj into both the K and the V slot (gemma4.py:97-107, utils.py:158-163: rsegs below);
    gate/up_proj -> mlp.gate_up_proj (MergedColumnParallelLinear), down_proj / o_proj RowParallelLinear (cols sharded);
    q_norm / k_norm (per-head width, shared by heads), the 4 sandwich norms and the final norm replicated (plain
    x * w RMSNorm, gemma4.py:617-627); layer_scalar is a persistent buffer (gemma4.py:709, loaded by AutoWeightsLoader);
    embed_tokens VocabParallelEmbedding, lm_head tied to it (gemma4.py:1447-1454)."""
    family = "gemma4"
    hf_prefix, v_prefix = "model.language_model.", "language_model.model."

    def __init__(self, cfg, tp, hf_names, vision_H=None):
        self.vision_H = vision_H     # read during super().__init__
        super().__init__(cfg, tp, hf_names)
        self.aliases = {"language_model.lm_head.weight": self.v_prefix + "embed_tokens.weight"} if cfg.tie else {}

    def _is_lm(self, n):
        return super()._is_lm(n) or bool(self.vision_H and _GEMMA_VIS_WRITER.fullmatch(n))

    def canon(self, name):
        n = name[len("model."):] if name.startswith("model.") else name
        if n.startswith("vision_tower."):
            for x in ("model." + n, "model." + n + ".weight"):
                if x in self.specs:
                    return x
            raise KeyError(f"unknown HF parameter {name!r}")
        return super().canon(name)

    def _spec(self, hf):
        c, T = self.cfg, self.tp
        m = _GEMMA_VIS_WRITER.fullmatch(hf)
        if m:     # HF's own module in vLLM: replicated, named as in HF without the leading model.
            return dict(hf=hf, vname=hf[len("model."):], kind="vision_post_norm", layer=int(m[1]), shape=[self.vision_H], shape2=[self.vision_H], dim=None, segs=[])
        n = hf[len(self.hf_prefix):]
        if n == "embed_tokens.weight":
            return self._vocab_spec(hf, self.v_prefix + n, "embed_tokens")
        if n == "norm.weight":
            return dict(hf=hf, vname=self.v_prefix + n, kind="final_norm", layer=None, shape=[c.H], shape2=[c.H],
                        dim=None, segs=[])
        m = re.fullmatch(r"layers\.(\d+)\.(.+)", n)
        if not m or not 0 <= int(m[1]) < c.L:
            raise KeyError(hf)
        i, rest = int(m[1]), m[2]
        nkv, hd, kev = c.attn(i)
        Q, K, H, I = c.nq * hd, nkv * hd, c.H, c.I
        tab = {
            "input_layernorm.weight": ("input_layernorm.weight", [H], None, []),
            "post_attention_layernorm.weight": ("post_attention_layernorm.weight", [H], None, []),
            "pre_feedforward_layernorm.weight": ("pre_feedforward_layernorm.weight", [H], None, []),
            "post_feedforward_layernorm.weight": ("post_feedforward_layernorm.weight", [H], None, []),
            "layer_scalar": ("layer_scalar", [1], None, []),
            "mlp.gate_proj.weight": ("mlp.gate_up_proj.weight", [I, H], 0, [(0, I, 0)]),
            "mlp.up_proj.weight": ("mlp.gate_up_proj.weight", [I, H], 0, [(0, I, I // T)]),
            "mlp.down_proj.weight": ("mlp.down_proj.weight", [H, I], 1, [(0, I, 0)]),
            "self_attn.q_proj.weight": ("self_attn.qkv_proj.weight", [Q, H], 0, [(0, Q, 0)]),
            "self_attn.k_proj.weight": ("self_attn.qkv_proj.weight", [K, H], 0, [(0, K, Q // T)]),
            "self_attn.o_proj.weight": ("self_attn.o_proj.weight", [H, Q], 1, [(0, Q, 0)]),
            "self_attn.q_norm.weight": ("self_attn.q_norm.weight", [hd], None, []),
            "self_attn.k_norm.weight": ("self_attn.k_norm.weight", [hd], None, []),
        }
        if not kev:
            tab["self_attn.v_proj.weight"] = ("self_attn.qkv_proj.weight", [K, H], 0, [(0, K, (Q + K) // T)])
        sp = self._layer_spec(hf, i, rest, tab)
        if rest == "self_attn.k_proj.weight" and kev:
            sp["rsegs"] = [(0, K, (Q + K) // T)]      # the same K rows also fill the V slot
        if rest == "layer_scalar":
            sp["buffer"] = True
        return sp

    def resid_gain(self, spec, side):
        k, L = spec["kind"], spec["layer"]
        if side == "out" and k == "o_proj":
            return f"layers.{L}.post_attention_layernorm.weight", "w"
        if side == "out" and k == "down_proj":
            return f"layers.{L}.post_feedforward_layernorm.weight", "w"
        if side == "in" and k in ("q_proj", "k_proj", "v_proj"):
            return f"layers.{L}.input_layernorm.weight", "w"
        if side == "in" and k in ("gate_proj", "up_proj"):
            return f"layers.{L}.pre_feedforward_layernorm.weight", "w"
        if side == "in" and k == "embed_tokens":
            return None, None
        raise ValueError(f"{spec['hf']}: side '{side}' does not touch the residual stream")


class Olmo3Layout(_FamilyLayout):
    """OLMo 3 under vLLM 0.33's Transformers modeling backend (registry.py:763).  vLLM names = HF names (base model
    prefix 'model', transformers/base.py:310-359), with the fusers' merged modules: q/k/v_proj -> self_attn.qkv_proj
    (QKVParallelLinear, fusers/qkv.py update_attrs), gate/up_proj -> mlp.gate_up_proj (fusers/glu.py), o_proj / down_proj
    row-parallel; q_norm [nq*hd] / k_norm [nkv*hd] and the two post-norms replicated (TPAwareRMSNorm all-gathers the
    sharded q / k, fusers/rms_norm.py:108-125, 286-297); embed_tokens / lm_head VocabParallelEmbedding / ParallelLMHead
    with the vocab padded to a multiple of 64 (100278 -> 100288) and split evenly over ranks."""
    family = "olmo3"
    hf_prefix, v_prefix = "model.", "model."

    def _spec(self, hf):
        c, T = self.cfg, self.tp
        if hf == "lm_head.weight":
            return self._vocab_spec(hf, "lm_head.weight", "lm_head")
        n = hf[len(self.hf_prefix):]
        if n == "embed_tokens.weight":
            return self._vocab_spec(hf, self.v_prefix + n, "embed_tokens")
        if n == "norm.weight":
            return dict(hf=hf, vname=self.v_prefix + n, kind="final_norm", layer=None, shape=[c.H], shape2=[c.H],
                        dim=None, segs=[])
        m = re.fullmatch(r"layers\.(\d+)\.(.+)", n)
        if not m or not 0 <= int(m[1]) < c.L:
            raise KeyError(hf)
        i, rest = int(m[1]), m[2]
        Q, K, H, I = c.nq * c.hd, c.nkv * c.hd, c.H, c.I
        tab = {
            "post_attention_layernorm.weight": ("post_attention_layernorm.weight", [H], None, []),
            "post_feedforward_layernorm.weight": ("post_feedforward_layernorm.weight", [H], None, []),
            "mlp.gate_proj.weight": ("mlp.gate_up_proj.weight", [I, H], 0, [(0, I, 0)]),
            "mlp.up_proj.weight": ("mlp.gate_up_proj.weight", [I, H], 0, [(0, I, I // T)]),
            "mlp.down_proj.weight": ("mlp.down_proj.weight", [H, I], 1, [(0, I, 0)]),
            "self_attn.q_proj.weight": ("self_attn.qkv_proj.weight", [Q, H], 0, [(0, Q, 0)]),
            "self_attn.k_proj.weight": ("self_attn.qkv_proj.weight", [K, H], 0, [(0, K, Q // T)]),
            "self_attn.v_proj.weight": ("self_attn.qkv_proj.weight", [K, H], 0, [(0, K, (Q + K) // T)]),
            "self_attn.o_proj.weight": ("self_attn.o_proj.weight", [H, Q], 1, [(0, Q, 0)]),
            "self_attn.q_norm.weight": ("self_attn.q_norm.weight", [Q], None, []),
            "self_attn.k_norm.weight": ("self_attn.k_norm.weight", [K], None, []),
        }
        return self._layer_spec(hf, i, rest, tab)

    def resid_gain(self, spec, side):
        k, L = spec["kind"], spec["layer"]
        if side == "out" and k == "o_proj":
            return f"layers.{L}.post_attention_layernorm.weight", "w"
        if side == "out" and k == "down_proj":
            return f"layers.{L}.post_feedforward_layernorm.weight", "w"
        if side == "in" and k in ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "embed_tokens"):
            return None, None        # post-norm architecture: blocks read the raw residual stream
        if side == "in" and k == "lm_head":
            return "norm.weight", "w"
        raise ValueError(f"{spec['hf']}: side '{side}' does not touch the residual stream")


def make_layout(cfg, tp, hf_names, vision_H=None):
    fam = getattr(cfg, "family", "qwen3_5")
    if fam == "gemma4":
        return Gemma4Layout(cfg, tp, hf_names, vision_H)
    if fam == "olmo3":
        return Olmo3Layout(cfg, tp, hf_names)
    return Layout(cfg, tp, hf_names)


def _as_index(x, N):
    """Index spec -> sorted, unique np.int64 array (raises if a given list is not strictly increasing)."""
    if x is None:
        return np.arange(N, dtype=np.int64)
    if isinstance(x, (int, np.integer)):
        a = np.array([int(x)], dtype=np.int64)
    elif isinstance(x, str):
        parts = [int(p) if p.strip() else None for p in x.split(":")]
        a = np.arange(N, dtype=np.int64)[slice(*parts)]
    elif isinstance(x, slice):
        a = np.arange(N, dtype=np.int64)[x]
    elif isinstance(x, range):
        a = np.arange(x.start, x.stop, x.step, dtype=np.int64)
    elif isinstance(x, dict):
        a = np.arange(x.get("start", 0), x.get("stop", N), x.get("step", 1), dtype=np.int64)
    else:
        a = np.asarray(list(x) if not isinstance(x, np.ndarray) else x, dtype=np.int64).reshape(-1)
    a = np.where(a < 0, a + N, a)
    if a.size and (a.min() < 0 or a.max() >= N):
        raise IndexError(f"index out of range [0,{N})")
    if a.size > 1 and not np.all(np.diff(a) > 0):
        raise ValueError("index lists must be strictly increasing (values/u/v are aligned to ascending order)")
    return a


def resolve_selection(layout: Layout, spec, sel):
    """Selection spec -> (R, C) HF index arrays (C is None for 1-D tensors)."""
    shape2 = spec["shape2"]
    nd = len(shape2)
    rows = cols = None
    if sel is None:
        pass
    elif isinstance(sel, tuple):
        rows, cols = sel
    elif isinstance(sel, dict):
        sel = dict(sel)
        heads = sel.pop("head", sel.pop("heads", None))
        part = sel.pop("part", None)
        neurons = sel.pop("neuron", sel.pop("neurons", None))
        rows = sel.pop("rows", sel.pop("tokens", None))
        cols = sel.pop("cols", None)
        if sel:
            raise ValueError(f"unknown selection keys {list(sel)}")
        for idx_fn, arg in ((lambda: layout.head_index(spec, heads, part), heads),
                            (lambda: layout.neuron_index(spec, neurons), neurons)):
            if arg is None:
                continue
            axis, idx = idx_fn()
            if axis == 0:
                if rows is not None:
                    raise ValueError("both rows and head/neuron along dim 0")
                rows = idx
            else:
                if cols is not None:
                    raise ValueError("both cols and head/neuron along dim 1")
                cols = idx
    else:
        rows = sel
    R = _as_index(rows, shape2[0])
    if nd == 1:
        if cols is not None:
            raise ValueError(f"{spec['hf']} is 1-D")
        return R, None
    return R, _as_index(cols, shape2[1])


# ======================================================================================================
# Local (per TP rank) plan: which local rows/cols of the vLLM parameter hold the selected HF indices
# ======================================================================================================
def _runs(loc, pos):
    """Split parallel index arrays into maximal runs where both increase by exactly 1."""
    if loc.size == 0:
        return []
    brk = np.nonzero((np.diff(loc) != 1) | (np.diff(pos) != 1))[0] + 1
    starts = np.concatenate([[0], brk])
    ends = np.concatenate([brk, [loc.size]])
    return [(int(loc[s]), int(loc[e - 1]) + 1, int(pos[s])) for s, e in zip(starts, ends)]


def _map_axis(sel, segs, rank, T):
    """HF indices `sel` along a sharded axis -> (local indices, positions in sel) held by `rank`."""
    locs, poss = [], []
    for sg in segs:
        ho, n, lo = sg[0], sg[1], sg[2]
        m = _seg_m(sg, T)
        a = ho + rank * m
        i0, i1 = np.searchsorted(sel, a), np.searchsorted(sel, min(a + m, ho + n))
        if i1 > i0:
            locs.append(lo + sel[i0:i1] - a)
            poss.append(np.arange(i0, i1))
    if not locs:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    loc, pos = np.concatenate(locs), np.concatenate(poss)
    o = np.argsort(loc, kind="stable")
    return loc[o], pos[o]


def local_plan(spec, rank, T, R, C):
    """-> (row_runs [(lr0, lr1, p0)], (col_loc, col_pos) or None)"""
    dim = spec["dim"]
    if dim == 0:
        lr, pr = _map_axis(R, spec["segs"], rank, T)
    else:
        lr, pr = R, np.arange(R.size)
    col = None
    if C is not None:
        if dim == 1:
            col = _map_axis(C, spec["segs"], rank, T)
        else:
            col = (C, np.arange(C.size))
    return _runs(lr, pr), col


# ======================================================================================================
# Worker side (runs inside each vLLM worker process; called through LLM.collective_rpc)
# ======================================================================================================
_WS = {"params": None, "backups": {}, "rank": 0, "tp": 1}


def _w_dispatch(worker, fname, blob):
    """Single RPC entry point: arguments and results travel as pickle blobs (bytes are msgpack-native)."""
    out = globals()[fname](worker, *pickle.loads(blob))
    return pickle.dumps(out, protocol=pickle.HIGHEST_PROTOCOL)


def _w_unwrap(worker):
    w = getattr(worker, "worker", worker)
    m = w.get_model()
    return m.unwrap() if hasattr(m, "unwrap") else m


def _w_params(worker):
    if _WS["params"] is None:
        from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
        model = _w_unwrap(worker)
        _WS["params"] = dict(model.named_parameters(remove_duplicate=False))
        _WS["rank"] = get_tensor_model_parallel_rank()
        _WS["tp"] = get_tensor_model_parallel_world_size()
    return _WS["params"]


def _p2(p):
    d = p.data
    return d.view(d.shape[0], -1) if d.dim() >= 3 else d       # conv1d (3-D) / vision patch_embed (5-D)


def _int_view(t):
    import torch
    return t.view({1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[t.element_size()])


def _bits_equal(a, b):
    import torch
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(_int_view(a.contiguous()), _int_view(b.contiguous()))


def _iter_blocks(P2, plan):
    """Yield (a, b, p0, csel, cpos) chunks: local rows [a,b) <-> selection positions [p0, p0+b-a);
    csel selects local cols (slice or index tensor), cpos = (GPU tensor, numpy) positions of those cols in C."""
    import torch
    runs, col = plan
    if col is None:
        csel = cpos = None
        ncol = 1
    else:
        cl, cp = col
        ncol = max(1, cl.size)
        if cl.size == 0:
            return
        if cl[-1] - cl[0] + 1 == cl.size:
            csel = slice(int(cl[0]), int(cl[-1]) + 1)
        else:
            csel = torch.as_tensor(cl, device=P2.device)
        cpos = (torch.as_tensor(cp, device=P2.device), cp)
    step = max(1, CHUNK_ELEMS // ncol)
    for lr0, lr1, p0 in runs:
        for a in range(lr0, lr1, step):
            b = min(lr1, a + step)
            yield a, b, p0 + (a - lr0), csel, cpos


def _blk_read(P2, a, b, csel):
    v = P2[a:b]
    if csel is None:
        return v
    return v[:, csel] if isinstance(csel, slice) else v.index_select(1, csel)


def _blk_write(P2, a, b, csel, new):
    v = P2[a:b]
    if csel is None:
        v.copy_(new)
    elif isinstance(csel, slice):
        v[:, csel].copy_(new)
    else:
        v.index_copy_(1, csel, new)


def _bcast(val, x, p0, cpos, nR, nC):
    """Slice a value (scalar / array broadcastable to the (|R|,|C|) or (|R|,) selection) for this chunk.
    Slicing happens on the CPU; only the chunk's slice is uploaded."""
    import torch
    if isinstance(val, (int, float)):
        return float(val)
    t = np.asarray(val, dtype=np.float32)
    nr = x.shape[0]
    if x.dim() == 1:
        if t.ndim == 1:
            assert t.shape == (nR,), f"value shape {t.shape} vs selection ({nR},)"
            t = t[p0:p0 + nr]
        else:
            assert t.ndim == 0, f"value shape {t.shape} vs selection ({nR},)"
    elif t.ndim == 1:
        assert t.shape[0] == nC, f"1-D value of length {t.shape[0]} must match selected cols {nC}"
        t = t[cpos[1]][None, :]
    elif t.ndim == 2:
        assert t.shape[0] in (1, nR) and t.shape[1] in (1, nC), f"value shape {t.shape} not broadcastable to ({nR},{nC})"
        if t.shape[0] == nR and nR > 1:
            t = t[p0:p0 + nr]
        if t.shape[1] == nC and nC > 1:
            t = t[:, cpos[1]]
    else:
        assert t.ndim == 0, f"value shape {t.shape}"
    return torch.as_tensor(np.ascontiguousarray(t), device=x.device)


def _compute(prim, blk, p0, cpos):
    """New values (param dtype) for one chunk.  Unchanged elements keep their exact bits."""
    import torch
    x = blk.float()
    nR, nC = len(prim["R"]), (len(prim["C"]) if prim["C"] is not None else None)
    op = prim["op"]
    if op == "scale":
        new = x * _bcast(prim["alpha"], x, p0, cpos, nR, nC)
    elif op == "set":
        v = _bcast(prim["value"], x, p0, cpos, nR, nC)
        new = torch.full_like(x, v) if isinstance(v, float) else torch.broadcast_to(v, x.shape).clone()
    elif op in ("add", "rank1"):
        if op == "add":
            d = _bcast(prim["value"], x, p0, cpos, nR, nC)
            d = torch.full_like(x, d) if isinstance(d, float) else torch.broadcast_to(d, x.shape)
        else:
            u = torch.as_tensor(np.ascontiguousarray(prim["u"][p0:p0 + x.shape[0]]), device=x.device)
            v = torch.as_tensor(np.ascontiguousarray(prim["v"][cpos[1]]), device=x.device)
            d = u[:, None] * v[None, :]
        new = torch.where(d == 0, x, x + d)        # exact no-op where the update is exactly 0 (keeps -0.0 bits)
    else:
        raise ValueError(op)
    return new.to(blk.dtype)


def _w_init(worker, vshape, buffers=()):
    """Check that every mapped vLLM parameter exists with the expected local shape.  `buffers`: names of module buffers
    that hold checkpoint values (e.g. Gemma 4 layer_scalar); they join the parameter dict and are edited like parameters."""
    P = _w_params(worker)
    if buffers:
        B = dict(_w_unwrap(worker).named_buffers(remove_duplicate=False))
        for b in buffers:
            if b in B and b not in P:
                P[b] = B[b]
    bad = {}
    for vn, shp in vshape.items():
        if vn not in P:
            bad[vn] = "missing"
        elif list(P[vn].shape) != list(shp):
            bad[vn] = f"shape {list(P[vn].shape)} != expected {list(shp)}"
    dtypes = {}
    for vn, p in P.items():
        key = re.sub(r"(layers|blocks)\.\d+\.", r"\1.N.", vn)
        dtypes[key] = str(p.dtype).replace("torch.", "")
    # unmapped names that are the very same tensor object as a mapped one (tied weights; modules registered under two
    # paths, e.g. vLLM Gemma4Model's self_decoder.decoder_layers = self.layers[:k], gemma4.py:1108-1112)
    ids = {id(P[vn]): vn for vn in vshape if vn in P}
    aliases = {n: ids[id(P[n])] for n in sorted(set(P) - set(vshape)) if id(P[n]) in ids}
    return dict(rank=_WS["rank"], tp=_WS["tp"], bad=bad, dtypes=dtypes, n_params=len(P),
                unmapped=sorted(set(P) - set(vshape)), aliases=aliases)


def _w_read(worker, prim):
    import torch
    P = _w_params(worker)
    p = P[prim["vname"]]
    P2 = _p2(p)
    plan = local_plan(prim["spec"], _WS["rank"], _WS["tp"], prim["R"], prim["C"])
    out = []
    with torch.no_grad():
        for a, b, p0, csel, cpos in _iter_blocks(P2, plan):
            out.append((p0, _blk_read(P2, a, b, csel).to("cpu", copy=True)))
    cp = None if plan[1] is None else plan[1][1]
    return dict(rank=_WS["rank"], blocks=out, cpos=cp)


def _w_stat(worker, prim):
    """Partial statistic for directional ops, in HF selection coordinates (zero-filled where not local), accumulated
    in fp64 (the driver sums ranks in fp64 and rounds once to fp32, so the result is ~TP-invariant).
    side 'out': s = u^T W (length |C|).  side 'in': s = W v (length |R|)."""
    import torch
    P = _w_params(worker)
    P2 = _p2(P[prim["vname"]])
    plan = local_plan(prim["spec"], _WS["rank"], _WS["tp"], prim["R"], prim["C"])
    side = prim["side"]
    nR, nC = len(prim["R"]), len(prim["C"])
    with torch.no_grad():
        dev = P2.device
        f64 = torch.float64
        if side == "out":
            u = torch.as_tensor(prim["dir"], device=dev).to(f64)
            s = torch.zeros(nC, dtype=f64, device=dev)
            for a, b, p0, csel, cpos in _iter_blocks(P2, plan):
                s.index_add_(0, cpos[0], u[p0:p0 + (b - a)] @ _blk_read(P2, a, b, csel).to(f64))
        else:
            v = torch.as_tensor(prim["dir"], device=dev).to(f64)
            s = torch.zeros(nR, dtype=f64, device=dev)
            for a, b, p0, csel, cpos in _iter_blocks(P2, plan):
                s[p0:p0 + (b - a)] = _blk_read(P2, a, b, csel).to(f64) @ v.index_select(0, cpos[0])
    return dict(rank=_WS["rank"], stat=s.cpu().numpy())


def _w_apply(worker, hid, prim):
    import torch
    P = _w_params(worker)
    p = P[prim["vname"]]
    P2 = _p2(p)
    plan = local_plan(prim["spec"], _WS["rank"], _WS["tp"], prim["R"], prim["C"])
    bk = _WS["backups"].setdefault(hid, [])
    changed = nelem = 0
    maxd = 0.0
    with torch.no_grad():
        for a, b, p0, csel, cpos in _iter_blocks(P2, plan):
            blk = _blk_read(P2, a, b, csel)
            orig = blk.to("cpu", copy=True)
            cdesc = csel if (csel is None or isinstance(csel, slice)) else csel.cpu()
            bk.append((prim["vname"], a, b, cdesc, orig))
            new = _compute(prim, blk, p0, cpos)
            ob = blk.clone()                        # blk may be a view of the parameter: copy before writing
            _blk_write(P2, a, b, csel, new)
            ne = (_int_view(new) != _int_view(ob)).sum().item()
            changed += ne
            nelem += new.numel()
            if ne:
                maxd = max(maxd, (new.float() - ob.float()).abs().max().item())
    torch.cuda.synchronize()
    return dict(rank=_WS["rank"], changed=changed, nelem=nelem, maxd=maxd)


def _w_revert(worker, hid):
    import torch
    P = _w_params(worker)
    ok, n = True, 0
    with torch.no_grad():
        for vname, a, b, cdesc, orig in reversed(_WS["backups"].get(hid, [])):
            P2 = _p2(P[vname])
            csel = cdesc if (cdesc is None or isinstance(cdesc, slice)) else cdesc.to(P2.device)
            _blk_write(P2, a, b, csel, orig.to(P2.device))
            back = _blk_read(P2, a, b, csel).to("cpu")
            ok = ok and _bits_equal(back, orig)
            n += 1
    torch.cuda.synchronize()
    _WS["backups"].pop(hid, None)
    return dict(rank=_WS["rank"], ok=ok, blocks=n)


def _w_mem(worker, reset):
    """CUDA memory of the worker process (bytes); optionally reset the peak counter afterwards."""
    import torch
    free, total = torch.cuda.mem_get_info()
    d = dict(rank=_WS.get("rank", 0), max_allocated=torch.cuda.max_memory_allocated(),
             allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved(),
             device_free=free, device_total=total)
    if reset:
        torch.cuda.reset_peak_memory_stats()
    return d


def _fp_tensor(t):
    """Position-weighted exact integer fingerprint of a tensor's bits (GPU, chunked)."""
    import torch
    x = _int_view(t.detach().reshape(-1))
    s1 = torch.zeros((), dtype=torch.int64, device=x.device)
    s2 = torch.zeros((), dtype=torch.int64, device=x.device)
    CH = 1 << 22
    for a in range(0, x.numel(), CH):
        c = x[a:a + CH].to(torch.int64)
        w = torch.arange(a, a + c.numel(), device=x.device, dtype=torch.int64) % 65521 + 1
        s1 += c.sum()
        s2 += (c * w).sum()
    return int(s1), int(s2)


def _w_fingerprint(worker, names=None):
    import torch
    P = _w_params(worker)
    with torch.no_grad():
        out = {n: _fp_tensor(P[n]) for n in (names if names is not None else sorted(P))}
    return dict(rank=_WS["rank"], fp=out)


def _w_verify_layout(worker, specs, model_dir, weight_map, vshape):
    """Bitwise compare every mapped local slice with the checkpoint tensor; coverage / overlap / aliasing."""
    import torch
    from safetensors import safe_open
    P = _w_params(worker)
    rank, T = _WS["rank"], _WS["tp"]
    by_file = {}
    for hf in specs:
        by_file.setdefault(weight_map[hf], []).append(hf)
    mism, nchk, nel = [], 0, 0
    cover = {vn: [] for vn in vshape}
    for f, names in sorted(by_file.items()):
        with safe_open(os.path.join(model_dir, f), "pt", device="cpu") as fh:
            for hf in names:
                sp = specs[hf]
                ck = fh.get_tensor(hf)
                p = P[sp["vname"]]
                ck2 = ck.view(ck.shape[0], -1) if ck.dim() >= 3 else ck
                P2 = _p2(p)
                if sp["dim"] is None:
                    pairs = [(P2, ck2)]
                    cover[sp["vname"]].append((0, P2.shape[0]))
                else:
                    pairs = []
                    for sg in sp["segs"] + sp.get("rsegs", []):      # rsegs: replicas of the same HF rows
                        ho, n, lo = sg[0], sg[1], sg[2]
                        m = _seg_m(sg, T)
                        a = ho + rank * m
                        m = max(0, min(a + m, ho + n) - a)            # == n/T unless the vocab is padded
                        if sp["dim"] == 0:
                            pairs.append((P2[lo:lo + m], ck2[a:a + m]))
                        else:
                            pairs.append((P2[:, lo:lo + m], ck2[:, a:a + m]))
                        cover[sp["vname"]].append((lo, lo + m))
                for loc, ref in pairs:
                    lc = loc.to("cpu")
                    rf = ref.to(lc.dtype)
                    if not _bits_equal(lc.contiguous(), rf.contiguous()):
                        d = (lc.float() - ref.float()).abs()
                        mism.append(dict(hf=hf, max_abs=float(d.max()), n_diff=int((d != 0).sum())))
                    nchk += 1
                    nel += lc.numel()
    cov_bad = {}
    vdim = {sp["vname"]: sp["dim"] for sp in specs.values()}
    for vn, iv in cover.items():
        dim = vdim[vn]
        ext = vshape[vn][dim if dim is not None else 0]
        iv = sorted(iv)
        pos, gaps, over = 0, [], []
        for a, b in iv:
            if a > pos:
                gaps.append((pos, a))
            if a < pos:
                over.append((a, pos))
            pos = max(pos, b)
        if pos < ext:
            gaps.append((pos, ext))
        if gaps or over:
            cov_bad[vn] = dict(gaps=gaps, overlaps=over)
    # storage aliasing between distinct parameters
    rngs = sorted((p.data.data_ptr(), p.data.data_ptr() + p.numel() * p.element_size(), n) for n, p in P.items())
    alias = [(rngs[k][2], rngs[k + 1][2]) for k in range(len(rngs) - 1) if rngs[k + 1][0] < rngs[k][1]]
    return dict(rank=rank, checked=nchk, elements=nel, mismatches=mism, coverage_bad=cov_bad, aliasing=alias)


def _w_scan(worker):
    """List every tensor held by model modules that is not a registered parameter (derived-tensor audit)."""
    import torch
    model = _w_unwrap(worker)
    pids = {id(p) for p in model.parameters()}
    rows = {}

    def add(kind, name, t):
        key = re.sub(r"(layers|blocks)\.\d+\.", r"\1.N.", name)
        r = rows.setdefault(key, dict(kind=kind, count=0, shape=list(t.shape), dtype=str(t.dtype), bytes=0))
        r["count"] += 1
        r["bytes"] += t.numel() * t.element_size()

    for mname, mod in model.named_modules():
        for bname, buf in mod.named_buffers(recurse=False):
            add("buffer", f"{mname}.{bname}", buf)
        for k, v in vars(mod).items():
            if k in ("_parameters", "_buffers", "_modules"):
                continue
            if isinstance(v, torch.Tensor) and id(v) not in pids:
                add("attr", f"{mname}.{k}", v)
            elif isinstance(v, (list, tuple)):
                for q, e in enumerate(v):
                    if isinstance(e, torch.Tensor) and id(e) not in pids:
                        add("attr-list", f"{mname}.{k}[{q}]", e)
    return dict(rank=_WS["rank"], tensors=rows)


# ======================================================================================================
# Driver side
# ======================================================================================================
@dataclass
class Handle:
    id: int
    edit: object
    prims: list
    touched: list
    fp_before: dict | None
    info: list = field(default_factory=list)
    reverted: bool = False

    @property
    def changed(self):
        return sum(i["changed"] for i in self.info)


DIRECTIONAL = ("project_out", "amplify")
_SEL_KEYS = ("head", "heads", "part", "neuron", "neurons", "rows", "cols", "tokens")
_ALL_OPS = ("scale", "add", "set", "rank1", "project_out", "amplify", "zero_block")


class Surgeon:
    """In-place weight surgery on an offline vllm.LLM serving Qwen3.8-27B (language model only by default;
    multimodal=True loads the vision tower too, maps its tensors and accepts image prompts)."""

    def __init__(self, model_dir, tp=1, max_model_len=16384, max_num_seqs=32, max_num_batched_tokens=8192,
                 gpu_memory_utilization=None, enable_prefix_caching=False, seed=0, enforce_eager=False,
                 inproc_engine=True, log=print, multimodal=False, limit_mm_per_prompt=None, **engine_kwargs):
        # inproc_engine: run vLLM's EngineCore in this process (VLLM_ENABLE_V1_MULTIPROCESSING=0). With the default
        # background EngineCore process, LLM.generate() streams requests over ZMQ while the core's busy loop already
        # schedules the first ones (core.py _process_input_queue drains only what has arrived), so the first prefill
        # batch composition is a race. Different batch shapes pick different GEMM kernels, and greedy logprobs of
        # bit-identical weights then differ by ~1e-4 between calls. In-process, all requests are queued before the
        # first step, so the schedule is a deterministic function of the inputs.
        if inproc_engine:
            os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
        pp = os.environ.get("PYTHONPATH", "")
        if ENGINE_DIR not in pp.split(":"):
            os.environ["PYTHONPATH"] = ENGINE_DIR + (":" + pp if pp else "")
        if gpu_memory_utilization is None:
            gpu_memory_utilization = round(float(os.environ["GPU_MEM_FRACTION"]) - 0.05, 4)
        self.log = log
        self.model_dir = model_dir
        self.tp = tp
        self.cfg = text_config(model_dir)
        self.family = getattr(self.cfg, "family", "qwen3_5")
        self.weight_map = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
        self.multimodal = bool(multimodal)
        if self.multimodal and self.family not in ("qwen3_5", "gemma4"):
            raise NotImplementedError(f"multimodal=True is available for Qwen3.8 and Gemma 4 only (family {self.family})")
        if self.family != "qwen3_5":
            self.vcfg = None
            self.layout = make_layout(self.cfg, tp, self.weight_map, json.load(open(os.path.join(model_dir, "config.json")))["vision_config"]["hidden_size"] if self.multimodal and self.family == "gemma4" else None)
            # Gemma 4 is a multimodal checkpoint: language_model_only=True sets every modality limit to 0
            # (config/multimodal.py get_limit_per_prompt), so _mark_tower_model skips the vision / audio towers
            # (interfaces.py:347-366, gemma4_mm.py:1101-1121).  OLMo 3 is text-only (no multimodal config).
            mm_kw = dict(language_model_only=True) if self.family == "gemma4" else {}
            if self.multimodal:                                  # Gemma 4 reads images; only its text stack is mapped for edits so far
                self.image_limit = int((limit_mm_per_prompt or {}).get("image", IMAGE_LIMIT))
                mm_kw = dict(limit_mm_per_prompt=dict(limit_mm_per_prompt or {"image": IMAGE_LIMIT, "video": 0, "audio": 0}))
                self.image_token_id = json.load(open(os.path.join(model_dir, "config.json")))["image_token_id"]
        elif self.multimodal:
            self.vcfg = VisionConfig(model_dir)
            if self.vcfg.deepstack:
                raise NotImplementedError("deepstack vision features are not mapped")
            vtp = 1 if engine_kwargs.get("mm_encoder_tp_mode") == "data" else tp
            self.layout = Layout(self.cfg, tp, self.weight_map, vision=self.vcfg, vision_tp=vtp)
            self.image_limit = int((limit_mm_per_prompt or {}).get("image", IMAGE_LIMIT))
            mm_kw = dict(limit_mm_per_prompt=dict(limit_mm_per_prompt or {"image": IMAGE_LIMIT, "video": 0}))
            self.image_token_id = json.load(open(os.path.join(model_dir, "config.json")))["image_token_id"]
        else:
            self.vcfg = None
            self.layout = Layout(self.cfg, tp, self.weight_map)
            mm_kw = dict(language_model_only=True)
        self.prefix_caching = enable_prefix_caching
        from vllm import LLM
        t0 = time.time()
        self.llm = LLM(model=model_dir, tensor_parallel_size=tp, max_model_len=max_model_len,
                       gpu_memory_utilization=gpu_memory_utilization, max_num_seqs=max_num_seqs,
                       max_num_batched_tokens=max_num_batched_tokens, enable_prefix_caching=enable_prefix_caching,
                       seed=seed, enforce_eager=enforce_eager, **mm_kw, **engine_kwargs)
        self.load_seconds = time.time() - t0
        self.max_model_len = max_model_len
        self.max_num_batched_tokens = max_num_batched_tokens
        self.engine_args = dict(tp=tp, max_model_len=max_model_len, max_num_seqs=max_num_seqs,
                                max_num_batched_tokens=max_num_batched_tokens,
                                gpu_memory_utilization=gpu_memory_utilization,
                                enable_prefix_caching=enable_prefix_caching, enforce_eager=enforce_eager,
                                inproc_engine=inproc_engine, **engine_kwargs)
        if self.multimodal:
            self.engine_args.update(multimodal=True, **mm_kw)
        if self.layout.buffers:
            init = self._rpc(_w_init, self.layout.vshape, list(self.layout.buffers))
        else:
            init = self._rpc(_w_init, self.layout.vshape)
        for r in init:
            if r["bad"]:
                raise RuntimeError(f"rank {r['rank']}: vLLM parameter layout differs from expectation: {r['bad']}")
        self.param_dtypes = init[0]["dtypes"]
        self.unmapped_params = init[0]["unmapped"]
        # parameters that are the same tensor object as a mapped one (declared ties + found at init); edits reach them too
        self.alias_params = dict(self.layout.aliases, **init[0].get("aliases", {}))
        self.unmapped_real = [n for n in self.unmapped_params if n not in self.alias_params]
        self.tokenizer = self.llm.get_tokenizer()
        if self.family == "gemma4":
            # Gemma 4 thinking block: <|channel>thought\n ... <channel|> answer (vllm/reasoning/gemma4_utils.py)
            self.think_start_id = self.tokenizer.convert_tokens_to_ids("<|channel>")
            self.think_end_id = self.tokenizer.convert_tokens_to_ids("<channel|>")
        elif self.family == "olmo3":
            # OLMo 3 Think: the template opens <think> in the prompt; </think> is plain text (3 tokens)
            self.think_start_id = self.think_end_id = None
        else:
            self.think_end_id = self.tokenizer.convert_tokens_to_ids("</think>")
        self._active: list[Handle] = []
        self._next = 1
        self.default_sampling = {}
        try:
            d = self.llm.get_default_sampling_params()
            for k in ("temperature", "top_p", "top_k", "min_p", "repetition_penalty", "presence_penalty",
                      "frequency_penalty"):
                self.default_sampling[k] = getattr(d, k)
        except Exception:
            pass
        log(f"[surgery] engine ready in {self.load_seconds:.1f}s (tp={tp}); unmapped vLLM params: "
            f"{self.unmapped_params if self.family == 'qwen3_5' else self.unmapped_real}"
            + (f"; {len(self.alias_params)} aliases of mapped params" if self.alias_params else ""))

    # ---------------- plumbing ----------------
    def _rpc(self, fn, *args):
        blob = pickle.dumps(args, protocol=pickle.HIGHEST_PROTOCOL)
        res = [pickle.loads(r) for r in self.llm.collective_rpc(_w_dispatch, args=(fn.__name__, blob))]
        return sorted(res, key=lambda r: r["rank"])

    def spec(self, name):
        return self.layout.specs[self.layout.canon(name)]

    def _prim_base(self, name, sel):
        hf = self.layout.canon(name)
        sp = self.layout.specs[hf]
        R, C = resolve_selection(self.layout, sp, sel)
        return dict(hf=hf, vname=sp["vname"], spec=sp, R=R, C=C)

    # ---------------- read ----------------
    def read(self, hf_name, index=None):
        """CPU copy of an HF-named parameter (vLLM's runtime dtype), or of a selection of it.
        Whole tensor -> HF shape; selection -> [|R|, |C|] (or [|R|] for 1-D tensors)."""
        import torch
        prim = self._prim_base(hf_name, index)
        res = self._rpc(_w_read, prim)
        sp = prim["spec"]
        if sp["dim"] is None:
            res = res[:1]
        nR = prim["R"].size
        nC = None if prim["C"] is None else prim["C"].size
        out = None
        for r in res:
            for p0, blk in r["blocks"]:
                if out is None:
                    out = torch.empty((nR,) if nC is None else (nR, nC), dtype=blk.dtype)
                if nC is None:
                    out[p0:p0 + blk.shape[0]] = blk
                else:
                    out[p0:p0 + blk.shape[0], torch.as_tensor(r["cpos"])] = blk
        if index is None and len(sp["shape"]) >= 3:
            out = out.view(*sp["shape"])
        return out

    def read_replica(self, hf_name, index=None):
        """Like read(), but from the replica slot of a tensor vLLM holds twice (spec 'rsegs'; Gemma 4 k_eq_v k_proj in
        the V slot of qkv_proj).  For tests: it must always equal read()."""
        import torch
        prim = self._prim_base(hf_name, index)
        sp = prim["spec"]
        if not sp.get("rsegs"):
            raise ValueError(f"{prim['hf']} has no replica")
        prim = dict(prim, spec=dict(sp, segs=sp["rsegs"]))
        res = self._rpc(_w_read, prim)
        nR, nC = prim["R"].size, (None if prim["C"] is None else prim["C"].size)
        out = None
        for r in res:
            for p0, blk in r["blocks"]:
                if out is None:
                    out = torch.empty((nR,) if nC is None else (nR, nC), dtype=blk.dtype)
                if nC is None:
                    out[p0:p0 + blk.shape[0]] = blk
                else:
                    out[p0:p0 + blk.shape[0], torch.as_tensor(r["cpos"])] = blk
        return out

    def sha256(self, hf_name, index=None):
        t = self.read(hf_name, index).contiguous()
        return hashlib.sha256(_int_view(t).numpy().tobytes()).hexdigest()

    # ---------------- edits ----------------
    def expand(self, edit):
        """Edit (dict or list of dicts) -> list of primitive ops with resolved HF index sets."""
        if isinstance(edit, (list, tuple)):
            return [p for e in edit for p in self.expand(e)]
        e = dict(edit)
        op = e.get("op")
        if op not in _ALL_OPS:
            raise ValueError(f"unknown op {op!r}; ops: {_ALL_OPS}")
        if op == "zero_block":
            L = int(e["layer"])
            parts = e.get("parts", ["attn", "mlp"])
            parts = [parts] if isinstance(parts, str) else parts
            subs = []
            for part in parts:
                nm = self.layout.block_writer(L, part)
                subs.append({"op": "set", "param": nm, "value": 0.0})
            return self.expand(subs)
        sel = e.get("sel", e.get("index"))
        if sel is None:
            top = {k: e[k] for k in _SEL_KEYS if k in e}
            sel = top or None
        prim = self._prim_base(e["param"], sel)
        prim["op"] = op
        nR = prim["R"].size
        nC = None if prim["C"] is None else prim["C"].size
        if op == "scale":
            prim["alpha"] = _num(e.get("alpha", 1.0))
        elif op in ("add", "set"):
            prim["value"] = _num(e["value"])
        elif op == "rank1":
            if nC is None:
                raise ValueError("rank1 needs a 2-D tensor")
            u, v = _vec(e["u"], nR, "u"), _vec(e["v"], nC, "v")
            prim["u"] = (np.float32(e.get("alpha", 1.0)) * u).astype(np.float32)
            prim["v"] = v
        else:  # directional
            if nC is None:
                raise ValueError(f"{op} needs a 2-D tensor")
            side = e.get("side", "out")
            if side not in ("out", "in"):
                raise ValueError("side must be 'out' (W <- (I+a uu^T) W) or 'in' (W <- W (I+a vv^T))")
            d = np.asarray(e["u"] if side == "out" else e.get("v", e.get("u")), dtype=np.float64).reshape(-1)
            if d.size != (nR if side == "out" else nC):
                raise ValueError(f"direction length {d.size} != selected {'rows' if side == 'out' else 'cols'}")
            space = e.get("space", "param")
            if space == "residual":
                # d is a residual-stream direction u.  The tensor reads / writes the stream through an RMSNorm with
                # gain g (writer: out = g * rms_norm(W a); reader: W (g * rms_norm(x))), so the parameter-space
                # direction is v = normalize(g * u): project_out makes the write exactly orthogonal to u (resp. the read
                # independent of the stream's u-component), amplify scales that component by (1+alpha) before the norm.
                axis = prim["R"] if side == "out" else prim["C"]
                if axis.size != self.cfg.H:
                    raise ValueError("space='residual' needs the full hidden axis selected on that side")
                gname, form = self.layout.resid_gain(prim["spec"], side)
                prim["gain"] = gname
                if gname is not None:
                    g = self.read(gname).float().numpy().astype(np.float64)
                    d = d * (1.0 + g if form == "1+w" else g)
            elif space != "param":
                raise ValueError("space must be 'param' or 'residual'")
            nrm = np.linalg.norm(d)
            if nrm == 0:
                raise ValueError("zero direction")
            prim["side"] = side
            prim["dir"] = (d / nrm).astype(np.float32)
            prim["alpha"] = -1.0 if op == "project_out" else float(e.get("alpha", 1.0))
        prim["edit"] = {k: (v if not isinstance(v, (list, np.ndarray)) or np.size(v) <= 8 else f"<{np.size(v)} values>")
                        for k, v in e.items()}
        return [prim]

    def _materialize(self, prim):
        """Directional op -> rank-1 update with the cross-rank statistic computed on current weights."""
        if prim["op"] not in DIRECTIONAL:
            return prim
        res = self._rpc(_w_stat, prim)
        s = res[0]["stat"].astype(np.float64).copy()
        if prim["spec"]["dim"] is not None:
            for r in res[1:]:
                s += r["stat"]
        s = s.astype(np.float32)
        a = np.float32(prim["alpha"])
        q = dict(prim)
        q["op"] = "rank1"
        if prim["side"] == "out":                 # W += a u (u^T W)
            q["u"], q["v"] = (a * prim["dir"]).astype(np.float32), s
        else:                                     # W += a (W v) v^T
            q["u"], q["v"] = (a * s).astype(np.float32), prim["dir"]
        q["stat"] = s
        return q

    def apply(self, edit, verify=True):
        """Apply an edit in place; returns a Handle for revert().  Warns if a non-identity op changed no bits."""
        prims = self.expand(edit)
        hid = self._next
        self._next += 1
        touched = sorted({p["vname"] for p in prims})
        fp0 = self._fingerprint(touched) if verify else None
        h = Handle(hid, edit, prims, touched, fp0)
        try:
            for p in prims:
                q = self._materialize(p)
                res = self._rpc(_w_apply, hid, q)
                if p["spec"]["dim"] is None:
                    res = res[:1]
                if p["spec"].get("rsegs"):            # same HF values held twice (Gemma 4 k_eq_v: K and V slot)
                    q2 = dict(q, spec=dict(q["spec"], segs=q["spec"]["rsegs"]))
                    res = res + self._rpc(_w_apply, hid, q2)
                info = dict(hf=p["hf"], op=p["op"], changed=sum(r["changed"] for r in res),
                            nelem=sum(r["nelem"] for r in res), maxd=max(r["maxd"] for r in res))
                h.info.append(info)
                if info["changed"] == 0 and not _is_identity(p):
                    warnings.warn(f"edit {p['edit']} changed 0 elements of {p['hf']} (update below bf16 resolution?)")
        except Exception:
            self._rpc(_w_revert, hid)
            raise
        self._active.append(h)
        self._after_weight_change(touched)
        return h

    def revert(self, h: Handle, verify=True):
        """Restore the exact original bits (LIFO order).  Raises if any restored block or fingerprint differs."""
        if not self._active or self._active[-1] is not h:
            raise RuntimeError("revert handles in LIFO order (edits may overlap)")
        res = self._rpc(_w_revert, h.id)
        ok = all(r["ok"] for r in res)
        rep = dict(blocks_bit_exact=ok, blocks=sum(r["blocks"] for r in res))
        if verify and h.fp_before is not None:
            fp1 = self._fingerprint(h.touched)
            rep["fingerprint_equal"] = fp1 == h.fp_before
            ok = ok and rep["fingerprint_equal"]
        self._active.pop()
        h.reverted = True
        self._after_weight_change(h.touched)
        if not ok:
            raise RuntimeError(f"revert of handle {h.id} NOT exact: {rep}")
        return rep

    def revert_all(self):
        while self._active:
            self.revert(self._active[-1])

    def edited(self, edit):
        """Context manager: with surgeon.edited(edit) as h: ..."""
        s = self

        class _Ctx:
            def __enter__(self_):
                self_.h = s.apply(edit)
                return self_.h

            def __exit__(self_, *a):
                s.revert(self_.h)
        return _Ctx()

    def _after_weight_change(self, touched=None):
        if self.prefix_caching:
            self.llm.reset_prefix_cache()      # cached KV / SSM states were computed with other weights
        if self.multimodal and (touched is None or any(v.startswith((VIS_V, "vision_tower.")) for v in touched)):
            self.reset_encoder_cache()

    def reset_encoder_cache(self):
        """Drop cached vision-encoder outputs.  vLLM V1 keeps finished requests' encoder outputs as reusable entries
        keyed by image hash (EncoderCacheManager.freeable), so after a vision-tower edit or revert a later request with
        the same image would otherwise reuse embeddings computed with the old weights.  Language-model edits do not
        change encoder outputs, so they keep the cache."""
        self.llm.llm_engine.engine_core.reset_encoder_cache()

    def _fingerprint(self, vnames=None):
        res = self._rpc(_w_fingerprint, vnames)
        return {(r["rank"], n): v for r in res for n, v in r["fp"].items()}

    def model_fingerprint(self):
        """Exact fingerprint of every vLLM parameter on every rank."""
        return self._fingerprint(None)

    def verify_layout(self, names=None):
        """Bitwise check of the HF->vLLM map for every mapped tensor on every rank (reads the checkpoint).
        names: optional subset of HF names (coverage is then checked for the vLLM parameters they map to)."""
        if names is None:
            return self._rpc(_w_verify_layout, self.layout.specs, self.model_dir, self.weight_map, self.layout.vshape)
        specs = {n: self.layout.specs[self.layout.canon(n)] for n in names}
        vns = {s["vname"] for s in specs.values()}
        return self._rpc(_w_verify_layout, specs, self.model_dir, self.weight_map,
                         {k: v for k, v in self.layout.vshape.items() if k in vns})

    def scan_tensors(self):
        return self._rpc(_w_scan)

    # ---------------- generation ----------------
    def render(self, prompt, enable_thinking=None, reasoning_effort=None, chat_template_kwargs=None, system=None):
        """The exact prompt string the chat template produces (for inspection)."""
        ctk = self._ctk(enable_thinking, reasoning_effort, chat_template_kwargs)
        return self.tokenizer.apply_chat_template(self._conv(prompt, system), tokenize=False,
                                                  add_generation_prompt=True, **ctk)

    @staticmethod
    def _conv(p, system):
        if isinstance(p, str):
            conv = [{"role": "user", "content": p}]
        else:
            conv = list(p)
        if system is not None and conv[0]["role"] != "system":
            conv = [{"role": "system", "content": system}] + conv
        return conv

    @staticmethod
    def _ctk(enable_thinking, reasoning_effort, chat_template_kwargs):
        """Same rule as vLLM's OpenAI server (ChatCompletionRequest.build_chat_params)."""
        ctk = dict(chat_template_kwargs or {})
        if reasoning_effort is not None:
            ctk["reasoning_effort"] = reasoning_effort
            if "enable_thinking" not in ctk and enable_thinking is None:
                ctk["enable_thinking"] = reasoning_effort != "none"
        if enable_thinking is not None:
            ctk["enable_thinking"] = bool(enable_thinking)
        return ctk

    def generate(self, prompts, n=1, sampling=None, max_tokens=4096, enable_thinking=None, reasoning_effort=None,
                 chat_template_kwargs=None, system=None, logprobs=None, use_tqdm=False, seeds=None, on_output=None):
        """Chat-templated generation.  prompts: list of str or of message lists; with multimodal=True a message's content
        may be a list of OpenAI-style parts ({"type": "image_url", "image_url": {"url": "data:..."}} / {"type": "text",
        "text": ...}, the format gen_mm.py sends; see image_message()), or a prompt may be an item dict
        {"prompt": text, "image_paths": [...]} (converted with image_message, "<image>" placeholders interleave).
        sampling: SamplingParams kwargs (defaults: the model's generation_config).  Returns one dict per prompt:
        {prompt_tokens, prompt_token_ids, samples: [{text, reasoning, content, think_closed, num_tokens, finish_reason,
        stop_reason, token_ids, [logprobs, top_logprobs]}]}.  Reasoning/answer split at the first </think> token.
        prompt_token_ids + token_ids is the exact sequence the model saw, ready for score().
        seeds: one int per prompt (vLLM gives sample k seed + k): two models sampled with the same seeds share their
        random numbers, so their outputs stay identical until an edit changes a token (common random numbers)."""
        from vllm import SamplingParams
        kw = dict(self.default_sampling)
        kw.update(sampling or {})
        if seeds is None:
            sp = SamplingParams(n=n, max_tokens=max_tokens, logprobs=logprobs, **kw)
        else:
            assert len(seeds) == len(prompts)
            sp = [SamplingParams(n=n, max_tokens=max_tokens, logprobs=logprobs, seed=int(s), **kw) for s in seeds]
        ctk = self._ctk(enable_thinking, reasoning_effort, chat_template_kwargs)
        thinking = ctk.get("enable_thinking", True)
        if self.family == "gemma4":
            thinking = bool(ctk.get("enable_thinking", False))   # Gemma 4 template default: thinking off
        elif self.family == "olmo3":
            thinking = True                                       # OLMo 3 Think: always reasons (template opens <think>)
        prompts = [image_message(p["prompt"], p.get("image_paths") or []) if isinstance(p, dict) else p for p in prompts]
        convs = [self._conv(p, system) for p in prompts]
        if self.multimodal:
            for i, c in enumerate(convs):
                k = sum(_n_images(m) for m in c)
                if k > self.image_limit:
                    raise ValueError(f"prompt {i}: {k} images > limit_mm_per_prompt {self.image_limit}")
        def one(o):
            samples = []
            for c in o.outputs:
                ids = list(c.token_ids)
                s = dict(text=c.text, num_tokens=len(ids), finish_reason=c.finish_reason, stop_reason=c.stop_reason,
                         token_ids=ids)
                if self.family != "qwen3_5":
                    s.update(self._split_thinking(ids, c.text, thinking))
                elif thinking:
                    if self.think_end_id in ids:
                        k = ids.index(self.think_end_id)
                        s["reasoning"] = self.tokenizer.decode(ids[:k], skip_special_tokens=True).strip()
                        s["content"] = self.tokenizer.decode(ids[k + 1:], skip_special_tokens=True).strip()
                        s["think_closed"] = True
                    else:
                        s["reasoning"] = self.tokenizer.decode(ids, skip_special_tokens=True).strip()
                        s["content"] = ""
                        s["think_closed"] = False
                else:
                    s["reasoning"], s["content"], s["think_closed"] = "", c.text.strip(), None
                if logprobs is not None and c.logprobs is not None:
                    s["logprobs"] = [d[t].logprob for d, t in zip(c.logprobs, ids)]
                    s["top_logprobs"] = [{int(k): v.logprob for k, v in d.items()} for d in c.logprobs]
                samples.append(s)
            return dict(prompt_tokens=len(o.prompt_token_ids), prompt_token_ids=list(o.prompt_token_ids), samples=samples)
        eng, orig_step = self.llm.llm_engine, None
        if on_output is not None:      # hands each prompt's result over as it finishes, without splitting the batch (a split makes every part wait for its slowest draw)
            base, orig_step = self.llm.request_counter.counter, eng.step

            def watched():
                got = orig_step()
                for x in got:
                    if x.finished:
                        try:
                            on_output(int(x.request_id.split("-")[0]) - base, one(x))
                        except Exception as e:
                            self.log(f"[surgery] on_output failed: {e!r}")
                return got
            eng.step = watched
        t0 = time.time()
        try:
            outs = self.llm.chat(convs, sp, use_tqdm=use_tqdm, chat_template_kwargs=ctk)
        finally:
            if orig_step is not None:
                eng.step = orig_step
        dt = time.time() - t0
        res = [one(o) for o in outs]
        ntok = sum(x["num_tokens"] for r in res for x in r["samples"])
        self.last_generate = dict(seconds=dt, generated_tokens=ntok, tok_per_s=ntok / max(dt, 1e-9),
                                  prompts=len(prompts), n=n)
        return res

    def _split_thinking(self, ids, text, thinking):
        """Gemma 4 / OLMo 3: reasoning / content split.  Adds think_end = index of the first token of the closing marker
        in token_ids (None if the thinking block never closed or there was none), so masks can use exact token spans.
        Gemma 4 (vllm/reasoning/gemma4_utils.py parse_thinking_output): <|channel>thought\n R <channel|> A; without any
        channel token the whole output is the answer (a leading 'thought\n' label is stripped).
        OLMo 3 (vllm/reasoning/olmo3_reasoning_parser.py): R </think> A, the prompt already ends with <think>."""
        dec = lambda t: self.tokenizer.decode(t, skip_special_tokens=True)

        def unlabel(x):
            x = x.lstrip()
            if x == "thought" or x.startswith("thought\n"):
                x = x[len("thought"):]
            return x.strip()

        if self.family == "gemma4":
            if self.think_end_id in ids:
                k = ids.index(self.think_end_id)
                return dict(reasoning=unlabel(dec(ids[:k])), content=dec(ids[k + 1:]).strip(), think_closed=True,
                            think_end=k)
            if thinking and self.think_start_id in ids:
                return dict(reasoning=unlabel(dec(ids)), content="", think_closed=False, think_end=None)
            return dict(reasoning="", content=unlabel(dec(ids)), think_closed=None if not thinking else True,
                        think_end=None)
        # olmo3
        i = text.find("</think>")
        if i < 0:
            return dict(reasoning=text.strip(), content="", think_closed=False, think_end=None)
        return dict(reasoning=text[:i].strip(), content=text[i + len("</think>"):].strip(), think_closed=True,
                    think_end=self._text_token_index(ids, len(text[:i])))

    def _text_token_index(self, ids, nchar):
        """Smallest k such that decode(ids[:k+1]) is longer than nchar characters (the token holding character nchar)."""
        lo, hi = 0, len(ids)
        while lo < hi:
            mid = (lo + hi) // 2
            if len(self.tokenizer.decode(ids[:mid + 1], skip_special_tokens=True)) > nchar:
                hi = mid
            else:
                lo = mid + 1
        return lo

    # ---------------- scoring (teacher forcing, no generation) ----------------
    def score(self, seqs, start=None, use_tqdm=False, final_via_sampler=None, images=None):
        """Log-probabilities of given token sequences under the current (edited) weights, without generating.

        seqs:  list of token-id lists, e.g. r["prompt_token_ids"] + s["token_ids"] from generate().
        start: first scored position (int for all, or one per sequence; default 1).
        Returns one dict per sequence: {logprobs: float32[len - start], sum, mean, n_tokens, start}, where
        logprobs[k] = log p(seq[start+k] | seq[:start+k]) -- raw logprobs (no temperature / penalties), the same
        quantity generate(..., logprobs=0) reports for sampled tokens.

        Implementation: vLLM prompt_logprobs=0, max_tokens=1, detokenize=False (one prefill per sequence).  vLLM
        refuses prompts of max_model_len tokens (no room for the 1 output token), so a sequence of exactly
        max_model_len tokens is sent as prompt seq[:-1] and its last token is forced with allowed_token_ids; the
        sampler takes raw logprobs before that mask, so the forced token's logprob is the unmasked one
        (final_via_sampler=True forces this path for any length, for testing).
        images (multimodal=True): one entry per sequence, None or the list of images (paths / bytes / PIL) of the prompt
        that produced it, in order.  Their expanded <|image_pad|> runs are collapsed to one token each and the images are
        sent as multi_modal_data, so vLLM re-expands them to the identical token sequence (checked below) and the
        prefill sees the same vision embeddings as generation did.
        Memory: prompt logprobs materialise [chunk, vocab] bf16 logits + fp32 log-softmax per request and step
        (~1.42 MiB per token at vocab 248320), outside vLLM's profiled budget; chunk <= max_num_batched_tokens."""
        from vllm import SamplingParams
        seqs = [[int(t) for t in s] for s in seqs]
        if start is None:
            starts = [1] * len(seqs)
        elif isinstance(start, (int, np.integer)):
            starts = [int(start)] * len(seqs)
        else:
            starts = [int(x) for x in start]
        if len(starts) != len(seqs):
            raise ValueError(f"{len(starts)} start positions for {len(seqs)} sequences")
        if images is not None:
            if not self.multimodal:
                raise ValueError("score(images=...) needs Surgeon(multimodal=True)")
            if len(images) != len(seqs):
                raise ValueError(f"{len(images)} image lists for {len(seqs)} sequences")
        L = self.max_model_len
        est = self.max_num_batched_tokens * self.cfg.V * 6
        if est > 4 * 2 ** 30 and not getattr(self, "_score_mem_warned", False):
            self._score_mem_warned = True
            warnings.warn(f"score(): prompt logprobs may need ~{est / 2 ** 30:.1f} GiB beyond vLLM's budget per step "
                          f"(max_num_batched_tokens={self.max_num_batched_tokens}); use <= 2048 on tight GPU slices")
        prompts, params, via = [], [], []
        for i, (q, st) in enumerate(zip(seqs, starts)):
            if len(q) < 2:
                raise ValueError(f"sequence {i}: needs at least 2 tokens")
            if len(q) > L:
                raise ValueError(f"sequence {i}: {len(q)} tokens > max_model_len={L}")
            if not 1 <= st <= len(q):
                raise ValueError(f"sequence {i}: start={st} outside [1, {len(q)}]")
            v = (len(q) == L) if final_via_sampler is None else bool(final_via_sampler)
            imgs = images[i] if images is not None else None
            if v:
                prompts.append(self._token_prompt(q[:-1], imgs))
                params.append(SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0, logprobs=0,
                                             allowed_token_ids=[q[-1]], detokenize=False))
            else:
                prompts.append(self._token_prompt(q, imgs))
                params.append(SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0, detokenize=False))
            via.append(v)
        t0 = time.time()
        outs = self.llm.generate(prompts, params, use_tqdm=use_tqdm)
        dt = time.time() - t0
        res = []
        for i, (q, st, v, o) in enumerate(zip(seqs, starts, via, outs)):
            pl = o.prompt_logprobs
            n_prompt = len(q) - 1 if v else len(q)
            if list(o.prompt_token_ids) != q[:n_prompt]:
                raise RuntimeError(f"sequence {i}: vLLM prompt ids differ from the input")
            lp = np.empty(len(q) - st, dtype=np.float32)
            for k in range(st, n_prompt):
                lp[k - st] = pl[k][q[k]].logprob
            if v and st <= len(q) - 1:
                c = o.outputs[0]
                if list(c.token_ids)[:1] != [q[-1]]:
                    raise RuntimeError(f"sequence {i}: forced last token not sampled ({list(c.token_ids)})")
                lp[-1] = c.logprobs[0][q[-1]].logprob
            tot = float(np.sum(lp, dtype=np.float64))
            res.append(dict(logprobs=lp, sum=tot, mean=tot / lp.size if lp.size else float("nan"), n_tokens=int(lp.size),
                            start=st))
        ntok = sum(len(q) for q in seqs)
        self.last_score = dict(seconds=dt, sequences=len(seqs), input_tokens=ntok,
                               scored_tokens=sum(r["n_tokens"] for r in res), tok_per_s=ntok / max(dt, 1e-9))
        return res

    def _token_prompt(self, ids, imgs):
        if not imgs:
            return {"prompt_token_ids": ids}
        tid = self.image_token_id
        col = [t for k, t in enumerate(ids) if t != tid or k == 0 or ids[k - 1] != tid]
        runs = sum(1 for k, t in enumerate(ids) if t == tid and (k == 0 or ids[k - 1] != tid))
        if runs != len(imgs):
            raise ValueError(f"{runs} image placeholder runs in the sequence but {len(imgs)} images given")
        return {"prompt_token_ids": col, "multi_modal_data": {"image": [load_image(x) for x in imgs]}}

    def gpu_memory(self, reset_peak=False):
        """Per-rank CUDA memory (bytes): allocated, reserved, peak allocated since the last reset, device free/total."""
        return self._rpc(_w_mem, bool(reset_peak))

    # ---------------- export ----------------
    def export(self, edits, out_dir, duplicate_layers=None, link="hard"):
        """New HF checkpoint with `edits` baked in (values read back from the live engine: bit-identical to what
        the forward pass used), optionally with layers i..j duplicated (inserted after j).  Originals untouched."""
        edited = {}
        if edits:
            prims = self.expand(edits)
            h = self.apply(edits)
            try:
                for hf in sorted({p["hf"] for p in prims}):
                    edited[hf] = self.read(hf)
            finally:
                self.revert(h)
        return write_checkpoint(self.model_dir, out_dir, edited, duplicate_layers, link, log=self.log)

    def duplicate_layers(self, layers, out_dir, edits=None, link="hard"):
        i, j = (layers[0], layers[-1])
        return self.export(edits, out_dir, duplicate_layers=(i, j), link=link)


def image_data_url(path):
    """Same encoding as code/mm/gen_mm.py data_url()."""
    import base64
    import mimetypes
    mime = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as f:
        return f"data:{mime};base64," + base64.b64encode(f.read()).decode()


def image_content(prompt, image_paths):
    """Message content for one item, identical to code/mm/gen_mm.py build_content(): a prompt with exactly
    len(image_paths) "<image>" placeholders is interleaved (text, image, text, ...); otherwise images first, then text."""
    imgs = [{"type": "image_url", "image_url": {"url": image_data_url(p)}} for p in image_paths]
    parts = prompt.split("<image>")
    if len(parts) - 1 == len(imgs) and imgs:
        content = []
        for k, txt in enumerate(parts):
            if txt.strip():
                content.append({"type": "text", "text": txt.strip()})
            if k < len(imgs):
                content.append(imgs[k])
        return content
    return imgs + [{"type": "text", "text": prompt}]


def image_message(prompt, image_paths):
    return [{"role": "user", "content": image_content(prompt, image_paths)}]


def _n_images(msg):
    c = msg.get("content") if isinstance(msg, dict) else None
    if not isinstance(c, list):
        return 0
    return sum(1 for part in c if isinstance(part, dict) and part.get("type") in ("image_url", "image", "image_pil"))


def load_image(x):
    """Decode an image exactly as vLLM's chat path does for data URLs (ImageMediaIO: RGB, RGBA on white)."""
    from vllm.multimodal.media.image import ImageMediaIO
    io_ = ImageMediaIO(image_mode="RGB")
    if isinstance(x, (str, os.PathLike)):
        with open(x, "rb") as f:
            x = f.read()
    if isinstance(x, (bytes, bytearray)):
        return io_.load_bytes(bytes(x)).media
    return io_._convert_image_mode(x)


def _num(x):
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    return np.asarray(x, dtype=np.float32)


def _vec(x, n, name):
    a = np.asarray(x, dtype=np.float32).reshape(-1)
    if a.size != n:
        raise ValueError(f"{name} has length {a.size}, selection needs {n}")
    return a


def _is_identity(p):
    op = p["op"]
    if op == "scale":
        return np.all(np.asarray(p["alpha"]) == 1.0)
    if op == "add":
        return np.all(np.asarray(p["value"]) == 0.0)
    if op == "rank1":
        return not np.any(np.asarray(p["u"])) or not np.any(np.asarray(p["v"]))
    if op in DIRECTIONAL:
        return p["alpha"] == 0.0
    return False   # set: may legitimately equal the current value


# ======================================================================================================
# Checkpoint writing (no engine needed): edited tensors + optional layer duplication
# ======================================================================================================
_ST_DT = {"BF16": "bfloat16", "F16": "float16", "F32": "float32", "F64": "float64", "I64": "int64", "I32": "int32",
          "U8": "uint8", "BOOL": "bool"}


def _st_header(path):
    import struct
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(n))


def _place(src, dst, link):
    """Unchanged shard: hard link (same inode, no data copied), symlink, or copy.  Never touches src."""
    if link == "hard":
        try:
            os.link(src, dst)
            return "hardlink"
        except OSError:
            pass
    elif link == "sym":
        os.symlink(src, dst)
        return "symlink"
    shutil.copy2(src, dst)
    return "copy"


def write_checkpoint(src_dir, out_dir, edited=None, duplicate=None, link="hard", shard_bytes=4 << 30, log=print):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    edited = dict(edited or {})
    if os.path.exists(out_dir) and os.listdir(out_dir):
        raise FileExistsError(f"{out_dir} exists and is not empty (never overwritten)")
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    idx = json.load(open(os.path.join(src_dir, "model.safetensors.index.json")))
    wm = idx["weight_map"]
    cfg = json.load(open(os.path.join(src_dir, "config.json")))
    tc = cfg.get("text_config", cfg)
    files = sorted(set(wm.values()))
    stat0 = {f: os.stat(os.path.join(src_dir, f)) for f in files}
    hdr = {f: _st_header(os.path.join(src_dir, f)) for f in files}
    for n in edited:
        if n not in wm:
            raise KeyError(n)
    pat = re.compile(r"^(model\.language_model\.layers\.)(\d+)(\..+)$")
    rename = lambda n: n
    dup = []
    if duplicate is not None:
        i, j = int(duplicate[0]), int(duplicate[1])
        L = tc["num_hidden_layers"]
        assert 0 <= i <= j < L
        k = j - i + 1

        def rename(n):
            m = pat.match(n)
            return f"{m[1]}{int(m[2]) + k}{m[3]}" if m and int(m[2]) > j else n

        for n in sorted(wm):
            m = pat.match(n)
            if m and i <= int(m[2]) <= j:
                dup.append((f"{m[1]}{int(m[2]) - i + j + 1}{m[3]}", n))
        lt = tc["layer_types"]
        tc["layer_types"] = lt[:j + 1] + lt[i:j + 1] + lt[j + 1:]
        tc["num_hidden_layers"] = L + k
    manifest = dict(source=os.path.abspath(src_dir), created=time.strftime("%Y-%m-%d %H:%M:%S"),
                    duplicate_layers=list(duplicate) if duplicate is not None else None,
                    edited={}, files={})

    def ckpt_tensor(n, fh):
        t = edited.get(n)
        ck_dt = getattr(torch, _ST_DT[hdr[wm[n]][n]["dtype"]])
        if t is None:
            return fh.get_tensor(n)
        t = t.contiguous()
        if list(t.shape) != list(hdr[wm[n]][n]["shape"]):
            raise ValueError(f"{n}: edited shape {list(t.shape)} != checkpoint {hdr[wm[n]][n]['shape']}")
        if t.dtype != ck_dt:
            c = t.to(ck_dt)
            if _bits_equal(c.to(t.dtype), t):
                t = c
            else:
                manifest["edited"].setdefault(n, {})["dtype_note"] = (
                    f"saved as {t.dtype} (engine dtype): edited values are not exactly representable in {ck_dt}")
        manifest["edited"].setdefault(n, {}).update(
            dtype=str(t.dtype), sha256=hashlib.sha256(_int_view(t).numpy().tobytes()).hexdigest())
        return t

    new_wm, total = {}, 0
    for f in files:
        names = [n for n in wm if wm[n] == f]
        src = os.path.join(src_dir, f)
        dst = os.path.join(out_dir, f)
        if not any(n in edited for n in names) and all(rename(n) == n for n in names):
            manifest["files"][f] = _place(src, dst, link)
            for n in names:
                new_wm[n] = f
        else:
            with safe_open(src, "pt", device="cpu") as fh:
                tensors = {rename(n): ckpt_tensor(n, fh) for n in names}
            save_file(tensors, dst, metadata={"format": "pt"})
            manifest["files"][f] = "rewritten"
            for n in names:
                new_wm[rename(n)] = f
        log(f"[export] {f}: {manifest['files'][f]}")
    if dup:
        groups, cur, size = [], [], 0
        for new, old in dup:
            h = hdr[wm[old]][old]
            nb = h["data_offsets"][1] - h["data_offsets"][0]
            if cur and size + nb > shard_bytes:
                groups.append(cur)
                cur, size = [], 0
            cur.append((new, old))
            size += nb
        groups.append(cur)
        handles = {}
        for gi, grp in enumerate(groups):
            fname = f"model-dup-{gi + 1:05d}-of-{len(groups):05d}.safetensors"
            tensors = {}
            for new, old in grp:
                f = wm[old]
                if f not in handles:
                    handles[f] = safe_open(os.path.join(src_dir, f), "pt", device="cpu")
                tensors[new] = ckpt_tensor(old, handles[f]).clone()
                new_wm[new] = fname
            save_file(tensors, os.path.join(out_dir, fname), metadata={"format": "pt"})
            manifest["files"][fname] = f"new ({len(grp)} duplicated tensors)"
            log(f"[export] {fname}: {len(grp)} duplicated tensors")
        handles.clear()
    for f in sorted(set(new_wm.values())):
        h = _st_header(os.path.join(out_dir, f))
        total += sum(v["data_offsets"][1] - v["data_offsets"][0] for k, v in h.items() if k != "__metadata__")
    json.dump({"metadata": {"total_size": total}, "weight_map": dict(sorted(new_wm.items()))},
              open(os.path.join(out_dir, "model.safetensors.index.json"), "w"), indent=2)
    for e in sorted(os.listdir(src_dir)):
        p = os.path.join(src_dir, e)
        if not os.path.isfile(p) or e.endswith(".safetensors") or e == "model.safetensors.index.json":
            continue
        if e == "config.json":
            json.dump(cfg, open(os.path.join(out_dir, e), "w"), indent=4)
        else:
            shutil.copy2(p, os.path.join(out_dir, e))
    stat1 = {f: os.stat(os.path.join(src_dir, f)) for f in files}
    changed = [f for f in files if (stat0[f].st_size, stat0[f].st_mtime_ns, stat0[f].st_ino)
               != (stat1[f].st_size, stat1[f].st_mtime_ns, stat1[f].st_ino)]
    if changed:
        raise RuntimeError(f"original files changed during export: {changed}")
    manifest["seconds"] = round(time.time() - t0, 1)
    manifest["originals_unchanged"] = True
    json.dump(manifest, open(os.path.join(out_dir, "surgery_manifest.json"), "w"), indent=2)
    log(f"[export] wrote {out_dir} in {manifest['seconds']}s")
    return manifest
