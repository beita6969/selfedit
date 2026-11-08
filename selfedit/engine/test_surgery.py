"""Verification suite and benchmark for surgery.py.  Each phase runs in its own process (one vLLM engine):

  core   TP1 correctness: determinism, HF->vLLM layout check, derived-tensor audit, every op class (weights vs an
         independent CPU reference, sibling regions untouched, logprobs move / stay identical, bit-exact revert),
         semantic-equivalence tests, chat template / reasoning effort, export of an edited and a layer-duplicated
         checkpoint.  Writes OUT/core_tp<T>_<stamp>.json (+ export reference).
  reload load the edited export: weights bit-identical to the in-engine edit, logprobs vs in-engine reference.
  dup    load the layer-duplicated export: layout check, copies identical, coherent generation.
  bench  TP1 timings: load, apply, 32 math prompts x 4k tokens, revert.
  tp2    the op / equivalence / revert suite at TP=2.
  vision multimodal=True: vision-tower layout check, image-prompt determinism, every op class on vision-tower / merger
         tensors (apply -> image-probe generation moves -> bit-exact revert -> probe identical), encoder-cache control,
         score(images=...) vs generation logprobs, thinking generation under a vision edit.
  textref text-only default: greedy probe logprobs vs a stored core-phase reference (--ref export_ref_*.json), i.e. the
         multimodal changes leave the default language-only engine byte-identical.
"""
import argparse
import hashlib
import json
import os
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

MODEL = "/data/fd/mingdazhang/selfedit/models/Qwen3.8-27B"
OUT = os.environ.get("SELFEDIT_ENGINE_OUT", "/data/fd/mingdazhang/selfedit/engine")
SETS = "/data/fd/mingdazhang/selfedit/data/sets"
MMDATA = os.environ.get("SELFEDIT_MM_DATA", "/data/fd/mingdazhang/selfedit/data/mm")   # sets/*.jsonl with image_paths
PROBES = ["What is the capital of France? Answer in one short sentence.",
          "List the first five prime numbers, separated by commas.",
          "Translate 'good morning, my friend' into German.",
          "In one sentence, why is the sky blue?"]
STAMP = time.strftime("%Y%m%d-%H%M%S")
H, I = 5120, 17408


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ------------------------------------------------------------------------------------------------ helpers
def probe(S, max_tokens=24, prompts=None):
    res = S.generate(prompts or PROBES, sampling={"temperature": 0.0}, max_tokens=max_tokens, enable_thinking=False, logprobs=5)
    return [dict(ids=s["token_ids"], lp=s["logprobs"], text=s["content"]) for r in res for s in r["samples"]]


def cmp(base, new):
    ident = all(a["ids"] == b["ids"] and a["lp"] == b["lp"] for a, b in zip(base, new))
    d0 = dd = 0.0
    ncmp = ndiv = 0
    for a, b in zip(base, new):
        for t in range(min(len(a["ids"]), len(b["ids"]))):
            if a["ids"][t] != b["ids"][t]:
                ndiv += 1
                break
            d = abs(a["lp"][t] - b["lp"][t])
            if t == 0:
                d0 = max(d0, d)
            else:
                dd = max(dd, d)
                ncmp += 1
    return dict(identical=ident, d_prefill=d0, d_decode=dd, n_decode_cmp=ncmp, n_diverged=ndiv)


def unit(rng, n):
    v = rng.standard_normal(n)
    return (v / np.linalg.norm(v)).astype(np.float32)


def sha(t):
    from surgery import _int_view
    return hashlib.sha256(_int_view(t.contiguous()).numpy().tobytes()).hexdigest()[:16]


def bits_equal(a, b):
    from surgery import _bits_equal
    return _bits_equal(a.contiguous(), b.contiguous())


class Ckpt:
    def __init__(self, d):
        from safetensors import safe_open
        self.d, self.open = d, safe_open
        self.wm = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
        self.h = {}

    def get(self, n):
        f = self.wm[n]
        if f not in self.h:
            self.h[f] = self.open(os.path.join(self.d, f), "pt", device="cpu")
        return self.h[f].get_tensor(n)


def big(S, hf):
    return int(np.prod(S.layout.specs[hf]["shape"])) > 120_000_000


def region(S, hf, prim_like=None, extra_rows=64, seed=0):
    """Index used to compare a tensor: whole tensor if small, else selected rows (+ random other rows)."""
    if not big(S, hf):
        return None
    rows = set()
    if prim_like is not None:
        rows |= set(prim_like["R"].tolist())
    rng = np.random.default_rng(seed)
    rows |= set(rng.choice(S.layout.specs[hf]["shape2"][0], size=extra_rows, replace=False).tolist())
    return {"rows": sorted(rows)}


def expected_after(S, edit, cur, rowmap):
    """Independent CPU reference of the op math on HF-coordinate tensors (cur: hf -> 2-D/1-D runtime-dtype tensor,
    updated in place; rowmap: hf -> HF row ids held in cur for tensors compared on a row subset).
    Returns the set of hf names whose result involves a reduction (directional ops)."""
    import torch
    approx = {}
    for p in S.expand(edit):
        if p["hf"] in approx:          # a later op on a directionally edited tensor: no error bound tracked
            approx[p["hf"]] = None
        W = cur[p["hf"]]
        Rh = p["R"]
        if p["hf"] in rowmap:
            pos = np.searchsorted(rowmap[p["hf"]], Rh)
            assert np.array_equal(rowmap[p["hf"]][pos], Rh)
            Rh = pos
        R = torch.as_tensor(Rh)
        if p["C"] is None:
            blk = W[R].float()
        else:
            C = torch.as_tensor(p["C"])
            blk = W[R][:, C].float()
        op = p["op"]

        def val(x):
            if isinstance(x, float):
                return torch.full_like(blk, x)
            return torch.broadcast_to(torch.as_tensor(np.asarray(x, np.float32)), blk.shape)

        if op == "scale":
            new = blk * (p["alpha"] if isinstance(p["alpha"], float) else val(p["alpha"]))
        elif op == "set":
            new = val(p["value"]).clone()
        elif op == "add":
            d = val(p["value"])
            new = torch.where(d == 0, blk, blk + d)
        elif op == "rank1":
            d = torch.as_tensor(p["u"])[:, None] * torch.as_tensor(p["v"])[None, :]
            new = torch.where(d == 0, blk, blk + d)
        else:
            dirv = torch.as_tensor(p["dir"]).double()
            a = np.float32(p["alpha"])
            if p["side"] == "out":
                s64 = dirv @ blk.double()
                d = torch.as_tensor(a * p["dir"])[:, None] * s64.float()[None, :]
                d64 = float(a) * dirv[:, None] * s64[None, :]
            else:
                s64 = blk.double() @ dirv
                d = (torch.as_tensor(a) * s64.float())[:, None] * torch.as_tensor(p["dir"])[None, :]
                d64 = float(a) * s64[:, None] * dirv[None, :]
            new = torch.where(d == 0, blk, blk + d)
            approx[p["hf"]] = (W, R, None if p["C"] is None else C, blk.double() + d64, blk.double().abs() + d64.abs())
        newc = new.to(W.dtype)
        if p["C"] is None:
            W[R] = newc
        else:
            W[R[:, None], C[None, :]] = newc
    return approx


def ulp_diff(a, b):
    from surgery import _int_view
    ia, ib = _int_view(a.contiguous()).long(), _int_view(b.contiguous()).long()
    d = (ia - ib).abs()
    return int((d != 0).sum()), int(d.max()) if d.numel() else 0


# ------------------------------------------------------------------------------------------------ op tests
def op_test(S, ck, name, edit, base, fp_ref, expect="change", full_fp=True, probe_fn=None):
    import torch
    probe_ = probe_fn or probe
    t_all = time.time()
    rec = dict(name=name, expect=expect)
    prims = S.expand(edit)
    hfs = sorted({p["hf"] for p in prims})
    vnames = sorted({p["vname"] for p in prims})
    sibs = sorted({hf for hf, sp in S.layout.specs.items() if sp["vname"] in vnames} - set(hfs))
    sel_of = {p["hf"]: p for p in prims}
    idx = {hf: region(S, hf, sel_of.get(hf)) for hf in hfs + sibs}
    # originals: engine read must equal the checkpoint (cast to runtime dtype) bit for bit
    orig = {}
    for hf in hfs + sibs:
        t = S.read(hf, idx[hf])
        c = ck.get(hf)
        c = c.view(c.shape[0], -1) if c.dim() >= 3 else c
        if idx[hf] is not None:
            c = c[torch.as_tensor(idx[hf]["rows"])]
        t2 = t.view(t.shape[0], -1) if t.dim() >= 3 else t
        assert bits_equal(t2, c.to(t2.dtype)), f"engine read != checkpoint for {hf}"
        orig[hf] = t2.clone()
    rec["sha_before"] = {hf: sha(orig[hf]) for hf in hfs}
    # expected result (independent math on full tensors or on the compared rows)
    cur = {hf: orig[hf].clone() for hf in hfs}
    rowmap = {hf: np.asarray(idx[hf]["rows"]) for hf in hfs if idx[hf] is not None}
    approx = expected_after(S, edit, cur, rowmap)
    t0 = time.time()
    h = S.apply(edit)
    rec["apply_s"] = round(time.time() - t0, 3)
    rec["changed_elems"] = h.changed
    rec["ops"] = [dict(op=i["op"], hf=i["hf"].replace("model.language_model.", "").replace("model.visual.", "visual."), changed=i["changed"],
                       maxd=i["maxd"]) for i in h.info]
    # weights after apply
    wchk = {}
    for hf in hfs:
        t = S.read(hf, idx[hf])
        t = t.view(t.shape[0], -1) if t.dim() >= 3 else t
        rec.setdefault("sha_edited", {})[hf] = sha(t)
        if approx.get(hf) is not None:
            nd, mx = ulp_diff(t, cur[hf])
            _, R_, C_, ex64, scale = approx[hf]
            got = (t[R_][:, C_] if C_ is not None else t[R_]).double()
            # error bound: correct bf16 rounding of the fp64-exact result + fp32 rounding of the rank-1 term
            bound = ex64.abs() * 2.0 ** -8 + scale * 2.0 ** -20 + 1e-30
            ratio = float(((got - ex64).abs() / bound).max())
            wchk[hf] = dict(exact=nd == 0, n_diff_vs_ref=nd, max_ulp_vs_ref=mx, numel=t.numel(),
                            max_err_over_bound=round(ratio, 4), within_bound=ratio <= 1.0)
        else:
            wchk[hf] = dict(exact=bits_equal(t, cur[hf]))
        if idx[hf] is not None:
            wchk[hf]["rows_compared"] = len(idx[hf]["rows"])
    rec["weights_vs_reference"] = wchk
    rec["siblings_untouched"] = all(bits_equal(
        (lambda t: t.view(t.shape[0], -1) if t.dim() >= 3 else t)(S.read(hf, idx[hf])), orig[hf]) for hf in sibs)
    rec["siblings"] = [s.replace("model.language_model.", "") for s in sibs]
    if full_fp:
        fp1 = S.model_fingerprint()
        changed_v = sorted({k[1] for k in fp1 if fp1[k] != fp_ref[k]})
        rec["fingerprint_changed_params"] = changed_v
        rec["only_touched_changed"] = set(changed_v) <= set(vnames)
    lp = probe_(S)
    rec["logprobs_vs_base"] = cmp(base, lp)
    rec["probe_text"] = [x["text"] for x in lp]
    t0 = time.time()
    rep = S.revert(h)
    rec["revert_s"] = round(time.time() - t0, 3)
    rec["revert"] = rep
    after = {}
    for hf in hfs + sibs:
        t = S.read(hf, idx[hf])
        after[hf] = t.view(t.shape[0], -1) if t.dim() >= 3 else t
    rec["sha_reverted"] = {hf: sha(after[hf]) for hf in hfs}
    rec["revert_reads_bit_exact"] = all(bits_equal(after[hf], orig[hf]) for hf in hfs + sibs)
    if full_fp:
        rec["revert_full_fingerprint_equal"] = S.model_fingerprint() == fp_ref
    lp2 = probe_(S)
    rec["logprobs_after_revert"] = cmp(base, lp2)
    # verdict
    ok = rec["revert"]["blocks_bit_exact"] and rec["revert_reads_bit_exact"] and rec["logprobs_after_revert"]["identical"]
    ok = ok and rec["siblings_untouched"] and all(v["exact"] or v.get("within_bound", False) for v in wchk.values())
    if full_fp:
        ok = ok and rec["only_touched_changed"] and rec["revert_full_fingerprint_equal"]
    lpc = rec["logprobs_vs_base"]
    if expect == "change":
        ok = ok and rec["changed_elems"] > 0 and not lpc["identical"] and (lpc["d_prefill"] > 0 or lpc["n_diverged"] > 0)
    else:
        ok = ok and lpc["identical"] and rec["changed_elems"] == 0
    rec["PASS"] = bool(ok)
    rec["secs"] = round(time.time() - t_all, 1)
    log(f"{'PASS' if ok else 'FAIL'} {name}: changed={rec['changed_elems']} lp={lpc} weights={wchk} "
        f"revert={rep} apply={rec['apply_s']}s revert={rec['revert_s']}s")
    return rec


def build_tests(S, rng, ck, tp):
    def rms(hf):
        t = ck.get(hf).float()
        return float(t.pow(2).mean().sqrt())

    L = "model.language_model.layers."
    toks = sorted(set(S.tokenizer.encode(" Paris France prime German blue", add_special_tokens=False)))
    t = []
    # per-head scale / add on GDN parameters (head 30+ lives on rank 1 at TP2)
    t.append(("scale A_log head (fp32 param)", {"op": "scale", "param": "layers.0.linear_attn.A_log", "head": 5, "alpha": 0.5}))
    t.append(("scale A_log head on rank1@TP2", {"op": "scale", "param": "layers.0.linear_attn.A_log", "head": 30, "alpha": 0.5}))
    t.append(("add dt_bias head", {"op": "add", "param": "layers.2.linear_attn.dt_bias", "head": [9, 40], "value": 1.0}))
    t.append(("scale in_proj_a head (decay gate)", {"op": "scale", "param": "layers.4.linear_attn.in_proj_a", "head": [10, 33], "alpha": 0.25}))
    t.append(("scale in_proj_b head (beta gate)", {"op": "scale", "param": "layers.5.linear_attn.in_proj_b", "head": [11, 44], "alpha": 0.25}))
    t.append(("add in_proj_z head (output gate)", {"op": "add", "param": "layers.6.linear_attn.in_proj_z", "head": [12, 47], "value": 0.05}))
    t.append(("add conv1d k-head taps", {"op": "add", "param": "layers.8.linear_attn.conv1d", "head": [3, 12], "part": "k", "value": 0.1}))
    t.append(("scale out_proj head cols", {"op": "scale", "param": "layers.13.linear_attn.out_proj", "head": [20, 40], "alpha": 3.0}))
    # full attention
    t.append(("scale q_proj output-gate rows", {"op": "scale", "param": "layers.3.self_attn.q_proj", "head": [2, 20], "part": "gate", "alpha": 0.25}))
    t.append(("set v_proj kv-head", {"op": "set", "param": "layers.7.self_attn.v_proj", "head": 3, "value": 0.0}))
    t.append(("scale o_proj head cols", {"op": "scale", "param": "layers.15.self_attn.o_proj", "head": [7, 21], "alpha": 3.0}))
    # MLP neurons
    t.append(("set gate_proj neuron rows", {"op": "set", "param": "layers.10.mlp.gate_proj", "neuron": [123, 4567, 17000], "value": 0.0}))
    t.append(("set down_proj neuron col", {"op": "set", "param": "layers.11.mlp.down_proj", "neuron": [456, 16000], "value": 0.0}))
    t.append(("scale up_proj neuron rows", {"op": "scale", "param": "layers.12.mlp.up_proj", "neuron": [789, 9000], "alpha": 4.0}))
    # rank-1 and directional ops
    hf = L + "20.mlp.down_proj.weight"
    t.append(("rank1 down_proj", {"op": "rank1", "param": hf, "u": unit(rng, H), "v": unit(rng, I),
                                  "alpha": 0.5 * rms(hf) * float(np.sqrt(H * I))}))
    hf = L + "21.linear_attn.in_proj_qkv.weight"
    t.append(("rank1 in_proj_qkv v-head block", {"op": "rank1", "param": hf, "head": 30, "part": "v", "u": unit(rng, 128),
                                                 "v": unit(rng, H), "alpha": 1.0 * rms(hf) * float(np.sqrt(128 * H))}))
    t.append(("project_out out-side out_proj", {"op": "project_out", "param": "layers.22.linear_attn.out_proj", "side": "out", "u": unit(rng, H)}))
    t.append(("project_out in-side gate_proj", {"op": "project_out", "param": "layers.23.mlp.gate_proj", "side": "in", "v": unit(rng, H)}))
    t.append(("project_out out-side gate_proj (sum over sharded rows)", {"op": "project_out", "param": "layers.24.mlp.gate_proj", "side": "out", "u": unit(rng, I)}))
    t.append(("amplify in-side down_proj (sum over sharded cols)", {"op": "amplify", "param": "layers.25.mlp.down_proj", "side": "in", "v": unit(rng, I), "alpha": 3.0}))
    t.append(("amplify out-side k_proj", {"op": "amplify", "param": "layers.27.self_attn.k_proj", "side": "out", "u": unit(rng, 1024), "alpha": 2.0}))
    t.append(("project_out in-side out_proj (sum over sharded cols)", {"op": "project_out", "param": "layers.26.linear_attn.out_proj", "side": "in", "v": unit(rng, 6144)}))
    t.append(("project_out in-side lm_head rows 0:8192", {"op": "project_out", "param": "lm_head", "rows": "0:8192", "side": "in", "v": unit(rng, H)}))
    t.append(("scale lm_head rows across TP boundary", {"op": "scale", "param": "lm_head", "rows": "124000:124400", "alpha": 2.0}))
    t.append(("scale embed_tokens rows", {"op": "scale", "param": "embed_tokens", "rows": toks, "alpha": 1.5}))
    # norms (replicated; Gemma-style (1+w) at runtime)
    t.append(("add q_norm", {"op": "add", "param": "layers.7.self_attn.q_norm", "value": 0.5}))
    t.append(("add input_layernorm", {"op": "add", "param": "layers.9.input_layernorm", "value": 0.5}))
    t.append(("scale linear_attn.norm", {"op": "scale", "param": "layers.16.linear_attn.norm", "alpha": 2.0}))
    t.append(("add final norm", {"op": "add", "param": "norm", "value": 0.2}))
    # block skipping
    t.append(("zero_block attn (GDN layer 30)", {"op": "zero_block", "layer": 30, "parts": ["attn"]}))
    t.append(("zero_block attn (full layer 31)", {"op": "zero_block", "layer": 31, "parts": ["attn"]}))
    t.append(("zero_block mlp (layer 33)", {"op": "zero_block", "layer": 33, "parts": ["mlp"]}))
    t.append(("composite edit list", [{"op": "scale", "param": "layers.40.linear_attn.A_log", "head": 1, "alpha": 0.7},
                                      {"op": "project_out", "param": "layers.41.mlp.up_proj", "side": "in", "v": unit(rng, H)},
                                      {"op": "zero_block", "layer": 42, "parts": ["mlp"]}]))
    # exact-zero edits: weights and logprobs must be bit-identical
    z = []
    z.append(("ZERO scale alpha=1 all A_log", {"op": "scale", "param": "layers.1.linear_attn.A_log", "alpha": 1.0}))
    z.append(("ZERO add 0 to in_proj_z", {"op": "add", "param": "layers.6.linear_attn.in_proj_z", "value": 0.0}))
    z.append(("ZERO rank1 alpha=0 down_proj", {"op": "rank1", "param": "layers.20.mlp.down_proj", "u": unit(rng, H), "v": unit(rng, I), "alpha": 0.0}))
    z.append(("ZERO amplify alpha=0 o_proj in-side", {"op": "amplify", "param": "layers.15.self_attn.o_proj", "side": "in", "v": unit(rng, 6144), "alpha": 0.0}))
    z.append(("ZERO amplify alpha=0 gate_proj out-side", {"op": "amplify", "param": "layers.24.mlp.gate_proj", "side": "out", "u": unit(rng, I), "alpha": 0.0}))
    z.append(("ZERO scale 1 q_norm", {"op": "scale", "param": "layers.3.self_attn.q_norm", "alpha": 1.0}))
    cur = S.read("layers.5.linear_attn.in_proj_b", {"head": [11, 44]}).float().numpy()
    z.append(("ZERO set in_proj_b heads to current values", {"op": "set", "param": "layers.5.linear_attn.in_proj_b", "head": [11, 44], "value": cur}))
    return t, z


EQUIV = [
    ("MLP neuron L15 n=1000: gate row == up row == down col",
     [{"op": "set", "param": "layers.15.mlp.gate_proj", "neuron": 1000, "value": 0.0},
      {"op": "set", "param": "layers.15.mlp.up_proj", "neuron": 1000, "value": 0.0},
      {"op": "set", "param": "layers.15.mlp.down_proj", "neuron": 1000, "value": 0.0}]),
    ("GDN v-head L16 h=20 (h=44): in_proj_qkv v rows == conv1d v rows == in_proj_z rows == out_proj cols",
     [{"op": "set", "param": "layers.16.linear_attn.in_proj_qkv", "head": [20, 44], "part": "v", "value": 0.0},
      {"op": "set", "param": "layers.16.linear_attn.conv1d", "head": [20, 44], "part": "v", "value": 0.0},
      {"op": "set", "param": "layers.16.linear_attn.in_proj_z", "head": [20, 44], "value": 0.0},
      {"op": "set", "param": "layers.16.linear_attn.out_proj", "head": [20, 44], "value": 0.0}]),
    ("Full-attn L19 kv-head 1 (and 3): v_proj rows == o_proj cols of its 6 query heads",
     [{"op": "set", "param": "layers.19.self_attn.v_proj", "head": [1, 3], "value": 0.0},
      {"op": "set", "param": "layers.19.self_attn.o_proj", "head": list(range(6, 12)) + list(range(18, 24)), "value": 0.0}]),
]


def equiv_tests(S, base):
    out = []
    for name, variants in EQUIV:
        res = []
        for v in variants:
            h = S.apply(v)
            res.append(probe(S))
            S.revert(h)
        same = all(r == res[0] for r in res[1:]) and all(
            a["ids"] == b["ids"] and a["lp"] == b["lp"] for r in res[1:] for a, b in zip(res[0], r))
        rec = dict(name=name, variants_bit_identical=same, differs_from_base=not cmp(base, res[0])["identical"],
                   vs_base=cmp(base, res[0]))
        rec["PASS"] = bool(same and rec["differs_from_base"])
        log(f"{'PASS' if rec['PASS'] else 'FAIL'} EQUIV {name}: {rec}")
        out.append(rec)
    return out


def summarize_layout(rep):
    return [dict(rank=r["rank"], checked=r["checked"], elements=r["elements"], mismatches=r["mismatches"][:10],
                 n_mismatch=len(r["mismatches"]), coverage_bad=r["coverage_bad"], aliasing=r["aliasing"]) for r in rep]


def suite(S, ck, tp, results):
    rng = np.random.default_rng(1)
    b1 = probe(S)
    b2 = probe(S)
    b3 = probe(S)
    results["determinism"] = cmp(b1, b2)
    results["determinism_3"] = cmp(b1, b3)
    log("determinism (identical batches):", results["determinism"], results["determinism_3"])
    # batch-composition sensitivity (information only): prompt 0 alone vs inside the 4-prompt probe batch
    solo = probe(S, prompts=PROBES[:1])
    results["batch_composition_prompt0_alone_vs_in_batch"] = cmp(b1[:1], solo)
    log("batch composition (prompt 0 alone vs in batch):", results["batch_composition_prompt0_alone_vs_in_batch"])
    t0 = time.time()
    lay = S.verify_layout()
    results["layout"] = summarize_layout(lay)
    results["layout_secs"] = round(time.time() - t0, 1)
    log("layout:", results["layout"], f"{results['layout_secs']}s")
    scan = S.scan_tensors()
    results["derived_scan"] = scan[0]["tensors"]
    results["param_dtypes"] = S.param_dtypes
    results["unmapped_params"] = S.unmapped_params
    log("non-parameter tensors in model modules:", json.dumps(scan[0]["tensors"], indent=1))
    log("param dtypes:", json.dumps(S.param_dtypes, indent=1))
    t0 = time.time()
    fp0 = S.model_fingerprint()
    results["fingerprint_secs"] = round(time.time() - t0, 2)
    full_fp = results["fingerprint_secs"] < 10
    log(f"full-model fingerprint: {results['fingerprint_secs']}s over {len(fp0)} (rank, param) entries; per-test={full_fp}")
    tests, zeros = build_tests(S, rng, ck, tp)
    recs = []
    for name, e in tests:
        try:
            recs.append(op_test(S, ck, name, e, b1, fp0, "change", full_fp))
        except Exception as ex:
            traceback.print_exc()
            recs.append(dict(name=name, PASS=False, error=repr(ex)))
            S.revert_all()
    for name, e in zeros:
        try:
            recs.append(op_test(S, ck, name, e, b1, fp0, "zero", full_fp))
        except Exception as ex:
            traceback.print_exc()
            recs.append(dict(name=name, PASS=False, error=repr(ex)))
            S.revert_all()
    results["op_tests"] = recs
    results["equivalence"] = equiv_tests(S, b1)
    results["final_full_fingerprint_equal"] = S.model_fingerprint() == fp0
    results["final_probe_identical"] = cmp(b1, probe(S))["identical"]
    log("final full-model fingerprint equal:", results["final_full_fingerprint_equal"],
        "final probe identical:", results["final_probe_identical"])
    return b1, fp0


def chat_checks(S, results):
    r = {}
    q = "What is 17 * 23?"
    for name, kw in [("default", {}), ("thinking_off", dict(enable_thinking=False)), ("effort_low", dict(reasoning_effort="low")),
                     ("effort_medium", dict(reasoning_effort="medium")), ("effort_xhigh", dict(reasoning_effort="xhigh")),
                     ("effort_none", dict(reasoning_effort="none"))]:
        r[f"render_{name}"] = S.render(q, **kw)
    try:
        S.render(q, reasoning_effort="high")
        r["render_effort_high"] = "accepted"
    except Exception as ex:
        r["render_effort_high"] = f"template error: {str(ex)[:160]}"
    math = [json.loads(l)["prompt"] for l in open(f"{SETS}/aime_2025.jsonl")][:2]
    t0 = time.time()
    g = S.generate(math, n=2, max_tokens=1024, reasoning_effort="low")
    r["gen_low_secs"] = round(time.time() - t0, 1)
    r["gen_low"] = [dict(prompt_tokens=x["prompt_tokens"], samples=[dict(
        num_tokens=s["num_tokens"], finish_reason=s["finish_reason"], think_closed=s["think_closed"],
        reasoning_head=s["reasoning"][:200], content_head=s["content"][:300]) for s in x["samples"]]) for x in g]
    g = S.generate(["What is 17 * 23? Reply with just the number."], max_tokens=2048, reasoning_effort="medium",
                   sampling={"temperature": 0.0})
    s = g[0]["samples"][0]
    r["gen_medium_greedy"] = dict(num_tokens=s["num_tokens"], finish_reason=s["finish_reason"],
                                  think_closed=s["think_closed"], reasoning=s["reasoning"][:400], content=s["content"])
    results["chat"] = r
    log("chat checks:", json.dumps(r, indent=1)[:4000])


def export_checks(S, b1, results, ref_path, rng):
    edits = [{"op": "scale", "param": "layers.0.linear_attn.A_log", "head": [3, 4], "alpha": 0.8},
             {"op": "rank1", "param": "layers.40.mlp.down_proj", "u": unit(rng, H), "v": unit(rng, I), "alpha": 3.0},
             {"op": "project_out", "param": "layers.43.self_attn.o_proj", "side": "out", "u": unit(rng, H)},
             {"op": "zero_block", "layer": 50, "parts": ["mlp"]},
             {"op": "set", "param": "layers.52.mlp.gate_proj", "neuron": [1, 2, 3], "value": 0.0}]
    hfs = sorted({p["hf"] for p in S.expand(edits)})
    h = S.apply(edits)
    lp_edit = probe(S)
    from surgery import _int_view
    sha_edit = {hf: hashlib.sha256(_int_view(S.read(hf).contiguous()).numpy().tobytes()).hexdigest() for hf in hfs}
    S.revert(h)
    out_dir = f"{OUT}/ckpt_edit_{STAMP}"
    t0 = time.time()
    man = S.export(edits, out_dir)
    secs = round(time.time() - t0, 1)
    ok_sha = all(man["edited"][hf]["sha256"] == sha_edit[hf] for hf in hfs)
    results["export_edit"] = dict(dir=out_dir, secs=secs, manifest_files=man["files"], edited=man["edited"],
                                  sha_matches_in_engine=ok_sha, lp_edit_vs_base=cmp(b1, lp_edit))
    log("export edit:", json.dumps(results["export_edit"], indent=1))
    ino = {f: (os.stat(f"{MODEL}/{f}").st_ino, os.stat(f"{out_dir}/{f}").st_ino, os.stat(f"{MODEL}/{f}").st_nlink)
           for f in man["files"] if man["files"][f] == "hardlink"}
    results["export_edit"]["hardlinks_same_inode"] = all(a == b for a, b, _ in ino.values())
    json.dump(dict(base=b1, edited=lp_edit, sha=sha_edit, edits_hf=hfs, ckpt=out_dir,
                   engine_args=S.engine_args), open(ref_path, "w"))
    # structural export: duplicate layers 28..31 (lin, lin, lin, full) -> inserted after 31
    dup_dir = f"{OUT}/ckpt_dup28-31_{STAMP}"
    t0 = time.time()
    man2 = S.duplicate_layers([28, 31], dup_dir)
    results["export_dup"] = dict(dir=dup_dir, secs=round(time.time() - t0, 1), files=man2["files"],
                                 config=json.load(open(f"{dup_dir}/config.json"))["text_config"]["num_hidden_layers"])
    log("export dup:", json.dumps(results["export_dup"], indent=1))
    return out_dir, dup_dir


# ------------------------------------------------------------------------------------------------ phases
def engine_kwargs(a):
    return dict(max_model_len=a.max_model_len, max_num_seqs=a.max_num_seqs, max_num_batched_tokens=a.max_num_batched_tokens)


def phase_core(a):
    from surgery import Surgeon
    S = Surgeon(a.model, tp=a.tp, **engine_kwargs(a))
    ck = Ckpt(a.model)
    results = dict(phase=f"core_tp{a.tp}", load_secs=round(S.load_seconds, 1), engine_args=S.engine_args)
    path = f"{OUT}/core_tp{a.tp}_{STAMP}.json"
    try:
        b1, fp0 = suite(S, ck, a.tp, results)
        if a.tp == 1 and not a.skip_extra:
            chat_checks(S, results)
            export_checks(S, b1, results, f"{OUT}/export_ref_{STAMP}.json", np.random.default_rng(7))
    finally:
        json.dump(results, open(path, "w"), indent=1, default=str)
        log("results ->", path)
    recs = results.get("op_tests", []) + results.get("equivalence", [])
    log(f"SUMMARY: {sum(r['PASS'] for r in recs)}/{len(recs)} passed; determinism={results.get('determinism')}")
    for r in recs:
        if not r["PASS"]:
            log("FAILED:", r["name"], r.get("error", ""))


def phase_reload(a):
    import torch
    from surgery import Surgeon
    ref = json.load(open(a.ref))
    ea = ref["engine_args"]
    S = Surgeon(ref["ckpt"], tp=ea["tp"], max_model_len=ea["max_model_len"], max_num_seqs=ea["max_num_seqs"],
                max_num_batched_tokens=ea["max_num_batched_tokens"])
    res = dict(phase="reload", ckpt=ref["ckpt"], load_secs=round(S.load_seconds, 1))
    res["layout_vs_export"] = summarize_layout(S.verify_layout())
    from surgery import _int_view
    res["weights_bit_identical_to_in_engine_edit"] = {
        hf: hashlib.sha256(_int_view(S.read(hf).contiguous()).numpy().tobytes()).hexdigest() == ref["sha"][hf]
        for hf in ref["edits_hf"]}
    lp = probe(S)
    res["probe_vs_in_engine_edited"] = cmp(ref["edited"], lp)
    res["probe_vs_base"] = cmp(ref["base"], lp)
    # set the edited tensors back to the original checkpoint values -> should reproduce the base probe
    ck = Ckpt(MODEL)
    sets = [{"op": "set", "param": hf, "value": ck.get(hf).float().reshape(S.layout.specs[hf]["shape2"]).numpy()}
            for hf in ref["edits_hf"]]
    h = S.apply(sets)
    lp0 = probe(S)
    res["after_set_to_original_vs_base"] = cmp(ref["base"], lp0)
    S.revert(h)
    res["after_revert_vs_export_probe"] = cmp(lp, probe(S))
    path = f"{OUT}/reload_{STAMP}.json"
    json.dump(res, open(path, "w"), indent=1, default=str)
    log(json.dumps(res, indent=1, default=str))


def phase_dup(a):
    from surgery import Surgeon
    S = Surgeon(a.ckpt, tp=1, max_model_len=a.max_model_len, max_num_seqs=a.max_num_seqs,
                max_num_batched_tokens=a.max_num_batched_tokens)
    res = dict(phase="dup", ckpt=a.ckpt, load_secs=round(S.load_seconds, 1), num_hidden_layers=S.cfg.L,
               layer_types_28_40=S.cfg.layer_types[28:40])
    res["layout_vs_export"] = summarize_layout(S.verify_layout())
    res["copies_identical"] = {}
    for k in range(4):
        for suf in ["mlp.down_proj", "input_layernorm"] + (["self_attn.o_proj"] if k == 3 else ["linear_attn.A_log", "linear_attn.in_proj_qkv"]):
            a_, b_ = S.sha256(f"layers.{28 + k}.{suf}"), S.sha256(f"layers.{32 + k}.{suf}")
            res["copies_identical"][f"{28 + k}->{32 + k} {suf}"] = a_ == b_
    lp = probe(S, max_tokens=48)
    res["probe_text"] = [x["text"] for x in lp]
    g = S.generate(["Explain briefly what a prime number is and give three examples.",
                    "Write a haiku about autumn leaves."], max_tokens=1024, reasoning_effort="low",
                   sampling={"temperature": 0.0})
    res["gen"] = [dict(num_tokens=s["num_tokens"], finish=s["finish_reason"], think_closed=s["think_closed"],
                       reasoning_head=s["reasoning"][:300], content=s["content"][:600]) for x in g for s in x["samples"]]
    path = f"{OUT}/dup_{STAMP}.json"
    json.dump(res, open(path, "w"), indent=1, default=str)
    log(json.dumps(res, indent=1, default=str))


def phase_bench(a):
    from surgery import Surgeon
    t0 = time.time()
    S = Surgeon(a.model, tp=a.tp, **engine_kwargs(a))
    res = dict(phase=f"bench_tp{a.tp}", load_secs=round(S.load_seconds, 1), wall_to_ready=round(time.time() - t0, 1),
               engine_args=S.engine_args)
    rng = np.random.default_rng(3)
    probe(S)
    tp = time.time()
    probe(S)
    res["probe_4x24_secs"] = round(time.time() - tp, 2)
    edit = ([{"op": "scale", "param": f"layers.{l}.linear_attn.A_log", "head": [1, 2, 3, 4], "alpha": 0.9} for l in (0, 4, 8, 12)]
            + [{"op": "rank1", "param": f"layers.{l}.mlp.down_proj", "u": unit(rng, H), "v": unit(rng, I), "alpha": 1.0}
               for l in (20, 30, 40, 50)]
            + [{"op": "project_out", "param": "layers.47.self_attn.o_proj", "side": "out", "u": unit(rng, H)},
               {"op": "amplify", "param": "layers.45.mlp.down_proj", "side": "in", "v": unit(rng, I), "alpha": 0.5}])
    prompts = [json.loads(l)["prompt"] for f in ("aime_2025", "aime_2026") for l in open(f"{SETS}/{f}.jsonl")][:32]
    for rep in range(a.reps):
        r = {}
        t = time.time()
        h = S.apply(edit)
        r["apply_secs"] = round(time.time() - t, 3)
        r["apply_secs_no_verify"] = None
        t = time.time()
        g = S.generate(prompts, max_tokens=a.gen_tokens)
        r["generate_secs"] = round(time.time() - t, 1)
        toks = [s["num_tokens"] for x in g for s in x["samples"]]
        fin = [s["finish_reason"] for x in g for s in x["samples"]]
        r.update(gen_tokens=sum(toks), tok_per_s=round(sum(toks) / r["generate_secs"], 1), mean_tokens=round(np.mean(toks)),
                 finish={k: fin.count(k) for k in set(fin)},
                 think_closed=sum(bool(s["think_closed"]) for x in g for s in x["samples"]))
        t = time.time()
        rep_ = S.revert(h)
        r["revert_secs"] = round(time.time() - t, 3)
        r["revert"] = rep_
        t = time.time()
        h = S.apply(edit, verify=False)
        r["apply_secs_no_verify"] = round(time.time() - t, 3)
        t = time.time()
        S.revert(h, verify=False)
        r["revert_secs_no_verify"] = round(time.time() - t, 3)
        res[f"rep{rep}"] = r
        log(f"bench rep{rep}:", r)
    path = f"{OUT}/bench_tp{a.tp}_{STAMP}.json"
    json.dump(res, open(path, "w"), indent=1, default=str)
    log(json.dumps(res, indent=1, default=str))


def diffstats(A, B):
    """|a - b| over aligned per-token logprob arrays (lists of equal-length arrays)."""
    a = np.concatenate([np.asarray(x, np.float64) for x in A]) if A else np.zeros(0)
    b = np.concatenate([np.asarray(x, np.float64) for x in B]) if B else np.zeros(0)
    assert a.shape == b.shape, (a.shape, b.shape)
    d = np.abs(a - b)
    sums = [abs(float(np.sum(np.asarray(x, np.float64)) - np.sum(np.asarray(y, np.float64)))) for x, y in zip(A, B)]
    return dict(n=int(d.size), identical=bool(d.size and d.max() == 0), max=float(d.max()) if d.size else 0.0,
                mean=float(d.mean()) if d.size else 0.0, p50=float(np.percentile(d, 50)) if d.size else 0.0,
                p99=float(np.percentile(d, 99)) if d.size else 0.0, frac_gt_1e2=float((d > 1e-2).mean()) if d.size else 0.0,
                abs_sum_diff=[round(x, 5) for x in sums],
                mean_lp_a=float(a.mean()) if a.size else 0.0, mean_lp_b=float(b.mean()) if b.size else 0.0)


def prefix_lp(x, y):
    """Logprobs of two generations of the same prompt on their common token prefix."""
    m = 0
    while m < min(len(x["token_ids"]), len(y["token_ids"])) and x["token_ids"][m] == y["token_ids"][m]:
        m += 1
    return x["logprobs"][:m], y["logprobs"][:m], m


def phase_score(a):
    from surgery import Surgeon
    S = Surgeon(a.model, tp=a.tp, **engine_kwargs(a))
    res = dict(phase=f"score_tp{a.tp}", load_secs=round(S.load_seconds, 1), engine_args=S.engine_args,
               default_sampling=S.default_sampling)
    prompts = [json.loads(l)["prompt"] for f in ("aime_2025", "aime_2026") for l in open(f"{SETS}/{f}.jsonl")][:32]
    S.gpu_memory(reset_peak=True)
    # (1) sampled generation (model defaults, thinking on) with logprobs=0 -> score the same sequences
    t0 = time.time()
    g = S.generate(prompts[:4], max_tokens=a.gen_tokens, logprobs=0)
    res["gen_sampled_secs"] = round(time.time() - t0, 1)
    flat = [(r, s) for r in g for s in r["samples"]]
    seqs = [r["prompt_token_ids"] + s["token_ids"] for r, s in flat]
    starts = [len(r["prompt_token_ids"]) for r, s in flat]
    glp = [s["logprobs"] for r, s in flat]
    res["gen_sampled"] = [dict(prompt_tokens=len(r["prompt_token_ids"]), tokens=s["num_tokens"], finish=s["finish_reason"])
                          for r, s in flat]
    sc1 = S.score(seqs, starts)
    res["score_4seq_secs"] = round(S.last_score["seconds"], 2)
    sc2 = S.score(seqs, starts)
    res["score_repeat_same_batch"] = dict(
        bit_identical=all(np.array_equal(x["logprobs"], y["logprobs"]) for x, y in zip(sc1, sc2)),
        sums_equal=all(x["sum"] == y["sum"] for x, y in zip(sc1, sc2)))
    res["score_n_tokens_match_generated"] = [x["n_tokens"] == len(l) for x, l in zip(sc1, glp)]
    res["score_vs_generate_sampled"] = diffstats(glp, [x["logprobs"] for x in sc1])
    log("score vs generate (sampled):", res["score_vs_generate_sampled"], "repeat:", res["score_repeat_same_batch"])
    # (2) noise references: (a) score of sequence 0 alone vs inside the 4-sequence batch (prefill path)
    solo = S.score(seqs[:1], starts[:1])
    res["noise_score_alone_vs_batch"] = diffstats([sc1[0]["logprobs"]], [solo[0]["logprobs"]])
    # (b) greedy generation of prompt 0 alone vs inside the batch of 4 (decode path), on the common prefix
    gb = S.generate(prompts[:4], max_tokens=a.control_tokens, logprobs=0, sampling={"temperature": 0.0})
    ga = S.generate(prompts[:1], max_tokens=a.control_tokens, logprobs=0, sampling={"temperature": 0.0})
    x, y, m = prefix_lp(gb[0]["samples"][0], ga[0]["samples"][0])
    res["noise_generate_alone_vs_batch"] = dict(common_prefix=m, **diffstats([x], [y]))
    # (c) score vs generate on the greedy batch
    gflat = [(r, s) for r in gb for s in r["samples"]]
    gsc = S.score([r["prompt_token_ids"] + s["token_ids"] for r, s in gflat], [len(r["prompt_token_ids"]) for r, s in gflat])
    res["score_vs_generate_greedy"] = diffstats([s["logprobs"] for r, s in gflat], [x["logprobs"] for x in gsc])
    log("noise refs:", res["noise_score_alone_vs_batch"], res["noise_generate_alone_vs_batch"],
        "score vs greedy:", res["score_vs_generate_greedy"])
    # (3) last token through the sampler (path used for len == max_model_len) vs the prompt-logprob path
    fv = S.score(seqs, starts, final_via_sampler=True)
    # (prompts are 1 token shorter, so the last prefill chunk has a different size: not necessarily bit-identical)
    res["final_via_sampler"] = dict(
        all_but_last=diffstats([x["logprobs"][:-1] for x in sc1], [y["logprobs"][:-1] for y in fv]),
        last_abs_diff=[float(abs(x["logprobs"][-1] - y["logprobs"][-1])) for x, y in zip(sc1, fv)])
    # (4) throughput: 32 sequences of ~8k tokens (chat prompt + the model's own text), one of exactly max_model_len
    L = S.max_model_len
    body = [t_ for _, s in flat + gflat for t_ in s["token_ids"]]
    rng = np.random.default_rng(5)
    lens = rng.integers(L - 512, L, size=32)
    lens[0] = L
    tseqs, tstarts, off = [], [], 0
    for p, n_ in zip(prompts, lens):
        pid = S.tokenizer.encode(S.render(p), add_special_tokens=False)
        need = int(n_) - len(pid)
        tail = [body[(off + k) % len(body)] for k in range(need)]
        off += need
        tseqs.append(pid + tail)
        tstarts.append(len(pid))
    res["throughput_input"] = dict(sequences=len(tseqs), total_tokens=int(sum(map(len, tseqs))),
                                   min_len=int(min(map(len, tseqs))), max_len=int(max(map(len, tseqs))))
    S.gpu_memory(reset_peak=True)
    outs = []
    for rep in range(2):
        o = S.score(tseqs, tstarts)
        outs.append(o)
        res[f"throughput_rep{rep}"] = dict(secs=round(S.last_score["seconds"], 2),
                                           input_tok_per_s=round(S.last_score["tok_per_s"], 1),
                                           scored_tok_per_s=round(S.last_score["scored_tokens"] / S.last_score["seconds"], 1))
        log(f"score throughput rep{rep}:", res[f"throughput_rep{rep}"])
    res["throughput_reps_bit_identical"] = all(np.array_equal(x["logprobs"], y["logprobs"]) for x, y in zip(*outs))
    res["max_len_sequence"] = dict(len=len(tseqs[0]), n_tokens=outs[0][0]["n_tokens"], mean=outs[0][0]["mean"])
    res["memory_after_throughput"] = S.gpu_memory()
    res["mean_logprob_generated_part"] = [round(x["mean"], 4) for x in sc1]
    path = f"{OUT}/score_tp{a.tp}_{STAMP}.json"
    json.dump(res, open(path, "w"), indent=1, default=str)
    log(json.dumps(res, indent=1, default=str))


# ------------------------------------------------------------------------------------------------ vision
def mm_items():
    """Fixed image prompts from the multimodal test sets: single image (CV-Bench, BabyVision), multi-image (BLINK, ERQA)."""
    picks = []
    for f, ok in (("cvbench", lambda r: True), ("babyvision", lambda r: True),
                  ("blink", lambda r: len(r["image_paths"]) >= 2), ("erqa", lambda r: len(r["image_paths"]) >= 3)):
        for line in open(f"{MMDATA}/sets/{f}.jsonl"):
            r = json.loads(line)
            if ok(r):
                picks.append(r)
                break
    return picks


def vprobe_fn(items):
    def vprobe(S, max_tokens=24, prompts=None):
        res = S.generate(prompts or items, sampling={"temperature": 0.0}, max_tokens=max_tokens, enable_thinking=False,
                         logprobs=5)
        return [dict(ids=s["token_ids"], lp=s["logprobs"], text=s["content"]) for r in res for s in r["samples"]]
    return vprobe


def vision_tests(rng):
    v = "visual."
    tests = [
        ("v_block13_attn_scale", [{"op": "scale", "param": v + "blocks.13.attn.proj.weight", "alpha": 0.5},
                                  {"op": "scale", "param": v + "blocks.13.attn.proj.bias", "alpha": 0.5}]),
        ("v_block20_mlp_zero", [{"op": "set", "param": v + "blocks.20.mlp.linear_fc2.weight", "value": 0.0},
                                {"op": "set", "param": v + "blocks.20.mlp.linear_fc2.bias", "value": 0.0}]),
        ("v_block5_heads_v_scale0", {"op": "scale", "param": v + "blocks.5.attn.qkv", "head": [2, 3], "part": "v", "alpha": 0.0}),
        ("v_block8_qkv_bias_k_add", {"op": "add", "param": v + "blocks.8.attn.qkv.bias", "head": [0], "part": "k", "value": 0.5}),
        ("v_block26_proj_head_cols0", {"op": "scale", "param": v + "blocks.26.attn.proj", "head": [15], "alpha": 0.0}),
        ("v_block24_fc1_neurons", [{"op": "scale", "param": v + "blocks.24.mlp.linear_fc1", "neuron": [1, 50, 999, 4303], "alpha": 3.0},
                                   {"op": "scale", "param": v + "blocks.24.mlp.linear_fc1.bias", "neuron": [1, 50, 999, 4303], "alpha": 3.0}]),
        ("v_block0_norm1_bias_add", {"op": "add", "param": v + "blocks.0.norm1.bias", "value": 0.3}),
        ("merger_norm_scale", {"op": "scale", "param": v + "merger.norm.weight", "alpha": 1.3}),
        ("merger_fc1_neurons", {"op": "scale", "param": v + "merger.linear_fc1", "neuron": [10, 20, 30, 4607], "alpha": 2.0}),
        ("merger_fc2_neuron_cols", {"op": "scale", "param": v + "merger.linear_fc2", "neuron": [7, 4000], "alpha": 0.0}),
        ("merger_fc2_out_project", {"op": "project_out", "param": v + "merger.linear_fc2", "side": "out", "u": unit(rng, 5120)}),
        ("merger_fc1_in_amplify", {"op": "amplify", "param": v + "merger.linear_fc1", "side": "in", "v": unit(rng, 4608), "alpha": 2.0}),
        ("merger_fc2_rank1", {"op": "rank1", "param": v + "merger.linear_fc2", "u": unit(rng, 5120), "v": unit(rng, 4608), "alpha": 5.0}),
        ("patch_embed_scale_5d", {"op": "scale", "param": v + "patch_embed.proj.weight", "alpha": 1.5}),
        ("pos_embed_rows_set", {"op": "set", "param": v + "pos_embed", "rows": list(range(0, 2304, 7)), "value": 0.0}),
    ]
    zeros = [("v_identity_scale", {"op": "scale", "param": v + "blocks.3.attn.proj.weight", "alpha": 1.0})]
    return tests, zeros


def phase_vision(a):
    from surgery import Surgeon, VIS_HF
    S = Surgeon(a.model, tp=a.tp, multimodal=True, **engine_kwargs(a))
    ck = Ckpt(a.model)
    results = dict(phase=f"vision_tp{a.tp}", load_secs=round(S.load_seconds, 1), engine_args=S.engine_args,
                   unmapped_params=S.unmapped_params,
                   vision_param_dtypes={k: d for k, d in S.param_dtypes.items() if k.startswith("visual.")})
    path = f"{OUT}/vision_tp{a.tp}_{STAMP}.json"
    log("unmapped vLLM params:", S.unmapped_params)
    try:
        vn = [n for n in S.layout.names if n.startswith(VIS_HF)]
        t0 = time.time()
        results["layout_vision"] = summarize_layout(S.verify_layout(vn))
        results["layout_vision"][0]["n_tensors"] = len(vn)
        log("vision layout:", results["layout_vision"], f"{time.time() - t0:.1f}s")
        items = mm_items()
        results["items"] = [dict(id=r["id"], n_images=len(r["image_paths"])) for r in items]
        vp = vprobe_fn(items)
        b1, b2 = vp(S), vp(S)
        results["determinism"] = cmp(b1, b2)
        results["probe_text"] = [x["text"] for x in b1]
        t1, t2 = probe(S), probe(S)
        results["text_determinism"] = cmp(t1, t2)
        if a.ref:
            results["text_probe_vs_text_only_ref"] = cmp(json.load(open(a.ref))["base"], t1)
        log("determinism:", results["determinism"], "text:", results["text_determinism"],
            "text vs text-only engine ref:", results.get("text_probe_vs_text_only_ref"))
        t0 = time.time()
        fp0 = S.model_fingerprint()
        full_fp = time.time() - t0 < 10
        results["fingerprint_secs"] = round(time.time() - t0, 2)
        rng = np.random.default_rng(11)
        tests, zeros = vision_tests(rng)
        recs = []
        zero_names = {n for n, _ in zeros}
        for name, e in tests + zeros:
            try:
                recs.append(op_test(S, ck, name, e, b1, fp0, "zero" if name in zero_names else "change", full_fp,
                                    probe_fn=vp))
            except Exception as ex:
                traceback.print_exc()
                recs.append(dict(name=name, PASS=False, error=repr(ex)))
                S.revert_all()
        results["op_tests"] = recs
        # a vision edit leaves text-only prompts bit-identical (they never run the vision tower)
        h = S.apply(tests[0][1])
        results["vision_edit_text_probe_identical"] = cmp(t1, probe(S))["identical"]
        S.revert(h)
        # control: without the encoder-cache reset, a vision edit is invisible to an image already seen (stale embeddings);
        # the revert above emptied the cache, so see the images once first
        vp(S)
        reset = S.reset_encoder_cache
        S.reset_encoder_cache = lambda: None
        try:
            h = S.apply(tests[0][1])
            stale = vp(S)
            S.revert(h)
        finally:
            S.reset_encoder_cache = reset
        S.reset_encoder_cache()
        results["encoder_cache_control"] = dict(edit_seen_without_reset=not cmp(b1, stale)["identical"],
                                                note="False = stale cached embeddings reused, so the reset is required")
        # score(images=...) reproduces the generation logprobs of image prompts; moves under an edit; exact after revert
        g = S.generate(items, sampling={"temperature": 0.0}, max_tokens=24, enable_thinking=False, logprobs=0)
        seqs = [r["prompt_token_ids"] + r["samples"][0]["token_ids"] for r in g]
        starts = [len(r["prompt_token_ids"]) for r in g]
        imgs = [r["image_paths"] for r in items]
        sc0 = S.score(seqs, starts, images=imgs)
        ad = [np.abs(np.asarray(s["logprobs"]) - np.asarray(r["samples"][0]["logprobs"])) for s, r in zip(sc0, g)]
        h = S.apply(tests[10][1])
        sc1 = S.score(seqs, starts, images=imgs)
        S.revert(h)
        sc2 = S.score(seqs, starts, images=imgs)
        results["score_images"] = dict(
            mean_abs_vs_generation=[round(float(x.mean()), 5) for x in ad], max_abs_vs_generation=[round(float(x.max()), 4) for x in ad],
            median_abs_vs_generation=[float(np.median(x)) for x in ad], prompt_tokens=starts,
            edit_moves=[round(x["mean"] - y["mean"], 5) for x, y in zip(sc1, sc0)],
            after_revert_bit_identical=all(np.array_equal(x["logprobs"], y["logprobs"]) for x, y in zip(sc0, sc2)))
        try:
            S.score(seqs[:1], starts[:1], images=[imgs[0] + imgs[0]])
            results["score_images"]["wrong_image_count_rejected"] = False
        except ValueError:
            results["score_images"]["wrong_image_count_rejected"] = True
        log("score(images):", results["score_images"])
        # thinking generation (as the loop runs it) under a vision edit, then a bit-exact revert
        h = S.apply(tests[1][1])
        gt = S.generate(items[:2], n=2, max_tokens=1024, reasoning_effort="low")
        rep = S.revert(h)
        results["thinking_under_edit"] = dict(revert=rep, samples=[dict(
            num_tokens=s["num_tokens"], finish=s["finish_reason"], think_closed=s["think_closed"],
            content=s["content"][-200:]) for r in gt for s in r["samples"]])
        results["final_full_fingerprint_equal"] = S.model_fingerprint() == fp0
        results["final_probe_identical"] = cmp(b1, vp(S))["identical"]
        log("final full-model fingerprint equal:", results["final_full_fingerprint_equal"],
            "final probe identical:", results["final_probe_identical"])
    finally:
        json.dump(results, open(path, "w"), indent=1, default=str)
        log("results ->", path)
    recs = results.get("op_tests", [])
    sc = results.get("score_images", {})
    ok = [r["PASS"] for r in recs] + [
        results.get("determinism", {}).get("identical", False), results.get("vision_edit_text_probe_identical", False),
        sc.get("after_revert_bit_identical", False), max(sc.get("mean_abs_vs_generation", [9])) < 0.02,
        results.get("final_full_fingerprint_equal", False), results.get("final_probe_identical", False),
        all(c == 0 for c in [len(results.get("layout_vision", [{}])[0].get("mismatches", [1]))])]
    log(f"SUMMARY: {sum(r['PASS'] for r in recs)}/{len(recs)} vision op tests passed; all checks pass: {all(ok)}")
    for r in recs:
        if not r["PASS"]:
            log("FAILED:", r["name"], r.get("error", ""))


def phase_textref(a):
    """Default (text-only) engine after the multimodal changes: greedy probe vs a reference.  --ref: a core-phase
    export_ref_*.json or a file written by --write-ref.  --engine-dir legacy: run the pre-multimodal surgery.py
    (engine/legacy/surgery.py) instead, e.g. to write a same-machine reference with --write-ref."""
    if a.engine_dir:
        sys.path.insert(0, os.path.join(HERE, a.engine_dir) if not os.path.isabs(a.engine_dir) else a.engine_dir)
    import surgery
    from surgery import Surgeon
    log("surgery module:", surgery.__file__)
    S = Surgeon(a.model, tp=a.tp, **engine_kwargs(a))
    b = probe(S)
    if a.write_ref:
        json.dump(dict(base=b, engine_args=S.engine_args, surgery=surgery.__file__), open(a.write_ref, "w"))
        log("reference written ->", a.write_ref)
        return
    ref = json.load(open(a.ref))
    res = dict(phase="textref", ref=a.ref, ref_engine_args=ref["engine_args"], engine_args=S.engine_args,
               vs_ref=cmp(ref["base"], b), unmapped_params=S.unmapped_params, n_specs=len(S.layout.specs))
    h = S.apply({"op": "scale", "param": "layers.0.linear_attn.A_log", "head": [3, 4], "alpha": 0.8})
    S.revert(h)
    res["after_edit_revert_vs_ref"] = cmp(ref["base"], probe(S))
    path = f"{OUT}/textref_{STAMP}.json"
    json.dump(res, open(path, "w"), indent=1, default=str)
    log(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["core", "reload", "dup", "bench", "tp2", "score", "vision", "textref"])
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--max-num-seqs", type=int, default=16)
    ap.add_argument("--max-num-batched-tokens", type=int, default=4096)
    ap.add_argument("--ckpt")
    ap.add_argument("--ref")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--gen-tokens", type=int, default=4096)
    ap.add_argument("--skip-extra", action="store_true")
    ap.add_argument("--control-tokens", type=int, default=1024)
    ap.add_argument("--engine-dir")                                 # textref: import surgery.py from this dir (e.g. legacy)
    ap.add_argument("--write-ref")                                  # textref: write the probe as a reference and stop
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    log("args:", vars(a))
    if a.phase == "tp2":
        a.tp = 2
        a.phase = "core"
    {"core": phase_core, "reload": phase_reload, "dup": phase_dup, "bench": phase_bench,
     "score": phase_score, "vision": phase_vision, "textref": phase_textref}[a.phase](a)
