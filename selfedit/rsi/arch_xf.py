"""Edit menu, validation, dosing and the random-edit control of the RSI loop for the non-Qwen architectures
(Gemma 4 dense, OLMo 3).  rsi/loop.py keeps its Qwen3.5 definitions unchanged and swaps these in when the model's
config.json names another family (loop.py, end of the module).

What the menu offers follows what each architecture computes (vLLM 0.33 model code):
- Gemma 4 (model_executor/models/gemma4.py): every block is sandwich-normed.  The attention block reads
  input_layernorm(x) and its q / k / v go through per-head RMS norms (q_norm, k_norm with gains; v_norm without,
  gemma4.py:432-446), attention uses scale 1.0 (gemma4.py:397-400), the output goes through post_attention_layernorm
  before the residual add; the MLP reads pre_feedforward_layernorm(x) and writes through post_feedforward_layernorm
  (gemma4.py:736-756); then the whole stream is multiplied by layer_scalar (gemma4.py:774).  All norms are plain
  x * w RMSNorms.  Consequences: scaling a whole o_proj / down_proj / up_proj, a head's q / k / v rows, or
  input_layernorm is undone by the following norm (no-op), so those are not offered; q_norm scales every head's
  attention logits (a sharpness knob); o_proj head columns scale what one head writes.
- OLMo 3 (Transformers backend, transformers models/olmo3/modeling_olmo3.py): post-norm blocks (x + norm(block(x))),
  q_norm / k_norm over the whole q / k projection (one gain per channel, so per-head blocks of it scale that head's
  logits), no pre-norms.  Scaling a whole o_proj / down_proj / up_proj is undone by the post-norm.
Residual directions are applied with space="residual" (engine/surgery.py Surgeon.expand): the engine maps the
residual-stream direction through the adjacent norm gain, so project_out removes it from what a block writes (writers)
or makes a block's read independent of it (readers)."""
import json, os, re, sys


_ENG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "engine")


def family(model_dir):
    """Family of the model at model_dir, by the rule of engine/surgery.py model_family() (read here from config.json so
    that importing loop.py for a Qwen model imports nothing new: tests swap in fake engines, e.g. rsi/t_confirm.py);
    loop.py's default (Qwen3.5) when the directory has no config.json (its menu needs nothing from the config)."""
    p = os.path.join(model_dir, "config.json")
    if not os.path.exists(p):
        return "qwen3_5"
    raw = json.load(open(p))
    if raw.get("model_type") == "gemma4" or raw.get("text_config", {}).get("model_type") == "gemma4_text":
        return "gemma4"
    if raw.get("model_type") == "olmo3":
        return "olmo3"
    return "qwen3_5"


def text_config(model_dir):
    if _ENG not in sys.path:
        sys.path.append(_ENG)
    import surgery                                    # numpy-only module scope (torch / vllm load lazily)
    assert surgery.model_family(model_dir) == family(model_dir)
    return surgery.text_config(model_dir)


def sampling(model_dir):
    """Fixed decoding of a lineage = the model's own generation_config.json recommendation (as loop.py's Qwen SAMPLING
    is Qwen's), with the remaining penalties neutral.  top_k 0 = no top-k cut (vLLM sampling_params.py:318-320)."""
    g = json.load(open(os.path.join(model_dir, "generation_config.json")))
    return dict(temperature=float(g.get("temperature", 1.0)), top_p=float(g.get("top_p", 1.0)),
                top_k=int(g.get("top_k", 0) or 0), min_p=0.0, presence_penalty=0.0, repetition_penalty=1.0)


def _as_list(x):
    return x if isinstance(x, list) else [x]


DOSE = ("Dose: the engine sweeps every edit (except zero_block) at 1x, 2x, 4x and 8x of its deviation from identity "
        "(scale alpha -> 1 + k(alpha - 1), amplify alpha -> k * alpha; project_out is alpha = -1), within hard limits, "
        "and keeps the dose with the best self-likelihood margin among doses that do not move your own traces too far. "
        "Lesson from an earlier lineage on another model (Qwen3.8-27B, generation 1): single-layer edits with small "
        "alpha changed its own-trace log-prob by only ~1e-4 nats/token and had no measurable effect; edits need to be "
        "strong enough to matter.")

EDIT_FORMAT = """Edit JSON (one object per edit; keep why and predict under 40 words each):
{"op": "scale" | "project_out" | "amplify" | "zero_block",
 "param": "<HF name>", "head": h | [h...], "neuron": n | [n...],
 "alpha": x, "side": "out" | "in", "u" or "v": "dir:...", "layer": L, "parts": [...],
 "writers": ["attn", "mlp"], "layers": [first, last],
 "why": "<the internal evidence>", "predict": "<expected behaviour change>", "confidence": 0..1}"""


class Arch:
    """Menu text and the edit functions of one non-Qwen model (loop.py binds them in place of its Qwen ones)."""

    def __init__(self, model_dir, args, resolve_dir):
        self.fam = family(model_dir)
        if self.fam not in ("gemma4", "olmo3"):
            raise ValueError(f"arch_xf covers gemma4 / olmo3, not {self.fam}")
        self.c = c = text_config(model_dir)
        self.args, self.resolve_dir = args, resolve_dir
        self.SAMPLING = sampling(model_dir)
        self.L, self.H, self.I, self.nq = c.L, c.H, c.I, c.nq
        self.full = [l for l, t in enumerate(c.layer_types) if t == "full_attention"]
        self.WRITERS = {"attn": lambda L: f"layers.{L}.self_attn.o_proj", "mlp": lambda L: f"layers.{L}.mlp.down_proj"}
        # scale targets: name -> (selection key or None, (lo, hi) bounds, dose clip)
        if self.fam == "gemma4":
            gains = ["post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm"]
            self.PARAM_RE = re.compile(r"^layers\.(\d+)\.(mlp\.(down|gate|up)_proj|self_attn\.(q|k|v|o)_proj|"
                                       r"self_attn\.q_norm|layer_scalar|" + "|".join(gains) + r")$")
            self.gains = gains + ["layer_scalar", "self_attn.q_norm"]
            self.nkv_max = None
        else:
            gains = ["post_attention_layernorm", "post_feedforward_layernorm"]
            self.PARAM_RE = re.compile(r"^layers\.(\d+)\.(mlp\.(down|gate|up)_proj|self_attn\.(q|k|v|o)_proj|"
                                       r"self_attn\.(q|k)_norm|" + "|".join(gains) + r")$")
            self.gains = gains
            self.nkv_max = c.nkv
        self.MENU = self._menu()
        self.DIR_HELP = ("Directions are referenced by name: \"dir:<contrast>:<kind>:<layer>\" or "
                         "\"dir:<contrast>:<kind>:<layer>:svd<k>\",\nkind in {resid, mixer, mlp}; a direction lives in the "
                         f"{self.H}-d residual space, so use it on the out side of o_proj/down_proj or the in side of the "
                         "input projections (the engine maps it through the block's norm gain). Available contrasts: {names}.")
        self.EDIT_FORMAT = EDIT_FORMAT

    # ------------------------------------------------------------------------------------------------ menu text
    def _menu(self):
        c, L1 = self.c, self.L - 1
        if self.fam == "gemma4":
            head = (f"layer L = 0..{L1}, full (global) attention iff L % 6 == 5 ({len(self.full)} layers), else "
                    f"sliding-window attention ({c.raw.get('text_config', c.raw).get('sliding_window')} tokens); every "
                    "block is sandwich-normed (a norm before and after it)")
            l2 = (f"attention heads `layers.L.self_attn.o_proj` with \"head\": h (0..{c.nq - 1}) — how strongly head h "
                  "writes (alpha 0 silences it); attention sharpness `layers.L.self_attn.q_norm` (no head: multiplies the "
                  "attention logits of every head in layer L; q and k are RMS-normalised per head and attention uses "
                  "scale 1); block gains `layers.L.post_attention_layernorm` (attention output), "
                  "`pre_feedforward_layernorm` (MLP input), `post_feedforward_layernorm` (MLP output) and "
                  "`layers.L.layer_scalar` (multiplies the whole residual stream after layer L)")
            l3_in = ("in side of `self_attn.q_proj/k_proj/v_proj` (full-attention layers have no v_proj: their k_proj "
                     "gives both keys and values), `mlp.gate_proj/up_proj`")
        else:
            head = (f"layer L = 0..{L1}, full attention iff L % 4 == 3 ({len(self.full)} layers), else sliding-window "
                    f"attention ({c.raw.get('sliding_window')} tokens); post-norm blocks: x <- x + norm(block(x))")
            l2 = (f"attention heads `layers.L.self_attn.o_proj` with \"head\": h (0..{c.nq - 1}) — how strongly head h "
                  "writes (alpha 0 silences it); per-head attention sharpness `layers.L.self_attn.q_norm` with \"head\": "
                  f"h (0..{c.nq - 1}) (multiplies head h's attention logits) and `self_attn.k_norm` with \"head\": g "
                  f"(0..{c.nkv - 1}) (the {c.nq // c.nkv} query heads sharing key group g); block output gains "
                  "`layers.L.post_attention_layernorm`, `post_feedforward_layernorm`")
            l3_in = "in side of `self_attn.q_proj/k_proj/v_proj`, `mlp.gate_proj/up_proj`"
        return f"""| level | targets (HF names; {head}) | ops | bounds |
|---|---|---|---|
| L1 neurons | `layers.L.mlp.down_proj` with "neuron": n (0..{c.I - 1}) (output column of one neuron); `layers.L.mlp.gate_proj` / `up_proj` with "neuron": n | scale (alpha in [0, 2]) | at most 64 neurons per edit |
| L2 heads and gains | {l2} | scale (heads of o_proj: alpha in [0, 2]; sharpness and gains: [0.5, 1.5]) | at most 16 heads per edit |
| L3 directions | out side of `self_attn.o_proj`, `mlp.down_proj`; {l3_in} — direction given by name (see library) | project_out, amplify (W <- (I + alpha u u^T) W; alpha in [-2, 2]: -1 removes the component, -2 reflects it, +1 doubles it) | one layer per edit |
| L3g global directions | every residual writer of a layer range at once: `writers` "attn" (o_proj) and/or "mlp" (down_proj), `layers` [first, last] inclusive | project_out, amplify (alpha in [-2, 2]); "u": "dir:<contrast>:<kind>:<L>" uses one direction everywhere, "dir:<contrast>:<kind>:*" uses each layer's own direction | any range (directional ablation over all layers = [0, {L1}]) |
| L4 structure | whole blocks | zero_block {{"layer": L, "parts": ["attn"] or ["mlp"]}} | one block per edit |

{DOSE}"""

    # ------------------------------------------------------------------------------------------------ validation
    def _dir(self, ref, L):
        v = self.resolve_dir(ref, L)
        if v.size != self.H:
            raise ValueError(f"direction has {v.size} dims, this model's residual stream has {self.H}")
        return v

    def validate(self, e):
        """Return (clean edit for the engine, None) or (None, reason)."""
        op, L1 = e.get("op"), self.L - 1
        if op == "zero_block":
            L = int(e.get("layer", -1))
            parts = _as_list(e.get("parts", []))
            if not (0 <= L <= L1) or not parts or any(p not in ("attn", "mlp") for p in parts) or len(parts) > 1:
                return None, f"zero_block needs layer 0..{L1} and exactly one part"
            return {"op": op, "layer": L, "parts": parts}, None
        if op in ("project_out", "amplify") and "writers" in e:
            writers = _as_list(e.get("writers"))
            lay = _as_list(e.get("layers", []))
            if not writers or any(w not in self.WRITERS for w in writers):
                return None, "writers must be a subset of ['attn', 'mlp']"
            if len(lay) != 2 or not 0 <= int(lay[0]) <= int(lay[1]) <= L1:
                return None, f"layers must be [first, last] with 0 <= first <= last <= {L1}"
            ref = e.get("u") or e.get("dir")
            try:
                self._dir(ref, int(lay[0]))
            except Exception as ex:
                return None, f"direction {ref!r}: {ex}"
            a = -1.0 if op == "project_out" else float(e.get("alpha", 0.0))
            if not -2.0 <= a <= 2.0:
                return None, "amplify alpha outside [-2, 2]"
            return {"op": op, "writers": writers, "layers": [int(lay[0]), int(lay[1])], "dir_name": ref, "alpha": a}, None
        p = str(e.get("param", ""))
        for pre in ("model.language_model.", "language_model.model.", "model."):
            if p.startswith(pre):
                p = p[len(pre):]
                break
        p = p[:-len(".weight")] if p.endswith(".weight") else p
        m = self.PARAM_RE.match(p)
        if not m or not 0 <= int(m.group(1)) <= L1:
            return None, f"param {p!r} not in the edit menu"
        L = int(m.group(1))
        rest = p.split(".", 2)[2]
        if self.fam == "gemma4" and rest == "self_attn.v_proj" and L in self.full:
            return None, f"layer {L} is full attention: it has no v_proj (k_proj gives keys and values)"
        if "part" in e:
            return None, "no head parts in this model (q, k, v and o are separate matrices)"
        out = {"op": op, "param": p}
        for k in ("head", "neuron"):
            if k in e:
                out[k] = sorted({int(i) for i in e[k]}) if isinstance(e[k], list) else int(e[k])
        if "head" in out and len(_as_list(out["head"])) > 16:
            return None, "at most 16 heads per edit"
        if "neuron" in out and len(_as_list(out["neuron"])) > 64:
            return None, "at most 64 neurons per edit"
        if "neuron" in out:
            if not rest.startswith("mlp."):
                return None, "neuron selects MLP neurons (mlp.down_proj / gate_proj / up_proj)"
            if not all(0 <= n < self.I for n in _as_list(out["neuron"])):
                return None, f"neurons are 0..{self.I - 1}"
        if "head" in out:
            nmax = self.nkv_max if rest == "self_attn.k_norm" else self.nq
            if rest not in ("self_attn.o_proj",) + (("self_attn.q_norm", "self_attn.k_norm") if self.fam == "olmo3" else ()):
                return None, (f"head selects o_proj heads{' and q_norm / k_norm heads' if self.fam == 'olmo3' else ''}; "
                              f"{rest} takes no head")
            if not all(0 <= h < nmax for h in _as_list(out["head"])):
                return None, f"{rest} heads are 0..{nmax - 1}"
        if op == "scale":
            if ("head" in out or "neuron" in out) and rest.endswith("_proj"):
                lo, hi = 0.0, 2.0
            elif rest in self.gains or (self.fam == "olmo3" and rest in ("self_attn.q_norm", "self_attn.k_norm")):
                lo, hi = 0.5, 1.5
            elif rest.startswith("mlp."):
                return None, ("scale an MLP matrix through neurons (\"neuron\"): scaling a whole down_proj / up_proj is "
                              "undone by the norm after the block")
            elif rest == "self_attn.o_proj":
                return None, "scale o_proj through heads (\"head\"): scaling all of it is undone by the norm after the block"
            else:
                why = ("q, k and v are RMS-normalised per head" if self.fam == "gemma4" else
                       "q and k pass through q_norm / k_norm, and v's scale is undone by the norm after the block")
                return None, (f"scaling {rest} has no effect here ({why}); use q_norm"
                              f"{' / k_norm heads' if self.fam == 'olmo3' else ''} for sharpness or o_proj heads for output")
            a = float(e.get("alpha", 1.0))
            if not lo <= a <= hi:
                return None, f"alpha {a} outside [{lo}, {hi}]"
            out["alpha"] = a
        elif op in ("project_out", "amplify"):
            if "head" in out or "neuron" in out:
                return None, "directional edits act on a whole matrix (no head / neuron selection)"
            side = e.get("side", "out")
            if side not in ("out", "in"):
                return None, "side must be 'out' or 'in'"
            key = "u" if side == "out" else "v"
            ref = e.get(key) or e.get("u") or e.get("v")
            out_side = rest in ("self_attn.o_proj", "mlp.down_proj")
            in_side = rest in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "mlp.gate_proj", "mlp.up_proj")
            if not (out_side or in_side):
                return None, f"{p} does not read or write the residual stream"
            if (side == "out") != out_side:
                return None, f"{p} takes residual directions on the {'out' if out_side else 'in'} side"
            try:
                self._dir(ref, L)
            except Exception as ex:
                return None, f"direction {ref!r}: {ex}"
            out["side"], out["dir_name"] = side, ref
            if op == "amplify":
                a = float(e.get("alpha", 0.0))
                if not -2.0 <= a <= 2.0:
                    return None, "amplify alpha outside [-2, 2]"
                out["alpha"] = a
        else:
            return None, f"op {op!r} not in the menu"
        return out, None

    # ------------------------------------------------------------------------------------------------ engine form
    def engine_edit(self, e):
        """Clean edit -> what the engine applies (directions resolved, in residual space; a global edit becomes one edit
        per writer)."""
        if "writers" in e:
            return [{"op": "amplify", "param": self.WRITERS[w](L), "side": "out", "alpha": e["alpha"],
                     "u": self._dir(e["dir_name"], L), "space": "residual"}
                    for L in range(e["layers"][0], e["layers"][1] + 1) for w in e["writers"]]
        if e["op"] in ("project_out", "amplify"):
            L = int(e["param"].split(".")[1])
            return dict({k: v for k, v in e.items() if k != "dir_name"}, space="residual",
                        **{"u" if e["side"] == "out" else "v": self._dir(e["dir_name"], L)})
        return e

    def dosed(self, e, k):
        """The same edit with k times its deviation from identity, clipped to hard limits; None if it cannot be dosed."""
        if k == 1:
            return e
        op, d = e["op"], dict(e)
        if op == "zero_block":
            return None
        if op == "scale":
            lo, hi = (0.0, 3.0) if ("neuron" in e or "head" in e) and e["param"].endswith("_proj") else (0.25, 2.0)
            d["alpha"] = round(min(hi, max(lo, 1 + k * (e["alpha"] - 1))), 4)
        else:
            a = -1.0 if op == "project_out" else e["alpha"]
            d["op"], d["alpha"] = "amplify", round(max(-self.args.alpha_cap, min(self.args.alpha_cap, k * a)), 4)
        return d

    def random_edit(self, rng, names):
        """Blind-search control: a random edit from the same menu (no evidence, no reasoning)."""
        r, L = rng.random(), rng.randrange(self.L)
        if r < 0.2:
            return {"op": "scale", "param": f"layers.{L}.mlp.down_proj", "alpha": round(rng.uniform(0, 2), 2),
                    "neuron": sorted(rng.sample(range(self.I), rng.choice([8, 16, 32, 64])))}
        if r < 0.45:
            kinds = ["ohead", "sharp", "gain"]
            kind = rng.choice(kinds)
            if kind == "ohead":
                return {"op": "scale", "param": f"layers.{L}.self_attn.o_proj",
                        "head": sorted(rng.sample(range(self.nq), rng.randint(1, 8))), "alpha": round(rng.uniform(0.5, 1.5), 2)}
            if kind == "sharp":
                e = {"op": "scale", "param": f"layers.{L}.self_attn.q_norm", "alpha": round(rng.uniform(0.5, 1.5), 2)}
                if self.fam == "olmo3":
                    e["head"] = sorted(rng.sample(range(self.nq), rng.randint(1, 8)))
                return e
            g = rng.choice([x for x in self.gains if x != "self_attn.q_norm"])
            return {"op": "scale", "param": f"layers.{L}.{g}", "alpha": round(rng.uniform(0.5, 1.5), 2)}
        c, kind = rng.choice(names), rng.choice(["resid", "mixer", "mlp"])
        if r < 0.7:
            return {"op": "amplify", "param": self.WRITERS[rng.choice(["attn", "mlp"])](L), "side": "out",
                    "u": f"dir:{c}:{kind}:{L}", "alpha": round(rng.uniform(-2, 2), 2)}
        if r < 0.95:
            a = rng.randrange(self.L)
            return {"op": "amplify", "writers": rng.choice([["attn"], ["mlp"], ["attn", "mlp"]]),
                    "layers": [a, rng.randrange(a, self.L)], "u": f"dir:{c}:{kind}:*", "alpha": round(rng.uniform(-2, 2), 2)}
        return {"op": "zero_block", "layer": L, "parts": [rng.choice(["attn", "mlp"])]}
