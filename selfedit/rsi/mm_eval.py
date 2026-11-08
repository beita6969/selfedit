"""Multimodal hooks for rsi/loop.py (--mm, lineage B: visual perception).

Pool / guard items are prep_mm.py records ({"id", "kind", "prompt", "image_paths", "answer", "options", ...}); evaluate()
sends them as image prompts (the message format of code/mm/gen_mm.py, via surgery.image_message) and grades with
code/mm/grade_mm.py. Items without "image_paths" (e.g. a text guard pool) are prompted and graded exactly as in text mode.
L5 of the edit menu: vision-tower parameters (block residual-write scale / zero, merger hidden units, merger in/out
directions). Text mode never imports this module.
"""
import json, os, sys

import numpy as np

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(CODE, "mm")]
import grade_mm                                   # noqa: E402
from grade import correct as text_correct, looped as text_looped   # noqa: E402
from surgery import image_message                 # noqa: E402

DEPTH, MERGER = 27, 4608                          # Qwen3.8-27B vision tower: 27 blocks, merger width 4608
NSVD = 32                                         # "wsvd:k", k < NSVD
FC1, FC2 = "visual.merger.linear_fc1", "visual.merger.linear_fc2"
_ctx = {}

MENU_L5 = (
    "| L5 vision tower (images only) | the 27-block ViT that encodes every image (blocks B = 0..26, width 1152, 16 heads): "
    "`visual.blocks.B.attn` / `visual.blocks.B.mlp` = that block's residual write (attn.proj / mlp.linear_fc2, weight "
    "and bias); the patch merger: `visual.merger.linear_fc1` (4608 -> 4608; input = 4 neighbouring patches of the last "
    "block, concatenated) and `visual.merger.linear_fc2` (4608 -> 5120; writes the image tokens into your residual "
    "stream before layer 0); merger hidden units \"neuron\": n (0..4607) = rows of linear_fc1 / columns of linear_fc2 | "
    "block scale {\"op\": \"scale\", \"vblock\": B, \"part\": \"attn\" or \"mlp\", \"alpha\": a} (alpha in [0.5, 1.5]); "
    "block zero {\"op\": \"zero_block\", \"vblock\": B, \"parts\": [\"attn\"] or [\"mlp\"]}; merger units: scale with "
    "\"param\": linear_fc1 or linear_fc2, \"neuron\": [...] (alpha in [0, 2]); merger directions: project_out / amplify "
    "(alpha in [-2, 2]) on the out side of `visual.merger.linear_fc2` with \"u\": \"dir:<contrast>:resid:0\" (a library "
    "direction; the bias is transformed too, so the component really leaves the image tokens) or \"u\": \"wsvd:k\" (k-th "
    "output singular direction of that matrix, k = 0..31, 0 = strongest), and on the in side of "
    "`visual.merger.linear_fc1` with \"v\": \"wsvd:k\" (k-th input singular direction) | one block or one merger "
    "matrix per edit; at most 64 units |")


def setup(args, resolve_dir):
    """Called once by loop.py after parsing: the model dir (for the singular directions) and the library resolver."""
    _ctx.update(model=args.model, run_dir=args.run_dir, resolve_dir=resolve_dir)


def extend_menu(menu):
    """Insert the L5 row after the L4 row of loop.MENU."""
    lines = menu.split("\n")
    k = max(i for i, l in enumerate(lines) if l.startswith("| L4"))
    return "\n".join(lines[:k + 1] + [MENU_L5] + lines[k + 1:])


def is_vision(e):
    return isinstance(e, dict) and ("vblock" in e or str(e.get("param", "")).replace("model.", "", 1).startswith("visual."))


# ---------------------------------------------------------------- evaluation
PARTIAL = None


def evaluate(S, probs, n, budget, effort, sampling, seeds=None):
    prompts = [image_message(p["prompt"], p["image_paths"]) if p.get("image_paths") else p["prompt"] for p in probs]
    hook = None
    if PARTIAL:         # traces reach the disk as they finish, so a run can be read before it ends
        def hook(i, o):
            one = []
            _collect(one, [probs[i]], [o])
            with open(PARTIAL, "a") as f:
                for r in one:
                    f.write(json.dumps({k: v for k, v in r.items() if k != "seq"}, ensure_ascii=False) + "\n")
    outs = S.generate(prompts, n=n, sampling=sampling, max_tokens=budget, reasoning_effort=effort, seeds=seeds, on_output=hook)
    recs = []
    _collect(recs, probs, outs)
    return recs


def _safe(r, f):
    """A grade never fails an item (11-08: a '-15%' gold crashed stock items): a grader exception counts the answer wrong (False),
    is kept on the record (grade_error) and logged with the row id and the answer text. hboost grades again at read time from these same fields."""
    try:
        return f()
    except Exception as e:
        r["grade_error"] = f"{type(e).__name__}: {str(e)[:160]}"
        print(f"grade error on {r.get('id')} sample {r.get('sample')}: {r['grade_error']}; answer {str(r.get('answer'))[:80]!r}; text {str(r.get('content'))[-160:]!r}", flush=True)
        return False


def _collect(recs, probs, outs):
    for p, o in zip(probs, outs):
        mm = bool(p.get("image_paths"))
        for k, s in enumerate(o["samples"]):
            r = {"id": p["id"], "kind": p["kind"], "answer": p["answer"], "sample": k, "reasoning": s["reasoning"],
                 "content": s["content"], "finish": s["finish_reason"], "tokens": s["num_tokens"],
                 "seq": o["prompt_token_ids"] + s["token_ids"], "start": len(o["prompt_token_ids"])}
            if mm:
                r.update(options=p.get("options") or [], category=p.get("category"), images=list(p["image_paths"]))
                if p.get("precision") is not None:
                    r["precision"] = p["precision"]
                if r["finish"] != "stop" or not r["content"]:      # no text: wrong, no answer, before any grader runs (grade_mm.correct says the same)
                    r["ok"] = False
                else:
                    r["ok"] = _safe(r, lambda: bool(grade_mm.correct(r)[0]))
                r["loop"] = _safe(r, lambda: bool(grade_mm.looped(dict(r, reasoning=r["reasoning"] or "", content=r["content"] or ""))))
            else:
                r["ok"] = _safe(r, lambda: text_correct(r))
                r["loop"] = _safe(r, lambda: text_looped(r))
            recs.append(r)


def score_images(recs):
    """Per-trace image lists for Surgeon.score(images=...) (None for text traces)."""
    return [r.get("images") for r in recs]


# ---------------------------------------------------------------- L5 edits
def _wsvd():
    """Singular directions of the ORIGINAL merger matrices (read from the checkpoint, so "wsvd:k" names one fixed vector
    for the whole lineage, whatever edits are active). Cached in the run dir."""
    if "wsvd" in _ctx:
        return _ctx["wsvd"]
    path = os.path.join(_ctx["run_dir"], "merger_wsvd.npz")
    if not os.path.exists(path):
        import torch
        from safetensors import safe_open
        wm = json.load(open(os.path.join(_ctx["model"], "model.safetensors.index.json")))["weight_map"]
        out = {}
        for name, key, side in ((FC2, "fc2_out", "U"), (FC1, "fc1_in", "V")):
            hf = "model." + name + ".weight"
            with safe_open(os.path.join(_ctx["model"], wm[hf]), "pt", device="cpu") as fh:
                W = fh.get_tensor(hf).float()
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            U, Sv, Vh = torch.linalg.svd(W.to(dev), full_matrices=False)
            vecs = U[:, :NSVD].T if side == "U" else Vh[:NSVD]
            out[key] = vecs.double().cpu().numpy()
            out[key + "_sv"] = Sv[:NSVD].double().cpu().numpy()
        np.savez(path + ".tmp.npz", **out)
        os.replace(path + ".tmp.npz", path)
    z = np.load(path)
    _ctx["wsvd"] = {k: z[k] for k in z.files}
    return _ctx["wsvd"]


def resolve_vdir(ref, param, side):
    """Unit direction for a merger edit: "wsvd:k" (singular direction) or a library name (out side of fc2 only)."""
    ref = str(ref)
    if ref.startswith("wsvd:"):
        k = int(ref.split(":")[1])
        if not 0 <= k < NSVD:
            raise ValueError(f"wsvd index must be 0..{NSVD - 1}")
        v = _wsvd()["fc2_out" if param == FC2 else "fc1_in"][k]
    else:
        if param != FC2 or side != "out":
            raise ValueError("library directions (5120-d residual space) only fit the out side of visual.merger.linear_fc2")
        v = _ctx["resolve_dir"](ref, 0)
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def _as_list(x):
    return x if isinstance(x, list) else [x]


def validate(e):
    """L5 proposal -> (clean edit, None) or (None, reason). Clean edits use the loop's op names so dosed() applies."""
    op = e.get("op")
    if "vblock" in e:
        try:
            B = int(e["vblock"])
        except (TypeError, ValueError):
            return None, "vblock must be an integer 0..26"
        if not 0 <= B < DEPTH:
            return None, f"vblock must be 0..{DEPTH - 1}"
        if op == "zero_block":
            parts = _as_list(e.get("parts", e.get("part", [])))
            if len(parts) != 1 or parts[0] not in ("attn", "mlp"):
                return None, "zero_block needs exactly one part: attn or mlp"
            return {"op": "zero_block", "vblock": B, "parts": parts}, None
        if op == "scale":
            part = e.get("part", (_as_list(e.get("parts", [None])) or [None])[0])
            if part not in ("attn", "mlp"):
                return None, "block scale needs part attn or mlp"
            a = float(e.get("alpha", 1.0))
            if not 0.5 <= a <= 1.5:
                return None, f"block scale alpha {a} outside [0.5, 1.5]"
            return {"op": "scale", "vblock": B, "part": part, "alpha": a}, None
        return None, "vision blocks take scale or zero_block"
    p = str(e.get("param", "")).replace("model.", "", 1).replace(".weight", "")
    if p not in (FC1, FC2):
        return None, f"param {p!r} not in the vision menu (merger linear_fc1 / linear_fc2, or vblock)"
    if op == "scale":
        if "neuron" not in e:
            return None, "merger scale needs \"neuron\": [...]"
        ns = sorted({int(x) for x in _as_list(e["neuron"])})
        if not ns or len(ns) > 64 or ns[0] < 0 or ns[-1] >= MERGER:
            return None, f"1..64 merger units in 0..{MERGER - 1}"
        a = float(e.get("alpha", 1.0))
        if not 0.0 <= a <= 2.0:
            return None, f"alpha {a} outside [0, 2]"
        return {"op": "scale", "param": p, "neuron": ns, "alpha": a}, None
    if op in ("project_out", "amplify"):
        side = "out" if p == FC2 else "in"
        if e.get("side", side) != side:
            return None, f"{p} takes directions on the {side} side"
        ref = e.get("u") or e.get("v") or e.get("dir")
        try:
            resolve_vdir(ref, p, side)
        except Exception as ex:
            return None, f"direction {ref!r}: {ex}"
        out = {"op": op, "param": p, "side": side, "dir_name": ref}
        if op == "amplify":
            a = float(e.get("alpha", 0.0))
            if not -2.0 <= a <= 2.0:
                return None, "amplify alpha outside [-2, 2]"
            out["alpha"] = a
        return out, None
    return None, f"op {op!r} not in the vision menu"


def _writer(B, part):
    return f"visual.blocks.{B}.attn.proj" if part == "attn" else f"visual.blocks.{B}.mlp.linear_fc2"


def engine_edit(S, e):
    """Clean L5 edit -> engine edits (HF vision names; the Surgeon must be multimodal)."""
    if "vblock" in e:
        if e["op"] == "zero_block":
            w = _writer(e["vblock"], e["parts"][0])
            return [{"op": "set", "param": w + ".weight", "value": 0.0}, {"op": "set", "param": w + ".bias", "value": 0.0}]
        w = _writer(e["vblock"], e["part"])
        return [{"op": "scale", "param": w + ".weight", "alpha": e["alpha"]},
                {"op": "scale", "param": w + ".bias", "alpha": e["alpha"]}]
    p = e["param"]
    if e["op"] == "scale":
        out = [{"op": "scale", "param": p + ".weight", "neuron": e["neuron"], "alpha": e["alpha"]}]
        if p == FC1:          # a hidden unit of linear_fc1 = its weight row and its bias entry
            out.append({"op": "scale", "param": p + ".bias", "neuron": e["neuron"], "alpha": e["alpha"]})
        return out
    a = -1.0 if e["op"] == "project_out" else float(e["alpha"])
    d = resolve_vdir(e["dir_name"], p, e["side"])
    if p == FC2:              # y = W x + b  ->  (I + a u u^T) y: transform the bias with the weight
        b = S.read(p + ".bias").double().numpy()
        db = (a * d * float(d @ b)).astype(np.float32)
        return [{"op": "amplify", "param": p + ".weight", "side": "out", "u": d, "alpha": a},
                {"op": "add", "param": p + ".bias", "value": db}]
    return [{"op": "amplify", "param": p + ".weight", "side": "in", "v": d, "alpha": a}]


def target(e):
    if "vblock" in e:
        return f"visual.blocks.{e['vblock']}.{e.get('part') or '+'.join(e.get('parts', []))}"
    return e.get("param")


def random_edit(rng):
    """Blind-search control for L5."""
    r = rng.random()
    if r < 0.35:
        return {"op": "scale", "vblock": rng.randrange(DEPTH), "part": rng.choice(["attn", "mlp"]),
                "alpha": round(rng.uniform(0.5, 1.5), 2)}
    if r < 0.45:
        return {"op": "zero_block", "vblock": rng.randrange(DEPTH), "parts": [rng.choice(["attn", "mlp"])]}
    if r < 0.7:
        return {"op": "scale", "param": rng.choice([FC1, FC2]), "alpha": round(rng.uniform(0, 2), 2),
                "neuron": sorted(rng.sample(range(MERGER), rng.choice([8, 16, 32, 64])))}
    p = rng.choice([FC1, FC2])
    return {"op": rng.choice(["project_out", "amplify"]), "param": p, "side": "out" if p == FC2 else "in",
            ("u" if p == FC2 else "v"): f"wsvd:{rng.randrange(NSVD)}", "alpha": round(rng.uniform(-2, 2), 2)}
