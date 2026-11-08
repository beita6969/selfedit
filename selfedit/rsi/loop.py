"""One RSI lineage on one GPU slice: the current model reads a dossier about itself, proposes edits to its own
internal parameters, the engine tries each edit on held-out pool problems, and confirmed edits become the next
generation. No gradients anywhere; prompts and decoding settings stay fixed across generations."""
import argparse, glob, json, os, random, re, sys, time, zlib

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [CODE, os.path.join(CODE, "engine")]
import numpy as np
from grade import correct, looped
from rsi.accept import per_problem, tokens_per_problem, paired_bootstrap, paired_bootstrap_tokens, decide, retention
from rsi.dossier import pool_summary, failure_excerpts, edit_history
from rsi.hazard import Hazard, own_traces as hazard_traces

ap = argparse.ArgumentParser()
ap.add_argument("--run-dir", required=True)
ap.add_argument("--model", default="/data/fd/mingdazhang/selfedit/models/Qwen3.8-27B")
ap.add_argument("--target", default="competition math reasoning under a fixed thinking budget")
ap.add_argument("--pool", nargs="+", required=True)              # selection-pool JSONL files (never test sets)
ap.add_argument("--guard", nargs="*", default=[])                # guard pool files (e.g. MMLU-Pro pool)
ap.add_argument("--calib", default=None)                         # per-problem accuracy file from an earlier calibration
ap.add_argument("--calib-samples", type=int, default=4)
ap.add_argument("--effort", default="medium")
ap.add_argument("--budget", type=int, default=8192)              # max generated tokens; truncation counts as wrong
ap.add_argument("--mb", type=int, default=32)                    # screen problems per generation
ap.add_argument("--confirm-mb", type=int, default=48)
ap.add_argument("--confirm-looks", type=int, default=2, choices=[1, 2])   # 2: two-look confirm (see the confirm block)
ap.add_argument("--guard-mb", type=int, default=64)
ap.add_argument("--samples", type=int, default=2)
ap.add_argument("--confirm-samples", type=int, default=4)   # samples per confirm problem: at 2 most of the confirm
# noise is per-problem sampling, so a true +3-5 pp edit rarely cleared the bound; confirms are rare, the compute is cheap
ap.add_argument("--base-samples", type=int, default=4)   # the screen base is the shared control of every screened
# candidate; its noise moves all of a generation's comparisons together, so it gets ~samples*sqrt(gen-screen) (Dunnett)
ap.add_argument("--gens", type=int, default=6)
ap.add_argument("--max-edits", type=int, default=6)
ap.add_argument("--surgeon-effort", default="xhigh")
ap.add_argument("--surgeon-tokens", type=int, default=36000)
ap.add_argument("--surgeon-fallback-effort", default="medium")   # last attempt thinks less so the JSON fits
ap.add_argument("--proposer", default="model", choices=["model", "random"])   # random = blind-search control
ap.add_argument("--random-cands", type=int, default=12)
ap.add_argument("--doses", default="1,2,4,8")                    # multiples of each edit's deviation from identity
ap.add_argument("--drift-cap", type=float, default=0.08)         # max mean drop of own-trace log-prob (nats/token)
ap.add_argument("--null-controls", type=int, default=3)         # random edits per generation that calibrate the margin's drift bias
ap.add_argument("--efficiency", action="store_true")            # also accept: same accuracy, clearly fewer tokens
ap.add_argument("--guard-tol", type=float, default=0.03)
ap.add_argument("--mentor", default="1-3:codesign,4-6:review,7-99:auto")
ap.add_argument("--review-timeout", type=int, default=3600)
ap.add_argument("--dirlib", default="/data/fd/mingdazhang/selfedit/probe/library")
ap.add_argument("--digests", nargs="*", default=[])              # probe digest .md files shown to the surgeon
ap.add_argument("--tp", type=int, default=1)
ap.add_argument("--max-model-len", type=int, default=49152)
ap.add_argument("--max-num-seqs", type=int, default=64)
ap.add_argument("--gen-screen", type=int, default=4)            # candidates that get a generation screen, ranked by the self-likelihood pre-screen
ap.add_argument("--score-traces", type=int, default=24)         # own traces per side (correct / failed) for the pre-screen
ap.add_argument("--gpu-util", type=float, default=None)         # vLLM budget; leave room for score() logits outside it
ap.add_argument("--max-num-batched-tokens", type=int, default=4096)
ap.add_argument("--kv-cache-dtype", default="auto")            # fp8 doubles concurrency; fixed within a lineage
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--crn", type=int, default=0)   # 1: common random numbers. Every sample of a problem has a fixed seed
# within a generation (from --seed, gen, problem id), so the base and each candidate draw the same random numbers and
# stay identical until the edit changes a token; the screen base then uses --samples (pairs sample k with sample k)
ap.add_argument("--checkpoint-cmd", default=None)                # run at each generation boundary; exit code 4 = yield the card
ap.add_argument("--mm", action="store_true")                    # multimodal lineage: image pools (rsi/mm_eval.py), L5 vision menu
# Judge additions (11-05). A tight-budget gain can be a trade of checking for speed (Gemma gen 7: +10.4 at 8192, -10.6 at 16384):
ap.add_argument("--retain-tol", type=float, default=None)       # confirm also needs accuracy on base-finished problems not to fall by more than this
ap.add_argument("--loose-budget", type=int, default=0)          # confirm survivors are also run at this budget on --loose-mb confirm problems
ap.add_argument("--loose-mb", type=int, default=48)
ap.add_argument("--loose-tol", type=float, default=0.02)        # reject if the mean falls by more than this ...
ap.add_argument("--loose-floor", type=float, default=-0.08)     # ... or the 90 % lower bound is below this
# Pipeline memory and self-adjustment (11-05): the model reads --cases each generation and may retune META_KEYS every
# --meta-every generations. Nothing in the judge is reachable from there.
ap.add_argument("--cases", default=None)
ap.add_argument("--hazard-traces", type=int, default=0)           # > 0: rank candidates by rsi/hazard.py's stop read on this many traces instead of self-likelihood
ap.add_argument("--hazard-min", type=float, default=0.3)          # stop read a candidate needs for a screen slot of its own (null edits read below ~0.1)
ap.add_argument("--hazard-explore", type=int, default=1)          # screen slots kept for candidates below it, picked at random
ap.add_argument("--hazard-key", default="cut", choices=["cut", "ans", "post"])   # which stop read ranks: Qwen's cut read does not predict tokens
ap.add_argument("--hazard-rounds", type=int, default=1)           # proposal rounds per generation; from the second on the model sees the stop reads so far
ap.add_argument("--min-retry-alpha", type=float, default=0.5)
ap.add_argument("--alpha-cap", type=float, default=3.0)           # dosing limit of a direction edit (Gemma: alpha -1 holds at the loose budget, -2 does not)
ap.add_argument("--no-prescreen", action="store_true")          # self-likelihood excess does not predict screens (rho -0.11, n 203): proposal order, strongest dose within the drift cap
ap.add_argument("--no-repeats", action="store_true")            # a target + direction already screened on this base is not screened again
ap.add_argument("--token-sd", type=float, default=0.0)          # null SD of a screen's dtok for this model: tokens replicate across problems (rho +0.46), screen accuracy does not (-0.40)
ap.add_argument("--meta-every", type=int, default=0)
args = ap.parse_args()

os.makedirs(args.run_dir, exist_ok=True)
LOG = os.path.join(args.run_dir, "log.jsonl")
ACCEPTED = os.path.join(args.run_dir, "accepted.jsonl")
SAMPLING = dict(temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0, repetition_penalty=1.0)


def say(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def jdump(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)


def load_jsonl(paths):
    return [json.loads(l) for p in paths for l in open(p)]


# ---------------------------------------------------------------- edit menu and validation
MENU = """| level | targets (HF names; layer L = 0..63, full attention iff L % 4 == 3, else Gated DeltaNet) | ops | bounds |
|---|---|---|---|
| L1 neurons | `layers.L.mlp.down_proj` with "neuron": n (0..17407) (output column of one neuron); `layers.L.mlp.gate_proj` / `up_proj` with "neuron": n | scale (alpha in [0, 2]) | at most 64 neurons per edit |
| L2 gates and memory | DeltaNet `layers.L.linear_attn.A_log` / `dt_bias` with "head": h (0..47) — memory decay; `linear_attn.in_proj_b` "head": h — write strength beta; `linear_attn.in_proj_z` "head": h — output gate; attention `layers.L.self_attn.q_proj` "head": h (0..23) "part": "gate" — output gate; norms `layers.L.input_layernorm`, `post_attention_layernorm` (stored as w, gain = 1 + w) | add (A_log: value in [-1, 1]; dt_bias: [-2, 2]) or scale (alpha in [0.5, 1.5]) | at most 16 heads per edit |
| L3 directions | out side of `self_attn.o_proj`, `linear_attn.out_proj`, `mlp.down_proj`; in side of `self_attn.q_proj/k_proj/v_proj`, `linear_attn.in_proj_qkv/z/b/a`, `mlp.gate_proj/up_proj` — direction given by name (see library) | project_out, amplify (W <- (I + alpha u u^T) W; alpha in [-2, 2]: -1 removes the component, -2 reflects it, +1 doubles it) | one layer per edit |
| L3g global directions | every residual writer of a layer range at once: `writers` "attn" (o_proj / out_proj) and/or "mlp" (down_proj), `layers` [first, last] inclusive | project_out, amplify (alpha in [-2, 2]); "u": "dir:<contrast>:<kind>:<L>" uses one direction everywhere, "dir:<contrast>:<kind>:*" uses each layer's own direction | any range (directional ablation over all layers = [0, 63]) |
| L4 structure | whole blocks | zero_block {"layer": L, "parts": ["attn"] or ["mlp"]} | one block per edit |

Dose: the engine sweeps every edit (except zero_block) at 1x, 2x, 4x and 8x of its deviation from identity (scale alpha -> 1 + k(alpha - 1), add value -> k * value, amplify alpha -> k * alpha; project_out is alpha = -1), within hard limits, and keeps the dose with the best self-likelihood margin among doses that do not move your own traces too far. Past lesson (generation 1): single-layer edits with small alpha changed your own-trace log-prob by only ~1e-4 nats/token and had no measurable effect; edits need to be strong enough to matter."""
DIR_HELP = """Directions are referenced by name: "dir:<contrast>:<kind>:<layer>" or "dir:<contrast>:<kind>:<layer>:svd<k>",
kind in {resid, mixer, mlp}; a direction lives in the 5120-d residual space, so use it on the out side of
o_proj/out_proj/down_proj or the in side of the input projections. Available contrasts: {names}."""
EDIT_FORMAT = """Edit JSON (one object per edit; keep why and predict under 40 words each):
{"op": "scale" | "add" | "project_out" | "amplify" | "zero_block",
 "param": "<HF name>", "head": h | [h...], "part": "gate", "neuron": n | [n...],
 "alpha": x, "value": x, "side": "out" | "in", "u" or "v": "dir:...", "layer": L, "parts": [...],
 "writers": ["attn", "mlp"], "layers": [first, last],
 "why": "<the internal evidence>", "predict": "<expected behaviour change>", "confidence": 0..1}"""
WRITERS = {"attn": lambda L: f"layers.{L}.self_attn.o_proj" if L % 4 == 3 else f"layers.{L}.linear_attn.out_proj",
           "mlp": lambda L: f"layers.{L}.mlp.down_proj"}
COMBO_FORMAT = """\nSeveral edits as one candidate: {"combo": [edit, edit, ...], "why": ..., "predict": ..., "confidence": ...}, 2 to 8 edits."""
PARAM_RE = re.compile(r"^layers\.(\d+)\.(mlp\.(down|gate|up)_proj|linear_attn\.(A_log|dt_bias|in_proj_b|in_proj_z|in_proj_a|in_proj_qkv|out_proj)|"
                      r"self_attn\.(q|k|v|o)_proj|input_layernorm|post_attention_layernorm)$")


def as_list(x):
    return x if isinstance(x, list) else [x]


def validate(e):
    """Return (clean edit for the engine, None) or (None, reason)."""
    if args.mm and mm_eval.is_vision(e):
        return mm_eval.validate(e)
    op = e.get("op")
    if op == "zero_block":
        L = int(e.get("layer", -1))
        parts = as_list(e.get("parts", []))
        if not (0 <= L < 64) or not parts or any(p not in ("attn", "mlp") for p in parts) or len(parts) > 1:
            return None, "zero_block needs layer 0..63 and exactly one part"
        return {"op": op, "layer": L, "parts": parts}, None
    if op in ("project_out", "amplify") and "writers" in e:
        writers = as_list(e.get("writers"))
        lay = as_list(e.get("layers", []))
        if not writers or any(w not in WRITERS for w in writers):
            return None, "writers must be a subset of ['attn', 'mlp']"
        if len(lay) != 2 or not 0 <= int(lay[0]) <= int(lay[1]) < 64:
            return None, "layers must be [first, last] with 0 <= first <= last <= 63"
        ref = e.get("u") or e.get("dir")
        try:
            resolve_dir(ref, int(lay[0]))
        except Exception as ex:
            return None, f"direction {ref!r}: {ex}"
        a = -1.0 if op == "project_out" else float(e.get("alpha", 0.0))
        if not -2.0 <= a <= 2.0:
            return None, "amplify alpha outside [-2, 2]"
        return {"op": op, "writers": writers, "layers": [int(lay[0]), int(lay[1])], "dir_name": ref, "alpha": a}, None
    p = str(e.get("param", "")).replace("model.language_model.", "").replace(".weight", "")
    m = PARAM_RE.match(p)
    if not m or not 0 <= int(m.group(1)) < 64:
        return None, f"param {p!r} not in the edit menu"
    L = int(m.group(1))
    is_attn = L % 4 == 3
    if ("linear_attn" in p and is_attn) or ("self_attn" in p and not is_attn):
        return None, f"layer {L} is {'full attention' if is_attn else 'DeltaNet'}; {p} does not exist"
    out = {"op": op, "param": p}
    for k in ("head", "part", "neuron"):
        if k in e:
            out[k] = e[k]
    for k in ("head", "neuron"):                      # the engine takes strictly increasing lists; one alpha for all
        if isinstance(out.get(k), list):
            out[k] = sorted({int(i) for i in out[k]})
    if "head" in out and len(as_list(out["head"])) > 16:
        return None, "at most 16 heads per edit"
    if "neuron" in out and len(as_list(out["neuron"])) > 64:
        return None, "at most 64 neurons per edit"
    if op == "scale":
        a = float(e.get("alpha", 1.0))
        lo, hi = (0.0, 2.0) if "neuron" in out else (0.5, 1.5)
        if not lo <= a <= hi:
            return None, f"alpha {a} outside [{lo}, {hi}]"
        out["alpha"] = a
    elif op == "add":
        v = float(e.get("value", 0.0))
        lim = 1.0 if p.endswith("A_log") else 2.0 if p.endswith("dt_bias") else 0.0
        if not lim or abs(v) > lim:
            return None, "add is only for A_log (|value| <= 1) and dt_bias (|value| <= 2)"
        out["value"] = v
    elif op in ("project_out", "amplify"):
        side = e.get("side", "out")
        key = "u" if side == "out" else "v"
        ref = e.get(key) or e.get("u") or e.get("v")
        out_side = any(s in p for s in ("o_proj", "out_proj", "down_proj"))
        if (side == "out") != out_side:
            return None, f"{p} takes residual directions on the {'out' if out_side else 'in'} side"
        try:
            resolve_dir(ref, L)
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


_dir_cache = {}


def resolve_dir(ref, layer=None):
    """Unit direction for a library name; layer "*" means the layer the edit is applied at."""
    parts = str(ref).split(":")
    if parts[0] == "dir" and len(parts) == 3 and layer is not None:
        parts.append("*")   # no layer given (OLMo gen 3: "dir:late_vs_early_correct:mlp"): the edit's own layer
    if parts[0] != "dir" or len(parts) not in (4, 5):
        raise ValueError("expected dir:<contrast>:<kind>:<layer>[:svd<k>]")
    name, kind = parts[1], parts[2]
    L = layer if parts[3] == "*" else int(parts[3])
    if L is None:
        raise ValueError("layer '*' needs a layer to apply at")
    if name not in _dir_cache:
        _dir_cache[name] = np.load(os.path.join(args.dirlib, name + ".npz"))
    z = _dir_cache[name]
    if len(parts) == 5:
        v = z[f"{kind}/svd_V"][L, int(parts[4].replace("svd", ""))]
    else:
        v = z[f"{kind}/dir"][L]
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def engine_edit(e):
    """Clean edit -> what the engine applies (directions resolved; a global edit becomes one edit per writer)."""
    if args.mm and mm_eval.is_vision(e):
        return mm_eval.engine_edit(_S, e)
    if "writers" in e:
        return [{"op": "amplify", "param": WRITERS[w](L), "side": "out", "alpha": e["alpha"],
                 "u": resolve_dir(e["dir_name"], L)}
                for L in range(e["layers"][0], e["layers"][1] + 1) for w in e["writers"]]
    if e["op"] in ("project_out", "amplify"):
        L = int(e["param"].split(".")[1])
        return dict({k: v for k, v in e.items() if k != "dir_name"},
                    **{"u" if e["side"] == "out" else "v": resolve_dir(e["dir_name"], L)})
    return e


def target(e):
    if args.mm and mm_eval.is_vision(e):
        return mm_eval.target(e)
    if "writers" in e:
        return f"layers {e['layers'][0]}-{e['layers'][1]} writers {'+'.join(e['writers'])}"
    return e.get("param", e.get("layer"))


def dosed(e, k):
    """The same edit with k times its deviation from identity, clipped to hard limits; None if it cannot be dosed."""
    if k == 1:
        return e
    op, d = e["op"], dict(e)
    if op == "zero_block":
        return None
    if op == "scale":
        lo, hi = (0.0, 3.0) if "neuron" in e else (0.25, 2.0)
        d["alpha"] = round(min(hi, max(lo, 1 + k * (e["alpha"] - 1))), 4)
    elif op == "add":
        lim = 2.0 if e["param"].endswith("A_log") else 4.0
        d["value"] = round(max(-lim, min(lim, k * e["value"])), 4)
    else:
        a = -1.0 if op == "project_out" else e["alpha"]
        d["op"], d["alpha"] = "amplify", round(max(-args.alpha_cap, min(args.alpha_cap, k * a)), 4)
    return d


def random_edit(rng, names):
    """Blind-search control: a random edit from the same menu (no evidence, no reasoning)."""
    if args.mm and rng.random() < 0.3:
        return mm_eval.random_edit(rng)
    r, L = rng.random(), rng.randrange(64)
    DN, AT = [l for l in range(64) if l % 4 != 3], [l for l in range(64) if l % 4 == 3]
    if r < 0.2:
        return {"op": "scale", "param": f"layers.{L}.mlp.down_proj", "alpha": round(rng.uniform(0, 2), 2),
                "neuron": sorted(rng.sample(range(17408), rng.choice([8, 16, 32, 64])))}
    if r < 0.45:
        kind = rng.choice(["A_log", "dt_bias", "in_proj_b", "in_proj_z", "qgate", "norm"])
        if kind == "qgate":
            return {"op": "scale", "param": f"layers.{rng.choice(AT)}.self_attn.q_proj", "part": "gate",
                    "head": sorted(rng.sample(range(24), rng.randint(1, 8))), "alpha": round(rng.uniform(0.5, 1.5), 2)}
        if kind == "norm":
            return {"op": "scale", "param": f"layers.{L}.{rng.choice(['input_layernorm', 'post_attention_layernorm'])}",
                    "alpha": round(rng.uniform(0.5, 1.5), 2)}
        p, heads = f"layers.{rng.choice(DN)}.linear_attn.{kind}", sorted(rng.sample(range(48), rng.randint(1, 16)))
        if kind == "A_log":
            return {"op": "add", "param": p, "head": heads, "value": round(rng.uniform(-1, 1), 2)}
        if kind == "dt_bias":
            return {"op": "add", "param": p, "head": heads, "value": round(rng.uniform(-2, 2), 2)}
        return {"op": "scale", "param": p, "head": heads, "alpha": round(rng.uniform(0.5, 1.5), 2)}
    c, kind = rng.choice(names), rng.choice(["resid", "mixer", "mlp"])
    if r < 0.7:
        return {"op": "amplify", "param": WRITERS[rng.choice(["attn", "mlp"])](L), "side": "out",
                "u": f"dir:{c}:{kind}:{L}", "alpha": round(rng.uniform(-2, 2), 2)}
    if r < 0.95:
        a = rng.randrange(64)
        return {"op": "amplify", "writers": rng.choice([["attn"], ["mlp"], ["attn", "mlp"]]),
                "layers": [a, rng.randrange(a, 64)], "u": f"dir:{c}:{kind}:*", "alpha": round(rng.uniform(-2, 2), 2)}
    return {"op": "zero_block", "layer": L, "parts": [rng.choice(["attn", "mlp"])]}


def contrast_names():
    return sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(args.dirlib, "*.npz")))


# ---------------------------------------------------------------- evaluation
CRN_GEN = 0                                                      # generation whose seeds evaluate() uses


def crn_seeds(probs):
    """One seed per problem for this generation (vLLM gives sample k seed + k); None without --crn."""
    if not args.crn:
        return None
    return [zlib.crc32(f"{args.seed}|{CRN_GEN}|{p['id']}".encode()) % (2 ** 31 - 1024) for p in probs]


def evaluate(S, probs, n):
    if args.mm:
        return mm_eval.evaluate(S, probs, n, args.budget, args.effort, SAMPLING, seeds=crn_seeds(probs))
    outs = S.generate([p["prompt"] for p in probs], n=n, sampling=SAMPLING, max_tokens=args.budget,
                      reasoning_effort=args.effort, seeds=crn_seeds(probs))
    recs = []
    for p, o in zip(probs, outs):
        for k, s in enumerate(o["samples"]):
            r = {"id": p["id"], "kind": p["kind"], "answer": p["answer"], "sample": k, "reasoning": s["reasoning"],
                 "content": s["content"], "finish": s["finish_reason"], "tokens": s["num_tokens"],
                 "seq": o["prompt_token_ids"] + s["token_ids"], "start": len(o["prompt_token_ids"])}
            r["ok"] = correct(r)
            r["loop"] = looped(r)
            recs.append(r)
    return recs


def compare(base, cand):
    main = paired_bootstrap(per_problem(base), per_problem(cand))
    if main["n"] == 0:
        return {"n": 0, "diff": 0.0, "lo90": 0.0, "hi90": 0.0, "dtok": 0.0, "acc_base": float("nan"), "acc_cand": float("nan")}
    bt, ct = tokens_per_problem(base), tokens_per_problem(cand)
    main["tok"] = paired_bootstrap_tokens(bt, ct)
    ids = sorted(set(bt) & set(ct))
    main["dtok"] = float(np.mean([np.mean(ct[i]) - np.mean(bt[i]) for i in ids])) if ids else 0.0
    main["acc_base"] = float(np.mean([r["ok"] for r in base]))
    main["acc_cand"] = float(np.mean([r["ok"] for r in cand]))
    return main


def own_traces(base, k, rng):
    """The model's own traces on the screen problems: up to k correct and k failed (wrong or truncated)."""
    good = [r for r in base if r["ok"]]
    bad = [r for r in base if not r["ok"]]
    rng.shuffle(good)
    rng.shuffle(bad)
    return good[:k], bad[:k]


def self_margin(S, good, bad, ref):
    """Self-likelihood pre-screen: how much the edit raises the mean token log-prob of the model's own correct
    traces relative to its own failed ones (prefill only, no generation)."""
    seqs = [r["seq"] for r in good + bad]
    if args.mm:
        sc = S.score(seqs, [r["start"] for r in good + bad], images=mm_eval.score_images(good + bad))
    else:
        sc = S.score(seqs, [r["start"] for r in good + bad])
    m = [x["mean"] for x in sc]
    if ref is None:
        return m
    d = [a - b for a, b in zip(m, ref)]
    dg, db = d[: len(good)], d[len(good):]
    return {"d_good": float(np.mean(dg)) if dg else 0.0, "d_bad": float(np.mean(db)) if db else 0.0,
            "margin": (float(np.mean(dg)) if dg else 0.0) - (float(np.mean(db)) if db else 0.0)}


def mentor_mode(gen):
    for part in args.mentor.split(","):
        rng, mode = part.split(":")
        a, b = (int(x) for x in rng.split("-"))
        if a <= gen <= b:
            return mode
    return "auto"


def extract_json(text):
    """The last JSON object with "edits" or "probes" in the text; failing that, every complete edit object after
    the last '"edits"' (salvages answers cut off by the token budget)."""
    dec = json.JSONDecoder()
    for i in reversed([m.start() for m in re.finditer(r"\{", text)]):
        try:
            obj, _ = dec.raw_decode(text, i)
        except ValueError:
            continue
        if isinstance(obj, dict) and ("edits" in obj or "probes" in obj):
            return obj
    k = text.rfind('"edits"')
    items, i = [], k
    while k >= 0:
        i = text.find("{", i + 1)
        if i < 0:
            break
        try:
            obj, end = dec.raw_decode(text, i)
        except ValueError:
            continue
        if isinstance(obj, dict) and "op" in obj:
            items.append(obj)
            i = end - 1
    return {"edits": items, "salvaged": True} if items else None


# ---------------------------------------------------------------- pipeline self-adjustment
# What the model may retune, with bounds. Screens, confirms, guards, budgets, intervals and the decision rules are absent on purpose.
META_KEYS = {"doses": "list of 1-5 numbers in [0.25, 16]: multiples of an edit's deviation swept in the pre-screen",
             "gen_screen": "integer in [2, 8]: candidates screened per generation",
             "max_edits": "integer in [2, 8]: proposals requested per generation",
             "use_prescreen": "true or false: false screens candidates in proposal order at the strongest dose within the drift cap, instead of ranking candidates and doses by self-likelihood",
             "focus": "text under 400 characters added to your own dossier: what you should try or avoid next"}
META = {"focus": "", "use_prescreen": not args.no_prescreen}


def meta_apply(cfg):
    out = {}
    if isinstance(cfg.get("doses"), list) and 1 <= len(cfg["doses"]) <= 5:
        d = [float(x) for x in cfg["doses"] if isinstance(x, (int, float)) and 0.25 <= x <= 16]
        if d:
            args.doses = ",".join(f"{x:g}" for x in d)
            out["doses"] = d
    for k, lo, hi in (("gen_screen", 2, 8), ("max_edits", 2, 8)):
        if isinstance(cfg.get(k), int) and lo <= cfg[k] <= hi:
            setattr(args, k, cfg[k])
            out[k] = cfg[k]
    if isinstance(cfg.get("use_prescreen"), bool) and not args.hazard_traces:
        META["use_prescreen"] = out["use_prescreen"] = cfg["use_prescreen"]
    if isinstance(cfg.get("focus"), str):
        META["focus"] = out["focus"] = cfg["focus"][:400]
    return out


def meta_step(S, gen, cases):
    state = {"doses": args.doses, "gen_screen": args.gen_screen, "max_edits": args.max_edits, **META}
    prompt = ("You are running the automated pipeline that edits your own weights. Below are the record of all edits tried on "
              "you in this run and notes from earlier runs. Decide how the pipeline should search next.\n\n"
              f"Generation {gen}. Current settings: {json.dumps(state)}\n\nSettings you may change:\n"
              + "\n".join(f"- {k}: {v}" for k, v in META_KEYS.items())
              + "\n\nThe tests that decide whether an edit is kept are fixed and you cannot change them.\n\n"
              f"Edit record:\n{edit_history(LOG)}\n\nNotes:\n{cases or '(none)'}\n\n"
              'Answer with one JSON object after your reasoning: {"config": {only the settings you change}, "why": "<under 60 words, citing the record>"}.')
    out = S.generate([prompt], n=1, sampling=SAMPLING, max_tokens=min(args.surgeon_tokens, 16000),
                     reasoning_effort=args.surgeon_effort)[0]["samples"][0]
    obj = extract_json(out["content"] or "") or extract_json(out["reasoning"] or "") or {}
    done = meta_apply(obj.get("config") or {})
    with open(os.path.join(args.run_dir, "meta.jsonl"), "a") as f:
        f.write(json.dumps({"gen": gen, "asked": obj.get("config"), "applied": done, "why": obj.get("why", "")}, ensure_ascii=False) + "\n")
    say(f"gen {gen}: pipeline settings {json.dumps(done, ensure_ascii=False)[:300]}")


# ---------------------------------------------------------------- main
def main():
    global _S, CRN_GEN
    from surgery import Surgeon
    rng = random.Random(args.seed)
    S = Surgeon(args.model, tp=args.tp, max_model_len=args.max_model_len, max_num_seqs=args.max_num_seqs,
                max_num_batched_tokens=args.max_num_batched_tokens, gpu_memory_utilization=args.gpu_util,
                kv_cache_dtype=args.kv_cache_dtype, **({"multimodal": True} if args.mm else {}))
    _S = S
    say("engine up")

    # Resume: re-apply edits accepted by earlier generations.
    accepted = load_jsonl([ACCEPTED]) if os.path.exists(ACCEPTED) else []
    for a in accepted:
        S.apply(engine_edit(a["edit"]))
    logged = load_jsonl([LOG]) if os.path.exists(LOG) else []
    gen0 = 1 + max([a["gen"] for a in accepted + logged], default=0)
    gen0 = max(gen0, 1 + max([int(d[-2:]) for d in glob.glob(os.path.join(args.run_dir, "gen[0-9][0-9]"))
                              if os.path.exists(os.path.join(d, "DONE"))], default=0))
    say(f"resumed with {len(accepted)} accepted edits; starting generation {gen0}")
    if os.path.exists(os.path.join(args.run_dir, "meta.jsonl")):
        for m in load_jsonl([os.path.join(args.run_dir, "meta.jsonl")]):
            meta_apply(m.get("applied") or {})

    pool = load_jsonl(args.pool)
    guard = load_jsonl(args.guard) if args.guard else []
    # Calibration of generation 0: keep problems the model sometimes but not always solves (the frontier).
    calib_path = args.calib or os.path.join(args.run_dir, "calib.jsonl")
    if not os.path.exists(calib_path):
        say(f"calibrating on {len(pool)} pool problems x {args.calib_samples}")
        recs = evaluate(S, pool, args.calib_samples)
        with open(calib_path, "w") as f:
            for r in recs:
                f.write(json.dumps({k: v for k, v in r.items() if k not in ("seq", "start")}, ensure_ascii=False) + "\n")
    calib = load_jsonl([calib_path])
    acc = {k: float(np.mean(v)) for k, v in per_problem(calib).items()}
    # Budget-limited failures (always truncated) also count: a more concise model could solve them.
    trunc = {k: float(np.mean(v)) for k, v in per_problem([dict(r, ok=r["finish"] == "length") for r in calib]).items()}
    frontier = [p for p in pool if 0 < acc.get(p["id"], 0) < 1 or (acc.get(p["id"]) == 0 and trunc.get(p["id"], 0) >= 0.5)]
    say(f"frontier {len(frontier)} / {len(pool)}; calibration summary:\n" + pool_summary(calib))

    for gen in range(gen0, gen0 + args.gens):
        gdir = os.path.join(args.run_dir, f"gen{gen:02d}")
        os.makedirs(gdir, exist_ok=True)
        rng.shuffle(frontier)
        screen, confirm = frontier[: args.mb], frontier[args.mb: args.mb + args.confirm_mb]
        gmb = rng.sample(guard, min(args.guard_mb, len(guard))) if guard else []
        CRN_GEN = gen
        cases = open(args.cases).read() if args.cases and os.path.exists(args.cases) else ""
        if args.meta_every and gen > 1 and (gen - 1) % args.meta_every == 0:
            meta_step(S, gen, cases)
        nb = args.samples if args.crn else args.base_samples
        say(f"gen {gen}: base eval on {len(screen)} screen problems x {nb}" + (" (common random numbers)" if args.crn else ""))
        base = evaluate(S, screen, nb)

        # Dossier and the model's own proposals.
        notes_path = os.path.join(args.run_dir, "mentor_notes.md")
        digests = "\n\n".join(open(d).read() for d in args.digests if os.path.exists(d)) or "(none yet)"
        prompt = open(os.path.join(CODE, "rsi", "surgeon_prompt.md")).read()
        fields = dict(gen=gen, target_family=args.target,
                      edit_menu=(mm_eval.extend_menu(MENU) if args.mm else MENU) + "\n\n" + DIR_HELP.replace("{names}", ", ".join(contrast_names()) or "(none)")
                      + "\n\n" + EDIT_FORMAT + COMBO_FORMAT,
                      pool_summary=pool_summary(base) + f"\n(decoding fixed: thinking effort {args.effort}, "
                      f"budget {args.budget} tokens, truncation counts as wrong)",
                      failure_excerpts=failure_excerpts(base, k=5, seed=gen),
                      edit_history=edit_history(LOG) + ("\n\nMentor notes:\n" + open(notes_path).read()
                                                        if os.path.exists(notes_path) else "")
                      + ("\n\nNotes from earlier runs:\n" + cases if cases else "")
                      + ("\n\nYour own note for this round: " + META["focus"] if META["focus"] else ""),
                      probe_digests=digests, max_probes=4, max_edits=args.max_edits,
                      probe_api="(describe what you want measured, e.g. "
                                "{\"probes\": [{\"contrast\": [\"<set A description>\", \"<set B description>\"], "
                                "\"mask\": \"think\"}]}; requests are queued for the mentor, who builds the feasible ones; "
                                "their digests appear in a later generation)")
        for k, v in fields.items():
            prompt = prompt.replace("{" + k + "}", str(v))
        # The surgeon's real budget is what the context leaves after the dossier: a 31k-token dossier at max-model-len
        # 49152 left 18k of the 36k, and answers were cut off (11-04). State the true number in the prompt.
        n_prompt = len(S.tokenizer(S.render(prompt, reasoning_effort=args.surgeon_effort),
                                   add_special_tokens=False)["input_ids"]) if hasattr(S, "render") else 0
        s_tokens = max(1024, min(args.surgeon_tokens, args.max_model_len - n_prompt - 128))
        if s_tokens < args.surgeon_tokens:
            say(f"gen {gen}: dossier {n_prompt} tokens leaves the surgeon {s_tokens} (max-model-len {args.max_model_len})")
        prompt += (f"\n\nBudget: about {s_tokens} tokens for your reasoning and the answer together. An answer cut "
                   "off by the budget is lost, so keep the reasoning focused and write the final JSON compactly.")
        open(os.path.join(gdir, "dossier.md"), "w").write(prompt)
        proposals, wishes = [], []
        for attempt in range(3 if args.proposer == "model" else 0):
            effort = args.surgeon_effort if attempt < 2 else args.surgeon_fallback_effort
            out = S.generate([prompt], n=1, sampling=SAMPLING, max_tokens=s_tokens,
                             reasoning_effort=effort)[0]["samples"][0]
            open(os.path.join(gdir, f"surgeon_raw{attempt}.txt"), "w").write(out["reasoning"] + "\n</think>\n" + out["content"])
            obj = extract_json(out["content"] or "") or extract_json(out["reasoning"] or "") or {}
            wishes += obj.get("probes", [])
            proposals = obj.get("edits", [])
            if proposals:
                say(f"gen {gen}: surgeon attempt {attempt} ({effort}, {out['num_tokens']} tokens) gave {len(proposals)} edits"
                    + (" (salvaged from a cut-off answer)" if obj.get("salvaged") else ""))
                break
            if wishes:
                prompt += ("\n\n(Your probe requests are queued for the next generation. This round, choose option B "
                           "and propose edits from the evidence you already have.)")
        if wishes:
            with open(os.path.join(args.run_dir, "probe_wishes.jsonl"), "a") as f:
                f.write(json.dumps({"gen": gen, "probes": wishes}, ensure_ascii=False) + "\n")
        cands = []
        if args.proposer == "random":
            names = contrast_names()
            proposals = [random_edit(rng, names) for _ in range(args.random_cands)]
        for i, e in enumerate(proposals[: args.max_edits if args.proposer == "model" else None]):
            clean, why = validate(e)
            cands.append({"idx": i, "author": args.proposer if args.proposer == "random" else "model", "proposal": e,
                          "edit": clean, "invalid": why})
        sig_path = os.path.join(args.run_dir, "screened.jsonl")
        seen_sigs = set()
        if args.no_repeats and os.path.exists(sig_path):
            seen_sigs = {r["sig"] for r in load_jsonl([sig_path]) if r["base"] == len(accepted)}
        for c in cands:
            if c["edit"] is not None:
                c["sig"] = json.dumps({k: v for k, v in c["edit"].items() if k not in ("alpha", "value")}, sort_keys=True)
                if c["sig"] in seen_sigs:
                    c["edit"], c["invalid"] = None, "this target and direction were already screened on these weights"
        jdump(os.path.join(gdir, "proposals.json"), [{k: v for k, v in c.items() if k not in ("edit", "sig")} for c in cands])
        say(f"gen {gen}: model proposed {len(proposals)} edits, {sum(c['edit'] is not None for c in cands)} valid")

        # Mentor review (co-design / review phases).
        mode = mentor_mode(gen)
        review_path = os.path.join(gdir, "review.json")
        if mode != "auto":
            open(os.path.join(gdir, "AWAITING_REVIEW"), "w").write(f"{mode} {time.time():.0f}")
            say(f"gen {gen}: AWAITING_REVIEW ({mode}) in {gdir}")
            t0 = time.time()
            while not os.path.exists(review_path) and time.time() - t0 < args.review_timeout:
                time.sleep(15)
        if os.path.exists(review_path):
            rev = json.load(open(review_path))
            for d in rev.get("decisions", []):
                c = next((c for c in cands if c["idx"] == d["idx"]), None)
                if c is None:
                    continue
                c["mentor"] = d.get("action")
                if d.get("action") == "reject":
                    c["edit"], c["invalid"] = None, "mentor: " + d.get("reason", "")
                elif d.get("action") == "modify":
                    c["edit"], c["invalid"] = validate(d["edit"])
                    c["author"] = "model+mentor"
            if mode == "codesign":
                for j, e in enumerate(rev.get("mentor_edits", [])):
                    clean, why = validate(e)
                    cands.append({"idx": 100 + j, "author": "mentor", "proposal": e, "edit": clean, "invalid": why})
            if rev.get("notes"):
                with open(notes_path, "a") as f:
                    f.write(f"- gen {gen}: {rev['notes']}\n")

        doses = [float(x) for x in args.doses.split(",")]
        if args.hazard_traces:
            # Rank by the causal read of the stop decision (rsi/hazard.py): strongest dose within the drift cap; candidates that
            # clear --hazard-min go first, --hazard-explore slots stay for the others, so the read itself keeps being tested.
            hz = Hazard(S, hazard_traces(S, base, args.hazard_traces, rng))
            cut0, end0 = hz.base_stop()[:2]
            say(f"gen {gen}: stop read on {args.hazard_traces} traces, {hz.tokens} tokens; base log p(stop) cuts {cut0:.2f}, natural end {end0:.2f}")
            reads = []

            def stop_read(c):
                pts, seen = [], set()
                for k in doses:
                    e = dosed(c["edit"], k)
                    key = json.dumps(e, sort_keys=True) if e else None
                    if e is None or key in seen:
                        continue
                    seen.add(key)
                    try:
                        h = S.apply(engine_edit(e))
                    except Exception as ex:
                        say(f"gen {gen} dose {k}: apply failed: {ex}")
                        continue
                    p = hz.effect()
                    S.revert(h)
                    p["dose"], p["edit"] = k, e
                    pts.append(p)
                    say(f"gen {gen} cand {c['idx']} dose {k:g}: stop {p['d_cut']:+.3f}+-{p['se_cut']:.3f} ans {p['d_ans']:+.3f} post {p['d_post']:+.3f} pre {p['d_pre']:+.3f} end {p['d_end']:+.3f} all {p['d_all']:+.4f} revisit {p['d_rev']:+.3f}")
                    reads.append(f"- {json.dumps(e)}: stop {p['d_cut']:+.3f}, after the answer {p['d_ans']:+.3f}, end {p['d_end']:+.3f}, all {p['d_all']:+.4f}, revisit {p['d_rev']:+.3f}")
                    if -p["d_all"] > args.drift_cap:                # doses ascend: stronger ones only drift further
                        break
                c["sweep"] = [{k: v for k, v in p.items() if k != "edit"} for p in pts]
                ok = [p for p in pts if -p["d_all"] <= args.drift_cap]
                if not ok:
                    c["edit"], c["invalid"] = None, "every dose moved the model's own traces beyond the drift cap" if pts else "apply failed"
                    return
                best = max(ok, key=lambda p: p["d_" + args.hazard_key])
                c["edit"], c["dose"] = best["edit"], best["dose"]
                c["pre"] = {k: v for k, v in best.items() if k != "edit"}

            for c in cands:
                if c["edit"] is not None:
                    stop_read(c)
            # The model sees what its edits did and proposes again: an experiment inside the generation, at a fraction of a screen's cost.
            for rnd in range(1, args.hazard_rounds if args.proposer == "model" else 1):
                follow = prompt + "\n\n" + open(os.path.join(CODE, "rsi", "stop_read_prompt.md")).read()
                for k, v in dict(n=args.hazard_traces, cut=f"{cut0:.2f}", end=f"{end0:.2f}", cap=args.drift_cap, table="\n".join(reads),
                                 max_edits=args.max_edits).items():
                    follow = follow.replace("{" + k + "}", str(v))
                out = S.generate([follow], n=1, sampling=SAMPLING, max_tokens=s_tokens, reasoning_effort=args.surgeon_effort)[0]["samples"][0]
                open(os.path.join(gdir, f"surgeon_round{rnd + 1}.txt"), "w").write(out["reasoning"] + "\n</think>\n" + out["content"])
                obj = extract_json(out["content"] or "") or extract_json(out["reasoning"] or "") or {}
                more = obj.get("edits", [])[: args.max_edits]
                say(f"gen {gen}: round {rnd + 1} ({out['num_tokens']} tokens) gave {len(more)} edits")
                known = {c["sig"] for c in cands if c.get("sig")}
                for i, e in enumerate(more):
                    clean, why = validate(e)
                    c = {"idx": 100 * (rnd + 1) + i, "author": "model", "proposal": e, "edit": clean, "invalid": why, "round": rnd + 1}
                    if clean is not None:
                        c["sig"] = json.dumps({k: v for k, v in clean.items() if k not in ("alpha", "value")}, sort_keys=True)
                        if c["sig"] in seen_sigs:
                            c["edit"], c["invalid"] = None, "this target and direction were already screened on these weights"
                        elif c["sig"] in known:
                            c["edit"], c["invalid"] = None, "this target and direction were already read this generation"
                        known.add(c["sig"])
                    cands.append(c)
                    if c["edit"] is not None:
                        stop_read(c)
            valid = sorted([c for c in cands if c["edit"] is not None], key=lambda c: -c["pre"]["d_" + args.hazard_key])
            top = [c for c in valid if c["pre"]["d_" + args.hazard_key] >= args.hazard_min][: max(0, args.gen_screen - args.hazard_explore)]
            rest = [c for c in valid if c not in top]
            rng.shuffle(rest)
            for c in rest[args.hazard_explore:]:
                c["skip"] = True
            cands = top + rest + [c for c in cands if c["edit"] is None]

        # Pre-screen every valid candidate by self-likelihood on its own traces; the best get a generation screen.
        good, bad = ([], []) if args.hazard_traces else own_traces(base, args.score_traces, rng)
        ref = self_margin(S, good, bad, None) if good or bad else None
        ng, nb = len(good), len(bad)

        def sweep_points(edit):
            pts, seen = [], set()
            for k in doses:
                e = dosed(edit, k)
                key = json.dumps(e, sort_keys=True) if e else None
                if e is None or key in seen:
                    continue
                seen.add(key)
                try:
                    h = S.apply(engine_edit(e))
                except Exception as ex:
                    say(f"gen {gen} dose {k}: apply failed: {ex}")
                    continue
                pre = self_margin(S, good, bad, ref)
                S.revert(h)
                pre["drift"] = -(ng * pre["d_good"] + nb * pre["d_bad"]) / max(1, ng + nb)
                pre["dose"], pre["edit"] = k, e
                pts.append(pre)
            return pts

        # Null line: any perturbation costs the (longer) failed traces more log-prob than the correct ones, so the raw
        # margin grows with drift whatever the edit does. Random control edits estimate that slope each generation;
        # candidates are ranked by their margin above it.
        beta, ctrl = 0.0, []
        if ref is not None and args.null_controls and META["use_prescreen"]:
            names = contrast_names()
            for j in range(args.null_controls):
                ce, _ = validate(random_edit(rng, names))
                if ce is not None:
                    ctrl += [p for p in sweep_points(ce) if 0 < p["drift"] <= 2 * args.drift_cap]
            if ctrl:
                beta = sum(p["drift"] * p["margin"] for p in ctrl) / sum(p["drift"] ** 2 for p in ctrl)
            say(f"gen {gen}: null slope {beta:+.3f} from {len(ctrl)} control points")
            with open(os.path.join(gdir, "null_controls.json"), "w") as f:
                json.dump({"beta": beta, "points": [{k: v for k, v in p.items() if k != "edit"} | {"target": target(p["edit"])}
                                                    for p in ctrl]}, f, indent=1)
        for c in cands:
            if c["edit"] is None or ref is None:
                continue
            sweep = sweep_points(c["edit"])
            for pre in sweep:
                pre["excess"] = pre["margin"] - beta * pre["drift"]
                say(f"gen {gen} cand {c['idx']} dose {pre['dose']:g}: margin {pre['margin']:+.4f} excess {pre['excess']:+.4f} "
                    f"(good {pre['d_good']:+.4f}, bad {pre['d_bad']:+.4f}, drift {pre['drift']:+.4f})")
            ok = [p for p in sweep if p["drift"] <= args.drift_cap]
            if not ok:
                c["edit"], c["invalid"] = None, "every dose moved the model's own traces beyond the drift cap" if sweep else "apply failed"
                continue
            best = max(ok, key=lambda p: (p["excess"], -p["dose"])) if META["use_prescreen"] else max(ok, key=lambda p: p["dose"])
            c["edit"], c["dose"] = best["edit"], best["dose"]
            c["pre"] = {k: v for k, v in best.items() if k != "edit"}
            c["sweep"] = [{k: v for k, v in p.items() if k != "edit"} for p in sweep]
        ranked = sorted([c for c in cands if c["edit"] is not None], key=lambda c: -c.get("pre", {}).get("excess", 0))
        if not META["use_prescreen"]:
            ranked = [c for c in cands if c["edit"] is not None]
        for c in ranked[args.gen_screen:]:
            c["skip"] = True

        # Screen the selected candidates on the same problems as the base.
        results = []
        for c in cands:
            if c.get("skip"):
                rec = {"gen": gen, "author": c["author"], "op": c["edit"]["op"], "target": target(c["edit"]),
                       "params": {k: v for k, v in c["edit"].items() if k not in ("op", "param")},
                       "verdict": "prescreen-skipped", "pre": c.get("pre"), "sweep": c.get("sweep"),
                       "why": c["proposal"].get("why", "")}
                with open(LOG, "a") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue
            if c["edit"] is None:
                rec = {"gen": gen, "author": c["author"], "op": c["proposal"].get("op"),
                       "target": c["proposal"].get("param", c["proposal"].get("layer")), "verdict": "invalid",
                       "reason": c["invalid"]}
                with open(LOG, "a") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue
            h = S.apply(engine_edit(c["edit"]))
            changed = h.changed if hasattr(h, "changed") else None
            cand = evaluate(S, screen, args.samples)
            S.revert(h)
            st = compare(base, cand)
            st["screen_pass"] = decide(st, [], "screen", efficiency=args.efficiency)
            st["tok_z"] = st["dtok"] / args.token_sd if args.token_sd else 0.0
            if st["tok_z"] <= -3 and st["diff"] >= -0.05:
                st["screen_pass"] = True
            if c.get("sig"):
                with open(sig_path, "a") as f:
                    f.write(json.dumps({"sig": c["sig"], "base": len(accepted), "gen": gen}) + "\n")
            results.append((c, st, changed))
            say(f"gen {gen} cand {c['idx']} [{c['author']}] {c['edit']['op']} {target(c['edit'])} x{c.get('dose', 1):g}: "
                f"diff {st['diff']:+.3f} [{st['lo90']:+.3f},{st['hi90']:+.3f}] dtok {st['dtok']:+.0f} "
                f"({100 * st.get('tok', {}).get('rel', 0):+.1f}%) pass={st['screen_pass']}")

        # Confirm the best screen survivors on fresh problems (and the guard); accept the best confirmed one.
        survivors = sorted([r for r in results if r[1]["screen_pass"]],
                           key=lambda r: -max(r[1]["diff"] / 0.06, -r[1]["tok_z"]))[:2] if confirm else []
        base_c = evaluate(S, confirm, args.confirm_samples) if survivors else []
        base_g = evaluate(S, gmb, 1) if survivors and gmb else []
        # Two-look confirm (group-sequential): look 1 accepts only on the 98% bound; a promising edit (accuracy up, or
        # with --efficiency accuracy not lower and tokens down) that misses it gets a second set of fresh problems and
        # guard problems, and the pooled look 2 is decided by the usual 90% rule. Simulated at the observed confirm
        # noise (SE 0.051 at 48 problems x 2 samples): false accepts 5.3% (one look: 5.0%), accepts of a true +5pp edit
        # 40% (25%); --confirm-samples 4 (11-04) lowers the per-problem sampling part of that noise.
        ext = frontier[args.mb + args.confirm_mb: args.mb + 2 * args.confirm_mb] if args.confirm_looks == 2 else []
        rng2 = random.Random(args.seed * 1000 + gen)    # own stream: the main rng sequence stays as before
        gmb2 = rng2.sample([g for g in guard if g not in gmb], min(args.guard_mb, len(guard) - len(gmb))) if ext and guard else []
        base_c2 = base_g2 = base_l = None
        best = None
        for c, st, _ in survivors:
            h = S.apply(engine_edit(c["edit"]))
            cc = evaluate(S, confirm, args.confirm_samples)
            cg = evaluate(S, gmb, 1) if gmb else []
            S.revert(h)
            stc = compare(base_c, cc)
            stg = compare(base_g, cg) if gmb else {"n": 0}
            looks = 1
            if ext:
                tok = stc.get("tok", {"n": 0})
                ok1 = decide(stc, [stg], "confirm", guard_tol=args.guard_tol, efficiency=args.efficiency, bound="98")
                promising = stc["n"] > 0 and (stc["diff"] > 0 or (args.efficiency and stc["diff"] >= 0
                                                                  and tok["n"] > 0 and tok["rel"] < 0))
                if not ok1 and promising:
                    say(f"  confirm cand {c['idx']} look 1: diff {stc['diff']:+.3f} [{stc['lo98']:+.3f},{stc['hi98']:+.3f}]98 "
                        f"tokens {100 * tok.get('rel', 0):+.1f}% guard {stg.get('diff', 0):+.3f}: promising, "
                        f"extending by {len(ext)} problems")
                    if base_c2 is None:
                        base_c2 = evaluate(S, ext, args.confirm_samples)
                        base_g2 = evaluate(S, gmb2, 1) if gmb2 else []
                    h = S.apply(engine_edit(c["edit"]))
                    cc += evaluate(S, ext, args.confirm_samples)
                    cg += evaluate(S, gmb2, 1) if gmb2 else []
                    S.revert(h)
                    st["confirm_look1"], st["guard_look1"] = stc, stg
                    stc = compare(base_c + base_c2, cc)
                    stg = compare(base_g + base_g2, cg) if gmb else {"n": 0}
                    looks = 2
            st["confirm"], st["guard"], st["looks"] = stc, stg, looks
            st["confirm_pass"] = (decide(stc, [stg], "confirm", guard_tol=args.guard_tol, efficiency=args.efficiency,
                                         bound="98" if ext and looks == 1 else "90"))
            ret = retention(base_c + (base_c2 if looks == 2 else []), cc)
            st["retain"] = ret
            tight_ok = st["confirm_pass"]
            if args.retain_tol is not None and ret["n"] and ret["diff"] < -args.retain_tol:
                st["confirm_pass"] = False
            if st["confirm_pass"] and args.loose_budget:
                probs_l, tight = (confirm + ext)[: args.loose_mb], args.budget
                args.budget = args.loose_budget
                try:
                    if base_l is None:
                        base_l = evaluate(S, probs_l, args.samples)
                    h = S.apply(engine_edit(c["edit"]))
                    cl = evaluate(S, probs_l, args.samples)
                    S.revert(h)
                finally:
                    args.budget = tight
                stl = compare(base_l, cl)
                st["loose"] = stl
                st["confirm_pass"] = stl["diff"] >= -args.loose_tol and stl["lo90"] >= args.loose_floor
                say(f"  loose cand {c['idx']} at {args.loose_budget}: diff {stl['diff']:+.3f} [{stl['lo90']:+.3f},{stl['hi90']:+.3f}] "
                    f"tokens {100 * stl.get('tok', {}).get('rel', 0):+.1f}% pass={st['confirm_pass']}")
            say(f"  confirm cand {c['idx']}{' (pooled, 2 looks)' if looks == 2 else ''}: diff {stc['diff']:+.3f} "
                f"[{stc['lo90']:+.3f},{stc['hi90']:+.3f}] tokens {100 * stc.get('tok', {}).get('rel', 0):+.1f}% "
                f"guard {stg.get('diff', 0):+.3f} retain {st['retain']['diff']:+.3f} (n {st['retain']['n']}) pass={st['confirm_pass']}")
            if st["confirm_pass"] and (best is None or stc["diff"] > best[1]["confirm"]["diff"]):
                best = (c, st)
            # Holds at the tight budget but costs answers it used to get: the same edit at half strength gets one more try per halving.
            weaker = dosed(c["edit"], 0.5) if c["edit"]["op"] in ("amplify", "combo") else None
            if args.loose_budget and tight_ok and not st["confirm_pass"] and weaker and max(abs(x["alpha"]) for x in weaker.get("edits", [weaker])) >= args.min_retry_alpha:
                say(f"  cand {c['idx']}: holds at {args.budget} but fails retention or the loose budget; retrying at half strength {target(weaker)}")
                c2 = dict(c, edit=weaker, dose=c.get("dose", 1) / 2, idx=f"{c['idx']}h")
                st2 = {k: v for k, v in st.items() if k not in ("confirm", "guard", "looks", "retain", "loose", "confirm_look1", "guard_look1", "confirm_pass")}
                survivors.append((c2, st2, None))
                results.append((c2, st2, None))
        for c, st, changed in results:
            verdict = "accepted" if best and c is best[0] else "confirm-failed" if "confirm" in st else \
                "screen-failed" if not st["screen_pass"] else "not-confirmed"
            rec = {"gen": gen, "author": c["author"], "op": c["edit"]["op"], "target": target(c["edit"]),
                   "params": {k: v for k, v in c["edit"].items() if k not in ("op", "param", "u", "v")},
                   "dose": c.get("dose"), "sweep": c.get("sweep"), "tok": st.get("tok"),
                   "why": c["proposal"].get("why", ""), "verdict": verdict, "diff": st["diff"], "dtok": st["dtok"],
                   "pre": c.get("pre"),
                   "screen": {k: st[k] for k in ("n", "diff", "lo90", "hi90", "acc_base", "acc_cand")},
                   "confirm": st.get("confirm"), "guard": st.get("guard"), "looks": st.get("looks"),
                   "retain": st.get("retain"), "loose": st.get("loose"),
                   "confirm_look1": st.get("confirm_look1"), "guard_look1": st.get("guard_look1"), "changed": changed}
            with open(LOG, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if best:
            S.apply(engine_edit(best[0]["edit"]))
            accepted.append(best[0])
            with open(ACCEPTED, "a") as f:
                f.write(json.dumps({"gen": gen, "author": best[0]["author"], "edit": best[0]["edit"],
                                    "confirm": best[1]["confirm"], "guard": best[1]["guard"]}, ensure_ascii=False) + "\n")
            say(f"gen {gen}: ACCEPTED cand {best[0]['idx']} ({best[0]['author']})")
        else:
            say(f"gen {gen}: no edit accepted")
        open(os.path.join(gdir, "DONE"), "w").write("accepted" if best else "none")
        if args.checkpoint_cmd:
            import subprocess
            rc = subprocess.call(args.checkpoint_cmd, shell=True)
            if rc == 4:
                say(f"gen {gen}: checkpoint asked to yield the card; exiting (resume restarts at gen {gen + 1})")
                return


# Other architectures (Gemma 4, OLMo 3): their own menu, validation, dosing, random control and decoding settings
# (rsi/arch_xf.py) replace the Qwen3.5 definitions above, which stay unchanged for Qwen models.
from rsi import arch_xf
FAMILY = arch_xf.family(args.model)
if FAMILY != "qwen3_5":
    if args.mm and FAMILY != "gemma4":
        raise SystemExit("--mm lineages exist only for Qwen3.5 and Gemma 4")
    _X = arch_xf.Arch(args.model, args, resolve_dir)
    MENU, DIR_HELP, EDIT_FORMAT, WRITERS, PARAM_RE, SAMPLING = (_X.MENU, _X.DIR_HELP, _X.EDIT_FORMAT, _X.WRITERS,
                                                                _X.PARAM_RE, _X.SAMPLING)
    validate, engine_edit, dosed, random_edit = _X.validate, _X.engine_edit, _X.dosed, _X.random_edit

_S = None                                                       # the Surgeon (L5 edits read the merger bias)
if args.mm:
    from rsi import mm_eval
    mm_eval.setup(args, resolve_dir)

# Several edits as one candidate. Stop reads add across layers while each edit's drift grows with the square of its strength.
_one = dict(validate=validate, engine_edit=engine_edit, dosed=dosed, target=target)


def validate(e):
    if "combo" not in e:
        return _one["validate"](e)
    parts = as_list(e["combo"])
    if not 2 <= len(parts) <= 8:
        return None, "a combo has 2 to 8 edits"
    out = []
    for x in parts:
        clean, why = _one["validate"](x)
        if clean is None:
            return None, f"combo part {x}: {why}"
        out.append(clean)
    return {"op": "combo", "edits": out}, None


def engine_edit(e):
    return [_one["engine_edit"](x) for x in e["edits"]] if e["op"] == "combo" else _one["engine_edit"](e)


def dosed(e, k):
    if e.get("op") != "combo":
        return _one["dosed"](e, k)
    parts = [_one["dosed"](x, k) for x in e["edits"]]
    return None if None in parts else {"op": "combo", "edits": parts}


def target(e):
    return " + ".join(str(_one["target"](x)) for x in e["edits"]) if e.get("op") == "combo" else _one["target"](e)

if __name__ == "__main__":
    main()
