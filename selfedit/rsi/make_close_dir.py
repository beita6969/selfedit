"""Directions for the decision to close, from the model's own base traces. Last token of a line, all from one forward per trace:
close  natural end of a finished correct trace (it closes here)
goon   a line stating the correct answer, followed by more thinking (it could have closed)
plain  a line after the first mention that does not state the answer
pre    a line before the first mention
Contrasts saved as <out>/<name>.npz with <kind>/dir per layer, kinds mlp (the writer's output in the residual stream) and resid:
close_vs_goon, concluded_vs_plain, close_vs_plain, answered_vs_unanswered. Reliability: cosine between directions from even and odd traces; separation: AUC on the other half.
--style: only fast_vs_slow_early, line ends in the first --style-tok thinking tokens of the fast and the slow correct trace of a problem (tokens differ by at least --style-ratio), so position is equal.
--end-only: only right_vs_wrong_end, natural ends of finished correct traces against finished wrong ones (a base at the loose budget has enough of them).
Usage: python -u rsi/make_close_dir.py --model M --base BASE.jsonl --pool POOL --effort low --out LIB [--device cuda:0]"""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rsi.hazard import answer_marks

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--base", required=True)
ap.add_argument("--pool", required=True)
ap.add_argument("--effort", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--max-traces", type=int, default=0)
ap.add_argument("--per-kind", type=int, default=3)
ap.add_argument("--segment", type=int, default=4096)
ap.add_argument("--end-only", action="store_true")
ap.add_argument("--style", action="store_true")
ap.add_argument("--style-tok", type=int, default=1536)
ap.add_argument("--style-ratio", type=float, default=1.5)
args = ap.parse_args()

from transformers import AutoConfig, AutoTokenizer
from transformers.cache_utils import DynamicCache
fam = json.load(open(os.path.join(args.model, "config.json"))).get("model_type")
full = AutoConfig.from_pretrained(args.model)
cfg = getattr(full, "text_config", full)
tk = AutoTokenizer.from_pretrained(args.model)
gemma = fam.startswith("gemma")
if gemma:
    from transformers.models.gemma4 import modeling_gemma4 as M
    model = M.Gemma4ForCausalLM.from_pretrained(args.model, config=cfg, dtype=torch.bfloat16, attn_implementation="sdpa",
                                                key_mapping={r"^model\.language_model\.": "model."})
    lead = tk("<|channel>thought\n", add_special_tokens=False).input_ids
else:
    from transformers.models.qwen3_5 import modeling_qwen3_5 as M
    model = M.Qwen3_5ForCausalLM.from_pretrained(args.model, config=cfg, dtype=torch.bfloat16, attn_implementation="sdpa")
    lead = []
model.to(args.device).eval().requires_grad_(False)
layers = model.model.layers
L, H = len(layers), cfg.hidden_size
print(f"{fam}: {L} layers, hidden {H}", flush=True)

store, want = {"mlp": {}, "resid": {}}, {}          # kind -> layer -> rows at the wanted positions; want["idx"]: those positions in the segment


def keep(kind, l):
    def hook(mod, inp, out):
        x = (out[0] if isinstance(out, tuple) else out)[0]
        store[kind].setdefault(l, []).append(x[want["idx"]].float().cpu())
    return hook


for l, layer in enumerate(layers):
    (layer.post_feedforward_layernorm if gemma else layer.mlp).register_forward_hook(keep("mlp", l))
    layer.register_forward_hook(keep("resid", l))

pool = {p["id"]: p for p in map(json.loads, open(args.pool))}
recs = [json.loads(x) for x in open(args.base)]
vecs = {k: {"mlp": [], "resid": []} for k in ("close", "goon", "plain", "pre", "wrong", "fast", "slow")}
if args.style:          # (record, its class, pair index): one forward per trace, cut at --style-tok
    by = {}
    for r in recs:
        if r["finish"] != "length" and r.get("ok") and r.get("reasoning") and r["id"] in pool:
            by.setdefault(r["id"], []).append(r)
    recs, tag = [], {}
    for k in sorted(by):
        rs = sorted(by[k], key=lambda r: r["tokens"])
        if len(rs) >= 2 and rs[-1]["tokens"] >= args.style_ratio * rs[0]["tokens"]:
            for r, c in ((rs[0], "fast"), (rs[-1], "slow")):
                tag[len(recs)] = (c, len(recs) // 2)
                recs.append(r)
    print(f"{len(recs) // 2} pairs", flush=True)
owner = {k: [] for k in vecs}
t0, used = time.time(), 0


def spread(xs, n):
    xs = sorted(xs)
    if len(xs) <= n:
        return xs
    return [xs[int(i * (len(xs) - 1) / (n - 1))] for i in range(n)] if n > 1 else [xs[-1]]


for i, r in enumerate(recs):
    if args.max_traces and used >= args.max_traces:
        break
    if not r.get("reasoning"):
        continue
    th = tk(r["reasoning"], add_special_tokens=False).input_ids
    if args.style:
        th = th[: args.style_tok]
    finished = r["finish"] != "length"
    e = len(th)
    while e > 1 and not tk.decode([th[e - 1]]).strip():
        e -= 1
    if args.style:
        pos = {tag[i][0]: spread([k for k, t in enumerate(th) if k >= 1 and "\n" in tk.decode([t])], 8)}
    elif args.end_only:
        if not finished:
            continue
        pos = {"close" if r.get("ok") else "wrong": [e]}
    else:
        first, lines = answer_marks(tk, th, r["answer"])
        if first is None or first < 0:
            continue
        breaks = [k for k, t in enumerate(th) if "\n" in tk.decode([t])]
        body = [k for k in lines if k < len(th) - 2]
        stated = set(lines)
        pos = {"goon": spread(body, args.per_kind),
               "plain": spread([k for k in breaks if k > first and k not in stated and k >= 1], args.per_kind),
               "pre": spread([k for k in breaks if int(0.5 * first) < k < first and k >= 1], args.per_kind),
               "close": [e] if finished and r.get("ok") else []}
    if not any(pos.values()):
        continue
    head = tk(tk.apply_chat_template([{"role": "user", "content": pool[r["id"]]["prompt"]}], tokenize=False, add_generation_prompt=True,
                                     reasoning_effort=args.effort, enable_thinking=True), add_special_tokens=False).input_ids + lead
    ids = torch.tensor(head + th, device=args.device)
    ats = sorted({len(head) + k - 1 for ks in pos.values() for k in ks})    # k: the break token (or end) index, so its line's last token is k - 1
    cache = DynamicCache(config=cfg) if len(ids) > args.segment else None
    for s in store.values():
        s.clear()
    with torch.inference_mode():
        for off in range(0, len(ids), args.segment):
            want["idx"] = torch.tensor([a - off for a in ats if off <= a < off + args.segment], dtype=torch.long, device=args.device)
            model.model(input_ids=ids[None, off:off + args.segment], past_key_values=cache, use_cache=cache is not None)
    del cache
    rows = {space: torch.stack([torch.cat(store[space][l]) for l in range(L)], 1).numpy() for space in store}     # [positions, layers, H]
    for kind, ks in pos.items():
        for k in ks:
            at = ats.index(len(head) + k - 1)
            for space in rows:
                vecs[kind][space].append(rows[space][at].astype(np.float16))
            owner[kind].append(tag[i][1] if args.style else i)
    used += 1
    if used % 20 == 0:
        print(f"{used} traces, {len(ids)} tokens last, {time.time() - t0:.0f}s, " + ", ".join(f"{k} {len(v)}" for k, v in owner.items()), flush=True)
print(f"{used} traces in {time.time() - t0:.0f}s: " + ", ".join(f"{k} {len(v)}" for k, v in owner.items()), flush=True)


def trace_mean(kind, space, mask=None):
    """Mean per trace, then across traces. [layers, H]; with mask, only traces whose index satisfies it."""
    own = np.array(owner[kind])
    x = np.stack(vecs[kind][space]).astype(np.float32)
    keep_t = [t for t in sorted(set(owner[kind])) if mask is None or mask(t)]
    return np.mean([x[own == t].mean(0) for t in keep_t], axis=0)


def contrast(A, B, space, mask=None):
    a = np.mean([trace_mean(k, space, mask) for k in A], axis=0)
    b = np.mean([trace_mean(k, space, mask) for k in B], axis=0)
    return a - b


def auc(p, q):
    r = np.argsort(np.argsort(np.r_[p, q])) + 1
    return (r[: len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(q))


CONTRASTS = {"close_vs_goon": (["close"], ["goon"]), "concluded_vs_plain": (["close", "goon"], ["plain"]),
             "close_vs_plain": (["close"], ["plain"]), "answered_vs_unanswered": (["plain"], ["pre"])}
if args.end_only:
    CONTRASTS = {"right_vs_wrong_end": (["close"], ["wrong"])}
if args.style:
    CONTRASTS = {"fast_vs_slow_early": (["fast"], ["slow"])}
os.makedirs(args.out, exist_ok=True)
report = {}
even, odd = (lambda t: t % 2 == 0), (lambda t: t % 2 == 1)
for name, (A, B) in CONTRASTS.items():
    out = {}
    for space in ("mlp", "resid"):
        d = contrast(A, B, space)
        out[f"{space}/dir"] = (d / np.maximum(1e-9, np.linalg.norm(d, axis=1, keepdims=True))).astype(np.float32)
        out[f"{space}/norm"] = np.linalg.norm(d, axis=1).astype(np.float32)
    d0, d1 = contrast(A, B, "mlp", even), contrast(A, B, "mlp", odd)
    rel = (d0 * d1).sum(1) / np.maximum(1e-9, np.linalg.norm(d0, axis=1) * np.linalg.norm(d1, axis=1))
    sep = np.zeros(L)
    for l in range(L):                              # projection on the even-trace direction, scored on odd traces, and the reverse
        s = []
        for dd, pick in ((d0, odd), (d1, even)):
            u = dd[l] / max(1e-9, np.linalg.norm(dd[l]))
            P = [float(x[l].astype(np.float32) @ u) for k in A for x, t in zip(vecs[k]["mlp"], owner[k]) if pick(t)]
            Q = [float(x[l].astype(np.float32) @ u) for k in B for x, t in zip(vecs[k]["mlp"], owner[k]) if pick(t)]
            s.append(auc(np.array(P), np.array(Q)))
        sep[l] = np.mean(s)
    out["mlp/half_cos"], out["mlp/auc"] = rel.astype(np.float32), sep.astype(np.float32)
    np.savez(os.path.join(args.out, f"{name}.npz"), **out)
    top = sorted(range(L), key=lambda l: -sep[l])[:12]
    report[name] = {"top": [dict(layer=l, auc=round(float(sep[l]), 3), half_cos=round(float(rel[l]), 3)) for l in top]}
    print(name, "top AUC " + ", ".join(f"L{l} {sep[l]:.2f} (cos {rel[l]:+.2f})" for l in top), flush=True)
json.dump(report, open(os.path.join(args.out, "close_dirs_report.json"), "w"), indent=1)
print("done", flush=True)
