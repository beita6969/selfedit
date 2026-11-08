"""How much the model looks at the image while it thinks: share of attention on image tokens at sampled positions of its own traces, per full-attention layer and head, image in the forward pass.
Compares right and failed traces of the same problem, early and late positions. Separates a model that has the image in view but misreads it from one that stops looking.
Writes OUT.npz (per trace: mass per position and layer; per layer and head: mean mass, right minus wrong) and OUT.txt (the summary).

Usage: python -u rsi/mm_look.py --model M --traces CALIB.jsonl --pool POOL --effort medium --out OUT [--max-traces 600] [--positions 24] [--device cuda:0]
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--traces", required=True)
ap.add_argument("--pool", required=True)
ap.add_argument("--effort", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--max-traces", type=int, default=600)
ap.add_argument("--positions", type=int, default=24)
ap.add_argument("--max-tok", type=int, default=4000)
args = ap.parse_args()

from transformers import AutoProcessor
from transformers.models.qwen3_5 import modeling_qwen3_5 as M
proc = AutoProcessor.from_pretrained(args.model)
tk = proc.tokenizer
model = M.Qwen3_5ForConditionalGeneration.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa")
model.to(args.device).eval().requires_grad_(False)
lang = model.model.language_model
IMG = model.config.image_token_id
FULL = [l for l in range(len(lang.layers)) if l % 4 == 3]

state = {"on": False, "idx": None, "img": None, "rows": [], "heads": []}
lang.register_forward_pre_hook(lambda mod, inp: state.update(on=True))
lang.register_forward_hook(lambda mod, inp, out: state.update(on=False))
orig = F.scaled_dot_product_attention


def wrapped(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kw):
    if state["on"]:
        idx = state["idx"]
        keys = k.float().repeat_interleave(q.size(1) // k.size(1), 1)           # 24 query heads share 4 key groups
        s = torch.matmul(q[:, :, idx, :].float(), keys.transpose(-1, -2)) * (scale if scale is not None else 1 / math.sqrt(k.size(-1)))
        cols = torch.arange(k.size(2), device=q.device)[None, :]
        s = s.masked_fill((cols > idx[:, None])[None, None], float("-inf"))     # a position sees itself and what came before
        m = s.softmax(-1)[..., state["img"]].sum(-1)[0]                         # [heads, positions]
        state["rows"].append(m.mean(0).cpu().numpy())
        state["heads"].append(m.mean(1).cpu().numpy())
    return orig(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kw)


F.scaled_dot_product_attention = wrapped
torch.nn.functional.scaled_dot_product_attention = wrapped
for mod in (M,):                                # a module that imported the function by name keeps the original otherwise
    if hasattr(mod, "scaled_dot_product_attention"):
        mod.scaled_dot_product_attention = wrapped

pool = {p["id"]: p for p in map(json.loads, open(args.pool))}
by = {}
for r in map(json.loads, open(args.traces)):
    if r.get("reasoning") and r["id"] in pool:
        by.setdefault(r["id"], []).append(r)
paired = [k for k in sorted(by) if any(r.get("ok") for r in by[k]) and any(not r.get("ok") for r in by[k])]
rest = [k for k in sorted(by) if k not in set(paired)]
rng = np.random.default_rng(0)
rng.shuffle(paired), rng.shuffle(rest)
order = [r for k in paired + rest for r in by[k]][: args.max_traces]
print(f"{len(order)} traces, {len(paired)} paired problems", flush=True)


def content(p):
    imgs = [{"type": "image", "image": q} for q in p["image_paths"]]
    parts = p["prompt"].split("<image>")
    if len(parts) - 1 == len(imgs) and imgs:
        c = []
        for j, txt in enumerate(parts):
            if txt.strip():
                c.append({"type": "text", "text": txt.strip()})
            if j < len(imgs):
                c.append(imgs[j])
        return c
    return imgs + [{"type": "text", "text": p["prompt"]}]


recs, heads_ok, heads_bad, t0 = [], [], [], time.time()
for r in order:
    th = tk(r["reasoning"], add_special_tokens=False).input_ids[: args.max_tok]
    if len(th) < 8:
        continue
    inp = proc.apply_chat_template([{"role": "user", "content": content(pool[r["id"]])}], tokenize=True, add_generation_prompt=True, return_dict=True,
                                   return_tensors="pt", reasoning_effort=args.effort, enable_thinking=True)
    head = inp["input_ids"][0]
    n = head.size(0)
    tail = torch.tensor(th, dtype=torch.long)[None]
    feed = {"input_ids": torch.cat([inp["input_ids"], tail], 1), "attention_mask": torch.ones((1, n + tail.shape[1]), dtype=torch.long)}
    if "mm_token_type_ids" in inp:
        feed["mm_token_type_ids"] = torch.cat([inp["mm_token_type_ids"], torch.zeros_like(tail)], 1)
    for key in ("pixel_values", "image_grid_thw"):
        if key in inp:
            feed[key] = inp[key]
    feed = {key: v.to(args.device) for key, v in feed.items()}
    feed["pixel_values"] = feed["pixel_values"].to(torch.bfloat16)
    T = feed["input_ids"].size(1)
    pos = np.unique(np.linspace(n, T - 1, args.positions).astype(int))
    state.update(idx=torch.tensor(pos, device=args.device), img=torch.nonzero(feed["input_ids"][0] == IMG).flatten(), rows=[], heads=[])
    with torch.inference_mode():
        model.model(**feed, use_cache=False)
    if len(state["rows"]) != len(FULL):
        raise RuntimeError(f"{len(state['rows'])} attention calls seen, {len(FULL)} full-attention layers expected")
    mass = np.stack(state["rows"], 1)           # [positions, full layers]
    n_img = int(state["img"].numel())
    recs.append(dict(id=r["id"], bench=r["id"].split("/")[0], ok=bool(r.get("ok")), mass=mass, frac=(pos - n) / max(1, len(th)), chance=n_img / (pos + 1)))
    (heads_ok if r.get("ok") else heads_bad).append(np.stack(state["heads"], 0))
    if len(recs) % 25 == 0:
        print(f"{len(recs)} traces, {T} tokens last, {time.time() - t0:.0f}s", flush=True)

w = [f"{len(recs)} traces; layers {FULL}"]
all_mass = np.concatenate([x["mass"] for x in recs], 0)
all_ch = np.concatenate([x["chance"] for x in recs], 0)
w.append(f"attention on image tokens: mean {all_mass.mean():.3f}; if attention were uniform over what came before: {all_ch.mean():.3f}")
for name, lo, hi in (("first 10 %", 0, .1), ("10-25 %", .1, .25), ("25-50 %", .25, .5), ("last half", .5, 1.01)):
    row = [x["mass"][(x["frac"] >= lo) & (x["frac"] < hi)].mean() for x in recs if ((x["frac"] >= lo) & (x["frac"] < hi)).any()]
    rc = [x["chance"][(x["frac"] >= lo) & (x["frac"] < hi)].mean() for x in recs if ((x["frac"] >= lo) & (x["frac"] < hi)).any()]
    w.append(f"  position {name}: {np.mean(row):.3f} (uniform {np.mean(rc):.3f})")
for l, name in enumerate(FULL):
    w.append(f"  L{name}: mean {all_mass[:, l].mean():.3f}")
idx = {}
for x in recs:
    idx.setdefault(x["id"], []).append(x)
rng = np.random.default_rng(1)


def within(val):
    """Right minus failed on the same problem, for a value per trace."""
    d = []
    for k, xs in idx.items():
        a = [val(x) for x in xs if x["ok"]]
        b = [val(x) for x in xs if not x["ok"]]
        if a and b:
            d.append(np.mean(a) - np.mean(b))
    d = np.array(d)
    m = d[rng.integers(0, len(d), (2000, len(d)))].mean(1)
    return d.mean(), np.quantile(m, .05), np.quantile(m, .95), len(d)


w.append("right minus failed on the same problem, mean attention on the image:")
m = within(lambda x: x["mass"].mean())
w.append(f"  all positions: {m[0]:+.4f} [{m[1]:+.4f},{m[2]:+.4f}] on {m[3]} problems")
for name, lo, hi in (("first 10 %", 0, .1), ("10-25 %", .1, .25), ("25-50 %", .25, .5)):
    def band(x, lo=lo, hi=hi):
        k = (x["frac"] >= lo) & (x["frac"] < hi)
        return x["mass"][k].mean() if k.any() else x["mass"].mean()
    m = within(band)
    w.append(f"  position {name}: {m[0]:+.4f} [{m[1]:+.4f},{m[2]:+.4f}]")
for l, name in enumerate(FULL):
    m = within(lambda x, l=l: x["mass"][:, l].mean())
    w.append(f"  L{name}: {m[0]:+.4f} [{m[1]:+.4f},{m[2]:+.4f}]")
for b in sorted({x["bench"] for x in recs}):
    xs = [x for x in recs if x["bench"] == b]
    w.append(f"  {b}: mean {np.mean([x['mass'].mean() for x in xs]):.3f}; right {np.mean([x['mass'].mean() for x in xs if x['ok']] or [0]):.3f}, failed {np.mean([x['mass'].mean() for x in xs if not x['ok']] or [0]):.3f}")
H_ok, H_bad = np.mean(heads_ok, 0), np.mean(heads_bad, 0)       # [full layers, heads]
top = np.argsort(-(H_ok + H_bad).ravel())[:16]
w.append("heads that look at the image most (layer, head, mean attention on the image, right, failed): " + "; ".join(f"L{FULL[i // H_ok.shape[1]]} h{i % H_ok.shape[1]} {(H_ok.ravel()[i] + H_bad.ravel()[i]) / 2:.3f} ({H_ok.ravel()[i]:.3f}, {H_bad.ravel()[i]:.3f})" for i in top))
text = "\n".join(w) + "\n"
print(text, flush=True)
open(args.out + ".txt", "w").write(text)
np.savez(args.out + ".npz", layers=np.array(FULL), heads_right=H_ok, heads_failed=H_bad, ids=np.array([x["id"] for x in recs]), ok=np.array([x["ok"] for x in recs]),
         mean_mass=np.array([x["mass"].mean() for x in recs]))
print("done", flush=True)
