"""Does looking at the image more change what the model reads from it? Causal test before any weight edit: thinking off, one forward per problem, log-odds of the right option at "\\boxed{"
(the read of rsi/mm_read.py), with a constant added to the attention score of image tokens in full-attention layers, for positions after the last image. Per arm: change of the read, change in
attention on the image at the last position. Blind arm: the same layers cannot see the image at all, which shows how much of the read goes through full attention.

Usage: python -u rsi/mm_boost.py --model M --pool POOL --effort medium --out OUT --look LOOK.npz [--thin blink:3,cvbench:10] [--max-problems 0] [--device cuda:0]
"""
import argparse, json, re, time
import numpy as np
import torch
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--pool", required=True)
ap.add_argument("--effort", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--look", required=True)
ap.add_argument("--thin", default="blink:3,cvbench:10")
ap.add_argument("--max-problems", type=int, default=0)
ap.add_argument("--device", default="cuda:0")
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
NH = model.config.text_config.num_attention_heads

z = np.load(args.look)
look = (z["heads_right"] + z["heads_failed"]) / 2                       # [full layers, heads]
TOP = {int(l): np.argsort(-look[i])[:6].tolist() for i, l in enumerate(z["layers"])}
INF = float("inf")
ARMS = [("base", 0.0, FULL, None)]
ARMS += [(f"all +{b}", b, FULL, None) for b in (0.5, 1.0, 2.0, 3.0)]
ARMS += [("all -1", -1.0, FULL, None), ("blind", -INF, FULL, None)]
ARMS += [("L3-27 +2", 2.0, [l for l in FULL if l <= 27], None), ("L31-47 +2", 2.0, [l for l in FULL if 31 <= l <= 47], None), ("L51-63 +2", 2.0, [l for l in FULL if l >= 51], None)]
ARMS += [("top 6 heads +2", 2.0, FULL, "top"), ("other heads +2", 2.0, FULL, "rest")]

state = {"on": False, "k": 0, "arm": None, "after": None, "img": None, "mass": None}
lang.register_forward_pre_hook(lambda mod, inp: state.update(on=True, k=0))
lang.register_forward_hook(lambda mod, inp, out: state.update(on=False))
orig = F.scaled_dot_product_attention


def wrapped(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kw):
    if not state["on"]:
        return orig(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kw)
    L = FULL[state["k"]]
    state["k"] += 1
    T = k.size(2)
    if attn_mask is None:
        mask = torch.zeros((T, T), dtype=q.dtype, device=q.device).masked_fill(torch.triu(torch.ones((T, T), dtype=torch.bool, device=q.device), 1), -INF)[None, None]
    else:
        mask = torch.zeros_like(attn_mask, dtype=q.dtype).masked_fill(~attn_mask, -INF)
    _, beta, layers, which = state["arm"]
    if beta and L in layers:
        add = torch.zeros((NH, T, T), dtype=q.dtype, device=q.device)
        heads = range(NH) if which is None else TOP[L] if which == "top" else [h for h in range(NH) if h not in TOP[L]]
        grid = add[list(heads)]
        grid[:, state["after"]:, state["img"]] = beta
        add[list(heads)] = grid
        mask = mask + add[None]
    if L in (39, 59):
        scale_ = scale if scale is not None else 1 / k.size(-1) ** .5
        s = torch.matmul(q[:, :, -1:, :].float(), k.float().repeat_interleave(NH // k.size(1), 1).transpose(-1, -2)) * scale_ + mask[:, :, -1:, :].float()
        state["mass"].append(float(s.softmax(-1)[..., state["img"]].sum(-1).mean()))
    return orig(q, k, v, attn_mask=mask, dropout_p=dropout_p, is_causal=False, scale=scale, **kw)


F.scaled_dot_product_attention = wrapped
torch.nn.functional.scaled_dot_product_attention = wrapped


def letters_of(p):
    found = re.findall(r"\(([A-Z])\)", p["prompt"]) or re.findall(r"(?:^|\s)([A-Z])\.\s", p["prompt"])
    return sorted(set(found) | {p["answer"]})


def content(p):
    imgs = [{"type": "image", "image": x} for x in p["image_paths"]]
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


def tok(t):
    return tk(t, add_special_tokens=False).input_ids


vis = model.model.visual
vis_forward = vis.forward
cache = {"out": None}


def vis_cached(*a, **k):                                                # one problem, many arms: the image goes through the tower once
    if cache["out"] is None:
        cache["out"] = vis_forward(*a, **k)
        cache["pool"] = cache["out"].pooler_output
    cache["out"].pooler_output = cache["pool"]                          # the caller replaces it with a tuple on every call
    return cache["out"]


vis.forward = vis_cached
thin = dict((kv.split(":")[0], int(kv.split(":")[1])) for kv in args.thin.split(",")) if args.thin else {}
seen, work = {}, []
for p in map(json.loads, open(args.pool)):
    if not p.get("image_paths") or p["kind"] != "mc":
        continue
    seen[p["bench"]] = seen.get(p["bench"], 0) + 1
    if seen[p["bench"]] % thin.get(p["bench"], 1):
        continue
    opts = letters_of(p)
    if len(opts) < 2:
        continue
    ids = [tok("\\boxed{" + x) for x in opts]
    n = next(i for i in range(len(ids[0]) + 1) if i == len(ids[0]) or any(i >= len(y) or y[i] != ids[0][i] for y in ids))
    if any(len(y) <= n for y in ids):
        continue
    work.append((p, ids[0][:n], [y[n] for y in ids], opts.index(p["answer"])))
if args.max_problems:
    work = work[: args.max_problems]
print(f"{len(work)} option problems, {len(ARMS)} arms", flush=True)

marg, masses, t0, kept = np.zeros((len(work), len(ARMS))), np.zeros((len(work), len(ARMS))), time.time(), []
for w, (p, prefix, cand, g) in enumerate(work):
    inp = proc.apply_chat_template([{"role": "user", "content": content(p)}], tokenize=True, add_generation_prompt=True, return_dict=True,
                                   return_tensors="pt", reasoning_effort=args.effort, enable_thinking=False)
    tail = torch.tensor(prefix, dtype=torch.long)[None]
    feed = {"input_ids": torch.cat([inp["input_ids"], tail], 1), "attention_mask": torch.ones((1, inp["input_ids"].shape[1] + tail.shape[1]), dtype=torch.long)}
    if "mm_token_type_ids" in inp:
        feed["mm_token_type_ids"] = torch.cat([inp["mm_token_type_ids"], torch.zeros_like(tail)], 1)
    for key in ("pixel_values", "image_grid_thw"):
        if key in inp:
            feed[key] = inp[key]
    feed = {key: v.to(args.device) for key, v in feed.items()}
    feed["pixel_values"] = feed["pixel_values"].to(torch.bfloat16)
    img = torch.nonzero(feed["input_ids"][0] == IMG).flatten()
    state.update(img=img, after=int(img[-1]) + 1)
    cache["out"] = None
    for a, arm in enumerate(ARMS):
        state.update(arm=arm, mass=[])
        with torch.inference_mode():
            logits = model(**feed, use_cache=False).logits[0, -1].float()
        lp = logits.log_softmax(-1)[torch.tensor(cand, device=logits.device)].cpu().numpy().astype(np.float64)
        marg[w, a] = lp[g] - np.logaddexp.reduce(np.delete(lp, g))
        masses[w, a] = np.mean(state["mass"])
    kept.append(p)
    if (w + 1) % 25 == 0:
        print(f"{w + 1} problems, {time.time() - t0:.0f}s; so far " + ", ".join(f"{ARMS[a][0]} {(marg[:w + 1, a] - marg[:w + 1, 0]).mean():+.3f}" for a in range(1, len(ARMS))), flush=True)

rng = np.random.default_rng(0)
base = marg[:, 0]
benches = sorted({p["bench"] for p in kept})
out = [f"{len(kept)} option problems; base right {(base > 0).mean():.3f}, attention on the image at the last position (L39, L59) {masses[:, 0].mean():.3f}"]
for a, arm in enumerate(ARMS[1:], 1):
    d = marg[:, a] - base
    m = d[rng.integers(0, d.size, (2000, d.size))].mean(1)
    out.append(f"{arm[0]}: margin {d.mean():+.3f} [{np.quantile(m, .05):+.3f},{np.quantile(m, .95):+.3f}] right {(marg[:, a] > 0).mean() - (base > 0).mean():+.3f} look {masses[:, a].mean():.3f} | " +
               " ".join(f"{b} {d[[i for i, p in enumerate(kept) if p['bench'] == b]].mean():+.3f}" for b in benches))
text = "\n".join(out) + "\n"
print(text, flush=True)
open(args.out + ".txt", "w").write(text)
np.savez(args.out + ".npz", marg=marg, masses=masses, ids=np.array([p["id"] for p in kept]), arms=np.array([a[0] for a in ARMS]))
print("done", flush=True)
