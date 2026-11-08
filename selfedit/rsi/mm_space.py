"""The model's second most frequent own proposal: keep 2-D spatial structure with a distance bias on attention. Causal test before any weight edit, same read as rsi/mm_read.py
(thinking off, one forward per problem, log-odds of the right option at "\\boxed{"): add -alpha * (|row distance| + |column distance|) to attention scores between tokens of one image,
in the vision tower (patch units) or in the text full-attention layers (merged-token units). Calibration: the finer patch moved this read by +0.115 [+0.032,+0.199].

Usage: python -u rsi/mm_space.py --model M --pool POOL --effort medium --out OUT [--thin blink:3,cvbench:10] [--max-problems 0] [--device cuda:0]
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
vis = model.model.visual
IMG = model.config.image_token_id
MERGE = model.config.vision_config.spatial_merge_size
DEPTH = model.config.vision_config.depth
INF = float("inf")

# (label, where, alpha, first block, last block); blocks only for the tower
ARMS = [("base", None, 0.0, 0, 0)]
ARMS += [(f"text -{a}", "text", a, 0, 0) for a in (0.05, 0.2, 0.5)]
ARMS += [(f"tower -{a}", "tower", a, 0, DEPTH - 1) for a in (0.02, 0.05, 0.1)]
ARMS += [("tower -0.1 blocks 0-8", "tower", 0.1, 0, 8), ("tower -0.1 blocks 18-26", "tower", 0.1, 18, 26), ("tower +0.02", "tower", -0.02, 0, DEPTH - 1)]

state = {"lang": False, "vision": False, "arm": ARMS[0], "calls": 0, "tdist": None, "vdist": None}
lang.register_forward_pre_hook(lambda mod, inp: state.update(lang=True))
lang.register_forward_hook(lambda mod, inp, out: state.update(lang=False))
orig = F.scaled_dot_product_attention


def wrapped(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kw):
    _, where, alpha, first, last = state["arm"]
    if state["vision"]:
        i = state["calls"]
        state["calls"] += 1
        n = len(state["vdist"])
        block = i // n
        if where != "tower" or not first <= block <= last:
            return orig(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kw)
        d = state["vdist"][i % n]
        if d.size(0) != q.size(2):
            raise RuntimeError(f"{q.size(2)} patches in the attention call, {d.size(0)} expected")
        return orig(q, k, v, attn_mask=(-alpha * d).to(q.dtype)[None, None], dropout_p=dropout_p, is_causal=False, scale=scale, **kw)
    if state["lang"] and where == "text":
        T = k.size(2)
        if attn_mask is None:
            mask = torch.zeros((T, T), dtype=q.dtype, device=q.device).masked_fill(torch.triu(torch.ones((T, T), dtype=torch.bool, device=q.device), 1), -INF)[None, None]
        else:
            mask = torch.zeros_like(attn_mask, dtype=q.dtype).masked_fill(~attn_mask, -INF)
        for idx, d in state["tdist"]:
            mask[:, :, idx[:, None], idx[None, :]] += (-alpha * d).to(q.dtype)
        return orig(q, k, v, attn_mask=mask, dropout_p=dropout_p, is_causal=False, scale=scale, **kw)
    return orig(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kw)


F.scaled_dot_product_attention = wrapped
torch.nn.functional.scaled_dot_product_attention = wrapped
vis_forward = vis.forward
cache = {"out": None, "pool": None}


def vis_arm(*a, **k):                                                   # arms that leave the tower alone share one pass through it
    tower = state["arm"][1] == "tower"
    if cache["out"] is not None and not tower:
        cache["out"].pooler_output = cache["pool"]                      # the caller replaces it with a tuple on every call
        return cache["out"]
    state.update(vision=True, calls=0)
    out = vis_forward(*a, **k)
    state["vision"] = False
    if not tower:
        cache.update(out=out, pool=out.pooler_output)
    return out


vis.forward = vis_arm


def l1(rows, cols, device):
    r, c = torch.as_tensor(rows, device=device, dtype=torch.float32), torch.as_tensor(cols, device=device, dtype=torch.float32)
    return (r[:, None] - r[None]).abs() + (c[:, None] - c[None]).abs()


def distances(grid, ids, device):
    """Per image: patch distances in the tower's own order (merge groups together), and merged-token distances at the image tokens' text positions."""
    vdist, tdist = [], []
    where = torch.nonzero(ids == IMG).flatten()
    start = 0
    for t, h, w in grid.tolist():
        hp = np.arange(h)[:, None].repeat(w, 1).reshape(h // MERGE, MERGE, w // MERGE, MERGE).transpose(0, 2, 1, 3).ravel()
        wp = np.arange(w)[None].repeat(h, 0).reshape(h // MERGE, MERGE, w // MERGE, MERGE).transpose(0, 2, 1, 3).ravel()
        vdist.append(l1(hp, wp, device))
        hm, wm = h // MERGE, w // MERGE
        tdist.append((where[start:start + hm * wm], l1(np.arange(hm).repeat(wm), np.tile(np.arange(wm), hm), device)))
        start += hm * wm
    if start != where.numel():
        raise RuntimeError(f"{where.numel()} image tokens, {start} from the grid")
    return vdist, tdist


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


def report(marg, done):
    rng = np.random.default_rng(0)
    keep = [i for i in done if not np.isnan(marg[i]).any()]
    base = marg[keep, 0]
    benches = sorted({work[i][0]["bench"] for i in keep})
    out = [f"{len(keep)} option problems read, {len(done) - len(keep)} skipped; base right {(base > 0).mean():.3f}"]
    for a, arm in enumerate(ARMS[1:], 1):
        d = marg[keep, a] - base
        m = d[rng.integers(0, d.size, (2000, d.size))].mean(1)
        out.append(f"{arm[0]}: margin {d.mean():+.3f} [{np.quantile(m, .05):+.3f},{np.quantile(m, .95):+.3f}] right {(marg[keep, a] > 0).mean() - (base > 0).mean():+.3f} | " +
                   " ".join(f"{b} {d[[j for j, i in enumerate(keep) if work[i][0]['bench'] == b]].mean():+.3f}" for b in benches))
    return "\n".join(out) + "\n"


marg, t0, done = np.full((len(work), len(ARMS)), np.nan), time.time(), []
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
    cache.update(out=None, pool=None)
    done.append(w)
    try:
        vdist, tdist = distances(feed["image_grid_thw"], feed["input_ids"][0], args.device)
        state.update(vdist=vdist, tdist=tdist)
        for a, arm in enumerate(ARMS):
            state["arm"] = arm
            with torch.inference_mode():
                logits = model(**feed, use_cache=False).logits[0, -1].float()
            lp = logits.log_softmax(-1)[torch.tensor(cand, device=logits.device)].cpu().numpy().astype(np.float64)
            marg[w, a] = lp[g] - np.logaddexp.reduce(np.delete(lp, g))
    except torch.OutOfMemoryError:
        marg[w] = np.nan
        state.update(vision=False, lang=False)
        torch.cuda.empty_cache()
        print(f"{p['id']}: out of memory, skipped", flush=True)
    if (w + 1) % 25 == 0:
        print(f"{w + 1} problems, {time.time() - t0:.0f}s\n" + report(marg, done), flush=True)
        np.savez(args.out + ".npz", marg=marg, arms=np.array([a[0] for a in ARMS]))

text = report(marg, done)
print(text, flush=True)
open(args.out + ".txt", "w").write(text)
np.savez(args.out + ".npz", marg=marg, arms=np.array([a[0] for a in ARMS]), ids=np.array([x[0]["id"] for x in work]))
print("done", flush=True)
