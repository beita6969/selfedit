"""Other half of the split of the patch gain, which no checkpoint on stock vLLM can hold: same tower on the same patches, four times the text tokens. The merger's input square has one patch shown four times,
so each patch gets its own token. Same read as rsi/mm_read.py (thinking off, log-odds of the right option at "\\boxed{"). Calibration: the finer patch moved this read by +0.115 [+0.032,+0.199].

Usage: python -u rsi/mm_split.py --model M --pool POOL --effort medium --out OUT [--thin blink:3,cvbench:10] [--max-problems 0] [--device cuda:0]
"""
import argparse, json, re, time
import numpy as np
import torch

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
vis = model.model.visual
merger_forward = vis.merger.forward


def set_merge(m):
    proc.image_processor.merge_size = m
    vis.spatial_merge_size = m
    vis.config.spatial_merge_size = m
    model.config.vision_config.spatial_merge_size = m
    vis.merger.forward = merger_forward if m == 2 else lambda x: merger_forward(x.repeat_interleave(4, 0))


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
print(f"{len(work)} option problems", flush=True)


def read(p, prefix, cand, g):
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
    with torch.inference_mode():
        logits = model(**feed, use_cache=False).logits[0, -1].float()
    lp = logits.log_softmax(-1)[torch.tensor(cand, device=logits.device)].cpu().numpy().astype(np.float64)
    return lp[g] - np.logaddexp.reduce(np.delete(lp, g)), int(feed["input_ids"].shape[1])


def report(marg, done):
    rng = np.random.default_rng(0)
    keep = [i for i in done if not np.isnan(marg[i]).any()]
    d = marg[keep, 1] - marg[keep, 0]
    m = d[rng.integers(0, d.size, (2000, d.size))].mean(1)
    out = [f"{len(keep)} option problems read, {len(done) - len(keep)} skipped; prompt tokens {np.mean([lens[i, 0] for i in keep]):.0f} -> {np.mean([lens[i, 1] for i in keep]):.0f}",
           f"tokens x4, same tower: margin {d.mean():+.3f} [{np.quantile(m, .05):+.3f},{np.quantile(m, .95):+.3f}]; right {(marg[keep, 0] > 0).mean():.3f} -> {(marg[keep, 1] > 0).mean():.3f}"]
    for b in sorted({work[i][0]["bench"] for i in keep}):
        sel = [j for j, i in enumerate(keep) if work[i][0]["bench"] == b]
        out.append(f"{b} {len(sel)}: margin {d[sel].mean():+.3f}; right {(marg[keep][sel, 0] > 0).mean():.3f} -> {(marg[keep][sel, 1] > 0).mean():.3f}")
    return "\n".join(out) + "\n"


marg, lens, t0, done = np.full((len(work), 2), np.nan), np.zeros((len(work), 2)), time.time(), []
for w, (p, prefix, cand, g) in enumerate(work):
    done.append(w)
    try:
        for a, m in enumerate((2, 1)):
            set_merge(m)
            marg[w, a], lens[w, a] = read(p, prefix, cand, g)
    except torch.OutOfMemoryError:
        marg[w] = np.nan
        torch.cuda.empty_cache()
        print(f"{p['id']}: out of memory, skipped", flush=True)
    if (w + 1) % 50 == 0:
        print(f"{w + 1} problems, {time.time() - t0:.0f}s\n" + report(marg, done), flush=True)
        np.savez(args.out + ".npz", marg=marg, lens=lens)

text = report(marg, done)
print(text, flush=True)
open(args.out + ".txt", "w").write(text)
np.savez(args.out + ".npz", marg=marg, lens=lens, ids=np.array([x[0]["id"] for x in work]))
print("done", flush=True)
