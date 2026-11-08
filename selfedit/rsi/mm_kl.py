"""What does an edit cost on what the model normally does? Teacher-forced on its own finished traces for real problems (image, question, thinking, answer): the next-token distribution at every position, edited against unedited, mean KL(unedited || edited).
The local law says KL grows with the square of the dose and adds over independent writers, so a dose split over n writers costs 1/n of one writer carrying it all. These arms test that:
the joint edits; two writers at several doses; one writer carrying a dose against four carrying a quarter each; writers with no signal at the same doses; each writer of the joint edits alone.

Usage: python -u rsi/mm_kl.py --model M --pred PRED.json --traces TRACES.jsonl --pool POOL.jsonl --out OUT [--n 60] [--max-tokens 800] [--top 100] [--seed 0] [--device cuda:0]; writes OUT.jsonl, OUT.txt
"""
import argparse, json, random, re, time
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--pred", required=True)
ap.add_argument("--traces", required=True)
ap.add_argument("--pool", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=60)
ap.add_argument("--max-tokens", type=int, default=800)
ap.add_argument("--top", type=int, default=100)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--device", default="cuda:0")
args = ap.parse_args()

from transformers import AutoProcessor
from transformers.models.qwen3_5 import modeling_qwen3_5 as M
proc = AutoProcessor.from_pretrained(args.model)
tk = proc.tokenizer
model = M.Qwen3_5ForConditionalGeneration.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa")
model.to(args.device).eval().requires_grad_(False)
pred = json.load(open(args.pred))
pool = {}
for l in open(args.pool):
    p = json.loads(l)
    pool[p["id"]] = p
rng = random.Random(args.seed)
seen, traces = set(), []
for l in open(args.traces):
    t = json.loads(l)
    if t["id"] in seen or t["finish"] != "stop" or t["tokens"] > args.max_tokens or t["id"] not in pool or not pool[t["id"]].get("image_paths"):
        continue
    seen.add(t["id"])
    traces.append(t)
rng.shuffle(traces)
traces = traces[: args.n]
print(f"{len(traces)} traces", flush=True)


def module(name):
    if name.startswith("v"):
        b, part = re.match(r"v(\d+)\.(attn|mlp)", name).groups()
        block = model.model.visual.blocks[int(b)]
        return block.attn.proj if part == "attn" else block.mlp.linear_fc2
    L = int(re.match(r"t(\d+)", name)[1])
    layer = model.model.language_model.layers[L]
    return layer.self_attn.o_proj if L % 4 == 3 else layer.linear_attn.out_proj


def feed_of(t):
    p = pool[t["id"]]
    inp = proc.apply_chat_template([{"role": "user", "content": [{"type": "image", "image": x} for x in p["image_paths"]] + [{"type": "text", "text": p["prompt"]}]}], tokenize=True, add_generation_prompt=True,
                                   return_dict=True, return_tensors="pt", reasoning_effort="medium")
    tail = torch.tensor(tk(t["reasoning"] + "\n</think>\n\n" + t["content"], add_special_tokens=False).input_ids, dtype=torch.long)[None]
    feed = {"input_ids": torch.cat([inp["input_ids"], tail], 1), "attention_mask": torch.ones((1, inp["input_ids"].shape[1] + tail.shape[1]), dtype=torch.long)}
    if "mm_token_type_ids" in inp:
        feed["mm_token_type_ids"] = torch.cat([inp["mm_token_type_ids"], torch.zeros_like(tail)], 1)
    feed.update(pixel_values=inp["pixel_values"], image_grid_thw=inp["image_grid_thw"])
    feed = {k: v.to(args.device) for k, v in feed.items()}
    feed["pixel_values"] = feed["pixel_values"].to(torch.bfloat16)
    return feed, inp["input_ids"].shape[1]


def read(feed, start):
    with torch.inference_mode():
        out = model(**feed, use_cache=False)
    return out.logits[0, start - 1:-1].float().log_softmax(-1)


t0 = time.time()
base = []
feeds = []
for t in traces:
    feed, start = feed_of(t)
    lp = read(feed, start)
    top = lp.topk(args.top, -1)
    base.append((top.indices, top.values))
    feeds.append((feed, start))
print(f"base read, {time.time() - t0:.0f}s, positions {sum(b[0].size(0) for b in base)}", flush=True)


def cost():
    total, flips, n, per = 0.0, 0, 0, []
    for (feed, start), (idx, val) in zip(feeds, base):
        q = read(feed, start)
        qt = q.gather(1, idx)
        p_rest = (1 - val.exp().sum(1)).clamp_min(1e-9)
        q_rest = (1 - qt.exp().sum(1)).clamp_min(1e-9)
        kl = (val.exp() * (val - qt)).sum(1) + p_rest * (p_rest.log() - q_rest.log())
        per.append(kl.mean().item())
        total += kl.sum().item()
        flips += (q.argmax(1) != idx[:, 0]).sum().item()
        n += idx.size(0)
    return total / n, flips / n, float(np.median(per))


def run(doses):
    """doses: writer name to fractional dose. Scales the writer's weight and bias, reads, restores the exact bytes."""
    saved = []
    for name, d in doses.items():
        for par in module(name).parameters():
            saved.append((par, par.data.clone()))
            par.data.mul_(1 + d)
    res = cost()
    for par, old in saved:
        par.data.copy_(old)
    return res


arms = []
writers = {}
for lab, v in pred.items():
    if "writers" in v:
        arms.append((lab, {k: d for k, d in v["writers"]}))
        writers.update({(k, round(d, 4)): d for k, d in v["writers"]})
union = {}
for lab in ("law:cell:K29", "law:trace:K18", "law:ring:K31"):
    for k, d in pred[lab]["writers"]:
        union[k] = union.get(k, 0) + d
arms.insert(0, ("all3", {k: max(-.45, min(.45, d)) for k, d in union.items()}))
for lab in ("law:cell:K8", "law:trace:K8", "law:ring:K8"):
    arms.append((lab.replace("law", "flip"), {k: -d for k, d in pred[lab]["writers"]}))
for w in ("t24.attn", "v8.mlp"):
    for d in (-.6, -.3, -.15, .15, .3, .6):
        arms.append((f"dose:{w}:{d:+.2f}", {w: d}))
arms.append(("split:text:one", {"t24.attn": .6}))
arms.append(("split:text:four", {k: .15 for k in ("t24.attn", "t23.attn", "t32.attn", "t62.attn")}))
arms.append(("split:tower:one", {"v8.mlp": .6}))
arms.append(("split:tower:four", {k: .15 for k in ("v8.mlp", "v25.mlp", "v4.attn", "v6.mlp")}))
everyone = [f"v{b}.{p}" for b in range(27) for p in ("attn", "mlp")] + [f"t{L}.attn" for L in range(64)]
rest = [k for k in everyone if k not in union]
rng.shuffle(rest)
arms.append(("other:K8", {k: (.3 if i % 2 else -.3) for i, k in enumerate(rest[:8])}))
arms.append(("other:K29", {k: (.3 if i % 2 else -.3) for i, k in enumerate(rest[:29])}))
singles = sorted(writers, key=lambda x: x[0])
for k, d in singles:
    arms.append((f"one:{k}:{d:+.3f}", {k: writers[(k, d)]}))
print(f"{len(arms)} arms", flush=True)
rows = {}
with open(args.out + ".jsonl", "w") as f:
    for lab, doses in arms:
        t1 = time.time()
        kl, flip, med = run(doses)
        rows[lab] = dict(arm=lab, kl=kl, flip=flip, median=med, doses=doses)
        f.write(json.dumps(rows[lab]) + "\n")
        f.flush()
        print(f"{lab}: KL {kl:.5f} top flip {flip:.3f} median trace {med:.5f} ({time.time() - t1:.0f}s)", flush=True)

lines = ["mean KL(unedited || edited) per position on the model's own finished traces, and the share of positions whose most likely token changes"]
one = {(k, round(d, 4)): rows[f"one:{k}:{d:+.3f}"]["kl"] for k, d in singles}
lines.append("does cost add? joint edit against the sum of its writers alone")
for lab, v in pred.items():
    if "writers" in v:
        s = sum(one[(k, round(d, 4))] for k, d in v["writers"])
        lines.append(f"  {lab}: joint {rows[lab]['kl']:.5f}, sum {s:.5f}, ratio {rows[lab]['kl'] / max(s, 1e-12):.2f}, top flip {rows[lab]['flip']:.3f}")
lines.append(f"  all3: {rows['all3']['kl']:.5f}, top flip {rows['all3']['flip']:.3f}")
lines.append("does cost grow with the square of the dose? KL / dose2")
for w in ("t24.attn", "v8.mlp"):
    lines.append(f"  {w}: " + " ".join(f"{d:+.2f}:{rows[f'dose:{w}:{d:+.2f}']['kl'] / d ** 2:.4f}" for d in (-.6, -.3, -.15, .15, .3, .6)))
lines.append("one writer carrying .6 against four carrying .15 each (the law says about a quarter, if their sensitivities match)")
for part in ("text", "tower"):
    a, b = rows[f"split:{part}:one"]["kl"], rows[f"split:{part}:four"]["kl"]
    lines.append(f"  {part}: one {a:.5f}, four {b:.5f}, ratio {b / max(a, 1e-12):.2f}")
lines.append(f"writers with no signal, same doses: K8 {rows['other:K8']['kl']:.5f}, K29 {rows['other:K29']['kl']:.5f}")
text = "\n".join(lines) + "\n"
print(text, flush=True)
open(args.out + ".txt", "w").write(text)
print("done", flush=True)
