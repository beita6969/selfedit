"""Directions from the model's own vision traces, read with the image in the forward pass. Last token of a line:
bad, better   the lines the model itself pointed at when it read its own failures (self_diagnose marks): first astray in a failed trace, what it did instead in a right one
guess         a line reasoning about the test instead of the image ("typical answer", "intended answer", "dataset")
giveup        a line saying the image cannot be read ("too detailed", "by eye", "error-prone")
look          a line describing the image
first_right, first_wrong   the first lines of a right and a failed trace on the same problem, where the image is first read
right_end, wrong_end       natural ends of finished right and wrong traces
image, text   image tokens and text tokens of the prompt
Contrasts saved as <out>/<name>.npz, same format as the text library (<kind>/dir per layer, kinds mlp and resid). Reliability: cosine between directions from two halves of the problems; separation: AUC across the halves.

Usage: python -u rsi/mm_dirs.py --model M --traces CALIB.jsonl --pool POOL --effort medium --out LIB [--marks RUN/marks.jsonl] [--max-traces 1200] [--device cuda:0]
"""
import argparse, json, os, re, sys, time, zlib
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--traces", required=True)
ap.add_argument("--pool", required=True)
ap.add_argument("--effort", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--marks", default=None)
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--max-traces", type=int, default=1200)
ap.add_argument("--per-kind", type=int, default=3)
ap.add_argument("--max-tok", type=int, default=6000)
ap.add_argument("--first-lines", type=int, default=12)
args = ap.parse_args()

from transformers import AutoProcessor
from transformers.models.qwen3_5 import modeling_qwen3_5 as M
proc = AutoProcessor.from_pretrained(args.model)
tk = proc.tokenizer
model = M.Qwen3_5ForConditionalGeneration.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa")
model.to(args.device).eval().requires_grad_(False)
layers = model.model.language_model.layers
L, H = len(layers), model.config.text_config.hidden_size
IMG = model.config.image_token_id
print(f"{L} layers, hidden {H}", flush=True)

GUESS = re.compile(r"intended (answer|pattern)|expected answer|test answer|common answer|typical (answer|question|test|vqa|pattern|puzzle)|dataset|benchmark|answer key|known (answer|puzzle)|I'?ll (just )?(guess|go with)|let me guess|best guess|maybe the puzzle|just (guess|pick)", re.I)
GIVEUP = re.compile(r"too (detailed|difficult|hard|complex|uncertain|fine|small|error)|error-prone|by eye|unable to|can'?t (tell|see|resolve|make out|read|determine)|cannot (tell|see|resolve|make out|read|determine)|hard to (see|tell|read|trace)|difficult to (see|tell|read|trace)|impossible to", re.I)
LOOK = re.compile(r"\b(looking at|i see|i can see|the image|image \d|in the (picture|figure|photo)|at the (top|bottom|left|right)|on the (left|right)|appears? to|shows?)\b", re.I)

store, want = {"mlp": {}, "resid": {}}, {}


def keep(kind, l):
    def hook(mod, inp, out):
        x = (out[0] if isinstance(out, tuple) else out)[0]
        store[kind][l] = x[want["idx"]].float().cpu()
    return hook


for l, layer in enumerate(layers):
    layer.mlp.register_forward_hook(keep("mlp", l))
    layer.register_forward_hook(keep("resid", l))

pool = {p["id"]: p for p in map(json.loads, open(args.pool))}
recs = [json.loads(x) for x in open(args.traces)]
marked_bad, marked_good = {}, {}
if args.marks:
    files = {}
    for m in map(json.loads, open(args.marks)):
        if m.get("bad_line") is not None:
            marked_bad.setdefault((m["id"], m["fail_line"]), []).append(m["bad_line"])
        if m.get("better_line") is not None and m.get("good_line") is not None:
            marked_good.setdefault((m["id"], m["good_line"]), []).append(m["better_line"])
by = {}
for i, r in enumerate(recs):
    r["line_no"] = i
    if r.get("reasoning") and r["id"] in pool:
        by.setdefault(r["id"], []).append(r)
order = []
for r in recs:                                  # traces the model pointed into come first, then problems with both a right and a failed trace, then the rest
    if (r["id"], r["line_no"]) in marked_bad or (r["id"], r["line_no"]) in marked_good:
        order.append(r)
seen = {id(r) for r in order}
paired = [k for k in sorted(by) if any(r.get("ok") for r in by[k]) and any(not r.get("ok") for r in by[k])]
rng = np.random.default_rng(0)
rng.shuffle(paired)
for k in paired:
    for r in by[k]:
        if id(r) not in seen:
            seen.add(id(r))
            order.append(r)
rest = [k for k in sorted(by) if k not in set(paired)]
rng.shuffle(rest)
for k in rest:
    for r in by[k]:
        if id(r) not in seen:
            seen.add(id(r))
            order.append(r)
order = [r for r in order if r.get("reasoning")][: args.max_traces]
print(f"{len(order)} traces, {sum((r['id'], r['line_no']) in marked_bad for r in order)} with a marked bad line, {sum((r['id'], r['line_no']) in marked_good for r in order)} with a marked better line", flush=True)

KINDS = ("bad", "better", "guess", "giveup", "look", "first_right", "first_wrong", "right_end", "wrong_end", "image", "text")
vecs = {k: {"mlp": [], "resid": []} for k in KINDS}
owner = {k: [] for k in KINDS}


def spread(xs, n):
    xs = sorted(xs)
    if len(xs) <= n:
        return xs
    return [xs[int(i * (len(xs) - 1) / (n - 1))] for i in range(n)] if n > 1 else [xs[-1]]


def content(p):
    imgs = [{"type": "image", "image": q} for q in p["image_paths"]]
    parts = p["prompt"].split("<image>")
    if len(parts) - 1 == len(imgs) and imgs:
        c = []
        for k, txt in enumerate(parts):
            if txt.strip():
                c.append({"type": "text", "text": txt.strip()})
            if k < len(imgs):
                c.append(imgs[k])
        return c
    return imgs + [{"type": "text", "text": p["prompt"]}]


t0, used = time.time(), 0
for r in order:
    text = r["reasoning"]
    enc = tk(text, add_special_tokens=False, return_offsets_mapping=True)
    th, off = enc.input_ids[: args.max_tok], enc.offset_mapping[: args.max_tok]
    if len(th) < 8:
        continue
    lines, ends, pos_char = text.split("\n"), [], 0
    k = 0
    for ln in lines:                            # token holding the line's last character, or None past the cut
        pos_char += len(ln)
        while k < len(off) and off[k][1] < pos_char:
            k += 1
        ends.append(k if k < len(off) and ln.strip() else None)
        pos_char += 1
    live = [j for j, e in enumerate(ends) if e is not None and e >= 1]
    if not live:
        continue
    finished = r["finish"] != "length"
    pos = {k: [] for k in KINDS}
    pos["bad"] = [ends[j] for j in marked_bad.get((r["id"], r["line_no"]), []) if j < len(ends) and ends[j] is not None]
    pos["better"] = [ends[j] for j in marked_good.get((r["id"], r["line_no"]), []) if j < len(ends) and ends[j] is not None]
    pos["guess"] = spread([ends[j] for j in live if GUESS.search(lines[j])], args.per_kind)
    pos["giveup"] = spread([ends[j] for j in live if GIVEUP.search(lines[j])], args.per_kind)
    pos["look"] = spread([ends[j] for j in live if LOOK.search(lines[j]) and not GUESS.search(lines[j]) and not GIVEUP.search(lines[j])], args.per_kind)
    early = spread([ends[j] for j in live[: args.first_lines]], args.per_kind)
    both = any(x.get("ok") for x in by[r["id"]]) and any(not x.get("ok") for x in by[r["id"]])
    if both:
        pos["first_right" if r.get("ok") else "first_wrong"] = early
    if finished and len(enc.input_ids) <= args.max_tok:
        pos["right_end" if r.get("ok") else "wrong_end"] = [ends[live[-1]]]
    inp = proc.apply_chat_template([{"role": "user", "content": content(pool[r["id"]])}], tokenize=True, add_generation_prompt=True, return_dict=True,
                                   return_tensors="pt", reasoning_effort=args.effort, enable_thinking=True)
    head = inp["input_ids"][0].tolist()
    n = len(head)
    last = max([e for v in pos.values() for e in v] + [0]) + 1
    tail = torch.tensor(th[: last + 1], dtype=torch.long)[None]
    feed = {"input_ids": torch.cat([inp["input_ids"], tail], 1), "attention_mask": torch.ones((1, n + tail.shape[1]), dtype=torch.long)}
    if "mm_token_type_ids" in inp:
        feed["mm_token_type_ids"] = torch.cat([inp["mm_token_type_ids"], torch.zeros_like(tail)], 1)
    for key in ("pixel_values", "image_grid_thw"):
        if key in inp:
            feed[key] = inp[key]
    feed = {key: v.to(args.device) for key, v in feed.items()}
    if "pixel_values" in feed:
        feed["pixel_values"] = feed["pixel_values"].to(torch.bfloat16)
    img_at = [i for i, t in enumerate(head) if t == IMG]
    txt_at = [i for i, t in enumerate(head) if t != IMG and i > 3]
    if used % 4 == 0 and img_at:                # image and text tokens of the prompt, from every fourth trace: their mean over a sample of positions
        pos["image"] = [("h", i) for i in spread(img_at, 16)]
        pos["text"] = [("h", i) for i in spread(txt_at, 16)]
    ats = sorted({(e[1] if isinstance(e, tuple) else n + e) for v in pos.values() for e in v})
    if not ats:
        continue
    want["idx"] = torch.tensor(ats, dtype=torch.long, device=args.device)
    with torch.inference_mode():
        model.model(**feed, use_cache=False)
    rows = {space: torch.stack([store[space][l] for l in range(L)], 1).numpy() for space in store}
    half = zlib.crc32(r["id"].encode()) % 2
    for kind, ks in pos.items():
        for e in ks:
            at = ats.index(e[1] if isinstance(e, tuple) else n + e)
            for space in rows:
                vecs[kind][space].append(rows[space][at].astype(np.float16))
            owner[kind].append(half)
    used += 1
    if used % 25 == 0:
        print(f"{used} traces, {feed['input_ids'].shape[1]} tokens last, {time.time() - t0:.0f}s, " + ", ".join(f"{k} {len(v)}" for k, v in owner.items()), flush=True)
print(f"{used} traces in {time.time() - t0:.0f}s: " + ", ".join(f"{k} {len(v)}" for k, v in owner.items()), flush=True)


def mean(kind, space, half=None):
    x = np.stack(vecs[kind][space]).astype(np.float32)
    own = np.array(owner[kind])
    return x[own == half].mean(0) if half is not None else x.mean(0)


def auc(p, q):
    r = np.argsort(np.argsort(np.r_[p, q])) + 1
    return (r[: len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(q))


CONTRASTS = {"better_vs_bad": ("better", "bad"), "guess_vs_look": ("guess", "look"), "giveup_vs_look": ("giveup", "look"), "first_right_vs_wrong": ("first_right", "first_wrong"),
             "right_vs_wrong_end": ("right_end", "wrong_end"), "image_vs_text": ("image", "text")}
os.makedirs(args.out, exist_ok=True)
report = {}
for name, (A, B) in CONTRASTS.items():
    if min(len(owner[A]), len(owner[B])) < 16 or min(np.bincount(owner[A] + owner[B], minlength=2)) < 8:
        print(f"{name}: too few ({len(owner[A])}, {len(owner[B])})", flush=True)
        continue
    out = {}
    for space in ("mlp", "resid"):
        d = mean(A, space) - mean(B, space)
        out[f"{space}/dir"] = (d / np.maximum(1e-9, np.linalg.norm(d, axis=1, keepdims=True))).astype(np.float32)
        out[f"{space}/norm"] = np.linalg.norm(d, axis=1).astype(np.float32)
    d0, d1 = mean(A, "mlp", 0) - mean(B, "mlp", 0), mean(A, "mlp", 1) - mean(B, "mlp", 1)
    rel = (d0 * d1).sum(1) / np.maximum(1e-9, np.linalg.norm(d0, axis=1) * np.linalg.norm(d1, axis=1))
    xa, xb = np.stack(vecs[A]["mlp"]).astype(np.float32), np.stack(vecs[B]["mlp"]).astype(np.float32)
    oa, ob = np.array(owner[A]), np.array(owner[B])
    sep = np.zeros(L)
    for l in range(L):
        s = []
        for dd, pick in ((d0, 1), (d1, 0)):
            u = dd[l] / max(1e-9, np.linalg.norm(dd[l]))
            s.append(auc(xa[oa == pick, l] @ u, xb[ob == pick, l] @ u))
        sep[l] = np.mean(s)
    out["mlp/half_cos"], out["mlp/auc"] = rel.astype(np.float32), sep.astype(np.float32)
    np.savez(os.path.join(args.out, f"{name}.npz"), **out)
    top = sorted(range(L), key=lambda l: -sep[l])[:10]
    report[name] = {"n": [len(owner[A]), len(owner[B])], "top": [dict(layer=l, auc=round(float(sep[l]), 3), half_cos=round(float(rel[l]), 3)) for l in top]}
    print(name, f"({len(owner[A])} vs {len(owner[B])}) top AUC " + ", ".join(f"L{l} {sep[l]:.2f} (cos {rel[l]:+.2f})" for l in top), flush=True)
json.dump(report, open(os.path.join(args.out, "mm_dirs_report.json"), "w"), indent=1)
print("done", flush=True)
