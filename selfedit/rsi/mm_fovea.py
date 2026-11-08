"""Is the model's address error a shift of its rotary lookup? The address exam shows row reads about one row too far on: is that in the phase of the query, and can a weight edit move it?
Text full attention rotates query and key by position, rows on frequencies 1,4,..,31 and columns on 2,5,..,29 of the 32 rotary pairs. Rotating the rows of q_proj that write those pairs by angle omega * delta adds delta to the query's position on that axis, for every token at that layer, as a pure weight edit.
Arms: delta on rows and on columns, at all or some full-attention layers; more weight on the rows and columns pairs (a narrower lookup). Thinking off, one forward pass per item, the same stripes as rsi/mm_address.py.

Usage: python -u rsi/mm_fovea.py --model M --out OUT [--levels 8,12] [--per 40] [--seed 7] [--device cuda:0]; writes OUT.jsonl, OUT.txt
"""
import argparse, json, os, random, time
import numpy as np
import torch
from PIL import Image, ImageDraw

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--levels", default="8,12")
ap.add_argument("--per", type=int, default=40)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--device", default="cuda:0")
args = ap.parse_args()

from transformers import AutoProcessor
from transformers.models.qwen3_5 import modeling_qwen3_5 as M
proc = AutoProcessor.from_pretrained(args.model)
tk = proc.tokenizer
model = M.Qwen3_5ForConditionalGeneration.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa")
model.to(args.device).eval().requires_grad_(False)
layers = model.model.language_model.layers
cfg = model.config.text_config
rp = cfg.rope_parameters
PAIRS = int(cfg.head_dim * rp["partial_rotary_factor"]) // 2
OMEGA = 1.0 / rp["rope_theta"] ** (torch.arange(0, 2 * PAIRS, 2, dtype=torch.float64) / (2 * PAIRS))
ROWS = list(range(1, rp["mrope_section"][1] * 3, 3))
COLS = list(range(2, rp["mrope_section"][2] * 3, 3))
HEADS, HEAD = cfg.num_attention_heads, cfg.head_dim
GROUPS = cfg.num_key_value_heads
COLOURS = {"red": (220, 30, 30), "green": (20, 160, 60), "blue": (30, 60, 220), "yellow": (235, 200, 20), "purple": (140, 40, 170), "orange": (240, 130, 20)}
NAMES = list(COLOURS)
LET = "ABCDEF"
SIZE = 384
MODES = ("row", "col", "both")
FULL = [L for L in range(len(layers)) if L % 4 == 3]


def label(mode, dr, dc):
    return dr % 6 if mode == "row" else dc % 6 if mode == "col" else dr % 3 + 3 * (dc % 2)


def errors(mode, k):
    def wrap(x, n):
        return x - n if x > n // 2 else x
    if mode == "row":
        return wrap(k, 6), None
    if mode == "col":
        return None, wrap(k, 6)
    return wrap(k % 3, 3), wrap(k // 3, 2)


def make(rng, mode, n, path):
    r, c = rng.randrange(n), rng.randrange(n)
    pal = NAMES[:]
    rng.shuffle(pal)
    im = Image.new("RGB", (SIZE, SIZE), (255, 255, 255))
    dr = ImageDraw.Draw(im)
    s = SIZE / n
    for a in range(n):
        for b in range(n):
            dr.rectangle((b * s, a * s, (b + 1) * s - 1, (a + 1) * s - 1), fill=COLOURS[pal[label(mode, a - r, b - c)]])
    im.save(path)
    opts = NAMES[:]
    rng.shuffle(opts)
    q = f"The image is a grid of {n} rows and {n} columns of coloured cells. What colour is the cell in row {r + 1}, column {c + 1}, counting from the top left?\n" + "\n".join(f"({LET[i]}) {x}" for i, x in enumerate(opts)) + "\nPut the letter of the correct option in \\boxed{}."
    return r, c, pal, opts, q


levels = [int(x) for x in args.levels.split(",")]
os.makedirs(args.out + "_img", exist_ok=True)
rng = random.Random(args.seed)
ids = [tk("\\boxed{" + x, add_special_tokens=False).input_ids for x in LET]
cut = next(i for i in range(len(ids[0]) + 1) if i == len(ids[0]) or any(i >= len(y) or y[i] != ids[0][i] for y in ids))
tail = torch.tensor(ids[0][:cut], dtype=torch.long)[None]
letters = torch.tensor([y[cut] for y in ids], device=args.device)
items = []
for n in levels:
    for mode in MODES:
        for k in range(args.per):
            path = os.path.abspath(f"{args.out}_img/{mode}_{n}_{k}.png")
            r, c, pal, opts, q = make(rng, mode, n, path)
            inp = proc.apply_chat_template([{"role": "user", "content": [{"type": "image", "image": path}, {"type": "text", "text": q}]}], tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt", enable_thinking=False)
            feed = {"input_ids": torch.cat([inp["input_ids"], tail], 1), "attention_mask": torch.ones((1, inp["input_ids"].shape[1] + tail.shape[1]), dtype=torch.long)}
            if "mm_token_type_ids" in inp:
                feed["mm_token_type_ids"] = torch.cat([inp["mm_token_type_ids"], torch.zeros_like(tail)], 1)
            feed.update(pixel_values=inp["pixel_values"], image_grid_thw=inp["image_grid_thw"])
            feed = {a: b.to(args.device) for a, b in feed.items()}
            feed["pixel_values"] = feed["pixel_values"].to(torch.bfloat16)
            items.append(dict(n=n, mode=mode, r=r, c=c, pal=pal, opts=opts, feed=feed))
print(f"{len(items)} items", flush=True)


def read():
    rows = []
    for p in items:
        with torch.inference_mode():
            out = model(**p["feed"], use_cache=False)
        lp = out.logits[0, -1].float().log_softmax(-1)[letters]
        prob = (lp - lp.logsumexp(0)).exp().cpu().numpy().astype(np.float64)
        rows.append([prob[p["opts"].index(name)] for name in p["pal"]])
    return np.array(rows)


def fit_axis(hist):
    best = None
    es = np.array(sorted(hist))
    h = np.array([hist[e] for e in es], dtype=np.float64)
    for mu in np.arange(-2, 2.01, .1):
        for sd in np.arange(.1, 4.01, .1):
            w = np.exp(-.5 * ((es - mu) / sd) ** 2)
            w /= w.sum()
            for lapse in (0, .2, .4, .6, .8):
                q = (1 - lapse) * w + lapse / len(es)
                ll = (h * np.log(q + 1e-9)).sum()
                if best is None or ll > best[0]:
                    best = (ll, mu, sd, lapse)
    return best[1:]


def describe(P):
    out, stats = [], {}
    for n in levels:
        for mode in MODES:
            take = [i for i, p in enumerate(items) if p["n"] == n and p["mode"] == mode]
            A = P[take]
            right = float(A[:, 0].mean())
            stats[f"{mode}{n}"] = dict(right=right)
            if mode == "both":
                out.append(f"{mode}{n} {right:.2f}")
                continue
            hist = {}
            for k in range(6):
                e = errors(mode, k)[0 if mode == "row" else 1]
                hist[e] = hist.get(e, 0) + A[:, k].mean()
            mu, sd, lapse = fit_axis(hist)
            stats[f"{mode}{n}"].update(bias=float(mu), sd=float(sd), lapse=float(lapse))
            out.append(f"{mode}{n} {right:.2f} b{mu:+.1f} s{sd:.1f}")
    return "; ".join(out), stats


def edit(arm):
    """Rotates the rotary pairs of q_proj for rows and columns, or weighs them more. Returns what to restore."""
    saved = []
    for L in arm["layers"]:
        sa = layers[L].self_attn
        w = sa.q_proj.weight.data
        saved.append((w, w.clone()))
        for axis, delta in ((ROWS, arm.get("dr", 0.0)), (COLS, arm.get("dc", 0.0))):
            if delta:
                for j in axis:
                    th = float(OMEGA[j]) * delta
                    c, s = np.cos(th), np.sin(th)
                    for a in range(HEADS):
                        i1, i2 = a * 2 * HEAD + j, a * 2 * HEAD + j + PAIRS
                        x1, x2 = w[i1].clone(), w[i2].clone()
                        w[i1] = c * x1 - s * x2
                        w[i2] = c * x2 + s * x1
        if arm.get("q"):
            for j in ROWS + COLS:
                for a in range(HEADS):
                    w[a * 2 * HEAD + j] *= arm["q"]
                    w[a * 2 * HEAD + j + PAIRS] *= arm["q"]
        if arm.get("k"):
            kw = sa.k_proj.weight.data
            saved.append((kw, kw.clone()))
            for j in ROWS + COLS:
                for a in range(GROUPS):
                    kw[a * HEAD + j] *= arm["k"]
                    kw[a * HEAD + j + PAIRS] *= arm["k"]
    return saved


LATE = [L for L in FULL if L >= 39]
MID = [39, 43, 47, 51]
arms = []
for d in (-1.0, -.5, .5, 1.0, -2.0):
    arms.append(dict(label=f"rows{d:+.1f}:all", dr=d, layers=FULL))
for d in (-1.0, -.5, .5, 1.0):
    arms.append(dict(label=f"cols{d:+.1f}:all", dc=d, layers=FULL))
for d in (-1.0, 1.0):
    arms.append(dict(label=f"rows{d:+.1f}:late", dr=d, layers=LATE))
    arms.append(dict(label=f"rows{d:+.1f}:mid", dr=d, layers=MID))
for s in (1.25, 1.5, 2.0):
    arms.append(dict(label=f"q{s}:all", q=s, layers=FULL))
    arms.append(dict(label=f"qk{s}:all", q=s, k=s, layers=FULL))
for s in (.75,):
    arms.append(dict(label=f"q{s}:all", q=s, layers=FULL))
t0 = time.time()
base = read()
line, stats = describe(base)
print(f"base: {line} ({time.time() - t0:.0f}s)", flush=True)
lines = [f"base: {line}"]
with open(args.out + ".jsonl", "w") as f:
    f.write(json.dumps(dict(label="base", stats=stats)) + "\n")
    for arm in arms:
        saved = edit(arm)
        P = read()
        for w, old in saved:
            w.copy_(old)
        line, stats = describe(P)
        print(f"{arm['label']}: {line}", flush=True)
        lines.append(f"{arm['label']}: {line}")
        f.write(json.dumps(dict(label=arm["label"], stats=stats)) + "\n")
        f.flush()
open(args.out + ".txt", "w").write("\n".join(lines) + "\n")
print("done", flush=True)
