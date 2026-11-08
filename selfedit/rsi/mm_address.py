"""How far off is the place the model fetches, and which link of the fetch is weak? Thinking off, one forward pass per item.
The colour of a cell labels how far it lies from the named cell, so the colour read gives the signed error of the address, not only right or wrong. Colours repeat with the distance, so no cell stands out in the image:
row: all cells of a row share a colour, period 6 over rows; error in rows from -2 to +3.
col: the same over columns.
both: period 3 over rows and 2 over columns, the two errors together.
Then the two sides of the fetch, by linear probes on items not trained on: does the state after the question hold the address (row and column), and does it hold it at the position where the answer is read.

Usage: python -u rsi/mm_address.py --model M --out OUT [--levels 8,12,16,24] [--per 60] [--seed 7] [--device cuda:0]; writes OUT.jsonl, OUT.txt
"""
import argparse, json, os, random, re, time
import numpy as np
import torch
from PIL import Image, ImageDraw

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--levels", default="8,12,16,24")
ap.add_argument("--per", type=int, default=60)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--doses", default=None, help="JSON of writer name to dose, read against the edited model")
args = ap.parse_args()

from transformers import AutoProcessor
from transformers.models.qwen3_5 import modeling_qwen3_5 as M
proc = AutoProcessor.from_pretrained(args.model)
tk = proc.tokenizer
model = M.Qwen3_5ForConditionalGeneration.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa")
model.to(args.device).eval().requires_grad_(False)
if args.doses:
    for name, d in json.load(open(args.doses)).items():
        b = re.match(r"v(\d+)\.(attn|mlp)", name)
        if b:
            block = model.model.visual.blocks[int(b[1])]
            mod = block.attn.proj if b[2] == "attn" else block.mlp.linear_fc2
        else:
            L = int(re.match(r"t(\d+)", name)[1])
            layer = model.model.language_model.layers[L]
            mod = layer.self_attn.o_proj if L % 4 == 3 else layer.linear_attn.out_proj
        for par in mod.parameters():
            par.data.mul_(1 + d)
COLOURS ={"red": (220, 30, 30), "green": (20, 160, 60), "blue": (30, 60, 220), "yellow": (235, 200, 20), "purple": (140, 40, 170), "orange": (240, 130, 20)}
NAMES = list(COLOURS)
LET = "ABCDEF"
SIZE = 384
MODES = ("row", "col", "both")
LAYERS = [0, 8, 16, 32, 48, 64]
DIM = 1024
PROJ = torch.randn(5120, DIM, generator=torch.Generator().manual_seed(0), dtype=torch.float32).to(args.device) / DIM ** .5


def label(mode, dr, dc):
    """Which palette index a cell at this offset from the target gets."""
    return dr % 6 if mode == "row" else dc % 6 if mode == "col" else dr % 3 + 3 * (dc % 2)


def errors(mode, k):
    """Back from palette index to the offset of the place read: (rows, cols), None for the axis this mode does not label."""
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
items = []
for n in levels:
    for mode in MODES:
        for k in range(args.per):
            path = os.path.abspath(f"{args.out}_img/{mode}_{n}_{k}.png")
            r, c, pal, opts, q = make(rng, mode, n, path)
            items.append(dict(n=n, mode=mode, r=r, c=c, pal=pal, opts=opts, prompt=q, path=path))
print(f"{len(items)} items", flush=True)
ids = [tk("\\boxed{" + x, add_special_tokens=False).input_ids for x in LET]
cut = next(i for i in range(len(ids[0]) + 1) if i == len(ids[0]) or any(i >= len(y) or y[i] != ids[0][i] for y in ids))
tail = torch.tensor(ids[0][:cut], dtype=torch.long)[None]
letters = torch.tensor([y[cut] for y in ids], device=args.device)
counting = tk(" counting", add_special_tokens=False).input_ids[0]
states = {"after": {l: [] for l in LAYERS}, "last": {l: [] for l in LAYERS}}
t0 = time.time()
rows = []
with open(args.out + ".jsonl", "w") as f:
    for w, p in enumerate(items):
        inp = proc.apply_chat_template([{"role": "user", "content": [{"type": "image", "image": p["path"]}, {"type": "text", "text": p["prompt"]}]}], tokenize=True, add_generation_prompt=True,
                                       return_dict=True, return_tensors="pt", enable_thinking=False)
        feed = {"input_ids": torch.cat([inp["input_ids"], tail], 1), "attention_mask": torch.ones((1, inp["input_ids"].shape[1] + tail.shape[1]), dtype=torch.long)}
        if "mm_token_type_ids" in inp:
            feed["mm_token_type_ids"] = torch.cat([inp["mm_token_type_ids"], torch.zeros_like(tail)], 1)
        feed.update(pixel_values=inp["pixel_values"], image_grid_thw=inp["image_grid_thw"])
        feed = {k: v.to(args.device) for k, v in feed.items()}
        feed["pixel_values"] = feed["pixel_values"].to(torch.bfloat16)
        with torch.inference_mode():
            out = model(**feed, use_cache=False, output_hidden_states=True)
        lp = out.logits[0, -1].float().log_softmax(-1)[letters]
        prob = (lp - lp.logsumexp(0)).exp().cpu().numpy().astype(np.float64)
        # letter order is the option order, shown in random order: back to palette index, so 0 is the right colour and the rest is the error
        pal = np.array([prob[p["opts"].index(name)] for name in p["pal"]])
        pos = (feed["input_ids"][0] == counting).nonzero().flatten()
        for l in LAYERS:
            hs = out.hidden_states[l][0]
            states["last"][l].append((hs[-1].float() @ PROJ).cpu())
            states["after"][l].append((hs[pos[-1].item()].float() @ PROJ).cpu())
        row = {k: p[k] for k in ("n", "mode", "r", "c")}
        row["p"] = pal.tolist()
        rows.append(row)
        f.write(json.dumps(row) + "\n")
        if (w + 1) % 100 == 0:
            print(f"{w + 1} items, {time.time() - t0:.0f}s", flush=True)


def ridge(xa, ya, xb):
    """Linear read-out with intercept, its penalty chosen on the last fifth of the training set."""
    xa, xb, ya = xa.to(args.device), xb.to(args.device), ya.to(args.device)
    mu, sd, my = xa.mean(0), xa.std(0).clamp_min(1e-6), ya.mean(0)
    xa, xb, ya = (xa - mu) / sd, (xb - mu) / sd, ya - my
    n = int(xa.size(0) * .8)

    def fit(x, y, lam):
        return torch.linalg.solve((x.T @ x).double() + lam * x.size(0) * torch.eye(x.size(1), device=x.device, dtype=torch.float64), (x.T @ y).double()).float()
    best = min((1e-1, 1, 10, 100, 1000), key=lambda lam: ((xa[n:] @ fit(xa[:n], ya[:n], lam) - ya[n:]) ** 2).mean().item())
    return (xb @ fit(xa, ya, best) + my).cpu()


def fit_axis(hist):
    """Discrete normal over the error with bias, spread and a lapse to the flat guess, by likelihood on a coarse grid."""
    best = None
    es = np.array(sorted(hist))
    h = np.array([hist[e] for e in es], dtype=np.float64)
    for mu in np.arange(-2, 2.01, .1):
        for sd in np.arange(.1, 6.01, .1):
            w = np.exp(-.5 * ((es - mu) / sd) ** 2)
            w /= w.sum()
            for lapse in (0, .1, .2, .3, .4, .5, .6, .7, .8):
                q = (1 - lapse) * w + lapse / len(es)
                ll = (h * np.log(q + 1e-9)).sum()
                if best is None or ll > best[0]:
                    best = (ll, mu, sd, lapse)
    return best[1:]


lines = []
lines.append("read by mode and grid: right (softmax mean of the right colour, chance .17 for row and col, .17 for both); error shares in cells; fit bias, spread in cells, spread as share of the image side, lapse")
for n in levels:
    for mode in MODES:
        sub = [x for x in rows if x["n"] == n and x["mode"] == mode]
        if not sub:
            continue
        P = np.array([x["p"] for x in sub])
        right = np.mean(P.argmax(1) == 0)
        parts = []
        for ax, idx in (("row", 0), ("col", 1)):
            hist = {}
            for k in range(6):
                e = errors(mode, k)[idx]
                if e is not None:
                    hist[e] = hist.get(e, 0) + P[:, k].mean()
            if hist:
                mu, sd, lapse = fit_axis(hist)
                parts.append(f"{ax} " + " ".join(f"{e:+d}:{hist[e]:.2f}" for e in sorted(hist)) + f" | bias {mu:+.1f} sd {sd:.1f} ({sd / n:.3f}) lapse {lapse:.1f}")
        lines.append(f"  grid {n} {mode}: right {right:.2f} ({P[:, 0].mean():.2f}); " + "; ".join(parts))
lines.append("by place of the named cell, row and col modes together: mean right colour share by distance from the nearest edge of that axis, and signed mean error per third of the axis (a compressed count reads short late in the axis)")
for n in levels:
    sub = [(x["r"] if x["mode"] == "row" else x["c"], x["p"], x["mode"]) for x in rows if x["n"] == n and x["mode"] in ("row", "col")]
    d = {}
    th = {}
    for at, p, mode in sub:
        d.setdefault(min(min(at, n - 1 - at), 5), []).append(p[0])
        th.setdefault(3 * at // n, []).append(sum(errors(mode, k)[0 if mode == "row" else 1] * p[k] for k in range(6)))
    lines.append(f"  grid {n}: edge distance " + " ".join(f"{k}:{np.mean(v):.2f}" for k, v in sorted(d.items())) + "; signed error by third " + " ".join(f"{k}:{np.mean(v):+.2f}" for k, v in sorted(th.items())))
lines.append("can the state hold the address? R2 of row and of column, read linearly on half of the items not trained on, per text layer: at the first token after the address; at the position where the answer is read")
y = torch.tensor([[x["r"] + 1, x["c"] + 1] for x in rows], dtype=torch.float32)
order = np.random.default_rng(0).permutation(len(rows))
tr, te = torch.tensor(order[: len(order) // 2]), torch.tensor(order[len(order) // 2:])
for where in ("after", "last"):
    out = []
    for l in LAYERS:
        x = torch.stack(states[where][l]).float()
        pred = ridge(x[tr], y[tr], x[te])
        r2 = 1 - ((pred - y[te]) ** 2).sum(0) / ((y[te] - y[te].mean(0)) ** 2).sum(0)
        out.append(f"L{l} {r2[0].item():.2f}/{r2[1].item():.2f}")
    lines.append(f"  {where}: " + " ".join(out))
text = "\n".join(lines) + "\n"
print(text, flush=True)
open(args.out + ".txt", "w").write(text)
print("done", flush=True)
