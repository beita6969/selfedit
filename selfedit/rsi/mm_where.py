"""Where does a named place get lost? Exam images, thinking off, one forward pass each, three looks at the same model:
1. the read itself on three tasks at the same cell size: cell (named by row and column), mark (shown by an outline, nothing to count), strip (one row, nothing to bind);
2. the image token over a cell: can its colour be decoded there (a linear probe trained on other images, the model unchanged), at the merger output and after text layers; can the token's own row and column;
3. attention from the text after the image back to the image tokens, in each full-attention layer: how much more on the named cell's token than on a random token.
So: is the colour there at all, does the model look at the right token, or both and the answer still fails.

Usage: python -u rsi/mm_where.py --model M --items ITEMS.jsonl --out OUT [--levels 4,8,12,16,24] [--device cuda:0]
"""
import argparse, json, random, re, time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--items", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--levels", default="4,8,12,16,24")
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
MERGE = model.config.vision_config.spatial_merge_size
COLOURS = {"red": (220, 30, 30), "green": (20, 160, 60), "blue": (30, 60, 220), "yellow": (235, 200, 20), "purple": (140, 40, 170), "orange": (240, 130, 20)}
NAMES = list(COLOURS)
LAYERS = [0, 8, 16, 32, 48, 64]                                         # hidden_states index: 0 is the input to the text, so the merger output at image tokens
DIM = 1024
SIZE = 384
PROJ = torch.randn(5120, DIM, generator=torch.Generator().manual_seed(0), dtype=torch.float32).to(args.device) / DIM ** .5

state = {"lang": False, "spy": None, "rows": []}
lang.register_forward_pre_hook(lambda mod, inp: state.update(lang=True))
lang.register_forward_hook(lambda mod, inp, out: state.update(lang=False))
orig = F.scaled_dot_product_attention


def wrapped(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kw):
    if state["lang"] and state["spy"] is not None:
        idx, tgt, ctl = state["spy"]
        with torch.inference_mode():
            first = idx[-1].item() + 1
            keys = k[0].float().repeat_interleave(q.size(1) // k.size(1), 0)                                        # grouped query attention: each key group serves several query heads
            s = (q[0, :, first:].float() @ keys.transpose(-1, -2)) * (scale if scale is not None else q.size(-1) ** -.5)    # heads, queries, keys
            T = k.size(2)
            mask = torch.arange(T, device=s.device)[None] <= torch.arange(first, T, device=s.device)[:, None]    # causal: a query sees itself and what came before
            w = s.masked_fill(~mask[None], float("-inf")).softmax(-1)[:, :, idx]
            lift = w / w.mean(-1, keepdim=True).clamp_min(1e-12)
            heads = lift.mean(0)
            state["rows"].append([heads[-1, tgt].item(), heads[-1, ctl].item(), heads[:, tgt].max().item(), heads[:, ctl].max().item(),
                                  lift[:, :, tgt].max().item(), lift[:, :, ctl].max().item(), float(heads[-1].argmax().item() == tgt)])
    return orig(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kw)


F.scaled_dot_product_attention = wrapped
torch.nn.functional.scaled_dot_product_attention = wrapped


def tok(t):
    return tk(t, add_special_tokens=False).input_ids


def grid_of(path, n):
    """The true colour of every cell, from the pixels."""
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    s = SIZE / n
    out = np.zeros((n, n), dtype=np.int64)
    ref = np.array([COLOURS[c] for c in NAMES], dtype=np.float32)
    for r in range(n):
        for c in range(n):
            out[r, c] = ((ref - a[int((r + .5) * s), int((c + .5) * s)]) ** 2).sum(1).argmin()
    return out


levels = [int(x) for x in args.levels.split(",")]
items = [p for p in map(json.loads, open(args.items)) if p["task"] in ("cell", "mark", "strip") and p["level"] in levels]
print(f"{len(items)} items", flush=True)
t0 = time.time()
reads, cells, attn = {}, {}, {}
rng = random.Random(0)
for w, p in enumerate(items):
    inp = proc.apply_chat_template([{"role": "user", "content": [{"type": "image", "image": p["image_paths"][0]}, {"type": "text", "text": p["prompt"]}]}], tokenize=True, add_generation_prompt=True,
                                   return_dict=True, return_tensors="pt", enable_thinking=False)
    opts = sorted(set(re.findall(r"\(([A-Z])\)", p["prompt"])) | {p["answer"]})
    ids = [tok("\\boxed{" + x) for x in opts]
    n = next(i for i in range(len(ids[0]) + 1) if i == len(ids[0]) or any(i >= len(y) or y[i] != ids[0][i] for y in ids))
    tail = torch.tensor(ids[0][:n], dtype=torch.long)[None]
    feed = {"input_ids": torch.cat([inp["input_ids"], tail], 1), "attention_mask": torch.ones((1, inp["input_ids"].shape[1] + tail.shape[1]), dtype=torch.long)}
    if "mm_token_type_ids" in inp:
        feed["mm_token_type_ids"] = torch.cat([inp["mm_token_type_ids"], torch.zeros_like(tail)], 1)
    feed.update(pixel_values=inp["pixel_values"], image_grid_thw=inp["image_grid_thw"])
    feed = {k: v.to(args.device) for k, v in feed.items()}
    feed["pixel_values"] = feed["pixel_values"].to(torch.bfloat16)
    h, wd = [int(x) // MERGE for x in feed["image_grid_thw"][0].tolist()[1:]]
    idx = torch.nonzero(feed["input_ids"][0] == IMG).flatten()
    L = p["level"]
    tgt = None
    if p["task"] == "cell":
        r, c = [int(x) - 1 for x in re.search(r"row (\d+), column (\d+)", p["prompt"]).groups()]
        tgt = int((r + .5) / L * h) * wd + int((c + .5) / L * wd)
        ctl = rng.randrange(idx.numel() - 1)
        ctl += ctl >= tgt
        state.update(spy=(idx, tgt, ctl), rows=[])
    with torch.inference_mode():
        out = model(**feed, use_cache=False, output_hidden_states=p["task"] == "cell")
    state["spy"] = None
    lp = out.logits[0, -1].float().log_softmax(-1)[torch.tensor([y[n] for y in ids], device=args.device)].cpu().numpy().astype(np.float64)
    g = opts.index(p["answer"])
    reads.setdefault((p["task"], L), []).append((lp[g] > np.delete(lp, g).max(), lp[g] - np.logaddexp.reduce(np.delete(lp, g))))
    if p["task"] == "cell":
        attn.setdefault(L, []).append(state["rows"])
        colours = grid_of(p["image_paths"][0], L)
        tokens = [(int((a + .5) / L * h) * wd + int((b + .5) / L * wd), a, b) for a in range(L) for b in range(L)]
        where = torch.tensor([t[0] for t in tokens], device=args.device)
        row = cells.setdefault(L, {"x": {l: [] for l in LAYERS}, "y": [], "img": [], "pos": []})
        for l in LAYERS:
            row["x"][l].append((out.hidden_states[l][0, idx[where]].float() @ PROJ).cpu())
        row["y"].append(colours.ravel())
        row["img"].append(np.full(L * L, w))
        row["pos"].append(np.array([(a / max(1, L - 1), b / max(1, L - 1)) for a in range(L) for b in range(L)], dtype=np.float32))
    if (w + 1) % 100 == 0:
        print(f"{w + 1} items, {time.time() - t0:.0f}s", flush=True)


def ridge(xa, ya, xb):
    """Linear read-out with its penalty chosen on the last fifth of the training set."""
    xa, xb = xa.to(args.device), xb.to(args.device)
    mu, sd = xa.mean(0), xa.std(0).clamp_min(1e-6)
    xa, xb = (xa - mu) / sd, (xb - mu) / sd
    my = ya.mean(0).to(args.device)                                     # without the intercept the position read came out below zero whatever the token held
    ya = ya.to(args.device) - my
    cut = int(xa.size(0) * .8)

    def fit(x, y, lam):
        return torch.linalg.solve((x.T @ x).double() + lam * x.size(0) * torch.eye(x.size(1), device=x.device, dtype=torch.float64), (x.T @ y).double()).float()
    best = min((1e-3, 1e-2, 1e-1, 1, 10, 100), key=lambda lam: ((xa[cut:] @ fit(xa[:cut], ya[:cut], lam) - ya[cut:]) ** 2).mean().item())
    return xb @ fit(xa, ya, best) + my


lines = []
lines.append("read, right (margin):")
for task in ("cell", "mark", "strip"):
    lines.append(f"  {task}: " + "  ".join(f"{L}: {np.mean([x[0] for x in reads[(task, L)]]):.2f} ({np.mean([x[1] for x in reads[(task, L)]]):+.1f})" for L in levels if (task, L) in reads))
lines.append("decode the cell's colour from the image token over it, on images not trained on (chance .17); row and column R2 of that token:")
for L in levels:
    if L not in cells or L * L < 64:
        continue
    d = cells[L]
    y = torch.tensor(np.concatenate(d["y"]))
    img = np.concatenate(d["img"])
    pos = torch.tensor(np.concatenate(d["pos"]))
    ims = np.unique(img)
    tr = torch.tensor(np.isin(img, ims[: len(ims) // 2]))
    one = F.one_hot(y, 6).float()
    out = []
    for l in LAYERS:
        x = torch.cat(d["x"][l]).float()
        pred = ridge(x[tr], one[tr], x[~tr])
        acc = (pred.argmax(1).cpu() == y[~tr]).float().mean().item()
        pp = ridge(x[tr], pos[tr], x[~tr]).cpu()
        r2 = 1 - ((pp - pos[~tr]) ** 2).sum().item() / ((pos[~tr] - pos[~tr].mean(0)) ** 2).sum().item()
        out.append(f"L{l} {acc:.2f}/{r2:.2f}")
    lines.append(f"  grid {L}: " + " ".join(out))
lines.append("attention lift on the named cell's token against a random token, per full-attention layer (1 = uniform): last position; best text position over heads' mean; best position and head; top token right at the last position")
for L in levels:
    if L not in attn:
        continue
    a = np.array(attn[L])                                               # items, layers, 7
    m = a.mean(0)
    lay = range(m.shape[0])
    show = [i for i in lay if i % 3 == 0 or i == m.shape[0] - 1]
    lines.append(f"  grid {L}: " + " ".join(f"#{i} {m[i, 0]:.1f}/{m[i, 1]:.1f} {m[i, 2]:.1f}/{m[i, 3]:.1f} {m[i, 4]:.1f}/{m[i, 5]:.1f} {m[i, 6]:.2f}" for i in show))
text = "\n".join(lines) + "\n"
print(text, flush=True)
open(args.out + ".txt", "w").write(text)
np.savez(args.out + ".npz", **{f"attn_{L}": np.array(v) for L, v in attn.items()})
print("done", flush=True)
