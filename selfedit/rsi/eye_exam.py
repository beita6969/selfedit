"""An eye exam for a vision-language model: drawn images with known answers, graded in difficulty, thinking off, one forward pass each. It finds what limits the model's sight without any benchmark:
acuity (ring gap, small square, line gap: nothing to locate), locating (colour of a named grid cell, the odd glyph), counting (dots), following (which end a line reaches). Each task has levels of pixel size,
so the result is accuracy against pixels, and against pixels per text token for that checkpoint. Thresholds per task say which part to change.

Usage: python rsi/eye_exam.py build DIR [--per-level 40] [--seed 0] [--only cell:8,12/trace:3,4]
       python -u rsi/eye_exam.py run DIR --label NAME [--tasks cell,trace] [--edits JSON] [--think BUDGET] [loop.py args, with --mm]     writes DIR/NAME.jsonl (one row per item) and adds to DIR/summary.jsonl
       JSON: [{"label": .., "engine": [engine edits]}]; each edit is read against the same checkpoint, with the change of margin per task and its 90 % interval.
       --think generates with thinking at the given budget instead of reading, and grades the last \\boxed letter: can reasoning rescue what the read fails?
       python rsi/eye_exam.py report DIR
"""
import json, math, os, random, re, sys

COLOURS = {"red": (220, 30, 30), "green": (20, 160, 60), "blue": (30, 60, 220), "yellow": (235, 200, 20), "purple": (140, 40, 170), "orange": (240, 130, 20)}
LET = "ABCD"
ENDS = "ABCDEFGH"
SIZE = 384


def canvas(mode="RGB", w=SIZE):
    from PIL import Image
    return Image.new(mode, (w, w), (255, 255, 255))


def options(right, wrong, rng):
    o = [right] + rng.sample(wrong, 3)
    rng.shuffle(o)
    return o, LET[o.index(right)]


def ask(q, opts):
    return q + "\n" + "\n".join(f"({LET[i]}) {x}" for i, x in enumerate(opts)) + "\nPut the letter of the correct option in \\boxed{}."


def ring(rng, d):
    """Acuity: a ring with a gap at a random place, nothing to search for but the ring."""
    from PIL import ImageDraw
    im = canvas()
    dr = ImageDraw.Draw(im)
    x, y = rng.randint(d, SIZE - 2 * d), rng.randint(d, SIZE - 2 * d)
    w = max(1, round(d / 5))
    dr.ellipse((x, y, x + d, y + d), outline=(0, 0, 0), width=w)
    way = rng.choice(["up", "down", "left", "right"])
    cx, cy = x + d / 2, y + d / 2
    box = {"up": (cx - w / 2, y - 1, cx + w / 2, y + w + 1), "down": (cx - w / 2, y + d - w - 1, cx + w / 2, y + d + 1),
           "left": (x - 1, cy - w / 2, x + w + 1, cy + w / 2), "right": (x + d - w - 1, cy - w / 2, x + d + 1, cy + w / 2)}[way]
    dr.rectangle(box, fill=(255, 255, 255))
    opts = ["up", "down", "left", "right"]
    return im, ask("The image shows one black ring with a small gap in it. Which way does the gap face?", opts), LET[opts.index(way)]


def square(rng, s):
    """Acuity for colour: one small square on white."""
    from PIL import ImageDraw
    im = canvas()
    c = rng.choice(list(COLOURS))
    x, y = rng.randint(8, SIZE - s - 8), rng.randint(8, SIZE - s - 8)
    ImageDraw.Draw(im).rectangle((x, y, x + s - 1, y + s - 1), fill=COLOURS[c])
    opts, ans = options(c, [k for k in COLOURS if k != c], rng)
    return im, ask("The image is white with one small coloured square. What colour is the square?", opts), ans


def gap(rng, g):
    """Acuity for a break: one straight line, whole or broken by a gap of g pixels."""
    from PIL import ImageDraw
    im = canvas()
    dr = ImageDraw.Draw(im)
    a, b = rng.randint(40, SIZE - 200), rng.randint(40, SIZE - 40)
    L = rng.randint(120, 160)
    broken = rng.random() < .5
    dr.line((a, b, a + L, b), fill=(0, 0, 0), width=2)
    if broken:
        m = a + rng.randint(30, L - 30)
        dr.rectangle((m, b - 3, m + g - 1, b + 3), fill=(255, 255, 255))
    opts = ["yes", "no", "cannot tell", "there are two images"]
    return im, ask("The image shows one horizontal black line. Is the line broken anywhere by a gap?", opts), LET[opts.index("yes" if broken else "no")]


def cell(rng, n):
    """Locating: the colour of a named cell in an n x n grid."""
    from PIL import ImageDraw
    im = canvas()
    dr = ImageDraw.Draw(im)
    s = SIZE / n
    names = list(COLOURS)
    grid = [[rng.choice(names) for _ in range(n)] for _ in range(n)]
    for r in range(n):
        for c in range(n):
            dr.rectangle((c * s, r * s, (c + 1) * s - 1, (r + 1) * s - 1), fill=COLOURS[grid[r][c]])
    r, c = rng.randrange(n), rng.randrange(n)
    opts, ans = options(grid[r][c], [k for k in names if k != grid[r][c]], rng)
    return im, ask(f"The image is a grid of {n} rows and {n} columns of coloured cells. What colour is the cell in row {r + 1}, column {c + 1}, counting from the top left?", opts), ans


def odd(rng, n):
    """Search and locating: one glyph differs in an n x n grid."""
    from PIL import ImageDraw
    im = canvas()
    dr = ImageDraw.Draw(im)
    s = SIZE / n
    w = max(1, round(s / 8))
    R, C = rng.randrange(n), rng.randrange(n)
    for r in range(n):
        for c in range(n):
            x0, y0, x1, y1 = c * s + s * .2, r * s + s * .2, c * s + s * .8, r * s + s * .8
            dr.ellipse((x0, y0, x1, y1), outline=(0, 0, 0), width=w)
            if (r, c) == (R, C):
                dr.line(((x0 + x1) / 2, (y0 + y1) / 2, x1, y1), fill=(0, 0, 0), width=w)
    right = f"row {R + 1}, column {C + 1}"
    wrong = set()
    while len(wrong) < 3:
        o = f"row {rng.randint(1, n)}, column {rng.randint(1, n)}"
        if o != right:
            wrong.add(o)
    opts, ans = options(right, sorted(wrong), rng)
    return im, ask(f"The image is a grid of {n} rows and {n} columns of circles. One circle has a short tail like the letter Q. Where is it, counting from the top left?", opts), ans


def dots(rng, r):
    """Counting: k separated dots of radius r."""
    from PIL import ImageDraw
    im = canvas()
    dr = ImageDraw.Draw(im)
    k = rng.randint(2, 9)
    pts = []
    while len(pts) < k:
        p = (rng.randint(2 * r + 4, SIZE - 2 * r - 4), rng.randint(2 * r + 4, SIZE - 2 * r - 4))
        if all(math.dist(p, q) > 3 * r + 4 for q in pts):
            pts.append(p)
    for x, y in pts:
        dr.ellipse((x - r, y - r, x + r, y + r), fill=(0, 0, 0))
    opts, ans = options(str(k), [str(v) for v in range(1, 11) if v != k], rng)
    return im, ask("The image shows some black dots on white. How many dots are there?", opts), ans


def trace(rng, n):
    """Following: n lines from numbered starts on the top to lettered ends on the bottom, all black, crossing each other."""
    from PIL import ImageDraw
    im = canvas()
    dr = ImageDraw.Draw(im)
    tops = [SIZE * (i + 1) / (n + 1) for i in range(n)]
    ends = tops[:]
    order = list(range(n))
    rng.shuffle(order)
    for i in range(n):
        x0, x1 = tops[i], ends[order[i]]
        b = rng.uniform(.2, .8) * SIZE
        pts = []
        for t in range(41):
            u = t / 40
            y = 24 + u * (SIZE - 48)
            x = (1 - u) ** 2 * x0 + 2 * u * (1 - u) * b + u ** 2 * x1
            pts.append((x, y))
        dr.line(pts, fill=(0, 0, 0), width=2)
        dr.text((x0 - 3, 6), str(i + 1), fill=(0, 0, 0))
        dr.text((ends[i] - 3, SIZE - 18), ENDS[i], fill=(0, 0, 0))
    start = rng.randrange(n)
    return im, f"Black lines run from the numbers on top to the letters at the bottom, and they cross each other. Which letter does the line from number {start + 1} reach? Put the letter in \\boxed{{}}.", ENDS[order[start]]


def mark(rng, n):
    """Same grid as cell, the cell shown by an outline instead of named: colour perception at the same pixels, nothing to count."""
    from PIL import ImageDraw
    im = canvas()
    dr = ImageDraw.Draw(im)
    s = SIZE / n
    names = list(COLOURS)
    grid = [[rng.choice(names) for _ in range(n)] for _ in range(n)]
    for r in range(n):
        for c in range(n):
            dr.rectangle((c * s, r * s, (c + 1) * s - 1, (r + 1) * s - 1), fill=COLOURS[grid[r][c]])
    r, c = rng.randrange(n), rng.randrange(n)
    dr.rectangle((c * s, r * s, (c + 1) * s - 1, (r + 1) * s - 1), outline=(0, 0, 0), width=max(1, round(s / 8)))
    opts, ans = options(grid[r][c], [k for k in names if k != grid[r][c]], rng)
    return im, ask(f"The image is a grid of {n} rows and {n} columns of coloured cells. One cell has a black outline. What colour is that cell?", opts), ans


def strip(rng, n):
    """Same cell size as cell, one row only: counting along one axis, no second axis to bind."""
    from PIL import ImageDraw
    im = canvas()
    dr = ImageDraw.Draw(im)
    s = SIZE / n
    names = list(COLOURS)
    row = [rng.choice(names) for _ in range(n)]
    y = rng.randrange(n) * s
    for c in range(n):
        dr.rectangle((c * s, y, (c + 1) * s - 1, y + s - 1), fill=COLOURS[row[c]])
    c = rng.randrange(n)
    opts, ans = options(row[c], [k for k in names if k != row[c]], rng)
    return im, ask(f"The image shows one row of {n} coloured cells on white. What colour is cell number {c + 1}, counting from the left?", opts), ans


TASKS = {"ring": (ring, (6, 8, 12, 16, 24, 32, 48, 64, 96), "pixels across the ring"), "square": (square, (1, 2, 3, 4, 6, 8, 12, 16, 24), "square side"), "gap": (gap, (1, 2, 3, 4, 6, 8, 12, 16, 24), "gap"),
         "cell": (cell, (4, 6, 8, 12, 16, 24, 32, 48), "grid side"), "odd": (odd, (3, 4, 6, 8, 12, 16, 24), "grid side"), "dots": (dots, (1, 2, 3, 4, 6, 8, 12), "dot radius"), "trace": (trace, (2, 3, 4, 6, 8), "lines"),
         "mark": (mark, (4, 8, 12, 16, 24), "grid side"), "strip": (strip, (4, 8, 12, 16, 24), "cells")}


def build(d, per, seed=0, only=None):
    os.makedirs(os.path.join(d, "img"), exist_ok=True)
    rng = random.Random(seed)
    only = {x.split(":")[0]: [int(v) for v in x.split(":")[1].split(",")] for x in only.split("/")} if only else None
    with open(os.path.join(d, "items.jsonl"), "w") as f:
        for task, (fn, levels, unit) in TASKS.items():
            if only and task not in only:
                continue
            for lev in only[task] if only else levels:
                for k in range(per):
                    im, q, ans = fn(rng, lev)
                    path = os.path.abspath(os.path.join(d, "img", f"{task}_{lev}_{k}.png"))
                    im.save(path)
                    f.write(json.dumps({"id": f"{task}/{lev}/{k}", "task": task, "level": lev, "unit": unit, "prompt": q, "answer": ans, "image_paths": [path], "bench": task, "kind": "mc"}) + "\n")
    print("items written", flush=True)


def pop(argv, flag, default=None):
    if flag not in argv:
        return default
    i = argv.index(flag)
    val = argv[i + 1]
    del argv[i:i + 2]
    return val


def run(d, argv):
    label = pop(argv, "--label")
    only = pop(argv, "--tasks")
    edits = pop(argv, "--edits")
    think = pop(argv, "--think")
    keep_items = "--save-items" in argv                         # per arm, per item margins in NAME.arms.jsonl, in the order of NAME.jsonl: lets a selection be cross-fitted later
    if keep_items:
        argv.remove("--save-items")
    sys.argv = ["loop.py", "--run-dir", d] + argv
    CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path[:0] = [CODE, os.path.join(CODE, "engine")]
    import numpy as np
    import rsi.loop as L
    from surgery import Surgeon, image_message
    from vllm import SamplingParams

    a = L.args
    S = Surgeon(a.model, tp=a.tp, max_model_len=a.max_model_len, max_num_seqs=a.max_num_seqs, max_num_batched_tokens=a.max_num_batched_tokens,
                gpu_memory_utilization=a.gpu_util, kv_cache_dtype=a.kv_cache_dtype, multimodal=True)
    items = L.load_jsonl([os.path.join(d, "items.jsonl")])
    if only:
        items = [p for p in items if p["task"] in only.split(",")]
    if think:
        outs = S.generate([image_message(p["prompt"], p["image_paths"]) for p in items], n=1, sampling=dict(L.SAMPLING), max_tokens=int(think), reasoning_effort=a.effort)
        rows, toks = {}, []
        with open(os.path.join(d, f"{label}.jsonl"), "w") as f:
            for p, o in zip(items, outs):
                s = o["samples"][0]
                m = re.findall(r"\\boxed\{\s*\(?([A-H])\)?\s*\}", s["content"] or "")
                right = bool(m) and m[-1] == p["answer"] and s["finish_reason"] == "stop"
                rows.setdefault((p["task"], p["level"]), []).append(right)
                toks.append(s["num_tokens"])
                f.write(json.dumps({"id": p["id"], "task": p["task"], "level": p["level"], "right": right, "tokens": s["num_tokens"], "finish": s["finish_reason"], "content": (s["content"] or "")[-200:], "reasoning": (s["reasoning"] or "")[-3000:]}) + "\n")
        with open(os.path.join(d, "summary.jsonl"), "a") as f:
            f.write(json.dumps({"label": label, "acc": {f"{t}/{l}": float(np.mean(v)) for (t, l), v in rows.items()}}) + "\n")
        L.say(f"eye_exam: {label} thinking, tokens {np.mean(toks):.0f}, " + "; ".join(f"{t} " + " ".join(f"{l}:{np.mean(v):.2f}" for (tt, l), v in sorted(rows.items(), key=lambda kv: kv[0][1]) if tt == t) for t in TASKS if any(k[0] == t for k in rows)))
        return
    heads = S.generate([image_message(p["prompt"], p["image_paths"]) for p in items], n=1, sampling=dict(L.SAMPLING), max_tokens=1, enable_thinking=False)

    def tok(t):
        return S.tokenizer(t, add_special_tokens=False).input_ids

    prompts, cands, keep = [], [], []
    for p, o in zip(items, heads):
        opts = list(ENDS[:int(p["level"])]) if p["task"] == "trace" else sorted(set(re.findall(r"\(([A-Z])\)", p["prompt"]) or re.findall(r"(?:^|\s)([A-Z])\.\s", p["prompt"])) | {p["answer"]})
        ids = [tok("\\boxed{" + x) for x in opts]
        n = next(i for i in range(len(ids[0]) + 1) if i == len(ids[0]) or any(i >= len(y) or y[i] != ids[0][i] for y in ids))
        if len(opts) < 2 or any(len(y) <= n for y in ids):
            continue
        prompts.append(S._token_prompt(list(o["prompt_token_ids"]) + ids[0][:n], p["image_paths"]))
        cands.append(([y[n] for y in ids], opts.index(p["answer"])))
        keep.append(dict(p, tokens=len(o["prompt_token_ids"])))
    params = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20, detokenize=False)

    def read():
        margin, right = np.zeros(len(keep)), np.zeros(len(keep), dtype=bool)
        for k, ((c, g), o) in enumerate(zip(cands, S.llm.generate(prompts, [params] * len(prompts), use_tqdm=False))):
            top = o.outputs[0].logprobs[0]
            floor = min(v.logprob for v in top.values()) - 1.0
            lp = np.array([top[t].logprob if t in top else floor for t in c], dtype=np.float64)
            rest = np.delete(lp, g)
            right[k], margin[k] = lp[g] > rest.max(), lp[g] - np.logaddexp.reduce(rest)
        return margin, right

    def table(right):
        rows = {}
        for p, r in zip(keep, right):
            rows.setdefault((p["task"], p["level"]), []).append(bool(r))
        return rows

    def line(rows):
        return "; ".join(f"{t} " + " ".join(f"{l}:{np.mean(v):.2f}" for (tt, l), v in sorted(rows.items(), key=lambda kv: kv[0][1]) if tt == t) for t in TASKS if any(k[0] == t for k in rows))

    def save(name, rows, **extra):
        with open(os.path.join(d, "summary.jsonl"), "a") as f:
            f.write(json.dumps({"label": name, "acc": {f"{t}/{l}": float(np.mean(v)) for (t, l), v in rows.items()}, **extra}) + "\n")

    base, right = read()
    with open(os.path.join(d, f"{label}.jsonl"), "w") as f:
        for p, m, r in zip(keep, base, right):
            f.write(json.dumps({"id": p["id"], "task": p["task"], "level": p["level"], "right": bool(r), "margin": float(m), "tokens": p["tokens"]}) + "\n")
    save(label, table(right))
    L.say(f"eye_exam: {label} done, " + line(table(right)))
    rng = np.random.default_rng(0)
    tasks = np.array([p["task"] for p in keep])
    for raw in json.load(open(edits)) if edits else []:
        h = S.apply(raw["engine"])
        m, r = read()
        S.revert(h)
        out, per = [], {}
        for t in dict.fromkeys(tasks):
            x = (m - base)[tasks == t]
            b = x[rng.integers(0, x.size, (2000, x.size))].mean(1)
            per[t] = {"mean": float(x.mean()), "lo": float(np.quantile(b, .05)), "hi": float(np.quantile(b, .95))}
            out.append(f"{t} {x.mean():+.3f} [{np.quantile(b, .05):+.3f},{np.quantile(b, .95):+.3f}]")
        save(f"{label}:{raw['label']}", table(r), margin=per)
        if keep_items:
            with open(os.path.join(d, f"{label}.arms.jsonl"), "a") as f:
                f.write(json.dumps({"label": raw["label"], "margin": [round(float(x), 4) for x in m]}) + "\n")
        L.say(f"eye_exam: {label}:{raw['label']} margin " + " ".join(out) + " | " + line(table(r)))
    if edits:
        open(os.path.join(d, "scan_done"), "w").close()


def report(d):
    rows = [json.loads(l) for l in open(os.path.join(d, "summary.jsonl"))]
    for task, (_, levels, unit) in TASKS.items():
        print(f"{task} ({unit})")
        print(" " * 18 + " ".join(f"{l:>5}" for l in levels))
        for r in rows:
            print(f"{r['label'][:18]:18s}" + " ".join(f"{r['acc'].get(f'{task}/{l}', float('nan')):5.2f}" for l in levels))


if __name__ == "__main__":
    cmd, d = sys.argv[1], sys.argv[2]
    if cmd == "build":
        build(d, int(pop(sys.argv, "--per-level", "40")), int(pop(sys.argv, "--seed", "0")), pop(sys.argv, "--only"))
    elif cmd == "run":
        run(d, sys.argv[3:])
    else:
        report(d)
