"""Turns one adjoint read into candidate edits, one per theory, all at the same estimated cost, so a theory is judged by what it builds and not by how large an edit it dares.
Rules, for writers w with mean gain g, standard error se, curvature h (all read per item):
  raw      top K by |g|, dose g / |h|, clipped: what the blind scan did
  index    Smith-Hazel selection index: g shrunk by its reliability, over the curvature, every writer
  bang     Pontryagin on a box: the same step on every writer whose sign is established
  natural  natural gradient: g / F, with F the cost per writer
  flip     the index with its sign reversed, a negative control
  random   the writers and sizes of the index with random signs, an impact-matched control
Cost of an edit: sum F_w d_w^2 / 2, with F from --cost (writer to KL per unit dose squared), or the curvature if absent. Each rule is rescaled to each budget in --budgets.

With --expr FILE the arms are only the rules in that file, [{name, expr, budget, clip}], each expr checked by rsi/mm_rsi.py's check_expr and evaluated over the read. ARMS.json.summary holds the read by group of writers.

Usage: python rsi/mm_select.py ADJ.base.npz --out ARMS.json [--cost COST.json] [--budgets a,b] [--clip .45] [--key G] [--writers attn] [--prefix r1] [--half 0] [--expr RULES.json]
"""
import argparse, json, re, zlib
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("adj")
ap.add_argument("--out", required=True)
ap.add_argument("--cost")
ap.add_argument("--budgets", default=".002,.01")
ap.add_argument("--clip", type=float, default=.45)
ap.add_argument("--key", default="G")
ap.add_argument("--top", type=int, default=16)
ap.add_argument("--writers", default="scan")
ap.add_argument("--prefix", default="r")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--half", type=int, default=-1)         # select only on items with crc32(id) % 2 == half, so the other half stays fresh
ap.add_argument("--expr")
a = ap.parse_args()
z = np.load(a.adj, allow_pickle=False)
names = [str(x) for x in z["names"]]
keep = [i for i, n in enumerate(names) if a.writers == "all" or not n.endswith(".mlp") or n.startswith("v")]
skip = set(int(x) for x in z["skipped"]) if "skipped" in z else set()
rows = [i for i in range(z["G"].shape[0]) if i not in skip and (a.half < 0 or zlib.crc32(str(z["ids"][i]).encode()) % 2 == a.half)]
key = a.key if a.key in z and np.abs(z[a.key]).sum() > 0 else "G"
G = z[key][rows][:, keep]
names = [names[i] for i in keep]
n = len(rows)
# items are given equal say by trimming: the tower's per item derivative is heavy tailed
srt = np.sort(G, 0)
cut = int(n * .05)
g = srt[cut:n - cut].mean(0)
se = srt[cut:n - cut].std(0) / np.sqrt(n - 2 * cut)
var = max(0.0, float(g.var() - (se ** 2).mean()))
rel = var / (var + se ** 2 + 1e-12)
h = -np.abs(z["diag"][rows][:, keep].mean(0)) if "diag" in z and np.abs(z["diag"]).sum() > 0 else -(np.sqrt(z["nf"][keep] * z["np"][keep]) + 1e-9)
if h.max() == h.min():
    h = -np.ones_like(h)
h = np.minimum(h, np.quantile(h, .5))                # no writer is free
cost = json.load(open(a.cost)) if a.cost else {}
F = np.array([cost.get(k, 0.0) for k in names]) if cost else -h / np.abs(h).mean()
F = np.maximum(F, np.quantile(F[F > 0], .1) if (F > 0).any() else 1.0)
rng = np.random.default_rng(a.seed)
rules = {}
top = np.argsort(-np.abs(g))[:a.top]
d = np.zeros_like(g); d[top] = g[top] / -h[top]
rules["raw"] = d
idx = rel * g / -h
rules["index"] = idx
d = np.zeros_like(g); on = np.abs(g) / se > 2.5
d[on] = np.sign(g[on])
rules["bang"] = d
rules["natural"] = rel * g / F
rules["flip"] = -idx
rules["random"] = np.abs(idx) * rng.choice([-1.0, 1.0], len(idx))


def engine(k, alpha):
    if k.startswith("v"):
        b, part = re.match(r"v(\d+)\.(attn|mlp)", k).groups()
        w = f"visual.blocks.{b}." + ("attn.proj" if part == "attn" else "mlp.linear_fc2")
        return [{"op": "scale", "param": w + ".weight", "alpha": alpha}, {"op": "scale", "param": w + ".bias", "alpha": alpha}]
    L = int(re.match(r"t(\d+)", k)[1])
    if k.endswith(".mlp"):
        return [{"op": "scale", "param": f"layers.{L}.mlp.down_proj", "alpha": alpha}]
    return [{"op": "scale", "param": f"layers.{L}.self_attn.o_proj" if L % 4 == 3 else f"layers.{L}.linear_attn.out_proj", "alpha": alpha}]


arms, info = [], {}


def build(label, d, b, clip):
    s = np.sqrt(2 * b / max(1e-12, float((F * d ** 2).sum())))
    dd = np.clip(s * d, -clip, clip)
    for _ in range(5):                                # clipped writers leave budget for the others
        room = b - float((F * dd ** 2).sum()) / 2
        free = np.abs(dd) < clip - 1e-9
        if room <= 1e-9 or not free.any():
            break
        s2 = np.sqrt(1 + 2 * room / max(1e-12, float((F[free] * dd[free] ** 2).sum())))
        dd[free] = np.clip(dd[free] * s2, -clip, clip)
    on = np.abs(dd) > .002
    arms.append({"label": label, "effort": "medium", "engine": [e for k, x in zip(np.array(names)[on], dd[on]) for e in engine(str(k), round(float(1 + x), 4))]})
    info[label] = dict(writers=int(on.sum()), predicted_gain=float(g @ dd + .5 * (h * dd ** 2).sum()), cost=float((F * dd ** 2).sum()) / 2, doses={k: round(float(x), 4) for k, x in zip(np.array(names)[on], dd[on])})
    print(f"{label}: {on.sum()} writers, max dose {np.abs(dd).max():.3f}, predicted margin gain {info[label]['predicted_gain']:+.3f}, cost {info[label]['cost']:.4f}")


z_ = g / (se + 1e-12)
tower = np.array([float(k.startswith("v")) for k in names])
attn = np.array([float(k.endswith(".attn")) for k in names])
at = np.array([int(re.match(r"[vt](\d+)", k)[1]) for k in names], dtype=float)
pos = np.where(tower > 0, at / 26, at / 63)
lines = []
for part, on in [(f"v{x}-{x + 8}.{p}", (tower > 0) & (attn == (p == "attn")) & (at >= x) & (at <= x + 8)) for x in (0, 9, 18) for p in ("attn", "mlp")] + [(f"t{x}-{x + 15}.{p}", (tower == 0) & (attn == (p == "attn")) & (at >= x) & (at <= x + 15)) for x in (0, 16, 32, 48) for p in ("attn", "mlp")]:
    if on.any():
        lines.append(f"{part}: {on.sum()} writers, mean g {g[on].mean():+.3f}, mean z {z_[on].mean():+.2f}, z>2.5 on {(z_[on] > 2.5).sum()}, z<-2.5 on {(z_[on] < -2.5).sum()}, mean h {h[on].mean():.3f}, mean rel {rel[on].mean():.2f}, mean F {F[on].mean():.3f}")
order = np.argsort(-z_)
lines += [f"{names[i]}: g {g[i]:+.3f}, z {z_[i]:+.2f}, h {h[i]:.3f}, F {F[i]:.3f}" for i in list(order[:8]) + list(order[-8:])]
with open(a.out + ".summary", "w") as f:
    f.write("\n".join(lines) + "\n")
if a.expr:
    import ast, os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from mm_rsi import check_expr
    env = dict(g=g, h=h, se=se, z=z_, rel=rel, F=F, pos=pos, tower=tower, text=1 - tower, attn=attn, mlp=1 - attn, abs=np.abs, sign=np.sign, sqrt=np.sqrt, tanh=np.tanh, exp=np.exp, where=np.where, clip=np.clip, maximum=np.maximum, minimum=np.minimum, mean=np.mean, median=np.median, quantile=np.quantile)
    for r in json.load(open(a.expr)):
        err = check_expr(r["expr"])
        if err:
            print(f"{r['name']}: {err}")
            continue
        try:
            d = np.broadcast_to(np.asarray(eval(compile(ast.parse(r["expr"], mode="eval"), "rule", "eval"), {"__builtins__": {}}, env), dtype=float), g.shape).copy()
        except Exception as ex:
            print(f"{r['name']}: {r['expr']}: {ex}")
            continue
        if not np.isfinite(d).all() or not np.abs(d).sum() > 0:
            print(f"{r['name']}: {r['expr']}: no finite non-zero edit")
            continue
        build(f"{a.prefix}_{r['name']}_{r['budget']:g}", d, r["budget"], r["clip"])
else:
    for b in (float(x) for x in a.budgets.split(",")):
        for rule, d in rules.items():
            build(f"{a.prefix}_{rule}_{b:g}", d, b, a.clip)
json.dump(arms, open(a.out, "w"))
json.dump(info, open(a.out + ".info", "w"), indent=1)
print(f"reliability of the gain: mean {rel.mean():.2f}; writers with |z| > 2.5: {(np.abs(g) / se > 2.5).sum()} of {len(g)}; key {key}, items {n}")
