"""Acceptance statistics for candidate edits: per-problem paired comparison against the current generation."""
import json, random
from collections import defaultdict


def per_problem(records, budget=None):
    """Map problem id -> list of 0/1 outcomes; with a budget, samples longer than it count as wrong."""
    out = defaultdict(list)
    for r in records:
        ok = bool(r["ok"]) and (budget is None or r["tokens"] <= budget)
        out[r["id"]].append(1.0 if ok else 0.0)
    return out


def tokens_per_problem(records):
    out = defaultdict(list)
    for r in records:
        out[r["id"]].append(r["tokens"])
    return out


def paired_bootstrap(base, cand, iters=10000, seed=0):
    """Mean per-problem accuracy difference (cand - base) over shared problems, with a 90% bootstrap interval."""
    ids = sorted(set(base) & set(cand))
    d = [sum(cand[i]) / len(cand[i]) - sum(base[i]) / len(base[i]) for i in ids]
    if not d:
        return {"n": 0}
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(iters))
    return {"n": len(d), "diff": sum(d) / len(d), "lo90": means[int(0.05 * iters)], "hi90": means[int(0.95 * iters)],
            "lo98": means[int(0.01 * iters)], "hi98": means[int(0.99 * iters)]}


def paired_bootstrap_tokens(base, cand, iters=10000, seed=0):
    """Relative change of mean tokens over shared problems, cand / base - 1 as the ratio of the per-problem means summed
    over problems, with bootstrap intervals over problems. Before 11-04 this averaged per-problem ratios, which the
    base's sampling noise in the denominator biases upward (E[1/x] > 1/E[x]): +4-12 % at 2-4 samples per problem with
    the absolute change near zero, so the efficiency route needed a true cut of ~10 % to clear its -5 % screen."""
    ids = sorted(i for i in set(base) & set(cand) if sum(base[i]) > 0)
    if not ids:
        return {"n": 0}
    c = [sum(cand[i]) / len(cand[i]) for i in ids]
    b = [sum(base[i]) / len(base[i]) for i in ids]
    n, rng = len(ids), random.Random(seed)
    means = []
    for _ in range(iters):
        k = [rng.randrange(n) for _ in range(n)]
        means.append(sum(c[j] for j in k) / sum(b[j] for j in k) - 1)
    means.sort()
    return {"n": n, "rel": sum(c) / sum(b) - 1, "lo90": means[int(0.05 * iters)], "hi90": means[int(0.95 * iters)],
            "lo98": means[int(0.01 * iters)], "hi98": means[int(0.99 * iters)]}


def decide(main, guards, stage, guard_tol=0.01, efficiency=False, bound="90"):
    """Screen: main lower bound > -0.01 and mean > 0; confirm (fresh problems): main lower bound > 0. Guards must not drop beyond tolerance.
    With efficiency=True an edit may also pass by using clearly fewer tokens at no accuracy cost
    (screen: accuracy not lower and tokens down >= 5% on average; confirm: accuracy not lower and the token interval below zero).
    bound="98" uses the 98% interval (first look of the two-look confirm, see loop.py)."""
    lo, hi = "lo" + bound, "hi" + bound
    ok_main = main["n"] > 0 and (main["diff"] > 0 and main["lo90"] > -0.01 if stage == "screen" else main[lo] > 0)
    tok = main.get("tok", {"n": 0})
    if efficiency and not ok_main and main["n"] > 0 and tok["n"] > 0 and main["diff"] >= 0:
        ok_main = tok["rel"] <= -0.05 if stage == "screen" else tok[hi] < 0
    ok_guard = all(g["n"] == 0 or g["diff"] > -guard_tol for g in guards)
    return ok_main and ok_guard


def retention(base, cand):
    """Accuracy on the problems the base finished in every sample: an edit that only frees truncated answers leaves it
    unchanged (WSP's originally_finished check: 93.86 -> 93.86), one that trades checking for speed lowers it."""
    fin = {}
    for r in base:
        fin[r["id"]] = fin.get(r["id"], True) and r["finish"] != "length"
    ids = {k for k, v in fin.items() if v}
    b = [float(r["ok"]) for r in base if r["id"] in ids]
    c = [float(r["ok"]) for r in cand if r["id"] in ids]
    if not b or not c:
        return {"n": 0, "base": None, "cand": None, "diff": 0.0}
    return {"n": len(ids), "base": sum(b) / len(b), "cand": sum(c) / len(c), "diff": sum(c) / len(c) - sum(b) / len(b)}
