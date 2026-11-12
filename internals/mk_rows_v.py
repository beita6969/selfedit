"""mk_rows_v (fable_int, Vesta): row sets for one routed procedure of a tip whose traces file holds the routed prefix per row (t005r: the counting entry g002c26 only).
Routed = signal rows with a prefix. TEST = rows with crc32(row) % 5 == 0 (never used in any fit), FIT = the others.
Files (ids and "row sample" items): proc_fit (routed fit rows, one read each), proc_fit2 (the same rows, up to NS reads each), proc_test (routed test rows, one read),
proc_unrouted (unrouted test rows, 4 per benchmark: leakage), proc_unrfit (unrouted fit rows, up to 14 per benchmark: route key).
python mk_rows_v.py TRACES OUT_PREFIX [NS]"""
import json, os, sys, zlib, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import intlib
D = json.load(open(sys.argv[1])); out = sys.argv[2]; NS = int(sys.argv[3]) if len(sys.argv) > 3 else 2
kw = dict(gen_lo=150, gen_hi=3000, prompt_hi=6500)
sig = sorted(r for r, v in D.items() if v.get("role") in ("show", "monitor"))
routed = [r for r in sig if D[r].get("prefix")]; un = [r for r in sig if not D[r].get("prefix")]
ist = lambda r: zlib.crc32(r.encode()) % 5 == 0
pk = lambda L: [(r, intlib.pick_sample(D, r, **kw)) for r in L if intlib.pick_sample(D, r, **kw) is not None]


def multi(L, n):
    o = []
    for r in L:
        ks = sorted(int(k) for k, s in D[r]["samples"].items() if kw["gen_lo"] <= int(s["tokens"]) <= kw["gen_hi"] and int(s.get("prompt_tokens") or 0) <= kw["prompt_hi"] and s.get("reasoning") and s.get("finish") == "stop")
        k0 = intlib.pick_sample(D, r, **kw)
        if k0 is None:
            continue
        ks = [k0] + [k for k in ks if k != k0]; o += [(r, k) for k in ks[:n]]
    return o


rng = np.random.default_rng(1111)
def perbench(L, n):
    o = []
    for b in sorted({D[r]["bench"] for r in L}):
        X = pk([r for r in L if D[r]["bench"] == b]); idx = rng.permutation(len(X))[:n]; o += [X[i] for i in sorted(idx)]
    return o


sets = dict(fit=pk([r for r in routed if not ist(r)]), fit2=multi([r for r in routed if not ist(r)], NS), test=pk([r for r in routed if ist(r)]),
            unrouted=perbench([r for r in un if ist(r)], 4), unrfit=perbench([r for r in un if not ist(r)], 14))
for name, items in sets.items():
    for suf, txt in (("ids", "".join(r + "\n" for r in dict.fromkeys(r for r, _ in items))), ("items", "".join(f"{r} {k}\n" for r, k in items))):
        fn = f"{out}_{name}_{suf}.txt"
        assert not os.path.exists(fn), fn
        open(fn, "w").write(txt)
    tk = [int(D[r]["samples"][str(k)]["tokens"]) for r, k in items]
    print(name, len(items), "reads on", len({r for r, _ in items}), "rows", dict(collections.Counter(D[r]["bench"] for r, _ in items)), "thinking tokens median", int(np.median(tk)), "min", min(tk), "max", max(tk))
v = D[routed[0]]
print("signal rows", len(sig), "routed", len(routed), "| samples per row", sorted(v["samples"].keys(), key=int)[:10], "| fields", sorted(k for k in v if k != "samples"), "| settings", v.get("settings"), "| prefix:", repr(v["prefix"][:70]), "| distinct prefixes", len({D[r]["prefix"] for r in routed}))
