"""mk_rows_c (fable_int, law table): one shared row set for components that apply to every row (template modes, system notes, always-on prefixes): signal rows WITHOUT a routed prefix
(for them the tip's read is a floor-only base read). TEST = rows with crc32(row) % 5 == 0, NT per benchmark; FIT = other rows, NF per benchmark; one read each (150-3000 thinking tokens,
prompt <= 6500 tokens, finish stop), seeded. Writes OUT_fit_items.txt and OUT_test_items.txt. python mk_rows_c.py TRACES OUT [NF NT]"""
import json, os, sys, zlib, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import intlib
D = json.load(open(sys.argv[1])); out = sys.argv[2]; NF = int(sys.argv[3]) if len(sys.argv) > 3 else 6; NT = int(sys.argv[4]) if len(sys.argv) > 4 else 2
kw = dict(gen_lo=150, gen_hi=3000, prompt_hi=6500); rng = np.random.default_rng(3333)
un = sorted(r for r, v in D.items() if v.get("role") in ("show", "monitor") and not v.get("prefix"))
res = {"fit": [], "test": []}
for b in sorted({D[r]["bench"] for r in un}):
    for name, n, cond in (("fit", NF, lambda r: zlib.crc32(r.encode()) % 5 != 0), ("test", NT, lambda r: zlib.crc32(r.encode()) % 5 == 0)):
        L = [(r, intlib.pick_sample(D, r, **kw)) for r in un if D[r]["bench"] == b and cond(r)]; L = [x for x in L if x[1] is not None]
        idx = rng.permutation(len(L))[:n]; res[name] += [L[i] for i in sorted(idx)]
for name, items in res.items():
    fn = f"{out}_{name}_items.txt"; assert not os.path.exists(fn); open(fn, "w").write("".join(f"{r} {k}\n" for r, k in items))
    print(name, len(items), dict(collections.Counter(D[r]["bench"] for r, _ in items)), "thinking tokens median", int(np.median([int(D[r]["samples"][str(k)]["tokens"]) for r, k in items])))
