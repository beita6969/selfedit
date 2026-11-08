"""paired.py RESULTS TAG_A TAG_B [val|select|val_focus]: paired change in macro between two evaluations on the same rows.
Noise level = standard error of the paired macro difference, from a bench-stratified paired bootstrap over rows (2000 draws, seed 0). A rise below 1 SE is reported as no rise; 2 SE is the level for calling a change."""
import json, os, random, sys, collections
R, A, B = sys.argv[1:4]; V = sys.argv[4] if len(sys.argv) > 4 else "val"
F = None
if V == "val_focus":
    F = set(json.load(open("/home/bedicloud/sharestore2/zxc/selfedit/baselines/data/focus.json"))["val_ids"]); V = "val"
ld = lambda t: {x["id"]: x for x in map(json.loads, open(f"{os.path.dirname(t)}/{V}_{os.path.basename(t)}.jsonl" if "/" in t else f"{R}/{V}_{t}.jsonl"))}   # TAG may be DIR/TAG, for stock from another run
a, b = ld(A), ld(B); ids = sorted(set(a) & set(b) if F is None else set(a) & set(b) & F)
assert ids and (F is not None or set(a) == set(b))
by = collections.defaultdict(list)
for i in ids:
    by[a[i]["bench"]].append((int(a[i]["ok"]), int(b[i]["ok"])))
mac = lambda d: sum(sum(y - x for x, y in v) / len(v) for v in d.values()) / len(d)
rng = random.Random(0); boots = []
for _ in range(2000):
    boots.append(mac({k: [rng.choice(v) for _ in v] for k, v in by.items()}))
m = sum(boots) / len(boots); se = (sum((x - m) ** 2 for x in boots) / (len(boots) - 1)) ** .5
out = dict(rows=len(ids), macro_change=round(mac(by), 4), paired_se=round(se, 4), z=round(mac(by) / se, 2) if se else None,
           per_bench={k: dict(n=len(v), change=round(sum(y - x for x, y in v) / len(v), 4), gained=sum(y > x for x, y in v), lost=sum(y < x for x, y in v)) for k, v in sorted(by.items())},
           gained=sum(y > x for v in by.values() for x, y in v), lost=sum(y < x for v in by.values() for x, y in v))
out["reading"] = "rise" if out["macro_change"] >= 2 * se else "drop" if out["macro_change"] <= -2 * se else "no clear change"
print(json.dumps(out))
