"""entry_effects (fable_int, ranking of the next edit; SIGNAL rows only): what each routed text entry of the tip t007c is worth on the signal rows where it fires.
Tip reads (export e1: t007c, 8 samples) against the floor-only base reads (export e13, 8 samples) on the same SIGNAL rows (roles show + monitor; JUDGE rows are not read).
Per entry: rows where it fires, by benchmark; accuracy with the tip's texts and without any text on those rows; the same on rows where it is the ONLY routed text; and its macro
contribution = mean over the six benchmarks of (difference on the entry's rows x share of the benchmark's signal rows it fires on). se: over rows (paired row means).
python entry_effects.py FIRED.jsonl.gz TIP_READS.jsonl.gz FLOOR_READS.jsonl.gz"""
import gzip, json, sys, collections
import numpy as np
fired_f, tip_f, floor_f = sys.argv[1:4]
L = [json.loads(l) for l in gzip.open(fired_f, "rt")]
tipdef = L[0]; ent = {e["cid"]: e for e in tipdef["entries"]}
rows = {}
for r in L[1:]:
    rid = r.get("row") or r.get("id"); f = r.get("fired") or r.get("cids") or []
    rows[rid] = [x["cid"] if isinstance(x, dict) else x for x in f]
print("fired-file keys of a row line:", sorted(L[1].keys()))
acc = {}
for name, f in (("tip", tip_f), ("floor", floor_f)):
    d = collections.defaultdict(list); role = {}
    for l in gzip.open(f, "rt"):
        r = json.loads(l); d[r["row"]].append(bool(r["ok"])); role[r["row"]] = r.get("role")
    acc[name] = {k: float(np.mean(v)) for k, v in d.items()}; acc[name + "_role"] = role
sig = sorted(r for r in acc["floor"] if acc["floor_role"][r] in ("show", "monitor") and r in acc["tip"] and r in rows)
bench = lambda r: r.split("/")[0]; B = sorted({bench(r) for r in sig}); NB = collections.Counter(bench(r) for r in sig)
print(f"signal rows with both kinds of read: {len(sig)}; per benchmark {dict(NB)}; tip {100 * np.mean([acc['tip'][r] for r in sig]):.2f}, floor-only {100 * np.mean([acc['floor'][r] for r in sig]):.2f}")
text = [c for c, e in ent.items() if e.get("procedure")]
print("entry | rows fired (per benchmark) | tip - floor on its rows, points +- se | rows where it is the only routed text: n, difference +- se | macro contribution, points +- se | condition")
for c in text:
    R = [r for r in sig if c in rows[r]]; solo = [r for r in R if [x for x in rows[r] if ent.get(x, {}).get("procedure")] == [c]]
    if not R:
        print(c, "| fires on no signal row"); continue
    d = np.array([acc["tip"][r] - acc["floor"][r] for r in R]) * 100; ds = np.array([acc["tip"][r] - acc["floor"][r] for r in solo]) * 100
    macro = 0.0; var = 0.0
    for b in B:
        db = np.array([acc["tip"][r] - acc["floor"][r] for r in R if bench(r) == b]) * 100
        if len(db):
            macro += db.sum() / NB[b] / len(B); var += (db.var(ddof=1) * len(db) if len(db) > 1 else 0.0) / NB[b] ** 2 / len(B) ** 2
    pb = collections.Counter(bench(r) for r in R)
    print(f"{c} | {len(R)} ({', '.join(f'{b[:5]} {pb[b]}' for b in B if pb[b])}) | {d.mean():+.2f} +- {d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0:.2f} | {len(solo)}, {ds.mean() if len(ds) else float('nan'):+.2f} +- {ds.std(ddof=1) / np.sqrt(len(ds)) if len(ds) > 1 else 0:.2f} | {macro:+.2f} +- {np.sqrt(var):.2f} | {ent[c]['condition'][:90]}")
    for b in B:
        Rb = [r for r in R if bench(r) == b]
        if len(Rb) >= 4:
            print(f"      {b}: rows {len(Rb)}, tip {100 * np.mean([acc['tip'][r] for r in Rb]):.1f}, floor-only {100 * np.mean([acc['floor'][r] for r in Rb]):.1f}")
