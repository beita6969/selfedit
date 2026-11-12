"""Per-set share of OOD v2 rows routed by each t007c condition, and share with no condition fired (the loop's own fires())."""
import collections, glob, json, sys
V4 = "/data/fd/mingdazhang/selfedit/v4"; RUN = f"{V4}/runs/main_1110"; LOOP = "/workspace/fudan/mingdazhang/selfedit-v4/loop"
sys.path.insert(0, LOOP)
import gen as G
tip = json.load(open(f"{RUN}/state.json"))["tip"]
R = G.Run.__new__(G.Run); R.routes = {}
rows = {}
for l in open(f"{V4}/rows/learn.jsonl"):
    d = json.loads(l)
    if d.get("v4_role") in sys.argv[1].split(","): rows[d["id"]] = d
last = {}
for l in open(f"{V4}/q/out/ood_routes.jsonl") if sys.argv[1] == "ood_v2" else []:
    x = json.loads(l); last[x["key"]] = x
if sys.argv[1] != "ood_v2":
    for f in glob.glob(f"{V4}/q/out/*.jsonl"):
        for l in open(f, "rb"):
            if l.startswith(b'{"key": "rt1|'): x = json.loads(l); last[x["key"]] = x
err = 0
for k, x in last.items():
    _, c, r = k.split("|", 2)
    if r in rows:
        if x.get("error"): err += 1; continue
        p = x.get("p"); R.routes[(c, r)] = bool(p >= 0.5) if isinstance(p, (int, float)) else bool(x.get("yes"))
ents = [e for e in R.flat(tip["entries"]) if e["cond"] != "always"]
names = {e["cond"]: f"e{i+1}" for i, e in enumerate(ents)}
miss = sum(1 for e in ents for r in rows if (e["cond"], r) not in R.routes)
print(f"rows {len(rows)}; route answers {len(last)}; errors {err}; missing decisions {miss}")
byb = collections.defaultdict(list)
for r, d in rows.items(): byb[d["bench"]].append(r)
hdr = "set (n) | " + " | ".join(names[e["cond"]] for e in ents) + " | any | none"
print(hdr)
fired = {}
for b in sorted(byb) + ["ALL"]:
    rs = list(rows) if b == "ALL" else byb[b]
    sh = [sum(R.routes.get((e["cond"], r), False) for r in rs) / len(rs) for e in ents]
    anyf = [r for r in rs if any(R.routes.get((e["cond"], r), False) for e in ents)]
    for r in anyf: fired[r] = [names[e["cond"]] for e in ents if R.routes.get((e["cond"], r), False)]
    print(f"{b} ({len(rs)}) | " + " | ".join(f"{100*s:.0f}" for s in sh) + f" | {100*len(anyf)/len(rs):.1f} | {100*(1-len(anyf)/len(rs)):.1f}")
print("conditions: " + " | ".join(f"{names[e['cond']]}: {e['condition'][:70]}" for e in ents))
json.dump({r: fired.get(r, []) for r in rows}, open(f"{V4}/scratch/ext/ood_fired.json", "w"))
