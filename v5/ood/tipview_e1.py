"""Tip view with entry IDX made unconditional (cond -> always; text, order, settings untouched): python tipview_e1.py TIP ROLES SAMPLES OUT.jsonl IDX ROUTED_OUT.json
Based on tipview.py. Tip view requests from the loop's own code (gen.py Run.read_req): python tipview.py TIP_ID ROLES SAMPLES OUT.jsonl
writes the tip's own read request per (row, sample) (key = the tip's key: x|sig|row|s, or p000|parent|row|s on stock rows)."""
import collections, glob, json, sys
V4 = "/data/fd/mingdazhang/selfedit/v4"; RUN = f"{V4}/runs/main_1110"; LOOP = "/workspace/fudan/mingdazhang/selfedit-v4/loop"
sys.path.insert(0, LOOP)
import gen as G
tip_id, roles, samples, out = sys.argv[1], set(sys.argv[2].split(",")), [int(x) for x in sys.argv[3].split(",")], sys.argv[4]
IDX, ROUTED_OUT = int(sys.argv[5]), sys.argv[6]
st = json.load(open(f"{RUN}/state.json"))
tip = st["tip"]
assert tip["id"] == tip_id, (tip["id"], tip_id)
R = G.Run.__new__(G.Run)
R.knobs = json.load(open(f"{LOOP}/knobs.json")); R.S = R.knobs["settings"]; R.stock = {k: R.S[k]["stock"] for k in R.S}
rows = {}
for l in open(f"{V4}/rows/learn.jsonl"):
    d = json.loads(l)
    if d.get("v4_role") in roles:
        rows[d["id"]] = d
conds = {e["cond"] for e in R.flat(tip["entries"]) if e["cond"] != "always"}
R.routes, last = {}, {}
for f in glob.glob(f"{V4}/q/out/*.jsonl"):
    for l in open(f, "rb"):
        if l.startswith(b'{"key": "rt1|'):
            x = json.loads(l); last[x["key"]] = x
for k, x in last.items():
    _, c, r = k.split("|", 2)
    if c in conds and r in rows:
        p = x.get("p")
        R.routes[(c, r)] = bool(p >= 0.5) if isinstance(p, (int, float)) else bool(x.get("yes"))
miss = sum(1 for c in conds for r in rows if (c, r) not in R.routes)
print(f"tip {tip_id}: {len(tip['entries'])} entries, {len(conds)} conditions, rows {len(rows)}, route decisions missing {miss}", file=sys.stderr)
import copy
E1 = copy.deepcopy(tip["entries"]); c1 = E1[IDX]["cond"]; assert c1 != "always" and not E1[IDX].get("parts") and not E1[IDX].get("settings") and not E1[IDX].get("ops") and not E1[IDX].get("edit"), E1[IDX]
E1[IDX]["cond"] = "always"
routed = {r: bool(R.routes.get((c1, r))) for r in rows}
json.dump(routed, open(ROUTED_OUT, "w"))
print(f"entry {IDX} cond {c1}: routed on {sum(routed.values())}/{len(rows)} rows", file=sys.stderr)
nchg = 0
n = collections.Counter()
with open(out, "w") as fo:
    for r in sorted(rows):
        for s in samples:
            q0, _ = R.read_req(tip["entries"], r, s, 0)
            q, sg = R.read_req(E1, r, s, 0)
            assert (q["prefix"] == q0["prefix"]) == routed[r] and q["settings"] == q0["settings"]
            if not routed[r]:
                p0 = q0["prefix"] or ""; e1 = tip["entries"][IDX]["procedure"]; assert e1 in q["prefix"] and q["prefix"].replace(e1, "", 1).replace("\n\n\n\n", "\n\n").strip() == p0.strip(); nchg += 1
            n[q["key"].split("|")[0]] += 1
            fo.write(json.dumps(q) + "\n")
print(f"wrote {sum(n.values())} requests {dict(n)}; prefix changed on {nchg} (row, sample)", file=sys.stderr)
