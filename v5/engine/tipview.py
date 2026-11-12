"""Tip view requests from the loop's own code (gen.py Run.read_req): python tipview.py TIP_ID ROLES SAMPLES OUT.jsonl
writes the tip's own read request per (row, sample) (key = the tip's key: x|sig|row|s, or p000|parent|row|s on stock rows)."""
import collections, glob, json, sys
V4 = "/data/fd/mingdazhang/selfedit/v4"; RUN = f"{V4}/runs/main_1110"; LOOP = "/workspace/fudan/mingdazhang/selfedit-v4/loop"
sys.path.insert(0, LOOP)
import gen as G
tip_id, roles, samples, out = sys.argv[1], set(sys.argv[2].split(",")), [int(x) for x in sys.argv[3].split(",")], sys.argv[4]
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
n = collections.Counter()
with open(out, "w") as fo:
    for r in sorted(rows):
        for s in samples:
            q, sg = R.read_req(tip["entries"], r, s, 0)
            n[q["key"].split("|")[0]] += 1
            fo.write(json.dumps(q) + "\n")
print(f"wrote {sum(n.values())} tip requests {dict(n)}", file=sys.stderr)
