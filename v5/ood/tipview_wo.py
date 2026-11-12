"""Leave-one-entry-out view: tip entries minus IDXS (comma list), ONLY on rows where any of IDXS is routed:
python tipview_wo.py TIP ROLES SAMPLES OUT.jsonl IDXS ROWS_OUT.json   (based on tipview.py). Tip view requests from the loop's own code (gen.py Run.read_req): python tipview.py TIP_ID ROLES SAMPLES OUT.jsonl
writes the tip's own read request per (row, sample) (key = the tip's key: x|sig|row|s, or p000|parent|row|s on stock rows)."""
import collections, glob, json, sys
V4 = "/data/fd/mingdazhang/selfedit/v4"; RUN = f"{V4}/runs/main_1110"; LOOP = "/workspace/fudan/mingdazhang/selfedit-v4/loop"
sys.path.insert(0, LOOP)
import gen as G
tip_id, roles, samples, out = sys.argv[1], set(sys.argv[2].split(",")), [int(x) for x in sys.argv[3].split(",")], sys.argv[4]
IDXS, ROWS_OUT = [int(x) for x in sys.argv[5].split(",")], sys.argv[6]
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
ents = tip["entries"]; assert all(ents[i]["cond"] != "always" and not ents[i].get("parts") for i in IDXS)
WO = [e for i, e in enumerate(ents) if i not in IDXS]
fired = sorted(r for r in rows if any(R.routes.get((ents[i]["cond"], r)) for i in IDXS))
json.dump(fired, open(ROWS_OUT, "w"))
print(f"wo {IDXS}: routed on {len(fired)}/{len(rows)} rows", file=sys.stderr)
n = collections.Counter(); tipkeys = []
with open(out, "w") as fo:
    for r in fired:
        for s in samples:
            q0, _ = R.read_req(ents, r, s, 0)
            q, sg = R.read_req(WO, r, s, 0)
            assert q["prefix"] != q0["prefix"] or any(ents[i].get("settings") for i in IDXS), (r, IDXS)
            assert all(ents[i]["procedure"] not in (q["prefix"] or "") for i in IDXS)
            q["tip_key"] = q0["key"]
            n[q["key"].split("|")[0]] += 1
            fo.write(json.dumps(q) + "\n")
print(f"wrote {sum(n.values())} wo requests {dict(n)}", file=sys.stderr)
