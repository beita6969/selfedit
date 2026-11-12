"""Per-row fired entries of the tip (loop code): python tip_fired.py OUT.json -> {"entries": [{i, condition, procedure_head, settings, ops}], "rows": {row: [i...]}}"""
import glob, json, sys
V4 = "/data/fd/mingdazhang/selfedit/v4"; RUN = f"{V4}/runs/main_1110"; LOOP = "/workspace/fudan/mingdazhang/selfedit-v4/loop"
sys.path.insert(0, LOOP)
import gen as G
st = json.load(open(f"{RUN}/state.json")); tip = st["tip"]
R = G.Run.__new__(G.Run)
rows = [json.loads(l)["id"] for l in open(f"{V4}/rows/learn.jsonl")]
ents = R.flat(tip["entries"]); conds = {e["cond"] for e in ents if e["cond"] != "always"}
R.routes, last = {}, {}
for f in glob.glob(f"{V4}/q/out/*.jsonl"):
    for l in open(f, "rb"):
        if l.startswith(b'{"key": "rt1|'):
            x = json.loads(l); last[x["key"]] = x
for k, x in last.items():
    _, c, r = k.split("|", 2)
    if c in conds:
        p = x.get("p"); R.routes[(c, r)] = bool(p >= 0.5) if isinstance(p, (int, float)) else bool(x.get("yes"))
out = dict(tip=tip["id"], entries=[dict(i=i, condition=e.get("condition"), procedure_head=(e.get("procedure") or "")[:90], settings=e.get("settings"), ops=e.get("ops"))
                                   for i, e in enumerate(ents)],
           rows={r: [i for i, e in enumerate(ents) if R.fires(e, r)] for r in rows})
json.dump(out, open(sys.argv[1], "w"), indent=1)
import collections
c = collections.Counter(i for v in out["rows"].values() for i in v)
for e in out["entries"]:
    print(e["i"], c[e["i"]], (e["condition"] or "")[:100])
