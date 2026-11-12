"""Route requests for OOD v2 rows from the loop's own builder (gen.Run.route_reqs) on tip t007c's entries: python3 mk_routes.py OUT.jsonl"""
import json, sys
V4 = "/data/fd/mingdazhang/selfedit/v4"; RUN = f"{V4}/runs/main_1110"; LOOP = "/workspace/fudan/mingdazhang/selfedit-v4/loop"
sys.path.insert(0, LOOP)
import gen as G
tip = json.load(open(f"{RUN}/state.json"))["tip"]; assert tip["id"] == "t007c", tip["id"]
R = G.Run.__new__(G.Run)
rows = sorted(json.loads(l)["id"] for l in open(f"{V4}/rows/learn.jsonl") if json.loads(l).get("v4_role") == "ood_v2")
reqs = [q for e in tip["entries"] for q in R.route_reqs(e, rows)]
for q in reqs: q["priority"] = -10
with open(sys.argv[1], "w") as f:
    for q in reqs: f.write(json.dumps(q) + "\n")
print(len(rows), "rows;", len(reqs), "route requests;", len({q["key"].split("|")[1] for q in reqs}), "conditions")
