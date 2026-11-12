"""Build tip-view extend/W batches. python3 mk_ext.py TIP ROLES SAMPLES --name BATCH --ops JSON --prio P [--engine E] [--edit FILE] [--alpha A]
[--rows-shard i/n] [--limit N] [--check]. Base key = the tip's own key for (row, sample): sig found from any q/in request with the same row, prefix,
settings and ops as the tip trace (runs/main_1110/signal/<tip>/traces.json; JUDGE rows from the tip's own requests). Writes q/in/<BATCH>.jsonl
(+ q/prio/<BATCH>.json)."""
import argparse, glob, hashlib, json, os, sys, time, collections
V4 = "/data/fd/mingdazhang/selfedit/v4"; RUN = f"{V4}/runs/main_1110"
ap = argparse.ArgumentParser()
ap.add_argument("tip"); ap.add_argument("roles"); ap.add_argument("samples")
ap.add_argument("--name", required=True); ap.add_argument("--ops", default="null"); ap.add_argument("--prio", type=int, default=-3)
ap.add_argument("--engine", default="E"); ap.add_argument("--edit"); ap.add_argument("--alpha", type=float)
ap.add_argument("--rows-shard", default="0/1"); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--check", action="store_true")
ap.add_argument("--tag", default=None)
a = ap.parse_args()
roles = set(a.roles.split(",")); samples = [int(x) for x in a.samples.split(",")]
rows = {}
for l in open(f"{V4}/rows/learn.jsonl"):
    d = json.loads(l)
    if d.get("v4_role") in roles:
        rows[d["id"]] = d
canon = lambda o: json.dumps(o, sort_keys=True)
# tip view per row: from all q/in requests, the sig whose requests match the tip's per-row prefix/settings/ops
tr = json.load(open(f"{RUN}/signal/{a.tip}/traces.json"))
want = {}
for rid, t in tr.items():
    want[rid] = canon([t.get("prefix") or None, t.get("settings") or None, t.get("ops") or None, t.get("edit") or None])
done = set()
for f in glob.glob(f"{V4}/q/out/*.jsonl"):
    pass
sig_of, req_of = {}, {}
for f in sorted(glob.glob(f"{V4}/q/in/g00*.jsonl")):
    for l in open(f):
        r = json.loads(l)
        if r.get("mode", "read") != "read" or r.get("row") not in rows or r.get("base_key"):
            continue
        c = canon([r.get("prefix") or None, r.get("settings") or None, r.get("ops") or None, r.get("edit") or None])
        sig = r["key"].split("|")[1]
        if r["row"] in want and c == want[r["row"]]:
            sig_of.setdefault(r["row"], set()).add(sig); req_of[r["row"]] = r
missing = [x for x in rows if x not in sig_of]
multi = {k: v for k, v in sig_of.items() if len(v) > 1}
print(f"rows {len(rows)}; tip view found {len(sig_of)}; missing {len(missing)} {missing[:5]}; multi-sig {len(multi)}", file=sys.stderr)
# base results present?
keys_needed = {f"x|{sorted(sig_of[x])[0]}|{x}|{s}" for x in sig_of for s in samples}
have = set()
for f in glob.glob(f"{V4}/q/out/*.jsonl"):
    for l in open(f, "rb"):
        if not l.startswith(b'{"key": "x|'):
            continue
        k = l[9:l.index(b'"', 9)].decode()
        if k in keys_needed and b'"content"' in l and b'"error"' not in l[:400]:
            have.add(k)
print(f"base keys needed {len(keys_needed)}, cached {len(have)}", file=sys.stderr)
if a.check:
    sys.exit()
i, n = (int(x) for x in a.rows_shard.split("/"))
ops = json.loads(a.ops)
edit = None
if a.edit:
    edit = json.loads(a.edit)        # edit form as JSON, e.g. {"op": "surgeon", "file": ".../XB_bias_all_+1.json"}
tag = a.tag or a.name
sigx = hashlib.sha256(canon([tag, ops, edit]).encode()).hexdigest()[:16]
out = []
for j, x in enumerate(sorted(sig_of)):
    if j % n != i:
        continue
    base = req_of[x]
    for s in samples:
        bk = f"x|{sorted(sig_of[x])[0]}|{x}|{s}"
        r = dict(key=f"{tag}|{sigx}|{x}|{s}", mode="read", engine=a.engine, priority=a.prio, row=x, sample=s,
                 prefix=base.get("prefix"), settings=base.get("settings"))
        if base.get("ops") or ops:
            r["ops"] = (base.get("ops") or []) + (ops or [])
        if ops:
            r["base_key"] = bk
        if edit is not None:
            r["edit"] = edit
        r["base_ref"] = bk
        out.append(r)
if a.limit:
    out = out[:a.limit]
with open(f"{V4}/q/in/{a.name}.jsonl.tmp", "w") as fo:
    for r in out:
        fo.write(json.dumps(r) + "\n")
os.replace(f"{V4}/q/in/{a.name}.jsonl.tmp", f"{V4}/q/in/{a.name}.jsonl")
print(f"wrote {len(out)} requests to q/in/{a.name}.jsonl", file=sys.stderr)
