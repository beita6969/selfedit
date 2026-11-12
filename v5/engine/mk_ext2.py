"""Build candidate batches on the tip view: python3 mk_ext2.py TIPREQS.jsonl --name BATCH [--samples 0,1] [--ops JSON] [--edit JSON] [--engine E|W]
[--prio P] [--shard i/n] [--rows FILE] [--limit N] [--check]. Each request = the tip's own request (same prefix/settings/ops) + the candidate's ops/edit;
base_key / base_ref = the tip's key (ops: run_extend reads the cached base under it, or reads it fresh)."""
import argparse, glob, hashlib, json, os, sys, time
V4 = "/data/fd/mingdazhang/selfedit/v4"
ap = argparse.ArgumentParser()
ap.add_argument("tipreqs"); ap.add_argument("--name", required=True); ap.add_argument("--samples", default="0,1,2,3")
ap.add_argument("--ops", default="null"); ap.add_argument("--edit", default="null"); ap.add_argument("--engine", default="E")
ap.add_argument("--prio", type=int, default=-3); ap.add_argument("--shard", default="0/1"); ap.add_argument("--rows", default="")
ap.add_argument("--limit", type=int, default=0); ap.add_argument("--check", action="store_true"); ap.add_argument("--tag", default=""); ap.add_argument("--pin", default=""); ap.add_argument("--outdir", default="")
a = ap.parse_args()
samples = {int(x) for x in a.samples.split(",")}
tq = [json.loads(l) for l in open(a.tipreqs)]
tq = [q for q in tq if q["sample"] in samples]
if a.rows:
    R = json.load(open(a.rows))
    if R and isinstance(R[0], dict):  # (row, sample) pairs, e.g. long_rows.py output
        keep = {(d["row"], int(d["sample"])) for d in R}; tq = [q for q in tq if (q["row"], q["sample"]) in keep]
    else:
        keep = set(R); tq = [q for q in tq if q["row"] in keep]
need = {q["key"] for q in tq}; have = set()
for f in glob.glob(f"{V4}/q/out/*.jsonl"):
    for l in open(f, "rb"):
        if l.startswith(b'{"key": "x|') or l.startswith(b'{"key": "p000|'):
            k = l[9:l.index(b'"', 9)].decode()
            if k in need and b'"content": "' in l and b'"content": ""' not in l and b'"error"' not in l[:300]:
                have.add(k)
print(f"tip requests {len(tq)}; base cached {len(have & need)}; missing {len(need - have)}", file=sys.stderr)
if a.check:
    sys.exit()
ops, edit = json.loads(a.ops), json.loads(a.edit)
tag = a.tag or a.name.split("_")[0]
sigx = hashlib.sha256(json.dumps([ops, edit], sort_keys=True).encode()).hexdigest()[:16]
i, n = (int(x) for x in a.shard.split("/"))
rows = sorted({q["row"] for q in tq}); ri = {r: j for j, r in enumerate(rows)}
out = []
for q in tq:
    if ri[q["row"]] % n != i:
        continue
    r = dict(key=f"{tag}|{sigx}|{q['row']}|{q['sample']}", mode="read", engine=a.engine, priority=a.prio, row=q["row"], sample=q["sample"],
             prefix=q.get("prefix"), settings=q.get("settings"), base_ref=q["key"])
    o = (q.get("ops") or []) + (ops or [])
    if o:
        r["ops"] = o
    if ops:
        r["base_key"] = q["key"]
    if edit:
        r["edit"] = edit
    if a.pin:
        r["pin"] = a.pin; r["edit"] = edit
    out.append(r)
if a.limit:
    out = out[:a.limit]
p = f"{a.outdir or V4 + '/q/in'}/{a.name}.jsonl"
with open(p + ".tmp", "w") as fo:
    for r in out:
        fo.write(json.dumps(r) + "\n")
os.replace(p + ".tmp", p)
if not a.outdir:                             # submission time per batch (JUDGE tables carry it: t_submit)
    with open(f"{V4}/scratch/ext/submit_times.jsonl", "a") as fs:
        fs.write(json.dumps(dict(batch=a.name, t_submit=round(time.time(), 1), n=len(out))) + "\n")
print(f"wrote {len(out)} -> {p}", file=sys.stderr)
