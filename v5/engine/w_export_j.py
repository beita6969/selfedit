"""JUDGE-row per-read tables (Builder request 11-11): python3 w_export_j.py OUTDIR BATCH... -> OUTDIR/<batch>.tsv
usual columns + pred (ecomp.pred = grade_mm.correct(r)[1] unless the record carries pred), base_pred, base_row_mean / base_row_n (t007c's cached
mean ok over all its cached samples 0-7 on the row: scratch/ext/tip_judge8.jsonl), t_submit (scratch/ext/submit_times.jsonl, latest per batch)."""
import glob, json, os, sys
V4 = "/data/fd/mingdazhang/selfedit/v4"
sys.path.insert(0, "/workspace/fudan/mingdazhang/selfedit-v4/loop")
import ecomp
od = sys.argv[1]; os.makedirs(od, exist_ok=True); bats = sys.argv[2:]
rows = {json.loads(l)["id"]: json.loads(l) for l in open(f"{V4}/rows/learn.jsonl")}
ts = {}
for l in open(f"{V4}/scratch/ext/submit_times.jsonl"):
    d = json.loads(l); ts[d["batch"]] = d["t_submit"]
t8 = [json.loads(l) for l in open(f"{V4}/scratch/ext/tip_judge8.jsonl")]
reqs = {}
for b in bats:
    for l in open(f"{V4}/q/in/{b}.jsonl"):
        q = json.loads(l); q["_b"] = b; reqs[q["key"]] = q
need = {q["base_ref"] for q in reqs.values()} | {q["key"] for q in t8}; base = {}
for f in glob.glob(f"{V4}/q/out/*.jsonl"):
    for l in open(f, "rb"):
        if l.startswith(b'{"key": "x|') or l.startswith(b'{"key": "p000|'):
            k = l[9:l.index(b'"', 9)].decode()
            if k in need:
                x = json.loads(l)
                if not x.get("error") and x.get("content") is not None:
                    base[k] = x
rm = {}
for q in t8:
    if q["key"] in base:
        rm.setdefault(q["row"], []).append(float(bool(base[q["key"]].get("ok"))))
for b in bats:
    res = {}
    if os.path.exists(f"{V4}/q/out/{b}.jsonl"):
        for l in open(f"{V4}/q/out/{b}.jsonl"):
            x = json.loads(l); res[x["key"]] = x
    with open(f"{od}/{b}.tsv", "w") as fo:
        fo.write("batch\tcell\trow\tsample\tok\ttokens\tfinish\tloop\tprompt_tokens\terror\tbase_key\tbase_ok\tbase_tokens\tbase_finish\tpred\tbase_pred\tbase_row_mean\tbase_row_n\tt_submit\n")
        for k, q in reqs.items():
            if q["_b"] != b:
                continue
            x = res.get(k, {}); bb = base.get(q["base_ref"], {}); row = rows[q["row"]]
            e = q.get("edit")
            cell = os.path.basename(e.get("file", "")).replace(".json", "") if isinstance(e, dict) and e else ("W-null" if not e else "list")
            pr = "" if not x or x.get("error") else ecomp.pred(x, row)
            bp = ecomp.pred(bb, row) if bb else ""
            m = rm.get(q["row"], [])
            fo.write("\t".join("" if v is None else str(v) for v in [b, cell, q["row"], q["sample"], int(bool(x.get("ok"))), x.get("tokens"), x.get("finish"), int(bool(x.get("loop"))),
                     x.get("prompt_tokens"), (x.get("error") or "")[:80].replace("\t", " "), q["base_ref"], int(bool(bb.get("ok"))) if bb else "", bb.get("tokens", ""),
                     bb.get("finish", ""), pr, bp, f"{sum(m)/len(m):.4f}" if m else "", len(m), ts.get(b, "")]) + "\n")
    print(b, len(res), "->", f"{od}/{b}.tsv")
