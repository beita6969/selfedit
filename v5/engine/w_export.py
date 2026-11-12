"""Compact per-read table of W batches: python3 w_export.py OUTDIR BATCH...  -> OUTDIR/<batch>.tsv
columns: batch, cell (edit spec file), row, sample, ok, tokens, finish, loop, prompt_tokens, base_key, base_ok, base_tokens, base_finish"""
import glob, json, os, sys
V4 = "/data/fd/mingdazhang/selfedit/v4"
od = sys.argv[1]; os.makedirs(od, exist_ok=True)
bats = sys.argv[2:]
reqs = {}
for b in bats:
    for l in open(f"{V4}/q/in/{b}.jsonl" if os.path.exists(f"{V4}/q/in/{b}.jsonl") else f"{V4}/q/hold/{b}.jsonl"):
        q = json.loads(l); q["_b"] = b; reqs[q["key"]] = q
need = {q["base_ref"] for q in reqs.values()}; base = {}
for f in glob.glob(f"{V4}/q/out/*.jsonl"):
    for l in open(f, "rb"):
        if l.startswith(b'{"key": "x|') or l.startswith(b'{"key": "p000|'):
            k = l[9:l.index(b'"', 9)].decode()
            if k in need:
                x = json.loads(l)
                if not x.get("error") and x.get("content") is not None:
                    base[k] = x
for b in bats:
    res = {}
    for l in open(f"{V4}/q/out/{b}.jsonl"):
        x = json.loads(l); res[x["key"]] = x
    with open(f"{od}/{b}.tsv", "w") as fo:
        fo.write("batch\tcell\trow\tsample\tok\ttokens\tfinish\tloop\tprompt_tokens\terror\tbase_key\tbase_ok\tbase_tokens\tbase_finish\n")
        for k, q in reqs.items():
            if q["_b"] != b:
                continue
            x = res.get(k, {}); bb = base.get(q["base_ref"], {})
            e = q.get("edit") or {}
            cell = (os.path.basename(e.get("file", "")).replace(".json", "") if isinstance(e, dict) and e else ("W-null" if not e else "list"))
            fo.write("\t".join(str(v) for v in [b, cell, q["row"], q["sample"], int(bool(x.get("ok"))), x.get("tokens"), x.get("finish"), int(bool(x.get("loop"))),
                     x.get("prompt_tokens"), (x.get("error") or "")[:80].replace("\t", " "), q["base_ref"], int(bool(bb.get("ok"))) if bb else "", bb.get("tokens", ""), bb.get("finish", "")]) + "\n")
    print(b, len(res), "->", f"{od}/{b}.tsv")
