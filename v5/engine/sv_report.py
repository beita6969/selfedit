"""short_vote unit-test report: python3 sv_report.py BATCH PARENT_INBATCH (the tip parent batch the test requests were copied from)
per row: op tokens, reads cut, votes, ok vs the tip's cached single reads (key x|TIP_SIG|row|s) mean tokens / mean ok."""
import glob, json, sys
V4 = "/data/fd/mingdazhang/selfedit/v4"
batch, pin = sys.argv[1], sys.argv[2]
SIG = {}
for l in open(f"{V4}/q/in/{pin}.jsonl"):
    d = json.loads(l); SIG[d["row"]] = d["key"].split("|")[1]
res = [json.loads(l) for l in open(f"{V4}/q/out/{batch}.jsonl")]
rows = {r["row"] for r in res}
tip = {}
for f in glob.glob(f"{V4}/q/out/*.jsonl"):
    with open(f) as fh:
        for l in fh:
            if '"key": "x|' not in l:
                continue
            d = json.loads(l)
            if d.get("row") in rows and not d.get("error") and d.get("finish") != "abort" and d["key"].startswith(f"x|{SIG.get(d['row'])}|{d['row']}|"):
                tip.setdefault(d["row"], {})[d["key"]] = d
nimg = {}
for l in open(f"{V4}/rows/learn.jsonl"):
    d = json.loads(l)
    if d["id"] in rows:
        nimg[d["id"]] = len(d.get("image_paths") or [])
T = dict(gopen=0, novote=0, n=0, tok=0, tipn=0, tiptok=0, reads=0, cut=0, ok=0, tipok=0, err=0)
print(f"{'row':60s} img  op_tok  tip_tok/read(n)  cut  preds -> ok | tip ok")
for r in sorted(res, key=lambda x: x["row"]):
    t = list(tip.get(r["row"], {}).values())
    tt = sum(x.get("tokens") or 0 for x in t) / max(1, len(t)); tok_ = sum(bool(x.get("ok")) for x in t) / max(1, len(t))
    if r.get("error") or r.get("op_skipped"):
        T["err"] += 1
        print(f"{r['row'][:60]:60s} {nimg.get(r['row'])}  ERROR {str(r.get('error') or r.get('op_skipped'))[:80]}")
        continue
    sv = next(o for o in r["op_trace"] if o["op"] == "short_vote")
    reads = sv["reads"]
    T["n"] += 1; T["tok"] += r["tokens"]; T["tipn"] += 1; T["tiptok"] += tt; T["reads"] += len(reads); T["cut"] += sum(x["cut"] for x in reads)
    T["ok"] += bool(r["ok"]); T["tipok"] += tok_
    T["gopen"] += sv.get("gate_open") is not False; T["novote"] += sum(x["pred"] is None for x in reads)
    print(f"{r['row'][:60]:60s} {nimg.get(r['row'])}  {r['tokens']:6d}  {tt:7.0f} ({len(t)})       {sum(x['cut'] for x in reads)}/{len(reads)}  "
          f"{[str(x['pred'])[:12] for x in reads]} -> {r['pred']!s:.12} {int(bool(r['ok']))} | {tok_:.2f}")
n = max(1, T["n"])
print(f"\nrows {T['n']} (errors {T['err']}); mean tokens per short_vote result {T['tok']/n:.0f} vs tip single read {T['tiptok']/n:.0f} "
      f"(ratio {T['tok']/max(1,T['tiptok']):.2f}); reads cut {T['cut']}/{T['reads']} ({T['cut']/max(1,T['reads']):.0%}); "
      f"gate open {T['gopen']}/{T['n']}; reads without a vote {T['novote']}/{T['reads']} ({T['novote']/max(1,T['reads']):.0%}); "
      f"accuracy short_vote {T['ok']/n:.2f} vs tip mean single-read {T['tipok']/n:.2f}")
