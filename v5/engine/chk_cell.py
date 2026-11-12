"""5-row check of a W cell batch vs the tip's cached same-index reads: python3 chk_cell.py BATCH"""
import glob, json, sys, statistics
V4 = "/data/fd/mingdazhang/selfedit/v4"; b = sys.argv[1]
reqs = [json.loads(l) for l in open(f"{V4}/q/in/{b}.jsonl")]
res = {json.loads(l)["key"]: json.loads(l) for l in open(f"{V4}/q/out/{b}.jsonl")}
need = {q["base_ref"] for q in reqs}; base = {}
for f in glob.glob(f"{V4}/q/out/*.jsonl"):
    for l in open(f, "rb"):
        if l.startswith(b'{"key": "x|') or l.startswith(b'{"key": "p000|'):
            k = l[9:l.index(b'"', 9)].decode()
            if k in need:
                x = json.loads(l)
                if not x.get("error"):
                    base[k] = x
rt = []
for q in reqs:
    x = res.get(q["key"], {}); bb = base.get(q["base_ref"], {})
    r = (x.get("tokens") or 0) / max(1, bb.get("tokens") or 1); rt.append(r)
    print(f"{q['row'][:44]:44s} prompt {bb.get('prompt_tokens')} -> {x.get('prompt_tokens')}  tokens {bb.get('tokens')} -> {x.get('tokens')} (x{r:.2f})  "
          f"finish {x.get('finish')}  loop {x.get('loop')}  ok {bb.get('ok')} -> {x.get('ok')}  {(x.get('error') or '')[:100]}")
print(f"mean token ratio x{statistics.mean(rt):.2f}, median x{statistics.median(rt):.2f}; prompt_tokens identical {sum(1 for q in reqs if res.get(q['key'],{}).get('prompt_tokens') == base.get(q['base_ref'],{}).get('prompt_tokens'))}/{len(reqs)}")
