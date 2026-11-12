"""python3 chk_vs_ref.py BATCH REF_BATCH,REF_BATCH [N_SNIPPETS]: per row cell vs the W-null read of the same row and sample, + snippets"""
import json, statistics, sys
V4 = "/data/fd/mingdazhang/selfedit/v4"; b = sys.argv[1]; refs = sys.argv[2].split(","); ns = int(sys.argv[3]) if len(sys.argv) > 3 else 2
reqs = [json.loads(l) for l in open(f"{V4}/q/in/{b}.jsonl")]
res = {json.loads(l)["key"]: json.loads(l) for l in open(f"{V4}/q/out/{b}.jsonl")}
ref = {}
for rb in refs:
    for l in open(f"{V4}/q/out/{rb}.jsonl"):
        x = json.loads(l)
        if not x.get("error"):
            k = x["key"].split("|"); ref[(k[2], int(k[3]))] = x
rat = []
for q in reqs:
    x = res.get(q["key"], {}); f = ref.get((q["row"], q["sample"]), {})
    r = (x.get("tokens") or 0) / max(1, f.get("tokens") or 1); rat.append(r)
    print(f"{q['row'][:30]:30s} s{q['sample']} prompt {f.get('prompt_tokens')}->{x.get('prompt_tokens')} tokens {f.get('tokens')}->{x.get('tokens')} (x{r:.2f}) "
          f"ok {int(bool(f.get('ok')))}->{int(bool(x.get('ok')))} finish {x.get('finish')} loop {int(bool(x.get('loop')))} {(x.get('error') or '')[:80]}")
print(f"token ratio mean x{statistics.mean(rat):.2f} median x{statistics.median(rat):.2f}; prompt identical "
      f"{sum(1 for q in reqs if res.get(q['key'],{}).get('prompt_tokens') == ref.get((q['row'], q['sample']),{}).get('prompt_tokens'))}/{len(reqs)}; "
      f"acc control {sum(bool(ref.get((q['row'], q['sample']),{}).get('ok')) for q in reqs)}/{len(reqs)} cell {sum(bool(res.get(q['key'],{}).get('ok')) for q in reqs)}/{len(reqs)}")
for q in reqs[:ns]:
    x = res.get(q["key"], {}); f = ref.get((q["row"], q["sample"]), {})
    print(f"--- {q['row']} s{q['sample']}\n  control: {repr((f.get('reasoning') or '')[:200])}\n  cell:    {repr((x.get('reasoning') or '')[:200])}")
