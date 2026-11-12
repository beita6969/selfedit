"""python3 long_rows.py OUT.json THRESH BATCH...: list (row, sample, tokens, ok, finish) for cell reads above THRESH tokens, + W-null tokens"""
import json, sys
V4 = "/data/fd/mingdazhang/selfedit/v4"; out, th, bs = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
ref = {}
for rb in ("wnull_sig_s0", "wnull_sig_s1"):
    for l in open(f"{V4}/q/out/{rb}.jsonl"):
        x = json.loads(l); k = x["key"].split("|")
        if not x.get("error"): ref[(k[2], int(k[3]))] = x
L = []
for b in bs:
    for l in open(f"{V4}/q/out/{b}.jsonl"):
        x = json.loads(l)
        if x.get("error"): continue
        k = x["key"].split("|"); row, s = k[2], int(k[3])
        if (x.get("tokens") or 0) > th:
            f = ref.get((row, s), {})
            L.append(dict(row=row, sample=s, tokens=x["tokens"], ok=int(bool(x.get("ok"))), finish=x.get("finish"),
                          wnull_tokens=f.get("tokens"), wnull_ok=int(bool(f.get("ok")))))
L.sort(key=lambda d: (d["row"], d["sample"]))
json.dump(L, open(out, "w"), indent=1)
rows = sorted({d["row"] for d in L}); bench = {}
for d in L: bench.setdefault(d["row"].split("/")[0], []).append(d)
print(f"{len(L)} reads on {len(rows)} rows above {th}; ok {sum(d['ok'] for d in L)}/{len(L)} (W-null same read ok {sum(d['wnull_ok'] for d in L)}/{len(L)}); "
      f"finish length {sum(d['finish']=='length' for d in L)}")
print("by bench: " + ", ".join(f"{k} {len(v)} reads ok {sum(d['ok'] for d in v)}" for k, v in sorted(bench.items())))
for d in L: print(f"{d['row']:28s} s{d['sample']} tok {d['tokens']:6d} ok {d['ok']} finish {d['finish']:6s} | W-null tok {d['wnull_tokens']} ok {d['wnull_ok']}")
