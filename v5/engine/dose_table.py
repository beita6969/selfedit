"""FL dose series on the 324 signal rows, three references (row-clustered se): (a) cell - tip cached 8-sample row mean; (b) cell - W-null on the
same (row, sample) [wnull_sig_s0,s1,s23]; (c) cell - mean(RANDFL, ROTFL) on the same (row, sample) [randfl_s0-3, rotfl_s0-3].
Per benchmark and ALL; tokens vs W-null, cuts, reads > 16k. usage: dose_table.py NAME=b1,b2 ..."""
import collections, glob, json, math, os, statistics, sys
V4 = "/data/fd/mingdazhang/selfedit/v4"
rows = {r["id"]: r for r in map(json.loads, open(f"{V4}/rows/learn.jsonl"))}
def load(bs):
    req, res = {}, {}
    for b in bs.split(","):
        p = f"{V4}/q/in/{b}.jsonl"; p = p if os.path.exists(p) else f"{V4}/q/hold/{b}.jsonl"
        for l in open(p): q = json.loads(l); req[(q["row"], q["sample"])] = q["key"]
        for l in open(f"{V4}/q/out/{b}.jsonl"): x = json.loads(l); res[x["key"]] = x
    return {rs: res[k] for rs, k in req.items() if k in res and not res[k].get("error") and res[k].get("content") is not None}
W = load("wnull_sig_s0,wnull_sig_s1,wnull_sig_s23"); RF = load("randfl_s0,randfl_s1,randfl_s2,randfl_s3"); RO = load("rotfl_s0,rotfl_s1,rotfl_s2,rotfl_s3")
t8 = [json.loads(l) for l in open(f"{V4}/scratch/ext/tip_signal8.jsonl")]; need = {q["key"] for q in t8}; tip = {}
for f in glob.glob(f"{V4}/q/out/*.jsonl"):
    for l in open(f, "rb"):
        if l.startswith(b'{"key": "x|') or l.startswith(b'{"key": "p000|'):
            k = l[9:l.index(b'"', 9)].decode()
            if k in need:
                x = json.loads(l)
                if not x.get("error") and x.get("content") is not None: tip[k] = x
m8 = collections.defaultdict(list)
for q in t8:
    if q["key"] in tip: m8[q["row"]].append(float(bool(tip[q["key"]].get("ok"))))
ok = lambda x: float(bool(x.get("ok")))
def st(diffs):
    byr = collections.defaultdict(list)
    for (r, s), d in diffs: byr[r].append(d)
    if len(byr) < 2: return "n/a"
    v = [sum(x) / len(x) for x in byr.values()]; m = sum(v) / len(v); se = statistics.pstdev(v) / math.sqrt(len(v) - 1)
    return f"{m*100:+5.2f}±{se*100:4.2f}"
print("cell | set | reads (samples) | acc | (a) vs tip mean8 | (b) vs W-null same (row,s) | (c) vs mean(RANDFL,ROTFL) same (row,s) | tok vs W-null | cuts (W-null) | >16k (W-null)")
for spec in sys.argv[1:]:
    name, bs = spec.split("=", 1); C = load(bs)
    for b in sorted({rows[r]["bench"] for r, _ in C}) + ["ALL"]:
        ks = sorted(k for k in C if b == "ALL" or rows[k[0]]["bench"] == b); sm = sorted({s for _, s in ks})
        kw = [k for k in ks if k in W]; kc = [k for k in ks if k in RF and k in RO]
        tr = sum(C[k]["tokens"] for k in kw) / max(1, sum(W[k]["tokens"] for k in kw))
        print(f"{name} | {b} | {len(ks)} (s{''.join(map(str, sm))}) | {sum(ok(C[k]) for k in ks)/len(ks):.3f} | {st([(k, ok(C[k]) - sum(m8[k[0]])/len(m8[k[0]])) for k in ks if m8[k[0]]])}"
              f" | {st([(k, ok(C[k]) - ok(W[k])) for k in kw])} [{len(kw)}] | {st([(k, ok(C[k]) - (ok(RF[k]) + ok(RO[k])) / 2) for k in kc])} [{len(kc)}] | x{tr:.2f}"
              f" | {sum(C[k].get('finish') == 'length' for k in ks)} ({sum(W[k].get('finish') == 'length' for k in kw)}) | {sum(C[k]['tokens'] > 16000 for k in ks)} ({sum(W[k]['tokens'] > 16000 for k in kw)})")
