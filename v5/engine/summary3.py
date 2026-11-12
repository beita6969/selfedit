"""python3 summary3.py [--pairs FILE] NAME=b1,b2 ...  Three references per cell, row-clustered se:
(a) cell - tip cached row mean (up to 8 samples); (b) cell - W-null on the same (row, sample) [W-null s0-3 = wnull_sig_s0,s1,s23];
(c) cell - mean(RAND1, RAND2) on the same (row, sample) (samples 0-1 only). Also tokens vs W-null, cuts, reads > 16k, gated (tip read > 1500) for (c).
--pairs: restrict to (row, sample) pairs (list of {row, sample})."""
import collections, glob, json, math, os, statistics, sys
V4 = "/data/fd/mingdazhang/selfedit/v4"
args = sys.argv[1:]; pairs = None; cand = None
if args and args[0] == "--cand":
    cand = args[1]; args = args[2:]
if args and args[0] == "--pairs":
    pairs = {(d["row"], int(d["sample"])) for d in json.load(open(args[1]))}; args = args[2:]
def load(bs):
    req, res = {}, {}
    for b in bs.split(","):
        p = f"{V4}/q/in/{b}.jsonl"; p = p if os.path.exists(p) else f"{V4}/q/hold/{b}.jsonl"
        for l in open(p):
            q = json.loads(l); req[(q["row"], q["sample"])] = q["key"]
        for l in open(f"{V4}/q/out/{b}.jsonl"):
            x = json.loads(l); res[x["key"]] = x
    out = {rs: res[k] for rs, k in req.items() if k in res and not res[k].get("error") and res[k].get("content") is not None}
    return {k: v for k, v in out.items() if pairs is None or k in pairs}
W = load("wnull_sig_s0,wnull_sig_s1,wnull_sig_s23"); CA = load(cand) if cand else {}; R1, R2 = load("rand1_all_p1"), load("rand2_all_p1")
t8 = [json.loads(l) for l in open(f"{V4}/scratch/ext/tip_signal8.jsonl")]; need = {q["key"] for q in t8}; tip = {}
for f in glob.glob(f"{V4}/q/out/*.jsonl"):
    for l in open(f, "rb"):
        if l.startswith(b'{"key": "x|') or l.startswith(b'{"key": "p000|'):
            k = l[9:l.index(b'"', 9)].decode()
            if k in need:
                x = json.loads(l)
                if not x.get("error") and x.get("content") is not None: tip[k] = x
m8 = collections.defaultdict(list); T = {}
for q in t8:
    if q["key"] in tip:
        m8[q["row"]].append(float(bool(tip[q["key"]].get("ok")))); T[(q["row"], q["sample"])] = tip[q["key"]]
ok = lambda x: float(bool(x.get("ok")))
def st(diffs):
    byr = collections.defaultdict(list)
    for (r, s), d in diffs: byr[r].append(d)
    if len(byr) < 2: return "n/a", None
    v = [sum(x) / len(x) for x in byr.values()]; m = sum(v) / len(v); se = statistics.pstdev(v) / math.sqrt(len(v) - 1)
    p = math.erfc(abs(m / se) / math.sqrt(2)) if se > 0 else 1.0
    return f"{m*100:+5.2f}±{se*100:4.2f} (p {p:.3f})", m
print("cell | reads (samples) | acc | (a) vs tip mean8 | (b) vs W-null same samples | (c) vs RAND mean [s0-1] | (c) gated tip>1500 | tok vs W-null | cuts (W-null) | >16k" + (" | (d) candidate - cell, paired" if cand else ""))
for spec in args:
    name, bs = spec.split("=", 1); C = load(bs); ks = sorted(C)
    sm = sorted({s for _, s in ks})
    a, _ = st([(k, ok(C[k]) - sum(m8[k[0]]) / len(m8[k[0]])) for k in ks if m8[k[0]]])
    kw = [k for k in ks if k in W]; b, _ = st([(k, ok(C[k]) - ok(W[k])) for k in kw])
    kr = [k for k in ks if k in R1 and k in R2]; c, _ = st([(k, ok(C[k]) - (ok(R1[k]) + ok(R2[k])) / 2) for k in kr])
    kg = [k for k in kr if k in T and (T[k].get("tokens") or 0) > 1500]; g, _ = st([(k, ok(C[k]) - (ok(R1[k]) + ok(R2[k])) / 2) for k in kg])
    tr = sum(C[k]["tokens"] for k in kw) / max(1, sum(W[k]["tokens"] for k in kw))
    cuts = sum(C[k].get("finish") == "length" for k in ks); wc = sum(W[k].get("finish") == "length" for k in kw)
    print(f"{name} | {len(ks)} (s{''.join(map(str, sm))}) | {sum(ok(C[k]) for k in ks)/len(ks):.3f} | {a} | {b} [{len(kw)}] | {c} [{len(kr)}] | {g} [{len(kg)}] | x{tr:.2f} | {cuts} ({wc}) | {sum(C[k]['tokens'] > 16000 for k in ks)}" + (f" | {st([(k, ok(CA[k]) - ok(C[k])) for k in ks if k in CA])[0]} [{sum(k in CA for k in ks)}]" if cand else ""))
