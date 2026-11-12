"""Zoom pilot table: zoom - tip (paired by row, sample; picture-clustered se) per bench + macro (+/- benches, bench-level se); by box area
(< 5 %, 5-20 %, 20-45 %, >= 45 %, no zoom); tokens of call 1 / call 2 and total as a multiple of one tip read; no-zoom share and reasons.
usage: zoom_table.py ANSNAME LOCNAME TIPBATCH"""
import collections, json, math, sys
V4 = "/data/fd/mingdazhang/selfedit/v4"; Z = f"{V4}/scratch/zoom"
ans, locn, tipb = sys.argv[1:4]
rows = {r["id"]: r for r in map(json.loads, open(f"{V4}/rows/learn.jsonl"))}
def load(b):
    o = {}
    for l in open(f"{V4}/q/out/{b}.jsonl"):
        x = json.loads(l)
        if not x.get("error") and (x.get("content") is not None or x.get("mode") == "read"): k = x["key"].split("|"); o[(k[-2], int(k[-1]))] = x
    return o
A, L, T = load(ans), load(locn), load(tipb)
M = {(m["row"], m["sample"]): m for m in map(json.loads, open(f"{Z}/{ans}_meta.jsonl"))}
ok = lambda x: float(bool(x.get("ok")))
def cl(pairs):
    byr = collections.defaultdict(list)
    for (r, s), d in pairs: byr[r].append(d)
    if len(byr) < 2: return None
    v = {r: sum(x) / len(x) for r, x in byr.items()}; n = len(v); m = sum(v.values()) / n; byc = collections.defaultdict(float)
    for r, d in v.items(): byc[rows[r].get("pic") or r] += d - m
    G = len(byc); return m, math.sqrt(sum(x * x for x in byc.values()) * G / max(G - 1, 1)) / n, n
fm = lambda t: "n/a" if not t else f"{100*t[0]:+.1f}±{100*t[1]:.1f} (n={t[2]})"
keys = sorted(k for k in A if k in T and k in M)
print(f"{ans}: zoom reads {len(A)} tip {len(T)} paired {len(keys)}; zoomed {sum(M[k]['zoom'] for k in keys)}; no zoom {dict(collections.Counter(M[k]['why'] for k in keys if not M[k]['zoom']))}")
def line(lab, ks):
    if not ks: return None
    t = cl([(k, ok(A[k]) - ok(T[k])) for k in ks]); av = lambda D, f: sum((D[k].get(f) or 0) for k in ks if k in D) / len(ks)
    tipt = av(T, "tokens") + av(T, "prompt_tokens"); tot = av(L, "tokens") + av(L, "prompt_tokens") + av(A, "tokens") + av(A, "prompt_tokens")
    print(f"{lab}: tip {100*sum(ok(T[k]) for k in ks)/len(ks):.1f} zoom {100*sum(ok(A[k]) for k in ks)/len(ks):.1f} | zoom-tip {fm(t)} | call1 prompt/gen {av(L,'prompt_tokens'):.0f}/{av(L,'tokens'):.0f}"
          f" | call2 prompt/gen {av(A,'prompt_tokens'):.0f}/{av(A,'tokens'):.0f} | tip prompt/gen {av(T,'prompt_tokens'):.0f}/{av(T,'tokens'):.0f} | total x{tot/max(tipt,1):.2f} (generated only x{(av(L,'tokens')+av(A,'tokens'))/max(av(T,'tokens'),1):.2f})"
          f" | zoomed {sum(M[k]['zoom'] for k in ks)}/{len(ks)} | cuts {sum(A[k].get('finish')=='length' for k in ks)} vs {sum(T[k].get('finish')=='length' for k in ks)}")
    return t
res = {}
for b in sorted({rows[k[0]]["bench"] for k in keys}): res[b] = line(b, [k for k in keys if rows[k[0]]["bench"] == b])
line("POOLED", keys)
v = [t[0] for t in res.values() if t]
if v:
    m = sum(v) / len(v); sd = math.sqrt(sum((x - m) ** 2 for x in v) / max(len(v) - 1, 1))
    print(f"MACRO zoom-tip {100*m:+.1f} (bench-level se {100*sd/math.sqrt(len(v)):.1f}); benches + {sum(x>0 for x in v)} / - {sum(x<0 for x in v)} / 0 {sum(x==0 for x in v)}")
for lab, f in (("area < 5%", lambda m: m["zoom"] and m["area"] < .05), ("area 5-20%", lambda m: m["zoom"] and .05 <= m["area"] < .2),
               ("area 20-45%", lambda m: m["zoom"] and .2 <= m["area"] < .45), ("area >= 45%", lambda m: m["zoom"] and m["area"] >= .45), ("no zoom", lambda m: not m["zoom"])):
    line(lab, [k for k in keys if f(M[k])])
line("single-image", [k for k in keys if not isinstance(rows[k[0]]["image_paths"], list) or len(rows[k[0]]["image_paths"]) == 1])
line("multi-image", [k for k in keys if isinstance(rows[k[0]]["image_paths"], list) and len(rows[k[0]]["image_paths"]) > 1])

# ---- equal-token comparison on the SAME rows from the tip's cached 8 samples (signal side only: scratch/ext/tip_signal8.jsonl)
import glob, os
if len(sys.argv) > 4 and sys.argv[4] == "--vote8":
    t8 = [json.loads(l) for l in open(f"{V4}/scratch/ext/tip_signal8.jsonl")]; need = {q["key"] for q in t8}; got = {}
    for f in glob.glob(f"{V4}/q/out/*.jsonl"):
        for l in open(f, "rb"):
            if l.startswith(b'{"key": "x|') or l.startswith(b'{"key": "p000|'):
                k = l[9:l.index(b'"', 9)].decode()
                if k in need:
                    x = json.loads(l)
                    if not x.get("error") and x.get("content") is not None: got[k] = x
    S = collections.defaultdict(dict)
    for q in t8:
        if q["key"] in got: S[q["row"]][q["sample"]] = got[q["key"]]
    rs = sorted({k[0] for k in keys} & {r for r, d in S.items() if len(d) == 8})
    cost = lambda x: (x.get("tokens") or 0) + (x.get("prompt_tokens") or 0)
    def vote(seq, stop2):
        c, okd, used = collections.Counter(), {}, 0
        for x in seq:
            used += cost(x); p = str(x.get("pred")); c[p] += 1; okd.setdefault(p, ok(x))
            if stop2 and c[p] >= 2: return okd[p], used
        top = max(c.values()); win = next(str(x.get("pred")) for x in seq if c[str(x.get("pred"))] == top)   # ties -> earliest
        return okd[win], used
    one = sum(cost(S[r][s]) for r in rs for s in range(8)) / (8 * len(rs))
    print(f"\nequal-token comparison on {len(rs)} rows with 8 cached tip samples (cyclic starts, all 8 averaged); one tip read = {one:.0f} tokens (prompt+generated)")
    for lab, m, stop2 in [("tip single read", 1, False), ("plurality of 2", 2, False), ("plurality of 3", 3, False), ("plurality of 4", 4, False),
                          ("early-stop vote (stop when 2 agree, max 8)", 8, True), ("early-stop vote (stop when 2 agree, max 4)", 4, True)]:
        a = u = 0
        for r in rs:
            for st in range(8):
                seq = [S[r][(st + i) % 8] for i in range(m)]; o, used = vote(seq, stop2); a += o; u += used
        n = 8 * len(rs); print(f"  {lab}: acc {100*a/n:.1f} at x{u/n/one:.2f}")
    zk = [k for k in keys if k[0] in rs]
    av = lambda D, f: sum((D[k].get(f) or 0) for k in zk if k in D) / len(zk)
    print(f"  zoom (same rows, s0-1): acc {100*sum(ok(A[k]) for k in zk)/len(zk):.1f} at x{(av(L,'tokens')+av(L,'prompt_tokens')+av(A,'tokens')+av(A,'prompt_tokens'))/one:.2f}; tip s0-1 on these rows {100*sum(ok(T[k]) for k in zk)/len(zk):.1f}")
