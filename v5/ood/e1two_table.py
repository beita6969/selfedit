"""PROBE e1two_probe table: e1two - tip and e1two - e1always (paired by row, sample; picture-clustered se), per bench + macro, split entry 1
routed in tip vs not, hallusionbench per category, benchmarks +/- with benchmark-level se, tokens; share of reads (reasoning + answer) matching
hint|trap|suggests that|red herring|boilerplate|generic instruction, per arm."""
import collections, json, math, os, re
V4 = "/data/fd/mingdazhang/selfedit/v4"; X = f"{V4}/scratch/ext"
rows = {r["id"]: r for r in map(json.loads, open(f"{V4}/rows/learn.jsonl"))}
HINT = re.compile(r"\bhint|\btrap|suggests that|red herring|boilerplate|generic instruction", re.I)
def load(b):
    o = {}
    p = f"{V4}/q/out/{b}.jsonl"
    if os.path.exists(p):
        for l in open(p):
            x = json.loads(l)
            if not x.get("error") and x.get("content") is not None: k = x["key"].split("|"); o[(k[-2], int(k[-1]))] = x
    return o
ok = lambda x: float(bool(x.get("ok")))
def cl(pairs):
    byr = collections.defaultdict(list)
    for (r, s), d in pairs: byr[r].append(d)
    if len(byr) < 2: return None
    v = {r: sum(x) / len(x) for r, x in byr.items()}; n = len(v); m = sum(v.values()) / n; byc = collections.defaultdict(float)
    for r, d in v.items(): byc[rows[r].get("pic") or r] += d - m
    G = len(byc); return m, math.sqrt(sum(x * x for x in byc.values()) * G / max(G - 1, 1)) / n, n
fm = lambda t: "n/a" if not t else f"{100*t[0]:+.1f}±{100*t[1]:.1f} (n={t[2]})"
def side(title, P, T, A, routed, hb=False):
    keys = sorted(set(P) & set(T) & set(A)); print(f"\n== {title}: probe {len(P)} tip {len(T)} e1always {len(A)} paired {len(keys)}")
    acc = lambda D, K: 100 * sum(ok(D[k]) for k in K) / len(K)
    tok = lambda D, K: sum(D[k].get("tokens") or 0 for k in K) / len(K)
    def line(lab, ks):
        if not ks: return None
        a, b = cl([(k, ok(P[k]) - ok(T[k])) for k in ks]), cl([(k, ok(P[k]) - ok(A[k])) for k in ks])
        print(f"{lab}: tip {acc(T, ks):.1f} e1always {acc(A, ks):.1f} e1two {acc(P, ks):.1f} | e1two-tip {fm(a)} | e1two-e1always {fm(b)} | gen tok {tok(T, ks):.0f}/{tok(A, ks):.0f}/{tok(P, ks):.0f}"
              f" | prompt tok {sum(P[k].get('prompt_tokens') or 0 for k in ks)/len(ks):.0f} | cuts {sum(D[k].get('finish') == 'length' for D in (T, A, P) for k in ks)}")
        return a, b
    res = {}
    for bn in sorted({rows[k[0]]["bench"] for k in keys}):
        ks = [k for k in keys if rows[k[0]]["bench"] == bn]; res[bn] = line(bn, ks)
        line("  e1 routed in tip", [k for k in ks if routed.get(k[0])]); line("  e1 not routed", [k for k in ks if not routed.get(k[0])])
        if hb and bn == "hallusionbench":
            for c in sorted({rows[k[0]]["category"] for k in ks}): line(f"    {c}", [k for k in ks if rows[k[0]]["category"] == c])
    line("ALL pooled", keys); line("  ALL e1 routed", [k for k in keys if routed.get(k[0])]); line("  ALL e1 not routed", [k for k in keys if not routed.get(k[0])])
    for j, lab in ((0, "e1two-tip"), (1, "e1two-e1always")):
        v = [t[j][0] for t in res.values() if t and t[j]]; m = sum(v) / len(v); sd = math.sqrt(sum((x - m) ** 2 for x in v) / max(len(v) - 1, 1))
        print(f"MACRO {lab}: {100*m:+.1f} (bench-level se {100*sd/math.sqrt(len(v)):.1f}; within-bench se {100*math.sqrt(sum(t[j][1]**2 for t in res.values() if t and t[j]))/len(v):.1f}); benches + {sum(x > 0 for x in v)} / - {sum(x < 0 for x in v)} / 0 {sum(x == 0 for x in v)}")
    for nm, D in (("tip", T), ("e1always", A), ("e1two", P)):
        h = [k for k in keys if HINT.search((D[k].get("reasoning") or "") + " " + (D[k].get("content") or ""))]
        print(f"hint/trap regex share {nm}: {100*len(h)/len(keys):.1f}% ({len(h)}/{len(keys)}); among e1-not-routed rows {100*sum(not routed.get(k[0]) for k in h)/max(1,sum(not routed.get(k[0]) for k in keys)):.1f}%")
side("OOD v2", load("ood_e1two_probe"), load("ood_tipE"), load("ood_e1always"), json.load(open(f"{X}/e1_routed_ood.json")), hb=True)
side("SIGNAL", load("sig_e1two_probe"), load("sig_tipE"), load("sig_e1always"), json.load(open(f"{X}/e1_routed_sig.json")))
