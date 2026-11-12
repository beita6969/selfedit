"""Entry-1-always arm tables. OOD v2: ood_e1always vs ood_tipE (paired, s0-1); signal: sig_e1always vs tip cached 8-sample row mean and vs
fresh sig_tipE (paired). Per bench + macro, picture-clustered se; split e1 already routed vs added by the arm; tokens; hallusionbench per category."""
import collections, glob, json, math, os
V4 = "/data/fd/mingdazhang/selfedit/v4"; X = f"{V4}/scratch/ext"
rows = {r["id"]: r for r in map(json.loads, open(f"{V4}/rows/learn.jsonl"))}
def load(b):
    o = {}
    p = f"{V4}/q/out/{b}.jsonl"
    if os.path.exists(p):
        for l in open(p):
            x = json.loads(l)
            if not x.get("error") and x.get("content") is not None: k = x["key"].split("|"); o[(k[-2], int(k[-1]))] = x
    return o
ok = lambda x: float(bool(x.get("ok")))
def cl(pairs):                                   # pairs: [((row, s), diff)] -> row means, clustered by picture
    byr = collections.defaultdict(list)
    for (r, s), d in pairs: byr[r].append(d)
    if len(byr) < 2: return None
    v = {r: sum(x) / len(x) for r, x in byr.items()}; n = len(v); m = sum(v.values()) / n; byc = collections.defaultdict(float)
    for r, d in v.items(): byc[rows[r].get("pic") or r] += d - m
    G = len(byc); se = math.sqrt(sum(x * x for x in byc.values()) * G / max(G - 1, 1)) / n
    return m, se, n
fm = lambda t: "n/a" if not t else f"{100*t[0]:+.1f}±{100*t[1]:.1f} (n={t[2]})"
acc = lambda D, K: 100 * sum(ok(D[k]) for k in K) / len(K) if K else float("nan")
tok = lambda D, K, f: sum(D[k].get(f) or 0 for k in K) / len(K) if K else float("nan")
def block(title, A, B, routed, ref8=None, groups=None):
    print(f"\n== {title}: arm reads {len(A)}, reference reads {len(B)}")
    keys = sorted(set(A) & set(B)); res = {}
    def line(lab, ks):
        if not ks: print(f"{lab}: n/a"); return None
        t = cl([(k, ok(A[k]) - ok(B[k])) for k in ks])
        s = f"{lab}: ref {acc(B, ks):.1f} arm {acc(A, ks):.1f} | arm-ref {fm(t)}"
        if ref8 is not None:
            t8 = cl([(k, ok(A[k]) - ref8[k[0]]) for k in ks if k[0] in ref8]); s += f" | arm-tip mean8 {fm(t8)}"
        s += (f" | prompt tok {tok(B, ks, 'prompt_tokens'):.0f}->{tok(A, ks, 'prompt_tokens'):.0f} | gen tok {tok(B, ks, 'tokens'):.0f}->{tok(A, ks, 'tokens'):.0f}"
              f" | cuts {sum(B[k].get('finish') == 'length' for k in ks)}->{sum(A[k].get('finish') == 'length' for k in ks)}")
        print(s); return t
    sets = sorted({rows[k[0]]["bench"] for k in keys})
    for b in sets:
        ks = [k for k in keys if rows[k[0]]["bench"] == b]; res[b] = line(b, ks)
        for lab, f in (("  e1 routed", lambda k: routed.get(k[0])), ("  e1 added", lambda k: not routed.get(k[0]))): line(lab, [k for k in ks if f(k)])
        if groups and b in groups:
            for c in sorted({rows[k[0]]["category"] for k in ks}):
                kc = [k for k in ks if rows[k[0]]["category"] == c]; line(f"  {c}", kc)
                for lab, f in (("    routed", lambda k: routed.get(k[0])), ("    added", lambda k: not routed.get(k[0]))): line(lab, [k for k in kc if f(k)])
    line("ALL (pooled)", keys)
    for lab, f in (("  ALL e1 routed", lambda k: routed.get(k[0])), ("  ALL e1 added", lambda k: not routed.get(k[0]))): line(lab, [k for k in keys if f(k)])
    v = [t for t in res.values() if t]
    if v: print(f"MACRO ({len(v)} sets) arm-ref {100*sum(t[0] for t in v)/len(v):+.1f}±{100*math.sqrt(sum(t[1]**2 for t in v))/len(v):.1f}")
ro = json.load(open(f"{X}/e1_routed_ood.json")) if os.path.exists(f"{X}/e1_routed_ood.json") else {}
block("OOD v2: ood_e1always vs ood_tipE (s0-1)", load("ood_e1always"), load("ood_tipE"), ro, groups={"hallusionbench"})
rs = json.load(open(f"{X}/e1_routed_sig.json")) if os.path.exists(f"{X}/e1_routed_sig.json") else {}
t8 = [json.loads(l) for l in open(f"{X}/tip_signal8.jsonl")]; need = {q["key"] for q in t8}; tip = {}
for f in glob.glob(f"{V4}/q/out/*.jsonl"):
    for l in open(f, "rb"):
        if l.startswith(b'{"key": "x|') or l.startswith(b'{"key": "p000|'):
            k = l[9:l.index(b'"', 9)].decode()
            if k in need:
                x = json.loads(l)
                if not x.get("error") and x.get("content") is not None: tip[k] = x
m8 = collections.defaultdict(list)
for q in t8:
    if q["key"] in tip: m8[q["row"]].append(ok(tip[q["key"]]))
ref8 = {r: sum(v) / len(v) for r, v in m8.items()}
print(f"\ntip mean8 rows {len(ref8)}")
block("SIGNAL: sig_e1always vs fresh sig_tipE (s0-1); last column vs tip cached mean8", load("sig_e1always"), load("sig_tipE"), rs, ref8=ref8)
