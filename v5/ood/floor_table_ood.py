"""floor8k minus tip (E path, same rows and samples), per set and macro, picture-clustered se; splits single/multi image, fired/none;
prompt and generated tokens per read; also floor8k minus W-null (W path) where xfer_wnull_s0/s1 exist."""
import collections, json, math, os
V4 = "/data/fd/mingdazhang/selfedit/v4"
rows = {r["id"]: r for r in map(json.loads, open(f"{V4}/rows/learn.jsonl")) if r.get("v4_role") == "ood_v2"}
fired = json.load(open(f"{V4}/scratch/ext/ood_fired.json"))
def load(*bs):
    o = {}
    for b in bs:
        p = f"{V4}/q/out/{b}.jsonl"
        if not os.path.exists(p): continue
        for l in open(p):
            x = json.loads(l); k = x["key"].split("|")
            if not x.get("error") and x.get("content") is not None: o[(k[-2], int(k[-1]))] = x
    return o
F, T, W = load("ood_floor8k"), load("ood_tipE"), load("ood_wnull_s0", "ood_wnull_s1")
nim = lambda r: len(r['image_paths']) if isinstance(r.get('image_paths'), list) else 1
ok = lambda x: float(bool(x.get("ok")))
def cl(pairs):
    byr = collections.defaultdict(list)
    for (r, s), d in pairs: byr[r].append(d)
    if len(byr) < 2: return None
    v = {r: sum(x) / len(x) for r, x in byr.items()}; n = len(v); m = sum(v.values()) / n; byc = collections.defaultdict(float)
    for r, d in v.items(): byc[rows[r].get("pic") or r] += d - m
    G = len(byc); se = math.sqrt(sum(x * x for x in byc.values()) * G / max(G - 1, 1)) / n
    return m, se, n
fm = lambda t: "n/a" if not t else f"{100*t[0]:+.1f}±{100*t[1]:.1f} ({t[2]})"
def line(lab, ks):
    kt = [k for k in ks if k in F and k in T]; kw = [k for k in ks if k in F and k in W]
    if not kt: return f"{lab}: n/a"
    tok = lambda D, K, f: sum(D[k][f] for k in K) / len(K)
    return (f"{lab}: floor8k {100*sum(ok(F[k]) for k in kt)/len(kt):.1f} tipE {100*sum(ok(T[k]) for k in kt)/len(kt):.1f} | floor8k-tipE {fm(cl([(k, ok(F[k]) - ok(T[k])) for k in kt]))}"
            f" | floor8k-W-null(W path) {fm(cl([(k, ok(F[k]) - ok(W[k])) for k in kw]))} | prompt tok {tok(T, kt, 'prompt_tokens'):.0f}->{tok(F, kt, 'prompt_tokens'):.0f}"
            f" | gen tok {tok(T, kt, 'tokens'):.0f}->{tok(F, kt, 'tokens'):.0f} | cuts {sum(T[k].get('finish') == 'length' for k in kt)}->{sum(F[k].get('finish') == 'length' for k in kt)}")
print(f"reads floor8k {len(F)} tipE {len(T)} W-null(s0-1) {len(W)}")
allk = sorted(set(F) | set(T)); sets = sorted({r["bench"] for r in rows.values()}); res = {}
for b in sets + ["ALL"]:
    ks = [k for k in allk if b == "ALL" or rows[k[0]]["bench"] == b]
    print(line(f"{b}", ks))
    for lab, f_ in (("single-image", lambda k: nim(rows[k[0]]) == 1), ("multi-image", lambda k: nim(rows[k[0]]) > 1),
                    ("procedure fired", lambda k: bool(fired.get(k[0]))), ("none fired", lambda k: not fired.get(k[0]))):
        sub = [k for k in ks if f_(k)]
        if sub: print("    " + line(lab, sub))
    t = cl([(k, ok(F[k]) - ok(T[k])) for k in ks if k in F and k in T]); res[b] = t
v = [res[b] for b in sets if res.get(b)]
if v: print(f"MACRO floor8k-tipE: {100*sum(x[0] for x in v)/len(v):+.1f}±{100*math.sqrt(sum(x[1]**2 for x in v))/len(v):.1f} over {len(v)} sets")
