"""Leave-one-entry-out table: per entry/group k and side (OOD v2 / learn): rows where k routed, tip acc there, tip - wo_k (paired by row and
sample, picture-clustered se), per bench where n >= 15 rows, tokens with/without; summary = contribution x routed share of the whole side."""
import collections, glob, json, math, os
V4 = "/data/fd/mingdazhang/selfedit/v4"; X = f"{V4}/scratch/ext"
rows = {r["id"]: r for r in map(json.loads, open(f"{V4}/rows/learn.jsonl"))}
ok = lambda x: float(bool(x.get("ok")))
def cl(pairs):
    byr = collections.defaultdict(list)
    for (r, s), d in pairs: byr[r].append(d)
    if len(byr) < 2: return None
    v = {r: sum(x) / len(x) for r, x in byr.items()}; n = len(v); m = sum(v.values()) / n; byc = collections.defaultdict(float)
    for r, d in v.items(): byc[rows[r].get("pic") or r] += d - m
    G = len(byc); return m, math.sqrt(sum(x * x for x in byc.values()) * G / max(G - 1, 1)) / n, n
fm = lambda t: "n/a" if not t else f"{100*t[0]:+.1f}±{100*t[1]:.1f} (n={t[2]})"
def res(b):
    o = {}
    p = f"{V4}/q/out/{b}.jsonl"
    if os.path.exists(p):
        for l in open(p):
            x = json.loads(l)
            if not x.get("error") and x.get("content") is not None: o[x["key"]] = x
    return o
# tip references: OOD = ood_tipE (keys otip|<tipkey>); learn = cached tip reads by their own keys (lrn_tip_ref.json)
OT = {k[5:]: x for k, x in res("ood_tipE").items()}
need = set(json.load(open(f"{X}/lrn_tip_ref.json"))["have"]); LT = {}
for f in glob.glob(f"{V4}/q/out/*.jsonl"):
    for l in open(f, "rb"):
        if l.startswith(b'{"key": "x|'):
            k = l[9:l.index(b'"', 9)].decode()
            if k in need:
                x = json.loads(l)
                if not x.get("error") and x.get("content") is not None: LT[k] = x
NSIDE = {"ood": sum(r.get("v4_role") == "ood_v2" for r in rows.values()), "lrn": sum(r.get("v4_role") in ("show", "monitor", "judge") for r in rows.values())}
# route confidence p(yes) per (cond, row): last rt1 answer per key, as tipview.py
PY = {}
for f in glob.glob(f"{V4}/q/out/*.jsonl"):
    for l in open(f, "rb"):
        if l.startswith(b'{"key": "rt1|'):
            x = json.loads(l); _, c, r = x["key"].split("|", 2)
            if not x.get("error"): PY[(c, r)] = (x.get("p"), x.get("p_src"))
TIPE = json.load(open(f"{V4}/runs/main_1110/state.json"))["tip"]["entries"]
GIX = {"E1": [1], "E2": [2], "E5": [5], "E6": [6], "E9": [9], "E11": [11], "H": [3, 4, 7, 8, 10]}
def pyes(k, r):   # group: highest p over its conditions (routed if any >= 0.5)
    v = [PY.get((TIPE[i]["cond"], r), (None, None)) for i in GIX[k]]
    if any(p is None for p, _ in v): return None, "missing"
    return max(p for p, _ in v), ",".join(sorted({src for _, src in v}))
qs = lambda v, f: sorted(v)[min(len(v) - 1, int(f * len(v)))]
summ = collections.defaultdict(dict); sharp = []
for side, TIP in (("ood", OT), ("lrn", LT)):
    print(f"\n===== {side} (whole side {NSIDE[side]} rows)")
    for k in ("E1", "E2", "E5", "E6", "E9", "E11", "H"):
        reqs = [json.loads(l) for l in open(f"{X}/wo_{side}_{k}.jsonl")]; W = res(f"{side}_wo_{k}"); tag = ("owo" if side == "ood" else "lwo") + k + "|"
        P = [((q["row"], q["sample"]), TIP[q["tip_key"]], W[tag + q["key"]]) for q in reqs if q["tip_key"] in TIP and tag + q["key"] in W]
        if not P: print(f"{k}: no paired reads yet ({len(reqs)} requested)"); continue
        def line(lab, PP):
            t = cl([(rs, ok(a) - ok(b)) for rs, a, b in PP]); n = len({rs[0] for rs, _, _ in PP})
            tk = lambda i, f: sum((p[i].get(f) or 0) for p in PP) / len(PP)
            print(f"{lab}: rows {n} tip acc {100*sum(ok(a) for _, a, _ in PP)/len(PP):.1f} wo acc {100*sum(ok(b) for _, _, b in PP)/len(PP):.1f} | tip-wo {fm(t)}"
                  f" | prompt tok {tk(1,'prompt_tokens'):.0f} vs {tk(2,'prompt_tokens'):.0f} | gen tok {tk(1,'tokens'):.0f} vs {tk(2,'tokens'):.0f}"
                  f" | cuts {sum(a.get('finish')=='length' for _, a, _ in PP)} vs {sum(b.get('finish')=='length' for _, _, b in PP)}")
            return t, n
        t, n = line(f"{k} [{len(P)}/{len(reqs)} reads]", P)
        byb = collections.defaultdict(list)
        for p in P: byb[rows[p[0][0]]["bench"]].append(p)
        for b, PP in sorted(byb.items()):
            if len({rs[0] for rs, _, _ in PP}) >= 15: line(f"   {b}", PP)
        if t: summ[k][side] = (t[0] * n / NSIDE[side], t[1] * n / NSIDE[side], n)
        # split by route confidence: terciles of p(yes) among the routed rows of this side
        pr = {r: pyes(k, r)[0] for r in {rs[0] for rs, _, _ in P}}; srcs = collections.Counter(pyes(k, r)[1] for r in pr)
        order = sorted(pr, key=lambda r: pr[r]); m = len(order); terc = {}
        for j, r in enumerate(order): terc[r] = min(2, 3 * j // max(m, 1))
        for ti in range(3):
            rr = [r for r in order if terc[r] == ti]
            if rr: line(f"   p(yes) tercile {ti+1} [{pr[rr[0]]:.4f}, {pr[rr[-1]]:.4f}]", [p for p in P if terc[p[0][0]] == ti])
        print(f"   p sources {dict(srcs)}")
        sideroles = ("ood_v2",) if side == "ood" else ("show", "monitor", "judge")
        allr = [r for r, d in rows.items() if d.get("v4_role") in sideroles]
        byb2 = collections.defaultdict(list)
        for r in pr: byb2[rows[r]["bench"]].append(pr[r])
        for b, v in sorted(byb2.items()):
            print(f"   p(yes) routed {b} n={len(v)}: min {min(v):.3f} q25 {qs(v,.25):.3f} med {qs(v,.5):.3f} q75 {qs(v,.75):.4f} max {max(v):.4f}")
        below = sorted(((pyes(k, r)[0], r) for r in allr if pyes(k, r)[0] is not None and pyes(k, r)[0] < 0.5), reverse=True)
        bands = [(0.2, 0.5), (0.05, 0.2), (0.01, 0.05)]
        print("   just below the decision (not routed): " + " | ".join(f"p in [{a},{b}): {sum(a <= p < b for p, _ in below)} " + str(dict(collections.Counter(rows[r]['bench'] for p, r in below if a <= p < b))) for a, b in bands))
        print("   top-10 not routed: " + ", ".join(f"{r}:{p:.3f}" for p, r in below[:10]))
        t3 = {ti: cl([(rs, ok(a) - ok(b)) for rs, a, b in P if terc[rs[0]] == ti]) for ti in range(3)}
        sharp.append((k, side, t3))
print("\n===== summary: contribution where routed x routed share = points of the whole side")
for k, d in summ.items():
    print(f"{k}: " + " | ".join(f"{s} {100*v[0]:+.2f}±{100*v[1]:.2f} pts ({v[2]} rows)" for s, v in d.items()))
print("\n===== stricter threshold? contribution by p(yes) tercile (low / mid / high)")
for k, side, t3 in sharp:
    print(f"{k} {side}: " + " / ".join(fm(t3[i]) for i in range(3)))
