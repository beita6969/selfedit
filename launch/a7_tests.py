"""Amendment 7 CPU tests T1-T3 (fable_1109 §1.4) on cal 7 draws: T1 member-track null calibration, T2 power (+3/+5/+8 pp, Holm x3 / x6), T3 re-judge of the
replay stage-2 candidates (member track vs the pool verdict of cal1109c_10) with the family-wise sign-flip size of the Holm-over-3 rule (C3).
usage: env (cal 7 env, CAL_A5S=1) a7_tests.py OUT [T1,T2,T3]"""
import os, sys, json
CAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cal_1109c.py")
OUT = sys.argv[1]; WHICH = (sys.argv[2] if len(sys.argv) > 2 else "T1,T2,T3").split(",")
g = dict(__file__=CAL, __name__="cal_setup"); sys.argv = [CAL, OUT]
exec(compile(open(CAL).read().split("TB, OUTR = [], {}")[0], CAL, "exec"), g)
H, data, say, NULLS, MEM, RES, FOURM, W = (g[k] for k in ("H", "data", "say", "NULLS", "MEM", "RES", "FOURM", "W"))
import numpy as np, zlib
from harness.adapters import member_cand as MC
from harness.anchor.stats import holm_bonferroni
R = g["new_run"](f"{OUT}/a7"); os.makedirs(R.work, exist_ok=True)
SEL, SCORE = g["SEL"], g["SCORE"]
res = {}
def four(m):
    pr = FOURM.get(m["name"]) or H.draw_of(H.REG.get(m["name"], m), 3)
    v = R.votes([m, pr])
    return H.merge4(v[m["name"]], v[pr["name"]])
if "T1" in WHICH:
    rows = R.no_id(list(SCORE) + list(SEL), "T1")
    ps = []
    for pn in g["PAIRS"]:
        par = MEM.get(pn) or RES.get(pn) or H.REG[pn]
        vp = four(par)
        nl = NULLS[pn]
        for x in nl:
            mt = MC.member_track(H, data, four(x), vp, rows, seed=zlib.crc32(x["name"].encode()) % 1000)
            ps.append(dict(parent=pn, null=x["name"] + "+" + FOURM[x["name"]]["name"], vs="parent+d3", rows=mt["rows"], D=mt["D"], p=mt["p"]))
        if len(nl) == 2:      # the two redraw pairs against each other: also a null
            mt = MC.member_track(H, data, four(nl[0]), four(nl[1]), rows, seed=7)
            ps.append(dict(parent=pn, null=nl[0]["name"] + " vs " + nl[1]["name"], vs="redraw pair", rows=mt["rows"], D=mt["D"], p=mt["p"]))
    st = MEM["stock"]; a, b = H.draw_of(st, 13), H.draw_of(st, 14)
    v = R.votes([a, b]); mt = MC.member_track(H, data, H.merge4(v[a["name"]], v[b["name"]]), four(st), rows, seed=11)
    ps.append(dict(parent="stock", null="d13+d14", vs="parent+d3", rows=mt["rows"], D=mt["D"], p=mt["p"]))
    pv = [x["p"] for x in ps if x["rows"] >= 150]
    try:
        from scipy.stats import kstest, binom
        ks = float(kstest(pv, "uniform").pvalue); k = sum(p <= .05 for p in pv); band = (binom.ppf(.025, len(pv), .05), binom.ppf(.975, len(pv), .05))
    except Exception as e:
        ks, k, band = None, sum(p <= .05 for p in pv), None
    res["T1"] = dict(reads=ps, n=len(pv), ks_p=ks, k_le_05=k, band95=band, passes=bool(ks is not None and ks > .1 and band and band[0] <= k <= band[1]))
    for x in ps: say(f"T1 {x}")
    say(f"T1: n {len(pv)} KS p {ks} k(p<=.05) {k} band {band} -> {'PASS' if res['T1']['passes'] else 'FAIL'}")
if "T2" in WHICH:
    import re
    st = MEM["stock"]; cb, cq = H.draw_of(st, 13), H.draw_of(st, 14)
    V0 = R.votes([st, cb, cq]); vp = four(st)
    draw = json.load(open(f"{W}/r001/rank.json"))["draw"]
    Rb = R.no_id([i for i in SCORE if i not in set(draw)] + list(SEL), "R")
    rows = sorted(g["LEARN_SS"])
    gold = {}
    for v in V0.values():
        for i, ss in v.items():
            for s_ in ss.values():
                if s_ and s_.get("ok") and s_.get("key") is not None:
                    gold.setdefault(i, s_["key"])
    for i in rows:
        if i in gold or i not in H.POOL:
            continue
        g_ = H.POOL[i].get("answer")
        for txt in (f"Final answer: \\boxed{{{g_}}}", f"Answer: {g_}", str(g_)):
            r_, ok_, p_ = H.regrade(dict(id=i, sample=0, content=txt, finish="stop"), H.POOL[i])
            if ok_ and p_ is not None:
                gold[i] = re.sub(r"\s+", " ", str(p_)).strip().lower() if r_["kind"] != "mc" else str(p_); break
    wrong = lambda v, i: bool(v.get(i)) and all(not x_.get("ok") for x_ in v[i].values() if x_)
    elig = [i for i in rows if i in gold and wrong(V0[cb["name"]], i) and wrong(V0[cq["name"]], i)]
    inj = lambda v, pick: {**v, **{i: {k: dict(x_, key=gold[i], ok=True) for k, x_ in v[i].items()} for i in pick}}
    out = {}
    for e in (0.0, 0.03, 0.05, 0.08):
        hits = {3: 0, 6: 0}; Ds = []
        for sd in range(200):
            rng = np.random.default_rng(100003 * sd + int(round(e * 1000)))
            n_ = min(len(elig), int(round(e * len(rows))))
            pick = list(rng.choice(elig, n_, replace=False)) if n_ else []
            vc = H.merge4(inj(V0[cb["name"]], pick), inj(V0[cq["name"]], pick))
            mt = MC.member_track(H, data, vc, vp, Rb, seed=sd)
            Ds.append(mt["D"])
            for k in (3, 6):
                hits[k] += bool(min(1.0, k * mt["p"]) <= .05 and mt["D"] > 0 and mt["up"] > mt["down"])
        out[e] = dict(holm3=hits[3] / 200, holm6=hits[6] / 200, mean_D=float(np.mean(Ds)), eligible=len(elig))
        say(f"T2 +{e*100:.0f}pp: power Holm x3 {hits[3]/200:.3f} x6 {hits[6]/200:.3f} mean D {np.mean(Ds):+.4f} (R {len(Rb)} rows)")
    res["T2"] = dict(power=out, passes=out[0.05]["holm3"] >= .8, size0=out[0.0]["holm3"])
if "T3" in WHICH:
    c10 = json.load(open(os.environ.get("CAL10", "/data/fd/mingdazhang/selfedit/rsi/mm/hb/res_test/cal1109c_10/result.json")))["rungs"]["J1"]["rounds"]
    t3 = []; fam = []
    for r in range(1, 6):
        mem = g["members_at"](r); by = {m["name"]: m for m in mem}
        kids, _ = g["rebuild"](R, r, mem)
        km = {m["name"]: (m, row) for m, row in kids}
        rr = c10[str(r)]
        draw = json.load(open(f"{W}/r{r:03d}/rank.json"))["draw"]
        Rb = R.no_id([i for i in SCORE if i not in set(draw)] + list(SEL), "R")
        st2 = [x for x in rr["stage2"] if not x.get("null")]
        top = sorted([x for x in st2 if x["name"] in km], key=lambda x: -float(km[x["name"]][1].get("zx_score") or 0))[:3]
        mts = {}
        for x in top:
            m, row = km[x["name"]]; p = by[m["parents"][0]]
            Rm = [i for i in Rb if i not in g["lineage_cards"](m)]
            vc = four(m)
            if not vc:
                continue
            mts[x["name"]] = (x, MC.member_track(H, data, vc, four(p), Rm, seed=r))
        adj = holm_bonferroni({n: mt["p"] for n, (x, mt) in mts.items()}) if mts else {}
        for n, (x, mt) in mts.items():
            ok = bool(adj[n] <= .05 and mt["D"] > 0 and mt["up"] > mt["down"] and all(x["checks"].get(k) in (True, "True") for k in ("no_harm", "card_vs_unseen", "critic", "fired")) and (x["global_"] or x["checks"].get("z") in (True, "True")))
            t3.append(dict(round=r, name=n, rows=mt["rows"], D=mt["D"], p=mt["p"], p_holm=adj[n], up=mt["up"], down=mt["down"], member_admit=ok, pool_admit=n in rr["accepted"], pool_dS=x["dS"], pool_p_holm=x.get("p_holm")))
            say(f"T3 r{r} {n}: member D {mt['D']:+.4f} p {mt['p']:.4f} holm {adj[n]:.4f} up/down {mt['up']}/{mt['down']} -> {'ADMIT' if ok else 'no'} | pool dS {x['dS']:+.4f} holm {x.get('p_holm')} {'ADMIT' if n in rr['accepted'] else 'no'}")
        if mts:          # C3: family-wise sign-flip size of the Holm-over-3 member rule, one sign per picture shared across the round's candidates
            L = [mt for _, mt in mts.values()]
            allc = sorted({c for mt in L for c in mt["clusters"]}); pos = {c: k for k, c in enumerate(allc)}
            rng = np.random.default_rng(r); hit = 0; NO, NI = 400, 500
            pre = []
            for mt in L:
                codes, k = __import__("harness.anchor.eproc", fromlist=["x"]).cluster_index(mt["clusters"])
                gi = np.array([pos[c] for c in mt["clusters"]]); pre.append((mt, codes, k, gi))
            for _ in range(NO):
                sg = rng.choice((-1.0, 1.0), size=len(allc)); ps = {}; okd = {}
                for j, (mt, codes, k, gi) in enumerate(pre):
                    x = mt["d"] * sg[gi]
                    s_ = np.bincount(codes, weights=x, minlength=k)
                    S = rng.choice((-1.0, 1.0), size=(NI, k))
                    ps[j] = float((np.sum((S @ s_) / len(x) >= x.mean() - 1e-12) + 1) / (NI + 1))
                    okd[j] = x.mean() > 0 and (x > 0).sum() > (x < 0).sum()
                a_ = holm_bonferroni(ps)
                hit += any(a_[j] <= .05 and okd[j] for j in ps)
            fam.append(dict(round=r, n=len(L), size=hit / NO))
            say(f"T3 r{r}: C3 family-wise size of the member Holm rule {hit/NO:.4f} ({len(L)} candidates, {NO}x{NI} flips)")
    res["T3"] = dict(rejudge=t3, family_sizes=fam, C3=max([f["size"] for f in fam] or [0]) <= .05)
json.dump(res, open(f"{OUT}/a7_tests.json", "w"), indent=1, default=str)
say(f"A7 TESTS DONE {OUT}/a7_tests.json")
