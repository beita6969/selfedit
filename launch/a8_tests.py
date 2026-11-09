"""Amendment 8 CPU checks: cross-round e-process for carried candidates (betting e-process over picture-cluster mean paired D of fresh 4-vs-4 reads,
eproc.eprocess_rows, e-Bonferroni over a fixed carry cap K=4: admit at E >= K/alpha = 80).
N1: real null reads on the cal 7 draws (stock and member redraws vs the parent), E path per lineage.
N2: parametric null and power from the real per-row stock accuracy (16 samples per row), real picture clusters, 4 rounds, drop rule (D <= 0 on two
consecutive reads), at the full carry rows and at a half-size scope.
usage: env (cal 7 env, CAL_A5S=1, CAL_EXCLUDE) a8_tests.py OUT"""
import os, sys, json
CAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cal_1109c.py")
OUT = sys.argv[1]
g = dict(__file__=CAL, __name__="cal_setup"); sys.argv = [CAL, OUT]
exec(compile(open(CAL).read().split("TB, OUTR = [], {}")[0], CAL, "exec"), g)
H, data, say, NULLS, MEM, RES, FOURM, W = (g[k] for k in ("H", "data", "say", "NULLS", "MEM", "RES", "FOURM", "W"))
import numpy as np, zlib
from harness.adapters import member_cand as MC
from harness.anchor import eproc
K, ALPHA = 4, 0.05
THR = K / ALPHA
R = g["new_run"](f"{OUT}/a8"); os.makedirs(R.work, exist_ok=True)
rows = sorted(g["LEARN_SS"])
res = {}
def four(m):
    pr = FOURM.get(m["name"]) or H.draw_of(H.REG.get(m["name"], m), 3)
    v = R.votes([m, pr])
    return H.merge4(v[m["name"]], v[pr["name"]])
def bets(mt):
    by = {}
    for c, x in zip(mt["clusters"], mt["d"]):
        by.setdefault(c, []).append(float(x))
    return [float(np.clip(np.mean(by[c]), -1, 1)) for c in sorted(by)]
# N1: real nulls, two fresh reads per lineage where they exist
n1 = []
for pn in g["PAIRS"]:
    par = MEM.get(pn) or RES.get(pn) or H.REG[pn]
    vp = four(par); reads = []
    for x in NULLS[pn]:
        mt = MC.member_track(H, data, four(x), vp, rows)
        if mt["rows"]:
            reads.append(bets(mt))
    if reads:
        e = eproc.eprocess_rows(reads)
        n1.append(dict(parent=pn, reads=len(reads), E=e["E"], maxE=max(e["path"]), bets=e["bets"]))
st = MEM["stock"]
a, b = H.draw_of(st, 13), H.draw_of(st, 14)
v = R.votes([a, b])
r1 = MC.member_track(H, data, four(NULLS["stock"][0]), four(st), rows)
r2 = MC.member_track(H, data, four(NULLS["stock"][1]), H.merge4(v[a["name"]], v[b["name"]]), rows)
e = eproc.eprocess_rows([bets(r1), bets(r2)])
n1.append(dict(parent="stock (fresh parent on read 2: d13+d14)", reads=2, E=e["E"], maxE=max(e["path"]), bets=e["bets"]))
for x in n1: say(f"N1 {x}")
res["N1"] = dict(reads=n1, max_E=max(x["maxE"] for x in n1), passes=all(x["maxE"] < THR for x in n1))
# N2: parametric from the real per-row stock accuracy
ds = [st] + [H.draw_of(st, d) for d in (3, 4, 6, 8, 9, 13, 14)]
V = R.votes(ds)
p, cl, ids = [], [], []
for i in rows:
    ok = [bool(s_.get("ok")) for m in ds for s_ in (V[m["name"]].get(i) or {}).values() if s_]
    if len(ok) >= 8:
        ids.append(i); p.append(np.mean(ok)); cl.append(str(data.cluster(i)))
p = np.array(p); codes, kc = eproc.cluster_index(cl)
say(f"N2: {len(ids)} rows with >= 8 stock samples, {kc} picture clusters, mean stock accuracy {p.mean():.3f}")
def sim(eff, n_rows, n_rounds=4, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    adm = np.zeros(n_rounds + 1); Ds = []
    for b in range(B):
        sub = np.sort(rng.choice(len(ids), n_rows, replace=False)) if n_rows < len(ids) else np.arange(len(ids))
        pc = p[sub].copy(); pp = p[sub]
        if eff > 0:            # a persistent fix: rows the parent mostly misses become right on every candidate sample
            el = np.where(pp <= .5)[0]
            n_fix = min(len(el), int(round(eff * len(sub) / max(1e-9, np.mean(1 - pp[el])))))
            pc[rng.choice(el, n_fix, replace=False)] = 1.0
        cs = codes[sub]; reads = []; neg = 0
        for t in range(n_rounds):
            d = rng.binomial(4, pc) / 4 - rng.binomial(4, pp) / 4
            Ds.append(d.mean())
            s_ = np.bincount(cs, weights=d, minlength=kc); n_ = np.bincount(cs, minlength=kc)
            reads.append([float(s_[k] / n_[k]) for k in range(kc) if n_[k]])
            E = eproc.eprocess_rows(reads)["E"]
            if E >= THR:
                adm[t + 1:] += 1; break
            neg = neg + 1 if d.mean() <= 0 else 0
            if neg >= 2:
                break
    return dict(admit_by_round={t: float(adm[t] / B) for t in range(1, n_rounds + 1)}, mean_D=float(np.mean(Ds)))
n2 = {}
for nr in (len(ids), min(200, len(ids))):
    for eff in (0.0, 0.02, 0.03, 0.05):
        x = sim(eff, nr, seed=int(eff * 1000) + nr)
        n2[f"rows{nr}_eff{eff}"] = x
        say(f"N2 rows {nr} effect +{eff*100:.0f} pp: admitted by round 1..4 {x['admit_by_round']} mean D {x['mean_D']:+.4f}")
res["N2"] = n2
res["N2_size_ok"] = all(v["admit_by_round"][4] <= ALPHA / K * 1.5 + 0.005 for k, v in n2.items() if k.endswith("eff0.0"))
json.dump(res, open(f"{OUT}/a8_tests.json", "w"), indent=1, default=str)
say(f"A8 TESTS DONE {OUT}/a8_tests.json")
