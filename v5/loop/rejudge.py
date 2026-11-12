"""Re-judge one finished generation on the per-routed-row scale (AM14) from the reads it already has (no new reads), with the full
cluster-robust covariance. Keeps the old numbers in judge_v0 / lfsr_v0 / accepted_v0 / D_v0, sets accepted from the new judgement
(lfsr <= alpha; generation 1 predates AM10, whose numbers are added for every candidate), leaves merged and the tip as they are.
Backs up state.json and ledger.jsonl (copies with suffix .pre_am14) before writing; rewrites that generation's ledger lines.
Run it only while the run's gen.py is stopped. usage: rejudge.py RUN_DIR GEN [ROOT] [--dry]"""
import json, os, shutil, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(os.path.dirname(HERE), "harness")]
import gen  # noqa: E402
import l1 as L1  # noqa: E402
import yield_l1 as YL  # noqa: E402

args = [x for x in sys.argv[1:] if not x.startswith("--")]
dry = "--dry" in sys.argv
RUN, g = args[0], int(args[1])
R = gen.Run(gen.args(["--run", RUN] + (["--root", args[2], "--rows", os.path.join(args[2], "rows/learn.jsonl")] if len(args) > 2 else [])))
cur_g, cur_tip = R.state.get("gen"), R.state["tip"]
R.g = g
G = R.G()
tips = {t["id"]: t for t in R.state["tips"] + [cur_tip]}
V = [c for c in G["cands"] if c.get("rung") and c.get("entry")]
R.state["tip"] = tips[V[0]["parent"]]
R.load_routes(R.state["tip"]["entries"], R.role["show"] + R.role["judge"] + R.role["monitor"])
R.load_routes([c["entry"] for c in V], R.judge_rows)
B = R.judge_numbers("obs", V)
R.state["tip"] = cur_tip

rows = []
for c in V:
    o = B.get(c["cid"]) or {}
    if o.get("too_few"):
        continue
    if "judge_v0" not in c:
        c["judge_v0"] = c.get("judge")
        c["accepted_v0"] = bool(c.get("accepted"))
        c["lfsr_v0"] = (c.get("judge") or {}).get("lfsr")
        c["D_v0"] = (c.get("judge") or {}).get("D")
        c["status_v0"] = c.get("status")
    acc = bool(o["lfsr"] <= gen.ALPHA)
    c["judge"] = dict(rung=1, scale="routed", rejudged="AM14", **{k: o[k] for k in ("D", "se", "clusters", "lfsr", "post_mean", "family", "K", "mu", "tau", "contrib")},
                      am10=o.get("am10"))
    c.update(dw=dict(d=o["dd"], w=o["w"], scale="routed"), contrib=o["contrib"], n_read=o["n"], accepted=acc,
             status=f"{'accepted' if acc else 'not accepted'} on all reads, re-judged per routed row (AM14, lfsr {o['lfsr']:.3f}; v0: lfsr {c['lfsr_v0']:.3f}, "
                    f"{'accepted' if c['accepted_v0'] else 'not accepted'})")
    a10 = o.get("am10") or {}
    rows.append(dict(id=c["cid"], author=c["author"], kind=c.get("kind"), src=c.get("card_source"), rows=o["n"], share=round(c["share"], 3), d=round(o["D"], 4),
                     se=round(o["se"], 4), lfsr=round(o["lfsr"], 3), post_mean=round(o["post_mean"], 4), contrib=round(o["contrib"], 4), accepted=acc,
                     lfsr_v0=c["lfsr_v0"], accepted_v0=c["accepted_v0"], merged=bool(c.get("merged")),
                     am10_P=a10.get("P_mu_le0"), am10_mu=a10.get("mu_mean"), am10_Ds=a10.get("D_s"), tok=(c.get("tok_cand"), c.get("tok_parent"))))
fam = {}
for c in V:
    j = c["judge"]
    fam[j["family"]] = (j["mu"], j["tau"], j["K"])

# L1 on the routed scale: full covariance and diagonal
allc = [c for G2 in R.state["gens"].values() for c in G2.get("cands", []) if int(c.get("gen", 0)) <= g]
P, summ = L1.posteriors(allc, ["B", "P0"], gen.ALPHA)
C = L1.judged(allc)
D, S = L1.family_all(C)
Sd = np.diag(np.diag(S))
lp = YL.unit_prior(D, Sd)
pd, sd_ = {}, {}
for a in ("B", "P0"):
    ix = [i for i, c in enumerate(C) if c["author"] == a]
    pd[a] = YL.yield_posterior_routed(D[ix], Sd[np.ix_(ix, ix)], np.array([C[i]["share"] for i in ix]), gen.ALPHA, lp)
    sd_[a] = YL.summary(*pd[a])
ycmp = dict(full=dict(Y=summ, P_P0_gt_B=YL.p_greater(*P["P0"], *P["B"])), diag=dict(Y=sd_, P_P0_gt_B=YL.p_greater(*pd["P0"], *pd["B"])))

out = dict(gen=g, rows=rows, families=fam, L1=ycmp)
print(json.dumps(out, indent=1, default=float))
if dry:
    sys.exit(0)

sp, lp_ = os.path.join(RUN, "state.json"), os.path.join(RUN, "ledger.jsonl")
for p in (sp, lp_):
    if os.path.exists(p) and not os.path.exists(p + ".pre_am14"):
        shutil.copy2(p, p + ".pre_am14")
R.save()
by = {c["cid"]: c for c in V}
lines = []
for l in open(lp_):
    x = json.loads(l)
    c = by.get(x.get("id"))
    if c is not None and x.get("gen") == g and not x.get("final"):
        j = c["judge"]
        x.update(D=j["D"], se=j["se"], lfsr=j["lfsr"], post_mean=j["post_mean"], contrib=j["contrib"], scale="routed", family=j["family"], accepted=c["accepted"],
                 status=c["status"], am10=j.get("am10"), D_v0=c["D_v0"], lfsr_v0=c["lfsr_v0"], accepted_v0=c["accepted_v0"], rejudged="AM14")
    lines.append(json.dumps(x, ensure_ascii=False))
with open(lp_ + ".tmp", "w") as f:
    f.write("\n".join(lines) + "\n")
os.replace(lp_ + ".tmp", lp_)
gen.jdump(os.path.join(RUN, f"rejudge_g{g:03d}_am14.json"), out)
print(f"written: state.json (backup state.json.pre_am14), ledger.jsonl (backup ledger.jsonl.pre_am14), rejudge_g{g:03d}_am14.json")
