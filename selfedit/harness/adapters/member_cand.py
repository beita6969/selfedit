# selfedit 1109c: NEW adapter (§8.2 item 3, MemberCandidate) + the §6.2 stage-2 rule. A pool member is wrapped as an RRSI selection.Candidate whose
# EvalResult is its refit-pool read on R; RRSI's judge (floor + cost rule, verbatim) runs with our §6.2 item 3 checks as its domain guards.
from __future__ import annotations

import numpy as np

from harness.anchor import eproc
from harness.anchor.evaluate import TaskResult, aggregate
from harness.anchor.selection import Candidate, judge as rrsi_judge
from harness.adapters import pool_score

GLOBAL_SHARE = 0.9          # a candidate is global iff it answers >= 90 % of R and has no gate scope or program


def ev_of(name, right, ids, tokens):
    """EvalResult with rows as tasks, one trial each (the pool decision), tokens = the member's mean tokens."""
    per = {i: TaskResult(rewards=[float(r)], tokens=[tokens]) for i, r in zip(ids, right)}
    return aggregate(name, 1, per)


def is_global(H, m, v, R_ids):
    if H.gate_scope(m) is not None or m.get("program") or m.get("gate"):
        return False
    have = sum(1 for i in R_ids if v.get(i))
    return have >= GLOBAL_SHARE * max(1, len(R_ids))


def stage2(H, run, members, m, parent, votes, R_ids, nulls, rung, delta, cfg, incumbent_counts, card_ok=True, critic_ok=True, base=None, seed=0):
    """§6.2 item 3 on R. nulls: the parent's real re-seeded draws (member dicts present in votes). Returns the record (admissible, reasons, every number)."""
    L = eproc.LADDER[rung]
    null_ref = None
    if NULLREF[0]:           # amendment 4 (11-09): the reference pool holds one real redraw of the parent, so adding a fresh draw is not itself a gain
        ref = next((x for x in nulls if x["name"] != m["name"] and votes.get(x["name"])), None)
        if ref is not None:
            members, base, null_ref = list(members) + [ref], None, ref["name"]
            nulls = [x for x in nulls if x["name"] != ref["name"]] or [ref]
    rd = pool_score.read_R(H, run, members, m, votes, R_ids, base=base)
    ids, tab, pa = rd["ids"], rd["f0"].tab, rd["pa"]
    def _v(x):               # amendment 5: 4-sample read where a paired second read exists
        pr = FOUR.get(x["name"]) if S4[0] else None
        return H.merge4(votes[x["name"]], votes[pr]) if pr and votes.get(pr) else votes[x["name"]]
    vc, vp = _v(m), _v(parent)
    glob = is_global(H, m, vc, ids)
    lb = eproc.cluster_boot_lower(rd["d_ce"], rd["clusters"], L["q"], seed=seed)
    p_boot = eproc.cluster_boot_p(rd["d_ce"], rd["clusters"], seed=seed)
    dacc = float(np.mean(rd["d_acc"])) if len(ids) else 0.0
    scope = [i for i in ids if vc.get(i)]
    # fired-set z, sigma from the parent's real redraws on the same fired rows
    F = H.fired_rows(tab, vc, pa, scope)
    dz = H.vs_pool(tab, vc, pa, F)
    dn, sc = [], []
    for x in nulls:          # a partial redraw is read on the fired rows it covers; its second moment is scaled up to |F|
        vx = _v(x)
        cov = [i for i in F if vx.get(i)]
        if cov:
            dn.append(list(H.vs_pool(tab, vx, pa, cov).values())); sc.append(len(F) / len(cov))
    z = eproc.fired_z(dz.values(), dn, sc)
    # no-harm against the parent, bound from the parent's redraws on the same rows
    Rm = H.harm(H.paired(vc, vp, scope))
    Rn = []
    for x in nulls:          # scaled up to |scope| when the redraw covers part of it
        vx = _v(x)
        cov = [i for i in scope if vx.get(i)]
        if cov:
            Rn.append(H.harm(H.paired(vx, vp, cov)) * len(scope) / len(cov))
    hthr = eproc.harm_threshold(Rn)
    weight = bool(H.has_engine(m))
    lobo = lobo_null = None
    bench_spec, why_bs = False, []
    if LOBO[0]:                # fable_1109d patch 15 + 16: LOBO transfer read, per-bench null band, bench_specific tag
        lc = getattr(run, "c9_lobo_cache", None)
        lobo = pool_score.read_lobo(H, run, members, m, votes, ids, cache=lc)
        lobo_null = {}
        for x in nulls:
            cov = [i for i in ids if votes[x["name"]].get(i)]
            if len(cov) >= 50:
                lobo_null[x["name"]] = pool_score.read_lobo(H, run, members, x, votes, cov, cache=lc)
        for b, v in lobo.items():
            sq = [nb[b]["inc"] ** 2 for nb in lobo_null.values() if b in nb and nb[b]["rows"]]
            v["band"] = max(float(1.65 * np.sqrt(np.mean(sq))) if sq else 0.0, 2.0 / max(1, v["rows"]))      # floor: 2 rows of the bench
            if v["inc"] < -v["band"]:
                bench_spec = True; why_bs.append(f"bench {b} LOBO {v['inc']:+.3f} below -{v['band']:.3f}")
        why_bs += trigger_tags(m)
        bench_spec = bench_spec or bool(trigger_tags(m))
    checks = dict(pool_lb=bool(np.isfinite(lb) and lb > H.GAIN_TOL), dacc=dacc > 0, gained_gt_lost=rd["up"] > rd["down"],
                  no_harm=hthr is not None and Rm <= hthr, card_vs_unseen=bool(card_ok), critic=bool(critic_ok), fired=len(F) >= 8)
    if not glob:
        checks["z"] = z["z"] is not None and z["z"] >= L["z"] and z["net"] > 0
    if lobo is not None:
        checks["lobo_sum"] = sum(v["inc"] * v["rows"] for v in lobo.values()) > 0
    tok_c, tok_p = pool_score.mean_tokens(vc, ids), pool_score.mean_tokens(vp, ids)
    ev1, ev0 = ev_of(m["name"], rd["right1"], ids, tok_c), ev_of("incumbent", rd["right0"], ids, tok_p)
    S_star = ev0.S
    within = (ev1.S - ev0.S) <= delta
    waive = {"z"} if (weight and within) else set()          # D4: a within-band weight member is admitted on novelty without the z line, flagged
    guards = [k for k, ok in checks.items() if not ok and k not in waive]
    comps = [str(o) for o in (m.get("own_ops") or m.get("ops") or [])]
    cand = Candidate(m["name"], [dict(id=f"C{k+1}", component=c) for k, c in enumerate(comps)], ev=ev1)
    dec = rrsi_judge(cand, ev0, S_star, delta, cfg, incumbent_counts, guards)
    mc = eproc.mcnemar_rows(rd["right0"], rd["right1"])
    bets = eproc.fired_cluster_d(ids, rd["clusters"], rd["d_acc"], F)        # A3: one bet per fired picture-cluster
    ep = dict(eproc.eprocess_rows_status([bets]), bets_read=bets)
    return dict(name=m["name"], parent=parent["name"], null_ref=null_ref, rung=rung, global_=glob, weight=weight, rows=len(ids), S0=ev0.S, S1=ev1.S, dS=ev1.S - ev0.S, delta=delta,
                C0=tok_p, C1=tok_c, lb=lb, q=L["q"], dacc=dacc, up=rd["up"], down=rd["down"], mcnemar_p=mc["p"], fired=len(F), z=z, harm=Rm, harm_null=Rn, harm_thr=hthr,
                checks=checks, waived=sorted(waive & {k for k, ok in checks.items() if not ok}), within_band=bool(within), admissible=bool(dec.admissible), reason=dec.reason,
                novelty=dec.novelty, eproc=ep, p_boot=p_boot, _pool=list(members), _ctx=(H, run, m, votes), alpha=1 - L["level"], lobo=lobo, lobo_null=lobo_null, bench_specific=bench_spec, bench_specific_why=why_bs, _rd=rd, _judge=(cand, ev0, S_star, delta, cfg, incumbent_counts, guards))


def decide_holm(rec, p_adj):
    """Amendment A1 (prereg_1109c 'a1'): the pool line of the guards is p_holm <= alpha instead of the per-candidate LB; every other guard, the floor and
    the cost rule are unchanged (RRSI judge re-run). Writes p_holm, admissible, reason."""
    cand, ev0, S_star, delta, cfg, counts, guards = rec["_judge"]
    g = [x for x in guards if x != "pool_lb"] + ([] if p_adj <= rec["alpha"] else ["pool_holm"])
    d_eff = a2_delta(rec, p_adj, delta)
    dec = rrsi_judge(cand, ev0, S_star, d_eff, cfg, counts, g)
    rec.update(p_holm=p_adj, admissible_unadjusted=rec["admissible"], admissible=bool(dec.admissible), reason=dec.reason, delta_judge=d_eff, branch="gain" if rec["dS"] > d_eff else "shaped")
    return rec


LOBO = [True]        # fable_1109d patch 15 (prereg_1109c 'a4'): LOBO transfer read; sum of LOBO increments > 0 is a guard; a bench under its null band tags bench_specific
S4 = [False]         # amendment 5 (cal 7): member statistics (fired z, no-harm, paired) on 4 samples: a member's votes merged with FOUR[name]'s as samples 2,3
FOUR = {}            # member name -> the name of its seed-paired second read (draw 3 of a candidate; the paired redraw of a null; a parent's draw 3)
NULLREF = [False]    # amendment 4 (11-09): reference pool = pool + one redraw of the parent (prereg c1109 null_ref; cal CAL_NULLREF)
TRIG = [(r"\bbench\b\s*(==|!=|in\b)", "program compares the bench"), (r"category", "program reads the category"), (r"startswith\(\s*['\"][a-z0-9_]+/", "program tests a row-id prefix"),
        (r"['\"](vlmsbiased|babyvision|lego|blink|charxiv|visulogic)['\"]", "program names a bench")]


def trigger_tags(m):
    """fable_1109d patch 16: input-only scopes. A program trigger that reads the bench or category string, or a cat:/bench: gate scope, is bench_specific."""
    import re
    out = []
    prog = m.get("program") or ""
    for pat, why in TRIG:
        if re.search(pat, prog):
            out.append(why)
    sc = ((m.get("gate") or {}).get("scope") or "")
    if sc.startswith(("cat:", "bench:")):
        out.append(f"gate scope {sc.split(':')[0]}:")
    return out


A2 = [True]          # amendment A2 (prereg_1109c 'a2', fable_1109d §1.3 item 2): a member whose pool line passes (p_holm <= alpha) is judged on the gain branch


def a2_delta(rec, p_adj, delta):
    """A2: the RRSI cost rule takes the gain branch (dC <= beta0 + beta1 dS) whenever the Holm pool line passes and dS > 0; the shaped branch stays only
    for D4 within-band weight members. Implemented as the judge's delta = min(delta, dS - 1e-9), which also leaves the floor dS >= -delta satisfied."""
    if A2[0] and p_adj <= rec["alpha"] and rec["dS"] > 0 and not (rec["weight"] and rec["within_band"]):
        return min(delta, rec["dS"] - 1e-9)
    return delta


def family_size(recs, cfg, n=2000, seed=0):
    """A1's C3: family-wise size of one round's stage-2 admission decision under H0, by picture-clustered sign flips shared across the round's
    candidates (one sign per picture, so their dependence is kept). Per flip: each candidate's bootstrap p, dS, up/down on its flipped rows; Holm across
    the candidates; the family errs if any candidate is admitted (p_holm <= alpha, dS > 0, up > down, floor, cost rule; fixed checks as observed)."""
    from harness.anchor.selection import cost_rule
    from harness.anchor.stats import holm_bonferroni
    live = [r for r in recs if all(v for k, v in r["checks"].items() if k in ("no_harm", "card_vs_unseen", "critic", "fired"))]
    if not live:
        return 0.0
    allc = sorted({c for r in live for c in r["_rd"]["clusters"]})
    pos = {c: k for k, c in enumerate(allc)}
    pre = []
    for j, r in enumerate(live):
        rd = r["_rd"]
        codes, k = eproc.cluster_index(rd["clusters"])
        rng = np.random.default_rng(seed + 7 + j)
        W = rng.multinomial(k, np.full(k, 1.0 / k), size=eproc.BOOT_B).astype(float)
        den = np.maximum(W @ np.bincount(codes, minlength=k).astype(float), 1e-12)
        glob_ix = np.array([pos[c] for c in rd["clusters"]], int)
        dC = 0.0 if not r["C0"] or not r["C1"] else (r["C1"] - r["C0"]) / r["C0"]
        pre.append((r, codes, k, W, den, glob_ix, dC))
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(int(n)):
        sg = rng.choice((-1.0, 1.0), size=len(allc))
        ps, ok = {}, {}
        for r, codes, k, W, den, gix, dC in pre:
            s = sg[gix]
            dce, dacc = r["_rd"]["d_ce"] * s, r["_rd"]["d_acc"] * s
            means = (W @ np.bincount(codes, weights=dce, minlength=k)) / den
            ps[r["name"]] = float((np.sum(means <= 0) + 1) / (len(means) + 1))
            dS = float(dacc.mean())
            ok[r["name"]] = (dS, dC, int((dacc > 0).sum()) > int((dacc < 0).sum()))
        adj = holm_bonferroni(ps)
        def good(r):
            dS, dC, updown = ok[r["name"]]
            d = a2_delta(dict(r, dS=dS), adj[r["name"]], r["delta"])
            return dS > 0 and updown and dS >= -d and cost_rule(dS, dC, r["novelty"], d, cfg)[0]
        if any(adj[r["name"]] <= live[0]["alpha"] and good(r) for r, *_ in pre):
            hits += 1
    return hits / float(n)


def size_test(rec, cfg, n=2000, seed=0):
    """C3: size of this candidate's admission test under H0, by picture-clustered sign flips of its per-row differences on R (CE gain and accuracy
    change flipped together). The parts that do not depend on the per-row differences (no-harm, card, critic, fired, z for a global member) are
    held at their observed values; floor and cost rule are re-run on the flipped dS."""
    from harness.anchor.selection import cost_rule
    rd = rec["_rd"]
    q, delta = rec["q"], rec["delta"]
    fixed_ok = all(v for k, v in rec["checks"].items() if k in ("no_harm", "card_vs_unseen", "critic", "fired"))
    if not fixed_ok:
        return 0.0
    codes, k = eproc.cluster_index(rd["clusters"])
    rng = np.random.default_rng(seed + 1)
    W = rng.multinomial(k, np.full(k, 1.0 / k), size=eproc.BOOT_B).astype(float)
    cnt = np.bincount(codes, minlength=k).astype(float)
    den = np.maximum(W @ cnt, 1e-12)
    dC = 0.0 if not rec["C0"] or not rec["C1"] else (rec["C1"] - rec["C0"]) / rec["C0"]
    def test(x):
        dce, dacc = x["d_ce"], x["d_acc"]
        s = np.bincount(codes, weights=dce, minlength=k)
        lb = float(np.quantile((W @ s) / den, q))
        up, down = int((dacc > 0).sum()), int((dacc < 0).sum())
        dS = float(dacc.mean())
        if not (lb > 0 and dS > 0 and up > down):
            return False
        if dS < -delta:                     # floor, S* = incumbent S
            return False
        ok, _ = cost_rule(dS, dC, rec["novelty"], delta, cfg)
        return ok
    return eproc.signflip_size(test, dict(d_ce=rd["d_ce"], d_acc=rd["d_acc"]), rd["clusters"], n=n, seed=seed)


def public(rec):
    return {k: v for k, v in rec.items() if not k.startswith("_")}


# ---------------------------------------------------------------- amendment 6 (11-09, after pilot r2): null rule = candidate rule; dominance void --------
A6P = [False]        # amendment 6' (11-09): pre-A6 null rule; a null voids the round only if admitted on the gain path (dS > delta); cost-path admits are logged
A6B = [True]         # A6 (b) dominance void; off: A6 (a) only, an admissible null voids the round as before (cal CAL_A6B=0, c1109 a6b)
A6 = [False]         # prereg_1109c 'a6' (c1109 a6; cal CAL_A6). Off: the null rule and the void rule are those of amendments 1-5, byte for byte.


def a6_null(rec):
    """A6 (a): a null is judged by the same rule as a candidate (A1 pool line with its own unadjusted p as a family of one, A2 branch), not by the
    pre-A2 rule. Before A6 a null whose pool line passed fell to the shaped branch (dS within delta), where the sign of its token change decided it:
    cheaper -> admissible (pilot r2 stock~d4, p .046, dC -2.7 %), dearer -> not (cal 6pn r1 stock~d4, p .011, dC +1.1 %). No candidate is judged
    that way under A2, so the null answered a different question than the one it stands in for."""
    rec["admissible_pre_a6"] = bool(rec["admissible"])
    return decide_holm(rec, rec["p_boot"])


def beats_null(rec, nrec, seed=0):
    """A6 (b): does admitted candidate rec beat admissible null nrec? Paired on the R rows both read: per-row CE gain of the candidate (over its reference
    pool) minus that of the null (over its own), one-sided picture-cluster bootstrap p of the mean <= the rung's alpha, and mean accuracy change of the
    candidate above the null's on those rows. A bias common to both (e.g. the ensembling gain of a fresh draw) cancels in the difference."""
    a, b = rec["_rd"], nrec["_rd"]
    same = [x["name"] for x in rec.get("_pool") or []] == [x["name"] for x in nrec.get("_pool") or []]
    if not same and rec.get("_ctx") and nrec.get("_pool") is not None:
        # fix 11-09 (A6 power): read the candidate on the null's own reference pool, so both gains are over the same base. Before, when the null was
        # the candidate's null_ref redraw, the candidate's gain was already net of the null and the null's gain was subtracted a second time.
        H_, run_, m_, votes_ = rec["_ctx"]
        a = pool_score.read_R(H_, run_, nrec["_pool"], m_, votes_, list(b["ids"]))
    ia = {i: k for k, i in enumerate(a["ids"])}
    ib = {i: k for k, i in enumerate(b["ids"])}
    com = [i for i in a["ids"] if i in ib]
    if not com:
        return dict(null=nrec["name"], rows=0, p=1.0, lb=float("nan"), dacc=0.0, ok=False)
    ka, kb = [ia[i] for i in com], [ib[i] for i in com]
    dce = np.asarray(a["d_ce"], float)[ka] - np.asarray(b["d_ce"], float)[kb]
    dac = np.asarray(a["d_acc"], float)[ka] - np.asarray(b["d_acc"], float)[kb]
    cl = [a["clusters"][k] for k in ka]
    p = eproc.cluster_boot_p(dce, cl, seed=seed)
    lb = eproc.cluster_boot_lower(dce, cl, rec["q"], seed=seed)
    ok = bool(p <= rec["alpha"] and float(dac.mean()) > 0)
    return dict(null=nrec["name"], rows=len(com), p=float(p), lb=float(lb), dce=float(dce.mean()), dacc=float(dac.mean()), ok=ok, reread=not same)


def a6_void(chosen_recs, null_recs, seed=0):
    """A6 (b): an admissible null voids an admitted candidate only if the candidate does not beat it (beats_null) -- every admissible null of the round,
    whatever its parent, is a bar each admitted candidate must clear. Returns (voided names, kept names, the pair tests)."""
    adm = [n for n in null_recs if n["admissible"]]
    tests, voided, kept = {}, [], []
    for rc in chosen_recs:           # in acceptance order; an admit read against an earlier admit that is void is void too (its read assumed it)
        tests[rc["name"]] = [beats_null(rc, n, seed=seed) for n in adm]
        (kept if not voided and all(t["ok"] for t in tests[rc["name"]]) else voided).append(rc["name"])
    return voided, kept, tests


def null_path(rec):
    """A6': the path an admissible null took under the pre-A6 rule: 'gain' (dS > the judge's delta) or 'cost' (dS within delta, shaped branch)."""
    return "gain" if rec["dS"] > rec.get("delta_judge", rec["delta"]) else "cost"


# ---------------------------------------------------------------- amendment 7 (11-09, fable_1109 §1.2): member track ----------------------------------
A7 = [False]         # prereg_1109c 'a7' (c1109 a7; cal CAL_A7): primary admission = per-member paired accuracy 4 vs 4 samples; pool line logged


def member_track(H, data, vc4, vp4, R_ids, B=2000, seed=0):
    """A7 member track. vc4 / vp4: 4-sample vote dicts of candidate and parent on the same seeds (H.merge4 of samples 0,1 and the seed-paired draw).
    Per row of R both read: D_i = right rate of the candidate's samples minus the parent's (H.paired: a NO_RUN / unparsed sample is wrong).
    One-sided picture-cluster sign-flip p of mean D (one sign per picture, B flips): (#{flipped mean >= observed} + 1) / (B + 1)."""
    d = H.paired(vc4, vp4, R_ids)
    ids = [i for i in R_ids if i in d]
    if not ids:
        return dict(rows=0, D=0.0, p=1.0, up=0, down=0, ids=[], d=np.zeros(0), clusters=[])
    x = np.array([d[i] for i in ids], float)
    cl = [str(data.cluster(i)) for i in ids]
    codes, k = eproc.cluster_index(cl)
    s_ = np.bincount(codes, weights=x, minlength=k)
    rng = np.random.default_rng(seed)
    obs = float(x.mean())
    S = rng.choice((-1.0, 1.0), size=(int(B), k))
    means = (S @ s_) / len(x)
    p = float((np.sum(means >= obs - 1e-12) + 1) / (B + 1))
    return dict(rows=len(ids), D=obs, p=p, up=int((x > 0).sum()), down=int((x < 0).sum()), ids=ids, d=x, clusters=cl)


def a7_decide(rec, mt, p_holm, alpha=0.05):
    """A7 admission on the member track: p_holm <= alpha, mean D > 0, rows up > down, and the fixed guards of the stage-2 record (no-harm, card, critic,
    fired; scoped members also fired-z as now). The RRSI cost rule runs on the gain branch (A2) with dS = mean D."""
    from harness.anchor.selection import cost_rule
    c = rec["checks"]
    fixed = all(c.get(k) in (True, "True") for k in ("no_harm", "card_vs_unseen", "critic", "fired")) and (rec["global_"] or c.get("z") in (True, "True"))
    dC = (rec["C1"] - rec["C0"]) / rec["C0"] if rec.get("C0") and rec.get("C1") else 0.0
    cost_ok = cost_rule(mt["D"], dC, rec.get("novelty", 0), min(rec["delta"], mt["D"] - 1e-9) if mt["D"] > 0 else rec["delta"], _CFG[0])[0] if _CFG[0] else True
    ok = bool(p_holm <= alpha and mt["D"] > 0 and mt["up"] > mt["down"] and fixed and cost_ok)
    why = [] if ok else [k for k, v in (("p_holm", p_holm <= alpha), ("D>0", mt["D"] > 0), ("up>down", mt["up"] > mt["down"]), ("fixed guards", fixed), ("cost", cost_ok)) if not v]
    return ok, why


_CFG = [None]        # the RRSI config (set by the caller: hw_round init / cal)
