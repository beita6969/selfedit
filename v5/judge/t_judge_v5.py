"""Tests of the v5 judge library (plain asserts; prints 'N ok, 0 failed'). Synthetic rows only; no engine, no export file.
Covers: the statistic (stats5), the policy evaluator and the frontier rule (policy5), the judge with the source-benchmark exclusion, the
entry audit, the null audit and the row-role guards (judge_v5, reads5), the declared-guess detector (guess5)."""
import itertools
import json
import math
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import guess5 as G  # noqa: E402
import judge_v5 as JV  # noqa: E402
import policy5 as P  # noqa: E402
import reads5 as R  # noqa: E402
import stats5 as ST  # noqa: E402

OK, BAD = [0], [0]
BEN = ["b0", "b1", "b2", "b3", "b4", "b5"]


def check(cond, what):
    if cond:
        OK[0] += 1
        print("ok  ", what)
    else:
        BAD[0] += 1
        print("FAIL", what)


def raises(fn, exc, what):
    try:
        fn()
    except exc:
        check(True, what)
    except Exception as e:  # noqa: BLE001
        check(False, f"{what} (raised {type(e).__name__}: {e})")
    else:
        check(False, what + " (nothing raised)")


def mk_rows(n_per=40, roles=("judge",), shared_pics=0):
    rows = {}
    for b in BEN:
        for role in roles:
            for i in range(n_per):
                rid = f"{b}/{role}{i}"
                pic = f"{b}-{role}-pic{i // 2}" if i < shared_pics else rid
                rows[rid] = dict(id=rid, bench=b, role=role, pic=pic, kind="mc", category=None)
    return rows


def mk_arm(rows, rng, n=8, p_right=None, tokens=1000.0, plant=None, answers="ABCD", prompt=4000.0):
    """reads per row: right with the row's probability (Beta(2,1.2) per row unless given); wrong answers spread over 3 options."""
    arm, pr = {}, {}
    for r, x in rows.items():
        p = p_right[r] if p_right is not None else float(rng.beta(2.0, 1.2))
        if plant and x["bench"] in plant:
            p = min(1.0, p + plant[x["bench"]])
        pr[r] = p
        xs = []
        for s in range(n):
            ok = rng.random() < p
            pred = "A" if ok else str(rng.choice(list(answers[1:])))
            xs.append(dict(ok=bool(ok), pred=pred, tokens=float(tokens * (0.5 + rng.random())), prompt_tokens=prompt, finish="stop", s=s, guess=False))
        arm[r] = xs
    return arm, pr


# ================================================================================================ stats5
def t_stats():
    rows = mk_rows(10, shared_pics=4)
    ids = sorted(rows)
    rng = np.random.default_rng(0)
    x = {r: float(rng.normal(0.02, 0.3)) for r in ids}
    E = ST.bench_effects(x, rows, floor_rows=0.0)
    b = "b0"
    xs = {}
    for r in ids:
        if rows[r]["bench"] == b:
            xs.setdefault(rows[r]["pic"], []).append(x[r])
    n = sum(len(v) for v in xs.values())
    d = sum(sum(v) for v in xs.values()) / n
    Gc = len(xs)
    se = math.sqrt(sum((sum(v) - d * len(v)) ** 2 for v in xs.values()) * Gc / (Gc - 1)) / n
    check(abs(E[b]["d"] - d) < 1e-12 and abs(E[b]["se"] - se) < 1e-12 and E[b]["G"] == Gc == 8 and E[b]["n"] == 10,
          "bench_effects: mean and cluster-robust se over picture clusters (2 shared pictures -> 8 clusters of 10 rows)")
    check(ST.bench_effects(x, rows, floor_rows=1.0)[b]["se"] >= 1.0 / n - 1e-12, "bench_effects: the se floor is floor_rows / n")
    pg, who = ST.p_gen(E, BEN)
    lob = [ST.p_macro(E, [c for c in BEN if c != bb]) for bb in BEN]
    check(abs(pg - max(lob)) < 1e-15 and who == BEN[int(np.argmax(lob))], "p_gen = the largest leave-one-benchmark-out p_macro, with its carrier")
    check(pg >= ST.p_macro(E, BEN) - 0.25, "p_gen is not far below p_macro on noise (sanity)")
    # exact benchmark sign-flip
    E6 = {bb: dict(d=v, se=0.01, n=10, G=10) for bb, v in zip(BEN, [.05, .04, .03, .02, .01, .06])}
    check(abs(ST.p_bench_signflip(E6, BEN) - 1 / 64) < 1e-12, "benchmark sign-flip: six positive effects -> 1 / 64")
    Ew = {bb: dict(d=v, se=0.01, n=10, G=10) for bb, v in zip(BEN, [.2, 0, 0, 0, 0, 0])}
    check(abs(ST.p_bench_signflip(Ew, BEN) - 0.5) < 1e-12, "benchmark sign-flip: an effect on one benchmark only -> 0.5 (ties count)")
    check(ST.p_bench_t(E6, BEN) < 0.01 and ST.p_bench_t(Ew, BEN) > 0.15, "benchmark t: six positive small p; one-benchmark wonder not significant")
    # a one-benchmark wonder: p_macro small, p_gen large
    xw = {r: (0.5 if rows[r]["bench"] == "b3" else 0.0) + float(rng.normal(0, 0.2)) for r in ids}
    Ew2 = ST.bench_effects(xw, rows)
    check(ST.p_macro(Ew2, BEN) < 0.01 and ST.p_gen(Ew2, BEN)[0] > 0.05 and ST.p_gen(Ew2, BEN)[1] == "b3",
          "one benchmark carries the effect: p_macro passes, p_gen does not, the carrier is named")
    check(ST.p_gen(Ew2, ["b0", "b1"])[0] == 1.0, "p_gen needs at least 3 scored benchmarks")
    check(ST.lobo_point({bb: dict(d=v, se=.01, n=1, G=2) for bb, v in zip(BEN, [.2, -.02, -.02, -.02, -.02, -.02])}, BEN) < 0, "lobo_point: negative without the carrier")
    # family
    check(list(ST.holm([0.001, 0.06, 0.03, 0.07], 0.10)) == [True, False, True, False], "Holm: 0.001 <= .025, 0.03 <= .0333, 0.06 > .05 stops (0.07 <= .10 is not reached)")
    check(list(ST.holm([0.06, 0.06], 0.10)) == [False, False] and list(ST.bh([0.06, 0.06], 0.10)) == [True, True], "Holm is stricter than BH on two p = .06")
    check(list(ST.holm([], 0.1)) == [], "Holm of an empty family")
    # negative benchmark
    xn = dict(x)
    for r in ids:
        if rows[r]["bench"] == "b1":
            xn[r] = -0.6 + float(rng.normal(0, 0.1))
    check([bb for bb, _ in ST.neg_bench(ST.bench_effects(xn, rows), BEN)] == ["b1"], "neg_bench names the significantly negative benchmark")
    # array form = scalar form
    D = ST.Design(ids, rows, BEN)
    X = np.array([[x[r] for r in ids], [xw[r] for r in ids]])
    st = D.stats(X)
    E1, E2 = ST.bench_effects(x, rows), ST.bench_effects(xw, rows)
    check(abs(st["p_macro"][0] - ST.p_macro(E1, BEN)) < 1e-12 and abs(st["p_gen"][1] - ST.p_gen(E2, BEN)[0]) < 1e-12
          and abs(st["p_bench"][0] - ST.p_bench_signflip(E1, BEN)) < 1e-12 and abs(st["p_bench_t"][1] - ST.p_bench_t(E2, BEN)) < 1e-9,
          "Design.stats (simulation form) equals the scalar functions")
    S5 = [bb for bb in BEN if bb != "b3"]
    check(abs(D.stats(X, S=S5)["p_macro"][1] - ST.p_macro(E2, S5)) < 1e-12, "Design.stats on a scored subset equals p_macro on that subset")
    # sign flip
    rows2 = mk_rows(74)
    ids2 = sorted(rows2)
    D2 = ST.Design(ids2, rows2, BEN)
    xd = np.random.default_rng(1).normal(0.03, 0.35, len(ids2))
    sf = D2.signflip(xd, n=20000, seed=3)
    Ed = ST.bench_effects(dict(zip(ids2, xd)), rows2)
    check(abs(sf["p_macro"] - ST.p_macro(Ed, BEN)) < 0.02 and abs(sf["p_gen"] - ST.p_gen(Ed, BEN)[0]) < 0.03, "sign-flip p-values agree with the t forms when many rows differ")
    xs_ = np.zeros(len(ids2))
    for j, r in enumerate(ids2):
        if rows2[r]["bench"] == "b0" and j % 74 < 4:
            xs_[j] = 1.0
    sf2 = D2.signflip(xs_, n=20000, seed=3)
    check(sf2["p_macro"] >= 0.8 / 16 and sf2["p_macro"] <= 1.3 / 16, "sign-flip: 4 active rows cannot give p below 2^-4 (t form would say less)")
    check(D2.signflip(np.zeros(len(ids2)))["p_macro"] == 1.0, "sign-flip: no difference at all -> p = 1")
    check(D2.signflip(xd, n=20000, seed=3)["p_gen"] == sf["p_gen"], "sign-flip is deterministic at a fixed seed")
    sfn = D2.signflip(-np.abs(xd) * (np.array([rows2[r]["bench"] for r in ids2]) == "b2"))
    check(sfn["p_neg"][2] < 0.001 and min(sfn["p_neg"][i] for i in (0, 1, 3, 4, 5)) > 0.5, "sign-flip lower tail marks the negative benchmark only")


# ================================================================================================ policy5
def brute(pol, tip, extra=None):
    """independent brute force: every ordered tuple, answer by explicit rules."""
    n0, trig, final, ne = pol["n0"], pol["trigger"], pol.get("final", "plur"), pol.get("n_extra", 1)
    own = pol.get("extra", "tip") == "tip"
    oks, toks, esc = [], [], []

    def plur(reads):
        cnt = {}
        for j, x in enumerate(reads):
            c = P.cls_of(x)
            cnt.setdefault(c, [0, j, x["ok"] and c != P.NONE])
            cnt[c][0] += 1
        best = sorted(cnt.values(), key=lambda v: (-v[0], v[1]))[0]
        return float(best[2])
    for tp in itertools.permutations(range(len(tip)), n0):
        init = [tip[i] for i in tp]
        dis = len({P.cls_of(x) for x in init}) > 1
        gs = any(P.guess_of(x, pol.get("gw")) for x in init)
        fire = {"none": False, "disagree": dis, "guess": gs, "either": dis or gs}[trig]
        if not fire:
            oks.append(plur(init)), toks.append(sum(x["tokens"] for x in init)), esc.append(0.0)
            continue
        pool = [tip[i] for i in range(len(tip)) if i not in tp] if own else extra
        o, t = [], []
        for cb in itertools.permutations(range(len(pool)), ne):
            ex = [pool[i] for i in cb]
            if final == "extra":
                a = next((x for x in ex if P.cls_of(x) != P.NONE), None)
                o.append(float(a["ok"]) if a is not None else plur(init + ex))
            else:
                o.append(plur(init + ex))
            t.append(sum(x["tokens"] for x in init + ex))
        oks.append(np.mean(o)), toks.append(np.mean(t)), esc.append(1.0)
    return float(np.mean(oks)), float(np.mean(toks)), float(np.mean(esc))


def t_policy():
    rng = np.random.default_rng(5)
    rows = mk_rows(6)
    tip, _ = mk_arm(rows, rng, n=6)
    ext, _ = mk_arm(rows, rng, n=3, tokens=2000.0)
    for r in list(tip)[:10]:
        for x in tip[r]:
            x["guess"] = bool(rng.random() < 0.3)
    check(P.cls_of(dict(pred="5.0")) == P.cls_of(dict(pred="5")) and P.cls_of(dict(pred=" Yes ")) == P.cls_of(dict(pred="yes"))
          and P.cls_of(dict(pred=None)) == P.NONE and P.cls_of(dict(pred="x", cls=("v", "Q"))) == ("v", "Q"), "cls_of: numbers by value, text lower-cased, None -> none, an exported class wins")
    a, b, c = dict(ok=True, pred="A", tokens=1), dict(ok=False, pred="B", tokens=1), dict(ok=False, pred="C", tokens=1)
    check(P._plur([b, a, c]) == (("v", "b"), False) and P._plur([a, b, b]) == (("v", "b"), False) and P._plur([a, b, c])[1] is True,
          "plurality: ties go to the earliest read; a majority wins")
    check(P._plur([dict(ok=True, pred=None, tokens=1)])[1] is False, "plurality: a read without an answer is wrong whatever its ok says")
    r0 = sorted(tip)[0]
    v1 = P.eval_row(dict(n0=1, trigger="none", extra="tip"), tip[r0])
    check(abs(v1["ok"] - np.mean([x["ok"] for x in tip[r0]])) < 1e-12 and abs(v1["tokens"] - np.mean([x["tokens"] for x in tip[r0]])) < 1e-9
          and abs(v1["total"] - v1["tokens"] - 4000.0) < 1e-9, "vote1 = the row's mean ok and mean tokens; total = generated + prompt")
    pols = [dict(n0=3, trigger="none", extra="tip"), dict(n0=2, trigger="disagree", extra="tip", final="plur"), dict(n0=2, trigger="disagree", extra="tip", final="extra"),
            dict(n0=2, trigger="disagree", extra="X", final="extra"), dict(n0=2, trigger="disagree", extra="X", final="plur"),
            dict(n0=1, trigger="guess", extra="tip", final="extra"), dict(n0=2, trigger="either", extra="X", final="extra"),
            dict(n0=3, trigger="disagree", extra="tip", n_extra=2, final="plur"), dict(n0=1, trigger="guess", extra="X", final="extra")]
    good = True
    for pol in pols:
        for r in sorted(tip)[:10]:
            v = P.eval_row(pol, tip[r], ext[r])
            bo, bt, be = brute(pol, tip[r], ext[r])
            good &= abs(v["ok"] - bo) < 1e-9 and abs(v["tokens"] - bt) < 1e-6 and abs(v["esc"] - be) < 1e-9
    check(good, "eval_row equals an independent brute force over every ordering (9 policies x 10 rows: ok, tokens, escalation rate)")
    es = P.evaluate(dict(n0=2, trigger="disagree", extra="tip", final="plur"), tip, sorted(tip))
    v3 = P.evaluate(dict(n0=3, trigger="none", extra="tip"), tip, sorted(tip))
    check(all(abs(es[r]["ok"] - v3[r]["ok"]) < 1e-9 for r in tip) and sum(es[r]["tokens"] for r in tip) < sum(v3[r]["tokens"] for r in tip),
          "the early-stopping vote (2 reads, third on disagreement) gives the vote of 3's answers on every row with fewer tokens")
    raises(lambda: P.eval_row(dict(n0=1, trigger="guess", extra="tip", final="extra"), [dict(ok=True, pred="A", tokens=1, guess=None)] * 3), ValueError,
           "a guess-triggered policy refuses reads without the declared-guess flag")
    check(P.eval_row(dict(n0=5, trigger="none", extra="tip"), tip[r0][:3]) is None and P.eval_row(dict(n0=2, trigger="disagree", extra="X"), tip[r0], None) is None,
          "too few samples (tip or extra arm) -> None")
    g = [dict(ok=False, pred="B", tokens=10, guess={600: False, 2500: True}), dict(ok=True, pred="A", tokens=10, guess={600: False, 2500: False})]
    lo = P.eval_row(dict(n0=1, trigger="guess", extra="tip", final="extra", gw=600), g)
    hi = P.eval_row(dict(n0=1, trigger="guess", extra="tip", final="extra", gw=2500), g)
    check(lo["esc"] == 0.0 and lo["ok"] == 0.5 and hi["esc"] == 0.5 and hi["ok"] == 1.0, "guess windows: the policy's gw picks the flag (window 2500 escalates the flagged wrong read and takes the fresh one)")
    names = [P.name_of(p) for p in P.grammar(("tip", "FL"), windows=(600,))]
    check(len(names) == len(set(names)) and "n1:guess@600->FL:extra" in names and "n2:disagree->tip" in names and not any(n.startswith("n1:disagree") for n in names)
          and not any(n.startswith("n1:") and not n.endswith(":extra") for n in names), "grammar: unique names; n0 = 1 cannot disagree and uses final = extra")
    # hull
    h = P.hull([(1, .70, "a"), (2, .705, "b"), (2.4, .72, "c"), (3, .72, "d"), (4, .725, "e"), (5, .74, "f")])
    check([p[2] for p in h] == ["a", "c", "f"], "hull: dominated points and points under a chord are dropped")
    rows8 = mk_rows(30)
    tip8, _ = mk_arm(rows8, np.random.default_rng(7), n=8)
    ids = sorted(rows8)
    F = P.Frontier(tip8, ids, rows8, policies=P.tip_policies(max_vote=4, with_guess=False))
    V = F.vertices
    check(V[0][2] == "vote1" and all(V[i][0] < V[i + 1][0] and V[i][1] < V[i + 1][1] for i in range(len(V) - 1)), "frontier: starts at one read; tokens and accuracy increase along the vertices")
    slopes = [(V[i + 1][1] - V[i][1]) / (V[i + 1][0] - V[i][0]) for i in range(len(V) - 1)]
    check(all(slopes[i] >= slopes[i + 1] - 1e-12 for i in range(len(slopes) - 1)), "frontier: concave (slopes do not increase)")
    T = 0.5 * (V[0][0] + V[1][0])
    ok_, tok_, desc = F.ref_at(T)
    check(abs(P.macro_mean({r: dict(v=tok_[r]) for r in ids}, rows8, "v") - T) < 1e-6 and abs(desc["w"] - 0.5) < 1e-9
          and abs(P.macro_mean({r: dict(v=ok_[r]) for r in ids}, rows8, "v") - 0.5 * (V[0][1] + V[1][1])) < 1e-9,
          "ref_at: the mixture of the two adjacent vertices has EXACTLY the asked tokens and the mixed accuracy")
    check(F.ref_at(T, strict=True)[2]["w"] == 1.0 and F.ref_at(V[0][0] * 0.9)[2]["hi"] == "vote1", "ref_at: strict takes the upper vertex; below one read the reference is one read")
    top = max(p[0] for p in F.points)
    above = F.ref_at(0.5 * (V[-1][0] + top)) if top > V[-1][0] + 1 else None
    check((above is None and top <= V[-1][0] + 1) or (above is not None and above[2]["flat"] and above[2]["hi"] == V[-1][2]), "ref_at: flat above the last vertex while evaluated policies exist there")
    check(F.ref_at(top * 1.5) is None, "ref_at: beyond the largest evaluated policy the frontier is unknown (None)")
    Ft = P.Frontier(tip8, ids, rows8, policies=list(F.pols.values()), cost="total", table=F.table)
    check(abs(Ft.base_tokens - F.base_tokens - 4000.0) < 1e-6 and Ft.table is not None, "frontier on the total-token axis: one read costs generated + prompt tokens")
    raises(lambda: P.Frontier(tip8, ids, rows8, policies=list(F.pols.values()), table=F.table, hull_from={"vote3"}), ValueError, "hull_from without the single read is refused")
    Fh = P.Frontier(tip8, ids, rows8, policies=list(F.pols.values()), table=F.table, hull_from={"vote1", "vote3"})
    check([p[2] for p in Fh.vertices] == ["vote1", "vote3"], "hull_from: the vertices named elsewhere (signal rows) are used as they are")


# ================================================================================================ views (stage 3)
def t_views():
    rng = np.random.default_rng(21)
    rows = mk_rows(4)
    a, _ = mk_arm(rows, rng, n=5)
    b, _ = mk_arm(rows, rng, n=4, tokens=2000.0)
    same = [(dict(n0=1, trigger="none", extra="tip"), P.views_policy(["a"])), (dict(n0=2, trigger="disagree", extra="tip", final="plur"), P.views_policy(["a", "a"], ["a"])),
            (dict(n0=2, trigger="disagree", extra="X", final="extra"), P.views_policy(["a", "a"], ["b"], "extra")),
            (dict(n0=3, trigger="disagree", extra="tip", n_extra=2, final="plur"), P.views_policy(["a", "a", "a"], ["a", "a"]))]
    good = True
    for old_, new_ in same:
        for r in sorted(rows)[:8]:
            x, y = P.eval_row(old_, a[r], b[r]), P.eval_views_row(new_, {"a": a[r], "b": b[r]}, max_init=10 ** 6, max_extra=10 ** 6)
            good &= abs(x["ok"] - y["ok"]) < 1e-9 and abs(x["tokens"] - y["tokens"]) < 1e-6 and abs(x["esc"] - y["esc"]) < 1e-9 and abs(x["total"] - y["total"]) < 1e-6
    check(good, "views policies on ONE view equal the single-view evaluator (one read, 2 + 1, 2 + extra arm, 3 + 2)")
    good = True
    for r in sorted(rows)[:8]:                                 # a mixed pair, by independent brute force
        oks, toks = [], []
        for i, j in itertools.product(range(5), range(4)):
            init = [a[r][i], b[r][j]]
            if P.cls_of(init[0]) == P.cls_of(init[1]):
                oks.append(float(init[0]["ok"])), toks.append(init[0]["tokens"] + init[1]["tokens"])
                continue
            o, t = [], []
            for k in range(5):
                if k == i:
                    continue
                rs = init + [a[r][k]]
                cl = [P.cls_of(x) for x in rs]
                win = rs[2] if cl[2] in cl[:2] else rs[0]            # plurality of three, ties to the earliest
                win = rs[cl.index(cl[2])] if cl[2] in cl[:2] else rs[0]
                o.append(float(win["ok"])), t.append(sum(x["tokens"] for x in rs))
            oks.append(np.mean(o)), toks.append(np.mean(t))
        y = P.eval_views_row(P.views_policy(["a", "b"], ["a"]), {"a": a[r], "b": b[r]})
        good &= abs(y["ok"] - np.mean(oks)) < 1e-9 and abs(y["tokens"] - np.mean(toks)) < 1e-6
    check(good, "a view-diverse vote (one read under each view, a third under the first on disagreement) equals an independent brute force")
    check(P.eval_views_row(P.views_policy(["a", "a", "a"], ["b", "b"]), {"a": a[sorted(rows)[0]][:2], "b": b[sorted(rows)[0]]}) is None, "too few reads under a view -> None")
    gv = P.grammar_views(["tip", "T06", "F2048"])
    names = [P.views_name(p) for p in gv]
    check(len(gv) == 7 * 6 and len(set(names)) == len(names) and "[tip+T06]->[tip]" in names and "[T06+T06+tip]->[T06+tip]" in names and all(len(set(p["init"] + p["extra_views"])) == 2 for p in gv),
          "grammar_views: 7 votes per ordered pair of different views, unique names, every vote uses both views")
    check([l for l, _ in P.budget_policies("v")] == ["x1.00", "default", "x4.12"] and P.views_name(P.budget_policies("v")[1][1]) == "[v+v]->[v]", "budget_policies: one read, 2 + 1 on disagreement, 3 + 2")
    big = P.eval_views_row(P.views_policy(["a", "a", "b"], ["a", "b"]), {"a": a[sorted(rows)[0]] * 2, "b": b[sorted(rows)[0]] * 2}, max_init=50, max_extra=10, rng=np.random.default_rng(0))
    check(big is not None and big["n"] == 50, "large assignment spaces are subsampled with a fixed-seed rng (and refuse without one)")
    raises(lambda: P.eval_views_row(P.views_policy(["a", "a", "b"], ["a", "b"]), {"a": a[sorted(rows)[0]] * 2, "b": b[sorted(rows)[0]] * 2}, max_init=50), ValueError, "subsampling without an rng raises")
    # in the judge: a "setting" that IS the tip (a fresh half of the same reads) gains nothing at any budget; a better view gains
    rows2 = mk_rows(60)
    rng2 = np.random.default_rng(22)
    tip, pr = mk_arm(rows2, rng2, n=8)
    Jg = JV.Judge(rows2, tip, benches=BEN, policies=P.tip_policies(max_vote=5, with_guess=False))
    same_v, _ = mk_arm(rows2, rng2, n=8, p_right=pr)
    better, _ = mk_arm(rows2, rng2, n=8, p_right=pr, plant={b_: 0.12 for b_ in BEN})
    recs = {}
    for nm, arm in (("same", same_v), ("better", better)):
        for lab, pol in P.budget_policies(nm):
            recs[(nm, lab)] = Jg.score(Jg.cand_from_views(f"{nm} {lab}", pol, {nm: arm}, kind="setting"))
    check(all(abs(recs[("same", lab)]["eq"]["d"]) < 0.03 and recs[("same", lab)]["eq"]["p_gen"] > 0.02 for lab in ("x1.00", "default", "x4.12")),
          "frontier against frontier: a settings arm that is the tip itself gains ~0 at one read, at the default budget and at the last vertex")
    check(all(recs[("better", lab)]["eq"]["d"] > 0.05 and recs[("better", lab)]["eq"]["p_gen"] < 0.01 for lab in ("x1.00", "default", "x4.12"))
          and 2.0 < recs[("better", "default")]["multiple"] < 2.9 and recs[("better", "default")]["ref"]["multiple"] > 2.0,
          "frontier against frontier: a better view gains at every budget; at the default budget it is held to the tip's frontier at ITS tokens (x2.4), not to one read")
    mix = Jg.score(Jg.cand_from_views("mix", P.views_policy(["tip", "same"], ["tip"]), {"same": same_v}))
    check(abs(mix["eq"]["d"]) < 0.02, "a view-diverse vote with a view that is the tip itself gains ~0 over the tip's own vote")


# ================================================================================================ judge_v5
def t_judge():
    rng = np.random.default_rng(11)
    rows = mk_rows(74)
    tip, pr = mk_arm(rows, rng, n=8)
    Jg = JV.Judge(rows, tip, benches=BEN, policies=P.tip_policies(max_vote=4, with_guess=False))
    check(len(Jg.J) == 444 and abs(Jg.base_total - Jg.base_tokens - 4000.0) < 1e-6, "Judge: 444 rows; total tokens of one read = generated + prompt")
    aa, _ = mk_arm(rows, rng, n=4, p_right=pr)
    hom, _ = mk_arm(rows, rng, n=4, p_right=pr, plant={b: 0.10 for b in BEN})
    one, _ = mk_arm(rows, rng, n=4, p_right=pr, plant={"b2": 0.45})
    neg, _ = mk_arm(rows, rng, n=4, p_right=pr, plant={"b0": 0.12, "b1": 0.12, "b2": 0.12, "b3": 0.12, "b4": 0.12, "b5": -0.5})
    recs = Jg.decide([Jg.cand_from_arm("A/A", aa, "cell"), Jg.cand_from_arm("homogeneous", hom, "cell"), Jg.cand_from_arm("one-bench", one, "cell"),
                      Jg.cand_from_arm("negative-b5", neg, "cell")])
    by = {r["name"]: r for r in recs}
    check(not by["A/A"]["accept"] and by["homogeneous"]["accept"], "decide: an A/A arm is rejected, a homogeneous +10 pp arm is accepted")
    check(not by["one-bench"]["accept"] and by["one-bench"]["eq"]["p_macro"] < 0.05 and by["one-bench"]["eq"]["carrier"] == "b2" and "carried by b2" in by["one-bench"]["reason"],
          "decide: an effect on ONE benchmark passes p_macro, fails p_gen, and the reason names the carrier")
    check(not by["negative-b5"]["accept"] and (("b5" in by["negative-b5"]["reason"]) or by["negative-b5"]["p"] > by["negative-b5"]["holm_threshold"]),
          "decide: a significantly negative scored benchmark blocks the accept")
    check(all(r["K"] == 4 for r in recs) and sorted(r["holm_threshold"] for r in recs) == sorted(0.10 / k for k in (4, 3, 2, 1)), "decide: Holm thresholds 0.10 / (K - rank) over the judged family")
    # source exclusion (P1)
    src = Jg.decide([Jg.cand_from_arm("entry from b2", one, "entry", source=["b2"])])[0]
    check(src["scored"] == [b for b in BEN if b != "b2"] and not src["accept"] and abs(src["eq"]["d"]) < 0.03 and src["pooled"]["d"] > 0.05,
          "source exclusion: the benchmark that produced the candidate is not scored; its effect shows only in the pooled line")
    two = Jg.decide([Jg.cand_from_arm("x", hom, "entry", source=["b0", "b1", "b2", "b3"])])[0]
    check(not two["accept"] and "scored benchmarks" in two["reason"], "fewer than 3 scored benchmarks cannot be judged")
    # equal compute: a candidate that IS the tip's vote of 3 at 3x tokens gains nothing
    v3 = {r: [dict(ok=bool(P._plur(tip[r][i:i + 3])[1]), pred="A" if P._plur(tip[r][i:i + 3])[1] else "B", tokens=sum(x["tokens"] for x in tip[r][i:i + 3]),
                   prompt_tokens=12000.0, finish="stop", s=i) for i in (0, 4)] for r in Jg.J}
    rv = Jg.score(Jg.cand_from_arm("tip vote of 3 as an arm", v3, "policy"))
    check(rv["raw"]["d"] > 0.005 and abs(rv["eq"]["d"]) < 0.02 and 2.7 < rv["multiple"] < 3.3 and rv["ref"]["multiple"] > 2.0,
          "equal compute: the tip's own vote of 3 read as an arm has a raw gain and ~0 at equal compute (reference = the frontier at its tokens)")
    pc = Jg.cand_from_policy(dict(n0=2, trigger="disagree", extra="tip", final="plur"))
    rp = Jg.score(pc)
    check(abs(rp["eq"]["d"]) < 1e-9 or rp["eq"]["d"] <= 1e-9, "a tip-only policy can never beat the tip's own frontier (it is on or under the hull)")
    check(rp["total"]["multiple"] < rp["multiple"] and rp["total"]["d"] is not None, "both token conventions are scored (escalations pay less of the prompt than of the generated tokens here)")
    # too many tokens for the frontier
    huge = {r: [dict(x, tokens=x["tokens"] * 30) for x in aa[r]] for r in Jg.J}
    rh = Jg.decide([Jg.cand_from_arm("x30 tokens", huge, "cell")])[0]
    check(not rh["judged"] and not rh["accept"] and "frontier" in rh["reason"], "tokens beyond the evaluated frontier: not judged (more tip samples needed)")
    # routed entry and the removal form (base = the tip without the entry)
    routed = [r for r in Jg.J if rows[r]["bench"] in ("b0", "b1", "b3")][:120]
    wo = {r: [dict(x, ok=False, pred="B") if (rows[r]["bench"] == "b0" and x["ok"] and j == 0) else x for j, x in enumerate(tip[r][:2])] for r in routed}
    c = Jg.cand_from_arm("entry", tip, "entry", source=["b0"], routed=routed, base=Jg.system_of_arm(wo, routed=routed))
    x, info = Jg.diffs(c)
    check(all(abs(x[r]) < 1e-12 for r in Jg.J if r not in set(routed)), "routed entry: rows where it does not fire contribute exactly 0")
    r1 = routed[0]
    exp = np.mean([v["ok"] for v in tip[r1]]) - np.mean([v["ok"] for v in wo[r1]])
    check(abs(info["raw"][r1] - exp) < 1e-12 and abs(info["multiple"] - 1.0) < 0.2, "removal form: raw difference = tip (all cached reads) minus the leave-one-out reads on the routed row")
    # entry audit layers
    idp = {r: [dict(x, ok=False, pred="B") if (rows[r]["bench"] == "b0" and x["ok"]) else x for x in tip[r][:2]] for r in Jg.J if rows[r]["bench"] in ("b0", "b1")}
    gen = {r: [dict(x, ok=False, pred="B") if (x["ok"] and rng.random() < 0.25) else x for x in tip[r][:4]] for r in Jg.J}
    non = {r: list(tip[r][4:6]) for r in routed}
    ra = Jg.audit_entries([
        Jg.cand_from_arm("id-specific", tip, "entry", source=["b0"], routed=sorted(idp), base=Jg.system_of_arm(idp, routed=sorted(idp))),
        Jg.cand_from_arm("general", tip, "entry", source=["b0"], base=Jg.system_of_arm(gen)),
        Jg.cand_from_arm("nothing", tip, "entry", source=["b1"], routed=routed, base=Jg.system_of_arm(non, routed=routed))])
    check([r["layer"] for r in ra] == ["ID_SPECIFIC", "GENERAL", "UNPROVEN"], "entry audit: ID_SPECIFIC (only its source benchmark), GENERAL (all benchmarks), UNPROVEN (no effect)")
    # null audit for weight cells
    shifted, _ = mk_arm(rows, rng, n=4, p_right=pr, plant={b: 0.08 for b in BEN})
    cell, _ = mk_arm(rows, rng, n=4, p_right=pr, plant={b: 0.09 for b in BEN})
    rn = Jg.decide([Jg.cand_from_arm("cell with a shifted null", cell, "cell", null=Jg.system_of_arm(shifted))])[0]
    check(rn["null"]["audit_ok"] is False and rn["gate"] == "vs_null" and not rn["accept"],
          "null audit: when a random edit itself shifts the score, the cell is judged against the null arm (and +1 pp over it is not enough)")
    rn2 = Jg.decide([Jg.cand_from_arm("cell with a clean null", hom, "cell", null=Jg.system_of_arm(aa))])[0]
    check(rn2["null"]["audit_ok"] is True and rn2["gate"] == "eq" and rn2["accept"], "null audit: a clean null leaves the tip's cache as the reference")
    # admission test of an in-distribution entry: its effect on its SOURCE benchmark only
    rt2 = [r for r in Jg.J if rows[r]["bench"] in ("b2", "b4")][::2]
    helped = {r: [dict(x, ok=True, pred="A") if rows[r]["bench"] == "b2" else x for x in tip[r][:4]] for r in rt2}
    same4 = {r: list(tip[r][4:8]) for r in rt2}
    ch, cn = Jg.cand_from_arm("helps b2 only", helped, "entry", routed=rt2), Jg.cand_from_arm("A/A", same4, "entry", routed=rt2)
    eh, en = Jg.source_effect(ch, ["b2"]), Jg.source_effect(cn, ["b2"])
    check(eh["judged"] and eh["d"] > 0.05 and eh["p"] < 0.001 and eh["rows"] == 74 and en["p"] > 0.05 and abs(Jg.source_effect(ch, ["b4"])["d"]) < 0.03,
          "source_effect: an entry that helps on ONE benchmark is significant on that benchmark's rows (74 rows, unrouted rows count 0), an A/A entry is not, another benchmark shows nothing")
    sub_rows = set([r for r in Jg.J if rows[r]["bench"] == "b2"][:20])
    es = Jg.source_effect(ch, ["b2"], rows=sub_rows)
    check(es["rows"] == 20 and es["judged"], "source_effect on a subset of rows (the MONITOR rows of the source benchmark in the loop)")
    check(Jg.source_effect(ch, ["nope"])["judged"] is False, "source_effect: no row of the benchmark -> not judged")
    check(list(ST.holm([eh["p"], en["p"], 0.5], 0.10)) == [True, False, False], "admission family: Holm over the entries tested on their source benchmarks")
    gen_only = Jg.decide([Jg.cand_from_arm("helps b2 only", helped, "entry", source=["b2"], routed=rt2)])[0]
    check(not gen_only["accept"], "the same entry is NOT a general gain: off-source it is rejected (it can only enter the specialisation layer)")
    # printing
    txt = JV.line(by["homogeneous"], BEN)
    check("ACCEPT" in txt and "generated tokens" in txt and "total tokens with prompts" in txt and "per benchmark (scored 6" in txt and "reference" in txt,
          "line(): verdict, both token conventions, the reference used, the per-benchmark effects and the scored set")
    check("* = source, not scored" in JV.line(src, BEN), "line(): the source benchmark is marked as not scored")
    with tempfile.TemporaryDirectory() as td:
        JV.dump(recs, os.path.join(td, "x.json"))
        check(len(json.load(open(os.path.join(td, "x.json")))) == 4, "dump(): records are plain json")
    # row-role guards
    rows2 = mk_rows(20, roles=("judge", "show", "monitor"))
    tip2, _ = mk_arm(rows2, rng, n=4)
    raises(lambda: JV.Judge(rows2, tip2, roles=("judge", "show")), JV.JudgeError, "JUDGE rows are never pooled with SHOW / MONITOR rows")
    Js = JV.Judge(rows2, {r: v for r, v in tip2.items() if rows2[r]["role"] != "judge"}, benches=BEN, policies=P.tip_policies(max_vote=3, with_guess=False), roles=("show", "monitor"))
    check(len(Js.J) == 240 and all(rows2[r]["role"] != "judge" for r in Js.J), "the signal-row pre-screen runs on SHOW + MONITOR rows only")
    jrow = next(r for r in rows2 if rows2[r]["role"] == "judge")
    raises(lambda: Js.diffs(dict(name="leak", kind="cell", value={jrow: 1.0}, tokens={jrow: 1.0})), JV.JudgeError, "a JUDGE row in a signal-row candidate raises")
    raises(lambda: JV.Judge(rows2, {r: v for r, v in tip2.items() if rows2[r]["role"] != "judge"}), JV.JudgeError, "a judge without the tip's cached reads on JUDGE rows is refused")
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "learn.jsonl")
        open(p, "w").write(json.dumps(dict(id="x/1", bench="x", v4_role="ood_v2")) + "\n")
        raises(lambda: R.load_rows(p), R.RowError, "reads5.load_rows: a rows file that is not the clean 768-row learn.jsonl (sha256) is refused")
        raises(lambda: R.load_rows(p, check_sha=False), R.RowError, "reads5.load_rows: a row with role ood_v2 is refused even without the sha check")


# ================================================================================================ guess5
def t_guess():
    check(G.flag(dict(seed="guess", window=600, patterns=[]), "... I'll just guess B.") and G.flag(dict(seed="guess", window=600, patterns=[]), "my GUESSING here")
          and not G.flag(dict(seed="guess", window=600, patterns=[]), "no doubt at all"), "flag: the seed stem (guess, guessing) in the tail window")
    check(not G.flag(dict(seed="guess", window=20, patterns=[]), "guess " + "x" * 100) and G.flag(dict(seed="guess", window=2500, patterns=[]), "guess " + "x" * 100),
          "flag: only the last `window` characters count")
    check(G.flag(dict(seed="guess", window=600, patterns=["not sure"]), "I am not sure about it") and not G.flag(dict(seed="guess", window=600, patterns=["not sure"]), "knot surely"),
          "flag: added patterns match whole word sequences")
    rng = np.random.default_rng(3)
    rows = mk_rows(30, roles=("show",))
    reads = []
    filler = ["the figure shows a grid", "count the items along the row", "compare the two shapes", "option looks right overall", "so the answer follows"]
    for r, x in rows.items():
        p = float(rng.beta(2, 2))
        for s in range(8):
            ok = rng.random() < p
            t = " ".join(rng.choice(filler, 3))
            if not ok and rng.random() < 0.6:
                t += " honestly cannot tell here"            # a general pattern of within-row wrong reads (all benchmarks)
            if x["bench"] == "b4" and not ok and rng.random() < 0.9:
                t += " zebra stripes unclear"                # a pattern of wrong reads in ONE benchmark only
            if p < 0.3 and rng.random() < 0.7:
                t += " hard picture indeed"                   # marks hard ROWS (right and wrong reads alike)
            reads.append(dict(row=r, ok=bool(ok), tokens=1000.0, tail=t))
    det = G.derive(reads, rows, price=0.01)
    pats = " | ".join(det["patterns"])
    check(any(w in pats for w in ("honestly", "cannot", "tell", "here")), "derive: a tail pattern that marks the worse READS of a row in every benchmark is selected")
    check("zebra" not in pats and "stripes" not in pats, "derive: a pattern tied to one benchmark cannot enter (MIN_BENCH)")
    check("hard picture" not in pats and "indeed" not in pats, "derive: a pattern that marks hard ROWS (no within-row value) is not selected: it only costs tokens")
    check(det["windows"][det["window"]]["lobo"]["p_signflip"] <= G.KEEP_P, "derive: additions are kept only when their leave-one-benchmark-out gain over the seed is positive across benchmarks (sign-flip p <= KEEP_P)")
    kept = 0
    for sd in range(12):                                       # pure-noise tails (the fake engine's random words): additions must almost never be kept
        r2 = np.random.default_rng(100 + sd)
        nz = [dict(r_, tail=" ".join(r2.choice(filler + ["look at the figure again", "then check which option", "the same order of marks", "each side matters here"], 6))) for r_ in reads]
        kept += bool(G.derive(nz, rows, price=0.01, windows=(600,))["patterns"])
    check(kept <= 2, f"derive: on pure-noise tails additions are kept in {kept} of 12 draws (the old 'better in 4 of 6' rule kept them one time in three)")
    noise = [dict(r, tail=" ".join(rng.choice(filler, 4))) for r in reads]
    det0 = G.derive(noise, rows, price=0.01)
    check(det0["patterns"] == [], "derive: on tails without signal nothing is added to the seed (the leave-one-benchmark-out value decides)")
    rows_j = mk_rows(2, roles=("judge",))
    raises(lambda: G.derive([dict(row=next(iter(rows_j)), ok=True, tokens=1.0, tail="guess")], rows_j, 0.01), ValueError, "derive: a JUDGE row raises")


if __name__ == "__main__":
    for f in (t_stats, t_policy, t_views, t_judge, t_guess):
        f()
    print(f"\n{OK[0]} ok, {BAD[0]} failed")
    sys.exit(1 if BAD[0] else 0)
