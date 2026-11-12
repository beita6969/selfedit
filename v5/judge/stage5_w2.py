"""Stage 5 (coordinator 11-12 06:40 CST): ONE pre-declared candidate, the loop's second weight edit stacked on the first:
W2 = the accepted artefact vp2m512 (tip v5w001) + PRUa64, a 4,160-unit MLP cell compiled in closed form from the evolved counting procedure
g002c26 (no text is sent; requests are stock). Learn rows only. Same code path as stage 4 (frontier against frontier; sign-flip p_gen).

  RULE AS FIXED BY THE COORDINATOR BEFORE e14 (11-12, after message 11): the cell PRUa64 is compiled from g002c26, source benchmark
     vlmsbiased, so the loop's own rules apply to it. GENERAL-LAYER verdict = the OFF-SOURCE line: one read of W2 against the W1 frontier
     at equal generated tokens, scored on the five other benchmarks, p_gen over the five, K = 1 (the six-benchmark line is printed as
     information). If it fails off-source: the ADMISSION test of the boundary patch, unchanged: W2 against W1 on the JUDGE rows of the
     source benchmark (mean over its 74 JUDGE rows, equal generated tokens, one-sided sign-flip of picture clusters, Holm 0.10 in its own
     family, K = 1); admitted = a SPECIALISATION component held in weights (ID_SPECIFIC), reported apart from the general system.
  C  the comparison itself, K = 1, declared before any JUDGE read. Candidate = W2 reads (2 per row); reference = the W1 system's OWN frontier
     (its 4 reads per row: e11 on signal rows, e12 on JUDGE rows; vertices chosen on signal rows, valued on JUDGE rows) at the candidate's
     generated tokens; six benchmarks scored. DECIDING POINT: ONE READ of W2 against W1's frontier at W2's one-read tokens (the cell is an
     always-on change of the base read; a 2-read candidate has no vote of its own to place). Accepted iff p_gen <= 0.10 and no benchmark
     significantly negative. Printed beside it: the same at equal total tokens; "2 W2 reads + a W1 read on disagreement" (the default-budget
     policy with its third read taken from the parent system, since W2 has 2 reads); the agreement of the two W2 reads; the split by the
     route of the cell's source condition g002c26 (the tip t007c's cached route decisions: e1_tip_fired_learn, entry 1).
  D  information, decides nothing: W2 at one read against the old request-floor tip t007c (floor + 11 text entries, 8 cached reads).
  shrinkage  signal rows -> JUDGE rows for C (the signal rows hold the cell's fit rows: its signal number is partly in-sample).
  tip        if C is accepted: tip v5w002 (general layer = vp2m512 weights + PRUa64 units; parent v5w001).
  RULING after the W2 verdict (coordinator 11-12): a component that is always on in the weights can touch every row, so the admission test
     also needs "no benchmark significantly negative" (one-sided sign-flip p < 0.10 / 6 on the six-benchmark comparison). W2 = PRUa64 at dose 1:
     NOT admitted (babyvision). W3 = vp2m512 + the ROUTE-KEYED cell PR3a64tE_g002c26_+1.8 is the next pre-declared candidate (K = 1): the same
     lines, plus as information the rows where g002c26 does not route (difference and generated-token median against W1: the test of the gate).
usage: stage5_w2.py signal|judge [FILE|-] [W2|W3]
       W2: e13b_w2_signal.jsonl.gz / e14_w2_judge.jsonl.gz -> OUT/stage5_w2_*.txt / .json, tip_v5w002_w2.json
       W3: its files (named by the engine agent)           -> OUT/stage6_w3_*.txt / .json, tip_v5w003_w3.json
"""
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import judge_v5 as JV  # noqa: E402
import policy5 as P  # noqa: E402
import reads5 as R  # noqa: E402
import stage1_data as SD  # noqa: E402
import stage4_vp2m as S4  # noqa: E402

B = R.BENCHES
OUT = S4.OUT
E13, E14 = "e13b_w2_signal.jsonl.gz", "e14_w2_judge.jsonl.gz"
CELL_ENTRY = 1                 # index of g002c26 in the tip t007c (the counting procedure the cells were compiled from)
# the pre-declared candidates of this script (each K = 1, declared before its JUDGE reads). Both are cells on top of vp2m512 (tip v5w001).
CANDS = {
    "W2": dict(hint="w2", out="stage5_w2", tip_id="v5w002", tip_file="tip_v5w002_w2.json", sig=E13, jud=E14, title="vp2m512 + PRUa64",
               units=dict(name="PRUa64", n=4160, always_on=True, compiled_from="g002c26",
                          kind="MLP cell compiled in closed form from the evolved counting procedure g002c26 (round 2 of an iterated residual fit); always on"),
               notes=["coordinator's note 11-12, REPORT ONLY (not read by the Builder, not used by any decision): the quick held-out pass of W2 (one sample) had babyvision +6.7 and "
                      "vlmsbiased +8.3 over W1 on the ID test rows, OOD flat (-0.1). The babyvision sign there is the opposite of the JUDGE-row -6.8 +- 2.6, so that benchmark's swing is "
                      "partly its own noise (long, guess-prone reads); the ruling (not admitted) does not depend on it."]),
    "W3": dict(hint="w3", out="stage6_w3", tip_id="v5w003", tip_file="tip_v5w003_w3.json", sig=None, jud=None, title="vp2m512 + PR3a64tE_g002c26_+1.8",
               units=dict(name="PR3a64tE_g002c26_+1.8", always_on=False, compiled_from="g002c26",
                          kind="route-keyed MLP cell of round 3 compiled from g002c26: every unit gated by silu(8 (route key - 0.25)); in the weights, gated inside the model")),
}
NAME, CFG = "W2", CANDS["W2"]


def setup(cand):
    global NAME, CFG
    if cand not in CANDS:
        raise SystemExit(f"unknown candidate {cand}: {sorted(CANDS)}")
    NAME, CFG = cand, CANDS[cand]


def w1_judges(rows, full, S, J, want_judge):
    """the W1 system (vp2m512, tip v5w001) as the reference: its own reads, its own frontier (vertices from its signal reads)."""
    a_s, _ = S4.load_candidate(os.path.join(SD.EXPORTS, S4.E11), rows, full, S)
    pols = P.tip_policies(max_vote=4, with_guess=False)
    Ws = JV.Judge(rows, a_s, benches=B, policies=pols, roles=R.SIGNAL_ROLES)
    Wg = None
    if want_judge:
        a_j, _ = S4.load_candidate(os.path.join(SD.EXPORTS, S4.E12), rows, full, J)
        Wg = JV.Judge(rows, a_j, benches=B, policies=pols, hull_from={p[2] for p in Ws.F.vertices}, hull_from_total={p[2] for p in Ws.Ft.vertices})
    return Ws, Wg


def point(Jx, lab, pol, arm):
    c = Jx.cand_from_views(f"{NAME} {lab}", pol, {NAME: arm}, kind="weights")
    n = len(Jx.J)
    gen, tot = sum(c["tokens"].values()) / n, sum(c["total"].values()) / n
    r = Jx.score(c)
    out = dict(budget=lab, acc=sum(c["value"].values()) / n, gen=gen, prompt=tot - gen, total=tot, multiple=gen / Jx.base_tokens, multiple_total=tot / Jx.base_total,
               judged=bool(r.get("judged")))
    if not r.get("judged"):
        out["error"] = r.get("error")
        return out, c, r
    e, t = r["eq"], r.get("total") or {}
    out.update(raw=r["raw"]["d"], raw_se=r["raw"]["se"], d=e["d"], se=e["se"], p_macro=e["p_macro"], p_gen=e["p_gen"], carrier=e["carrier"], npos=e["npos"], neg=[b for b, _ in e["neg"]],
               per_bench={b: e["per_bench"][b]["d"] for b in B}, per_bench_se={b: e["per_bench"][b]["se"] for b in B},
               ref=r["ref"], d_total=t.get("d"), se_total=t.get("se"), p_gen_total=t.get("p_gen"), ref_total=t.get("ref"))
    return out, c, r


def subset(Jx, x, rs):
    """mean and picture-cluster se of the row differences x over a set of rows (one stratum)."""
    rs = [r for r in rs if r in x]
    if not rs:
        return None
    cl = {}
    for r in rs:
        cl.setdefault(Jx.rows[r]["pic"], []).append(x[r])
    n, G = len(rs), len(cl)
    d = sum(x[r] for r in rs) / n
    se = math.sqrt(sum((sum(v) - d * len(v)) ** 2 for v in cl.values()) * G / max(G - 1, 1)) / n
    return dict(rows=n, d=d, se=se)


def route_split(Jx, c, routed):
    """the candidate's one-read effect on rows where the g002c26 condition is routed and on the others: raw (against one W1 read) and at
    equal generated tokens; rows per benchmark."""
    x, info = Jx.diffs(c)
    out = {}
    for lab, rs in (("routed", [r for r in Jx.J if r in routed]), ("not routed", [r for r in Jx.J if r not in routed])):
        import collections
        out[lab] = dict(eq=subset(Jx, x, rs), raw=subset(Jx, info["raw"], rs), by_bench=dict(collections.Counter(Jx.rows[r]["bench"] for r in rs)))
    return out


SOURCE_BENCH = "vlmsbiased"     # the source benchmark of g002c26 in the ledger (card_source): the procedure the cell was compiled from


def off_source(Jx, c):
    """printed, not the declared decision: the same one-read comparison scored WITHOUT the source benchmark of the procedure the cell was
    compiled from (the loop's rule P1 for text entries: a candidate is not scored on the benchmark that produced it)."""
    x, _ = Jx.diffs(c)
    S5 = [b for b in B if b != SOURCE_BENCH]
    blk = Jx._block(x, S5)
    return dict(scored=S5, d=blk["d"], se=blk["se"], p_macro=blk["p_macro"], p_gen=blk["p_gen"], npos=blk["npos"], carrier=blk["carrier"])


def tails(arm, ref_arm, rows_):
    """generated tokens of the candidate against the reference: mean, median, 95th percentile, reads not ending in stop (heavy-tail check)."""
    import statistics
    c = [x["tokens"] for r in rows_ for x in arm[r]]
    w = [x["tokens"] for r in rows_ for x in ref_arm[r]]
    q = lambda v, f: sorted(v)[min(len(v) - 1, int(f * len(v)))]   # noqa: E731
    return dict(mean=[sum(c) / len(c), sum(w) / len(w)], median=[statistics.median(c), statistics.median(w)], p95=[q(c, .95), q(w, .95)], max=[max(c), max(w)],
                not_stop=[sum(x.get("finish") != "stop" for r in rows_ for x in arm[r]), sum(x.get("finish") != "stop" for r in rows_ for x in ref_arm[r])],
                reads=[len(c), len(w)])


def gate_info(Jx, arm, routed, c=None):
    """information, the test of the gate: on the rows where the source condition does NOT route (and on the routed rows likewise) the
    candidate's generated tokens against W1's: mean, median; the paired row difference with its picture-cluster se, in tokens and as the
    log ratio of the row means (robust to the long tail); the same two numbers for W1 against itself (reads s0, s1 against reads s2, s3:
    the size of this comparison when no weight changes); and the raw one-read effect per benchmark."""
    import statistics
    out = {}
    raw = Jx.diffs(c)[1]["raw"] if c is not None else None
    mt = lambda vs: sum(v["tokens"] for v in vs) / len(vs)       # noqa: E731
    for lab, rs in (("routed", [r for r in Jx.J if r in routed]), ("not routed", [r for r in Jx.J if r not in routed])):
        cc = [x["tokens"] for r in rs for x in arm[r]]
        w = [x["tokens"] for r in rs for x in Jx.tip[r]]
        g = dict(rows=len(rs), mean=[sum(cc) / len(cc), sum(w) / len(w)], median=[statistics.median(cc), statistics.median(w)])
        g["tok"] = subset(Jx, {r: mt(arm[r]) - mt(Jx.tip[r]) for r in rs}, rs)
        g["log"] = subset(Jx, {r: math.log(mt(arm[r])) - math.log(mt(Jx.tip[r])) for r in rs}, rs)
        aa = {}
        for r in rs:
            u = sorted(Jx.tip[r], key=lambda v: v["s"])
            if len(u) >= 4:
                aa[r] = (mt(u[:2]), mt(u[2:4]))
        if len(aa) == len(rs):
            g["aa_tok"] = subset(Jx, {r: v[0] - v[1] for r, v in aa.items()}, rs)
            g["aa_log"] = subset(Jx, {r: math.log(v[0]) - math.log(v[1]) for r, v in aa.items()}, rs)
        if raw is not None:
            g["by_bench"] = {b_: subset(Jx, raw, [r for r in rs if Jx.rows[r]["bench"] == b_]) for b_ in B}
        out[lab] = g
    return out


def agreement(arm, rows_):
    """the two W2 reads of a row: how often they agree, accuracy when they agree and when they do not."""
    ag = [(P.cls_of(arm[r][0]) == P.cls_of(arm[r][1]), (arm[r][0]["ok"] + arm[r][1]["ok"]) / 2) for r in rows_ if len(arm[r]) >= 2]
    a, d_ = [v for k, v in ag if k], [v for k, v in ag if not k]
    return dict(rows=len(ag), agree=len(a) / max(len(ag), 1), acc_agree=sum(a) / max(len(a), 1), acc_disagree=sum(d_) / max(len(d_), 1))


def same_index(Jx, arm, routed):
    """printed, never deciding: W2 read s against the W1 read of the SAME row and sample index (the engine uses common random seeds per
    (row, sample)): the paired difference, and how often the two reads are the same read (same answer and same number of generated
    tokens), i.e. where the added cell changes nothing."""
    x, ident = {}, {"routed": [0, 0], "not routed": [0, 0]}
    for r in Jx.J:
        w = {v["s"]: v for v in Jx.tip[r]}
        ds = []
        for v in arm[r]:
            if v["s"] in w:
                u = w[v["s"]]
                ds.append(float(v["ok"]) - float(u["ok"]))
                k = "routed" if r in routed else "not routed"
                ident[k][1] += 1
                ident[k][0] += int(P.cls_of(v) == P.cls_of(u) and v["tokens"] == u["tokens"])
        if ds:
            x[r] = sum(ds) / len(ds)
    if len(x) < len(Jx.J):
        return None
    blk = Jx._block(x, B)
    return dict(d=blk["d"], se=blk["se"], p_macro=blk["p_macro"], p_gen=blk["p_gen"], per_bench={b: blk["per_bench"][b]["d"] for b in B},
                identical={k: (v[0] / v[1] if v[1] else None) for k, v in ident.items()}, pairs={k: v[1] for k, v in ident.items()},
                routed=subset(Jx, x, [r for r in Jx.J if r in routed]), not_routed=subset(Jx, x, [r for r in Jx.J if r not in routed]))


def compute(Jx, arm, routed):
    pts = [point(Jx, "one read", P.views_policy([NAME]), arm)]
    if min(len(arm[r]) for r in Jx.J) >= 2:
        pts.append(point(Jx, "2 reads + a W1 read on disagreement", P.views_policy([NAME, NAME], ["tip"]), arm))
    return pts, route_split(Jx, pts[0][1], routed), tails(arm, Jx.tip, Jx.J), dict(agreement(arm, Jx.J), same_index=same_index(Jx, arm, routed), off_source=off_source(Jx, pts[0][1]),
                                                                                    gate=gate_info(Jx, arm, routed, pts[0][1]))


def lines_C(L, Jx, pts, split, tl, ag, title):
    L += ["", title, S4.base_line(Jx, "W1 = vp2m512 (tip v5w001; 4 reads per row)")]
    for x, _, _ in pts:
        L.append(S4.text(x))
    L.append(f"  split of the ONE-READ effect by the route of the cell's source condition g002c26 (tip t007c's cached routes):")
    for lab, v in split.items():
        if v["eq"]:
            L.append(f"    {lab:11s} {v['eq']['rows']:3d} rows: raw vs 1 W1 read {100 * v['raw']['d']:+6.2f} +- {100 * v['raw']['se']:4.2f} | at equal generated tokens {100 * v['eq']['d']:+6.2f} +- {100 * v['eq']['se']:4.2f} | rows by benchmark {v['by_bench']}")
    gt = ag.get("gate")
    if gt:
        L.append("  THE GATE (information): generated tokens " + NAME + " / W1 by route: " + "; ".join(
            f"{lab} ({v['rows']} rows) mean {v['mean'][0]:.0f} / {v['mean'][1]:.0f} (x{v['mean'][0] / v['mean'][1]:.3f}), median {v['median'][0]:.0f} / {v['median'][1]:.0f} (x{v['median'][0] / v['median'][1]:.3f})"
            for lab, v in gt.items()))
    for lab, v in (gt or {}).items():
        if v.get("tok"):
            t_, l_ = v["tok"], v["log"]
            x_ = (f"    {lab}: paired by row, {NAME} minus W1 generated tokens {t_['d']:+.0f} +- {t_['se']:.0f} ({100 * t_['d'] / v['mean'][1]:+.1f} +- {100 * t_['se'] / v['mean'][1]:.1f} % of W1's mean);"
                  f" ratio of the row means, geometric x{math.exp(l_['d']):.3f} (log {l_['d']:+.3f} +- {l_['se']:.3f})")
            if v.get("aa_tok"):
                x_ += (f" | W1 against itself (reads s0, s1 minus s2, s3): {v['aa_tok']['d']:+.0f} +- {v['aa_tok']['se']:.0f} tokens, geometric x{math.exp(v['aa_log']['d']):.3f}"
                       f" (log {v['aa_log']['d']:+.3f} +- {v['aa_log']['se']:.3f})")
            L.append(x_)
        if v.get("by_bench"):
            L.append(f"    {lab}: raw one read minus one W1 read by benchmark: " + " ".join(
                f"{b_[:5]} {100 * e['d']:+.1f}+-{100 * e['se']:.1f} ({e['rows']})" for b_, e in v["by_bench"].items() if e))
    L.append(f"  generated tokens {NAME} / W1: mean {tl['mean'][0]:.0f} / {tl['mean'][1]:.0f} (x{tl['mean'][0] / tl['mean'][1]:.2f}), median {tl['median'][0]:.0f} / {tl['median'][1]:.0f}"
             f" (x{tl['median'][0] / tl['median'][1]:.2f}), 95th percentile {tl['p95'][0]:.0f} / {tl['p95'][1]:.0f}, max {tl['max'][0]:.0f} / {tl['max'][1]:.0f},"
             f" reads not ending in stop {tl['not_stop'][0]} of {tl['reads'][0]} / {tl['not_stop'][1]} of {tl['reads'][1]}")
    L.append(f"  the two {NAME} reads agree on {100 * ag['agree']:.1f}% of rows (accuracy {100 * ag['acc_agree']:.1f} when they agree, {100 * ag['acc_disagree']:.1f} when they do not)")
    of = ag.get("off_source")
    if of:
        L.append(f"  OFF-SOURCE (the general-layer rule): one read scored WITHOUT {SOURCE_BENCH} (the source benchmark of g002c26, the procedure the cell was compiled from; rule P1 for text entries):"
                 f" {100 * of['d']:+.2f} +- {100 * of['se']:.2f} over the other five benchmarks, p_macro {of['p_macro']:.4f}, p_gen {of['p_gen']:.4f}, positive {of['npos']}/5")
    si = ag.get("same_index")
    if si:
        L.append(f"  printed, not deciding: {NAME} read s minus the W1 read of the same row and sample index (common seeds): {100 * si['d']:+.2f} +- {100 * si['se']:.2f} (p_macro {si['p_macro']:.4f}, p_gen {si['p_gen']:.4f});"
                 f" routed {100 * si['routed']['d']:+.2f} +- {100 * si['routed']['se']:.2f}, not routed {100 * si['not_routed']['d']:+.2f} +- {100 * si['not_routed']['se']:.2f};"
                 f" the two reads are the SAME read (same answer, same token count) on {100 * (si['identical']['routed'] or 0):.1f}% of routed and {100 * (si['identical']['not routed'] or 0):.1f}% of unrouted (row, sample) pairs")


def info_D(Jx, arm, t007c):
    """W2 at one read against the old request-floor tip t007c at one read (floor + 11 text entries): paired by row."""
    m = len(Jx.J)
    v = {r: sum(x["ok"] for x in arm[r]) / len(arm[r]) for r in Jx.J}
    tv = {r: sum(x["ok"] for x in t007c[r]) / len(t007c[r]) for r in Jx.J}
    blk = Jx._block({r: v[r] - tv[r] for r in Jx.J}, B)
    mean = lambda a, k: sum(sum(k(x) for x in a[r]) / len(a[r]) for r in Jx.J) / m   # noqa: E731
    g = [mean(arm, lambda x: x["tokens"]), mean(t007c, lambda x: x["tokens"])]
    p_ = [mean(arm, lambda x: float(x.get("prompt_tokens") or 0)), mean(t007c, lambda x: float(x.get("prompt_tokens") or 0))]
    return dict(acc=sum(v.values()) / m, acc_t007c=sum(tv.values()) / m, d=blk["d"], se=blk["se"], p_two_sided=min(1.0, 2 * min(blk["p_macro"], 1 - blk["p_macro"])), npos=blk["npos"],
                per_bench={b: blk["per_bench"][b]["d"] for b in B}, per_bench_se={b: blk["per_bench"][b]["se"] for b in B}, gen=g, prompt=p_, total=[g[0] + p_[0], g[1] + p_[1]])


def text_D(x):
    return (f"  one read: {NAME} {100 * x['acc']:.2f} vs t007c {100 * x['acc_t007c']:.2f}: {100 * x['d']:+.2f} +- {100 * x['se']:.2f} (two-sided p {x['p_two_sided']:.3f}; positive {x['npos']}/6) | "
            + " ".join(f"{b[:5]} {100 * x['per_bench'][b]:+5.1f}+-{100 * x['per_bench_se'][b]:3.1f}" for b in B)
            + f"\n  tokens {NAME} / t007c: generated {x['gen'][0]:.0f} / {x['gen'][1]:.0f} (x{x['gen'][0] / x['gen'][1]:.2f}), prompt {x['prompt'][0]:.0f} / {x['prompt'][1]:.0f} (x{x['prompt'][0] / x['prompt'][1]:.2f}),"
            f" total {x['total'][0]:.0f} / {x['total'][1]:.0f} (x{x['total'][0] / x['total'][1]:.2f})")


def load_all(rows, full, path, role_rows):
    arm, meta = S4.load_candidate(path, rows, full, role_rows, name_hint=CFG["hint"])
    tip, fired, _ = SD.load_e1(rows)
    SD.classify(tip, full)
    routed = {r for r, f in fired.items() if CELL_ENTRY in f}
    return arm, meta, tip, routed


def cmd_signal(path=None):
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    rows, full = SD.rows(), SD.full_rows()
    S, J = R.rows_of(rows, R.SIGNAL_ROLES), R.rows_of(rows, "judge")
    if not (path or CFG["sig"]):
        raise SystemExit(f"{NAME}: give the signal file")
    arm, meta, t007c, routed = load_all(rows, full, path or os.path.join(SD.EXPORTS, CFG["sig"]), S)
    Ws, _ = w1_judges(rows, full, S, J, False)
    pts, split, tl, ag = compute(Ws, arm, routed)
    Dm = info_D(Ws, arm, t007c)
    L = [f"{NAME} = {CFG['title']} on the SIGNAL rows (324 rows; they include the cell's fit rows): the dry computation; never a decision",
         f"candidate reads: {json.dumps(meta)}"]
    lines_C(L, Ws, pts, split, tl, ag, "C. against the W1 system's own frontier (frontier against frontier at equal generated tokens)")
    se_ = Ws.source_effect(pts[0][1], [SOURCE_BENCH])
    L.append(f"  for the record (signal rows hold the cell's fit rows): the admission form on the {SOURCE_BENCH} signal rows: {100 * se_['d']:+.2f} +- {100 * se_['se']:.2f} (p {se_['p']:.4f}; in-sample)")
    L += ["", "D. INFORMATION: against the old request-floor tip t007c (floor + 11 text entries; 8 cached reads; decides nothing)", text_D(Dm)]
    L.append(f"\n({time.time() - t0:.0f} s)")
    open(os.path.join(OUT, CFG["out"] + "_signal.txt"), "w").write("\n".join(L) + "\n")
    JV.dump(dict(meta=meta, C=[x for x, _, _ in pts], split=split, tokens=tl, agreement=ag, D=Dm), os.path.join(OUT, CFG["out"] + "_signal.json"))
    print("\n".join(L))


def cmd_judge(path=None):
    t0 = time.time()
    rows, full = SD.rows(), SD.full_rows()
    S, J = R.rows_of(rows, R.SIGNAL_ROLES), R.rows_of(rows, "judge")
    if not (path or CFG["jud"]):
        raise SystemExit(f"{NAME}: give the JUDGE file")
    arm, meta, t007c, routed = load_all(rows, full, path or os.path.join(SD.EXPORTS, CFG["jud"]), J)
    sig = json.load(open(os.path.join(OUT, CFG["out"] + "_signal.json")))
    Ws, Wg = w1_judges(rows, full, S, J, True)
    pts, split, tl, ag = compute(Wg, arm, routed)
    cand = pts[0][1]                                             # the deciding point: ONE READ
    rec = Wg.decide([dict(cand, source=[SOURCE_BENCH])], family=f"{NAME} general layer, off-source, K = 1")[0]
    acc = bool(rec["accept"])
    six = pts[0][0]
    adm = Wg.source_effect(cand, [SOURCE_BENCH]) if not acc else None
    # coordinator's ruling 11-12 (after the W2 verdict): a component in the weights can touch every row, so condition (iii) of the judge (no
    # benchmark significantly negative: one-sided sign-flip p < 0.10 / 6 on the six-benchmark comparison) applies in the admission test too
    src_ok = bool(adm and adm.get("judged") and adm["d"] > 0 and adm["p"] <= JV.ALPHA)
    admitted = bool(src_ok and not six.get("neg"))
    why_not = None if (admitted or acc) else ("significantly negative on " + ", ".join(six["neg"]) if (src_ok and six.get("neg")) else "no significant gain on the source benchmark")
    Dm = info_D(Wg, arm, t007c)
    e5 = rec["eq"]
    L = [f"VERDICT for {NAME} = {CFG['title']} ({CFG['units']['kind']}; source benchmark {SOURCE_BENCH}; stock requests, no text)",
         f"444 JUDGE rows; candidate reads: {json.dumps(meta)}", "",
         f"GENERAL-LAYER VERDICT (off-source: five benchmarks without {SOURCE_BENCH}; one read against W1's frontier at equal generated tokens; K = 1): "
         f"{'ACCEPTED' if acc else 'NOT ACCEPTED'}: {100 * e5['d']:+.2f} +- {100 * e5['se']:.2f}, p_macro {e5['p_macro']:.4f}, p_gen {e5['p_gen']:.4f}, positive {e5['npos']}/5; {rec['reason']}",
         f"  information, six benchmarks scored: {100 * six['d']:+.2f} +- {100 * six['se']:.2f}, p_macro {six['p_macro']:.4f}, p_gen {six['p_gen']:.4f} (worst without {six['carrier']}), positive {six['npos']}/6"]
    if adm is not None:
        L.append(f"ADMISSION TEST (the source benchmark's own JUDGE rows, family K = 1, Holm 0.10, sign-flip of picture clusters, equal generated tokens; AND no benchmark significantly negative): "
                 + (f"{'ADMITTED as a specialisation component held in weights (ID_SPECIFIC)' if admitted else 'NOT ADMITTED (' + str(why_not) + ')'}: {SOURCE_BENCH} {100 * adm['d']:+.2f} +- {100 * adm['se']:.2f} on its {adm['rows']} JUDGE rows"
                    f" (raw {100 * adm['raw']:+.2f}), p {adm['p']:.4f}; significantly negative benchmarks: {six.get('neg') or 'none'}"
                    + ("".join(f" ({b} {100 * six['per_bench'][b]:+.1f} +- {100 * six['per_bench_se'][b]:.1f})" for b in six.get("neg") or [])) if adm.get("judged") else f"not judged: {adm.get('error')}"))
        nr = split.get("not routed") or {}
        gt = ag.get("gate") or {}
        if nr.get("raw") and gt.get("not routed"):
            g_ = gt["not routed"]
            L.append(f"  THE GATE (information): the {nr['raw']['rows']} JUDGE rows where g002c26 does not route: {NAME} minus W1 {100 * nr['raw']['d']:+.2f} +- {100 * nr['raw']['se']:.2f} raw"
                     f" ({100 * nr['eq']['d']:+.2f} at equal generated tokens); generated tokens median {g_['median'][0]:.0f} vs {g_['median'][1]:.0f} (x{g_['median'][0] / g_['median'][1]:.3f}),"
                     f" mean x{g_['mean'][0] / g_['mean'][1]:.3f}")
        L.append(f"  rows that did not select it: the cell was fitted on SIGNAL rows only, so these {SOURCE_BENCH} JUDGE rows are themselves untouched by the fit; its signal-row number is in-sample"
                 f" ({SOURCE_BENCH} signal rows {100 * sig['C'][0]['per_bench'][SOURCE_BENCH]:+.2f}). The effect on the {SOURCE_BENCH} ID-test rows belongs to the held-out report, not to this decision.")
    lines_C(L, Wg, pts, split, tl, ag, "C. against the W1 system's own frontier (frontier against frontier at equal generated tokens; six benchmarks scored)")
    sd, jd = sig["C"][0], pts[0][0]
    kept = jd["d"] / sd["d"] if (sd.get("judged") and jd.get("judged") and sd["d"] > 0) else None
    L += ["", f"SHRINKAGE for C (one read, equal generated tokens): signal rows {100 * sd['d']:+.2f} +- {100 * sd['se']:.2f} (raw {100 * sd['raw']:+.2f}; in-sample on the cell's fit rows)"
          f" -> JUDGE rows {100 * jd['d']:+.2f} +- {100 * jd['se']:.2f} (raw {100 * jd['raw']:+.2f})" + (f"; kept {100 * kept:.0f}%" if kept is not None else "")
          + "; routed rows raw: signal " + (f"{100 * sig['split']['routed']['raw']['d']:+.2f}" if sig["split"]["routed"]["raw"] else "n/a")
          + " -> JUDGE " + (f"{100 * split['routed']['raw']['d']:+.2f}" if split["routed"]["raw"] else "n/a")]
    L += ["", f"D. INFORMATION (decides nothing): the weights-only system {NAME} against the text-and-settings system t007c (floor + 11 text entries), one read", text_D(Dm)]
    out = dict(meta=meta, candidate=NAME, accepted_general=acc, admitted_specialisation=admitted, not_admitted_because=why_not, reason=rec["reason"], C=[x for x, _, _ in pts], split=split, tokens=tl, agreement=ag, D=Dm, signal=sig,
               shrinkage=dict(signal=sd.get("d"), judge=jd.get("d"), kept=kept),
               decision=dict(point="one read", rule="off-source (five benchmarks), K = 1", scored=rec["scored"], p_gen=e5["p_gen"], p_macro=e5["p_macro"], d=e5["d"], se=e5["se"], K=1,
                             alpha=JV.ALPHA, neg=[b for b, _ in e5["neg"]], six=dict(d=six["d"], se=six["se"], p_gen=six["p_gen"], p_macro=six["p_macro"])),
               admission=adm)
    out["tip_file"] = write_tip(meta, sig, rec, pts, split, general=acc, adm=adm, admitted=admitted, why_not=why_not)
    L += ["", f"TIP FILE ({'general' if acc else 'admitted: specialisation layer in weights' if admitted else 'NOT ADMITTED: kept as a record, the system stays v5w001'}): {out['tip_file']}"]
    L.append(f"\n({time.time() - t0:.0f} s)")
    open(os.path.join(OUT, CFG["out"] + "_judge.txt"), "w").write("\n".join(L) + "\n")
    JV.dump(out, os.path.join(OUT, CFG["out"] + "_judge.json"))
    print("\n".join(L))


def write_tip(meta, sig, rec, pts, split, general, adm, admitted, why_not):
    """The candidate's tip file, parent v5w001, always written (it is the record of the verdict):
    general        the cell passed off-source -> general layer = vp2m512 weights + the cell.
    admitted       general layer = [vp2m512 weights]; SPECIALISATION layer IN WEIGHTS = [the cell, source vlmsbiased]. The served artefact
                   carries both; the general system's reported numbers are those of v5w001.
    not admitted   general layer = [vp2m512 weights], specialisation layer empty, the cell under "rejected" with every number; the
                   system stays v5w001."""
    parent = json.load(open(os.path.join(OUT, "tip_v5w001_vp2m512.json")))
    w = [json.loads(x) for x in meta["weights"]]
    art = w[0] if len(w) == 1 else w
    one = pts[0][0]
    v = dict(gen=CFG["out"], cid=CFG["units"]["name"], benches=list(B), scored=rec["scored"], d=rec["eq"]["d"], se=rec["eq"]["se"], p_gen=rec["eq"]["p_gen"],
             six_benchmarks=dict(d=one.get("d"), se=one.get("se"), p_gen=one.get("p_gen")), one_read_raw=one.get("raw"),
             routed_raw=(split["routed"]["raw"] or {}).get("d"), not_routed_raw=(split["not routed"]["raw"] or {}).get("d"),
             negative_benchmarks=list(one.get("neg") or []), per_bench=one.get("per_bench"), per_bench_se=one.get("per_bench_se"))
    if adm and adm.get("judged"):
        v.update(source_d=adm["d"], source_se=adm["se"], source_p=adm["p"], source_raw=adm["raw"], source_rows=adm["rows"], K=1,
                 untouched_rows="the cell was fitted on signal rows only: the source benchmark's JUDGE rows did not select it",
                 signal_source_d_in_sample=sig["C"][0]["per_bench"].get(SOURCE_BENCH))
    cell = dict(condition="always", cond="always", procedure=None, settings={}, ops=[], weights=art, cid=CFG["units"]["name"], units=CFG["units"], source=[SOURCE_BENCH],
                author="loop (compiled cell; pre-declared, coordinator 11-12)")
    spec_sys = dict(artefact=art, one_read_judge=one["acc"], one_read_signal=sig["C"][0]["acc"], reads_per_row=2, tokens=dict(generated=one["gen"], prompt=one["prompt"], total=one["total"]))
    entries, special, rejected, policy, system = list(parent["entries"]), [], [], dict(parent["policy"]), None
    if general:
        status = "general"
        entries.append(dict(cell, audit="GENERAL", verdict=dict(v, layer="GENERAL", basis="off-source (five benchmarks), one read against the W1 frontier at equal generated tokens, K = 1")))
        note = f"General layer = the VP2M512 weights + the cell {CFG['units']['name']} (it passed the off-source rule), stock requests, no text; the specialisation layer is empty."
        policy = dict(default=parent["policy"]["default"], extras=[], guess=None,
                      frontier=dict(vertices=["vote1"], points=[dict(name="vote1", vertex=True, multiple=1.0, multiple_total=1.0, acc_judge=one["acc"], acc_signal=sig["C"][0]["acc"])],
                                    default="vote1", report=["vote1"], base_tokens=one["gen"], base_total=one["total"], reads_per_row=2,
                                    note="2 reads per row: only the one-read point is valued; the default budget and the last vertex need 8 reads per learn row"))
    elif admitted:
        status = "admitted: specialisation layer in weights (ID_SPECIFIC)"
        special.append(dict(cell, held_in="weights", verdict=dict(v, layer="ID_SPECIFIC", basis="admission test: the source benchmark's JUDGE rows (K = 1, Holm 0.10) and no benchmark significantly negative")))
        system = spec_sys
        note = (f"General layer = the VP2M512 weights only. SPECIALISATION layer IN WEIGHTS = the cell {CFG['units']['name']} (source benchmark vlmsbiased; it failed the off-source rule and passed "
                "the admission test: positive on its source benchmark's JUDGE rows and no benchmark significantly negative). The SERVED ARTEFACT CARRIES BOTH; the GENERAL system's reported "
                "numbers are those of tip v5w001 (the vp2m512 artefact without the cell). The artefact with the cell is the specialisation system, reported separately, never as a general gain.")
    else:
        status = f"not admitted ({'always-on; ' if CFG['units'].get('always_on') else ''}{why_not})"
        rejected.append(dict(cell, verdict=dict(v, layer="NOT_ADMITTED", why=why_not, basis="general layer: off-source rule failed; admission: the source benchmark's JUDGE rows (K = 1, Holm 0.10) AND no "
                                                "benchmark significantly negative (coordinator's ruling 11-12: a component in the weights can touch every row)")))
        note = (f"NOT ADMITTED. The cell {CFG['units']['name']} is neither in the general layer (off-source rule failed) nor in the specialisation layer ({why_not}). The system stays tip v5w001 "
                "(the vp2m512 artefact without the cell); this file is the record of the verdict with every number.")
    tip = dict(id=CFG["tip_id"], status=status, born=f"{CFG['out']} (11-12)", parent="v5w001", kind="weights system", note=note,
               weights=dict(name=f"{NAME} = {CFG['title']}", artefact=art, general_artefact=parent["weights"]["artefact"], requests="stock: no min_image_tokens floor, effort medium"),
               entries=entries, special=special, rejected=rejected, policy=policy, specialisation_system=system, candidate_artefact_reads=spec_sys, notes=list(CFG.get("notes") or []),
               reads=dict(signal=sig["meta"], judge=meta))
    p = os.path.join(OUT, CFG["tip_file"])
    JV.dump(tip, p)
    return p


if __name__ == "__main__":
    # stage5_w2.py signal|judge [FILE] [CANDIDATE]   (CANDIDATE W2 by default; W3 needs its files)
    setup(sys.argv[3] if len(sys.argv) > 3 else "W2")
    {"signal": cmd_signal, "judge": cmd_judge}[sys.argv[1]](sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "-" else None)
