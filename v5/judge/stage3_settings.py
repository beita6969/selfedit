"""Stage 3 (coordinator 11-11): the settings arm and the view-diverse votes on the GENERAL base (floor-only tip, 8 cached reads per row).

  signal   (needs e9_settings_signal.jsonl.gz) the signal-row pre-screen: each settings arm at its three budget points (one read; 2 reads +
           a third on disagreement = the default budget; 3 reads + 2 unless unanimous) against the tip's frontier at equal generated
           tokens ("frontier against frontier"); every view-diverse vote of the grammar; which candidates go to the JUDGE rows (macro
           gain at equal compute > 0 with p_macro <= 0.25; a setting on its default budget; the best vote per pair of views; at most 6).
  judge    (needs e10_settings_judge.jsonl.gz) the JUDGE verdicts of what the pre-screen sent, and ONLY of that: the JUDGE reads of an arm
           that was not sent are not loaded. Family = the sent candidates (Holm). Shrinkage signal -> JUDGE per candidate.
  aa       false-accept rate of "frontier against frontier" on tip-vs-tip resamples (the arm is a 4-read half of the tip's own reads).
  reuse    outcome-blind check that reused cache reads and new reads of an arm agree on JUDGE rows (answers and tokens only, never ok).
Outputs: OUT/stage3_signal.txt / .json, OUT/stage3_judge.txt / .json, OUT/stage3_aa.txt, OUT/stage3_reuse.txt.
usage: stage3_settings.py signal|judge|aa|reuse
"""
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import guess5 as G  # noqa: E402
import judge_v5 as JV  # noqa: E402
import policy5 as P  # noqa: E402
import reads5 as R  # noqa: E402
import stage1_data as SD  # noqa: E402
import stats5 as ST  # noqa: E402

B = R.BENCHES
WINDOWS = (600, 1500)
SCREEN_P, K_MAX = 0.25, 6
E9, E10 = "e9_settings_signal.jsonl.gz", "e10_settings_judge.jsonl.gz"
OUT = os.path.join(SD.HOME, "b5_restart", "stage3")


def base(rows, full, with_stock=False):
    floor, stock = SD.load_e2(rows)
    SD.classify(floor, full)
    if with_stock:
        SD.classify(stock, full)
    for xs in floor.values():
        for x in xs:
            x["guess"] = {w: G.flag(dict(seed="guess", window=w, patterns=[]), x.get("tail")) for w in WINDOWS}
    return (floor, stock) if with_stock else floor


def judges(rows, floor):
    S, J = R.rows_of(rows, R.SIGNAL_ROLES), R.rows_of(rows, "judge")
    pols = P.tip_policies(max_vote=5, with_guess=True, windows=WINDOWS)
    Js = JV.Judge(rows, {r: floor[r] for r in S}, benches=B, policies=pols, roles=R.SIGNAL_ROLES)
    return Js, S, J, pols


def arms_of(name, rows, full, role_rows, only=None):
    """arms of an export on the given rows; only = arm names to load (the others are not even parsed into reads)."""
    L = [v for v in SD._lines(name) if "row" in v and (only is None or v.get("arm") in only)]
    A = SD._arms(L, rows)
    out = {}
    for a, arm in A.items():
        arm = {r: arm[r] for r in role_rows if r in arm}
        if arm:
            out[a] = SD.classify(arm, full)
    return out


def brief(r):
    e = r["eq"]
    return dict(name=r["name"], acc=r["acc"], multiple=r["multiple"], multiple_total=(r.get("total") or {}).get("multiple"), raw=r["raw"]["d"], d=e["d"], se=e["se"],
                p_macro=e["p_macro"], p_gen=e["p_gen"], carrier=e["carrier"], npos=e["npos"], per_bench={b: e["per_bench"][b]["d"] for b in B},
                d_total=(r.get("total") or {}).get("d"), ref=r.get("ref"))


def row_text(x, lab=""):
    return (f"{lab}{x['name']:34s} acc {100 * x['acc']:6.2f} x{x['multiple']:4.2f} (total x{(x['multiple_total'] or 0):4.2f}) | raw {100 * x['raw']:+5.2f} | at equal compute {100 * x['d']:+5.2f} +- {100 * x['se']:4.2f}"
            f" p_macro {x['p_macro']:.3f} p_gen {x['p_gen']:.3f} pos {x['npos']}/6 | total-token convention {100 * (x['d_total'] or 0):+5.2f} | "
            + " ".join(f"{b[:5]} {100 * x['per_bench'][b]:+5.1f}" for b in B))


def screen(Jx, views):
    """the pre-screen computation on one row set: settings at three budgets, every view-diverse vote."""
    sets = {}
    for v, arm in sorted(views.items()):
        n = min(len(x) for x in arm.values())
        sets[v] = {lab: brief(Jx.score(Jx.cand_from_views(f"{v} {lab}", pol, {v: arm}, kind="setting"))) for lab, pol in P.budget_policies(v)
                   if len(pol["init"]) + len(pol["extra_views"]) <= n}
    votes = []
    vn = ["tip"] + sorted(views)
    for i, va in enumerate(vn):
        for vb in vn[i + 1:]:
            for pol in P.grammar_views([va, vb]):
                try:
                    r = brief(Jx.score(Jx.cand_from_views(P.views_name(pol), pol, views)))
                except ValueError:                             # a view with too few cached reads for this vote (STOCK has 4)
                    continue
                votes.append(dict(r, pair=[va, vb], policy=pol))
    return sets, votes


def cmd_signal():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    rows, full = SD.rows(), SD.full_rows()
    floor, stock = base(rows, full, with_stock=True)
    Js, S, J, pols = judges(rows, floor)
    views = arms_of(E9, rows, full, S) if os.path.exists(os.path.join(SD.EXPORTS, E9)) else {}
    views["STOCK"] = {r: stock[r] for r in S}                  # the stock view (no image floor; 4 cached reads): a view for the votes (coordinator)
    L = ["STAGE 3, signal-row pre-screen on the general base (floor-only tip v5t000; 324 SHOW + MONITOR rows; never a decision)",
         f"views ({E9}" + ("" if os.path.exists(os.path.join(SD.EXPORTS, E9)) else " NOT YET THERE") + " + the stock reads): " + ", ".join(f"{a} ({min(len(v) for v in arm.values())}-{max(len(v) for v in arm.values())} reads on {len(arm)} rows)" for a, arm in sorted(views.items())),
         "tip frontier on signal rows: " + " | ".join(f"{p[2]} {100 * p[1]:.2f} x{p[0] / Js.F.base_tokens:.2f}" for p in Js.F.vertices), ""]
    sets, votes = screen(Js, views)
    L.append("A. SETTINGS ARM, frontier against frontier (the arm's own policy at each budget against the tip's frontier at the arm's tokens)")
    sent = []
    for v, bud in sets.items():
        for lab, x in bud.items():
            L.append(row_text(x, f"  {lab:8s} "))
        d = bud["default"]
        ok = d["d"] > 0 and d["p_macro"] <= SCREEN_P and v != "STOCK"     # STOCK is a view for the votes, not a settings candidate (the floor was accepted against it)
        L.append(f"  -> {v}: " + ("PASSES the pre-screen on the default budget" if ok else "screened out (default budget)"))
        if ok:
            sent.append(dict(kind="setting", name=v, p=d["p_macro"], views=[v], policy=P.budget_policies(v)[1][1], signal=d))
    L += ["", "B. VIEW-DIVERSE VOTES (reads through two views; offline from the caches). Every vote of the grammar; * = beats the tip's frontier (d > 0, p_macro <= 0.25)"]
    best = {}
    for x in sorted(votes, key=lambda x: (x["pair"], x["p_macro"])):
        ok = x["d"] > 0 and x["p_macro"] <= SCREEN_P
        L.append(row_text(x, "  * " if ok else "    "))
        k = tuple(x["pair"])
        if ok and (k not in best or x["p_macro"] < best[k]["p_macro"]):
            best[k] = x
    for k, x in sorted(best.items()):
        sent.append(dict(kind="viewpolicy", name=x["name"], p=x["p_macro"], views=[v for v in k if v != "tip"], policy=x["policy"], signal={q: x[q] for q in x if q != "policy"}))
    sent.sort(key=lambda c: c["p"])
    for i, c in enumerate(sent):
        c["sent"] = i < K_MAX
    L += ["", f"TO THE JUDGE (the best vote per pair of views, settings on their default budget; at most {K_MAX}, smallest p_macro first):"]
    for c in sent:
        L.append(f"  {'SENT    ' if c['sent'] else 'not sent'} {c['kind']:10s} {c['name']:34s} signal {100 * c['signal']['d']:+5.2f} +- {100 * c['signal']['se']:4.2f} p_macro {c['p']:.3f}")
    if not sent:
        L.append("  nothing passes the pre-screen: no JUDGE read is used")
    L.append(f"\n({time.time() - t0:.0f} s)")
    open(os.path.join(OUT, "stage3_signal.txt"), "w").write("\n".join(L) + "\n")
    JV.dump(dict(settings=sets, votes=[{k: v for k, v in x.items()} for x in votes], sent=sent, frontier=[dict(name=p[2], tokens=p[0], acc=p[1]) for p in Js.F.vertices]),
            os.path.join(OUT, "stage3_signal.json"))
    print("\n".join(L))


def cmd_judge():
    t0 = time.time()
    rows, full = SD.rows(), SD.full_rows()
    floor, stock = base(rows, full, with_stock=True)
    Js, S, J, pols = judges(rows, floor)
    sent = [c for c in json.load(open(os.path.join(OUT, "stage3_signal.json")))["sent"] if c.get("sent")]
    need = sorted({v for c in sent for v in c["views"]})
    L = [f"STAGE 3, JUDGE verdicts (444 JUDGE rows) of the {len(sent)} candidates the signal-row pre-screen sent; JUDGE reads loaded only for the arms {need}", ""]
    if not sent:
        L.append("nothing was sent: no JUDGE read is loaded, the general tip is unchanged")
        open(os.path.join(OUT, "stage3_judge.txt"), "w").write("\n".join(L) + "\n")
        print("\n".join(L))
        return
    Jg = JV.Judge(rows, {r: floor[r] for r in J}, benches=B, policies=pols, hull_from={p[2] for p in Js.F.vertices}, hull_from_total={p[2] for p in Js.Ft.vertices})
    views = arms_of(E10, rows, full, J, only=set(need) - {"STOCK"}) if set(need) - {"STOCK"} else {}
    if "STOCK" in need:
        views["STOCK"] = {r: stock[r] for r in J}
    cands = [Jg.cand_from_views(c["name"], c["policy"], views, kind=c["kind"]) for c in sent]
    recs = Jg.decide(cands, family="stage 3: settings arm + view-diverse votes")
    out = []
    for c, r in zip(sent, recs):
        L.append(JV.line(r, B))
        kept = r["eq"]["d"] / c["signal"]["d"] if c["signal"]["d"] > 0 else None
        L.append(f"    signal rows {100 * c['signal']['d']:+.2f} +- {100 * c['signal']['se']:.2f} -> JUDGE rows {100 * r['eq']['d']:+.2f} +- {100 * r['eq']['se']:.2f}" + (f" (kept {100 * kept:.0f}%)" if kept is not None else ""))
        if c["kind"] == "setting":                              # the three budget points on JUDGE rows (reported; the default budget decides)
            for lab, pol in P.budget_policies(c["name"]):
                L.append(row_text(brief(Jg.score(Jg.cand_from_views(f"{c['name']} {lab}", pol, views, kind="setting"))), f"    {lab:8s} "))
        out.append(dict(name=c["name"], kind=c["kind"], accept=bool(r["accept"]), reason=r["reason"], signal=c["signal"], judge=brief(r), kept=kept))
        L.append("")
    acc = [o for o in out if o["accept"]]
    L.append("ACCEPTED: " + (", ".join(o["name"] for o in acc) if acc else "none: the general tip stays floor-only + the tip's own voting frontier"))
    L.append("tip frontier on JUDGE rows: " + " | ".join(f"{p[2]} {100 * p[1]:.2f} x{p[0] / Jg.F.base_tokens:.2f}" for p in Jg.F.vertices))
    L.append(f"\n({time.time() - t0:.0f} s)")
    open(os.path.join(OUT, "stage3_judge.txt"), "w").write("\n".join(L) + "\n")
    JV.dump(dict(verdicts=out, family=len(recs)), os.path.join(OUT, "stage3_judge.json"))
    print("\n".join(L))


def cmd_reuse():
    """Outcome-blind check that REUSED cache reads and NEW reads of one arm agree on the JUDGE rows (e10 marks them with "reused"). Only
    arm, row, s, pred, tokens, finish and reused are read: ok is never touched, so the JUDGE outcome of a screened-out arm stays unread.
    Per arm: reads, generated tokens, and the rate at which two reads of the same row give the same answer class, for (reused, new) pairs
    against (new, new) pairs: a cache read that came from another request (wrong view, wrong settings) would agree less with the new reads
    than the new reads agree among themselves."""
    import collections
    import itertools
    rows, full = SD.rows(), SD.full_rows()
    ec = SD._ecomp()
    J = set(R.rows_of(rows, "judge"))
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for v in SD._lines(E10):
        if "row" not in v or v.get("finish") == "abort":
            continue
        if v["row"] not in J:
            raise R.RowError(f"{E10}: row {v['row']} is not a JUDGE row")
        p = v.get("pred")
        by[v["arm"]][v["row"]].append(dict(cls=ec.answer_class(p, full[v["row"]]), tokens=float(v.get("tokens") or 0), reused=bool(v.get("reused")), s=int(v["s"])))
    L = [f"Reused against new reads on JUDGE rows ({E10}); OUTCOME-BLIND: answers and tokens only, ok is not read",
         f"{'arm':8s} {'reads':>6s} {'reused':>7s} {'new':>6s} | {'tokens reused':>13s} {'tokens new':>10s} | same answer: {'reused-new':>10s} {'new-new':>8s} {'reused-reused':>13s} | rows with both"]
    doc = {}
    for a in sorted(by):
        xs = [x for r in by[a].values() for x in r]
        ru, nw = [x for x in xs if x["reused"]], [x for x in xs if not x["reused"]]
        agree = collections.defaultdict(list)
        both = 0
        for r, rs in by[a].items():
            both += any(x["reused"] for x in rs) and any(not x["reused"] for x in rs)
            for x, y in itertools.combinations(rs, 2):
                k = "reused-new" if x["reused"] != y["reused"] else ("new-new" if not x["reused"] else "reused-reused")
                agree[(k, r)].append(float(x["cls"] == y["cls"]))
        rate = {}
        for k in ("reused-new", "new-new", "reused-reused"):
            v = [sum(z) / len(z) for (kk, r), z in agree.items() if kk == k]          # row means, then the mean over rows
            rate[k] = (sum(v) / len(v), len(v)) if v else (float("nan"), 0)
        tk = lambda q: sum(x["tokens"] for x in q) / len(q) if q else float("nan")   # noqa: E731
        # tokens matched by row (a reused read and a new read of the SAME row; the pooled means mix different rows when reuse is uneven)
        mr = [(tk([x for x in rs if x["reused"]]), tk([x for x in rs if not x["reused"]])) for rs in by[a].values()
              if any(x["reused"] for x in rs) and any(not x["reused"] for x in rs)]
        m_ru, m_nw = sum(p[0] for p in mr) / len(mr), sum(p[1] for p in mr) / len(mr)
        doc[a] = dict(reads=len(xs), reused=len(ru), new=len(nw), tokens_reused=tk(ru), tokens_new=tk(nw), tokens_reused_row_matched=m_ru, tokens_new_row_matched=m_nw,
                      agree={k: v[0] for k, v in rate.items()}, rows_both=both)
        L.append(f"{a:8s} {len(xs):6d} {len(ru):7d} {len(nw):6d} | {tk(ru):13.0f} {tk(nw):10.0f} | {'':12s} {100 * rate['reused-new'][0]:9.1f}% {100 * rate['new-new'][0]:7.1f}% {100 * rate['reused-reused'][0]:12.1f}% | {both}"
                 f" | tokens matched by row: reused {m_ru:.0f} vs new {m_nw:.0f} (x{m_nw / m_ru:.3f})")
    open(os.path.join(OUT, "stage3_reuse.txt"), "w").write("\n".join(L) + "\n")
    json.dump(doc, open(os.path.join(OUT, "stage3_reuse.json"), "w"), indent=1)
    print("\n".join(L))


# ------------------------------------------------------------------------------------------------ A/A of frontier against frontier
_G = {}


def _aa_rep(seed):
    if not _G:
        rows, full = SD.rows(), SD.full_rows()
        floor = base(rows, full)
        J = R.rows_of(rows, "judge")
        _G.update(rows=rows, J=J, floor=floor)
    rows, J, floor = _G["rows"], _G["J"], _G["floor"]
    rng = np.random.default_rng(seed)
    tipH, armH = {}, {}
    for r in J:
        p = rng.permutation(len(floor[r]))
        tipH[r] = [floor[r][i] for i in p[:4]]
        armH[r] = [floor[r][i] for i in p[4:8]]
    F = P.Frontier(tipH, J, rows, policies=[dict(n0=1, trigger="none", extra="tip"), dict(n0=2, trigger="disagree", extra="tip", final="plur"),
                                            dict(n0=2, trigger="disagree", extra="tip", final="extra"), dict(n0=3, trigger="none", extra="tip"), dict(n0=4, trigger="none", extra="tip")])
    Jg = JV.Judge(rows, tipH, benches=B, frontier=F)
    out = {}
    for name, pol in (("setting x1.00", P.views_policy(["arm"])), ("setting default", P.views_policy(["arm", "arm"], ["arm"])),
                      ("vote [tip+arm]->[tip]", P.views_policy(["tip", "arm"], ["tip"])), ("vote [tip+tip]->[arm]", P.views_policy(["tip", "tip"], ["arm"]))):
        x, info = Jg.diffs(Jg.cand_from_views(name, pol, {"arm": armH}))
        out[name] = (np.array([x[r] for r in J]), info["multiple"])
    return out


def cmd_aa(reps=400, proc=6):
    os.makedirs(OUT, exist_ok=True)
    with mp.Pool(proc) as pool:
        rs = pool.map(_aa_rep, [5000 + i for i in range(reps)], chunksize=4)
    rows = SD.rows()
    D = ST.Design(R.rows_of(rows, "judge"), rows, B)
    L = [f"A/A of 'frontier against frontier' (stage 3): the settings arm is a 4-read half of the general base's own reads, the tip is the other half; 444 JUDGE rows, {reps} replicates",
         "true gain = 0 for every line; accept rates at nominal 0.10 (Holm K = 1) and at the first Holm threshold of K = 3 and 6",
         f"{'candidate':26s} {'tokens':>7s} {'mean d':>7s} {'sd d':>6s} {'mean se':>7s} {'p_macro':>8s} {'p_gen':>7s} {'K=3':>6s} {'K=6':>6s}"]
    doc = {}
    for name in rs[0]:
        X = np.stack([r[name][0] for r in rs])
        st = D.stats(X)
        doc[name] = dict(multiple=float(np.mean([r[name][1] for r in rs])), d=float(st["d"].mean()), sd=float(st["d"].std()), se=float(st["se"].mean()),
                         p_macro=float((st["p_macro"] <= .10).mean()), p_gen=float((st["p_gen"] <= .10).mean()),
                         K3=float(((st["p_gen"] <= .10 / 3) & ~st["neg"]).mean()), K6=float(((st["p_gen"] <= .10 / 6) & ~st["neg"]).mean()))
        v = doc[name]
        L.append(f"{name:26s} x{v['multiple']:6.2f} {100 * v['d']:+7.2f} {100 * v['sd']:6.2f} {100 * v['se']:7.2f} {v['p_macro']:8.3f} {v['p_gen']:7.3f} {v['K3']:6.3f} {v['K6']:6.3f}")
    open(os.path.join(OUT, "stage3_aa.txt"), "w").write("\n".join(L) + "\n")
    json.dump(doc, open(os.path.join(OUT, "stage3_aa.json"), "w"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    {"signal": cmd_signal, "judge": cmd_judge, "aa": cmd_aa, "reuse": cmd_reuse}[sys.argv[1]]()
