"""Stage 1 (b), (c): what would the v5 rule have accepted? Offline, from reads that exist (exports E1-E6 of 11-11 + the JUDGE tables of
the hand judge). Learn rows only; signals from SHOW + MONITOR, verdicts on JUDGE.

Sections of the output (OUT/verdicts.txt, verdicts.json):
  0  inputs, the in-distribution declared-guess detector
  1  the tip's frontier on SIGNAL rows and on JUDGE rows (tip-only read policies; both token conventions)
  2  signal-row pre-screen (never a decision): operators / probes / cells / escalation policies against the tip's frontier
  3  JUDGE verdicts: image floor (vs stock) | entry audit off-source | this generation's family (cells, screened policies)
  4  tip-only read policies: raw gain per vertex, generality of the increments
usage: stage1_verdicts.py [--out DIR]
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import guess5 as G  # noqa: E402
import judge_v5 as JV  # noqa: E402
import policy5 as P  # noqa: E402
import reads5 as R  # noqa: E402
import stage1_data as SD  # noqa: E402
import stats5 as ST  # noqa: E402

WINDOWS = (600, 2500)
SCREEN_P = 0.25          # signal-row pre-screen: equal-compute macro gain > 0 with p_macro <= SCREEN_P (numbers in the output)
K_MAX = 6                # at most this many screened candidates per generation go to the JUDGE rows
B = R.BENCHES


def sub(arm, ids):
    return {r: arm[r] for r in ids if r in arm}


def frontier_text(F, title):
    return [title] + ["  " + l for l in F.lines()]


def pol_rows(Jg, pols, plain, sig_vertices=None):
    """tip-only policies: raw gain over one read, position on the frontier, gain over the plain-vote hull at equal tokens."""
    out = []
    V = {p[2] for p in Jg.F.vertices}
    for pol in pols:
        n = P.name_of(pol)
        if n not in Jg.F.table:
            continue
        t = Jg.F.table[n]
        raw = {r: t[r]["ok"] - Jg.tip_v[r] for r in Jg.J}
        blk = Jg._block(raw, B)
        tok = P.macro_mean(t, Jg.rows, "tokens")
        tot = P.macro_mean(t, Jg.rows, "total")
        ref = plain.ref_at(tok)
        eqp = None
        if ref is not None:
            x = {r: t[r]["ok"] - ref[0][r] for r in Jg.J}
            eqp = Jg._block(x, B)
        out.append(dict(name=n, acc=P.macro_mean(t, Jg.rows, "ok"), multiple=tok / Jg.F.base_tokens, multiple_total=tot / Jg.Ft.base_tokens,
                        esc=P.macro_mean(t, Jg.rows, "esc"), raw=blk, vs_plain_votes=eqp, vertex=n in V,
                        vertex_signal=None if sig_vertices is None else n in sig_vertices))
    return out


def pol_line(r):
    e = r["raw"]
    pb = " ".join(f"{b[:5]} {100 * e['per_bench'][b]['d']:+5.1f}" for b in B)
    s = (f"{r['name']:30s} acc {100 * r['acc']:5.2f} tokens x{r['multiple']:4.2f} (total x{r['multiple_total']:4.2f}) esc {100 * r['esc']:4.1f}% |"
         f" raw vs 1 read {100 * e['d']:+5.2f} +- {100 * e['se']:4.2f} p_gen {e['p_gen']:.4f} | {pb} |")
    if r["vs_plain_votes"]:
        v = r["vs_plain_votes"]
        s += f" vs plain votes at equal tokens {100 * v['d']:+5.2f} +- {100 * v['se']:4.2f} p_gen {v['p_gen']:.4f} |"
    s += " VERTEX" if r["vertex"] else " dominated"
    if r["vertex_signal"] is not None:
        s += " (signal rows: " + ("vertex)" if r["vertex_signal"] else "dominated)")
    return s


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=SD.OUT)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    rows, full = SD.rows(), SD.full_rows()
    J, S = R.rows_of(rows, "judge"), R.rows_of(rows, R.SIGNAL_ROLES)
    L, doc = [], dict(t=time.time())

    # ------------------------------------------------------------------ 0 inputs
    tip, fired, ents = SD.load_e1(rows)
    SD.classify(tip, full)
    floor, stock = SD.load_e2(rows)
    SD.classify(floor, full), SD.classify(stock, full)
    wo, _ref, wo_rows, _pair = SD.load_e3(rows)
    for arm in wo.values():
        SD.classify(arm, full)
    W4 = SD.load_e4(rows)
    for arm in W4.values():
        SD.classify(arm, full)
    src = SD.load_e6()
    E5 = SD._arms(SD._lines("e5_signal_E_arms.jsonl.gz"), rows)
    zmeta = {(v["row"], int(v["s"])): v for v in SD._lines("e5_signal_E_arms.jsonl.gz") if v.get("arm") == "sig_zoom"}
    for arm in E5.values():
        SD.classify(arm, full)
    jarms, _base = R.load_tsv(SD.W_JUDGE, rows)
    for arm in jarms.values():
        SD.classify(arm, full)
    aborts = sum(8 - len(tip[r]) for r in tip)
    L += ["selfedit v5 stage 1: verdicts of the v5 judge on reads that exist (learn rows only; no OOD / ID-test row is read)",
          f"tip t007c: E-cache reads of {len(tip)} rows ({sum(len(v) for v in tip.values())} reads; {aborts} abort records dropped: those rows have 7 reads)",
          f"rows: JUDGE {len(J)} (74 per benchmark), SIGNAL {len(S)} (SHOW 180 + MONITOR 144)",
          "tokens: generated tokens gate (v4 convention); total = generated + prompt tokens of every read, printed beside it", ""]

    # the declared-guess detector, derived on SHOW + MONITOR reads
    tipS, tipJ = sub(tip, S), sub(tip, J)
    Fp = P.Frontier(tipS, S, rows, policies=[dict(n0=1, trigger="none", extra="tip"), dict(n0=2, trigger="disagree", extra="tip", final="plur"),
                                             dict(n0=2, trigger="disagree", extra="tip", final="extra"), dict(n0=3, trigger="none", extra="tip")])
    V = Fp.vertices
    price = (V[1][1] - V[0][1]) / ((V[1][0] - V[0][0]) / V[0][0])
    det = G.derive([dict(row=r, ok=x["ok"], tokens=x["tokens"], tail=x["tail"]) for r in S for x in tipS[r]], rows, price)
    json.dump(det, open(os.path.join(a.out, "guess_detector.json"), "w"), indent=1)
    SD.set_guess(tip, det, WINDOWS)
    L.append("0. DECLARED-GUESS DETECTOR (derived on SHOW + MONITOR reads; guess5.py)")
    L.append(f"   token price of the tip's own vote on signal rows: {100 * price:.2f} pp per x1 tokens ({V[0][2]} -> {V[1][2]})")
    for W, rec in det["windows"].items():
        sd, lo, fi = rec["seed"], rec["lobo"], rec["final"]
        L.append(f"   tail window {W:4d}: seed 'guess' flags {100 * sd['share']:4.1f}% of reads (accuracy {100 * sd['acc_flagged']:.1f} flagged / {100 * sd['acc_unflagged']:.1f} not);"
                 f" re-reading flagged reads: gain {100 * sd['G']:+.2f} pp at +x{sd['C']:.3f} tokens, value over the vote's price {100 * sd['V']:+.3f}")
        L.append(f"                    + selected tail patterns ({rec['candidates']} candidates): in-sample value {100 * fi['V']:+.3f} with {rec['patterns'][:4]}...;"
                 f" LEAVE-ONE-BENCHMARK-OUT value {100 * lo['V']:+.3f} (better than the seed in {lo['better_in']}/6 benchmarks) -> "
                 + ("kept" if lo.get("p_signflip", 1.0) <= G.KEEP_P and lo["V"] > sd["V"] else "NOT kept: selected patterns do not transfer across benchmarks"))
    L.append(f"   => detector = the stem 'guess' alone (no added pattern survives leave-one-benchmark-out); windows {WINDOWS} are carried as a policy parameter")
    L.append("")
    doc["guess_detector"] = dict(window=det["window"], patterns=det["patterns"], price=price,
                                 windows={str(W): dict(seed=r_["seed"], lobo=r_["lobo"], final=r_["final"], patterns=r_["patterns"]) for W, r_ in det["windows"].items()})

    # ------------------------------------------------------------------ 1 frontiers
    tpols = P.tip_policies(max_vote=5, with_guess=True, windows=WINDOWS)
    Js = JV.Judge(rows, tipS, benches=B, policies=tpols, roles=R.SIGNAL_ROLES)
    sigV = {p[2] for p in Js.F.vertices}
    sigVt = {p[2] for p in Js.Ft.vertices}
    Jg = JV.Judge(rows, tipJ, benches=B, policies=tpols, hull_from=sigV, hull_from_total=sigVt)    # vertices chosen on SIGNAL rows, valued on JUDGE rows
    plain_names = ("vote1", "vote3", "vote5")
    plainS = P.Frontier(tipS, S, rows, policies=[Js.F.pols[n] for n in plain_names], table=Js.F.table)
    plainJ = P.Frontier(tipJ, J, rows, policies=[Jg.F.pols[n] for n in plain_names], table=Jg.F.table)
    L.append("1. THE TIP'S FRONTIER (tip-only read policies, exact expectation over the 8 cached reads; * = vertex). The vertices are CHOSEN on the signal rows")
    L.append("   (upper hull there) and only VALUED on the JUDGE rows: the equal-compute reference of a candidate is the mixture of the two adjacent vertices")
    L.append("   with exactly the candidate's tokens.")
    for nm, Jx in (("SIGNAL rows (324)", Js), ("JUDGE rows (444)", Jg)):
        L.append(f"   {nm}: generated tokens | total tokens with prompts")
        Vg, Vt = {p[2] for p in Jx.F.vertices}, {p[2] for p in Jx.Ft.vertices}
        tg = {p[2]: p for p in Jx.F.points}
        tt = {p[2]: p for p in Jx.Ft.points}
        for n in sorted(tg, key=lambda n: tg[n][0]):
            L.append(f"     {n:30s} acc {100 * tg[n][1]:6.2f} | x{tg[n][0] / Jx.F.base_tokens:4.2f} {'*' if n in Vg else ' '} | x{tt[n][0] / Jx.Ft.base_tokens:4.2f} {'*' if n in Vt else ' '}")
    L.append("")
    doc["frontier"] = dict(signal=[dict(name=p[2], tokens=p[0], acc=p[1], vertex=p[2] in sigV) for p in sorted(Js.F.points)],
                           judge=[dict(name=p[2], tokens=p[0], acc=p[1], vertex=p[2] in {q[2] for q in Jg.F.vertices}) for p in sorted(Jg.F.points)],
                           base_tokens=dict(signal=Js.F.base_tokens, judge=Jg.F.base_tokens), base_total=dict(signal=Js.Ft.base_tokens, judge=Jg.Ft.base_tokens))

    # ------------------------------------------------------------------ 2 signal-row pre-screen
    FL, RANDFL, ROTFL = SD.FL, "RANDFLs8_L24-34_+0.75", "ROTFLs8_L24-34_+0.75"
    cs = []
    # operator: zoom (answer call + locate call tokens)
    zoom = {}
    for r, xs in E5["sig_zoom"].items():
        zoom[r] = []
        for x in xs:
            m = zmeta[(r, x["s"])]
            zoom[r].append(dict(x, tokens=x["tokens"] + float(m.get("loc_tokens") or 0), prompt_tokens=x["prompt_tokens"] + float(m.get("loc_prompt_tokens") or 0)))
    zshare = sum(bool(m.get("zoomed")) for m in zmeta.values()) / len(zmeta)
    cs.append(Js.cand_from_arm("zoom operator (pilot build)", zoom, "operator", note=f"zoomed on {100 * zshare:.0f}% of reads"))
    cs.append(Js.cand_from_arm("probe: entry 1 on every row", E5["sig_e1always"], "probe"))
    cs.append(Js.cand_from_arm("probe: two-sided always-on text", E5["sig_e1two_probe"], "probe"))
    hdr_rows = sorted(E5["sig_hdr_probe"])
    cs.append(Js.cand_from_arm("probe: header wording (routed rows)", E5["sig_hdr_probe"], "probe", routed=hdr_rows))
    cs.append(Js.cand_from_arm("audit: tip re-read on E (sig_tipE)", E5["sig_tipE"], "audit"))
    for nm, arm in (("audit: W-null (W engine, no edit)", "W-null"), ("null: RANDFL (random edit)", RANDFL), ("null: ROTFL (rotated edit)", ROTFL),
                    ("null: RAND1", "RAND1_bias_all_+1"), ("null: RAND2", "RAND2")):
        cs.append(Js.cand_from_arm(nm, W4[arm], "audit"))
    nullsys = Js.system_of_arm({r: W4[RANDFL][r] + [dict(x, s=x["s"] + 4) for x in W4[ROTFL][r]] for r in S})
    cs.append(Js.cand_from_arm("FL +0.75 always-on (fitted on signal traces)", W4[FL], "cell", null=nullsys))
    fl_pols = [p for p in P.grammar(("FL",), windows=WINDOWS) if p["trigger"] != "none"]
    for pol in fl_pols:
        cs.append(Js.cand_from_policy(pol, extra_arm=W4[FL]))
    srec = [Js.score(c) for c in cs]
    L.append(f"2. SIGNAL-ROW PRE-SCREEN (324 SHOW + MONITOR rows; never a decision). A candidate goes to the JUDGE rows when its macro gain at equal compute")
    L.append(f"   is positive with p_macro <= {SCREEN_P}; at most {K_MAX} per generation (smallest p first). Audit / null arms are measurements of the ruler.")
    screened = []
    for r in srec:
        e = r["eq"]
        ok = r["kind"] in ("cell", "policy", "operator", "setting", "entry") and e["d"] > 0 and e["p_macro"] <= SCREEN_P
        r["screen"] = bool(ok)
        t = r.get("total") or {}
        pb = " ".join(f"{b[:5]} {100 * e['per_bench'][b]['d']:+5.1f}" for b in B)
        L.append(f"   {r['name']:46s} {r['kind']:8s} acc {100 * r['acc']:5.2f} | generated x{r['multiple']:4.2f}: raw {100 * r['raw']['d']:+5.2f}, at equal compute {100 * e['d']:+5.2f} +- {100 * e['se']:4.2f}"
                 f" p_macro {e['p_macro']:.3f} p_gen {e['p_gen']:.3f} | total x{t.get('multiple', float('nan')):4.2f}: {100 * t.get('d', float('nan')):+5.2f} | {pb} | "
                 + ("-> JUDGE" if ok else ("screened out" if r["kind"] in ("cell", "policy", "operator", "setting", "entry") else "measurement")))
        if r.get("null"):
            n = r["null"]
            L.append(f"   {'':46s}          random-edit null (RANDFL + ROTFL, 8 reads) vs the tip's cache {100 * n['d']:+.2f} +- {100 * n['se']:.2f} at x{n['multiple']:.2f}"
                     f" (two-sided p {n['p_two_sided']:.2f}: audit {'ok' if n['audit_ok'] else 'FAILED'}); cell - null at equal compute {100 * n['vs_null']['d']:+.2f} +- {100 * n['vs_null']['se']:.2f}"
                     f" p_macro {n['vs_null']['p_macro']:.4f} p_gen {n['vs_null']['p_gen']:.4f}")
        if ok:
            screened.append(r)
    allscr = sorted([r for r in screened if r["kind"] == "policy"], key=lambda r: r["eq"]["p_macro"])
    # one policy per extra component: the escalation policies of one cell are near-duplicates (same reads, different trigger / final
    # rule); the pre-screen keeps the best of them (smallest signal-row p_macro at equal compute) so the JUDGE family stays small
    screened_pol, seen = [], set()
    for r in allscr:
        ex = r["policy"]["extra"]
        if ex not in seen:
            seen.add(ex)
            screened_pol.append(r)
    L.append(f"   policies passing the screen: {[r['name'] for r in allscr]}")
    L.append(f"   sent to the JUDGE rows (the best policy per extra component): {[r['name'] for r in screened_pol]}")
    L.append("")
    doc["prescreen"] = srec

    # ------------------------------------------------------------------ 3 JUDGE verdicts
    L.append("3. JUDGE VERDICTS (444 JUDGE rows). accept iff p_gen passes Holm at 0.10 in its family, no scored benchmark significantly negative, >= 3 scored benchmarks")
    L.append("")
    # 3a floor vs stock (the tip of generation 1 was the stock model)
    spol = P.tip_policies(max_vote=4, with_guess=False)
    Jfs = JV.Judge(rows, sub(stock, S), benches=B, policies=spol, roles=R.SIGNAL_ROLES)
    Jf = JV.Judge(rows, sub(stock, J), benches=B, policies=spol, hull_from={p[2] for p in Jfs.F.vertices}, hull_from_total={p[2] for p in Jfs.Ft.vertices})
    rf = Jf.decide([Jf.cand_from_arm("image floor min_image_tokens 4096 (g001c34)", floor, "setting")], family="generation 1 settings (K 1 shown; 9 setting cells were read)")
    L.append("3a. IMAGE FLOOR against the stock model (reference = the stock model's own frontier from its 4 cached reads)")
    L += frontier_text(Jf.F, "    stock frontier, generated tokens:")
    for r in rf:
        L.append(JV.line(r, B))
        L.append(f"    family note: p_gen {r['eq']['p_gen']:.2e}; Holm threshold with the 9 setting cells of generation 1 as the family: {0.10 / 9:.4f} -> {'ACCEPT' if r['eq']['p_gen'] <= 0.10 / 9 else 'reject'}")
    L.append("")
    doc["floor"] = rf

    # 3b entry audit
    names = {"E1": "E1 g002c26 count / size", "E2": "E2 g003c00 count types / path", "E5": "E5 g005c00 count / steps", "E6": "E6 g005c11 named view",
             "E9": "E9 g005c23 3D verify", "E11": "E11 g005c29 rule / pattern", "H": "H g003c16+g004c11+g005c13+g005c22+g005c28 heights"}
    ac = []
    for k, ix in SD.GROUPS.items():
        rr = [r for r in wo_rows[k] if r in Jg.tip]
        srcs = sorted({src[i] for i in ix})
        base = Jg.system_of_arm(wo[k], routed=rr)
        c = Jg.cand_from_arm(names[k], tipJ, "entry", source=srcs, routed=rr, base=base, unrouted_identical=True)
        ac.append(c)
    ra = Jg.audit_entries(ac)
    L.append("3b. ENTRY AUDIT: each entry (group) as a candidate on top of the tip without it (leave-one-entry-out reads, 2 samples, on its routed JUDGE rows;")
    L.append("    tip side = the 8 cached reads). Scored on the benchmarks that did NOT produce it (source = the benchmark of the proposer's card, from the ledger).")
    for r in ra:
        L.append(JV.line(r, B))
        p = r["pooled"]
        L.append(f"    all six benchmarks (source included): {100 * p['d']:+5.2f} +- {100 * p['se']:4.2f} p_macro {p['p_macro']:.4f} | routed JUDGE rows {r['routed_rows']}")
    L.append("")
    doc["entry_audit"] = ra

    # 3c this generation's family on tip t007c
    FLj, XSj = jarms[SD.FL], jarms[SD.XS]
    cg = [Jg.cand_from_arm("FL +0.75 always-on", FLj, "cell"), Jg.cand_from_arm("XS +0.75 always-on", XSj, "cell")]
    for r in screened_pol[:K_MAX]:
        cg.append(Jg.cand_from_policy(r["policy"], extra_arm=FLj))
    rg = Jg.decide(cg, family="candidates on t007c (2 cells read on JUDGE rows + the screened policies)")
    L.append("3c. CANDIDATES ON THE TIP t007c: the two weight cells that were read on the JUDGE rows (always-on) and the escalation policies that pass the pre-screen")
    for r in rg:
        L.append(JV.line(r, B))
    # for the record: every FL policy of the grammar on JUDGE rows (not in the family: no extra reads were needed, shown for the reader)
    L.append("    (for the record, not in the family) every FL escalation policy of the grammar on the JUDGE rows:")
    allp = [Jg.score(Jg.cand_from_policy(p, extra_arm=FLj)) for p in fl_pols]
    for r in allp:
        e = r["eq"]
        L.append(f"      {r['name']:28s} acc {100 * r['acc']:5.2f} x{r['multiple']:4.2f} esc {100 * (r['esc'] or 0):4.1f}% raw {100 * r['raw']['d']:+5.2f} | eq {100 * e['d']:+5.2f} +- {100 * e['se']:4.2f} p_macro {e['p_macro']:.3f} p_gen {e['p_gen']:.3f}"
                 f" | total x{r['total']['multiple']:4.2f}: {100 * r['total']['d']:+5.2f} | " + " ".join(f"{b[:5]} {100 * e['per_bench'][b]['d']:+5.1f}" for b in B))
    wn = Jg.score(Jg.cand_from_arm("audit: W-null on JUDGE rows", jarms[SD.WNULL], "audit"))
    L.append(f"    audit: W-null (W engine, no edit; one fixed draw per row and sample) vs the tip's E cache on JUDGE rows: {100 * wn['raw']['d']:+.2f} +- {100 * wn['raw']['se']:.2f}"
             f" at x{wn['multiple']:.2f} (two-sided p {min(1.0, 2 * min(wn['raw']['p_macro'], 1 - wn['raw']['p_macro'])):.3f})")
    # sensitivity: the same two cells against the W-null reads as the tip (same engine; its own frontier from 4 reads)
    Jw = JV.Judge(rows, sub(jarms[SD.WNULL], J), benches=B, policies=P.tip_policies(max_vote=4, with_guess=False), hull_from={p[2] for p in Jfs.F.vertices} & {"vote1", "n2:disagree->tip", "n2:disagree->tip:extra", "vote4"})
    for nm, arm in (("FL +0.75 always-on", FLj), ("XS +0.75 always-on", XSj)):
        rw = Jw.score(Jw.cand_from_arm(nm, arm, "cell"))
        e = rw["eq"]
        L.append(f"    sensitivity, {nm} against W-null as the tip (its own frontier, 4 reads): raw {100 * rw['raw']['d']:+.2f}, at equal compute {100 * e['d']:+.2f} +- {100 * e['se']:.2f}"
                 f" p_macro {e['p_macro']:.4f} p_gen {e['p_gen']:.4f} (x{rw['multiple']:.2f}; reference {JV.ref_text(rw['ref'])})")
        doc.setdefault("sensitivity_wnull", []).append(rw)
    L.append("")
    doc["generation"] = rg
    doc["fl_policies_judge"] = allp
    doc["wnull_judge"] = wn

    # ------------------------------------------------------------------ 4 tip-only policies
    L.append("4. TIP-ONLY READ POLICIES (they ARE the reference class; no extra read is needed). Raw gain over one read, and over plain votes (1 / 3 / 5) at equal tokens")
    pj = pol_rows(Jg, tpols, plainJ, sigV)
    ps = pol_rows(Js, tpols, plainS)
    L.append("   JUDGE rows:")
    L += ["   " + pol_line(r) for r in sorted(pj, key=lambda r: r["multiple"])]
    L.append("   SIGNAL rows:")
    L += ["   " + pol_line(r) for r in sorted(ps, key=lambda r: r["multiple"])]
    # increments along the JUDGE hull
    L.append("   increments along the hull (JUDGE rows; is spending the next step of tokens a general gain?):")
    Vj = Jg.F.vertices
    inc = []
    for k in range(1, len(Vj)):
        a_, b_ = Jg.F.table[Vj[k - 1][2]], Jg.F.table[Vj[k][2]]
        blk = Jg._block({r: b_[r]["ok"] - a_[r]["ok"] for r in Jg.J}, B)
        inc.append(dict(lo=Vj[k - 1][2], hi=Vj[k][2], multiple=Vj[k][0] / Jg.F.base_tokens, blk=blk))
        L.append(f"     {Vj[k - 1][2]} -> {Vj[k][2]} (x{Vj[k][0] / Jg.F.base_tokens:.2f}): {100 * blk['d']:+5.2f} +- {100 * blk['se']:4.2f} p_macro {blk['p_macro']:.4f} p_gen {blk['p_gen']:.4f}"
                 f" positive {blk['npos']}/6 | " + " ".join(f"{b[:5]} {100 * blk['per_bench'][b]['d']:+5.1f}" for b in B))
    doc["tip_policies"] = dict(judge=pj, signal=ps, increments=inc)
    L.append("")

    # ------------------------------------------------------------------ 5 the general base (floor-only) and the specialisation layer on top
    for xs in floor.values():
        for x in xs:
            x["guess"] = {w: G.flag(dict(det, window=w, patterns=[]), x.get("tail")) for w in WINDOWS}
    fS, fJ = sub(floor, S), sub(floor, J)
    Jbs = JV.Judge(rows, fS, benches=B, policies=tpols, roles=R.SIGNAL_ROLES)
    Jbg = JV.Judge(rows, fJ, benches=B, policies=tpols, hull_from={p[2] for p in Jbs.F.vertices}, hull_from_total={p[2] for p in Jbs.Ft.vertices})
    L.append("5. THE GENERAL BASE = floor-only settings (min_image_tokens 4096, no text entry; 8 cached reads on every learn row) and the SPECIALISATION layer on top")
    L.append("   frontier of the general base (vertices chosen on its signal rows, valued on JUDGE rows): generated tokens | total tokens")
    Vg, Vt = {p[2] for p in Jbg.F.vertices}, {p[2] for p in Jbg.Ft.vertices}
    tg = {p[2]: p for p in Jbg.F.points}
    tt = {p[2]: p for p in Jbg.Ft.points}
    ts_ = {p[2]: p for p in Jbs.F.points}
    for n in sorted(tg, key=lambda n: tg[n][0]):
        if n in Vg or n in Vt or n in ("vote1", "vote3", "vote5", "n2:disagree->tip", "n3:disagree->tipx2"):
            L.append(f"     {n:30s} JUDGE acc {100 * tg[n][1]:6.2f} (signal {100 * ts_[n][1]:6.2f}) | x{tg[n][0] / Jbg.F.base_tokens:4.2f} {'*' if n in Vg else ' '} | x{tt[n][0] / Jbg.Ft.base_tokens:4.2f} {'*' if n in Vt else ' '}")
    L.append("   budget points on the JUDGE rows: general base | + specialisation layer (t007c = the base + the 11 routed entries) | increment of the layer, per benchmark")
    pts = []
    for n, lab in (("vote1", "x1.00  one read"), ("n2:disagree->tip", "x2.42  2 reads + third on disagreement"), ("n3:disagree->tipx2", "x4.12  3 reads + 2 unless unanimous")):
        a_, b_ = Jbg.F.table[n], Jg.F.table[n]
        blk = Jbg._block({r: b_[r]["ok"] - a_[r]["ok"] for r in J}, B)
        raw = Jbg._block({r: a_[r]["ok"] - Jbg.tip_v[r] for r in J}, B)
        ma, mb = P.macro_mean(a_, rows, "ok"), P.macro_mean(b_, rows, "ok")
        pts.append(dict(policy=n, base=ma, base_multiple=P.macro_mean(a_, rows, "tokens") / Jbg.F.base_tokens, with_special=mb,
                        special_multiple=P.macro_mean(b_, rows, "tokens") / Jbg.F.base_tokens, increment=blk, base_gain_over_one_read=raw))
        L.append(f"     {lab:40s} base {100 * ma:6.2f} (x{pts[-1]['base_multiple']:4.2f}; over one read {100 * raw['d']:+5.2f} +- {100 * raw['se']:4.2f}, p_gen {raw['p_gen']:.4f}, positive {raw['npos']}/6)"
                 f" | with the layer {100 * mb:6.2f} | layer {100 * blk['d']:+5.2f} +- {100 * blk['se']:4.2f} (p_macro {blk['p_macro']:.4f}, p_gen {blk['p_gen']:.4f}, carried by {blk['carrier']}):"
                 + " ".join(f" {b[:5]} {100 * blk['per_bench'][b]['d']:+5.1f}" for b in B))
    L.append("   The layer's increment sits on vlmsbiased (entry 1's source benchmark); it is reported as in-distribution specialisation, never as a general gain.")
    # the same increment on the SIGNAL rows (the entries were accepted on JUDGE rows: their JUDGE-row effect carries the selection)
    a_, b_ = Jbs.F.table["vote1"], Js.F.table["vote1"]
    blk = Jbs._block({r: b_[r]["ok"] - a_[r]["ok"] for r in S}, B)
    L.append(f"   the layer's increment on the SIGNAL rows (one read; rows it was not selected on): {100 * blk['d']:+5.2f} +- {100 * blk['se']:4.2f} (p_macro {blk['p_macro']:.4f}):"
             + " ".join(f" {b[:5]} {100 * blk['per_bench'][b]['d']:+5.1f}" for b in B))
    doc["special_layer_signal_rows"] = blk
    doc["general_base"] = dict(frontier=[dict(name=p[2], tokens=p[0], acc=p[1], vertex=p[2] in Vg) for p in sorted(Jbg.F.points)], points=pts,
                               base_tokens=Jbg.F.base_tokens, signal_vertices=sorted(p[2] for p in Jbs.F.vertices))
    L.append("")
    L.append(f"(computed in {time.time() - t0:.0f} s)")
    open(os.path.join(a.out, "verdicts.txt"), "w").write("\n".join(L) + "\n")
    JV.dump(doc, os.path.join(a.out, "verdicts.json"))
    print("\n".join(L))


if __name__ == "__main__":
    main()
