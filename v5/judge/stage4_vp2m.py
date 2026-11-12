"""Stage 4 (coordinator 11-12 03:35 CST): ONE pre-declared candidate, a real weight change: VP2M512 = the model artefact whose vision
patch-embedding kernel is resampled to half the patch edge, with its own preprocessor bounds (2,048-4,096 merged image tokens); requests are
STOCK (no min_image_tokens floor, effort medium). Learn rows only. No new statistic: the frontier-against-frontier path of stage 3
(judge_v5.Judge.cand_from_views + policy5.budget points), p_gen by sign-flip of picture clusters, K = 1 (declared before any JUDGE read).

  A  the acceptance test. Candidate system = VP2M512 weights with stock requests; reference = the STOCK model's own frontier (one read;
     2 reads + a third on disagreement; the last vertex its 4 cached reads allow) at the candidate's generated tokens; all six benchmarks
     scored. Accepted on the DEFAULT budget (as the settings arm); one read, default budget and the vote of 4 are printed, per benchmark,
     with generated, prompt and total tokens.
  B  information, decides nothing. The same candidate against the request-floor base v5t000 (floor-only tip, 8 cached reads), frontier
     against frontier, tokens in the three conventions: is the weight form within noise of the request floor, at what total-token ratio.
  shrinkage  signal rows -> JUDGE rows for A (default budget).
  tip        if A is accepted: the tip file of the "weights" system (general layer = the VP2M512 weights + its read policy from its own
             reads; the request floor is NOT in it).
usage: stage4_vp2m.py signal [FILE]   (e11_vp2m512_signal.jsonl.gz; 324 signal rows x 4)   -> OUT/stage4_signal.txt / .json
       stage4_vp2m.py judge  [FILE]   (e12_vp2m512_judge.jsonl.gz; 444 JUDGE rows x 4)      -> OUT/stage4_judge.txt / .json [+ tip file]
"""
import gzip
import hashlib
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

B = R.BENCHES
WINDOWS = (600, 1500)
SCREEN_P = 0.25
E11, E12 = "e11_vp2m512_signal.jsonl.gz", "e12_vp2m512_judge.jsonl.gz"
OUT = os.environ.get("STAGE4_OUT") or os.path.join(SD.HOME, "b5_restart", "stage4")     # STAGE4_OUT: stand-in runs write elsewhere
NAME = "VP2M512"
DEFAULT_POLICY = "n2:disagree->tip"


def budgets(view, n):
    """the three printed points of a view with n cached reads per row: one read, the default budget (2 reads + a third on disagreement),
    and the last point its reads allow (the vote of n; with >= 5 reads the stage-3 point 3 reads + 2 unless unanimous)."""
    out = [("one read", P.views_policy([view])), ("default budget", P.views_policy([view, view], [view]))]
    if n >= 5:
        out.append(("3 reads + 2 unless unanimous", P.views_policy([view, view, view], [view, view])))
    elif n >= 4:
        out.append((f"vote of {n}", P.views_policy([view] * n)))
    return out


def load_candidate(path, rows, full, role_rows, name_hint=None):
    """-> arm on role_rows, meta (weights, image tokens, sha256). A row outside the expected role raises; a held-out row cannot be loaded."""
    raw = open(path, "rb").read()
    L = [json.loads(l) for l in gzip.open(path, "rt") if l.strip()]
    rs = set(role_rows)
    bad = sorted({v["row"] for v in L if "row" in v and v["row"] not in rs})
    if bad:
        raise R.RowError(f"{path}: {len(bad)} rows are not in the expected row set (e.g. {bad[0]})")
    arms = SD._arms([v for v in L if "row" in v], rows)
    if len(arms) != 1:                                     # several arms in the file: only the pre-declared one is read, by its name
        pick = [a for a in arms if (name_hint or NAME).lower() in a.lower()]
        if len(pick) != 1:
            raise ValueError(f"{path}: expected ONE arm (the pre-declared candidate {name_hint or NAME}), found {sorted(arms)}")
        L = [v for v in L if v.get("arm") == pick[0]]
        arms = {pick[0]: arms[pick[0]]}
    name, arm = next(iter(arms.items()))
    SD.classify(arm, full)
    it = [v.get("image_tokens") for v in L if v.get("image_tokens") is not None]
    it = [sum(x) if isinstance(x, list) else x for x in it]
    meta = dict(file=os.path.basename(path), path=os.path.abspath(path), sha256=hashlib.sha256(raw).hexdigest(), arm=name, lines=len(L), rows=len(arm),
                reads_per_row=[min(len(v) for v in arm.values()), max(len(v) for v in arm.values())],
                weights=sorted({json.dumps(v.get("weights"), sort_keys=True) for v in L if "weights" in v}),
                settings=sorted({json.dumps(v.get("settings"), sort_keys=True) for v in L if "settings" in v}),
                image_tokens=dict(min=min(it), mean=sum(it) / len(it), max=max(it)) if it else None,
                finish=dict(__import__("collections").Counter(v.get("finish") for v in L if "row" in v)))
    miss = [r for r in role_rows if r not in arm]
    if miss:
        raise ValueError(f"{path}: {len(miss)} of {len(role_rows)} rows have no read (e.g. {miss[0]})")
    return arm, meta


def references(rows, full):
    """STOCK arm (4 cached reads per learn row) and the request-floor base v5t000 (8 reads, guess flags by the seed stem)."""
    floor, stock = SD.load_e2(rows)
    SD.classify(floor, full)
    SD.classify(stock, full)
    for xs in floor.values():
        for x in xs:
            x["guess"] = {w: G.flag(dict(seed="guess", window=w, patterns=[]), x.get("tail")) for w in WINDOWS}
    return stock, floor


def judges(rows, stock, floor, S, J, want_judge):
    """(signal judge, JUDGE judge or None) for the STOCK reference and for the floor base; vertices chosen on signal rows."""
    sp = P.tip_policies(max_vote=4, with_guess=False)
    fp = P.tip_policies(max_vote=5, with_guess=True, windows=WINDOWS)
    Ss = JV.Judge(rows, {r: stock[r] for r in S}, benches=B, policies=sp, roles=R.SIGNAL_ROLES)
    Fs = JV.Judge(rows, {r: floor[r] for r in S}, benches=B, policies=fp, roles=R.SIGNAL_ROLES)
    Sg = Fg = None
    if want_judge:
        Sg = JV.Judge(rows, {r: stock[r] for r in J}, benches=B, policies=sp, hull_from={p[2] for p in Ss.F.vertices}, hull_from_total={p[2] for p in Ss.Ft.vertices})
        Fg = JV.Judge(rows, {r: floor[r] for r in J}, benches=B, policies=fp, hull_from={p[2] for p in Fs.F.vertices}, hull_from_total={p[2] for p in Fs.Ft.vertices})
    return Ss, Fs, Sg, Fg


def point(Jx, lab, pol, arm):
    """one budget point of the candidate against the reference frontier of Jx. -> record (None fields when the frontier has no value there)."""
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
    out.update(raw=r["raw"]["d"], d=e["d"], se=e["se"], p_macro=e["p_macro"], p_gen=e["p_gen"], carrier=e["carrier"], npos=e["npos"], neg=[b for b, _ in e["neg"]],
               p_bench_t=e["p_bench_t"], per_bench={b: e["per_bench"][b]["d"] for b in B}, per_bench_se={b: e["per_bench"][b]["se"] for b in B},
               ref=r["ref"], d_total=t.get("d"), se_total=t.get("se"), p_gen_total=t.get("p_gen"), ref_total=t.get("ref"))
    return out, c, r


def text(x):
    s = (f"  {x['budget']:28s} acc {100 * x['acc']:6.2f} | tokens generated {x['gen']:6.0f} + prompt {x['prompt']:6.0f} = total {x['total']:6.0f}"
         f" (x{x['multiple']:4.2f} generated, x{x['multiple_total']:4.2f} total of one reference read)")
    if not x["judged"]:
        return s + f" | NOT REFERENCED: {x.get('error')}"
    s += (f"\n  {'':28s} reference {JV.ref_text(x['ref'])} | raw vs 1 reference read {100 * x['raw']:+6.2f} | at equal generated tokens {100 * x['d']:+6.2f} +- {100 * x['se']:4.2f}"
          f" p_macro {x['p_macro']:.4f} p_gen {x['p_gen']:.4f} (worst without {x['carrier']}) positive {x['npos']}/6"
          + (f" | NEGATIVE: {x['neg']}" if x["neg"] else "")
          + f"\n  {'':28s} per benchmark: " + " ".join(f"{b[:5]} {100 * x['per_bench'][b]:+5.1f}+-{100 * x['per_bench_se'][b]:3.1f}" for b in B))
    if x.get("d_total") is not None:
        s += f"\n  {'':28s} at equal TOTAL tokens (printed, not gating): {100 * x['d_total']:+6.2f} +- {100 * x['se_total']:4.2f} p_gen {x['p_gen_total']:.4f} | reference {JV.ref_text(x['ref_total'])}"
    return s


def base_line(Jx, name):
    V = {p[2]: p for p in Jx.F.points}
    one = Jx.F.table["vote1"]
    tot = sum(v["total"] for v in one.values()) / len(one)
    return (f"  {name}: one read {100 * V['vote1'][1]:.2f} (generated {Jx.base_tokens:.0f} + prompt {tot - Jx.base_tokens:.0f} = total {tot:.0f}); frontier vertices: "
            + " | ".join(f"{p[2]} {100 * p[1]:.2f} x{p[0] / Jx.F.base_tokens:.2f}" for p in Jx.F.vertices))


def compute(Jx, arm, Jb):
    n = min(len(arm[r]) for r in Jx.J)
    A = [point(Jx, lab, pol, {r: arm[r] for r in Jx.J}) for lab, pol in budgets(NAME, n)]
    Bi = [point(Jb, lab, pol, {r: arm[r] for r in Jb.J}) for lab, pol in budgets(NAME, n)]
    return A, Bi


def section(L, title, pts, Jx, name):
    L += ["", title, base_line(Jx, name)]
    for x, _, _ in pts:
        L.append(text(x))


def like_for_like(Jb, arm):
    """B, the SAME policy on both systems (decides nothing): candidate policy value minus the request-floor base's value of the same policy
    (the base's exact expectation over its cached reads), paired by row; two-sided sign-flip p; token ratios in the three conventions."""
    n = min(len(arm[r]) for r in Jb.J)
    names = {"one read": "vote1", "default budget": "n2:disagree->tip", f"vote of {n}": f"vote{n}", "3 reads + 2 unless unanimous": "n3:disagree->tipx2"}
    out = []
    for lab, pol in budgets(NAME, n):
        fn = names.get(lab)
        if fn not in Jb.F.table:
            continue
        c = Jb.cand_from_views(f"{NAME} {lab}", pol, {NAME: {r: arm[r] for r in Jb.J}}, kind="weights")
        ft = Jb.F.table[fn]
        blk = Jb._block({r: c["value"][r] - ft[r]["ok"] for r in Jb.J}, B)
        m = len(Jb.J)
        cg, ct = sum(c["tokens"].values()) / m, sum(c["total"].values()) / m
        fg, ftot = sum(v["tokens"] for v in ft.values()) / m, sum(v["total"] for v in ft.values()) / m
        out.append(dict(budget=lab, policy=fn, acc=sum(c["value"].values()) / m, acc_floor=sum(v["ok"] for v in ft.values()) / m, d=blk["d"], se=blk["se"],
                        p_two_sided=min(1.0, 2 * min(blk["p_macro"], 1 - blk["p_macro"])), npos=blk["npos"], per_bench={b: blk["per_bench"][b]["d"] for b in B},
                        gen_ratio=cg / fg, prompt_ratio=(ct - cg) / (ftot - fg), total_ratio=ct / ftot, gen=cg, prompt=ct - cg, total=ct, floor_gen=fg, floor_prompt=ftot - fg, floor_total=ftot))
    return out


def lfl_text(x):
    return (f"  {x['budget']:28s} weights {100 * x['acc']:6.2f} vs request floor {100 * x['acc_floor']:6.2f}: {100 * x['d']:+5.2f} +- {100 * x['se']:4.2f} (two-sided p {x['p_two_sided']:.3f}; positive {x['npos']}/6)"
            f" | tokens weights / floor: generated {x['gen']:.0f} / {x['floor_gen']:.0f} = x{x['gen_ratio']:.2f}, prompt {x['prompt']:.0f} / {x['floor_prompt']:.0f} = x{x['prompt_ratio']:.2f},"
            f" total {x['total']:.0f} / {x['floor_total']:.0f} = x{x['total_ratio']:.2f} | " + " ".join(f"{b[:5]} {100 * x['per_bench'][b]:+5.1f}" for b in B))


def cmd_signal(path=None):
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    rows, full = SD.rows(), SD.full_rows()
    S, J = R.rows_of(rows, R.SIGNAL_ROLES), R.rows_of(rows, "judge")
    arm, meta = load_candidate(path or os.path.join(SD.EXPORTS, E11), rows, full, S)
    stock, floor = references(rows, full)
    Ss, Fs, _, _ = judges(rows, stock, floor, S, J, False)
    A, Bi = compute(Ss, arm, Fs)
    L = [f"STAGE 4, {NAME} on the SIGNAL rows (324 SHOW + MONITOR rows): the dry computation of the verdict; never a decision",
         f"candidate reads: {json.dumps(meta)}"]
    section(L, "A. against the STOCK model's own frontier (frontier against frontier at equal generated tokens)", A, Ss, "STOCK (4 cached reads)")
    d = A[1][0]
    ok = bool(d["judged"] and d["d"] > 0 and d["p_macro"] <= SCREEN_P)
    L.append(f"  -> pre-screen rule (default budget: gain > 0 with p_macro <= {SCREEN_P}): {'PASSES' if ok else 'does NOT pass'} (the candidate is pre-declared; its JUDGE reads are read whatever this line says)")
    section(L, "B. INFORMATION: against the request-floor base v5t000 (floor-only tip; decides nothing)", Bi, Fs, "request floor v5t000 (8 cached reads)")
    lfl = like_for_like(Fs, arm)
    L += ["  the SAME policy on both systems (the candidate has 4 reads per row: its last point is a plain vote of 4, the base's frontier there is the early-stopping vote of 5):"] + [lfl_text(x) for x in lfl]
    L.append(f"\n({time.time() - t0:.0f} s)")
    open(os.path.join(OUT, "stage4_signal.txt"), "w").write("\n".join(L) + "\n")
    JV.dump(dict(meta=meta, A=[x for x, _, _ in A], B=[x for x, _, _ in Bi], B_same_policy=lfl, prescreen_pass=ok), os.path.join(OUT, "stage4_signal.json"))
    print("\n".join(L))


def cmd_judge(path=None):
    t0 = time.time()
    rows, full = SD.rows(), SD.full_rows()
    S, J = R.rows_of(rows, R.SIGNAL_ROLES), R.rows_of(rows, "judge")
    arm, meta = load_candidate(path or os.path.join(SD.EXPORTS, E12), rows, full, J)
    sig = json.load(open(os.path.join(OUT, "stage4_signal.json")))
    stock, floor = references(rows, full)
    Ss, Fs, Sg, Fg = judges(rows, stock, floor, S, J, True)
    A, Bi = compute(Sg, arm, Fg)
    # the acceptance test: the default budget, K = 1 (the candidate was declared before any JUDGE read)
    rec = Sg.decide([A[1][1]], family=f"{NAME}, pre-declared, K = 1")[0]
    acc = bool(rec["accept"])
    L = [f"STAGE 4, VERDICT for {NAME} (a real weight change: vision patch-embedding kernel at half the patch edge + its preprocessor bounds; STOCK requests)",
         f"444 JUDGE rows; candidate reads: {json.dumps(meta)}", "",
         f"VERDICT A: {'ACCEPTED' if acc else 'NOT ACCEPTED'} on the default budget: {rec['reason']}"]
    section(L, "A. against the STOCK model's own frontier (frontier against frontier at equal generated tokens; all six benchmarks scored; K = 1)", A, Sg, "STOCK (4 cached reads)")
    sd, jd = sig["A"][1], A[1][0]
    kept = jd["d"] / sd["d"] if (sd.get("judged") and jd.get("judged") and sd["d"] > 0) else None
    L += ["", "SHRINKAGE for A (default budget, equal generated tokens): signal rows "
          + (f"{100 * sd['d']:+.2f} +- {100 * sd['se']:.2f}" if sd.get("judged") else "n/a") + " -> JUDGE rows "
          + (f"{100 * jd['d']:+.2f} +- {100 * jd['se']:.2f}" if jd.get("judged") else "n/a") + (f" (kept {100 * kept:.0f}%)" if kept is not None else "")
          + "; one read: signal " + (f"{100 * sig['A'][0]['d']:+.2f}" if sig["A"][0].get("judged") else "n/a") + " -> JUDGE " + (f"{100 * A[0][0]['d']:+.2f}" if A[0][0].get("judged") else "n/a")]
    section(L, "B. INFORMATION (decides nothing): against the request-floor base v5t000, frontier against frontier", Bi, Fg, "request floor v5t000 (8 cached reads)")
    lfl = like_for_like(Fg, arm)
    L += ["  the SAME policy on both systems (the candidate has 4 reads per row: its last point is a plain vote of 4, the base's frontier there is the early-stopping vote of 5):"] + [lfl_text(x) for x in lfl]
    if lfl:
        x1 = lfl[0]
        L.append(f"  -> one read, weights minus request floor: {100 * x1['d']:+.2f} +- {100 * x1['se']:.2f} pp ("
                 + ("within noise of the request floor" if x1["p_two_sided"] > 0.10 else "NOT within noise of the request floor")
                 + f", two-sided p {x1['p_two_sided']:.3f}) at x{x1['total_ratio']:.2f} total tokens (generated x{x1['gen_ratio']:.2f}, prompt x{x1['prompt_ratio']:.2f})")
    out = dict(meta=meta, accepted=acc, reason=rec["reason"], A=[x for x, _, _ in A], B=[x for x, _, _ in Bi], B_same_policy=lfl, signal=sig, shrinkage=dict(signal=sd.get("d"), judge=jd.get("d"), kept=kept),
               decision=dict(p_gen=rec["eq"]["p_gen"], p_macro=rec["eq"]["p_macro"], d=rec["eq"]["d"], se=rec["eq"]["se"], K=1, alpha=JV.ALPHA, neg=[b for b, _ in rec["eq"]["neg"]]))
    if acc:
        out["tip_file"] = write_tip(rows, full, arm, meta, rec, A, sig, S, J)
        L += ["", f"TIP FILE of the weights system: {out['tip_file']}"]
    L.append(f"\n({time.time() - t0:.0f} s)")
    open(os.path.join(OUT, "stage4_judge.txt"), "w").write("\n".join(L) + "\n")
    JV.dump(out, os.path.join(OUT, "stage4_judge.json"))
    print("\n".join(L))


def write_tip(rows, full, arm_j, meta, rec, A, sig, S, J):
    """The tip of the "weights" system: general layer = the VP2M512 weights (stock requests; NO request floor) + its read policy, the
    frontier being the candidate's OWN (vertices chosen on its signal reads, valued on its JUDGE reads)."""
    arm_s, _ = load_candidate(sig["meta"]["path"], rows, full, S)
    n = min(min(len(v) for v in arm_s.values()), min(len(v) for v in arm_j.values()))
    pols = P.tip_policies(max_vote=min(5, n), with_guess=False)
    Ws = JV.Judge(rows, arm_s, benches=B, policies=pols, roles=R.SIGNAL_ROLES)
    Wg = JV.Judge(rows, arm_j, benches=B, policies=pols, hull_from={p[2] for p in Ws.F.vertices}, hull_from_total={p[2] for p in Ws.Ft.vertices})
    V = [p[2] for p in sorted(Ws.F.vertices)]
    pts = []
    for nm in sorted(Wg.F.table, key=lambda k: P.macro_mean(Wg.F.table[k], rows, "tokens")):
        tj, ts = Wg.F.table[nm], Ws.F.table[nm]
        pts.append(dict(name=nm, vertex=nm in V, multiple=P.macro_mean(tj, rows, "tokens") / Wg.F.base_tokens, multiple_total=P.macro_mean(tj, rows, "total") / Wg.Ft.base_tokens,
                        acc_judge=P.macro_mean(tj, rows, "ok"), acc_signal=P.macro_mean(ts, rows, "ok")))
    default = DEFAULT_POLICY if DEFAULT_POLICY in Wg.F.table else "vote1"
    w = [json.loads(x) for x in meta["weights"]]
    tip = dict(id="v5w001", born="stage 4 (11-12)", parent="t000 (the stock model)", kind="weights system",
               note=("General layer = the VP2M512 weights with STOCK requests. The request floor (g001c34, min_image_tokens 4096) is NOT in this system; "
                     "the specialisation layer is empty (its entries were audited on the request-floor base, not on these weights)."),
               weights=dict(name=NAME, artefact=w[0] if len(w) == 1 else w, description="vision patch-embedding kernel resampled to half the patch edge (kernel' = B^T kernel); "
                            "preprocessor bounds 2,048-4,096 merged image tokens", requests="stock: no min_image_tokens floor, effort medium", image_tokens=meta.get("image_tokens")),
               entries=[dict(condition="always", cond="always", procedure=None, settings={}, ops=[], weights=w[0] if len(w) == 1 else w, cid="vp2m512", author="hand (pre-declared, coordinator 11-12)",
                             source=[], audit="GENERAL",
                             verdict=dict(layer="GENERAL", basis="stage 4 verdict A: against the STOCK model's own frontier, default budget, K = 1", d=rec["eq"]["d"], se=rec["eq"]["se"],
                                          p_gen=rec["eq"]["p_gen"], scored=list(B), benches=list(B), one_read_raw=A[0][0].get("raw")))],
               special=[],
               policy=dict(default=default, extras=[], guess=None,
                           frontier=dict(vertices=V, points=pts, default=default, report=["vote1", default, V[-1]], base_tokens=Wg.F.base_tokens, base_total=Wg.Ft.base_tokens,
                                         reads_per_row=n, note=f"from the candidate's own {n} reads per row; vertices chosen on signal rows, valued on JUDGE rows")),
               reads=dict(signal=sig["meta"], judge=meta))
    p = os.path.join(OUT, "tip_v5w001_vp2m512.json")
    JV.dump(tip, p)
    return p


if __name__ == "__main__":
    {"signal": cmd_signal, "judge": cmd_judge}[sys.argv[1]](*sys.argv[2:3])
