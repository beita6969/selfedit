"""V1 ("2 base reads; ONE PY read decides on disagreement") on a WEIGHTS base (VP2M512, stock requests), JUDGE rows, K = 1 (coordinator
11-12 04:27). Same path as judge1 (py_judge.py): the policy against THIS base's own frontier at equal generated tokens (4 cached base
reads per row: tip_policies with max_vote 4; vertex names chosen on the signal rows from the base's signal reads), judge_v5.Judge
.cand_from_views / decide, sign-flip of picture clusters, accept iff p_gen <= 0.10 and no benchmark significantly negative.
usage: py_judge_vp.py E_SIGNAL.jsonl.gz E_JUDGE.jsonl.gz ARM [PY_JUDGE_COMPACT] [--out FILE]"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import py_judge as PJ          # noqa: E402  (py_arm, policies, printing helpers; it puts code-v5/v5 on the path)
import guess5 as G             # noqa: E402
import judge_v5 as JV          # noqa: E402
import policy5 as P            # noqa: E402
import reads5 as R             # noqa: E402
import stage1_data as SD       # noqa: E402
import stage3_settings as S3   # noqa: E402

B, say = R.BENCHES, PJ.say


def main():
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
    if out in a:
        a.remove(out)
    e_sig, e_jud, arm = a[0], a[1], a[2]
    rows, full = SD.rows(), SD.full_rows()
    S, J = R.rows_of(rows, R.SIGNAL_ROLES), R.rows_of(rows, "judge")
    sig = S3.arms_of(e_sig, rows, full, S)[arm]
    jud = S3.arms_of(e_jud, rows, full, J)[arm]
    for t in (sig, jud):
        for xs in t.values():
            for x in xs:
                x["guess"] = {w: G.flag(dict(seed="guess", window=w, patterns=[]), x.get("tail")) for w in S3.WINDOWS}
    pols = P.tip_policies(max_vote=4, with_guess=True, windows=S3.WINDOWS)
    Js = JV.Judge(rows, {r: sig[r] for r in S}, benches=B, policies=pols, roles=R.SIGNAL_ROLES)
    Jg = JV.Judge(rows, {r: jud[r] for r in J}, benches=B, policies=pols, hull_from={p[2] for p in Js.F.vertices}, hull_from_total={p[2] for p in Js.Ft.vertices})
    say(f"V1 ON THE WEIGHTS BASE {arm} (stock requests): '2 base reads; if their answers disagree, ONE PY read decides'; JUDGE rows, K = 1")
    say(f"base frontier, signal rows ({len(S)}): " + " | ".join(f"{p[2]} {100 * p[1]:.2f} x{p[0] / Js.F.base_tokens:.2f}" for p in Js.F.vertices)
        + f"; one read = {Js.base_tokens:.0f} generated / {Js.base_total:.0f} total tokens")
    say(f"base frontier, JUDGE rows ({len(J)}; vertex names from signal rows): " + " | ".join(f"{p[2]} {100 * p[1]:.2f} x{p[0] / Jg.F.base_tokens:.2f}" for p in Jg.F.vertices)
        + f"; one read = {Jg.base_tokens:.0f} generated / {Jg.base_total:.0f} total tokens; all evaluated points: "
        + ", ".join(f"{n} {100 * acc:.2f} x{tk / Jg.F.base_tokens:.2f}" for tk, acc, n in sorted(Jg.F.points) if n in ("vote1", "vote3", "vote4", "n2:disagree->tip")))
    if len(a) < 4:
        return
    py, n_j, none_j = PJ.py_arm(a[3], rows, full, set(J))
    need = [r for r in J if len({P.cls_of(x) for x in jud[r]}) > 1]
    miss = [r for r in need if not py.get(r)]
    if miss:
        raise SystemExit(f"{len(miss)} JUDGE rows with non-unanimous base reads have no PY read (e.g. {miss[0]})")

    def filled(ok, tok):
        ph = dict(ok=ok, tokens=tok, pred=None if not ok else "x", cls=P.NONE if not ok else ("v", "placeholder"), guess=None, finish="stop", s=0, prompt_tokens=tok)
        return {r: (py[r] if r in py else [dict(ph)]) for r in J}

    a1, a2 = filled(False, 0.0), filled(True, 1e6)
    c1, c2 = Jg.cand_from_views(PJ.NAME, PJ.CAND, {"py": a1}), Jg.cand_from_views(PJ.NAME, PJ.CAND, {"py": a2})
    same = all(abs(c1["value"][r] - c2["value"][r]) < 1e-12 and abs(c1["tokens"][r] - c2["tokens"][r]) < 1e-9 for r in J)
    if not same:
        raise SystemExit("the placeholder PY read changes the policy's value on some row")
    say("")
    say(f"PY reads {n_j} on {len(py)} JUDGE rows ({len(need)} rows have 4 cached base reads that are not unanimous in their answer class); without an answer {none_j} "
        f"({100 * none_j / max(n_j, 1):.1f}%); ran code {100 * sum(x['n_runs'] > 0 for xs in py.values() for x in xs) / max(n_j, 1):.1f}%; the other {len(J) - len(py)} rows never escalate "
        f"(placeholder check {same})")
    r = Jg.decide([c1], family="V1 on the weights base (K = 1)")[0]
    say(JV.line(r, B))
    say(f"    escalated {100 * r['esc']:.1f}%; benchmarks significantly negative: {r['eq']['neg'] if r['eq']['neg'] else 'none'}")
    rp = Jg.score(Jg.cand_from_views("plurality variant (information)", PJ.PLUR, {"py": a1}))
    say(f"    information only, plurality variant: acc {100 * rp['acc']:.2f} at x{rp['multiple']:.2f}; at equal compute {PJ.block_text(rp['eq'], B)}")
    bl, (ca, da, ct, dt, cT, dT, esc) = PJ.raw_vs_default(Jg, {"py": a1})
    say(f"    RAW against '2 base reads + a third base read' (not at equal tokens): {100 * ca:.2f} vs {100 * da:.2f}; generated tokens x{ct / Jg.base_tokens:.2f} vs x{dt / Jg.base_tokens:.2f}; "
        f"total tokens x{cT / Jg.base_total:.2f} vs x{dT / Jg.base_total:.2f}")
    say(f"        difference {PJ.block_text(bl, B)}")
    rr = sorted(py)
    pv = sum(sum(x["ok"] for x in py[q]) / len(py[q]) for q in rr) / len(rr)
    bv = sum(sum(x["ok"] for x in jud[q]) / len(jud[q]) for q in rr) / len(rr)
    pt = sum(sum(x["tokens"] for x in py[q]) / len(py[q]) for q in rr) / len(rr)
    bt = sum(sum(x["tokens"] for x in jud[q]) / len(jud[q]) for q in rr) / len(rr)
    pT = sum(sum(x["tokens"] + x["prompt_tokens"] for x in py[q]) / len(py[q]) for q in rr) / len(rr)
    bT = sum(sum(x["tokens"] + x["prompt_tokens"] for x in jud[q]) / len(jud[q]) for q in rr) / len(rr)
    say(f"    on the {len(rr)} rows read: one PY read {100 * pv:.2f} vs one base read {100 * bv:.2f}; generated tokens {pt:.0f} vs {bt:.0f} (x{pt / bt:.2f}); total {pT:.0f} vs {bT:.0f} (x{pT / bT:.2f})")
    say(f"VERDICT: {'ACCEPT' if r['accept'] else 'REJECT'}: {r['reason']}")
    if out:
        open(out, "w").write("\n".join(PJ.OUT) + "\n")
        JV.dump(dict(judge=r, plurality=rp), out.replace(".txt", ".json"))


if __name__ == "__main__":
    main()
