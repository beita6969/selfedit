"""The ONE JUDGE-row test of the PY escalation policy (coordinator 11-11 21:20), with the Builder's judge, on stage 3's own path:
stage1_data loaders + classify (grader answer classes) + guess windows, the tip (floor-only base) frontier of policy5.tip_policies with its
vertex NAMES chosen on the signal rows, judge_v5.Judge.cand_from_views / decide (p_gen = largest sign-flip p of the macro mean with one
benchmark left out; accept iff p_gen <= 0.10 and no benchmark significantly negative; family K = 1).

Candidate (fixed on signal rows before any JUDGE read): views vote [tip, tip] -> [py], final "extra" = 2 base reads; if they disagree, ONE PY
read decides. PY reads exist on the JUDGE rows whose 8 cached base reads are not unanimous in their answer class; on the other rows two
base reads never disagree, the PY read is never drawn, and a placeholder stands in (the script checks that its content changes nothing).
Printed as information only: the plurality variant, and the raw difference against the default policy "2 base reads + a third base read".
Also (signal rows, description only): the policy's escalations split by a declared guess in the two disagreeing base reads, per benchmark.
usage: py_judge.py PY_SIGNAL_COMPACT.jsonl.gz [PY_JUDGE_COMPACT.jsonl(.gz)] [--out FILE.txt]"""
import collections
import gzip
import json
import os
import sys

V5 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "v5")
sys.path.insert(0, V5)
import judge_v5 as JV          # noqa: E402
import policy5 as P            # noqa: E402
import reads5 as R             # noqa: E402
import stage1_data as SD       # noqa: E402
import stage3_settings as S3   # noqa: E402

B = R.BENCHES
OUT = []


def say(s=""):
    OUT.append(s)
    print(s)


def opn(p):
    return gzip.open(p, "rt") if p.endswith(".gz") else open(p)


def py_arm(path, rows, full, role_rows):
    arm, none, n = collections.defaultdict(dict), 0, 0
    for l in opn(path):
        x = json.loads(l)
        if x.get("error") or x.get("ok") is None or x["row"] not in role_rows:
            continue
        n += 1
        none += x.get("pred") is None
        arm[x["row"]][int(x["sample"])] = dict(ok=bool(x["ok"]), tokens=float(x["tokens"]), pred=x.get("pred"), guess=None, finish=x.get("finish"),
                                               s=int(x["sample"]), prompt_tokens=float(x["prompt_tokens"]), n_runs=x.get("n_runs"))
    arm = {r: [d[s] for s in sorted(d)] for r, d in arm.items()}
    SD.classify(arm, full)
    return arm, n, none


CAND = P.views_policy(["tip", "tip"], ["py"], "extra")
PLUR = P.views_policy(["tip", "tip"], ["py"])
DEFAULT = P.views_policy(["tip", "tip"], ["tip"])
NAME = "2 base reads; PY read decides on disagreement"


def block_text(b, benches):
    return (f"{100 * b['d']:+5.2f} +- {100 * b['se']:4.2f}  p_macro {b['p_macro']:.4f}  p_gen {b['p_gen']:.4f} (worst without {b['carrier']})  positive {b['npos']}/{b['k']}  | "
            + " ".join(f"{x[:5]} {100 * b['per_bench'][x]['d']:+5.1f}+-{100 * b['per_bench'][x]['se']:3.1f}" for x in benches))


def raw_vs_default(Jx, arms):
    """candidate minus the default policy (2 base reads + a third base read on disagreement), NOT at equal tokens."""
    A = {"tip": Jx.tip}
    A.update(arms)
    c, d = P.evaluate_views(CAND, A, Jx.J), P.evaluate_views(DEFAULT, A, Jx.J)
    x = {r: c[r]["ok"] - d[r]["ok"] for r in Jx.J}
    m = lambda t, k: P.macro_mean(t, Jx.rows, k)
    return Jx._block(x, list(B)), (m(c, "ok"), m(d, "ok"), m(c, "tokens"), m(d, "tokens"), m(c, "total"), m(d, "total"), m(c, "esc"))


def describe(Jx, arms, label):
    rs = Jx.score(Jx.cand_from_views(NAME, CAND, arms))
    rp = Jx.score(Jx.cand_from_views("plurality variant (information)", PLUR, arms))
    say(f"-- {label}")
    say(JV.line(dict(rs, accept=None, reason="(see verdict)"), B).replace("| reject: (see verdict)", "|"))
    say(f"    escalated {100 * rs['esc']:.1f}% of rows-orderings; frontier vertices: " + " | ".join(f"{p[2]} {100 * p[1]:.2f} x{p[0] / Jx.F.base_tokens:.2f}" for p in Jx.F.vertices))
    say(f"    information only, plurality variant: acc {100 * rp['acc']:.2f} at x{rp['multiple']:.2f}; at equal compute {block_text(rp['eq'], B)}")
    bl, (ca, da, ct, dt, cT, dT, esc) = raw_vs_default(Jx, arms)
    say(f"    RAW against the default policy '2 base reads + a third base read' (not at equal tokens): {100 * ca:.2f} vs {100 * da:.2f}; generated tokens "
        f"x{ct / Jx.base_tokens:.2f} vs x{dt / Jx.base_tokens:.2f}; total tokens x{cT / Jx.base_total:.2f} vs x{dT / Jx.base_total:.2f}")
    say(f"        difference {block_text(bl, B)}")
    return rs


def guess_split(Js, py, window):
    """signal rows, description: over all ordered pairs of two different base reads of a row that DISAGREE (the escalations), by whether one
    of the two ends in a declared guess: accuracy when the PY read decides vs the default (plurality with a third base read)."""
    acc = collections.defaultdict(lambda: [0.0, 0.0, 0.0])            # key -> [weight, py ok, default ok]
    for r in Js.J:
        xs = Js.tip[r]
        n = len(xs)
        pv = sum(x["ok"] and P.cls_of(x) != P.NONE for x in py[r]) / len(py[r])
        w = 1.0 / (n * (n - 1))
        for i in range(n):
            for j in range(n):
                if i == j or P.cls_of(xs[i]) == P.cls_of(xs[j]):
                    continue
                g = bool(P.guess_of(xs[i], window)) or bool(P.guess_of(xs[j], window))
                thirds = [k for k in range(n) if k not in (i, j)]
                dv = sum(P._plur([xs[i], xs[j], xs[k]])[1] for k in thirds) / len(thirds)
                for key in (("ALL", g), (Js.rows[r]["bench"], g), ("ALL", None), (Js.rows[r]["bench"], None)):
                    a = acc[key]
                    a[0] += w
                    a[1] += w * pv
                    a[2] += w * dv
    say(f"   guess window {window} characters of the tail (weights = expected escalations per row; 324 rows)")
    for b in ["ALL"] + list(B):
        parts = []
        for g, lab in ((None, "all escalations"), (True, "a declared guess in the pair"), (False, "no declared guess")):
            a = acc.get((b, g))
            if not a or a[0] <= 0:
                parts.append(f"{lab}: none")
                continue
            parts.append(f"{lab}: {a[0]:5.1f} rows-equiv, PY decides {100 * a[1] / a[0]:5.1f} vs third base read {100 * a[2] / a[0]:5.1f} ({100 * (a[1] - a[2]) / a[0]:+5.1f})")
        say(f"   {b:11s} " + " | ".join(parts))


def main():
    use_tip = "--tip" in sys.argv        # the same policy on top of the specialisation system: 2 TIP (t007c) reads, PY decides (for the record)
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
    if out in args:
        args.remove(out)
    rows, full = SD.rows(), SD.full_rows()
    if use_tip:
        import guess5 as G
        floor = SD.load_e1(rows)[0]
        SD.classify(floor, full)
        for xs in floor.values():
            for x in xs:
                x["guess"] = {w: G.flag(dict(seed="guess", window=w, patterns=[]), x.get("tail")) for w in S3.WINDOWS}
    else:
        floor = S3.base(rows, full)
    Js, S, J, pols = S3.judges(rows, floor)
    py_s, n_s, none_s = py_arm(args[0], rows, full, set(S))
    say("PY ESCALATION POLICY: '2 base reads (" + ("TIP t007c = general base + specialisation layer; FOR THE RECORD, not a decision" if use_tip else "floor-only base")
        + ", effort medium); if their answers disagree, ONE PY read (floor-only PY read) decides'")
    say("judge = code-v5/v5 judge_v5.py on stage 3's path (generated tokens gate; reference = the base's own frontier at the policy's tokens; "
        "sign-flip of picture clusters; accept iff p_gen <= 0.10 and no benchmark significantly negative; family K = 1)")
    say("")
    sig = describe(Js, {"py": py_s}, f"SIGNAL rows (324 show + monitor; pre-screen, never a decision); PY reads {n_s}, without an answer {none_s} ({100 * none_s / n_s:.1f}%)")
    say("")
    say("-- SIGNAL rows, description only: the escalations by a declared guess in the two disagreeing base reads, and by benchmark")
    for w in S3.WINDOWS:
        guess_split(Js, py_s, w)
    if len(args) < 2:
        if out:
            open(out, "w").write("\n".join(OUT) + "\n")
        return
    # ---------------------------------------------------------------- JUDGE rows
    Jset = set(J)
    py_j, n_j, none_j = py_arm(args[1], rows, full, Jset)
    need = [r for r in J if len({P.cls_of(x) for x in floor[r]}) > 1]
    miss = [r for r in need if not py_j.get(r)]
    if miss:
        raise SystemExit(f"{len(miss)} JUDGE rows with non-unanimous base reads have no PY read (e.g. {miss[0]}): not judged")
    extra = [r for r in py_j if r not in set(need)]
    Jg = JV.Judge(rows, {r: floor[r] for r in J}, benches=B, policies=pols, hull_from={p[2] for p in Js.F.vertices},
                  hull_from_total={p[2] for p in Js.Ft.vertices})

    def filled(ok, tokens):
        ph = dict(ok=ok, tokens=tokens, pred=None if not ok else "x", cls=P.NONE if not ok else ("v", "placeholder"), guess=None, finish="stop", s=0, prompt_tokens=tokens)
        return {r: (py_j[r] if r in py_j else [dict(ph)]) for r in J}

    a1, a2 = filled(False, 0.0), filled(True, 1e6)
    c1, c2 = Jg.cand_from_views(NAME, CAND, {"py": a1}), Jg.cand_from_views(NAME, CAND, {"py": a2})
    same = all(abs(c1["value"][r] - c2["value"][r]) < 1e-12 and abs(c1["tokens"][r] - c2["tokens"][r]) < 1e-9 for r in J)
    if not same:
        raise SystemExit("the placeholder PY read changes the policy's value on some row: a row without PY reads can escalate")
    say("")
    say(f"-- JUDGE rows (444): PY reads {n_j} on {len(py_j)} rows whose 8 cached base reads are not unanimous in their answer class ({len(need)} needed; "
        f"{len(extra)} extra); without an answer {none_j} ({100 * none_j / max(n_j, 1):.1f}%); the other {len(J) - len(py_j)} rows can never escalate "
        f"(placeholder check: identical values with two different placeholders = {same})")
    recs = Jg.decide([c1], family="PY escalation (K = 1)")
    r = recs[0]
    say(JV.line(r, B))
    say(f"    escalated {100 * r['esc']:.1f}%; JUDGE frontier (vertex names from signal rows): " + " | ".join(f"{p[2]} {100 * p[1]:.2f} x{p[0] / Jg.F.base_tokens:.2f}" for p in Jg.F.vertices))
    neg = r["eq"]["neg"]
    say(f"    benchmarks significantly negative: {neg if neg else 'none'}")
    rp = Jg.score(Jg.cand_from_views("plurality variant (information)", PLUR, {"py": a1}))
    say(f"    information only, plurality variant: acc {100 * rp['acc']:.2f} at x{rp['multiple']:.2f}; at equal compute {block_text(rp['eq'], B)}")
    bl, (ca, da, ct, dt, cT, dT, esc) = raw_vs_default(Jg, {"py": a1})
    say(f"    RAW against the default policy '2 base reads + a third base read' (not at equal tokens): {100 * ca:.2f} vs {100 * da:.2f}; generated tokens "
        f"x{ct / Jg.base_tokens:.2f} vs x{dt / Jg.base_tokens:.2f}; total tokens x{cT / Jg.base_total:.2f} vs x{dT / Jg.base_total:.2f}")
    say(f"        difference {block_text(bl, B)}")
    kept = r["eq"]["d"] / sig["eq"]["d"] if sig["eq"]["d"] > 0 else None
    say("")
    say(f"SHRINKAGE at equal generated tokens: signal rows {100 * sig['eq']['d']:+.2f} +- {100 * sig['eq']['se']:.2f} -> JUDGE rows {100 * r['eq']['d']:+.2f} +- {100 * r['eq']['se']:.2f}"
        + (f" (kept {100 * kept:.0f}%)" if kept is not None else ""))
    say(f"VERDICT: {'ACCEPT' if r['accept'] else 'REJECT'}: {r['reason']}")
    # PY reads themselves on the rows read (information): accuracy of one PY read vs one base read on the same rows
    rows_read = sorted(py_j)
    pv = sum(sum(x["ok"] for x in py_j[q]) / len(py_j[q]) for q in rows_read) / len(rows_read)
    bv = sum(sum(x["ok"] for x in floor[q]) / len(floor[q]) for q in rows_read) / len(rows_read)
    pt = sum(sum(x["tokens"] for x in py_j[q]) / len(py_j[q]) for q in rows_read) / len(rows_read)
    bt = sum(sum(x["tokens"] for x in floor[q]) / len(floor[q]) for q in rows_read) / len(rows_read)
    say(f"information: on the {len(rows_read)} rows read, one PY read {100 * pv:.2f} vs one base read {100 * bv:.2f} (pooled rows); generated tokens {pt:.0f} vs {bt:.0f} (x{pt / bt:.2f})")
    if out:
        open(out, "w").write("\n".join(OUT) + "\n")
        JV.dump(dict(signal=sig, judge=r, plurality_judge=rp), out.replace(".txt", ".json"))


if __name__ == "__main__":
    main()
