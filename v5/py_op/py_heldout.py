"""Held-out tables of the finished system (REPORTING ONLY; coordinator 11-11 22:20). ID test and OOD v2, per benchmark and macro.
Same code path as the JUDGE verdict: policy5.evaluate_views = exact expectation over the orderings of the cached base reads; answer classes
= policy5.cls_of of the stored pred; picture-clustered se (stats5.bench_effects, pic from the export), 95 % interval = 1.96 se, macro se =
sqrt(sum se_b^2) / k. Tokens relative to ONE STOCK read of the same set (generated, and generated + prompt of every call).
 lines  stock / floor (general base) / tip (general + specialisation layer): 1 read; 2 reads + a third read of the same system on
        disagreement; floor and tip also: 2 reads + ONE PY read decides on disagreement (the accepted policy; the PY read is the floor-only
        PY read in both systems).
A row without a PY read can never escalate (its base reads are unanimous); a placeholder stands in and the script checks that its
content changes nothing.
usage: py_heldout.py EXPORT.jsonl.gz [PY_COMPACT.jsonl(.gz)] [--samples 0] [--out FILE.txt]"""
import collections
import gzip
import json
import math
import os
import sys

V5 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "v5")
sys.path.insert(0, V5)
import policy5 as P     # noqa: E402
import stats5 as ST     # noqa: E402

OUT = []
V = P.views_policy
ONE, THIRD, PYP = V(["sys"]), V(["sys", "sys"], ["sys"]), V(["sys", "sys"], ["py"], "extra")
VENUS = {("id", "floor"): 70.0, ("ood", "floor"): 70.0, ("id", "tip"): 73.0, ("ood", "tip"): 70.1, ("id", "stock"): 63.1, ("ood", "stock"): 66.8}


def say(s=""):
    OUT.append(s)
    print(s)


def opn(p):
    return gzip.open(p, "rt") if p.endswith(".gz") else open(p)


def main():
    a = sys.argv[1:]
    out = a[a.index("--out") + 1] if "--out" in a else None
    only = [int(x) for x in a[a.index("--samples") + 1].split(",")] if "--samples" in a else None
    pos = [x for i, x in enumerate(a) if not x.startswith("--") and (i == 0 or a[i - 1] not in ("--out", "--samples"))]
    rows, arms = {}, collections.defaultdict(lambda: collections.defaultdict(dict))
    for l in opn(pos[0]):
        x = json.loads(l)
        rows[x["row"]] = dict(bench=x["bench"], pic=x.get("pic") or x["row"], set=x["set"])
        arms[x["system"]][x["row"]][int(x["sample"])] = dict(ok=bool(x["ok"]), tokens=float(x["tokens"] or 0), pred=x.get("pred"), cls=None, guess=None,
                                                             finish=x.get("finish"), s=int(x["sample"]), prompt_tokens=float(x.get("prompt_tokens") or 0))
    arms = {s: {r: [d[k] for k in sorted(d)] for r, d in t.items()} for s, t in arms.items()}
    py, pyinfo = None, {}
    if len(pos) > 1:
        t = collections.defaultdict(dict)
        for l in opn(pos[1]):
            x = json.loads(l)
            if x.get("error") or x.get("ok") is None or x["row"] not in rows or (only is not None and int(x["sample"]) not in only):
                continue
            t[x["row"]][int(x["sample"])] = dict(ok=bool(x["ok"]), tokens=float(x["tokens"]), pred=x.get("pred"), cls=None, guess=None, finish=x.get("finish"),
                                                 s=int(x["sample"]), prompt_tokens=float(x["prompt_tokens"]), n_runs=x.get("n_runs") or 0, flags=x.get("flags") or {})
        py = {r: [d[k] for k in sorted(d)] for r, d in t.items()}
    nonun = lambda s, r: len({P.cls_of(x) for x in arms[s][r]}) > 1
    say("HELD-OUT TABLES OF THE FINISHED SYSTEM (reporting only; nothing is chosen from these reads)" + (f"; PY samples {only}" if only is not None else ""))
    for st in ("id", "ood"):
        R = sorted(r for r in rows if rows[r]["set"] == st)
        benches = sorted({rows[r]["bench"] for r in R})
        mm = lambda t, k: P.macro_mean(t, rows, k)
        bm = lambda t, k, b: sum(t[r][k] for r in R if rows[r]["bench"] == b) / sum(rows[r]["bench"] == b for r in R)
        T = {}
        for s in ("stock", "floor", "tip"):
            A = {"sys": {r: arms[s][r] for r in R}}
            T[(s, "1 read")] = P.evaluate_views(ONE, A, R)
            T[(s, "2 reads + third")] = P.evaluate_views(THIRD, A, R)
        g0, t0 = mm(T[("stock", "1 read")], "tokens"), mm(T[("stock", "1 read")], "total")
        say("")
        say(f"================ {st.upper()} ({len(R)} rows; benchmarks {' '.join(benches)}); one stock read = {g0:.0f} generated / {t0:.0f} total tokens")
        say("check against the Venus numbers (2 reads + third on disagreement): " + "; ".join(
            f"{s} {100 * mm(T[(s, '2 reads + third')], 'ok'):.1f} (Venus {VENUS[(st, s)]:.1f})" for s in ("floor", "tip", "stock")))
        have_py = False
        if py is not None:
            need = [r for r in R if nonun("floor", r) or nonun("tip", r)]
            miss = [r for r in need if not py.get(r)]
            if miss:
                say(f"PY reads missing on {len(miss)} of the {len(need)} rows that can escalate: PY lines not computed for this set")
            else:
                have_py = True
                def filled(ok, tok):
                    ph = dict(ok=ok, tokens=tok, pred=None if not ok else "x", cls=P.NONE if not ok else ("v", "placeholder"), guess=None, finish="stop", s=0, prompt_tokens=tok)
                    return {r: (py[r] if py.get(r) else [dict(ph)]) for r in R}
                for s in ("floor", "tip"):
                    A1 = {"sys": {r: arms[s][r] for r in R}, "py": filled(False, 0.0)}
                    A2 = {"sys": A1["sys"], "py": filled(True, 1e6)}
                    t1, t2 = P.evaluate_views(PYP, A1, R), P.evaluate_views(PYP, A2, R)
                    if any(abs(t1[r]["ok"] - t2[r]["ok"]) > 1e-12 or abs(t1[r]["tokens"] - t2[r]["tokens"]) > 1e-9 for r in R):
                        raise SystemExit(f"{st} {s}: the placeholder changes the policy value: a row without PY reads can escalate")
                    T[(s, "2 reads + PY decides")] = t1
                reads = [x for r in need for x in py[r]]
                pyinfo[st] = dict(rows=len(need), reads=len(reads), none=sum(x["pred"] is None for x in reads), code=sum(x["n_runs"] > 0 for x in reads),
                                  ctx=sum(bool(x["flags"].get("ctx_stop")) for x in reads))
        order = [("stock", "1 read"), ("stock", "2 reads + third"), ("floor", "1 read"), ("floor", "2 reads + third"), ("floor", "2 reads + PY decides"),
                 ("tip", "1 read"), ("tip", "2 reads + third"), ("tip", "2 reads + PY decides")]
        order = [k for k in order if k in T]
        say(f"{'system / policy':34s} " + " ".join(f"{b[:10]:>10s}" for b in benches) + f" {'MACRO':>7s} {'gen x':>6s} {'total x':>7s} {'escal.':>6s}")
        for k in order:
            t = T[k]
            say(f"{k[0] + ': ' + k[1]:34s} " + " ".join(f"{100 * bm(t, 'ok', b):10.1f}" for b in benches)
                + f" {100 * mm(t, 'ok'):7.2f} {mm(t, 'tokens') / g0:6.2f} {mm(t, 'total') / t0:7.2f} {100 * mm(t, 'esc'):5.1f}%")

        def diff(k1, k2, per_bench=False):
            x = {r: T[k1][r]["ok"] - T[k2][r]["ok"] for r in R}
            E = ST.bench_effects(x, rows, floor_rows=0.0)
            M = ST.macro(E, benches)
            s = f"{100 * M['d']:+6.2f} [{100 * (M['d'] - 1.96 * M['se']):+5.1f}, {100 * (M['d'] + 1.96 * M['se']):+5.1f}]"
            if per_bench:
                s += "   " + " ".join(f"{b[:6]} {100 * E[b]['d']:+5.1f}±{196 * E[b]['se_raw']:.1f}" for b in benches) + f"   benchmarks + {sum(E[b]['d'] > 0 for b in benches)} / - {sum(E[b]['d'] < 0 for b in benches)}"
            return s
        say("differences, macro, picture-clustered 95 % interval:")
        for k in order[2:]:
            say(f"  {k[0] + ': ' + k[1]:32s} minus stock 1 read {diff(k, ('stock', '1 read'))}   minus stock 2 reads + third {diff(k, ('stock', '2 reads + third'))}")
        if have_py:
            say("what the accepted component adds on held-out rows (PY decides minus the same system's '2 reads + third base read'; per benchmark ± = 95 %):")
            for s in ("floor", "tip"):
                say(f"  {s:5s} {diff((s, '2 reads + PY decides'), (s, '2 reads + third'), per_bench=True)}")
                tk = (mm(T[(s, '2 reads + PY decides')], 'tokens') - mm(T[(s, '2 reads + third')], 'tokens')) / g0
                say(f"        extra generated tokens: +x{tk:.2f} of one stock read (x{mm(T[(s, '2 reads + PY decides')], 'tokens') / g0:.2f} vs x{mm(T[(s, '2 reads + third')], 'tokens') / g0:.2f})")
            i = pyinfo[st]
            say(f"PY reads: {i['reads']} on {i['rows']} rows; without an answer {i['none']} ({100 * i['none'] / i['reads']:.1f}%), of which request did not fit the context window {i['ctx']}; "
                f"ran code {100 * i['code'] / i['reads']:.1f}%")
            say("on escalations (expected, in row-equivalents): PY decides vs a third base read")
            for s in ("floor", "tip"):
                acc = collections.defaultdict(lambda: [0.0, 0.0, 0.0])
                for r in R:
                    xs = arms[s][r]
                    n = len(xs)
                    if not py.get(r):
                        continue
                    pv = sum(x["ok"] and x["pred"] is not None for x in py[r]) / len(py[r])
                    for i1 in range(n):
                        for j1 in range(n):
                            if i1 == j1 or P.cls_of(xs[i1]) == P.cls_of(xs[j1]):
                                continue
                            th = [k for k in range(n) if k not in (i1, j1)]
                            dv = sum(P._plur([xs[i1], xs[j1], xs[k]])[1] for k in th) / len(th)
                            for key in ("ALL", rows[r]["bench"]):
                                acc[key][0] += 1.0 / (n * (n - 1))
                                acc[key][1] += pv / (n * (n - 1))
                                acc[key][2] += dv / (n * (n - 1))
                say(f"  {s:5s} " + " | ".join(f"{b[:10]} {acc[b][0]:.1f}: {100 * acc[b][1] / max(acc[b][0], 1e-9):.0f} vs {100 * acc[b][2] / max(acc[b][0], 1e-9):.0f}" for b in ["ALL"] + benches))
    if out:
        open(out, "w").write("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
