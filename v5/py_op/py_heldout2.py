"""Held-out tables with several systems side by side (REPORTING ONLY; coordinator 11-12 04:27): stock, the request-floor system (floor;
tip = floor + specialisation layer) and a WEIGHTS system (e.g. vp2m512, stock requests). Same code path as py_heldout.py / the JUDGE
verdicts: policy5.evaluate_views (exact expectation over the orderings of the cached base reads of a system), answer classes =
policy5.cls_of(pred), picture-clustered se (stats5.bench_effects), 95 % interval = 1.96 se, macro se = sqrt(sum se_b^2) / k. Tokens are
multiples of ONE STOCK read of the same set (generated / generated + prompt of every call).
Lines per system: 1 read; 2 reads + a third read of the same system on disagreement; 2 reads + ONE PY read decides on disagreement (the PY
read made on that system's own base: --py SYSTEM[,SYSTEM]=COMPACT).
usage: py_heldout2.py EXPORT[,EXPORT...] --py floor,tip=FILE --py vp2m512=FILE [--out FILE.txt]"""
import collections
import gzip
import json
import os
import sys

V5 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "v5")
sys.path.insert(0, V5)
import policy5 as P     # noqa: E402
import stats5 as ST     # noqa: E402

OUT = []
V = P.views_policy
ONE, THIRD, PYP = V(["sys"]), V(["sys", "sys"], ["sys"]), V(["sys", "sys"], ["py"], "extra")


def say(s=""):
    OUT.append(s)
    print(s)


def opn(p):
    return gzip.open(p, "rt") if p.endswith(".gz") else open(p)


def main():
    a = sys.argv[1:]
    out = a[a.index("--out") + 1] if "--out" in a else None
    pyspec = [a[i + 1] for i, x in enumerate(a) if x == "--py"]
    rows, arms = {}, collections.defaultdict(lambda: collections.defaultdict(dict))
    for f in a[0].split(","):
        for l in opn(f):
            x = json.loads(l)
            rows.setdefault(x["row"], dict(bench=x["bench"], pic=x.get("pic") or x["row"], set=x["set"]))
            arms[x["system"]][x["row"]][int(x["sample"])] = dict(ok=bool(x["ok"]), tokens=float(x["tokens"] or 0), pred=x.get("pred"), cls=None, guess=None,
                                                                 finish=x.get("finish"), s=int(x["sample"]), prompt_tokens=float(x.get("prompt_tokens") or 0))
    arms = {s: {r: [d[k] for k in sorted(d)] for r, d in t.items()} for s, t in arms.items()}
    pyof = {}
    for spec in pyspec:
        names, path = spec.split("=", 1)
        t = collections.defaultdict(dict)
        for l in opn(path):
            x = json.loads(l)
            if x.get("error") or x.get("ok") is None or x["row"] not in rows:
                continue
            t[x["row"]][int(x["sample"])] = dict(ok=bool(x["ok"]), tokens=float(x["tokens"]), pred=x.get("pred"), cls=None, guess=None, finish=x.get("finish"),
                                                 s=int(x["sample"]), prompt_tokens=float(x["prompt_tokens"]), n_runs=x.get("n_runs") or 0, flags=x.get("flags") or {})
        arm = {r: [d[k] for k in sorted(d)] for r, d in t.items()}
        for n in names.split(","):
            pyof[n] = arm
    systems = [s for s in ("stock", "floor", "tip") if s in arms] + sorted(s for s in arms if s not in ("stock", "floor", "tip"))
    say("HELD-OUT TABLES (reporting only; nothing is chosen from these reads); systems: " + ", ".join(f"{s} ({min(len(v) for v in arms[s].values())} cached reads per row)" for s in systems))
    for st in ("id", "ood"):
        R = sorted(r for r in rows if rows[r]["set"] == st and all(r in arms[s] for s in systems))
        benches = sorted({rows[r]["bench"] for r in R})
        mm = lambda t, k: P.macro_mean(t, rows, k)
        bm = lambda t, k, b: sum(t[r][k] for r in R if rows[r]["bench"] == b) / sum(rows[r]["bench"] == b for r in R)
        T, info = {}, {}
        for s in systems:
            A = {"sys": {r: arms[s][r] for r in R}}
            T[(s, "1 read")] = P.evaluate_views(ONE, A, R)
            T[(s, "2 reads + third")] = P.evaluate_views(THIRD, A, R)
            py = pyof.get(s)
            if py is None:
                continue
            need = [r for r in R if len({P.cls_of(x) for x in arms[s][r]}) > 1]
            miss = [r for r in need if not py.get(r)]
            if miss:
                say(f"{st} {s}: PY reads missing on {len(miss)} of the {len(need)} rows that can escalate: PY line not computed")
                continue

            def filled(ok, tok):
                ph = dict(ok=ok, tokens=tok, pred=None if not ok else "x", cls=P.NONE if not ok else ("v", "placeholder"), guess=None, finish="stop", s=0, prompt_tokens=tok)
                return {r: (py[r] if py.get(r) else [dict(ph)]) for r in R}
            t1 = P.evaluate_views(PYP, {"sys": A["sys"], "py": filled(False, 0.0)}, R)
            t2 = P.evaluate_views(PYP, {"sys": A["sys"], "py": filled(True, 1e6)}, R)
            if any(abs(t1[r]["ok"] - t2[r]["ok"]) > 1e-12 or abs(t1[r]["tokens"] - t2[r]["tokens"]) > 1e-9 for r in R):
                raise SystemExit(f"{st} {s}: the placeholder changes the policy value")
            T[(s, "2 reads + PY decides")] = t1
            reads = [x for r in need for x in py[r]]
            acc = collections.defaultdict(lambda: [0.0, 0.0, 0.0])
            for r in need:
                xs = arms[s][r]
                n = len(xs)
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
            info[s] = dict(rows=len(need), reads=len(reads), none=sum(x["pred"] is None for x in reads), code=sum(x["n_runs"] > 0 for x in reads),
                           ctx=sum(bool(x["flags"].get("ctx_stop")) for x in reads), esc=acc)
        g0, t0 = mm(T[("stock", "1 read")], "tokens"), mm(T[("stock", "1 read")], "total")
        say("")
        say(f"================ {st.upper()} ({len(R)} rows; benchmarks {' '.join(benches)}); one stock read = {g0:.0f} generated / {t0:.0f} total tokens")
        order = [(s, p) for s in systems for p in ("1 read", "2 reads + third", "2 reads + PY decides") if (s, p) in T]
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
        for k in order:
            if k[0] == "stock":
                continue
            say(f"  {k[0] + ': ' + k[1]:32s} minus stock 1 read {diff(k, ('stock', '1 read'))}   minus stock 2 reads + third {diff(k, ('stock', '2 reads + third'))}")
        ws = [s for s in systems if s not in ("stock", "floor", "tip")]
        for w in ws:
            for p in ("1 read", "2 reads + third", "2 reads + PY decides"):
                if (w, p) in T and ("floor", p) in T:
                    say(f"  {w}: {p:22s} minus floor: same policy {diff((w, p), ('floor', p), per_bench=True)}")
        say("what the PY escalation adds (PY decides minus the same system's '2 reads + third base read'; per benchmark ± = 95 %):")
        for s in systems:
            if (s, "2 reads + PY decides") not in T:
                continue
            say(f"  {s:8s} {diff((s, '2 reads + PY decides'), (s, '2 reads + third'), per_bench=True)}")
            i = info[s]
            say(f"           PY reads {i['reads']} on {i['rows']} rows; without an answer {i['none']} ({100 * i['none'] / i['reads']:.1f}%), did not fit the window {i['ctx']}; ran code {100 * i['code'] / i['reads']:.1f}%; "
                f"extra generated tokens +x{(mm(T[(s, '2 reads + PY decides')], 'tokens') - mm(T[(s, '2 reads + third')], 'tokens')) / g0:.2f} of one stock read")
            e = i["esc"]
            say("           on escalations, PY decides vs a third base read: " + " | ".join(f"{b[:10]} {e[b][0]:.1f}: {100 * e[b][1] / max(e[b][0], 1e-9):.0f} vs {100 * e[b][2] / max(e[b][0], 1e-9):.0f}" for b in ["ALL"] + benches))
    if out:
        open(out, "w").write("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
