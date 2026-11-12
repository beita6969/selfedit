"""Stage 1 (a): false-accept rate of the v5 statistic on A/A comparisons built from the tip's own samples, and its power on planted effects.
JUDGE rows only (444 rows; the tip's reads with answers: E-cache samples + W-null samples = the same tip, 8 reads per row).

Part A  (statistic alone, x1 tokens). Per replicate and row, a random split of the row's tip samples: NC "candidate" reads vs the rest as the
        reference (mean). Planted effects: in the chosen benchmarks a share of rows becomes "helped" (each wrong candidate read turns right
        with probability Q) so that the benchmark gains delta pp in expectation. Rules compared at nominal 0.10: p_macro; the v4 form
        (p_macro and the leave-one-benchmark-out POINT estimate > 0); benchmark-level t (5 df) and exact sign-flip; p_gen (v5); and p_gen at
        the first Holm threshold of a family of K = 3, 6, 10.
Part B  (the whole judge: frontier reference, equal compute, floors). Per replicate and row, a random split into a tip half (4 reads: the
        cached tip of the replicate, from which the frontier is built) and an arm half (4 reads of a pseudo candidate arm that IS the tip).
        Candidates: arm always-on; "2 tip reads, on disagreement read the arm" with the plurality and with the arm's answer as final. The
        reference is the hull of the tip half at the candidate's tokens. Under A/A every one of them has a true gain of exactly 0. The same
        with planted effects in the arm (the arm is a better reader).
Output: OUT/aa_power.json, OUT/aa_power.txt.   usage: stage1_aa.py [--reps-a 6000] [--reps-b 600] [--proc 8]
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import judge_v5 as JV  # noqa: E402
import policy5 as P  # noqa: E402
import reads5 as R  # noqa: E402
import stage1_data as SD  # noqa: E402
import stats5 as ST  # noqa: E402

Q = 0.75
SCEN = [("A/A (no effect)", {}),
        ("homogeneous +2", {b: .02 for b in range(6)}), ("homogeneous +3", {b: .03 for b in range(6)}),
        ("homogeneous +4", {b: .04 for b in range(6)}), ("homogeneous +6", {b: .06 for b in range(6)}),
        ("floor-like (+14.9 +14.2 +5.4 +8.1 0 +2.7)", {0: .149, 1: .142, 2: .054, 3: .081, 5: .027}),
        ("ONE benchmark +18 (visulogic)", {4: .18}), ("ONE benchmark +18 (babyvision)", {0: .18}), ("ONE benchmark +9 (vlmsbiased)", {5: .09}),
        ("TWO benchmarks +9", {0: .09, 4: .09})]
KS = (1, 3, 6, 10)
_G = {}


def _load():
    if not _G:
        rows = SD.rows()
        arms, base, tip8 = SD.judge_tables(rows)
        J = R.rows_of(rows, "judge")
        _G.update(rows=rows, J=J, tip8=tip8, D=ST.Design(J, rows, R.BENCHES))
    return _G


def part_a(reps, nc, seed=1):
    g = _load()
    D, J = g["D"], g["J"]
    ok = np.array([[x["ok"] for x in g["tip8"][r]] for r in J], float)
    rng = np.random.default_rng(seed)
    out = {}
    for name, plant in SCEN:
        X = np.zeros((reps, len(J)))
        for i in range(reps):
            o = np.take_along_axis(ok, np.argsort(rng.random(ok.shape), axis=1), 1)
            c, ref = o[:, :nc].copy(), o[:, nc:]
            for b, delta in plant.items():
                m = D.b == b
                share = min(1.0, delta / (Q * (1 - ok[m].mean())))
                helped = rng.random(m.sum()) < share
                c[m] = np.where((rng.random((m.sum(), nc)) < Q) & helped[:, None], 1.0, c[m])
            X[i] = c.mean(1) - ref.mean(1)
        res = {}
        for fl in (0.0, 0.5, 1.0):
            st = D.stats(X, floor_rows=fl)
            a = 0.10
            res[f"floor{fl}"] = dict(
                d=float(st["d"].mean()), sd=float(st["d"].std()), se=float(st["se"].mean()),
                p_macro=float((st["p_macro"] <= a).mean()), v4_macro_lobo_point=float(((st["p_macro"] <= a) & (st["lobo_min"] > 0)).mean()),
                bench_t=float((st["p_bench_t"] <= a).mean()), bench_signflip=float((st["p_bench"] <= a).mean()),
                p_gen=float((st["p_gen"] <= a).mean()), accept=float(((st["p_gen"] <= a) & ~st["neg"]).mean()), neg=float(st["neg"].mean()),
                holm={f"K{K}": float(((st["p_gen"] <= a / K) & ~st["neg"]).mean()) for K in KS},
                holm_macro={f"K{K}": float((st["p_macro"] <= a / K).mean()) for K in KS})
        out[name] = res
    return out


def _plant_arm(arm, rows, J, D, plant, rng, ok_mean):
    """arm half with planted help: a helped row's wrong reads turn right (answer = the row's right answer when a read shows it)."""
    if not plant:
        return arm
    out = dict(arm)
    for j, r in enumerate(J):
        b = int(D.b[j])
        if b not in plant:
            continue
        share = min(1.0, plant[b] / (Q * (1 - ok_mean[b])))
        if rng.random() >= share:
            continue
        gold = _G["gold"].get(r)
        xs = []
        for x in arm[r]:
            if not x["ok"] and rng.random() < Q:
                x = dict(x, ok=True, pred=gold if gold is not None else "__gold__")
            xs.append(x)
        out[r] = xs
    return out


def _rep_b(args):
    seed, plant = args
    g = _load()
    rows, J, D = g["rows"], g["J"], g["D"]
    if "gold" not in g:
        g["gold"] = {r: next((x["pred"] for x in g["tip8"][r] if x["ok"]), None) for r in J}
        ok = np.array([[x["ok"] for x in g["tip8"][r]] for r in J], float)
        g["ok_mean"] = {b: float(ok[D.b == b].mean()) for b in range(len(D.benches))}
    rng = np.random.default_rng(seed)
    tipH, armH = {}, {}
    for r in J:
        p = rng.permutation(8)
        tipH[r] = [g["tip8"][r][i] for i in p[:4]]
        armH[r] = [g["tip8"][r][i] for i in p[4:]]
    armH = _plant_arm(armH, rows, J, D, plant, rng, g["ok_mean"])
    pols = [dict(n0=1, trigger="none", extra="tip"), dict(n0=2, trigger="disagree", extra="tip", final="plur"),
            dict(n0=2, trigger="disagree", extra="tip", final="extra"), dict(n0=3, trigger="none", extra="tip"), dict(n0=4, trigger="none", extra="tip")]
    F = P.Frontier(tipH, J, rows, policies=pols)
    out = {}
    for strict in (False, True):
        Jg = JV.Judge(rows, tipH, benches=R.BENCHES, frontier=F, strict=strict)
        cands = [Jg.cand_from_arm("always", armH, "cell"),
                 Jg.cand_from_policy(dict(n0=2, trigger="disagree", extra="arm", final="plur"), extra_arm=armH, name="esc_plur"),
                 Jg.cand_from_policy(dict(n0=2, trigger="disagree", extra="arm", final="extra"), extra_arm=armH, name="esc_extra")]
        for c in cands:
            x, info = Jg.diffs(c)
            out[(c["name"], strict)] = (np.array([x[r] for r in J]), info["multiple"])
    return out


def part_b(reps, proc, seed=1000):
    g = _load()
    D = g["D"]
    scen = [("A/A (the arm is the tip)", {}), ("arm better: homogeneous +3", {b: .03 for b in range(6)}),
            ("arm better: homogeneous +6", {b: .06 for b in range(6)}), ("arm better: ONE benchmark +18 (visulogic)", {4: .18})]
    out = {}
    with mp.Pool(proc) as pool:
        for si, (name, plant) in enumerate(scen):
            n = reps if not plant else max(reps // 3, 100)
            rs = pool.map(_rep_b, [(seed + 100000 * si + i, plant) for i in range(n)], chunksize=4)
            res = {}
            for key in rs[0]:
                X = np.stack([r[key][0] for r in rs])
                mult = float(np.mean([r[key][1] for r in rs]))
                for fl in (0.0, 0.5, 1.0):
                    st = D.stats(X, floor_rows=fl)
                    res[f"{key[0]}|{'strict' if key[1] else 'hull'}|floor{fl}"] = dict(
                        multiple=mult, d=float(st["d"].mean()), sd=float(st["d"].std()), se=float(st["se"].mean()),
                        p_macro=float((st["p_macro"] <= .10).mean()), p_macro_05=float((st["p_macro"] <= .05).mean()),
                        p_gen=float((st["p_gen"] <= .10).mean()), accept=float(((st["p_gen"] <= .10) & ~st["neg"]).mean()),
                        holm={f"K{K}": float(((st["p_gen"] <= .10 / K) & ~st["neg"]).mean()) for K in KS}, reps=len(rs))
            out[name] = res
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps-a", type=int, default=6000)
    ap.add_argument("--reps-b", type=int, default=600)
    ap.add_argument("--proc", type=int, default=8)
    ap.add_argument("--nc", type=int, default=4)
    ap.add_argument("--out", default=SD.OUT)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    t = time.time()
    A = part_a(a.reps_a, a.nc)
    print(f"part A done {time.time() - t:.0f}s", flush=True)
    B = part_b(a.reps_b, a.proc) if a.reps_b else {}
    print(f"part B done {time.time() - t:.0f}s", flush=True)
    doc = dict(note="A/A and planted effects from the tip's own samples on the 444 JUDGE rows (E-cache samples 0-3 + W-null samples 0-3)",
               reps_a=a.reps_a, reps_b=a.reps_b, nc=a.nc, Q=Q, part_a=A, part_b=B, floor_default=ST.FLOOR_ROWS)
    json.dump(doc, open(os.path.join(a.out, "aa_power.json"), "w"), indent=1)
    L = [f"v5 statistic: false accepts (A/A) and power (planted), 444 JUDGE rows, {a.nc} candidate reads vs {8 - a.nc} reference reads per row, {a.reps_a} replicates",
         "accept rates at nominal 0.10; Holm columns = p_gen at the first threshold 0.10 / K and no significantly negative benchmark", ""]
    for fl in (0.5, 0.0, 1.0):
        L.append(f"PART A, se floor {fl} row(s) per benchmark" + (" (the default)" if fl == ST.FLOOR_ROWS else ""))
        L.append(f"{'scenario':46s} {'macro d':>8s} {'p_macro':>8s} {'v4 form':>8s} {'bench t':>8s} {'bench sf':>8s} {'p_gen':>8s} | Holm K=1 {'K=3':>6s} {'K=6':>6s} {'K=10':>6s}")
        for name, _ in SCEN:
            r = A[name][f"floor{fl}"]
            L.append(f"{name:46s} {100 * r['d']:+8.2f} {r['p_macro']:8.3f} {r['v4_macro_lobo_point']:8.3f} {r['bench_t']:8.3f} {r['bench_signflip']:8.3f} {r['p_gen']:8.3f} |"
                     f" {r['holm']['K1']:8.3f} {r['holm']['K3']:6.3f} {r['holm']['K6']:6.3f} {r['holm']['K10']:6.3f}")
        L.append("")
    if B:
        L.append(f"PART B, the whole judge (frontier of a 4-read tip half; the arm is the other half); {a.reps_b} replicates for A/A")
        L.append(f"{'scenario':42s} {'candidate':10s} {'ref':6s} {'floor':>5s} {'tokens':>7s} {'mean d':>7s} {'sd d':>6s} {'mean se':>7s} {'p_macro':>8s} {'p_gen':>7s} | Holm K=1 {'K=3':>6s} {'K=6':>6s}")
        for name, res in B.items():
            for key, r in res.items():
                cn, ref, fl = key.split("|")
                L.append(f"{name:42s} {cn:10s} {ref:6s} {fl[5:]:>5s} x{r['multiple']:6.2f} {100 * r['d']:+7.2f} {100 * r['sd']:6.2f} {100 * r['se']:7.2f} {r['p_macro']:8.3f} {r['p_gen']:7.3f} |"
                         f" {r['holm']['K1']:8.3f} {r['holm']['K3']:6.3f} {r['holm']['K6']:6.3f}")
            L.append("")
    open(os.path.join(a.out, "aa_power.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
