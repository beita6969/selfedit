"""PY operator table: PY reads against the cached base reads of the same rows (floor-only v5t000, 8 samples per row; signal rows only).

 - accuracy per benchmark and macro; PY - base paired by row (row mean of the PY samples minus row mean of the 8 base samples),
   picture-clustered se inside a benchmark (stats5.bench_effects = the cl() of zoom_table.py), macro se = sqrt(sum se_b^2) / k;
 - tokens as a multiple of one base read: generated only, and prompt + generated of every call;
 - share of reads that ran code, runs per read, run outcomes (ok / error / refused / timeout / killed);
 - accuracy split by "ran code" / "answered directly", each against the base on the same rows;
 - EQUAL generated tokens: the base's own vote frontier on these rows (policy5.Frontier over the cached base samples: one read; 2 reads +
   a third on disagreement; 3 reads + 2 more unless unanimous), interpolated between its vertices at the PY read's mean generated tokens
   (macro mean over benchmarks, the Frontier's convention); PY - frontier paired by row, same se; p_macro and p_gen of stats5 beside it.
usage: py_table.py PY_READS.jsonl[.gz] BASE_EXPORT.jsonl.gz LEARN.jsonl [--rows-out FILE.tsv]"""
import collections
import gzip
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for d in (os.path.join(HERE, "..", "v5"), "/workspace/fudan/mingdazhang/selfedit-v5/v5"):
    if os.path.exists(os.path.join(d, "policy5.py")):
        sys.path.insert(0, d)
        break
import policy5
import reads5
import stats5


def opn(p):
    return gzip.open(p, "rt") if p.endswith(".gz") else open(p)


def main():
    py_path, base_path, rows_path = sys.argv[1:4]
    rows_out = sys.argv[sys.argv.index("--rows-out") + 1] if "--rows-out" in sys.argv else None
    rows = reads5.load_rows(rows_path)
    sig = set(reads5.rows_of(rows, reads5.SIGNAL_ROLES))
    # ---- PY reads
    PY, bad = collections.defaultdict(dict), 0
    for l in opn(py_path):
        if not l.strip():
            continue
        x = json.loads(l)
        if x.get("error") or x.get("ok") is None:
            bad += 1
            continue
        if x["row"] not in sig:
            raise SystemExit(f"row {x['row']} is not a signal row")
        PY[x["row"]][int(x["sample"])] = x
    # ---- base reads (8 cached samples)
    base = collections.defaultdict(dict)
    for l in opn(base_path):
        v = json.loads(l)
        if v["row"] in sig and v.get("ok") is not None and v.get("finish") != "abort":
            base[v["row"]][int(v["s"])] = dict(ok=bool(v["ok"]), tokens=float(v["tokens"]), pred=v.get("pred"), cls=v.get("cls"), guess=None,
                                               finish=v.get("finish"), s=int(v["s"]), prompt_tokens=v.get("prompt_tokens"))
    base = {r: [d[s] for s in sorted(d)] for r, d in base.items()}
    R = sorted(r for r in PY if r in base)
    benches = sorted({rows[r]["bench"] for r in R})
    nb = collections.Counter(len(base[r]) for r in R)
    n_reads = sum(len(PY[r]) for r in R)
    print(f"PY reads {n_reads} on {len(R)} signal rows (unread / error lines {bad}); samples per row {dict(collections.Counter(len(PY[r]) for r in R))}; "
          f"base samples per row {dict(nb)}")

    mean = lambda v: sum(v) / len(v) if v else float("nan")
    py_ok = {r: mean([float(x["ok"]) for x in PY[r].values()]) for r in R}
    py_gen = {r: mean([float(x["tokens"]) for x in PY[r].values()]) for r in R}
    py_tot = {r: mean([float(x["tokens"]) + float(x["prompt_tokens"]) for x in PY[r].values()]) for r in R}
    b_ok = {r: mean([float(x["ok"]) for x in base[r]]) for r in R}
    b_gen = {r: mean([x["tokens"] for x in base[r]]) for r in R}
    b_tot = {r: mean([x["tokens"] + float(x["prompt_tokens"] or 0) for x in base[r]]) for r in R}
    b_same = {r: mean([float(x["ok"]) for x in base[r] if x["s"] in PY[r]]) for r in R}

    def bmean(d, b):
        return mean([d[r] for r in R if rows[r]["bench"] == b])

    def macro(d):
        return mean([bmean(d, b) for b in benches])

    def eff(x, rs=None):
        """paired differences {row: d} -> (per-benchmark effects, macro dict)"""
        x = {r: v for r, v in x.items() if rs is None or r in rs}
        E = stats5.bench_effects(x, rows)
        return E, stats5.macro(E, sorted(E))

    f = lambda m: f"{100 * m['d']:+6.2f} ± {100 * m['se']:.2f}"
    print("\n== accuracy (PY: mean over its samples; base: mean over the 8 cached samples of the same rows)")
    print(f"{'bench':12s} {'rows':>4s} {'base8':>7s} {'base@s':>7s} {'PY':>7s}  {'PY-base8 (se)':>16s}   gen tokens base -> PY (x)      total tokens base -> PY (x)")
    D = {r: py_ok[r] - b_ok[r] for r in R}
    E, M = eff(D)
    for b in benches:
        n = sum(rows[r]["bench"] == b for r in R)
        print(f"{b:12s} {n:4d} {100 * bmean(b_ok, b):7.2f} {100 * bmean(b_same, b):7.2f} {100 * bmean(py_ok, b):7.2f}  {f(E[b]):>16s}   "
              f"{bmean(b_gen, b):6.0f} -> {bmean(py_gen, b):6.0f} (x{bmean(py_gen, b) / bmean(b_gen, b):4.2f})    "
              f"{bmean(b_tot, b):6.0f} -> {bmean(py_tot, b):6.0f} (x{bmean(py_tot, b) / bmean(b_tot, b):4.2f})")
    xg, xt = macro(py_gen) / macro(b_gen), macro(py_tot) / macro(b_tot)
    print(f"{'MACRO':12s} {len(R):4d} {100 * macro(b_ok):7.2f} {100 * macro(b_same):7.2f} {100 * macro(py_ok):7.2f}  {f(M):>16s}   "
          f"{macro(b_gen):6.0f} -> {macro(py_gen):6.0f} (x{xg:4.2f})    {macro(b_tot):6.0f} -> {macro(py_tot):6.0f} (x{xt:4.2f})")
    print(f"   benchmarks + {sum(E[b]['d'] > 0 for b in benches)} / - {sum(E[b]['d'] < 0 for b in benches)}; p_macro {stats5.p_macro(E, benches):.3f}; "
          f"p_gen {stats5.p_gen(E, benches)[0]:.3f} (carrier {stats5.p_gen(E, benches)[1]}); pooled tokens x gen "
          f"{mean(list(py_gen.values())) / mean(list(b_gen.values())):.2f}, total {mean(list(py_tot.values())) / mean(list(b_tot.values())):.2f}")

    # ---- code use
    reads = [(r, s, x) for r in R for s, x in sorted(PY[r].items())]
    ran = [(r, s, x) for r, s, x in reads if x.get("n_runs")]
    print("\n== code use")
    print(f"{'bench':12s} {'reads':>5s} {'ran code':>9s} {'runs/read':>9s} {'2 runs':>7s}  run outcomes                         reads with a failed / refused run   code after the limit   cut (length)")
    def use_line(lab, rr):
        k = [x for _, _, x in rr if x.get("n_runs")]
        runs = [u for _, _, x in rr for u in x.get("runs", [])]
        oc = collections.Counter(u["status"] for u in runs)
        nfail = sum(any(u["status"] != "ok" for u in x.get("runs", [])) for x in k)
        print(f"{lab:12s} {len(rr):5d} {100 * len(k) / max(len(rr), 1):8.1f}% {len(runs) / max(len(rr), 1):9.2f} {100 * sum(x['n_runs'] >= 2 for x in k) / max(len(rr), 1):6.1f}%  "
              f"{str(dict(sorted(oc.items()))):36s} {nfail:4d} ({100 * nfail / max(len(k), 1):.0f}% of code reads)          "
              f"{sum(bool((x.get('flags') or {}).get('code_after_limit')) for _, _, x in rr):4d}               {sum(x.get('finish') == 'length' for _, _, x in rr)}")
    for b in benches:
        use_line(b, [t for t in reads if rows[t[0]]["bench"] == b])
    use_line("ALL", reads)
    runs = [u for _, _, x in reads for u in x.get("runs", [])]
    if runs:
        oc = collections.Counter(u["status"] for u in runs)
        print(f"   runs {len(runs)}: ok {100 * oc['ok'] / len(runs):.1f}%, error {100 * oc['error'] / len(runs):.1f}%, refused {100 * oc['refused'] / len(runs):.1f}%, "
              f"timeout / killed {100 * (oc['timeout'] + oc['killed']) / len(runs):.1f}%; runs that returned an image {100 * sum(bool(u.get('n_sent')) for u in runs) / len(runs):.1f}%")
        ref = collections.Counter((u.get("error") or "")[:70] for u in runs if u["status"] == "refused")
        err = collections.Counter((u.get("error") or "").split(":")[0][:40] for u in runs if u["status"] == "error")
        print("   refusals:", dict(ref.most_common(8)))
        print("   errors:", dict(err.most_common(8)))
    flags = collections.Counter(k for _, _, x in reads for k, v in (x.get("flags") or {}).items() if v)
    print("   flags:", dict(flags))

    # ---- split by ran code / answered directly (read level; base = the row's 8-sample mean)
    print("\n== accuracy by what the read did (read level; base8 = mean of the row's 8 base samples, weighted like the reads; se picture-clustered over all rows of the group)")
    def cl(pairs):
        byr = collections.defaultdict(list)
        for r, d in pairs:
            byr[r].append(d)
        if len(byr) < 2:
            return None
        v = {r: sum(x) / len(x) for r, x in byr.items()}
        n = len(v)
        m = sum(v.values()) / n
        byc = collections.defaultdict(float)
        for r, d in v.items():
            byc[(rows[r]["bench"], rows[r]["pic"])] += d - m
        G = len(byc)
        return m, math.sqrt(sum(x * x for x in byc.values()) * G / max(G - 1, 1)) / n, n
    def grp(lab, rr):
        if not rr:
            print(f"{lab:34s} reads    0")
            return
        t = cl([(r, float(x["ok"]) - b_ok[r]) for r, s, x in rr])
        print(f"{lab:34s} reads {len(rr):4d} rows {len({r for r, _, _ in rr}):3d}  PY {100 * mean([float(x['ok']) for _, _, x in rr]):6.2f}  base8 {100 * mean([b_ok[r] for r, _, _ in rr]):6.2f}  "
              f"PY-base8 {'n/a' if not t else f'{100 * t[0]:+6.2f} ± {100 * t[1]:.2f}'}   gen tokens PY {mean([float(x['tokens']) for _, _, x in rr]):6.0f} vs base {mean([b_gen[r] for r, _, _ in rr]):6.0f}")
    grp("ran code", ran)
    grp("answered directly", [t for t in reads if not t[2].get("n_runs")])
    grp("ran code, every run ok", [t for t in ran if all(u["status"] == "ok" for u in t[2]["runs"])])
    grp("ran code, some run failed/refused", [t for t in ran if any(u["status"] != "ok" for u in t[2]["runs"])])
    grp("ran code, 1 run", [t for t in ran if t[2]["n_runs"] == 1])
    grp("ran code, 2 runs", [t for t in ran if t[2]["n_runs"] >= 2])
    grp("ran code, an image came back", [t for t in ran if any(u.get("n_sent") for u in t[2]["runs"])])
    for b in benches:
        grp(f"  {b}: ran code", [t for t in ran if rows[t[0]]["bench"] == b])
        grp(f"  {b}: direct", [t for t in reads if not t[2].get("n_runs") and rows[t[0]]["bench"] == b])
    # rows where the base is undecided or wrong: where a tool could add something
    for lab, pred in (("base8 row rate <= 25 %", lambda v: v <= 0.25), ("base8 row rate 25-75 %", lambda v: 0.25 < v < 0.75),
                      ("base8 row rate >= 75 %", lambda v: v >= 0.75)):
        grp(lab + ", ran code", [t for t in ran if pred(b_ok[t[0]])])
        grp(lab + ", direct", [t for t in reads if not t[2].get("n_runs") and pred(b_ok[t[0]])])

    # ---- equal tokens against the base's own vote frontier on these rows
    pols = [dict(n0=1, trigger="none", extra="tip"), dict(n0=2, trigger="disagree", extra="tip", final="plur"),
            dict(n0=3, trigger="disagree", extra="tip", n_extra=2, final="plur"), dict(n0=3, trigger="none", extra="tip"),
            dict(n0=5, trigger="none", extra="tip")]
    arm = {r: base[r] for r in R}
    F = policy5.Frontier(arm, R, rows, policies=pols, cost="tokens")
    Ft = policy5.Frontier(arm, R, rows, policies=pols, cost="total", table=F.table)
    print("\n== the base's vote frontier on these rows (generated tokens | prompt + generated tokens)")
    for a, b in zip(F.lines(), Ft.lines()):
        print("  " + a + "   |   " + b.split("tokens", 1)[1].strip())
    for lab, Fr, T in (("EQUAL GENERATED TOKENS", F, macro(py_gen)), ("equal TOTAL tokens (prompt + generated of every call)", Ft, macro(py_tot))):
        ref = Fr.ref_at(T)
        print(f"\n== {lab}: PY at {T:.0f} tokens = x{T / Fr.base_tokens:.2f} of one base read")
        if ref is None:
            print("   beyond the evaluated frontier")
            continue
        rok, rtok, desc = ref
        print(f"   frontier there: between {desc['lo']} and {desc['hi']} (weight {desc['w']:.3f} on the upper{', FLAT above the last vertex' if desc['flat'] else ''}) = {100 * desc['acc']:.2f}")
        Dq = {r: py_ok[r] - rok[r] for r in R}
        Eq, Mq = eff(Dq)
        for b in benches:
            print(f"   {b:12s} PY {100 * bmean(py_ok, b):6.2f}  frontier {100 * bmean(rok, b):6.2f}  PY-frontier {f(Eq[b])}")
        pg = stats5.p_gen(Eq, benches)
        print(f"   {'MACRO':12s} PY {100 * macro(py_ok):6.2f}  frontier {100 * macro(rok):6.2f}  PY-frontier {f(Mq)}   benchmarks + {sum(Eq[b]['d'] > 0 for b in benches)} / - "
              f"{sum(Eq[b]['d'] < 0 for b in benches)}; p_macro {stats5.p_macro(Eq, benches):.3f}; p_gen {pg[0]:.3f} (carrier {pg[1]})")
    # vertices for reference
    print("\n   reference points on these rows: " + "; ".join(f"{n} {100 * acc:.2f} at x{tk / F.base_tokens:.2f}" for tk, acc, n in F.vertices))

    if rows_out:
        with open(rows_out, "w") as fo:
            fo.write("row\tbench\tcategory\tbase8_ok\tbase_preds\tpy_ok\tpy_preds\tpy_runs\tpy_gen_tokens\tbase_gen_tokens\n")
            for r in R:
                bp = collections.Counter(str(x["pred"]) for x in base[r])
                fo.write("\t".join([r, rows[r]["bench"], str(rows[r].get("category")), f"{b_ok[r]:.3f}", json.dumps(dict(bp.most_common())),
                                    f"{py_ok[r]:.2f}", json.dumps([str(x.get("pred")) for _, x in sorted(PY[r].items())]),
                                    json.dumps([x.get("n_runs") for _, x in sorted(PY[r].items())]), f"{py_gen[r]:.0f}", f"{b_gen[r]:.0f}"]) + "\n")
        print(f"\nrow table -> {rows_out}")


if __name__ == "__main__":
    main()
