"""PY reads as ONE component of a read policy (offline, signal rows, cached base samples x PY samples): mixed votes of base and PY reads
against the base's own vote frontier at equal generated tokens (policy5.evaluate_views; picture-clustered se as in py_table.py).
usage: py_mix.py PY_COMPACT.jsonl[.gz] BASE.jsonl.gz LEARN.jsonl"""
import collections, gzip, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
for d in (os.path.join(HERE, "..", "v5"), "/workspace/fudan/mingdazhang/selfedit-v5/v5"):
    if os.path.exists(os.path.join(d, "policy5.py")):
        sys.path.insert(0, d); break
import policy5, reads5, stats5
opn = lambda p: gzip.open(p, "rt") if p.endswith(".gz") else open(p)
py_p, base_p, rows_p = sys.argv[1:4]
rows = reads5.load_rows(rows_p)
sig = set(reads5.rows_of(rows, reads5.SIGNAL_ROLES))
PY, base = collections.defaultdict(dict), collections.defaultdict(dict)
for l in opn(py_p):
    x = json.loads(l)
    if not x.get("error") and x.get("ok") is not None and x["row"] in sig:
        PY[x["row"]][int(x["sample"])] = dict(ok=bool(x["ok"]), tokens=float(x["tokens"]), pred=x.get("pred"), cls=None, guess=None, finish=x.get("finish"),
                                               s=int(x["sample"]), prompt_tokens=float(x["prompt_tokens"]))
for l in opn(base_p):
    v = json.loads(l)
    if v["row"] in sig:
        base[v["row"]][int(v["s"])] = dict(ok=bool(v["ok"]), tokens=float(v["tokens"]), pred=v.get("pred"), cls=None, guess=None, finish=v.get("finish"), s=int(v["s"]),
                                           prompt_tokens=float(v["prompt_tokens"] or 0))
R = sorted(r for r in PY if len(PY[r]) >= 2 and len(base.get(r, {})) >= 8)
arms = {"base": {r: [base[r][s] for s in sorted(base[r])] for r in R}, "py": {r: [PY[r][s] for s in sorted(PY[r])] for r in R}}
benches = sorted({rows[r]["bench"] for r in R})
pols = [dict(n0=1, trigger="none", extra="tip"), dict(n0=2, trigger="disagree", extra="tip", final="plur"),
        dict(n0=3, trigger="disagree", extra="tip", n_extra=2, final="plur"), dict(n0=3, trigger="none", extra="tip"), dict(n0=5, trigger="none", extra="tip"), dict(n0=7, trigger="none", extra="tip")]
F = policy5.Frontier(arms["base"], R, rows, policies=pols, cost="tokens")
V = policy5.views_policy
cands = [("one PY read", V(["py"])),
         ("2 PY reads + base read on disagreement", V(["py", "py"], ["base"])),
         ("base + PY, third = base on disagreement", V(["base", "py"], ["base"])),
         ("base + PY, third = PY on disagreement", V(["base", "py"], ["py"])),
         ("2 base reads, PY read on disagreement (plurality)", V(["base", "base"], ["py"])),
         ("2 base reads, PY read decides on disagreement", V(["base", "base"], ["py"], "extra")),
         ("3 base reads; base + PY more unless unanimous", V(["base", "base", "base"], ["base", "py"])),
         ("reference: 2 base reads + base on disagreement", V(["base", "base"], ["base"]))]
print(f"rows {len(R)}; base one read {100 * policy5.macro_mean(F.table['vote1'], rows, 'ok'):.2f} at {F.base_tokens:.0f} generated tokens")
print(f"{'policy':52s} {'acc':>6s} {'tokens x':>8s} {'escal.':>6s} {'frontier':>8s} {'policy - frontier':>18s}  per benchmark (policy - frontier)")
for name, pol in cands:
    t = policy5.evaluate_views(pol, arms, R)
    acc, tok = policy5.macro_mean(t, rows, "ok"), policy5.macro_mean(t, rows, "tokens")
    ref = F.ref_at(tok)
    if ref is None:
        print(f"{name:52s} {100 * acc:6.2f} x{tok / F.base_tokens:6.2f}  beyond the frontier")
        continue
    rok, _, desc = ref
    E = stats5.bench_effects({r: t[r]["ok"] - rok[r] for r in R}, rows)
    M = stats5.macro(E, benches)
    print(f"{name:52s} {100 * acc:6.2f} x{tok / F.base_tokens:6.2f} {100 * policy5.macro_mean(t, rows, 'esc'):5.1f}% {100 * desc['acc']:8.2f} {100 * M['d']:+8.2f} ± {100 * M['se']:.2f}   "
          + " ".join(f"{b[:5]} {100 * E[b]['d']:+5.1f}" for b in benches) + f"  p_gen {stats5.p_gen(E, benches)[0]:.2f}")
