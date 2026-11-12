"""B4 checks (coordinator 11-11, Track B): signal_cells.py (cell maker, calibration) and signal_plan.py (measurement planner).
 1. Statistics: Student t and BH against scipy / judge_bh, the cluster-robust se against judge_eb.family, the loop's constants.
 2. The guard: a judge row or an unknown row in the tip table, a cell's reads, a class, a traces file, engine result files, the
    planner's ingest or its request lines raises; nothing is written.
 3. Planted effects (synthetic tables on the real row map, independent draws): a global gain, a gain only on long-read rows, a cell
    that gains more at 3x tokens, pure noise -> the frontier keeps the global cell, the 3x cell does not fit the budget (and is docked
    by the vote of 3 tip reads on the judge's scale), noise cells get no positive bound (also over 20 all-noise tables), the long-row
    gain shows in the read-length split and is kept as a ROUTED cell on the router class that holds those rows.
 4. The two pairings: coupled reads (common random numbers) -> "key" has the smaller se and is used; independent reads -> "row";
    pred is the row (judge-convention) value when the two agree within their se.
 5. Budget: no stated token multiple -> error; RUN/cards.json g_token_multiple (number or object) with its source; a larger budget
    lets the 3x cell in; lambda is the ratio of the first increment that did not fit.
 6. The judge's test: p_pass falls with K (prior_family), agrees with a direct simulation of the LITE judge, and a cell that cannot
    pass at its K is held, not sent.
 7. Engine result files + a mapping -> the same document as the tables; tip lines by mapping too. Per-read tables (tsv, the form the
    engine agent writes per batch) with batch_cells.json: cell specs from the names, the concurrent control (W-null) as the primary
    reference with the tip's cached reads as secondary, measurement-only families, cells without an edit path held, reads that hit the
    token budget counted, vote gains without preds, router classes from prefix hashes, the watch that appends each table.
 8. cells.json round-trips through author_g.g_read_cells (gen.Run): rank order, kinds, numbers, cells_sha; routed cells are skipped
    by today's reader; no row id anywhere in cells.json or the table.
 9. Calibration: a planted slope is recovered, new predictions are shrunk by it, the raw prediction comes from the archived cells file,
    the quality number rises from the generation that overclaims to the generations written with the fitted slope.
10. Planner: restrict_edit, the race order, a simulated race (the planted best band wins, budget kept, only signal rows requested,
    explicit cells raced as they are, waiting while reads are pending), engine request lines.
11. With the patch applied (signal_cells_1111.patch; skipped on the unpatched tree): author G proposes routed cells, the judge's BH
    family of G = its candidates + the largest prior_family, realised.jsonl carries pred_raw / prior_family / p_pass.
12. B4b: prior_family required for weight cells (integer >= 0) in make and the planner; the family = cells sent + the SUM over distinct
    libraries (signal_cells.family_prior = author_g.g_prior); the reader skips an invalid prior_family / library.
usage: python3 t_signal_cells.py"""
import collections, json, math, os, sys, tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(os.path.dirname(HERE), "harness")]
os.chdir(HERE)
import signal_cells as SC  # noqa: E402
import signal_plan as SP  # noqa: E402
import signal_export as SX  # noqa: E402

ROWS = os.path.join(HERE, "..", "data", "learn.jsonl")
ok = fail = 0


def check(name, cond):
    global ok, fail
    print(("ok   " if cond else "FAIL ") + name, flush=True)
    ok, fail = ok + bool(cond), fail + (not cond)


def raises(fn, text="may be used"):
    try:
        fn()
    except ValueError as ex:
        return text in str(ex)
    return False


R = SC.Rows.from_learn(ROWS)
JUDGE = sorted(r for r, m in R.meta.items() if m["role"] == "judge")
BUD = dict(M=2.0, source="test", base=None)


def world(seed, effects, S=8, m=2, coupled=False, rows=None):
    """Synthetic tables on the real signal rows. A row has a true accuracy and is short (about 800 tokens) or long (about 4000, less
    accurate), half of each benchmark. effects = {name: (gain on every row, extra gain on long rows, token multiple[, kind[, family]])};
    cells of one family are alternatives of each other (one menu).
    coupled: a cell's sample s uses the tip's sample-s random number (common random numbers)."""
    rng = np.random.default_rng(seed)
    rows = rows or R.sig
    byb = collections.defaultdict(list)
    for r in rows:
        byb[R.bench(r)].append(r)
    long_ = {}
    for b, rs in byb.items():
        for i, j in enumerate(rng.permutation(len(rs))):
            long_[rs[j]] = i % 2 == 0
    p = {r: float(np.clip(rng.beta(4, 2) - (0.25 if long_[r] else 0.0), 0.05, 0.8)) for r in rows}
    base = {r: 4000.0 if long_[r] else 800.0 for r in rows}
    U = {r: rng.random(64) for r in rows}
    rd = lambda okv, tok: dict(ok=float(okv), tokens=float(tok), pred="A" if okv else "B", skip=False)   # noqa: E731
    tip = {r: {s: rd(U[r][s] < p[r], base[r] * math.exp(0.3 * rng.standard_normal())) for s in range(S)} for r in rows}
    cells, reads = {}, {}
    for name, e in effects.items():
        g, gl, tm = e[:3]
        kind = e[3] if len(e) > 3 else "weight"
        cells[name] = dict(kind=kind, name=name, family=e[4] if len(e) > 4 else "F", **({"prior_family": 0} if kind == "weight" else {}),   # B4b: required for weight
                           **({"edit": {"op": "surgeon", "file": f"/data/fd/mingdazhang/x/{name}.json"}} if kind == "weight"
                                                              else {"settings": {"temperature": 0.8}} if kind == "setting" else {}))
        reads[name] = {r: {s: rd((U[r][s] if coupled else rng.random()) < p[r] + g + (gl if long_[r] else 0.0), tm * base[r] * math.exp(0.3 * rng.standard_normal()))
                           for s in range(m)} for r in rows}
    return tip, cells, reads, long_


def by_name(doc, key="cells"):
    return {it["name"]: it for it in doc[key]}


# ---------------------------------------------------------------------------------------------------------------- 1. statistics
try:
    from scipy.stats import t as _t
    err = max([abs(SC.t_sf(x, nu) - _t.sf(x, nu)) for nu in (1, 2, 5, 30, 317) for x in (-3, -1, 0, 0.5, 1.3, 2.5, 6)]
              + [abs(SC.t_ppf(q, nu) - _t.ppf(q, nu)) / max(1.0, abs(_t.ppf(q, nu))) for nu in (1, 2, 5, 30, 317) for q in (0.6, 0.9, 0.99, 0.9984, 0.9999)])
    check(f"Student t (sf, ppf) agrees with scipy (max error {err:.1e})", err < 1e-9)
    import judge_bh as BH
    import judge_eb as EB
    rng = np.random.default_rng(0)
    ps = list(rng.random(40) ** 3)
    check("BH equals judge_bh.bh; ALPHA equals judge_bh.ALPHA", SC.bh(ps) == set(np.nonzero(BH.bh(ps))[0].tolist()) and SC.ALPHA == BH.ALPHA)
    one = [r for r in R.sig if R.bench(r) == "lego"]
    vals = {r: float(rng.normal(0.05, 0.4)) for r in one}
    cl = collections.defaultdict(list)
    for r in one:
        cl[R.cluster(r)].append(vals[r])
    D, S_, nu = EB.family([dict(name="x", d={c: float(np.mean(v)) for c, v in cl.items()}, w={c: len(v) for c, v in cl.items()})])
    st = SC.strat_stat(vals, R)
    check(f"one stratum: mean and cluster-robust se equal judge_eb.family ({st['d']:+.5f} / {st['se']:.5f}, nu {st['nu']:.0f})",
          abs(st["d"] - D[0]) < 1e-12 and abs(st["se"] - math.sqrt(S_[0, 0])) < 1e-12 and st["nu"] == nu[0] and not st["strat"])
except ImportError:
    print("skip scipy / judge_bh comparisons (scipy not installed)")
check("SIGNAL_ROLES equals signal_export's; 324 signal rows, 6 benchmarks at equal shares; JUDGE counted only (74 per benchmark)",
      SC.SIGNAL_ROLES == SX.SIGNAL_ROLES and len(R.sig) == 324 and len(R.W) == 6 and max(R.W.values()) - min(R.W.values()) < 1e-12 and set(R.n_judge.values()) == {74})
rng = np.random.default_rng(1)
allv = {r: float(rng.normal(0.0, 0.5) + (0.3 if R.bench(r) == "blink" else 0.0)) for r in R.sig}
sub = {r: v for r, v in allv.items() if R.bench(r) != "blink" or rng.random() < 0.3}          # one benchmark under-sampled
st = SC.strat_stat(sub, R)
plain = float(np.mean(list(sub.values())))
check(f"post-stratified by benchmark at the signal rows' shares: an under-sampled benchmark keeps its weight ({st['d']:+.4f} vs plain {plain:+.4f})",
      st["strat"] and abs(st["d"] - np.mean([np.mean([v for r, v in sub.items() if R.bench(r) == b]) for b in R.W])) < 1e-12 and abs(st["d"] - plain) > 1e-4)
check("ref_m follows ecomp (round: 1.49 -> none, 1.59 -> 2, 2.6 -> 3; ceil: 1.5 -> 2, 2.1 -> 3; off)",
      [SC.ref_m(c) for c in (1.49, 1.59, 2.49, 2.6)] == [None, 2, 2, 3] and [SC.ref_m(c, "ceil") for c in (1.4, 1.5, 2.1)] == [None, 2, 3] and SC.ref_m(3.0, "off") is None)
try:
    import ecomp
    check("ref_m equals ecomp.ref_m_round / ecomp.ref_m on a grid", all(SC.ref_m(c) == ecomp.ref_m_round(c) and SC.ref_m(c, "ceil") == ecomp.ref_m(c) for c in np.arange(0.8, 4.6, 0.05)))
except Exception as ex:                                           # noqa: BLE001
    print(f"skip ecomp comparison ({type(ex).__name__})")
tipv = {R.sig[0]: {s: dict(ok=float(s in (0, 1, 2, 5)), tokens=100.0, pred="A" if s in (0, 1, 2, 5) else "B", skip=False) for s in range(8)}}
v2, v3 = SC.vote_rows(tipv, R, 2)[R.sig[0]], SC.vote_rows(tipv, R, 3)[R.sig[0]]
check(f"vote gain from the tip's own samples: m 2 is exactly 0 (a tie goes to the first), m 3 = plurality of every window minus the mean ({v3:+.4f})",
      abs(v2) < 1e-12 and abs(v3 - (3 / 8 - 0.5)) < 1e-12)   # windows AAA AAB ABB BBA BAB ABB BBA BAA: 3 of 8 right

# ---------------------------------------------------------------------------------------------------------------- 2. the guard
tip, cells, reads, long_ = world(3, {"a": (0.08, 0.0, 1.3)})
jr = JUDGE[0]
bad = dict(reads, a=dict(reads["a"], **{jr: {0: dict(ok=1.0, tokens=10.0, pred="A", skip=False)}}))
check("a judge row in a cell's reads raises", raises(lambda: SC.make("t", tip, cells, bad, R, BUD), "has role 'judge'"))
check("a judge row in the tip table raises", raises(lambda: SC.make("t", dict(tip, **{jr: tip[R.sig[0]]}), cells, reads, R, BUD), "has role 'judge'"))
check("an unknown row raises", raises(lambda: SC.make("t", dict(tip, nope=tip[R.sig[0]]), cells, reads, R, BUD), "not in the role map"))
check("a judge row in a class raises", raises(lambda: SC.make("t", tip, cells, reads, R, BUD, classes=[dict(name="c", type="cond", condition="x", rows={jr})]), "has role 'judge'"))
tr = {R.sig[0]: dict(role="show", bench="b", samples={"0": dict(ok=True, tokens=5)}), jr: dict(role="judge", bench="b", samples={"0": dict(ok=True, tokens=5)})}
check("traces.json with a judge row raises (tip_table); a show row that says role judge raises too",
      raises(lambda: SC.tip_table(tr, R), "has role 'judge'") and raises(lambda: SC.tip_table({R.sig[0]: dict(role="judge", samples={})}, R), "has role 'judge'"))
check("reads_table refuses a judge row", raises(lambda: SC.reads_table([dict(cell="a", row=jr, sample=0, ok=True, tokens=3)], R), "has role 'judge'"))
d0 = tempfile.mkdtemp(prefix="t_sc_guard_")
rf = os.path.join(d0, "wsig_a.jsonl")
with open(rf, "w") as fh:
    fh.write(json.dumps(dict(key="k1", row=R.sig[0], sample=0, ok=True, tokens=9)) + "\n" + json.dumps(dict(key="k2", row=jr, sample=0, ok=True, tokens=9)) + "\n")
check("engine result files with a judge row raise in ingest_results", raises(lambda: SC.ingest_results([rf], {"cells": [dict(name="a", kind="weight", match={"batch": "wsig_a"})]}, R), "has role 'judge'"))
check("error / cancelled / abort lines and lines without a grade are no reads; op_skipped is a skip",
      all(SC.rec_of(x) is None for x in (dict(error="boom"), dict(cancelled=True, ok=False), dict(finish="abort", ok=False, tokens=3), dict(tokens=3), None))
      and SC.rec_of(dict(op_skipped=True, ok=False))["skip"] and SC.rec_of(dict(ok=False, tokens=7, finish="length")) == dict(ok=0.0, tokens=7.0, pred=None, skip=False, cut=True)
      and "cut" not in SC.rec_of(dict(ok=True, tokens=7, finish="stop")))

# ---------------------------------------------------------------------------------------------------------------- 3. planted effects
EFF = {"glob": (0.08, 0.0, 1.3), "longgain": (0.0, 0.16, 1.5), "costly": (0.20, 0.0, 3.0)}
EFF.update({f"noise{i}": (0.0, 0.0, 1.0 + 0.02 * i) for i in range(6)})
tip, cells, reads, long_ = world(11, EFF)
Lrows = {r for r in tip if long_[r]}
CL = [dict(name="cond:long", type="cond", condition="Does the question need a long derivation?", rows=Lrows)]
doc = SC.make("t000", tip, cells, reads, R, BUD, classes=CL)
C, NS = by_name(doc), by_name(doc, "not_selected")
glob = [it for it in doc["cells"] if it["kind"] == "weight"]
check(f"the frontier keeps the global cell: {[it['name'] for it in glob]} (pred {100 * C['glob']['pred']:+.1f} pp, lb {100 * C['glob']['lb']:+.1f}, x{C['glob']['tokens_multiple']})",
      [it["name"] for it in glob] == ["glob"] and C["glob"]["lb"] > 0 and 0.03 < C["glob"]["pred"] < 0.13 and abs(C["glob"]["tokens_multiple"] - 1.3) < 0.06)
check(f"the 3x cell gains more ({100 * NS['costly']['gain']:+.1f} pp) but does not fit the budget ({NS['costly']['why'][:44]}...); lambda = that increment's ratio "
      f"({doc['lam_pp_per_ktok']} pp / 1k tokens)", "costly" not in C and "does not fit the budget" in NS["costly"]["why"] and NS["costly"]["gain"] > C["glob"]["gain"]
      and doc["lam"] is not None and doc["lam"] > 0)
check(f"judge scale: tokens x{NS['costly']['tokens_multiple']} -> reference = plurality of {NS['costly']['ref_m']} tip reads, pred = gain - vote gain "
      f"({100 * NS['costly']['gain']:+.2f} - {100 * doc['votes']['3']['gain']:.2f} = {100 * NS['costly']['pred_raw']:+.2f})",
      NS["costly"]["ref_m"] == 3 and doc["votes"]["3"]["gain"] > 0.005 and abs(NS["costly"]["gain"] - doc["votes"]["3"]["gain"] - NS["costly"]["pred_raw"]) < 2e-3
      and C["glob"]["ref_m"] is None and abs(C["glob"]["gain"] - C["glob"]["pred_raw"]) < 0.02)
noise = [NS[f"noise{i}"] for i in range(6) if f"noise{i}" in NS]
check(f"noise cells: none sent, no positive bound (largest lb {100 * max(x['lb_raw'] for x in noise):+.2f} pp)", len(noise) == 6 and all(x["lb_raw"] <= 0 for x in noise)
      and not any(n_.startswith("noise") for n_ in C))
lg = NS.get("longgain") or C.get("longgain")
ro = [it for it in doc["cells"] + doc["held"] if it["kind"] == "routed" and it["id"] == "longgain" and it["route"]["type"] == "cond"]
check(f"the long-row gain: not the global choice; kept as a ROUTED cell on the router class ({[(it['name'], round(100 * it['pred'], 1), it['share']) for it in ro]})",
      "longgain" in NS and len(ro) == 1 and ro[0]["route"]["condition"] == CL[0]["condition"] and ro[0]["route"]["before_read"] and 0.09 < ro[0]["pred"] < 0.23
      and abs(ro[0]["share"] - 0.5) < 0.02 and ro[0]["base_kind"] == "weight" and ro[0]["edit"] == cells["longgain"]["edit"] and ro[0]["lb"] > 0)
doc1 = SC.make("t000", tip, {"longgain": cells["longgain"]}, {"longgain": reads["longgain"]}, R, BUD, classes=CL)
bc = by_name(doc1)["longgain"]["by_class"]
ln = sorted(k for k in bc if k.startswith("len"))
check(f"the same split by class: on the long router class {100 * bc['cond:long']['gain']:+.1f} pp; by the tip's read length {ln[0]} {100 * bc[ln[0]]['gain']:+.1f} / "
      f"{ln[1]} {100 * bc[ln[1]]['gain']:+.1f} pp (boundary = the data's median)", bc["cond:long"]["gain"] > 0.09 and bc[ln[1]]["gain"] > bc[ln[0]]["gain"] + 0.05
      and abs(bc[ln[0]]["gain"]) < 0.06 and abs(doc1["classes"][ln[1]]["share"] - 0.5) < 0.02 and doc1["classes"][ln[1]]["quantiles"] == [0.5, 1.0])
med = float(np.median([q["tokens"] for sm in tip.values() for q in sm.values()]))
check(f"length classes come from the data: boundary {doc1['classes'][ln[1]]['gt']} = the median of the tip's read lengths ({med:.1f}); 3 bins = terciles",
      abs(doc1["classes"][ln[1]]["gt"] - med) < 0.06 and [c["quantiles"] for c in SC.length_classes(tip, 3)] == [[0.0, 1 / 3], [1 / 3, 2 / 3], [2 / 3, 1.0]])
lo_c, hi_c = bc[ln[0]], bc[ln[1]]
check(f"a length class costs the base read: all-rows added tokens of the long class {hi_c['tokens_added_all_rows']:.0f} = P(long) x the cell's whole read "
      f"(not only its excess), class tokens ratio x{hi_c['tokens_ratio']:.2f} > 2", hi_c["tokens_added_all_rows"] > 0.5 * 4000 * 1.5 * 0.85 and hi_c["tokens_ratio"] > 2.0
      and bc["cond:long"]["tokens_ratio"] < 1.7)
check("the free control of a length class is in the document (re-read with the tip itself, next seed)", set(doc1["reread"]) == set(ln)
      and all(abs(x["pred_all_rows"]) < 0.03 for x in doc1["reread"].values()))
fp = fpr = 0
for sd in range(50):
    t_, c_, r_, _ = world(1000 + sd, {f"n{i}": (0.0, 0.0, 1.0 + 0.03 * i) for i in range(8)})
    dn = SC.make("t", t_, c_, r_, R, BUD)
    fp += bool([it for it in dn["cells"] + dn["held"] if it["kind"] != "routed"] or any(x["lb_raw"] > 0 for x in dn["not_selected"]))
    fpr += bool([it for it in dn["cells"] + dn["held"] if it["kind"] == "routed"])
check(f"50 all-noise tables of 8 cells: {fp} with a positive global bound (BH at 0.10 over the 8 cells: 5 expected at most; 300 tables read 8.7 %), {fpr} with a routed "
      f"(length-class) cell (its own family of 16 options)", fp <= 9 and fpr <= 9)

# ---------------------------------------------------------------------------------------------------------------- 4. the two pairings
tc, cc, rc, _ = world(21, {"a": (0.08, 0.0, 1.3)}, coupled=True)
ti, ci, ri, _ = world(21, {"a": (0.08, 0.0, 1.3)}, coupled=False)
ec, ei = by_name(SC.make("t", tc, cc, rc, R, BUD))["a"], by_name(SC.make("t", ti, ci, ri, R, BUD))["a"]
check(f"coupled reads: key se {100 * ec['est']['key']['se']:.2f} < row se {100 * ec['est']['row']['se']:.2f} -> key in use; independent reads: row se "
      f"{100 * ei['est']['row']['se']:.2f} < key se {100 * ei['est']['key']['se']:.2f} -> row in use", ec["est"]["use"] == "key" and ec["est"]["key"]["se"] < ec["est"]["row"]["se"]
      and ei["est"]["use"] == "row" and ei["est"]["row"]["se"] < ei["est"]["key"]["se"])
seen = {}
for sd in range(21, 40):
    t_, c_, r_, _ = world(sd, {"a": (0.08, 0.0, 1.3)}, coupled=True)
    e_ = by_name(SC.make("t", t_, c_, r_, R, BUD)).get("a")
    if e_ and e_["est"]["use"] == "key":
        seen.setdefault(e_["est"]["agree"], e_)
ea, ed = seen.get(True), seen.get(False)
check(f"both estimates are in the record; when they differ by less than the larger se, pred is the row (judge-convention) value ({100 * ea['pred_raw']:+.2f} = row "
      f"{100 * ea['est']['row']['pred']:+.2f}; key {100 * ea['est']['key']['pred']:+.2f}); se and lb are those of the pairing in use, lb never above pred",
      abs(ea["pred_raw"] - ea["est"]["row"]["pred"]) < 1e-4 and abs(ea["se_raw"] - ea["est"]["key"]["pred_se"]) < 1e-4 and ea["lb"] <= ea["pred"] + 1e-9
      and abs(ea["est"]["row"]["pred"] - ea["est"]["key"]["pred"]) < max(ea["est"]["row"]["pred_se"], ea["est"]["key"]["pred_se"]) and ea["n_reads"] == 648 and ea["samples_per_row"] == [2])
check(f"when they differ by more, pred is the value of the pairing in use ({100 * ed['pred_raw']:+.2f} = key {100 * ed['est']['key']['pred']:+.2f}; row "
      f"{100 * ed['est']['row']['pred']:+.2f}) and the record says they disagree", abs(ed["pred_raw"] - ed["est"]["key"]["pred"]) < 1e-4 and ed["est"]["agree"] is False
      and abs(ed["est"]["row"]["pred"] - ed["est"]["key"]["pred"]) >= max(ed["est"]["row"]["pred_se"], ed["est"]["key"]["pred_se"]))
t9, c9, r9, _ = world(22, {"a": (0.08, 0.0, 1.3)}, coupled=True)
for r in r9["a"]:                                                 # the cell read on other sample indices than the tip has: only "row" exists
    r9["a"][r] = {s + 40: q for s, q in r9["a"][r].items()}
e9 = by_name(SC.make("t", t9, c9, r9, R, BUD))["a"]
check("no common sample index: only the row pairing exists and is used", e9["est"]["use"] == "row" and "key" not in e9["est"] and e9["est"]["agree"] is None)
x_ = {R.sig[0]: {0: dict(ok=None, tokens=None, pred=None, skip=True), 1: dict(ok=1.0, tokens=300.0, pred="A", skip=False)}}
rf_ = {R.sig[0]: {0: dict(ok=0.0, tokens=100.0, pred="B", skip=False), 1: dict(ok=1.0, tokens=200.0, pred="A", skip=False)}}
V_ = SC.views(x_, rf_)
check("a skipped read (op_skipped) counts as the reference: row view a = (0.5 + 1) / 2, key view a = (0 + 1) / 2, tokens likewise",
      abs(V_["row"][R.sig[0]]["a"] - 0.75) < 1e-12 and abs(V_["key"][R.sig[0]]["a"] - 0.5) < 1e-12 and abs(V_["key"][R.sig[0]]["t"] - 200.0) < 1e-12)

# ---------------------------------------------------------------------------------------------------------------- 5. budget
run = tempfile.mkdtemp(prefix="t_sc_run_")
check("no stated token multiple: error (no default in code)", raises(lambda: SC.read_budget(run), "no stated token budget"))
json.dump({"w": True, "g_token_multiple": 2.08, "g_token_multiple_source": "stock xhigh / medium, 5965 / 2864 on 110 monitor rows (11-11)"}, open(os.path.join(run, "cards.json"), "w"))
b1 = SC.read_budget(run)
json.dump({"g_token_multiple": {"value": 2.08, "source": "ref table", "base": 2864}}, open(os.path.join(run, "cards.json"), "w"))
b2 = SC.read_budget(run)
check(f"RUN/cards.json g_token_multiple: number + source, or an object with value / source / base ({b1['M']}, '{b1['source'][-40:]}'; base {b2['base']})",
      b1["M"] == 2.08 and "5965 / 2864" in b1["source"] and "cards.json" in b1["source"] and b1["base"] is None and b2 == dict(M=2.08, source=b2["source"], base=2864.0)
      and "ref table" in b2["source"] and raises(lambda: SC.read_budget(None, 0.5), "must be >= 1"))
docb = SC.make("t000", tip, cells, reads, R, dict(M=3.3, source="test", base=None))
check(f"a larger budget lets the 3x cell in ({[it['name'] for it in docb['cells'] if it['kind'] == 'weight']}; budget +{docb['budget']['added_tokens_allowed']:.0f} tokens per read)",
      "costly" in by_name(docb) and by_name(docb)["costly"]["frontier"]["over"] == "glob" and by_name(docb)["costly"]["plan_choice"] and not by_name(docb)["glob"]["plan_choice"])
t0 = doc["tip_stats"]["tokens"]
docs = SC.make("t000", tip, cells, reads, R, dict(M=2.0, source="test", base=t0 / 2.0))
check("the multiple refers to a stated base (the medium reference) when given: M x base - the tip's tokens = nothing left -> no cell that adds tokens",
      abs(docs["budget"]["added_tokens_allowed"]) < 1.0 and not [it for it in docs["cells"] if it["tokens_added"] > 0] and docs["budget"]["base_is"] == "stated medium reference")
refp = [dict(name="effort xhigh (stock)", gain=0.063, se=0.018, tokens_ratio=2.08, rows=110, source="reference table")]
txt = SC.table(SC.make("t000", tip, cells, reads, R, BUD, refs=refp))
check("the table prints the xhigh reference point next to the frontier (or says it is not supplied), the votes and the budget's source",
      "reference point effort xhigh (stock): gain +6.30 pp (se 1.80), tokens x2.08, 110 rows" in txt and "NOT SUPPLIED" in SC.table(doc) and "vote of m tip reads" in txt and "source: test" in txt)
cref = dict(cells, xhigh=dict(kind="reference", name="effort xhigh (measured)", family="effort"))
docr = SC.make("t000", tip, cref, dict(reads, xhigh=reads["costly"]), R, BUD)
xr = by_name(docr, "not_selected")["effort xhigh (measured)"]
check(f"a measured reference cell is reported (gain {100 * xr['gain']:+.1f} pp at x{xr['tokens_multiple']}) and never proposed; it does not count in K",
      xr["why"].startswith("reference point") and "effort xhigh (measured)" not in by_name(docr) and docr["plans"]["global"]["K"] == doc["plans"]["global"]["K"]
      and "NOT SUPPLIED" not in SC.table(docr))

# ---------------------------------------------------------------------------------------------------------------- 6. the judge's test
parts = dict(tau2=0.0, wc=0.21, wt=0.21)
nb = {b: 74 for b in R.W}
pp = [SC.pass_prob(0.04, 0.0, parts, nb, {}, K, True, True) for K in (1, 4, 61, 600)]
check(f"p_pass falls with K: {[round(x, 3) for x in pp]} (K 1, 4, 61, 600; +4 pp, 444 JUDGE rows x 4 samples)", all(a >= b for a, b in zip(pp, pp[1:])) and pp[0] > pp[-1] + 0.2)
rng = np.random.default_rng(7)


def direct(delta, K, is_w, n_sim=1500):
    """The LITE judge itself on Bernoulli rows: parent mean of 8, cell samples 1..4, quarter and step-1 futility, t test at alpha / K."""
    p0 = np.clip(rng.beta(4, 2, 444), 0.1, 0.9)
    q = np.clip(p0 + delta, 0, 1)
    win, qi = 0, np.concatenate([np.arange(19) + 74 * b for b in range(6)])
    tq = SC.t_ppf(1 - SC.ALPHA / K, 443)
    for _ in range(n_sim):
        par = (rng.random((444, 8)) < p0[:, None]).mean(1)
        x = rng.random((444, 4)) < q[:, None]
        d1 = x[:, 0] - par
        if (is_w and d1[qi].mean() <= 0) or d1.mean() <= 0:
            continue
        d = x.mean(1) - par
        tot = np.array([d[74 * b:74 * b + 74].sum() for b in range(6)])
        win += (d.mean() / (d.std(ddof=1) / math.sqrt(444)) >= tq) and bool(np.all(d.sum() - tot > 0))
    return win / n_sim, float(np.mean(q * (1 - q))), float(np.mean(p0 * (1 - p0)))


for delta, K, is_w in ((0.04, 1, True), (0.05, 61, True), (0.03, 9, False)):
    dv, wc_, wt_ = direct(delta, K, is_w)
    mine = SC.pass_prob(delta, 0.0, dict(tau2=0.0, wc=wc_, wt=wt_), nb, {}, K, is_w, True)
    check(f"p_pass agrees with a direct simulation of the judge (+{100 * delta:.0f} pp, K {K}, {'weight' if is_w else 'setting'} cell): {mine:.3f} vs {dv:.3f}", abs(mine - dv) < 0.07)
cpf = {k: dict(v, prior_family=60) for k, v in cells.items()}
dpf = SC.make("t000", tip, cpf, reads, R, BUD)
g0, g60 = by_name(doc)["glob"], by_name(dpf)["glob"]
check(f"prior_family 60 from the library entry is in the cell record; K = cells sent + 60 = {g60['K']} (was {g0['K']}); p_pass {g0['p_pass']} -> {g60['p_pass']}",
      g60["prior_family"] == 60 and g0["prior_family"] == 0 and g60["K"] == len(dpf["cells"]) + 60 and g0["K"] == len(doc["cells"]) and g60["p_pass"] < g0["p_pass"]
      and g60["judge_plan"]["rows"] == 444 and g60["judge_plan"]["samples"] == SC.JUDGE_M and g60["judge_plan"]["K"] == g60["K"])
for sd in range(31, 80):                                          # a weak cell that the frontier keeps (measured +2..4 pp, bound > 0)
    tw, cw, rw, _ = world(sd, {"weak": (0.03, 0.0, 1.1)}, m=4)
    w0 = by_name(SC.make("t", tw, cw, rw, R, BUD)).get("weak")
    if w0 and 0.02 < w0["pred"] < 0.04:
        break
dwk = SC.make("t", tw, {"weak": dict(cw["weak"], prior_family=10 ** 6)}, rw, R, BUD)
hw = [it for it in dwk["held"] if it["name"] == "weak"]
check(f"a cell that cannot pass at its K is held, not sent: pred {100 * w0['pred']:+.1f} pp, p_pass {w0['p_pass']} at K {w0['K']} -> sent; with prior_family 2000 p_pass "
      f"{hw[0]['p_pass'] if hw else None} at K {hw[0]['K'] if hw else None} -> held ({(hw[0]['why_held'] if hw else '')[:40]}...)", w0["p_pass"] >= SC.ALPHA and len(hw) == 1
      and hw[0]["p_pass"] < SC.ALPHA and "cannot pass" in hw[0]["why_held"] and "weak" not in by_name(dwk) and hw[0]["K"] > 10 ** 6 and "held weak" in SC.table(dwk))
caps = SC.make("t000", tip, cpf, reads, R, BUD, classes=CL, caps={"g_cells": 1, "g_settings": 0})
check("K counts what author G will propose under its caps (g_cells 1) plus the routed cells", by_name(caps)["glob"]["K"] == 1 + 60 + sum(1 for it in caps["cells"] if it["kind"] == "routed"))

# ---------------------------------------------------------------------------------------------------------------- 7. engine result files
d7 = tempfile.mkdtemp(prefix="t_sc_res_")
files, spec7 = [], []
for name in ("glob", "costly", "noise0"):
    f = os.path.join(d7, f"wsig_{name}_0.jsonl")
    with open(f, "w") as fh:
        for r, sm in reads[name].items():
            for s, q in sm.items():
                fh.write(json.dumps(dict(key=f"x|{name}|{r}|{s}", mode="read", row=r, sample=s, ok=bool(q["ok"]), tokens=q["tokens"], pred=q["pred"], finish="stop",
                                         edit=cells[name]["edit"], edit_sha=SC.esha(cells[name]["edit"]), ops=[], settings={}, prefix=None)) + "\n")
        fh.write(json.dumps(dict(key="e1", row=R.sig[0], sample=5, error="boom", edit=cells[name]["edit"])) + "\n")
        fh.write(json.dumps(dict(key="rt1|c|r", mode="route", row=R.sig[0], yes=True)) + "\n")
    files.append(f)
    spec7.append(dict(cells[name], match={"batch": f"wsig_{name}_*"} if name != "costly" else {"edit_sha": SC.esha(cells[name]["edit"])}))
ft = os.path.join(d7, "g007_parent_0.jsonl")
with open(ft, "w") as fh:
    for r, sm in tip.items():
        for s, q in sm.items():
            fh.write(json.dumps(dict(key=f"p|{r}|{s}", mode="read", row=r, sample=s, ok=bool(q["ok"]), tokens=q["tokens"], pred=q["pred"], finish="stop")) + "\n")
c7, r7, t7, info = SC.ingest_results(files + [ft], {"cells": spec7, "tip": {"match": {"batch": "g007_parent_*", "no_edit": True}}}, R)
d7doc = SC.make("t000", t7, c7, r7, R, BUD)
dref = SC.make("t000", tip, {k: cells[k] for k in ("glob", "costly", "noise0")}, {k: reads[k] for k in ("glob", "costly", "noise0")}, R, BUD)
strip = lambda d: json.dumps({k: v for k, v in d.items() if k != "written"}, sort_keys=True)   # noqa: E731
check(f"result files + mapping (by batch name, by edit_sha; the tip's lines by batch) give the same document as the tables ({info})",
      strip(d7doc) == strip(dref) and info["glob"] == 648 and info["tip"] == 324 * 8 and info["no sample (error / abort)"] == 3 and info["not a read"] == 3)
check("an unknown match key is an error", raises(lambda: SC.ingest_results(files, {"cells": [dict(name="x", kind="weight", match={"bach": "x"})]}, R), "not known"))

# ---------------------------------------------------------------------------------------------------------------- 7b. per-read tables (tsv)
RULES = dict(control_cell="W-null", proposable_families=["XS"], prior_family=50)
n1, n2, n3, n4 = (SC.cell_from_name(n_, None, RULES) for n_ in ("XS_bias_L24-63noL63_+1", "XB_bias_all_+2", "W-null", "RAND1_bias_all_+1"))
check("cell specs from the tables' cell names: family = first word (the menu), labels alpha / band from the name, a family that is not proposable is measurement only, "
      "the control cell is kind control", (n1["family"], n1["band"], n1["alpha"], n1.get("propose"), n1["prior_family"], n1["kind"]) == ("XS", [24, 62], 1.0, None, 50, "weight")
      and (n2["family"], n2["band"], n2["alpha"], n2["propose"]) == ("XB", [0, 63], 2.0, False) and n3["kind"] == "control" and n3["prior_family"] == 0 and n4["propose"] is False
      and SC.cell_from_name("XS_bias_noL63_+0.5")["band"] == [0, 62] and SC.cell_from_name("XS_bias_L63only_+1")["band"] == [63, 63]
      and SC.cell_from_name("unit_giveup")["family"] == "unit" and "band" not in SC.cell_from_name("unit_giveup"))
tW, cW, rW, lW = world(61, {"XS_bias_all_+1": (0.11, 0.0, 1.4), "XS_bias_mid_+1": (0.10, 0.0, 1.2), "XB_bias_all_+1": (0.12, 0.0, 1.5), "W-null": (0.03, 0.0, 1.02), "RAND1_bias_all_+1": (0.03, 0.0, 1.0)})
d7b = tempfile.mkdtemp(prefix="t_sc_tsv_")
HEAD = "batch cell row sample ok tokens finish loop prompt_tokens error base_key base_ok base_tokens base_finish".split()
tsvs = []
for name, tb in rW.items():
    f = os.path.join(d7b, f"{name.split('_')[0].lower()}_{len(tsvs)}.tsv")
    with open(f, "w") as fh:
        fh.write("\t".join(HEAD) + "\n")
        for i, (r, sm) in enumerate(tb.items()):
            for s_, q in sm.items():
                cut = name.startswith("XB") and i % 40 == 0
                fh.write("\t".join(str(v) for v in (os.path.basename(f)[:-4], name, r, s_, int(q["ok"]) if not cut else 0, round(q["tokens"]) if not cut else 32768, "length" if cut else "stop",
                                                    0, 4000, "", f"x|k|{r}|{s_}", int(tW[r][s_]["ok"]), round(tW[r][s_]["tokens"]), "stop")) + "\n")
        fh.write("\t".join([os.path.basename(f)[:-4], name, R.sig[0], "7", "", "", "", "", "", "CUDA out of memory", "", "", "", ""]) + "\n")
    tsvs.append(f)
BC = {"prior_family": 50, "cells": {"xs_0": {"cell": "XS_bias_all_+1", "edit": "/data/fd/mingdazhang/selfedit/v4/cells/xs/edits/XS_bias_all_+1.json", "family": "XS"},
                                    "xb_2": {"cell": "XB_bias_all_+1", "note": "direction built partly on JUDGE rows"}}}
tWr = {r: {s_: dict(q, tokens=float(round(q["tokens"]))) for s_, q in sm.items()} for r, sm in tW.items()}
tWn = {r: {s_: {k: v for k, v in q.items() if k != "pred"} for s_, q in sm.items()} for r, sm in tWr.items()}      # a compact tip table: no pred
c7b, r7b, base7, info7 = SC.ingest_tsv(tsvs, R, BC, RULES)
check(f"per-read tables: {info7}; error lines are no reads; the base columns equal the tip table ({SC.base_check(base7, tWr)} differ)",
      info7["XS_bias_all_+1"] == 648 and info7["no sample (error / abort)"] == 5 and SC.base_check(base7, tWr).startswith("0 of") and set(c7b) == set(rW)
      and c7b["XS_bias_all_+1"]["edit"] == {"op": "surgeon", "file": BC["cells"]["xs_0"]["edit"]} and c7b["XB_bias_all_+1"]["propose"] is False and c7b["W-null"]["kind"] == "control")
with open(tsvs[0], "a") as fh:
    fh.write("\t".join(["xs_0", "XS_bias_all_+1", jr, "0", "1", "900", "stop", "0", "4000", "", "", "1", "900", "stop"]) + "\n")
check("a per-read table with a judge row raises", raises(lambda: SC.ingest_tsv(tsvs, R, BC, RULES), "has role 'judge'"))
keep = open(tsvs[0]).readlines()[:-1]
open(tsvs[0], "w").write("".join(keep))
d7c = SC.make("t000", tWn, c7b, r7b, R, BUD, control="W-null")
C7, N7 = by_name(d7c), by_name(d7c, "not_selected")
H7 = by_name(d7c, "held")
xs = C7["XS_bias_all_+1"]
check(f"the concurrent control: a weight cell's primary gain is against W-null ({100 * xs['gain']:+.1f} pp; planted +8 over the control), its gain against the tip's cached "
      f"reads is kept as secondary ({100 * xs['est']['vs_tip']['row']['d']:+.1f} pp; planted +11)", xs["est"]["against"] == "control cell W-null" and 0.04 < xs["gain"] < 0.12
      and xs["est"]["vs_tip"]["row"]["d"] > xs["gain"] + 0.01 and d7c["control"]["state"] == "in use" and "vs the tip's cached reads" in SC.table(d7c))
check(f"the control and the measurement-only cells are reported and never proposed: {[(k, v['why'][:34]) for k, v in N7.items() if v.get('propose') is False]}; K counts the "
      f"proposable cells only ({d7c['plans']['global']['K']})", N7["W-null"]["why"].startswith("concurrent control") and N7["W-null"]["est"]["against"] == "the tip's cached reads"
      and N7["XB_bias_all_+1"]["why"].startswith("measurement only") and "JUDGE rows" in N7["XB_bias_all_+1"]["why"] and N7["RAND1_bias_all_+1"]["propose"] is False
      and d7c["plans"]["global"]["K"] == 2 and not any(k.startswith(("XB", "RAND", "W-null")) for k in list(C7) + list(H7)) and "by_class" in N7["XB_bias_all_+1"])
noed = {"XS_bias_all_+1": {k: v for k, v in c7b["XS_bias_all_+1"].items() if k != "edit"}, "W-null": c7b["W-null"]}
H7m = by_name(SC.make("t000", tWn, noed, r7b, R, BUD, control="W-null"), "held")
check(f"a proposable cell without an edit path cannot be sent: held ({[(k, v['why_held'][:28]) for k, v in H7m.items()]}); with its path it is sent with prior_family 50",
      "XS_bias_all_+1" in H7m and "no edit" in H7m["XS_bias_all_+1"]["why_held"] and xs["prior_family"] == 50 and xs["edit"]["file"].endswith("XS_bias_all_+1.json")
      and xs["K"] >= 51)
ns7 = N7["XB_bias_all_+1"]["est"]["not_stopped"]
check(f"reads that hit the token budget are counted: XB not stopped {ns7['cell']} (control {ns7['ref']}), tokens x{N7['XB_bias_all_+1']['tokens_multiple']} over all reads "
      f"(the judge's ratio), x{ns7['tokens_multiple_stopped']} among the reads that stopped", 0.02 < ns7["cell"] < 0.04 and ns7["ref"] == 0.0
      and ns7["tokens_multiple_stopped"] < N7["XB_bias_all_+1"]["tokens_multiple"] and "not stopped" in SC.table(d7c))
check(f"a tip table without pred: the vote gains use right / wrong classes and the document says so ({d7c['votes_from'][:40]}...); with preds: the tip's preds",
      "lower bound" in d7c["votes_from"] and SC.make("t000", tWr, c7b, r7b, R, BUD)["votes_from"] == "the tip's preds" and abs(d7c["votes"]["2"]["gain"]) < 1e-9)
d7d = SC.make("t000", tWn, c7b, {k: v for k, v in r7b.items() if k != "W-null"}, R, BUD, control="W-null")
check("before the control has reads the cells are against the tip's cached reads and the document says so", by_name(d7d)["XS_bias_all_+1"]["est"]["against"] == "the tip's cached reads"
      and "vs_tip" not in by_name(d7d)["XS_bias_all_+1"]["est"] and "no reads yet" in d7d["control"]["state"] and by_name(d7d)["XS_bias_all_+1"]["gain"] > xs["gain"]
      and "not in the input yet" in SC.make("t000", tWn, {"XS_bias_all_+1": c7b["XS_bias_all_+1"]}, {"XS_bias_all_+1": r7b["XS_bias_all_+1"]}, R, BUD, control="W-null")["control"]["state"])
bc7 = xs["by_class"]
check(f"the read-length classes add up to the cell's gain under its pairing in use ({xs['est']['use']}): sum share x gain = {100 * sum(c['share'] * c['gain'] for c in bc7.values()):+.3f} "
      f"= {100 * xs['gain']:+.3f} pp", abs(sum(c["share"] * c["gain"] for k, c in bc7.items() if k.startswith("len")) - xs["gain"]) < 2e-4)
import hashlib  # noqa: E402
ents = [dict(condition=f"Does the question ask about thing {i}?", procedure=f"Procedure {i}: do the thing {i}.\n\nCheck it twice.", from_cand=f"g00{i}c0{i}") for i in range(4)]
ents = [dict(condition="always", procedure=None, settings={"min_image_tokens": 4096})] + ents
fire = {r: [i for i in range(4) if (hash_ := int(hashlib.sha256(f"{i}|{r}".encode()).hexdigest(), 16)) % (3 + i) == 0] for r in tW}
trs = {r: dict(role=R.meta[r]["role"], bench=R.bench(r), prefix_sha=hashlib.sha256("\n\n".join(ents[1 + i]["procedure"] for i in f).encode()).hexdigest()[:12]) for r, f in fire.items()}
trp_ = {r: dict(role=R.meta[r]["role"], bench=R.bench(r), prefix="\n\n".join(ents[1 + i]["procedure"] for i in f) or None) for r, f in fire.items()}
cl_s, cl_p = SC.router_classes(trs, ents), SC.router_classes(trp_, ents)
check(f"router classes of the tip's conditions from prefix hashes (a compact tip table) equal those from the prefix text: {[len(c['rows']) for c in cl_s]} rows; "
      "no class for the always entry", len(cl_s) == 4 and all(c["rows"] == {r for r, f in fire.items() if i in f} and c["undecoded"] == 0 for i, c in enumerate(cl_s))
      and [c["rows"] for c in cl_s] == [c["rows"] for c in cl_p] and [c["condition"] for c in cl_s] == [e["condition"] for e in ents[1:]] and cl_s[0]["name"].startswith("cond:"))
trs[R.sig[0]]["prefix_sha"] = "0" * 12
check("a prefix hash that no set of the tip's procedures gives is counted, its row is in no class", SC.router_classes(trs, ents)[0]["undecoded"] == 1
      and all(R.sig[0] not in c["rows"] for c in SC.router_classes(trs, ents)))
wdir = tempfile.mkdtemp(prefix="t_sc_watch_")
import shutil  # noqa: E402
shutil.copy(tsvs[0], wdir)
json.dump({r: dict(role=R.meta[r]["role"], bench=R.bench(r), samples={str(s_): dict(ok=bool(q["ok"]), tokens=q["tokens"], finish="stop") for s_, q in sm.items()}) for r, sm in tWr.items()},
          open(os.path.join(wdir, "tip.json"), "w"))
json.dump(BC, open(os.path.join(wdir, "batch_cells.json"), "w"))
wargs = ["watch", "--once", "--dir", wdir, "--run", os.path.join(wdir, "dryrun"), "--tip", "t000", "--rows", ROWS, "--traces", os.path.join(wdir, "tip.json"),
         "--batch-cells", os.path.join(wdir, "batch_cells.json"), "--proposable", "XS", "--control", "W-null", "--budget", "2.08", "--budget-source", "test"]
_out = sys.stdout
sys.stdout = open(os.devnull, "w")
try:
    SC.main(wargs)
    shutil.copy(tsvs[3], wdir)
    SC.main(wargs)
finally:
    sys.stdout = _out
md = open(os.path.join(wdir, "cells_table.md")).read()
check("watch: each run appends one dated table (CST) to cells_table.md and names the new tables; nothing else is written (dry)", md.count("\n## ") == 2 and md.count(" CST: 1 tables; new: xs_0.tsv") == 1
      and "CST: 2 tables; new: w-null_3.tsv" in md and "XS_bias_all_+1" in md and "W-null" in md.split("\n## ")[2] and sorted(os.listdir(wdir)) == ["batch_cells.json", "cells_table.md", "cells_table.md.state", "tip.json", "w-null_3.tsv", "xs_0.tsv"])
sys.stdout = open(os.devnull, "w")
try:
    SC.main(wargs)                                                # nothing new: a restarted watch appends nothing
finally:
    sys.stdout = _out
check("watch: a restart with the same tables appends nothing (the last table set is kept in cells_table.md.state)", open(os.path.join(wdir, "cells_table.md")).read() == md)

# ---------------------------------------------------------------------------------------------------------------- 8. round trip: author G
import gen  # noqa: E402
import t_versions as T  # noqa: E402

gen.time.sleep = lambda s: None
LOG = []
gen.log = lambda m: LOG.append(m)
root = tempfile.mkdtemp(prefix="t_sc_g_")
Rg = T.make(root, cls=gen.Run, slots=4, extra=("--no-versions", "--no-ecomp", "--no-w", "--e-cards", "8"))
json.dump({"w": True, "e_cards": 6, "g_token_multiple": 2.0, "g_token_multiple_source": "test"}, open(os.path.join(Rg.rd, "cards.json"), "w"))
Rg.g = 1
Rg.stamp_cards(Rg.state["gens"].setdefault("1", dict(t0=0.0, done=[], times={})))
EFF8 = {"glob": (0.08, 0.0, 1.3), "temp": (0.07, 0.0, 1.0, "setting", "T"), "longgain": (0.0, 0.16, 1.5), "noise0": (0.0, 0.0, 1.0)}
t8, c8, r8, l8 = world(41, EFF8)
CL8 = [dict(name="cond:long", type="cond", condition="Does the question need a long derivation?", rows={r for r in t8 if l8[r]})]
doc8 = SC.make("t000", t8, c8, r8, R, SC.read_budget(Rg.rd), classes=CL8)
path8, sha8 = SC.write_cells(os.path.join(Rg.rd, "signal", "t000"), doc8)
cs, info = Rg.g_read_cells()
sent8 = [(it["kind"], it["name"]) for it in doc8["cells"]]
check(f"cells.json goes through author_g.g_read_cells: {[(c['rank'], c['kind'], c['name']) for c in cs]}; state '{info['state']}', sha {info['sha']}",
      info["state"] == "read" and info["sha"] == sha8 and [(c["kind"], c["name"]) for c in cs] == [x for x in sent8 if x[0] in ("weight", "setting")]
      and {c["kind"] for c in cs} == {"weight", "setting"} and all(isinstance(c["pred"], float) and isinstance(c["se"], float) and c["lb"] > 0 for c in cs)
      and info["scale"].startswith("accuracy per row, show+monitor rows"))
sk = {s["name"]: s["why"] for s in info["skipped"]}
check(f"today's reader logs and skips the routed cell; the patched one waits for cards.json g_routed ({sk})",
      all("routed" in v and ("not handled" in v or "wait for RUN/cards.json g_routed" in v) for v in sk.values()) and len(sk) == sum(1 for k, _ in sent8 if k == "routed") >= 1)
props = []
sl = Rg.w_budget_after_g(props, {"B": 2, "P0": 4, "BW": 3})
gm = [p["cand"]["g"] for p in props if p["author"] == "G"]
check(f"author G proposes them: {[(p['cand']['cell'], 'edit' in p['cand'], p['cand']['settings']) for p in props]}; g meta carries pred / se / lb / tokens ratio / family / cells_sha",
      len(props) == len(cs) >= 2 and all(p["cand"].get("edit") == c8[p["cand"]["cell"]]["edit"] and p["cand"]["cell"] in ("glob", "longgain") for p in props[:-1])
      and props[-1]["cand"]["settings"] == {"temperature": 0.8} and "edit" not in props[-1]["cand"] and [p["slot"] for p in props] == list(range(len(props)))
      and all(g["cells_sha"] == sha8 and g["family"] in ("F", "T") and isinstance(g["pred_tokens_ratio"], float) and g["lb"] > 0 for g in gm))
w8 = [it for it in doc8["cells"] if it["kind"] == "weight"]
check(f"cells of one family are one menu: one final vertex ({[it['name'] for it in w8 if it['plan_choice']]}; earlier vertices stay as cheaper alternatives: "
      f"{[it['name'] for it in w8 if not it['plan_choice']]}); another family (the setting) is a coordinate of its own and is sent beside it",
      sum(1 for it in w8 if it["plan_choice"]) == 1 and ("setting", "temp") in sent8 and by_name(doc8)["temp"]["plan_choice"] and doc8["plans"]["global"]["menus"] == 2)
raw8 = open(path8).read()
tab8 = open(os.path.join(Rg.rd, "signal", "t000", "cells_table.txt")).read()
ids = set(R.meta)


def has_ids(o):
    if isinstance(o, dict):
        return any(has_ids(v) or (isinstance(k, str) and k in ids) for k, v in o.items())
    if isinstance(o, list):
        return any(has_ids(v) for v in o)
    return isinstance(o, str) and o in ids


check("no row id in cells.json or the table; the archived copy cells_<sha>.json is byte-identical; strict JSON (no NaN / Infinity)",
      not has_ids(json.loads(raw8)) and not any(i in tab8 for i in list(ids)[:800]) and open(os.path.join(Rg.rd, "signal", "t000", f"cells_{sha8}.json")).read() == raw8
      and "NaN" not in raw8 and "Infinity" not in raw8)
check("the table lists every measured cell with both pairings, its class split, the plans and the calibration line",
      all(x in tab8 for x in ("row d (se)", "key d (se)", "p_pass", "plan global:", "on cond:long", "calibration:", "noise0", "bound")))

# ---------------------------------------------------------------------------------------------------------------- 9. calibration
sig = os.path.join(Rg.rd, "signal")
c0 = SC.calibrate(sig)
check(f"no realised pair: predictions pass through ({c0['state']}); shrink 1, marked uncalibrated", c0["pairs"] == 0 and c0["quality"] is None and SC.shrink_of(c0, "weight")[0] == 1.0
      and "uncalibrated" in SC.shrink_of(c0, "weight")[2] and doc8["cells"][0]["shrink"] == 1.0 and os.path.exists(os.path.join(Rg.rd, "signal_calibration.json"))
      and not os.path.exists(os.path.join(sig, "calibration.json")))
rng = np.random.default_rng(9)
SLOPE = 0.6


def realised(tid, gen_, n, shrink, kind="weight", slope=SLOPE):
    """n judged G cells of one generation: cells file written with `shrink` applied, realised d = slope x raw + judge noise."""
    d = os.path.join(sig, tid)
    os.makedirs(d, exist_ok=True)
    its = []
    for i in range(n):
        raw, se_raw = float(rng.uniform(0.01, 0.09)), 0.012
        its.append(dict(kind=kind, name=f"{tid}-c{i}", pred=shrink * raw, se=shrink * se_raw, pred_raw=raw, se_raw=se_raw, shrink=shrink,
                        **({"edit": {"op": "surgeon", "file": f"/x/{tid}{i}.json"}} if kind == "weight" else {"settings": {"temperature": 0.7}})))
    _, sha = SC.write_cells(d, dict(doc8, tip=tid, cells=its, held=[], not_selected=[]))
    with open(os.path.join(d, "realised.jsonl"), "a") as fh:
        for i, it in enumerate(its):
            se = 0.012
            fh.write(json.dumps(dict(gen=gen_, cid=f"g{gen_:03d}c{i:02d}", tip=tid, rank=i, kind=kind, name=it["name"], family="F", cells_sha=sha, scale="s", pred=it["pred"],
                                     pred_se=it["se"], d=slope * it["pred_raw"] + float(rng.normal(0, se)), se=se, se_used=se, p=0.01, n_rows=444, samples_per_row=4,
                                     accepted=bool(i % 2), merged=False)) + "\n")
        fh.write(json.dumps(dict(gen=gen_, cid=f"g{gen_:03d}c98", tip=tid, kind=kind, name="dup of the tip", pred=0.02, pred_se=0.01, d=None, se=None, duplicate_of="tip")) + "\n")
        fh.write(json.dumps(dict(gen=gen_, cid=f"g{gen_:03d}c99", tip=tid, kind=kind, name=its[0]["name"], cells_sha=sha, pred=its[0]["pred"], pred_se=0.01, d=0.5, se=0.01,
                                 n_rows=444, duplicate_of=f"g{gen_:03d}c00")) + "\n")


realised("tA", 10, 12, 1.0)                                       # generation 10: written uncalibrated (raw predictions overclaim: slope 0.6)
ca = SC.calibrate(sig)
bw = ca["slopes"]["weight"]
check(f"calibration recovers the planted slope {SLOPE}: weight b {bw['b']:.3f} (sd {bw['sd']:.3f}, n {bw['n']}; least squares {bw['ls']:.3f}; the prior toward 0 pulls a little)",
      bw["n"] == 12 and abs(bw["b"] - SLOPE) < 3 * bw["sd"] + 0.02 and 0 < bw["b"] <= bw["ls"] + 1e-9 and bw["sd"] < 0.15 and bw["p_pos"] > 0.99 and ca["pairs"] == 12)
check("lines without measured numbers (duplicate of the tip) and a second line of the same measurement are not pairs; realised.jsonl is read per tip directory",
      len(SC.load_pairs(sig)) == 12 and not os.path.exists(os.path.join(sig, "realised.jsonl")))
doc9 = SC.make("t000", t8, c8, r8, R, SC.read_budget(Rg.rd), classes=CL8, calib=ca)
g9, g8_ = by_name(doc9)["glob"], by_name(doc8)["glob"]
check(f"new predictions are shrunk by it: pred {100 * g8_['pred']:+.2f} -> {100 * g9['pred']:+.2f} pp (shrink {g9['shrink']}, {g9['shrink_from']}); pred_raw unchanged; "
      f"the setting cell borrows the pooled slope", abs(g9["pred"] - bw["b"] * g9["pred_raw"]) < 1e-4 and g9["pred_raw"] == g8_["pred_raw"] and g9["shrink"] == round(bw["b"], 4)
      and g9["se"] > bw["b"] * g9["se_raw"] and g9["lb"] < g9["pred"] and "pooled" in by_name(doc9)["temp"]["shrink_from"] and "weight, 12 pairs" in g9["shrink_from"])
q10 = ca["by_gen"][-1]
realised("tB", 11, 12, bw["b"])                                   # generations 11, 12: written with the fitted slope
realised("tC", 12, 12, bw["b"])
cb = SC.calibrate(sig)
Q = [g["Q"] for g in cb["by_gen"]]
check(f"signal quality Q rises once predictions are written with the fitted slope: per generation {[g['Q_gen'] for g in cb['by_gen']]}, cumulative {Q}; "
      f"coverage {cb['by_gen'][-1]['coverage']}, rank r {cb['by_gen'][-1]['spearman']}", Q[0] == q10["Q"] and Q[0] < Q[1] < Q[2] and cb["by_gen"][1]["Q_gen"] > cb["by_gen"][0]["Q_gen"]
      and cb["quality"] is not None and cb["by_gen"][-1]["spearman"] > 0.5 and cb["by_gen"][-1]["n_cum"] == 36 and 0.75 <= cb["by_gen"][-1]["coverage"] <= 1.0)
check("the raw prediction of a pair comes from the archived cells_<sha>.json (generation 11 was written shrunk: pred_raw = pred / shrink)",
      all(abs(x["pred_raw"] - x["pred"] / bw["b"]) < 1e-6 for x in SC.load_pairs(sig) if x["gen"] == 11) and abs(cb["slopes"]["weight"]["b"] - SLOPE) < 0.1)
qt = open(os.path.join(Rg.rd, "signal_quality.txt")).read()
check(f"RUN/signal_calibration.json and RUN/signal_quality.txt are written for the report, beside RUN/signal (which keeps one directory per tip and nothing else: "
      f"{sorted(os.listdir(sig))}) ({SC.quality_line(sig)[:50]}...)", "signal quality" in qt and "slope weight" in qt and len(qt.splitlines()) >= 7
      and json.load(open(os.path.join(Rg.rd, "signal_calibration.json")))["quality"] == cb["quality"] and SC.quality_line(sig).startswith("signal quality Q")
      and all(os.path.isdir(os.path.join(sig, x)) for x in os.listdir(sig)))
realised("tD", 13, 6, 1.0, kind="setting", slope=-0.3)
cd = SC.calibrate(sig, write=False)
check(f"per kind: the setting slope is its own ({cd['slopes']['setting']['b']:+.2f}, P(b > 0) {cd['slopes']['setting']['p_pos']:.3f}); a slope below 0 shrinks to 0, never negative",
      cd["slopes"]["setting"]["b"] < 0 and cd["slopes"]["setting"]["p_pos"] < 0.1 and SC.shrink_of(cd, "setting")[0] == 0.0 and abs(cd["slopes"]["weight"]["b"] - cb["slopes"]["weight"]["b"]) < 1e-9)
one = SC.fit_slope([dict(pred_raw=0.05, d=-0.03, se=0.02, se_raw=0.01)])
check("one realised pair gives no statement about the sign of a slope (p_pos None)", one["p_pos"] is None and one["n"] == 1 and one["b"] < 0)

# ---------------------------------------------------------------------------------------------------------------- 10. planner
prims = []
for l in range(64):
    prims += [{"op": "set", "param": f"layers.{l}.mlp.gate_proj", "sel": {"neurons": [7]}, "value": {"npz": "/v/a.npz", "key": f"k_{l}"}},
              {"op": "scale", "param": f"layers.{l}.mlp.gate_proj", "sel": {"neurons": [7]}, "alpha": 8.0},
              {"op": "set", "param": f"layers.{l}.mlp.up_proj", "sel": {"neurons": [7]}, "value": {"npz": "/v/a.npz", "key": f"k_{l}"}},
              {"op": "set", "param": f"layers.{l}.mlp.down_proj", "sel": {"neurons": [7]}, "value": {"npz": "/v/a.npz", "key": f"down_XB_{l}"}},
              {"op": "scale", "param": f"layers.{l}.mlp.down_proj", "sel": {"neurons": [7]}, "alpha": 1.0}]
e1 = SP.restrict_edit(prims, [21, 42], 2.0)
lay = lambda E: sorted({int(p["param"].split(".")[1]) for p in E})   # noqa: E731
check("restrict_edit: the band's layers only, the down_proj scale = alpha, the gate scale (8) and the arrays untouched, the base list unchanged",
      lay(e1) == list(range(21, 43)) and len(e1) == 22 * 5 and {p["alpha"] for p in e1 if p["op"] == "scale" and p["param"].endswith("down_proj")} == {2.0}
      and {p["alpha"] for p in e1 if p["op"] == "scale" and p["param"].endswith("gate_proj")} == {8.0} and prims[4]["alpha"] == 1.0
      and [p for p in e1 if p["op"] == "set"] == [p for p in prims if p["op"] == "set" and 21 <= int(p["param"].split(".")[1]) <= 42])
check("restrict_edit: base_alpha rescales (a base written at dose 2, alpha 1 -> 0.5); alpha = base and the full band = the base list itself",
      {p["alpha"] for p in SP.restrict_edit(prims, [0, 63], 1.0, base_alpha=2.0) if p["param"].endswith("down_proj") and p["op"] == "scale"} == {0.5}
      and SP.restrict_edit(prims, [0, 63], 1.0) == prims)
check("restrict_edit refuses a list without a dose primitive, a primitive without a layer, an empty band",
      raises(lambda: SP.restrict_edit([p for p in prims if p["op"] == "set"], [0, 5], 2.0), "no dose primitive")
      and raises(lambda: SP.restrict_edit([{"op": "scale", "param": "vision.merger", "alpha": 1.1}], [0, 5], 2.0), "without a layer")
      and raises(lambda: SP.restrict_edit(prims, [70, 80], 1.0), "no primitive in layers"))
check("the coarse bands are the thirds of the declared layer range (0-63: early 0-20, mid 21-42, late 43-63)", SP.thirds([0, 63]) == [[0, 20], [21, 42], [43, 63]]
      and SP.thirds([21, 63]) == [[21, 34], [35, 49], [50, 63]])
tp, _, _, lp = world(51, {})
order = SP.race_order(tp, R, seed=0)
pre = lambda n: collections.Counter(R.bench(r) for r in order[:n])   # noqa: E731
lng = lambda n: sum(1 for r in order[:n] if lp[r]) / n               # noqa: E731
check(f"race order: all 324 signal rows once; every prefix balanced by benchmark (12: {sorted(pre(12).values())}, 84: {sorted(pre(84).values())}) and by read length "
      f"(long share {lng(12):.2f} / {lng(84):.2f} / {lng(324):.2f}); fixed by the seed", sorted(order) == sorted(tp) and set(pre(12).values()) == {2} and set(pre(84).values()) == {14}
      and abs(lng(12) - 0.5) <= 0.17 and abs(lng(84) - 0.5) < 0.08 and order == SP.race_order(tp, R, seed=0) and order != SP.race_order(tp, R, seed=1))
dpl = tempfile.mkdtemp(prefix="t_sc_plan_")
json.dump(prims, open(os.path.join(dpl, "XB_all_c1.json"), "w"))
LIB = [{"family": "XB", "kind": "weight", "prior_family": 60, "dose": {"range": [0.5, 3.0]},
        "restrict": {"base_edit": os.path.join(dpl, "XB_all_c1.json"), "base_alpha": 1.0, "layers": [0, 63], "server_dir": "/data/fd/mingdazhang/selfedit/v4/cells/xb_sig"}},
       {"family": "temperature", "kind": "setting", "knob": "temperature", "dose": {"range": [0.5, 1.5]}, "origin": 1.0},
       {"family": "units", "kind": "weight", "prior_family": 0, "cells": [{"name": "give-up state unit", "edit": {"op": "surgeon", "file": "/data/fd/mingdazhang/selfedit/v4/cells/u/giveup.json"}},
                                                       {"name": "opening position unit", "edit": {"op": "surgeon", "file": "/data/fd/mingdazhang/selfedit/v4/cells/u/open.json"}}]}]
rngp = np.random.default_rng(5)
pr_ = {r: float(np.clip(rngp.beta(4, 2), 0.05, 0.85)) for r in tp}
bs_ = {r: 4000.0 if lp[r] else 800.0 for r in tp}
Up = {r: rngp.random(64) for r in tp}
tipp = {r: {s: dict(ok=float(Up[r][s] < pr_[r]), tokens=bs_[r] * math.exp(0.3 * rngp.standard_normal()), pred="A" if Up[r][s] < pr_[r] else "B", skip=False) for s in range(8)} for r in tp}


def truth(spec):
    """Planted dose-response: the gain lives in the middle band and saturates with the dose; tokens grow with dose x width."""
    if spec["kind"] == "setting":
        return 0.0, 1.0
    if spec["family"] == "units":
        return (0.05, 1.15) if "give-up" in spec["name"] else (0.0, 1.0)
    a, (lo, hi) = spec["alpha"], spec["band"]
    wmid, wall = sum(1 for l in range(lo, hi + 1) if 21 <= l <= 42) / 22.0, (hi - lo + 1) / 64.0
    return 0.10 * wmid * (1 - math.exp(-a / 0.8)) - 0.036 * max(a - 1.5, 0) * wall, 1.0 + 0.45 * a * (0.5 * wmid + 0.5 * wall)


def engine(batch):
    out = []
    for q in batch["requests"]:
        g, tm = truth(q["spec"])
        for r in q["rows"]:
            for s in q["samples"]:
                okv = Up[r][s] < pr_[r] + g
                out.append(dict(key=batch["key"].format(tip=batch["tip"], sha=q["sha"], row=r, s=s), row=r, sample=s, ok=bool(okv), finish="stop",
                                tokens=tm * bs_[r] * math.exp(0.3 * rngp.standard_normal()), pred="A" if okv else "B"))
    return out


BP = dict(M=2.08, source="test", base=None)
P = SP.Plan.create(os.path.join(dpl, "plan"), "t000", tipp, R, LIB, 6000)
cellsP = P.st["cells"]
check(f"coarse grid: 3 doses x (3 thirds + the full band) for the restricted direction, 2 knob values (the origin is not a cell), 2 explicit cells = {len(cellsP)}; "
      f"K = ceil(log2) = {P.st['K']} rounds; levels {P.st['levels']}", len(cellsP) == 16 and P.st["K"] == 4 and P.st["levels"][-2:] == [[324, 1], [324, 2]]
      and sorted({c["dose"] for c in cellsP.values() if c["spec"]["family"] == "XB"}) == [0.5, 1.75, 3.0] and "temperature|1.0" not in cellsP
      and cellsP["units|give-up state unit"]["spec"]["edit"]["file"].endswith("giveup.json") and cellsP["units|give-up state unit"]["arm"] is None
      and all(c["spec"]["prior_family"] == (60 if c["spec"]["family"] == "XB" else 0) for c in cellsP.values()))
ef = cellsP["XB|a1.75|L21-42"]["spec"]
loc = json.load(open(ef["edit_local"]))
check(f"a derived edit is a file of its own, content hash in its name, server path in the request ({os.path.basename(ef['edit']['file'])})",
      ef["edit"]["file"].startswith("/data/fd/mingdazhang/selfedit/v4/cells/xb_sig/XB_a1.75_L21-42_") and loc == SP.restrict_edit(prims, [21, 42], 1.75)
      and os.path.basename(ef["edit_local"]) == os.path.basename(ef["edit"]["file"]) and open(ef["edit_local"]).read() == json.dumps(loc))
b0 = P.next_batch(tipp, R, BP)
allrows = [r for q in b0["requests"] for r in q["rows"]]
check(f"round 0: every coarse cell on the same prefix of the race order, one sample, within the round's budget ({b0['reads']} <= {6000 // 4}); only show / monitor rows",
      len(b0["requests"]) == 16 and all(q["rows"] == order_ for q in b0["requests"] for order_ in [P.st["order"][:b0["level"][0]]]) and b0["reads"] <= 1500 and b0["level"][1] == 1
      and all(R.meta[r]["role"] in SC.SIGNAL_ROLES for r in allrows) and all(q["samples"] == [0] for q in b0["requests"])
      and "edit_local" not in b0["requests"][0]["spec"] and os.path.exists(os.path.join(P.d, "batch_0.json")))
check("no reads yet: the planner waits", P.next_batch(tipp, R, BP) is None and "waiting" in P.st["note"] and len(P.st["rounds"]) == 1)
res0 = engine(b0)
check("ingest refuses a judge row before anything is written", raises(lambda: P.ingest(res0[:5] + [dict(cell="units|give-up state unit", row=jr, sample=0, ok=True, tokens=5)], R), "has role 'judge'")
      and not os.path.exists(os.path.join(P.d, "reads.jsonl")))
n_in = P.ingest(res0, R)
check(f"ingest: {n_in} reads appended by key; a second ingest of the same lines adds none", n_in == b0["reads"] and P.ingest(res0, R) == 0)
P = SP.Plan.load(P.d)                                             # a new process: the state file is all it needs
used = [b0["reads"]]
while True:
    b = P.next_batch(tipp, R, BP)
    if b is None:
        break
    used.append(b["reads"])
    assert all(R.meta[r]["role"] in SC.SIGNAL_ROLES for q in b["requests"] for r in q["rows"])
    P.ingest(engine(b), R)
stp = P.st
act = [cid for cid, c in stp["cells"].items() if c["status"] == "active"]
logs = [l for r_ in stp["rounds"] for l in r_["log"]]
check(f"the race ends inside the budget ({stp['used']} of {stp['budget']} reads in {len(used)} batches {used}); finished; the leader is in the planted band: {act}",
      stp["finished"] and stp["used"] == sum(used) <= 6000 and len(used) <= stp["K"] and len(act) >= 1 and all(stp["cells"][c]["band"] == [21, 42] for c in act)
      and stp["cells"][act[0]]["level"][0] == 324)
stops = [c for c in stp["cells"].values() if c["status"] == "stopped"]
check(f"cells stop by row-paired bounds ({len(stops)} stopped, e.g. '{stops[0]['stop']['why'][:70]}'); halving parks, it does not stop",
      len(stops) >= 6 and any("paired upper bound of value against" in c["stop"]["why"] for c in stops) and all(c["stop"]["why"] for c in stops)
      and all("halving" in c["stop"]["why"] or "budget" in c["stop"]["why"] for c in stp["cells"].values() if c["status"] == "parked"))
newc = [cid for cid, c in stp["cells"].items() if c["born"] > 0]
check(f"refinement: new doses between the coarse ones next to the best dose, and band moves for the leader ({newc})", any(l.startswith("new XB|a") and "refine" in l for l in logs)
      and all(stp["cells"][c]["spec"]["family"] == "XB" for c in newc) and any(stp["cells"][c]["dose"] not in (0.5, 1.75, 3.0) for c in newc))
check("the explicit cells (state / position units) were raced as they are: no derived cell, no restrict_edit",
      sum(1 for c in stp["cells"].values() if c["spec"]["family"] == "units") == 2 and not any("units" in c and stp["cells"][c]["born"] > 0 for c in stp["cells"]))
readsP = SC.reads_table((json.loads(l) for l in open(os.path.join(P.d, "reads.jsonl"))), R)
docP = SC.make("t000", tipp, {cid: c["spec"] for cid, c in stp["cells"].items()}, readsP, R, BP)
topP = docP["cells"][0]
check(f"the cell maker on the planner's reads: first cell {topP['name']} (pred {100 * topP['pred']:+.1f} pp, x{topP['tokens_multiple']}, prior_family {topP['prior_family']}, "
      f"K {topP['K']}, p_pass {topP['p_pass']})", topP["band"] == [21, 42] and topP["prior_family"] == 60 and topP["K"] >= 61 and topP["kind"] == "weight"
      and topP["edit"]["op"] == "surgeon" and "edit_local" not in json.dumps(docP["cells"]))
trp = {r: dict(role=R.meta[r]["role"], prefix="PFX", settings={"min_image_tokens": 4096, "temperature": None}, ops=[{"op": "zoom"}], edit={"op": "vision_scale", "alpha": 1.1})
       for r in tipp}
Ls = SP.engine_lines(json.load(open(os.path.join(P.d, "batch_0.json"))), trp)
w0 = next(l for l in Ls["W"] if l["cell"] == "XB|a1.75|L21-42")
e0 = next(l for l in Ls["E"] if l["cell"].startswith("temperature|"))
check("engine request lines: the tip's view of the row (prefix, settings, ops) + the cell; a weight cell's edit comes after the tip's edit (engine W, edit_sha); "
      "a setting cell replaces the tip's value", len(Ls["W"]) + len(Ls["E"]) == b0["reads"] and w0["edit"] == [{"op": "vision_scale", "alpha": 1.1}, ef["edit"]]
      and w0["edit_sha"] == SC.esha(w0["edit"]) and w0["prefix"] == "PFX" and w0["mode"] == "read" and w0["settings"]["min_image_tokens"] == 4096 and w0["ops"] == [{"op": "zoom"}]
      and w0["key"] == f"sc|t000|{cellsP['XB|a1.75|L21-42']['sha']}|{w0['row']}|0" and e0["settings"]["temperature"] in (0.5, 1.5))
check("engine lines refuse a judge row", raises(lambda: SP.engine_lines(dict(b0, requests=[dict(b0["requests"][0], rows=[jr])]), dict(trp, **{jr: dict(role="judge")})), "has role 'judge'"))
check("a budget too small for one round of the coarse grid is an error", raises(lambda: SP.Plan.create(os.path.join(dpl, "p2"), "t000", tipp, R, LIB, 300).next_batch(tipp, R, BP), "do not fit"))

# ---------------------------------------------------------------------------------------------------------------- 11. with the patch
if hasattr(gen.Run, "g_prior_pad"):
    json.dump({"w": True, "e_cards": 6, "g_token_multiple": 2.0, "g_token_multiple_source": "test", "g_routed": True}, open(os.path.join(Rg.rd, "cards.json"), "w"))
    Rg.g = 2
    Rg.stamp_cards(Rg.state["gens"].setdefault("2", dict(t0=0.0, done=[], times={})))
    c11 = {k: dict(v, prior_family=60 if k == "glob" else 0) for k, v in c8.items()}
    doc11 = SC.make("t000", t8, c11, r8, R, SC.read_budget(Rg.rd), classes=CL8)
    _, sha11 = SC.write_cells(os.path.join(Rg.rd, "signal", "t000"), doc11)
    cs, info = Rg.g_read_cells()
    props = []
    Rg.w_budget_after_g(props, {"B": 2, "P0": 4, "BW": 3})
    rp = [p for p in props if p["cand"]["g"]["kind"] == "routed"]
    check(f"patched reader: the routed cell is a conditional candidate with its edit ({[(p['cand']['condition'][:30], 'edit' in p['cand']) for p in rp]}); g meta carries "
          f"prior_family / pred_raw / shrink / p_pass / route", len(rp) == 1 and rp[0]["cand"]["condition"] == CL8[0]["condition"] and rp[0]["cand"]["edit"] == c8["longgain"]["edit"]
          and all(k in p["cand"]["g"] for p in props for k in ("prior_family", "pred_raw", "shrink", "p_pass", "K")) and rp[0]["cand"]["g"]["route"] == "cond"
          and max(p["cand"]["g"]["prior_family"] for p in props) == 60 and info["state"] == "read"
          and all("route type 'length' cannot be proposed" in x["why"] for x in info["skipped"]))
    json.dump({"w": True, "g_token_multiple": 2.0}, open(os.path.join(Rg.rd, "cards.json"), "w"))
    Rg.g = 3
    Rg.stamp_cards(Rg.state["gens"].setdefault("3", dict(t0=0.0, done=[], times={})))
    props = []
    Rg.w_budget_after_g(props, {"B": 2, "P0": 4, "BW": 3})
    check("without cards.json g_routed the routed cell is not proposed (logged as waiting)", not [p for p in props if p["cand"]["g"]["kind"] == "routed"] and len(props) >= 2)
    cands = [dict(cid=f"g003c{i:02d}", author="G", gen=3, g=dict(prior_family=pf)) for i, pf in enumerate((60, 0, 5))] + [dict(cid="g003c09", author="B", gen=3)]
    pad = Rg.g_prior_pad(("G", 3), [c["cid"] for c in cands[:3]], {c["cid"]: c for c in cands})
    check(f"judge side: the BH family of (G, generation) is padded by the largest prior_family among its candidates ({len(pad)} extra hypotheses with p = 1); other authors: none",
          len(pad) == 60 and all(BH.pval(x["d"], x["se"], x["n"], x["nu"])[0] > 0.999999 for x in pad) and Rg.g_prior_pad(("B", 3), ["g003c09"], {c["cid"]: c for c in cands}) == [])
    xs = [dict(name="a", d=0.05, se=0.018, n=444, nu=400, share=1.0, is_global=False), dict(name="b", d=0.03, se=0.018, n=444, nu=400, share=1.0, is_global=False)]
    j0 = {r["name"]: r["accept"] for r in BH.judge(xs)}
    j1 = {r["name"]: r["accept"] for r in BH.judge(xs + pad) if not r["name"].startswith("__prior")}
    check(f"with the pad BH's bar is alpha x k / (K + 60): a z 2.8 cell passes alone ({j0}) and not with 60 prior cells ({j1})", j0 == {"a": True, "b": True} and j1 == {"a": False, "b": False})
else:
    print("skip 11 (the tree is not patched: signal_cells_1111.patch adds routed G cells, the prior_family pad of G's BH family and the realised fields)")

# ---------------------------------------------------------------------------------------------------------------- 12. B4b (coordinator 11-11)
# prior_family REQUIRED for weight cells (an integer >= 0), the honest family = cells sent + the SUM over distinct libraries
tpb, cpb, rpb, lpb = world(11, EFF)
for bad_pf, txt in ((None, "no prior_family"), ("50", "not an integer"), (-1, "not an integer"), (True, "not an integer"), (2.5, "not an integer")):
    cb_ = {k: dict(v) for k, v in cpb.items()}
    if bad_pf is None:
        cb_["glob"].pop("prior_family")
    else:
        cb_["glob"]["prior_family"] = bad_pf
    check(f"make refuses a weight cell with prior_family {bad_pf!r} ({txt})", raises(lambda: SC.make("t000", tpb, cb_, rpb, R, BUD), txt))
cs_ = {k: dict(v) for k, v in c8.items()}
cs_["temp"].pop("prior_family", None)
check("a setting cell needs no prior_family (make runs)", isinstance(SC.make("t000", t8, cs_, r8, R, SC.read_budget(Rg.rd), classes=CL8).get("cells"), list))
check("family_prior: the sum over distinct libraries, the largest within one library; a missing library falls back to the family",
      SC.family_prior([dict(prior_family=30, library="L1"), dict(prior_family=20, library="L2"), dict(prior_family=10, library="L1"),
                       dict(prior_family=5, family="F9"), dict(prior_family=None, library="L3")]) == (55, {"L1": 30, "L2": 20, "F9": 5}))
cl_ = {k: dict(v) for k, v in cpb.items()}
for k, lib, pf in (("glob", "L1", 30), ("costly", "L2", 20), ("longgain", "L1", 10)):
    cl_[k].update(library=lib, prior_family=pf)
dlb = SC.make("t000", tpb, cl_, rpb, R, BUD, classes=CL)
snt = dlb["cells"]
want = len(snt) + SC.family_prior(snt)[0]
check(f"K of every sent item = items sent + the libraries' sum ({[(it['name'], it.get('library'), it['prior_family']) for it in snt]}, K {snt[0]['K'] if snt else None}, want {want})",
      bool(snt) and all(it["K"] == want for it in snt) and all(it.get("library") for it in snt))
n1b = SC.cell_from_name("XS_bias_all_+1", {"edit": "/x/a.json"}, dict(prior_family=50, library="batch_cells"))
n2b = SC.cell_from_name("FLk8_L0-34_+1", {"edit": "/x/b.json"}, dict(prior_family=50, library="batch_cells"))
n3b = SC.cell_from_name("FLs8_L24-34_+1", {"edit": "/x/c.json", "prior_family": 7}, dict(prior_family=50, library="batch_cells"))
n4b = SC.cell_from_name("XS_bias_all_+1", {"edit": "/x/a.json"}, {})
check("batch_cells.json: one file-level count is ONE library (XS + FL count 50, not 100); a per-cell count is its family's library; none stated -> None (make refuses)",
      SC.family_prior([n1b, n2b])[0] == 50 and n3b["library"] == "FLs8" and n3b["prior_family"] == 7 and SC.family_prior([n1b, n2b, n3b])[0] == 57
      and n4b["prior_family"] is None)
check("the planner refuses a weight library entry without prior_family",
      raises(lambda: SP.Plan.create(os.path.join(dpl, "p_b4b"), "t000", tipp, R, [dict(LIB[2], prior_family=None)], 5000), "prior_family is required"))
if hasattr(gen.Run, "g_prior"):
    json.dump({"w": True, "e_cards": 6, "g_token_multiple": 2.0, "g_token_multiple_source": "test"}, open(os.path.join(Rg.rd, "cards.json"), "w"))
    Rg.g = 4
    Rg.stamp_cards(Rg.state["gens"].setdefault("4", dict(t0=0.0, done=[], times={})))
    its = [dict(kind="weight", name=f"w{i}", edit={"op": "surgeon", "file": f"/data/fd/mingdazhang/x/b4b_{i}.json"}, pred=0.05, se=0.01, **x)
           for i, x in enumerate([dict(prior_family=3, library="L"), dict(prior_family="3"), dict(prior_family=-2), dict(prior_family=True),
                                  dict(prior_family=2.5), dict(prior_family=1, library=""), dict(prior_family=1, library=7), dict()])]
    json.dump(dict(tip="t000", scale=SC.SCALE, cells=its), open(Rg.g_cells_path("t000"), "w"))
    csb, infb = Rg.g_read_cells()
    skp = {x["name"]: x["why"] for x in infb["skipped"]}
    check(f"reader: an item whose prior_family is present but not an integer >= 0 (or whose library is not a non-empty string) is skipped ({sorted(skp)})",
          [c["name"] for c in csb] == ["w0", "w7"] and set(skp) == {"w1", "w2", "w3", "w4", "w5", "w6"}
          and all("prior_family must be an integer" in skp[k] for k in ("w1", "w2", "w3", "w4")) and all("library must be" in skp[k] for k in ("w5", "w6")))
    pb = []
    Rg.w_budget_after_g(pb, {"B": 2, "P0": 4, "BW": 3})
    check("reader: library travels in c['g']", any(p["cand"]["g"].get("library") == "L" and p["cand"]["g"].get("prior_family") == 3 for p in pb))
    cg = [dict(cid=f"g004c{i:02d}", author="G", gen=4, g=g) for i, g in enumerate([dict(prior_family=30, library="L1"), dict(prior_family=20, library="L2"),
                                                                                   dict(prior_family=10, library="L1"), dict(prior_family=5, family="F9"), dict(kind="weight")])]
    bc = {c["cid"]: c for c in cg}
    n_, lib_ = Rg.g_prior([c["cid"] for c in cg], bc)
    pad = Rg.g_prior_pad(("G", 4), [c["cid"] for c in cg], bc)
    check(f"judge side: the pad is the SUM over distinct libraries ({lib_}) = {n_} p = 1 hypotheses, as signal_cells.family_prior; other authors none",
          n_ == 55 and lib_ == {"L1": 30, "L2": 20, "F9": 5} and len(pad) == 55 and Rg.g_prior_pad(("P0", 4), [c["cid"] for c in cg], bc) == []
          and n_ == SC.family_prior([c["g"] for c in cg])[0])
else:
    print("skip 12 judge side (the tree is not patched)")

print(f"\n{ok} ok, {fail} failed")
sys.exit(1 if fail else 0)
