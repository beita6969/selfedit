"""Full result of one generation of a v4 run (read only; never writes state): per author, per candidate, invalid entries by source,
read durations and budget hits by candidate, AM10 numbers for every candidate, tip decision, wall per step.
usage: gen_report.py RUN_DIR GEN [ROOT]"""
import collections, json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(os.path.dirname(HERE), "harness")]
import gen  # noqa: E402

RUN, g = sys.argv[1], int(sys.argv[2])
argv = ["--run", RUN] + (["--root", sys.argv[3], "--rows", os.path.join(sys.argv[3], "rows/learn.jsonl")] if len(sys.argv) > 3 else [])
R = gen.Run(gen.args(argv))
R.g = g
G = R.G()
tips = {t["id"]: t for t in R.state["tips"] + [R.state["tip"]]}
C = G["cands"]
par = C[0]["parent"]
cur_tip = R.state["tip"]
R.state["tip"] = tips[par]
R.load_routes(R.state["tip"]["entries"], R.role["show"] + R.role["judge"] + R.role["monitor"])
R.load_routes([c["entry"] for c in C if c.get("entry")], R.judge_rows)
V = [c for c in C if c.get("rung")]
AB = getattr(R.io, "aborted", {})    # keys with a finish "abort" line (coordinator 11-11 01:45); in R.io.res only if a good line came later
tipe = R.state["tip"]["entries"]
par_ab = {}
for r in R.role["show"] + R.role["judge"] + R.role["monitor"]:
    sg = R.read_req(tipe, r, 0, 5)[1]
    for s_ in range(gen.P_FULL):
        k = R.read_key(sg, r, s_)
        if k in AB:
            par_ab[k] = (r, s_, (R.io.res.get(k) or {}).get("t_done"))
B = R.judge_numbers("obs", V)        # judge_eb estimates + AM10 (report only; decisions are AM15, in c["judge"])


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(math.ceil(p * len(xs))) - 1)] if xs else None


rep = json.load(open(os.path.join(RUN, f"report_g{g:03d}.json")))
print(f"GENERATION {g} of {RUN}: tip {par} -> {rep.get('tip')} change {rep.get('tip_change')}; wall {rep['wall_s'] / 60:.1f} min")
print("wall per step (min): " + ", ".join(f"{k} {v / 60:.1f}" for k, v in (G.get("times") or {}).items()))
print("slots:", G.get("slots"), "| sources:", G.get("sources"), "| repairs sent:", G.get("repairs"))
print("Y per author:", json.dumps(rep.get("Y")))
print("authors:", json.dumps(rep.get("authors")))
print("by source:", json.dumps(rep.get("sources")))
print("\nINVALID (author, source, reason):")
for c in C:
    if not c.get("valid0"):
        print(f"  {c['cid']} {c['author']} {c.get('card_source')} | {c['status'][:240]}")
print("\nCANDIDATES:")
dur_all, hit_rows = [], collections.Counter()
for c in C:
    e = c.get("entry") or {}
    j, o, rs = c.get("judge") or {}, B.get(c["cid"]) or {}, c.get("route") or {}
    durs, hits, toks = [], 0, []
    if c.get("rung"):
        for r in R.rung_rows(c, 1):
            x = R.io.get(R.read_req(R.cand_entries(c), r, 0, 1)[0]["key"])
            if x is None:
                continue
            if x.get("t_done") and x.get("t_submit"):
                durs.append(x["t_done"] - x["t_submit"])
            toks.append(x.get("tokens") or 0)
            if x.get("finish") == "length" or (x.get("tokens") or 0) >= 32768:
                hits += 1
                hit_rows[r] += 1
    dur_all += durs
    a10 = o.get("am10") or {}
    print(f"- {c['cid']} {c['author']} src={c.get('card_source')} kind={c.get('kind')} target={c.get('target', 'E.H')} valid={bool(c.get('valid0'))} "
          f"accepted={bool(c.get('accepted'))} merged={bool(c.get('merged'))}" + (f" cell={c.get('cell')} ({c.get('cell_group')})" if c.get("cell") else "")
          + (f" lab_turns={c.get('lab_turns')} lab_tested={c.get('lab_tested')}" if c.get("lab_turns") else "") + (f" stack_of={c.get('stack_of')}" if c.get("stack_of") else ""))
    print(f"    condition: {json.dumps(e.get('condition'), ensure_ascii=False)}")
    print(f"    procedure: {json.dumps(e.get('procedure'), ensure_ascii=False)}")
    print(f"    settings: {json.dumps(e.get('settings'))}" + (f" ops: {json.dumps(e.get('ops'))}" if e.get("ops") else "")
          + (f" edit: {json.dumps(e.get('edit'))}" if e.get("edit") else ""))
    if c.get("rung"):
        f4 = lambda v, f: "n/a" if v is None else format(v, f)
        print(f"    routed {c.get('routed')}/{R.NJ} share {c.get('share', 0):.3f} unsure {rs.get('unsure')} | rows {j.get('n') or c.get('n_read')} d {f4(j.get('D'), '+.4f')} "
              f"se {f4(j.get('se'), '.4f')} (used {f4(j.get('se_used'), '.4f')}, nu {j.get('nu')}) | AM15: p {f4(j.get('p'), '.4f')} BH {j.get('bh')} (K {j.get('K')}, {j.get('group')})"
              + (f" LOBO {j.get('lobo')}" if j.get("is_global") else "") + f" lb {f4(j.get('lb'), '+.4f')} contribution_lb {f4(j.get('contribution_lb'), '+.4f')} "
              f"share x d {f4(j.get('share_d'), '+.4f')} | est. lfsr {f4(j.get('lfsr'), '.3f')} pm {f4(j.get('post_mean'), '+.4f')} | tok {c.get('tok_cand') or 0:.0f} vs parent "
              f"{c.get('tok_parent') or 0:.0f} (c {((c.get('tok_cand') or 0) / max(c.get('tok_parent') or 1, 1)):.2f})")
        a21 = c.get("am21") or {}
        if a21.get("step1"):                                     # AM21 LITE: the step-1 line, then the pooled line above is on a21["n"] samples
            s1 = a21["step1"]
            print(f"    AM21 LITE step 1 (1 sample): d {f4(s1.get('d'), '+.4f')} se {f4(s1.get('se'), '.4f')} p {f4(s1.get('p'), '.4f')} rows {s1.get('n_rows')} "
                  f"tok {s1.get('tok') or 0:.0f}/{s1.get('tok_parent') or 0:.0f} c {s1.get('c')} -> "
                  + ("left (p = 1)" if a21.get("left") else f"samples 1-{int(a21.get('n', 1)) - 1} read; pooled numbers above on {a21.get('n')} samples per row")
                  + (f"; box ended at {a21['box_at']} samples per row" if a21.get("box_at") is not None else ""))
        if c.get("bundle"):                                      # AM23 bundle: members, union, members holding per row, its own samples
            bd = c["bundle"]
            print(f"    AM23 bundle: {len(bd.get('members') or [])} members {bd.get('members')}" + (f" (carried {bd.get('carried')})" if bd.get("carried") else "")
                  + f"; union {bd.get('union')} rows, members holding per row {bd.get('holders')}, rows with >= 2 members {bd.get('rows_2plus')}; "
                  f"samples {a21.get('s0')}-{int(a21.get('s0') or 0) + int(a21.get('M') or 0) - 1} (new on every row)"
                  + (f"; built with step 1 incomplete for {bd.get('incomplete')}" if bd.get("incomplete") else ""))
        if c.get("merged_through"):
            print(f"    merged through bundle {c['merged_through']}")
        ck = R.cand_keys(c)
        cab = [k for k in ck if k in AB]
        if cab:
            print(f"    abort reads: {len(cab)} of its {len(ck)} keys had a finish \"abort\" line ({sum(1 for k in cab if k in R.io.res)} re-read since); "
                  f"loops before the 11-11 01:45 fix graded them wrong, from the fix on they are not read")
        hit = sorted({par_ab[k][0] for k in par_ab if par_ab[k][0] in set(R.rung_rows(c, 1))
                      and c.get("decided_at") and (par_ab[k][2] is None or float(c["decided_at"]) < float(par_ab[k][2]))})
        if hit:
            print(f"    decided before the re-read of an aborted parent sample on {len(hit)} of its routed rows (that parent sample counted wrong): {hit}")
        if c.get("skipped_ctx"):
            print(f"    skipped_ctx {c['skipped_ctx']} rows (prompt over the context; zero difference): {c.get('skipped_ctx_rows')}"
                  + ("  ** MOSTLY UNUSABLE (> 1/4 of its routed rows) **" if c.get("mostly_unusable") else ""))
        if c.get("accepted_v1") is not None:
            print(f"    v1 (before AM15): accepted {c.get('accepted_v1')} merged {c.get('merged_v1')} | {str(c.get('status_v1'))[:200]}")
        if c.get("final"):
            fj = c["final"]
            print(f"    final: d {f4(fj.get('D'), '+.4f')} p {f4(fj.get('p'), '.4f')} accept {fj.get('accept')} contribution_lb {f4(fj.get('contribution_lb'), '+.4f')}")
        print(f"    AM10 across sources: P(mu<=0) {a10.get('P_mu_le0')} mu {a10.get('mu_mean')} D_s {a10.get('D_s')} se_s {a10.get('se_s')} n_s {a10.get('n_s')}")
        ec = c.get("ecomp") or o.get("ec") or {}
        print(f"    equal compute: c {ec.get('c')} reference {'plurality of ' + str(ec['m']) + ' tip samples' if ec.get('m') else 'tip mean'}; "
              f"op_skipped rows {ec.get('skipped', 0)}, error rows {ec.get('errors', 0)}; SHOW rows routed {c.get('show_routed')}")
        if durs:
            print(f"    read s: p50 {q(durs, .5):.0f} p90 {q(durs, .9):.0f} p99 {q(durs, .99):.0f} max {max(durs):.0f} | budget hits {hits}/{len(toks)}")
    print(f"    status: {c['status']}")
print(f"\nABORT READS (finish \"abort\"): {sum(1 for c in V for k in R.cand_keys(c) if k in AB)} among the candidates' keys; parent cache of {par}: "
      f"{len(par_ab)} (re-read since: {sum(1 for k in par_ab if par_ab[k][2])}) on rows {sorted({v[0] for v in par_ab.values()})}")
BD = G.get("bundle") or {}
if BD:                                                           # AM23: the build decision and what the bundle would have held
    el = [c for c in C if c["cid"] in set(BD.get("eligible") or [])]
    pos = [c for c in el if ((c.get("am21") or {}).get("step1") or {}).get("d", 0) > 0 and not (c.get("am21") or {}).get("left")]
    print(f"AM23: state {BD.get('state')} ({BD.get('why', '')}); eligible {len(el)}; decision at read share {BD.get('read_frac')} with step 1 complete for "
          f"{BD.get('step1_complete')}; eligible with step-1 d > 0 by the end: " + (", ".join(
              f"{c['cid']} {c['author']} d {c['am21']['step1']['d']:+.4f} p {c['am21']['step1'].get('p', float('nan')):.3f} rows {c.get('routed')}" for c in pos) or "none"))
L21 = [c for c in G.get("cands", []) if (c.get("am21") or {}).get("step1") or (c.get("am21") and c.get("rung"))]
if L21:                                                          # AM21 LITE table: step 1 -> left / pooled at 4 -> decision
    print(f"\nAM21 LITE ({len(L21)} read candidates): left after step 1 {sum(1 for c in L21 if c['am21'].get('left'))}, "
          f"to samples 1-3 {sum(1 for c in L21 if int(c['am21'].get('n', 1)) > 1)}, accepted {sum(1 for c in L21 if c.get('accepted'))}, "
          f"box-ended {sum(1 for c in L21 if c['am21'].get('box_at') is not None)}")
    print("  cid      author rows  step1 d   p      | pooled d  se      p      BH    | decision")
    for c in sorted(L21, key=lambda c: c["cid"]):
        a21, j, s1 = c["am21"], c.get("judge") or {}, c["am21"].get("step1") or {}
        if a21.get("bundle"):
            print(f"  {c['cid']} bundle {c.get('routed') or 0:4d}  (no step 1: samples {a21.get('s0')}-{int(a21.get('s0') or 0) + int(a21.get('M') or 0) - 1}) | "
                  f"{(format(j['D'], '+.4f') + ' ' + format(j['se'], '.4f') + ' ' + format(j['p'], '.4f') + ' ' + str(j.get('bh'))) if j.get('D') is not None else 'n/a':33s} | "
                  f"{'ACCEPTED' if c.get('accepted') else 'not accepted'}{' (merged)' if c.get('merged') else ''}")
            continue
        fm = lambda v, f: "   n/a" if v is None else format(v, f)
        pooled = "left" if a21.get("left") else f"{fm(j.get('D'), '+.4f')} {fm(j.get('se'), '.4f')} {fm(j.get('p'), '.4f')} {str(j.get('bh')):5s}"
        print(f"  {c['cid']} {c['author']:6s} {c.get('routed') or 0:4d} {fm(s1.get('d'), '+.4f')} {fm(s1.get('p'), '.4f')} | {pooled:33s} | "
              f"{'ACCEPTED' if c.get('accepted') else 'not accepted'}{' (merged)' if c.get('merged') else ''}")
print(f"\nALL READS s: p50 {q(dur_all, .5):.0f} p90 {q(dur_all, .9):.0f} p99 {q(dur_all, .99):.0f} max {max(dur_all):.0f} (n {len(dur_all)}); "
      f"budget hits {sum(hit_rows.values())} on {len(hit_rows)} rows; rows with the most hits: {hit_rows.most_common(10)}")
# stock reference: budget hits among the launch cache on JUDGE rows
st = [R.io.get(f"p000|parent|{r}|{s}") for r in R.judge_rows for s in range(4)]
st = [x for x in st if x]
print(f"stock launch cache on JUDGE: {sum(1 for x in st if x.get('finish') == 'length')}/{len(st)} samples hit the budget")
print(f"tip file: {os.path.join(RUN, 'tip.json')} (current {cur_tip['id']})")
if rep.get("deployment"):
    print("deployment (k-sample plurality, MONITOR):", json.dumps(rep["deployment"]))
if os.path.exists(os.path.join(RUN, "signal_calibration.json")):               # B4: author G's predictions against the judge
    print(gen.SCELLS.quality_line(os.path.join(RUN, "signal")))
if G.get("l1"):
    print("L1:", json.dumps(G["l1"], default=str)[:6000])
if G.get("l1_alloc"):
    print("L1 slots:", json.dumps(G["l1_alloc"], default=str))
