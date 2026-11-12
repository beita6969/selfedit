"""Dry run of ONE full v5 generation on RECORDED reads (no engine, no GPU, no request written).

The recorded reads (exports E1-E8 of 11-11 and the JUDGE tables of the hand judge; learn rows only) are written as engine result lines
into a scratch root, under the keys the LOOP ITSELF asks for:
  - the tip t007c's cached reads, the floor-only reads and the leave-one-entry-out reads keep their recorded keys, and the dry run CHECKS
    that gen5 / gen.Run.read_req builds exactly those keys from the tip file, the routes and knobs.json (prompt signature per row);
  - the FL cell (signal rows: E4, JUDGE rows: the hand-judge tables), its random-edit null (RANDFL, signal rows) and the zoom pilot
    (signal rows: E5, tokens of both calls) were read under hand-made keys: they are re-keyed to the loop's request keys of the
    corresponding candidate (cells.json item / knobs "operators" entry of the dry run).
Then gen5 runs generation 1 offline from a v5 state made of the v4 tip t007c (all 11 entries in the general layer, audit pending) and the
result is compared with the stage-1 verdict table: the audit layers, the FL verdict, the zoom screen-out, the new general tip and its
cache. usage: dry5.py [--out ~/selfedit/b5_restart/dry_run]
"""
import argparse
import collections
import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(os.path.dirname(HERE), "v5")]
import gen5  # noqa: E402
import reads5 as R5  # noqa: E402
import stage1_data as SD  # noqa: E402

HOME = os.path.expanduser("~/selfedit")
FL_EDIT = {"op": "surgeon", "file": "/data/fd/mingdazhang/selfedit/v4/cells/fable_f1/edits/FLs8_L24-34_+0.75.json"}
RAND_EDIT = {"op": "surgeon", "file": "/data/fd/mingdazhang/selfedit/v4/cells/fable_f1/edits/RANDFLs8_L24-34_+0.75.json"}
ZOOM_OP = {"name": "zoom (pilot build)", "ops": [{"op": "zoom"}]}
GROUP_H = ["g003c16", "g004c11", "g005c13", "g005c22", "g005c28"]


def line(key, row, s, x, **kw):
    """an engine result line of a recorded read."""
    return dict(key=key, mode="read", row=row, sample=int(s), ok=bool(x["ok"]), pred=x.get("pred"), tokens=x["tokens"], prompt_tokens=x.get("prompt_tokens"),
                finish=x.get("finish") or "stop", content="(recorded)" if x.get("pred") is not None else "", reasoning=x.get("tail") or "", recorded=True, **kw)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HOME, "b5_restart", "dry_run"))
    a = ap.parse_args(argv)
    root, run = os.path.join(a.out, "root"), os.path.join(a.out, "run")
    for d in (root, run):
        if os.path.isdir(d):
            shutil.rmtree(d)                                   # the dry run's own scratch directories on the Mac
    os.makedirs(os.path.join(root, "q", "out"))
    rows = SD.rows()
    tip, fired, ents = SD.load_e1(rows)
    floor, stock = SD.load_e2(rows)
    wo, _ref, wo_rows, _pair = SD.load_e3(rows)
    W4 = SD.load_e4(rows)
    E5 = SD._arms(SD._lines("e5_signal_E_arms.jsonl.gz"), rows)
    zmeta = {(v["row"], int(v["s"])): v for v in SD._lines("e5_signal_E_arms.jsonl.gz") if v.get("arm") == "sig_zoom"}
    jarms, _ = R5.load_tsv(SD.W_JUDGE, rows)
    src = {e["cid"]: [SD.load_e6()[e["i"]]] for e in ents if e["i"] in SD.load_e6()}
    # --- state, knobs, cells.json of the dry run
    tipfile = json.load(open(os.path.join(HOME, "v4_tips", "t007c.json")))
    st = gen5.init_state(tipfile, src, None, note="dry run on recorded reads: v4 tip t007c, every entry in the general layer, audit pending")
    os.makedirs(run)
    gen5.jdump(os.path.join(run, "state.json"), st)
    knobs = json.load(open(os.path.join(HERE, "knobs.json")))
    knobs["operators"] = [ZOOM_OP]
    kp = os.path.join(a.out, "knobs_dry.json")
    gen5.jdump(kp, knobs)
    cp = os.path.join(a.out, "cells_dry.json")
    gen5.jdump(cp, dict(tip=st["tip"]["id"], cells=[dict(kind="weight", name="FLs8_L24-34_+0.75", edit=FL_EDIT, null_edit=RAND_EDIT, pred=0.0375,
                                                      note="recorded reads of 11-11; fitted on signal-row traces")]))
    gp = os.path.join(a.out, "audit_groups_dry.json")
    gen5.jdump(gp, {"H": GROUP_H})
    # --- routes first (the loop's keys need them)
    out = open(os.path.join(root, "q", "out", "recorded_routes.jsonl"), "w")
    conds = {e["i"]: e["cond"] for e in ents if e["cond"] != "always"}
    for r, f in fired.items():
        for i, c in conds.items():
            out.write(json.dumps(dict(key=f"rt1|{c}|{r}", mode="route", row=r, yes=i in f, p=1.0 if i in f else 0.0, p_src="recorded")) + "\n")
    out.close()
    argv5 = ["--run", run, "--root", root, "--rows", SD.LEARN, "--knobs", kp, "--offline", "--p0-slots", "0", "--cells", cp, "--audit-groups", gp, "--gens", "1"]
    Rn = gen5.Run5(gen5.args5(argv5))
    T = Rn.tip()["entries"]
    allr = Rn.SIGR + Rn.JR
    Rn.load_routes(T, allr)
    checks = collections.OrderedDict()
    # --- recorded reads under the loop's keys
    out = open(os.path.join(root, "q", "out", "recorded_reads.jsonl"), "w")
    bad, written = 0, set()
    for r in allr:                                             # the tip's cache: recorded key == the loop's key
        for x in tip[r]:
            k = Rn.read_req(T, r, x["s"], 0)[0]["key"]
            bad += k != x["key"]
            written.add(x["key"])
            out.write(json.dumps(line(x["key"], r, x["s"], x)) + "\n")
    checks["tip t007c: the loop's read keys equal the recorded cache keys (768 rows x 8)"] = (bad == 0, f"{bad} differ")
    F = [e for e in T if e["cond"] == "always"]
    bad = 0
    for r in allr:                                             # the general base (floor-only view)
        for x in floor[r]:
            k = Rn.read_req(F, r, x["s"], 0)[0]["key"]
            bad += k != x["key"]
            if x["key"] not in written:                        # a row where no entry fires: the tip's view IS the floor-only view (one cache read)
                written.add(x["key"])
                out.write(json.dumps(line(x["key"], r, x["s"], x)) + "\n")
    checks["general base: the floor-only view's keys equal the recorded keys (768 rows x 8)"] = (bad == 0, f"{bad} differ")
    bad = n = same_view = 0
    for g, arm in wo.items():                                  # leave-one-entry-out reads: recorded key minus its batch prefix == the loop's key
        mem = set(GROUP_H) if g == "H" else {next(e["cid"] for e in ents if e["i"] == int(g[1:]))}
        V = [e for e in T if e["cid"] not in mem]
        for r, xs in arm.items():
            for x in xs:
                k = Rn.read_req(V, r, x["s"], 0)[0]["key"]
                rec = x["key"].split("|", 1)[1]
                bad += k != rec
                n += 1
                if k in written:                               # the view without the entry is a view the cache already holds (e.g. the floor-only
                    same_view += 1                             # view where only this entry fires): the loop reads nothing there and uses the
                    continue                                   # cache's 8 reads; the pilot's second read of the same request is not written
                written.add(k)
                out.write(json.dumps(line(k, r, x["s"], x)) + "\n")
    checks[f"leave-one-entry-out: the loop's keys equal the recorded keys ({n} reads)"] = (
        bad == 0, f"{bad} differ; {same_view} of them are views the cache already holds with 8 reads (the loop uses those)")
    base = dict(condition="always", cond="always", procedure=None, settings={}, ops=[])
    fl, nl = dict(base, edit=FL_EDIT, author="G"), dict(base, edit=RAND_EDIT, author="G")
    zo = dict(base, ops=gen5.PR.canon_ops(ZOOM_OP["ops"]), author="O")
    n = 0
    for arm, ent in ((W4[SD.FL], fl), (jarms[SD.FL], fl), (W4["RANDFLs8_L24-34_+0.75"], nl)):   # re-keyed: FL on signal + JUDGE rows, its null on signal rows
        for r, xs in arm.items():
            for x in xs:
                q = Rn.read_req(T + [ent], r, x["s"], 0)[0]
                assert q["engine"] in ("W", "T") and q["edit"] == ent["edit"]
                out.write(json.dumps(line(q["key"], r, x["s"], x, engine_type="W")) + "\n")
                n += 1
    for r, xs in E5["sig_zoom"].items():                       # re-keyed: the zoom pilot, tokens of both calls
        for x in xs:
            m = zmeta[(r, x["s"])]
            q = Rn.read_req(T + [zo], r, x["s"], 0)[0]
            assert q.get("ops")
            y = dict(x, tokens=x["tokens"] + float(m.get("loc_tokens") or 0), prompt_tokens=x["prompt_tokens"] + float(m.get("loc_prompt_tokens") or 0))
            out.write(json.dumps(line(q["key"], r, x["s"], y)) + "\n")
            n += 1
    out.close()
    checks[f"re-keyed to the loop's requests: FL (signal + JUDGE), RANDFL null (signal), zoom pilot (signal): {n} reads"] = (n > 0, "")
    # --- one generation, offline
    t0 = time.time()
    Rn = gen5.Run5(gen5.args5(argv5))
    Rn.main()
    G = Rn.state["gens"]["1"]
    by = {c["name"]: c for c in G["cands"]}
    # --- compare with the stage-1 verdict table
    s1 = json.load(open(os.path.join(HOME, "b5_restart", "stage1", "verdicts.json")))
    name1 = {"E1": "g002c26", "E2": "g003c00", "E5": "g005c00", "E6": "g005c11", "E9": "g005c23", "E11": "g005c29", "H": "H"}
    ok, det = True, []
    for r in s1["entry_audit"]:
        key = "audit " + name1[r["name"].split(" ")[0]]
        c = by[key]
        # the loop uses the cache's 8 reads where the view without the entry is a cached view, and its guess windows are (600, 1500): the
        # effects may differ from the stage-1 table (2 pilot reads everywhere, windows 600 / 2500) by sampling noise, the layers may not
        same = c.get("layer") == r["layer"] and abs(c["judge"]["d"] - r["eq"]["d"]) < 0.01 and abs(c["judge"]["pooled"] - r["pooled"]["d"]) < 0.01
        ok &= same
        det.append(f"{key}: {c.get('layer')} off-source {100 * c['judge']['d']:+.2f} (stage 1 {100 * r['eq']['d']:+.2f}), all benchmarks {100 * c['judge']['pooled']:+.2f} ({100 * r['pooled']['d']:+.2f})")
    checks["entry audit: the same layers as the stage-1 table, effects within 1 pp (7 units; the loop has 8 cached reads where stage 1 had 2)"] = (ok, "; ".join(det))
    c = by["FLs8_L24-34_+0.75"]
    f1 = next(r for r in s1["generation"] if r["name"].startswith("FL"))
    checks["FL cell: passes the signal-row pre-screen, rejected on JUDGE rows, same equal-compute effect as stage 1"] = (
        c["screen"]["passed"] and c.get("sent") and not c["accept"] and abs(c["judge"]["d"] - f1["eq"]["d"]) < 0.002,
        f"signal {100 * c['screen']['d']:+.2f} -> JUDGE {100 * c['judge']['d']:+.2f} (stage 1 {100 * f1['eq']['d']:+.2f})")
    z = by[ZOOM_OP["name"]]
    checks["zoom operator: screened out on signal rows, no JUDGE read"] = (z["screen"]["judged"] and not z["screen"]["passed"] and not z.get("sent") and not z.get("decided"),
                                                                           f"signal {100 * z['screen']['d']:+.2f} +- {100 * z['screen']['se']:.2f}")
    pol = [c for c in G["cands"] if c["kind"] == "policy"]
    checks["escalation policy with the cell: built offline from the grammar, sent, rejected on JUDGE rows"] = (
        len(pol) == 1 and pol[0].get("sent") and not pol[0]["accept"], "; ".join(f"{p['name']} signal {100 * p['screen']['d']:+.2f} -> JUDGE {100 * p['judge']['d']:+.2f}" for p in pol))
    t = Rn.tip()
    checks["new tip: general layer = the image floor only; the 11 entries in the specialisation layer in their original order"] = (
        [e["cid"] for e in t["entries"]] == ["g001c34"] and [e["cid"] for e in t["special"]] == [e["cid"] for e in ents if e["cond"] != "always"],
        f"{t['id']}: general {[e['cid'] for e in t['entries']]}, specialisation {len(t['special'])}")
    nc = G.get("new_tip_cache") or {}
    checks["cache of the new tip: 8 recorded reads on all 768 learn rows"] = (nc.get("rows_by_reads") == {"8": 768} and nc.get("missing_reads") == 0, str(nc))
    sh = Rn.state["shrinkage"]
    checks["shrinkage history: signal-row vs JUDGE-row effect recorded for the cell and the policy"] = (
        len(sh) == 2 and all(r["signal_d"] > r["judge_d"] for r in sh), "; ".join(f"{r['kind']} kept {100 * r['kept']:.0f}%" for r in sh))
    fr = G["frontier"]
    f_s1 = {p["name"]: p["acc"] for p in s1["frontier"]["judge"]}
    checks["frontier of the tip: JUDGE values equal the stage-1 table; default budget = 2 reads + third on disagreement"] = (
        fr["default"] == gen5.DEFAULT_POLICY and all(abs(p["acc_judge"] - f_s1[p["name"]]) < 1e-9 for p in fr["points"] if p["name"] in f_s1 and "@2500" not in p["name"]),
        ", ".join(f"{p['name']} {100 * p['acc_judge']:.2f} x{p['multiple']:.2f}" for p in fr["points"] if p["name"] in fr["report"]))
    rep = open(os.path.join(run, "report5_g001.txt")).read()
    checks["report: both systems on JUDGE rows (general 67.45; general + specialisation 70.61)"] = ("general                    acc  67.45" in rep and "general + specialisation   acc  70.61" in rep, "")
    # a second generation on the new general tip: nothing to audit, the frontier of the general base from its recorded cache
    Rn2 = gen5.Run5(gen5.args5(argv5[:-1] + ["1", "--no-audit", "--g-cells", "0"]))
    Rn2.knobs["operators"] = []
    Rn2.main()
    fr2 = Rn2.state["gens"]["2"]["frontier"]
    gb = {p["name"]: p["acc"] for p in s1["general_base"]["frontier"]}
    checks["generation 2 on the general tip: its frontier equals the stage-1 general-base frontier (67.45 / 69.26 / 70.41)"] = (
        all(abs(p["acc_judge"] - gb[p["name"]]) < 1e-9 for p in fr2["points"] if p["name"] in ("vote1", "n2:disagree->tip", "n3:disagree->tipx2")),
        ", ".join(f"{p['name']} {100 * p['acc_judge']:.2f}" for p in fr2["points"] if p["name"] in fr2["report"]))
    L = [f"v5 dry run on recorded reads ({time.strftime('%Y-%m-%d %H:%M')}; {time.time() - t0:.0f} s for two generations; no request written)"]
    for k, (v, d) in checks.items():
        L.append(f"{'ok  ' if v else 'FAIL'} {k}" + (f" | {d}" if d else ""))
    nbad = sum(not v for v, _ in checks.values())
    L.append(f"{len(checks) - nbad} ok, {nbad} failed")
    open(os.path.join(a.out, "dry_run_result.txt"), "w").write("\n".join(L) + "\n")
    for f in ("report5_g001.txt", "report5_g002.txt"):
        shutil.copy(os.path.join(run, f), os.path.join(a.out, f))
    print("\n".join(L))
    return nbad


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
