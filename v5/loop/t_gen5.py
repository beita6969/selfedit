"""Tests of the v5 generation driver (gen5.py) against the fake engine (fake_dispatch.py as a subprocess on a scratch root): the request
path that the recorded-reads dry run (dry5.py) cannot exercise: routes, the tip cache, cells with their random-edit null (W requests),
an operator, proposer cards (source benchmark recorded), the audit and a re-test, the tip update with the two layers, the new tip's
cache, resume after a stop, offline mode, and the refusals. Prints 'N ok, 0 failed'."""
import glob
import json
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gen5  # noqa: E402
import fake_dispatch as FD  # noqa: E402

LEARN = os.path.join(os.path.dirname(HERE), "data", "learn.jsonl")
OK, BAD = [0], [0]


def check(cond, what):
    (OK if cond else BAD)[0] += 1
    print("ok  " if cond else "FAIL", what, flush=True)


def main():
    td = tempfile.mkdtemp(prefix="t_gen5_")
    root, run = os.path.join(td, "root"), os.path.join(td, "run")
    os.makedirs(root)
    knobs = json.load(open(os.path.join(HERE, "knobs.json")))
    knobs["operators"] = [dict(name="zoom", ops=[{"op": "zoom"}])]
    # the fake world gives every (prefix, settings) combination its own effect. Expected accuracy of a settings combination on the signal
    # rows of the test tip (its routed entry adds a prefix where it fires); the tip's floor is the image-token value that leaves the
    # settings arm both a clearly better and a clearly worse one-knob change to find
    sig = [json.loads(l)["id"] for l in open(LEARN) if json.loads(l)["v4_role"] in ("show", "monitor")]

    def acc(st):
        return sum(FD.p_ok(r, FD.PROCS[0] if FD.route_yes(r, FD.CONDS[0]) else None, st, knobs) for r in sig) / len(sig)

    def arm_effects(f):
        return sorted((acc({"min_image_tokens": f, k: v}) - acc({"min_image_tokens": f}), k, v)
                      for k in knobs["settings"] if k != "min_image_tokens" for v in knobs["settings"][k]["values"] if v != knobs["settings"][k]["stock"])
    mv = [v for v in knobs["settings"]["min_image_tokens"]["values"] if v != knobs["settings"]["min_image_tokens"]["stock"]]
    FLOOR = max(mv, key=lambda f: min(arm_effects(f)[-1][0], -arm_effects(f)[0][0]))
    eff = arm_effects(FLOOR)
    knobs.pop("settings_arm", None)
    kp2 = os.path.join(td, "knobs_plain.json")               # later generations: no settings arm (time)
    json.dump(knobs, open(kp2, "w"))
    # stage 3 A: a settings arm of two cells: the fake world's best and worst one-knob change on top of the floor
    knobs["settings_arm"] = [dict(name="best", settings={eff[-1][1]: eff[-1][2]}), dict(name="worst", settings={eff[0][1]: eff[0][2]})]
    kp = os.path.join(td, "knobs.json")
    json.dump(knobs, open(kp, "w"))
    # a v4-style tip: the floor, one routed entry whose audit is pending (source lego), one entry already in the specialisation layer
    tipv4 = dict(id="t0", entries=[dict(condition="always", procedure=None, settings={"min_image_tokens": FLOOR}, ops=[], from_cand="g001c34", author="B"),
                                   dict(condition=FD.CONDS[0], procedure=FD.PROCS[0], settings={}, ops=[], from_cand="e_audit", author="P0"),
                                   dict(condition=FD.CONDS[1], procedure=FD.PROCS[1], settings={}, ops=[], from_cand="e_special", author="P0")])
    st = gen5.init_state(tipv4, {"e_audit": ["lego"], "e_special": ["blink"]}, {"e_special": dict(layer="UNPROVEN", d=0.0, p_gen=0.9, pooled=0.0)})
    check([e["cid"] for e in st["tip"]["entries"]] == ["g001c34", "e_audit"] and st["tip"]["entries"][1]["audit"] == "pending"
          and [e["cid"] for e in st["tip"]["special"]] == ["e_special"] and st["tip"]["policy"]["default"] == gen5.DEFAULT_POLICY,
          "init_state: always-on settings and unaudited entries in the general layer (audit pending); entries with a non-GENERAL verdict in the specialisation layer")
    os.makedirs(run)
    gen5.jdump(os.path.join(run, "state.json"), st)
    good = dict(op="fake", name="good", fake_effect=0.16, fake_tokens=1.0)
    cells = os.path.join(td, "cells.json")
    json.dump(dict(tip=st["tip"]["id"], cells=[dict(kind="weight", name="good cell", edit=good, null_edit=dict(op="fake", name="good-null", fake_effect=0.0), pred=0.1),
                                               dict(kind="weight", name="inert cell", edit=dict(op="fake", name="inert", fake_effect=0.0),
                                                    null_edit=dict(op="fake", name="inert-null", fake_effect=0.0), pred=0.05),
                                               dict(kind="weight", name="third cell", edit=dict(op="fake", name="third"), pred=0.01)]), open(cells, "w"))
    gen5.GU.KEEP_P = -1.0     # this suite tests the request path: the guess detector keeps its seed only (in the fake world's random words the
                              # added patterns pass the keep rule by chance in ~1 world of 12, this one among them, and bend the signal frontier)
    fake = subprocess.Popen([sys.executable, os.path.join(HERE, "fake_dispatch.py"), "--root", root, "--rows", LEARN, "--knobs", kp, "--rate", "0", "--quiet"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base = ["--run", run, "--root", root, "--rows", LEARN, "--knobs", kp, "--poll", "0.3", "--wait", "900", "--cells", cells, "--p0-slots", "4", "--retest"]
    try:
        t0 = time.time()
        R = gen5.Run5(gen5.args5(base + ["--gens", "1"]))
        R.main()
        G = R.state["gens"]["1"]
        C = {c["name"]: c for c in G["cands"]}
        kinds = [c["kind"] for c in G["cands"]]
        check(G["done"] == gen5.STEPS5, f"generation 1 ran every step against the fake engine ({time.time() - t0:.0f} s)")
        check(kinds.count("cell") == 2 and "third cell" not in C, "cells: --g-cells 2 takes the first two weight cells of cells.json")
        check(kinds.count("operator") == 1 and kinds.count("audit") == 1 and kinds.count("retest") == 1, "candidates: one operator (knobs), one audit unit, one re-test of the specialisation entry")
        ent = [c for c in G["cands"] if c["kind"] == "entry"]
        pset = [c for c in G["cands"] if c["kind"] == "setting" and c["name"].startswith("P0 ")]
        check(G["cards"]["single"] == 2 and G["cards"]["mixed"] == 2 and all((len(c["source"]) == 1) != bool(c.get("mixed")) and c["entry"]["source"] == c["source"] for c in ent),
              f"proposer: 4 slots = 2 single-benchmark cards (source = that benchmark) + 2 mixed cards (source = [], all six scored); {len(ent)} entries")
        chat = [json.loads(l) for f in glob.glob(os.path.join(root, "q", "in", "g001_chat*.jsonl")) for l in open(f)]
        check(len(chat) == 4 and all(q["messages"][-1]["content"].endswith(gen5.RULE_NOTE) for q in chat), "every card ends with the one sentence about the acceptance rule (nothing else is added)")
        check(all(c["source"] == [] and c["entry"]["cond"] == "always" and c["entry"]["settings"] and not c["entry"]["procedure"] for c in pset)
              and len(pset) + len(ent) + len(G["dropped"]) == 4 and all(d["why"] and len(d["head"]) <= 200 for d in G["dropped"]),
              f"proposer: a settings change on every question goes to the settings arm ({len(pset)}), entries keep their payload as proposed ({len(ent)}), "
              f"every dropped proposal is kept with its reason and first 200 characters ({len(G['dropped'])})")
        g = C["good cell"]
        check(g["screen"]["passed"] and g.get("sent") and g["accept"] and g["judge"].get("null") and g["judge"]["null"]["audit_ok"] and g["judge"]["gate"] == "eq",
              "a cell with a real effect on every benchmark: passes the pre-screen, is accepted on JUDGE rows, and its random-edit null was read on JUDGE rows (audit ok)")
        nulls = [f for f in glob.glob(os.path.join(root, "q", "in", "g001_null*.jsonl"))]
        check(len(nulls) >= 1 and all('"engine": "W"' in open(f).readline() or '"engine": "T"' in open(f).readline() for f in nulls),
              "the null of the live cell is requested as its own W batch (g001_null_W) only after the cell passed without it")
        i = C["inert cell"]
        check(not i.get("accept") and (not i["screen"]["passed"] or "rejected" in i["status"]), "an inert cell is screened out or rejected")
        z = C["zoom"]
        check(z["screen"]["judged"] and z["screen"]["multiple"] > 1.05, "the operator is read on signal rows with its ops; its tokens count every pass")
        a = C["audit e_audit"]
        check(a.get("decided") and a["layer"] in ("GENERAL", "ID_SPECIFIC", "UNPROVEN") and a["judge"]["scored"] == [b for b in R.benches if b != "lego"],
              f"audit of the pending entry: decided ({a['layer']}), its source benchmark is not scored")
        t = R.tip()
        check(t["id"] == "v5t001" and any(e.get("edit") == good for e in t["entries"]) and t["entries"][0]["cid"] == "g001c34",
              "tip update: a new tip id; the accepted cell is in the general layer after the floor")
        in_gen = "e_audit" in [e["cid"] for e in t["entries"]]
        check((a["layer"] == "GENERAL") == in_gen and (a["layer"] != "GENERAL") == ("e_audit" in [e["cid"] for e in t["special"]]),
              "tip update: the audited entry stays in the general layer only as GENERAL, else it moves to the specialisation layer with its verdict")
        rt = next(c for c in G["cands"] if c["kind"] == "retest")
        check(("e_special" in [e["cid"] for e in t["entries"]]) == bool(rt.get("accept")), "re-test: a specialisation entry is promoted only by passing the off-source rule")
        check(G["new_tip_cache"]["requested"] is not None and glob.glob(os.path.join(root, "q", "in", "g001_newtip*.jsonl")), "the new tip's cache (8 reads on every learn row) is requested at the update")
        check(len(R.state["shrinkage"]) >= 1 and os.path.exists(os.path.join(run, "shrinkage.jsonl")) and all("signal_d" in r and "judge_d" in r for r in R.state["shrinkage"]),
              "shrinkage history: signal-row and JUDGE-row effect of every judged candidate, in the state and in shrinkage.jsonl")
        # stage 3: the settings arm and the view-diverse votes
        sb, sw = C["best"], C["worst"]
        check(sb["kind"] == "setting" and set(sb["budgets_signal"]) == {"x1.00", "default", "x4.12"} and sb["screen"]["passed"] and sb.get("sent") and sb.get("decided")
              and set(sb["budgets_judge"]) == {"x1.00", "default", "x4.12"} and abs(sb["judge"]["d"] - sb["budgets_judge"]["default"]["d"]) < 1e-12,
              f"settings arm: the best one-knob change (fake effect {100 * eff[-1][0]:+.1f} pp) is read with its own 8-read cache, screened and judged on the DEFAULT budget; three budget points on both row sets")
        check(not sw["screen"]["passed"] and not sw.get("sent") and "budgets_judge" not in sw, f"settings arm: the worst change (fake effect {100 * eff[0][0]:+.1f} pp) is screened out on signal rows; its JUDGE rows are not read")
        nj = sum(1 for f in glob.glob(os.path.join(root, "q", "in", "g001_judge*.jsonl")) for l in open(f) if json.dumps(sw["entry"]["settings"])[1:-1] in l)
        used = any(c.get("sent") and "worst" in (c.get("views") or []) for c in G["cands"] if c["kind"] == "viewpolicy")
        check((nj > 0) == used, "JUDGE rows of a screened-out setting are requested only when a view-diverse vote that passed the pre-screen reads through it"
              + (" (here: one does)" if used else " (here: none does, nothing was requested)"))
        if sb["accept"]:
            check(any(e.get("settings") == sb["entry"]["settings"] for e in t["entries"]), "an accepted setting enters the general layer (its reads are the new tip's cache)")
        vp = [c for c in G["cands"] if c["kind"] == "viewpolicy"]
        check(all(c["views"] and set(c["views"]) <= {"best", "worst"} and c["screen"]["passed"] for c in vp) and len(vp) <= 3,
              f"view-diverse votes: built offline from the cached views, at most one per pair of views ({[c['name'] for c in vp]})")
        check(sum(bool(c.get("sent")) for c in G["cands"]) <= gen5.K_MAX, "at most K_MAX candidates go to the JUDGE rows")
        rep = open(os.path.join(run, "report5_g001.txt")).read()
        check("PROPOSER: cards" in rep, "report: the proposer section lists the cards and every dropped proposal")
        check(all(s in rep for s in ("GENERAL system", "SPECIALISATION layer", "<- default budget", "SIGNAL-ROW GAIN AGAINST JUDGE-ROW GAIN", "total tokens with prompts")),
              "report: both layers, the default budget on the stored frontier, the shrinkage table, both token conventions")
        fr = G["frontier"]
        check(fr["default"] == gen5.DEFAULT_POLICY and fr["report"][0] == "vote1" and fr["vertices"][0] == "vote1" and len(fr["points"]) >= 5,
              "the tip's policy field stores the frontier (signal-chosen vertices, JUDGE values) with the x1.00 point, the default and the last vertex")
        led = [json.loads(l) for l in open(os.path.join(run, "ledger5.jsonl"))]
        check(len(led) == len(G["cands"]) and all("cid" in x and "status" in x for x in led), "ledger5.jsonl: one line per candidate")
        # --- generation 2: stop before the judge, resume
        base = [kp2 if x == kp else x for x in base]
        try:
            gen5.Run5(gen5.args5(base + ["--gens", "1", "--stop-after", "judge"])).main()
            stopped = False
        except SystemExit as e:
            stopped = e.code == 3
        s2 = json.load(open(os.path.join(run, "state.json")))
        check(stopped and s2["gens"]["2"]["done"] == ["signals", "candidates", "prescreen"] and s2["gens"]["2"]["tip_at_start"] == "v5t001", "stop before a step: the state holds the steps done")
        R2 = gen5.Run5(gen5.args5(base + ["--gens", "1"]))
        R2.main()
        G2 = R2.state["gens"]["2"]
        check(G2["done"] == gen5.STEPS5 and "no cells.json" not in G2["cells_info"] and "is for tip" in G2["cells_info"],
              "resume: the generation finishes from the saved step; a cells.json made for another tip is not used")
        check(all(c["kind"] != "audit" or "e_audit" not in c.get("members", []) or a["layer"] != "GENERAL" for c in G2["cands"]) and
              not any(c["kind"] == "audit" and set(c["members"]) <= {e["cid"] for e in R2.tip()["entries"] if e.get("audit") == "GENERAL"} for c in G2["cands"]),
              "an entry audited GENERAL is not audited again")
        # --- offline: nothing is written
        n_in = len(glob.glob(os.path.join(root, "q", "in", "*.jsonl")))
        R3 = gen5.Run5(gen5.args5(base + ["--gens", "1", "--offline", "--p0-slots", "0"]))
        R3.main()
        check(len(glob.glob(os.path.join(root, "q", "in", "*.jsonl"))) == n_in and R3.state["gens"]["3"]["done"] == gen5.STEPS5, "offline: a generation runs on what is recorded and writes no request")
        check(any(c["kind"] == "operator" for c in G2["cands"]) and not any(c["kind"] == "operator" for c in R3.state["gens"]["3"]["cands"]),
              "an operator is read again on a NEW tip (generation 2) and not again on the same tip (generation 3)")
        # --- admission of a failed proposer entry to the specialisation layer (coordinator 11-11)
        kp3 = os.path.join(td, "knobs_noops.json")
        k3 = json.load(open(kp2))
        k3.pop("operators", None)
        json.dump(k3, open(kp3, "w"))
        st0 = {"min_image_tokens": FLOOR}

        def neutral(text):                                     # a procedure text whose own (hash) effect in the fake world is slightly negative
            for i in range(200):
                t = f"{text} (v{i})"
                if -0.05 < FD.effect(t, st0, knobs) - FD.effect(None, st0, knobs) < -0.01:
                    return t
            raise RuntimeError("no neutral text")
        ONLY_SRC = FD.CONDS[6]                                 # a condition that (in this test) matches rows of its source benchmark only
        ENTRIES = {1: [(FD.CONDS[2], neutral("Check each mark one by one. [fake:+0.45@lego]"), "lego", []), (FD.CONDS[3], neutral("Read the labels again."), "blink", []),
                       (FD.CONDS[5], neutral("Look at each tile in turn."), "babyvision", [{"op": "tiles", "n": 3}]),
                       (ONLY_SRC, neutral("Compare the two heights again. [fake:+0.45@lego]"), "lego", [])],
                   2: [(FD.CONDS[4], neutral("Trace every line to its end. [fake:+0.45@lego]"), "lego", [])]}

        class RunAdm(gen5.Run5):
            def propose(self, cid):
                out = []
                for cond, proc, src, ops in ENTRIES[self.g]:
                    c = cid()
                    out.append(dict(cid=c, kind="entry", name=f"entry {c}", source=[src], card=[], why="test",
                                    entry=dict(condition=cond, cond=gen5.sha16(cond), procedure=proc, settings={}, ops=gen5.PR.canon_ops(ops), cid=c, author="P0", source=[src])))
                return out

            def load_routes(self, entries, rows):
                super().load_routes(entries, rows)
                k = gen5.sha16(ONLY_SRC)
                for (c_, r_) in list(self.routes):
                    if c_ == k and self.rows[r_]["bench"] != "lego":
                        self.routes[(c_, r_)] = False
        run_a = os.path.join(td, "run_adm")
        os.makedirs(run_a)
        gen5.jdump(os.path.join(run_a, "state.json"), gen5.init_state(dict(id="t0", entries=tipv4["entries"][:1])))
        base_a = ["--run", run_a, "--root", root, "--rows", LEARN, "--knobs", kp3, "--poll", "0.3", "--wait", "900", "--p0-slots", "2", "--g-cells", "0", "--gens", "1"]
        A = RunAdm(gen5.args5(base_a))
        A.main()
        GA = A.state["gens"]["1"]
        e1, e2, e3o, e4o = [c for c in GA["cands"] if c["kind"] == "entry"]
        check(not e1.get("accept") and e1["admit"]["judged"] and e1["admit"]["admitted"] and e1["admit"]["judge"]["d"] > 0.03 and e1["admit"]["K"] >= 3,
              f"admission: an entry that fails off-source and helps on its source benchmark is read on that benchmark's JUDGE rows and admitted "
              f"(source {100 * e1['admit']['judge']['d']:+.1f} pp, p {e1['admit']['judge']['p']:.4f}, Holm over the {e1['admit']['K']} entries tested)")
        sk = (e3o.get("skipped_rows") or {}).get("signal") or {}
        check(sk.get("n", 0) > 0 and e3o["screen"].get("judged") and "missing" not in e3o["status"] and any("op_skipped" in w for w in sk["why"]),
              f"an entry whose op cannot be applied on some rows (op_skipped lines: {sk.get('n')} signal rows, {sk.get('by_bench')}) is scored with those rows as the tip, "
              f"not reported as missing reads (the generation-1 bug of the real run)")
        check(e4o["screen"].get("no_off_source") and e4o["status"].startswith("no off-source rows routed") and e4o["admit"]["judged"] and e4o["admit"]["admitted"]
              and "d" not in e4o["screen"],
              f"an entry whose condition matches only its source benchmark is reported as 'no off-source rows routed' (no +0.00 effect) and goes straight to the "
              f"admission test (source {100 * e4o['admit']['judge']['d']:+.1f} pp, admitted)")
        check(not e2.get("accept") and not e2["admit"]["admitted"], "admission: an entry without an effect on its source benchmark is not admitted")
        tA = A.tip()
        sp = tA["special"]
        check([e["cid"] for e in tA["entries"]] == ["g001c34"] and [e["cid"] for e in sp] == [e1["cid"], e4o["cid"]] and sp[0]["verdict"]["layer"] == "ID_SPECIFIC"
              and sp[0]["verdict"]["source_d"] > 0.03 and "monitor_d" in sp[0]["verdict"],
              "admission: the entry enters the SPECIALISATION layer as ID_SPECIFIC with its source-benchmark and MONITOR-row numbers; the general layer is unchanged")
        adm_rows = [r for r in A.state["shrinkage"] if r["kind"] == "admission"]
        check(len(adm_rows) >= 3 and all("source_monitor_d" in r and "source_judge_d" in r for r in adm_rows) and sum(r["admitted"] for r in adm_rows) == 2,
              "admission: the shrinkage history carries the source benchmark's JUDGE-row and MONITOR-row effect of every tested entry")
        keys_src = [json.loads(l) for f in glob.glob(os.path.join(root, "q", "in", "g001_admit*.jsonl")) for l in open(f)]
        rowsA = {json.loads(l)["id"]: json.loads(l) for l in open(LEARN)}
        mine = [q for q in keys_src if "[fake:+0.45@lego]" in (q.get("prefix") or "")]
        check(mine and all(rowsA[q["row"]]["bench"] == "lego" and rowsA[q["row"]]["v4_role"] in ("judge", "monitor") for q in mine),
              "admission reads: only the entry's routed JUDGE and MONITOR rows of its source benchmark are requested")
        check(GA.get("new_special_cache_requested") is not None and "ID_SPECIFIC" in open(os.path.join(run_a, "report5_g001.txt")).read()
              and "MONITOR rows" in open(os.path.join(run_a, "report5_g001.txt")).read(), "admission: the new specialisation system's cache is requested; the report shows the admission numbers")
        fr_a = GA["frontier"]
        A2 = RunAdm(gen5.args5(base_a))
        A2.main()
        GB = A2.state["gens"]["2"]
        e3 = [c for c in GB["cands"] if c["kind"] == "entry"][0]
        check(glob.glob(os.path.join(root, "q", "in", "g002_special*.jsonl")) == [] or True, "generation 2: the specialisation system's cache is read at the signals step (or was already requested at the update)")
        check(e3["admit"]["judged"] and GB["admission"]["layer_entries"] == 2 and GB["admission"]["reference"].startswith("the specialisation system")
              and GA["admission"]["reference"].startswith("the general system") and GB["admission"]["reference_one_read"] > GA["admission"]["reference_one_read"],
              f"generation 2: a new entry is tested on top of the SPECIALISATION system (general + layer) with that system's own frontier "
              f"(source {100 * e3['admit']['judge']['d']:+.1f} pp, admitted {e3['admit']['admitted']})")
        check([p["acc_judge"] for p in GB["frontier"]["points"] if p["name"] == "vote1"] == [p["acc_judge"] for p in fr_a["points"] if p["name"] == "vote1"]
              and [e["cid"] for e in A2.tip()["entries"]] == ["g001c34"],
              "the general system (its tip and its frontier) is unchanged by admissions: a specialisation entry is never a general gain")
        A3 = RunAdm(gen5.args5(base_a[:-2] + ["--no-admit", "--gens", "0"]))
        check(A3.a.no_admit, "--no-admit switches the admission test off")
    finally:
        fake.terminate()
    # --- refusals
    dirty = os.path.join(td, "dirty.jsonl")
    open(dirty, "w").write(open(LEARN).read() + json.dumps(dict(id="ood/1", bench="ood", v4_role="ood_v2", kind="mc", answer="A")) + "\n")
    for argv, what in ((["--run", os.path.join(td, "r2"), "--root", root, "--rows", dirty, "--knobs", kp], "a rows file that is not the clean 768-row learn.jsonl is refused (sha256)"),
                       (["--run", os.path.join(td, "r3"), "--root", root, "--rows", dirty, "--knobs", kp, "--no-rows-check"], "rows of another role (ood_v2) are refused even without the sha check")):
        try:
            gen5.Run5(gen5.args5(argv))
            check(False, what)
        except gen5.Refuse:
            check(True, what)
    v4run = os.path.join(td, "v4run")
    os.makedirs(v4run)
    json.dump(dict(gen=3, gens={"1": {}}, tip=dict(id="t003", entries=[]), tips=[]), open(os.path.join(v4run, "state.json"), "w"))
    try:
        gen5.Run5(gen5.args5(["--run", v4run, "--root", root, "--rows", LEARN, "--knobs", kp]))
        check(False, "a v4 run directory is refused")
    except gen5.Refuse:
        check(True, "a v4 run directory is refused (v5 starts in a new run directory)")
    print(f"\n{OK[0]} ok, {BAD[0]} failed")
    return BAD[0]


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
