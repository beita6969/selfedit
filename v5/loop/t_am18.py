"""AM18 checks (coordinator 11-10, lab fix): the lab reads LAB_SAMPLES samples of the draft on every routed lab row; feedback per row
as fractions; one summary line per group (mean of entry accuracy - tip accuracy, se, n) for shown and hidden rows; no verdict words;
the 99 % / 5-min-after-90 % close rule (fake IO whose last reads never arrive, patched clock); lab_tested True / False / None on the
candidate and in the ledger; P0 prompts unchanged (byte-identical to the pristine base tree when it is there).
usage: python3 t_am18.py   (base tree for the P0 comparison: $AM18_BASE, default /Users/admin/selfedit/code-v4-w_base)"""
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gen                       # noqa: E402
import proposer as PR            # noqa: E402
import lab as LAB                # noqa: E402
import t_lab as TL               # noqa: E402  (InstantIO, make_run, the scripted proposer, run_until)

N = [0, 0]
BASE = os.environ.get("AM18_BASE", "/Users/admin/selfedit/code-v4-w_base")
VERDICT = re.compile(r"\b(fix\w*|broke\w*|currently|improv\w*|worse\w*|better|gain\w*|loss\w*|lost|regress\w*|success\w*|fail\w*|harm\w*|"
                     r"benefit\w*|good|bad|correct\w*|incorrect\w*|verdict\w*)\b|->", re.I)
MEAN = re.compile(r"^Mean of \(accuracy with your entry - accuracy of the tip\) over the (\d+) questions? with samples of both the tip and your entry: "
                  r"([+-]\d\.\d{3})(?:, standard error (\d\.\d{3}) \(sd / sqrt\(n\)\), n = (\d+)\.|; standard error not computed \(n = (\d+); it needs n >= 2\)\.)$", re.M)


def check(name, cond):
    N[0 if cond else 1] += 1
    print(("ok   " if cond else "FAIL ") + name, flush=True)


def fb_of(rec):
    return rec["req"]["messages"][-1]["content"]


def stats(d):
    n = len(d)
    m = sum(d) / n if n else None
    se = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1)) / math.sqrt(n) if n >= 2 else None
    return n, m, se


def raw_row(R, e, rid):
    """(routed, tip samples, entry samples) of one lab row from the raw route / read results."""
    tip = [x for x in (R.io.get(f"p000|parent|{rid}|{s}") for s in range(gen.P_FULL)) if x is not None]
    routed = e["cond"] == "always" or R.io.get(f"rt1|{e['cond']}|{rid}")["p"] >= 0.5
    xs = [x for x in (R.io.get(k) for k in R.lab_keys(e, rid)) if x is not None] if routed else []
    return routed, tip, xs


def other_of(R, rec, P):
    return [i for i in R.role["show"] if R.rows[i]["bench"] == P[rec["slot"]]["card_source"] and i not in rec["shown_ids"]]


# ---------------------------------------------------------------------------------------------------------------------- 1-4
def t_reads_and_feedback():
    root = TL.new_root("am18A")
    R = TL.make_run(root)
    asked = []
    sub = R.io.submit

    def submit(name, reqs):
        if "_labread" in name:
            asked.append((name, list(reqs)))
        return sub(name, reqs)
    R.io.submit = submit
    TL.run_until(R, "validate")
    recs, P = R.lab_load(), TL.props_by_slot(R)
    sent = [(t, s) for (t, s), r in recs.items() if r.get("key")]
    # 1. four samples per routed lab row
    ok = True
    for t in (1, 2):
        got = [q for n, b in asked if n.endswith(f"_labread{t}") for q in b]
        want = []
        for (tt, s), r in recs.items():
            if tt == t and r.get("key"):
                for rid in r["shown_ids"] + other_of(R, r, P):
                    if raw_row(R, r["entry"], rid)[0]:
                        want += R.lab_keys(r["entry"], rid)
        ok &= sorted(q["key"] for q in got) == sorted(want) and all(q["priority"] == 1 and q["sample"] in range(LAB.LAB_SAMPLES) for q in got)
        ok &= all(r["shown"]["asked"] == LAB.LAB_SAMPLES * r["shown"]["routed"] and r["other"]["asked"] == LAB.LAB_SAMPLES * r["other"]["routed"]
                  for (tt, s), r in recs.items() if tt == t and r.get("key"))
    check(f"reads: exactly samples 0..{LAB.LAB_SAMPLES - 1} (priority 1) on every routed lab row, shown and hidden, nothing on rows where the "
          f"condition did not hold ({sum(len(b) for _, b in asked)} lab reads asked over {len(sent)} lab turns)", ok and asked)
    check("reads: LAB_SAMPLES = 4 and the close constants are 0.99 / 0.90 / 300 s", (LAB.LAB_SAMPLES, LAB.CLOSE_ALL, LAB.CLOSE_MOST, LAB.CLOSE_AFTER) == (4, .99, .90, 300.0))
    # 2. per-row fraction lines
    ok, n_rows, n_held = True, 0, 0
    for k in sent:
        r = recs[k]
        fb = fb_of(r)
        blocks = re.findall(r"^### Question (\d+): (.*)\n(.*)$", fb, re.M)
        ok &= [int(b[0]) for b in blocks] == list(range(1, len(r["shown_ids"]) + 1))
        for (q, head, line), rid in zip(blocks, r["shown_ids"]):
            routed, tip, xs = raw_row(R, r["entry"], rid)
            tf = f"Tip right {sum(bool(x['ok']) for x in tip)} of {len(tip)} samples"
            n_rows += 1
            if routed:
                n_held += 1
                ok &= line == tf + f"; with your entry right {sum(bool(x['ok']) for x in xs)} of {len(xs)} sample{'' if len(xs) == 1 else 's'}."
            else:
                ok &= line == tf + ". Your entry has no effect on this question."
    check(f"feedback: every shown question has the fraction line 'Tip right a of b samples; with your entry right c of d samples.' from the raw "
          f"samples ({n_rows} questions, {n_held} where the entry ran)", ok and n_held)
    # 3. summary lines, shown and hidden
    ok, seen = True, 0
    for k in sent:
        r = recs[k]
        fb = fb_of(r)
        ms = MEAN.findall(fb)
        groups = [("shown", r["shown_ids"])] + ([("other", other_of(R, r, P))] if r["n_other"] else [])
        want = []
        for gk, ids in groups:
            d = []
            for rid in ids:
                routed, tip, xs = raw_row(R, r["entry"], rid)
                if routed and tip and xs:
                    d.append(sum(bool(x["ok"]) for x in xs) / len(xs) - sum(bool(x["ok"]) for x in tip) / len(tip))
            if d:
                want.append((gk,) + stats(d))
        got = [(int(a), float(m), float(se) if se else None, int(n or n1)) for a, m, se, n, n1 in ms]
        ok &= len(got) == len(want) and all(g[0] == w[1] == g[3] and abs(g[1] - w[2]) < 5e-4 and ((g[2] is None and w[3] is None) or abs(g[2] - w[3]) < 5e-4)
                                            for g, w in zip(got, want))
        ok &= all(abs(r[w[0]]["mean_diff"] - w[2]) < 1e-12 and r[w[0]]["read"] == w[1] for w in want)
        ok &= "Summary of the shown questions:" in fb and (not r["n_other"] or "not shown to you)\n" in fb)
        seen += len(got)
    check(f"feedback: one summary line per group (shown questions; other questions not shown) with the mean of (entry accuracy - tip accuracy), "
          f"its se (sd / sqrt(n)) and n, equal to the raw numbers ({seen} lines)", ok and seen)
    rows = [dict(routed=True, ps=[dict(ok=i < 3) for i in range(8)], xs=[dict(ok=i < 2) for i in range(4)], asked=4),     # tip 3/8, entry 2/4
            dict(routed=True, ps=[dict(ok=i < 2) for i in range(8)], xs=[dict(ok=True) for i in range(4)], asked=4),      # tip 2/8, entry 4/4
            dict(routed=True, ps=[dict(ok=i < 4) for i in range(8)], xs=[dict(ok=i < 1) for i in range(3)], asked=4),     # tip 4/8, entry 1/3
            dict(routed=True, ps=[dict(ok=True)], xs=[], asked=4), dict(routed=False, ps=[dict(ok=True)], xs=[], asked=0)]
    for z in rows:
        z.update(tip_acc=LAB._acc(z["ps"]), acc=LAB._acc(z["xs"]))
    c = LAB.LabMixin.lab_counts(rows)
    n, m, se = stats([2 / 4 - 3 / 8, 4 / 4 - 2 / 8, 1 / 3 - 4 / 8])
    L = LAB.LabMixin._count_lines(c, False, "shown questions")
    check(f"summary numbers: tip 3/8 vs 2/4, 2/8 vs 4/4, 4/8 vs 1/3 (+ one routed row without samples, one not routed): mean {m:+.4f}, se {se:.4f}, n 3",
          (c["routed"], c["read"], c["missing"], c["partial"], c["asked"], c["samples_in"]) == (4, 3, 1, 1, 16, 11)
          and abs(c["mean_diff"] - m) < 1e-12 and abs(c["se_diff"] - se) < 1e-12
          and L == ["The condition held on 4 of 5 shown questions.",
                    f"Mean of (accuracy with your entry - accuracy of the tip) over the 3 questions with samples of both the tip and your entry: {m:+.3f}, "
                    f"standard error {se:.3f} (sd / sqrt(n)), n = 3.",
                    "Samples with your entry available in time: 11 of 16 (4 per question where your entry ran)."])
    c1 = LAB.LabMixin.lab_counts(rows[:1] + rows[3:])
    c0 = LAB.LabMixin.lab_counts(rows[3:])
    L1 = LAB.LabMixin._count_lines(c1, False, "other questions")
    L0 = LAB.LabMixin._count_lines(c0, True, "other questions")
    check("summary with n < 2: the se is not computed and the line says so; with n = 0 the line says there is no such question",
          L1[1] == "Mean of (accuracy with your entry - accuracy of the tip) over the 1 question with samples of both the tip and your entry: +0.125; "
                   "standard error not computed (n = 1; it needs n >= 2)." and c1["se_diff"] is None
          and L0[:2] == ["Your entry ran on all 2 other questions.",
                         "Mean of (accuracy with your entry - accuracy of the tip): no question with samples of both the tip and your entry."]
          and c0["mean_diff"] is None)
    # 4. no verdict words, no transition counts
    bad = sorted({m.group(0).lower() for k in sent for m in VERDICT.finditer(TL.template_only(fb_of(recs[k])))})
    adv = sorted({m.group(0).lower() for k in sent for m in TL.ADVICE.finditer(TL.template_only(fb_of(recs[k])))})
    check(f"feedback: no verdict words and no transition counts (fixed / broken / currently / ->: {bad}; advice words: {adv})", not bad and not adv)
    check("feedback: a single sample's tails are labelled as one of the samples ('- Sample j with your entry (right|wrong)'), never as the question's result",
          all(len(re.findall(r"^- Sample [1-4] with your entry \((?:right|wrong)\)$", fb_of(recs[k]), re.M)) == fb_of(recs[k]).count("  End of reasoning: ") for k in sent)
          and not any("(one sample)" in fb_of(recs[k]) for k in sent))
    check("stats: read_tokens = the tokens of every entry sample in (all 4 per routed row), tok_entry per sample",
          all(abs(recs[k]["read_tokens"] - sum((R.io.get(q) or {}).get("tokens") or 0 for rid in recs[k]["shown_ids"] + other_of(R, recs[k], P)
                                                for q in (R.lab_keys(recs[k]["entry"], rid) if raw_row(R, recs[k]["entry"], rid)[0] else []))) < 1e-6
              for k in sent))
    ex = os.path.join(root, "example_am18_feedback.txt")
    with open(ex, "w") as f:
        f.write(fb_of(recs[(1, 0)]) + "\n")
    print(f"     example feedback (slot 0, turn 1): {ex}")
    return R


# ---------------------------------------------------------------------------------------------------------------------- 5
class Clock:
    def __init__(self, t):
        self.t = float(t)

    def now(self):
        return self.t

    def sleep(self, s):
        self.t += s


class SlowIO:
    """Results arrive at given clock times (None = never); keys in errors get an error line instead of a result."""

    def __init__(self, clock, arrivals, errors=()):
        self.clock, self.arr, self.errors, self.res, self.err = clock, arrivals, set(errors), {}, {}

    def poll(self):
        for k, t in self.arr.items():
            if t is not None and t <= self.clock.t and k not in self.res and k not in self.err:
                if k in self.errors:
                    self.err[k] = dict(key=k, error="engine error")
                else:
                    self.res[k] = dict(key=k, ok=True)
        return 0

    def get(self, k):
        return self.res.get(k)


def patched(clock):
    LAB._now, LAB._sleep = clock.now, clock.sleep


def t_close_rule():
    real = (LAB._now, LAB._sleep)
    try:
        t0 = 1000.0
        keys = [f"k{i}" for i in range(100)]
        # A: 90 % in at t0 + 100 (keys 0..89 at t0 + 10 + i), 5 more at t0 + 200, 5 never -> closes at t0 + 400 (5 min after 90 %)
        arr = {k: t0 + 10 + i if i < 90 else (t0 + 200 if i < 95 else None) for i, k in enumerate(keys)}
        ck = Clock(t0)
        patched(ck)
        done, miss, info = LAB.wait_close(SlowIO(ck, arr), keys, deadline=t0 + 3600, every=5.0, log=lambda *a: None)
        check(f"close: last reads never arrive -> closes 5 min after 90 % are in ({info})",
              ck.t == t0 + 400 and info["why"] == "300 s after 90% in" and info["most_in_s"] == 100 and len(done) == 95 and miss == keys[95:] and info["k_in"] == 95)
        # B: 99 % in at t0 + 50 -> closes at once, without the 5 minutes
        arr = {k: t0 + min(i, 50) if i < 99 else None for i, k in enumerate(keys)}
        ck = Clock(t0)
        patched(ck)
        done, miss, info = LAB.wait_close(SlowIO(ck, arr), keys, deadline=t0 + 3600, log=lambda *a: None)
        check("close: 99 % in -> closes then (first poll with 99 in)", ck.t == t0 + 50 and info["why"] == "99% in" and len(done) == 99 and miss == ["k99"])
        # C: the deadline comes before the 5 minutes
        arr = {k: t0 + 10 + i if i < 90 else (t0 + 200 if i < 95 else None) for i, k in enumerate(keys)}
        ck = Clock(t0)
        patched(ck)
        done, miss, info = LAB.wait_close(SlowIO(ck, arr), keys, deadline=t0 + 250, log=lambda *a: None)
        check("close: never later than the deadline (deadline t0 + 250 before t0 + 400)", ck.t == t0 + 250 and info["why"] == "deadline" and len(done) == 95)
        # D: error lines count as in (95 results + 4 errors = 99 in), but are not results
        arr = {k: t0 + 20 if i < 99 else None for i, k in enumerate(keys)}
        ck = Clock(t0)
        patched(ck)
        done, miss, info = LAB.wait_close(SlowIO(ck, arr, errors=keys[95:99]), keys, deadline=t0 + 3600, log=lambda *a: None)
        check("close: a read with an error line counts as in (95 results + 4 errors -> 99 % at t0 + 20); done lists results only",
              ck.t == t0 + 20 and info["why"] == "99% in" and info["errors"] == 4 and done == keys[:95] and miss == ["k99"])
        # E: 90 % never reached -> waits to the deadline
        arr = {k: t0 + 5 if i < 89 else None for i, k in enumerate(keys)}
        ck = Clock(t0)
        patched(ck)
        done, miss, info = LAB.wait_close(SlowIO(ck, arr), keys, deadline=t0 + 1000, log=lambda *a: None)
        check("close: below 90 % the step waits for the deadline (89 of 100)", ck.t == t0 + 1000 and info["why"] == "deadline" and len(miss) == 11)
        # F: all in at once, and no keys
        ck = Clock(t0)
        patched(ck)
        d1, m1, i1 = LAB.wait_close(SlowIO(ck, {k: t0 for k in keys}), keys, deadline=t0 + 10, log=lambda *a: None)
        d2, m2, i2 = LAB.wait_close(SlowIO(ck, {}), [], deadline=t0 + 10, log=lambda *a: None)
        check("close: all in -> returns at once; no keys -> returns at once", ck.t == t0 and i1["why"] == "all in" and len(d1) == 100 and i2["why"] == "all in" and not d2 + m2)
        # G: inside the lab step: the last reads of turn 1 never arrive -> the step closes 5 min after 90 %, and the feedback shows what it has
        root = TL.new_root("am18close")
        io = TL.InstantIO(root, TL.script)
        R = TL.make_run(root, io=io)
        drop = set()
        sub = io.submit

        def submit(name, reqs):
            if name.endswith("_labread1"):
                card1 = set(TL.props_by_slot(R)[1]["card"])
                sg1 = R.sig(None, R.settings_of([dict(settings={"temperature": 0.8})]))
                groups = list(dict.fromkeys(q["key"].rsplit("|", 1)[0] for q in reqs))
                mine = [g for g in groups if g.split("|")[1] == sg1 and g.split("|")[2] in card1]
                for i, g in enumerate(groups):
                    if i % 6 == 0:
                        drop.add(g + "|3")
                drop.update(f"{mine[0]}|{s}" for s in range(4))                          # one shown question: no sample in
                drop.update({f"{mine[1]}|3", f"{mine[2]}|2", f"{mine[2]}|3"})          # 3 of 4 in, 2 of 4 in
                drop.discard(f"{mine[3]}|3")
                n = sub(name, reqs)
                for k in drop:
                    io.res.pop(k, None)
                return n
            return sub(name, reqs)
        io.submit = submit
        ck = Clock(time.time())
        patched(ck)
        start = ck.t
        TL.run_until(R, "validate")
        recs = R.lab_load()
        r1 = recs[(1, 1)]
        fb = fb_of(r1)
        cl = r1["close"]
        n1 = cl["n"]
        check(f"close in the lab step: turn-1 reads with {len(drop)} of {n1} never arriving close 5 min after 90 % ({cl}); turn 2 all in",
              cl["why"] == "300 s after 90% in" and cl["k_in"] == n1 - len(drop) and 0.9 * n1 <= cl["k_in"] < 0.99 * n1 and abs(ck.t - start - 300) < 1e-6
              and recs[(2, 0)]["close"]["why"] == "all in")
        check("close in the lab step: a shown question with 3 of 4 samples in says 'right c of 3 samples. (1 of the 4 samples not available in time.)'",
              re.search(r"^Tip right \d of 8 samples; with your entry right \d of 3 samples\. \(1 of the 4 samples not available in time\.\)$", fb, re.M) is not None)
        check("close in the lab step: 2 of 4 in -> 'right c of 2 samples. (2 of the 4 ...)'; none in -> 'with your entry: no sample was available in time.'",
              re.search(r"; with your entry right \d of 2 samples\. \(2 of the 4 samples not available in time\.\)$", fb, re.M) is not None
              and re.search(r"^Tip right \d of 8 samples; with your entry: no sample was available in time\.$", fb, re.M) is not None)
        sh = r1["shown"]
        check(f"close in the lab step: the summary counts what is in (shown: {sh['samples_in']} of {sh['asked']} samples, mean over {sh['read']} of "
              f"{sh['routed']} questions) and says how many samples were not in time",
              sh["read"] == sh["routed"] - 1 and sh["samples_in"] < sh["asked"]
              and f"Samples with your entry available in time: {sh['samples_in']} of {sh['asked']} (4 per question where your entry ran)." in fb
              and f"over the {sh['read']} questions with samples of both" in fb)
    finally:
        LAB._now, LAB._sleep = real


# ---------------------------------------------------------------------------------------------------------------------- 6
def t_lab_tested():
    root = TL.new_root("am18T")
    R = TL.make_run(root)
    R.generation(1)
    G = R.G()
    C = {c["slot"]: c for c in G["cands"] if c["author"] == "P0"}
    led = {}
    for l in open(os.path.join(R.rd, "ledger.jsonl")):
        x = json.loads(l)
        led[x["id"]] = x
    want = {0: False, 1: True, 2: False, 3: None, 4: None, 5: False}
    got = {s: C[s].get("lab_tested") for s in range(6)}
    check(f"lab_tested: slot 0 revised at both turns -> False; slot 1 kept -> True; slot 2 replaced after turn 1 -> False; slot 3 no lab turn ran "
          f"(draft did not parse) -> None; slot 4 lab_turns 0 -> None; slot 5 broken final answer repaired into a new entry -> False ({got})", got == want)
    check("lab_tested: B candidates None; every ledger line has the key, with the candidate's value",
          all(c.get("lab_tested") is None for c in G["cands"] if c["author"] == "B")
          and all("lab_tested" in led[c["cid"]] and led[c["cid"]]["lab_tested"] == c.get("lab_tested") for c in G["cands"]))
    check("lab_tested: lab_turns stays what the slot says, no lab turn after the last one (slot 0: turns 1 and 2 only)",
          C[0]["lab_turns"] == 2 and sorted(t for t, s in R.lab_load() if s == 0) == [1, 2] and not any(t > 2 for t, _ in R.lab_load()))

    def chat(q):                                                    # slot 0 goes back to its draft at turn 2; slot 5 keeps its draft at turn 1
        s, t = TL.slot_turn(q["key"])
        if (s, t) == (0, 2):
            return TL.P0_ANS[0]
        if (s, t) == (5, 1):
            return TL.P0_ANS[5]
        return TL.script(q)
    root = TL.new_root("am18T2")
    R2 = TL.make_run(root, chat_fn=chat)
    R2.generation(1)
    C2 = {c["slot"]: c for c in R2.G()["cands"] if c["author"] == "P0"}
    check("lab_tested: slot 5 keeps its draft after one lab turn -> True; slot 0 returns at turn 2 to its turn-1-tested draft -> True (an entry a lab turn ran)",
          C2[5].get("lab_tested") is True and C2[0].get("lab_tested") is True and C2[2].get("lab_tested") is False)
    p = dict(slot=9, lab_turns=0)
    check("lab_tested: direct calls (lab_turns 0 -> None; lab_turns 1 without a lab record -> None; cand None after a lab turn -> False)",
          R.lab_tested(p, {"condition": "always"}, {}) is None and R.lab_tested(dict(slot=9, lab_turns=1), {"condition": "always"}, {}) is None
          and R.lab_tested(dict(slot=1, lab_turns=2), None, R.lab_load()) is False)


# ---------------------------------------------------------------------------------------------------------------------- 7
CODE = """
import json, os, sys
sys.path.insert(0, os.getcwd())
import t_lab as TL, t_versions as TV
root_a, root_b = sys.argv[1], sys.argv[2]
R = TL.make_run(root_a)
TL.run_until(R, "lab")
B = TV.make(root_b)
TV.run_to(B, "lab")
"""


def p0_prompts(loopdir, tag):
    ra, rb = tempfile.mkdtemp(prefix=f"t_am18_{tag}_s1_"), tempfile.mkdtemp(prefix=f"t_am18_{tag}_v_")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    out = subprocess.run([sys.executable, "-B", "-c", CODE, ra, rb], cwd=loopdir, env=env, capture_output=True, text=True)
    if out.returncode != 0:
        print(out.stdout[-2000:], out.stderr[-2000:])
        return None
    return [open(os.path.join(r, "run", "prompts_g001.jsonl"), "rb").read() for r in (ra, rb)]


def t_p0():
    mine = p0_prompts(HERE, "here")
    check("P0: prompts file written on the Stage-1 path and on the versions path", mine is not None and all(mine))
    bl = os.path.join(BASE, "loop")
    if os.path.isdir(bl) and os.path.abspath(bl) != HERE:
        base = p0_prompts(bl, "base")
        check(f"P0: prompts byte-identical to the base tree {BASE} (Stage-1 path {len(mine[0])} bytes, versions path {len(mine[1])} bytes)",
              base is not None and base == mine)
    else:
        print(f"     (skipped: no base tree at {BASE} for the byte comparison; t_versions.py's P0-identity test still applies)")
    lines = [json.loads(l) for l in mine[1].decode().splitlines()]
    check("P0: no lab text in any P0 request (no 'Trial of your entry', no 'samples; with your entry')",
          lines and not any("Trial of your entry" in json.dumps(q) or "with your entry right" in json.dumps(q) for q in lines))


def main():
    t_reads_and_feedback()
    t_close_rule()
    t_lab_tested()
    t_p0()
    print(f"\n{N[0]} ok, {N[1]} failed")
    sys.exit(1 if N[1] else 0)


if __name__ == "__main__":
    main()
