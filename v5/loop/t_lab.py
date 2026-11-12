"""Tests of lab.py (proposer lab, S2-A) on the real learn rows. Run: python3 t_lab.py -> ALL GREEN.
Unit scenarios use an in-memory engine (fake_dispatch's simulator answering at submit time, scripted proposer answers); the last
part is an end-to-end dry run through the fake engine process (fake_dispatch.py) with lab_dry.LabRun, lab_turns 1 and 2, two
generations. Work dirs are fresh temp dirs (never deleted by this script)."""
import argparse
import collections
import json
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
import fake_dispatch as FD       # noqa: E402
import lab as LAB                # noqa: E402
from lab_dry import LabRun       # noqa: E402

ROWS = os.path.join(HERE, "..", "data", "learn.jsonl")
KNOBS = os.path.join(HERE, "knobs.json")
ADVICE = re.compile(r"\b(should|tends?|better|worse|lessons?|hints?|recommend\w*|suggest\w*|consider|works?|effective|improv\w*|"
                    r"help\w*|hurt\w*|avoid|prefer\w*|best|good|bad)\b", re.I)


def check(name, cond):
    if not cond:
        raise AssertionError(name)
    print("ok  ", name)


def J(cond, proc=None, settings=None, why="w"):
    return "Reasoning.\n```json\n" + json.dumps(dict(why=why, condition=cond, procedure=proc, settings=settings or {})) + "\n```"


C0 = "The question asks to count objects in the image."
LONG = "The question " + "and the image " * 20 + "show objects."
BROKEN = '```json\n{"why": "x", "condition": "always",, }\n```'
P0_ANS = {0: J(C0, "List every item you can see before answering."), 1: J("always", None, {"temperature": 0.8}),
          2: J(LONG, "Describe the objects first."), 3: BROKEN, 4: J("always", None, {"max_image_tokens": 8192}),
          5: J("The image contains a chart or a plot with labelled axes.", "Read both axes first.")}
LAB_ANS = {(0, 1): J(C0, "Name each item you can see, then answer."),
           (0, 2): J("The image contains small text or numbers that must be read.", "Read every label first, then answer."),
           (1, 1): J("always", None, {"temperature": 0.8}),                                  # kept unchanged
           (2, 1): J("The image shows several objects.", "Describe the objects first."),    # the too-long condition fixed
           (5, 1): BROKEN}                                                                   # a broken final answer -> repair round
PATTERN = (2, 2, 1, 1, 0, 1)        # lab_turns by slot


def slot_turn(key):
    return int(key.split("|")[4]), key.count("|lab")


def script(q):
    s, t = slot_turn(q["key"])
    if q["key"].endswith("|fix1"):
        return J("The image shows a table.", "Read the header row first.")
    return P0_ANS[s] if t == 0 else LAB_ANS.get((s, t), J("always", None, {"min_image_tokens": 2048}))


class InstantIO:
    """In-memory engine: every request is answered at submit time (fake_dispatch.Fake.handle for read / route, chat_fn for chat;
    chat_fn returning None = never answered). Records each batch (name, new requests)."""

    def __init__(self, root, chat_fn, read_hook=None):
        a = argparse.Namespace(root=root, rows=ROWS, knobs=KNOBS, rate=0, jitter=0.0, effect_scale=1.0, once=True, max_results=None, quiet=True)
        self.fake, self.chat_fn, self.read_hook = FD.Fake(a), chat_fn, read_hook
        self.res, self.err, self.batches, self.fail_on = {}, {}, [], None

    def submit(self, name, reqs):
        if self.fail_on and self.fail_on in name:
            raise RuntimeError(f"simulated crash at submit {name}")
        new, seen = [], set()
        for q in reqs:
            if q["key"] not in self.res and q["key"] not in seen:
                seen.add(q["key"])
                new.append(q)
        self.batches.append((name, new))
        for q in new:
            if q["mode"] == "chat":
                c = self.chat_fn(q)
                if c is None:
                    continue
                rec = dict(key=q["key"], mode="chat", content=c, reasoning="r", finish="stop", tokens=1000 + len(c))
            else:
                rec = self.fake.handle(q)
                if q["mode"] == "read" and self.read_hook:
                    rec = self.read_hook(q, rec)
            self.res[q["key"]] = rec
        return len(new)

    def get(self, k):
        return self.res.get(k)

    def have(self, k):
        return k in self.res

    def wait(self, keys, deadline=None, every=5.0, label="", tick=300):
        keys = list(dict.fromkeys(keys))
        return [k for k in keys if k in self.res], [k for k in keys if k not in self.res]

    def tok_rate(self):
        return None

    def poll(self):
        return 0

    def named(self, part):
        return [(n, b) for n, b in self.batches if part in n]


def new_root(tag):
    return tempfile.mkdtemp(prefix=f"t_lab_{tag}_")


def make_run(root, io=None, chat_fn=script, read_hook=None, slots=6, pattern=PATTERN):
    a = gen.args(["--root", root, "--rows", ROWS, "--run", os.path.join(root, "run"), "--slots", str(slots), "--box", "3600",
                  "--no-versions", "--no-w", "--no-stack"])          # the lab on its own (Stage 1 propose path); versions are tested in t_versions
    R = LabRun(a, lab_pattern=pattern)
    R.io = io or InstantIO(root, chat_fn, read_hook)
    return R


def run_until(R, stop, g=1):
    R.a.stop_after = stop
    try:
        R.generation(g)
    except SystemExit as e:
        assert e.code == 3, e.code
    R.a.stop_after = None


def props_by_slot(R):
    return {p["slot"]: p for p in R.G()["props"] if p["author"] == "P0"}


def user_of(msgs):
    return next(m["content"] for m in msgs if m["role"] == "user")


def template_only(text):
    return "\n".join(l for l in text.split("\n") if not l.startswith(("  End of reasoning:", "  Final text:")))


# ---------------------------------------------------------------------------------------------------------------------- unit
def scenario_main():
    root = new_root("A")
    R = make_run(root)
    run_until(R, "validate")
    G, P, io = R.G(), props_by_slot(R), R.io
    recs = R.lab_load()
    check("steps: lab runs between propose and validate", G["done"] == ["parent", "propose", "lab"] and gen.STEPS.index("lab") == gen.STEPS.index("propose") + 1)
    check("lab_turns set by slot (2, 2, 1, 1, 0, 1)", [P[s].get("lab_turns") for s in range(6)] == list(PATTERN))

    # slot outcomes
    check("slot 0 (lab_turns 2, revised twice): turns 1 and 2 sent, key_final = turn-2 key",
          recs[(1, 0)].get("key") and recs[(2, 0)].get("key") and P[0]["key_final"] == recs[(2, 0)]["key"])
    check("slot 1 (lab_turns 2, kept): turn 2 not sent because the entry is unchanged; key_final = turn-1 key",
          recs[(2, 1)].get("skip", "").startswith("entry kept unchanged") and P[1]["key_final"] == recs[(1, 1)]["key"])
    check("slot 2 (repairable-only condition length): lab runs on the entry as written and shows the validator message",
          recs[(1, 2)].get("key") and "Your entry as written cannot be judged: condition length" in user_of(recs[(1, 2)]["req"]["messages"][-1:])
          and P[2]["lab"][0]["ran_invalid"] and P[2]["lab"][0]["answer"] == "valid")
    check("slot 3 (answer does not parse): no lab, reason recorded, final_key = P0 key",
          recs[(1, 3)]["skip"].startswith("answer does not parse: bad_json") and "key_final" not in P[3] and LAB.final_key(P[3]) == P[3]["key"]
          and P[3]["lab"] == [dict(turn=1, skip=recs[(1, 3)]["skip"])])
    check("slot 4 (lab_turns 0): untouched (no lab record, no key_final)", "lab" not in P[4] and "key_final" not in P[4] and not any(k[1] == 4 for k in recs))
    check("B props: untouched", all("lab" not in p and "key_final" not in p for p in G["props"] if p["author"] == "B"))
    check("p['lab'] per turn: rows routed, right before/after counts, tokens",
          all(set(("turn", "key", "shown", "other", "read_tokens", "answer_tokens", "answered")) <= set(t) for t in P[0]["lab"])
          and P[0]["lab"][1]["turn"] == 2 and G["lab"]["final_from_lab"] == 4)

    # keys
    ok_keys = True
    for (t, s), r in recs.items():
        if not r.get("key"):
            continue
        msgs = r["req"]["messages"]
        prev = P[s]["key"] if t == 1 else recs[(t - 1, s)]["key"]
        prev_msgs = R.lab_p0_requests()[prev]["messages"] if t == 1 else recs[(t - 1, s)]["req"]["messages"]
        ok_keys &= r["key"] == f"{prev}|lab{t}|{LAB._sha16(json.dumps(msgs, ensure_ascii=False, sort_keys=True))[:8]}"
        ok_keys &= msgs[:len(prev_msgs)] == prev_msgs and msgs[-2] == dict(role="assistant", content=io.get(prev)["content"]) and msgs[-1]["role"] == "user"
        ok_keys &= r["req"]["key"] == r["key"] and r["req"]["mode"] == "chat" and r["req"]["max_tokens"] == gen.PROP_MAX and r["req"]["temperature"] == gen.PROP_TEMP
    check("keys: <previous key>|lab<t>|<sha16(messages)[:8]>; messages = previous messages + assistant(previous answer) + user(feedback)", ok_keys)
    check("keys: turn-2 key extends the turn-1 key", recs[(2, 0)]["key"].startswith(recs[(1, 0)]["key"] + "|lab2|"))
    check("keys: all lab keys distinct", len({r["key"] for r in recs.values() if r.get("key")}) == sum(1 for r in recs.values() if r.get("key")))

    # batching and rows
    for t in (1, 2):
        nb = [len(io.named(f"g001_lab{w}{t}")) for w in ("route", "read", "chat")]
        check(f"batching: lab turn {t} sends one route batch, one read batch, one chat batch for all slots ({nb})", nb == [1, 1, 1])
    labq = [q for n, b in io.batches if "_labroute" in n or "_labread" in n for q in b]
    check(f"lab rows: all {len(labq)} lab routes / reads are on SHOW rows", labq and all(R.rows[q["row"]]["v4_role"] == "show" for q in labq))
    src_ok = True
    for (t, s), r in recs.items():
        if r.get("key"):
            src = P[s]["card_source"]
            src_ok &= all(R.rows[i]["bench"] == src for i in r["shown_ids"]) and r["n_other"] == 30 - len(r["shown_ids"])
    check("lab rows: the card's shown questions + the other SHOW rows of the same source (30 per source)", src_ok)
    lr = [q for n, b in io.batches if "_labread" in n for q in b]
    per = collections.Counter((q["key"].rsplit("|", 1)[0]) for q in lr)
    check(f"lab reads (AM18): samples 0..{LAB.LAB_SAMPLES - 1}, priority 1, {LAB.LAB_SAMPLES} per (view, row)",
          lr and all(q["sample"] in range(LAB.LAB_SAMPLES) and q["priority"] == 1 for q in lr) and set(per.values()) == {LAB.LAB_SAMPLES})

    # feedback content
    r10 = recs[(1, 0)]
    fb = r10["req"]["messages"][-1]["content"]
    card_user = user_of(R.lab_p0_requests()[P[0]["key"]]["messages"])
    hc = re.findall(r"^### Question (\d+): currently (?:right|wrong) \((\d+) of (\d+) samples right\)", card_user, re.M)
    hf = re.findall(r"^### Question (\d+): .*\nTip right (\d+) of (\d+) samples", fb, re.M)
    check(f"numbering: Question 1..{len(hc)} in card order, each with the same tip samples as on the card", hc and hc == hf
          and [int(k) for k, *_ in hf] == list(range(1, len(P[0]["card"]) + 1)))
    check("numbering: lab_numbering of a P0 card = card ids in order", R.lab_numbering(P[0]["card"], R.lab_p0_requests()[P[0]["key"]]["messages"])
          == list(enumerate(P[0]["card"], 1)))
    e = r10["entry"]
    other = [i for i in R.role["show"] if R.rows[i]["bench"] == P[0]["card_source"] and i not in r10["shown_ids"]]
    par = {i: PR._samples({i: [io.get(f"p000|parent|{i}|{s}") for s in range(gen.P_FULL)]}, i) for i in r10["shown_ids"] + other}

    def counts(ids):
        routed, d = 0, []
        for i in ids:
            rt = io.get(f"rt1|{e['cond']}|{i}")
            if rt["p"] >= 0.5:
                routed += 1
                xs = [io.get(f"x|{R.sig(e['procedure'], R.settings_of([e]))}|{i}|{s}") for s in range(LAB.LAB_SAMPLES)]
                xs = [x for x in xs if x is not None]
                if xs:
                    d.append(sum(bool(x["ok"]) for x in xs) / len(xs) - sum(bool(v["ok"]) for v in par[i]) / len(par[i]))
        m = sum(d) / len(d) if d else None
        se = (sum((x - m) ** 2 for x in d) / (len(d) - 1)) ** 0.5 / len(d) ** 0.5 if len(d) >= 2 else None
        return routed, len(d), m, se

    def same(a, c):
        return a[:2] == (c["routed"], c["read"]) and all((x is None and y is None) or (x is not None and y is not None and abs(x - y) < 1e-12)
                                                          for x, y in zip(a[2:], (c["mean_diff"], c["se_diff"])))
    check("counts: shown and other numbers (routed, rows of the mean, mean difference, se) recomputed from the raw route / read / parent results",
          same(counts(r10["shown_ids"]), r10["shown"]) and same(counts(other), r10["other"]))
    n_routed = sum(1 for l in fb.split("\n") if l.startswith("### Question") and "the condition held (p = " in l)
    check(f"feedback: every routed shown question has the executor's reasoning and final text ({n_routed} routed)",
          fb.count("  End of reasoning: ") == n_routed == r10["shown"]["routed"] and fb.count("  Final text: ") == n_routed)
    check("feedback: a question where the condition did not hold says the entry has no effect",
          fb.count("Your entry has no effect on this question.") == len(r10["shown_ids"]) - n_routed)
    sh = {PR._ws(R.rows[i]["prompt"]) for i in r10["shown_ids"]}
    leaks = [i for i in other if len(PR._ws(R.rows[i]["prompt"])) >= 16 and PR._ws(R.rows[i]["prompt"]) not in sh
             and PR._ws(R.rows[i]["prompt"]) in PR._ws(json.dumps(r10["req"]["messages"]))]
    check("feedback: other SHOW rows appear as aggregates only (none of their question texts in the request)", not leaks)
    check("feedback: tails as on the card (proposer._tail, REASON_TAIL / CONTENT_TAIL)",
          all(len(l) <= len("  End of reasoning: ") + PR.REASON_TAIL + 3 for l in fb.split("\n") if l.startswith("  End of reasoning: ")))
    adv = sorted({m.group(0).lower() for rr in recs.values() if rr.get("key") for m in ADVICE.finditer(template_only(rr["req"]["messages"][-1]["content"]))})
    check(f"feedback: data, not advice (no advice words in the fixed text: {adv})", not adv)
    check("feedback: ends with the request to keep / revise / replace and the same answer format",
          all(rr["req"]["messages"][-1]["content"].endswith(PR.ANSWER_FORMAT) and "Keep your entry, revise it or replace it" in rr["req"]["messages"][-1]["content"]
              for rr in recs.values() if rr.get("key")))
    check("feedback: a non-last turn says one more trial follows; the last turn says its entry is judged",
          "tried in the same way once more" in fb and "tried in the same way" not in recs[(2, 0)]["req"]["messages"][-1]["content"]
          and "The entry in your next answer is the one that is judged." in recs[(1, 2)]["req"]["messages"][-1]["content"])
    fa = recs[(1, 1)]["req"]["messages"][-1]["content"]
    check("feedback: an 'always' entry runs on every question (no router p shown)",
          'Your condition is "always"' in fa and "p = " not in fa and recs[(1, 1)]["shown"]["routed"] == len(recs[(1, 1)]["shown_ids"])
          and recs[(1, 1)]["other"]["routed"] == recs[(1, 1)]["n_other"])
    gi = [PR.guard_messages(rr["req"]["messages"], R.rows) for rr in recs.values() if rr.get("key")]
    check(f"guard: every lab request passes proposer.guard_messages ({len(gi)} requests: slot 0 x2, slots 1, 2, 5)", len(gi) == 5)
    check("guard: no JUDGE / MONITOR row id in any lab request",
          not any(PR._ids_in(json.dumps(rr["req"]), {i for i, x in R.rows.items() if x["v4_role"] != "show"}) for rr in recs.values() if rr.get("key")))

    # resume: re-entry after the step reuses the file, sends nothing new
    before = open(R.lab_file()).read()
    nb0 = len(io.batches)
    R2 = make_run(root, io=io)
    R2.g, R2.deadline = 1, time.time() + 3600
    R2.state["gens"]["1"]["done"].remove("lab")
    R2.step_lab()
    P2 = props_by_slot(R2)
    check("resume: a second step_lab sends no new request and gives the same key_final and lab records",
          sum(len(b) for _, b in io.batches[nb0:]) == 0 and open(R.lab_file()).read() == before
          and all(P2[s].get("key_final") == P[s].get("key_final") and P2[s].get("lab") == P[s].get("lab") for s in P))
    R2.state["gens"]["1"]["done"].append("validate")
    nb0 = len(io.batches)
    R2.step_lab()
    check("resume: step_lab is a no-op once validate has run", len(io.batches) == nb0)

    unit_small(R)                     # (still on the stock tip: the lab rows of a slot need tip samples)

    # continue: validate reads final_key(p); a broken final lab answer gets the repair turn with the lab messages
    R.generation(1)
    G = R.G()
    C = {c["slot"]: c for c in G["cands"] if c["author"] == "P0"}
    fin = {s: PR.parse(io.get(LAB.final_key(P[s]))["content"])[0] for s in (0, 1, 2)}
    check("validate: candidates of lab slots carry the final turn's entry",
          all(C[s]["entry"]["condition"] == fin[s]["condition"] and C[s]["entry"]["procedure"] == fin[s]["procedure"] for s in (0, 1, 2))
          and C[0]["key_final"] == recs[(2, 0)]["key"] and C[0]["lab"][0]["turn"] == 1)
    check("validate: p['key'] is restored after the call (state keeps the P0 key)", all(p["key"].count("|lab") == 0 for p in G["props"] if p.get("key")))
    p5 = props_by_slot(R)[5]
    q5 = io.get(p5["repair_key"]) and [q for n, b in io.batches if "chatfix" in n for q in b if q["key"] == p5["repair_key"]][0]
    check("repair: a broken final lab answer gets one follow-up turn built on the lab messages (key <lab key>|fix1)",
          p5["repair_key"] == recs[(1, 5)]["key"] + "|fix1" and q5["messages"][:len(recs[(1, 5)]["req"]["messages"])] == recs[(1, 5)]["req"]["messages"]
          and C[5].get("repaired") is True)
    check("repair: a broken P0 answer without a lab still gets its usual repair turn", props_by_slot(R)[3]["repair_key"] == P[3]["key"] + "|fix1")
    check("generation finishes with the lab step", G["done"] == gen.STEPS and os.path.exists(os.path.join(R.rd, "report_g001.json")))
    return R


def scenario_crash():
    root = new_root("B")
    io = InstantIO(root, script)
    io.fail_on = "_labchat1"
    R = make_run(root, io=io)
    try:
        run_until(R, "validate")
        crashed = False
    except RuntimeError:
        crashed = True
    lines1 = [l for l in open(R.lab_file()) if json.loads(l)["turn"] == 1]
    check("crash: the turn-1 requests are on disk before the chat submit that crashed", crashed and len(lines1) == 5)
    P = props_by_slot(R)
    rid = P[0]["card"][0]
    k0 = f"p000|parent|{rid}|0"
    io.res[k0] = dict(io.res[k0], ok=not io.res[k0]["ok"])            # the tip's samples change between the crash and the restart
    io.fail_on = None
    R2 = make_run(root, io=io)
    run_until(R2, "validate")
    rec10 = json.loads(next(l for l in lines1 if json.loads(l)["slot"] == 0))
    fresh, _ = R2.lab_feedback(rec10["entry"], list(enumerate(rec10["shown_ids"], 1)), [i for i in R2.role["show"] if R2.rows[i]["bench"] == P[0]["card_source"]
                                                                                       and i not in rec10["shown_ids"]], rec10["sources"], rec10["note"], False)
    check("crash: a rebuilt turn-1 feedback would differ now (the test is live)", fresh != rec10["req"]["messages"][-1]["content"])
    after1 = [l for l in open(R2.lab_file()) if json.loads(l)["turn"] == 1]
    sent = [q["key"] for n, b in io.named("_labchat1") for q in b]
    check("crash: on restart the stored turn-1 requests are reused byte for byte and exactly those keys are submitted",
          after1 == lines1 and sorted(sent) == sorted(json.loads(l)["key"] for l in lines1 if json.loads(l).get("key")))
    check("crash: the restarted lab completes turn 2", R2.lab_load().get((2, 0), {}).get("key", "").startswith(json.loads(lines1[0])["key"] if json.loads(lines1[0])["slot"] == 0 else rec10["key"]))


def scenario_guard():
    root = new_root("C")
    rows = PR.load_rows(ROWS)
    showt = " ".join(PR._ws(r["prompt"]) for r in rows.values() if r["v4_role"] == "show")
    jt = next(PR._ws(r["prompt"]) for i, r in sorted(rows.items()) if r["v4_role"] == "judge" and 40 <= len(PR._ws(r["prompt"])) <= 400
              and PR._ws(r["prompt"]) not in showt)

    def hook(q, rec):
        if q["key"].startswith("x|"):
            rec = dict(rec, reasoning="filler words " * 30 + jt)
        return rec
    R = make_run(root, read_hook=hook)
    run_until(R, "validate")
    recs, P = R.lab_load(), props_by_slot(R)
    check("guard: a lab feedback carrying a JUDGE row's text is not sent (slot skipped, reason recorded, P0 answer kept)",
          recs[(1, 1)]["skip"].startswith("guard: question text of non-show row") and "key_final" not in P[1])
    check("guard: no chat request of the lab carries the JUDGE text", not any(jt in PR._ws(json.dumps(q["messages"])) for n, b in R.io.named("_labchat") for q in b))


def scenario_unanswered_and_context():
    root = new_root("D")

    def chat(q):
        s, t = slot_turn(q["key"])
        return None if (s, t) == (0, 1) else script(q)
    R = make_run(root, chat_fn=chat)
    run_until(R, "validate")
    recs, P = R.lab_load(), props_by_slot(R)
    check("unanswered: a lab turn without an answer by the deadline leaves the slot on its previous answer",
          "key_final" not in P[0] and P[0]["lab"][0]["answered"] is False and recs[(2, 0)]["skip"] == "no answer to the previous turn")
    root = new_root("E")
    R = make_run(root)
    R.LAB_CTX = 4000
    run_until(R, "validate")
    recs = R.lab_load()
    check("context: a lab turn whose prompt + max_tokens would not fit the engine is not sent",
          all(r["skip"].startswith("context: ") for (t, s), r in recs.items() if t == 1 and s in (0, 1, 2, 5)) and not any(b for _, b in R.io.named("_labchat")))


def unit_small(R):
    rows = R.rows
    check("final_key: key_final when set, else key", LAB.final_key({"key": "a"}) == "a" and LAB.final_key({"key": "a", "key_final": "b"}) == "b"
          and LabRun.final_key({"key": "a", "key_final": "b"}) == "b")
    tv = R.tip_view()
    e, note, why = R.lab_entry(dict(content=J("", "Do this.")), tv, None)
    check("lab_entry: an empty condition gives no entry (skip)", e is None and why)
    e, note, why = R.lab_entry(dict(content=J("always", None, {"temperature": 0.7})), tv, None)
    check("lab_entry: a non-repairable failure (value off the menu) gives no entry", e is None and "not one of" in why)
    e, note, why = R.lab_entry(dict(content=J("The image shows a table.", "Read the header row first.")), tv, None)
    check("lab_entry: a valid answer gives the entry dict of step_validate (cond = sha16(condition))",
          e == dict(condition="The image shows a table.", cond=gen.sha16("The image shows a table."), procedure="Read the header row first.", settings={}, ops=[])
          and note is None and why is None)
    # numbering under card_messages: dropped questions, images, identical texts
    par = {i: [R.io.get(f"p000|parent|{i}|{s}") for s in range(4)] for i in R.role["show"]}
    knobs = json.load(open(KNOBS))
    p0 = props_by_slot(R)[0]
    msgs, info = PR.card_messages(rows, par, p0["card"], tv, knobs, {}, max_prompt_tokens=4000)
    num = R.lab_numbering(p0["card"], msgs)
    check(f"numbering: a card with dropped questions maps only the kept ones ({info['n_shown']} of {info['n_card']})",
          info["dropped"] > 0 and num == list(enumerate(info["card_ids"], 1)))
    (num2, other, srcs), why = R.lab_rows(dict(p0), msgs)
    check("numbering: dropped questions count as other (not shown) rows", why is None and set(info["dropped_ids"]) <= set(other) and srcs == [p0["card_source"]])
    mi, _ = PR.card_messages(rows, par, p0["card"], tv, knobs, {"card_images": True})
    check("numbering: an image card (list-of-parts content) numbers the same", R.lab_numbering(p0["card"], mi) == list(enumerate(p0["card"], 1)))
    p1 = props_by_slot(R)[1]
    check("numbering: messages of another card do not match (fail closed)", R.lab_numbering(p0["card"], R.lab_p0_requests()[p1["key"]]["messages"]) is None)
    byp = collections.defaultdict(list)
    for i in R.role["show"]:
        byp[(PR._ws(rows[i]["prompt"]), tuple(PR._gold_lines(rows[i])))].append(i)
    dup = next(v for v in byp.values() if len(v) >= 2)
    byt = collections.defaultdict(list)
    for i in R.role["show"]:
        byt[PR._ws(rows[i]["prompt"])].append(i)
    dupt = next(v for v in byt.values() if len(v) >= 3)
    card = dupt[:3] + dup[:2]
    s_, u_ = PR.card_prompt(rows, par, card, tv, knobs)
    check(f"numbering: identical question texts map in card order ({len(dupt)} rows share one text)",
          R.lab_numbering(card, [dict(role="system", content=s_), dict(role="user", content=u_)]) == list(enumerate(card, 1)))
    check("lab_request: P0 and lab keys found, unknown key None",
          R.lab_request(p0["key"])["key"] == p0["key"] and R.lab_request(p0["key_final"])["key"] == p0["key_final"] and R.lab_request("chat|nope|lab1|x") is None)


# ---------------------------------------------------------------------------------------------------------------------- e2e
def e2e():
    D = new_root("e2e")
    for d in ("rows", "q/in", "q/out", "q/cancel"):
        os.makedirs(os.path.join(D, d), exist_ok=True)
    with open(ROWS) as f, open(os.path.join(D, "rows", "learn.jsonl"), "w") as g:
        g.write(f.read())
    rows = PR.load_rows(os.path.join(D, "rows", "learn.jsonl"))
    reqs = [dict(key=f"p000|parent|{i}|{s}", mode="read", engine="E", priority=2, row=i, sample=s, prefix=None,
                 settings={"min_image_tokens": None, "max_image_tokens": None, "temperature": None}) for i in rows for s in range(4)]
    p = os.path.join(D, "q", "in", "p000_parent_0.jsonl")
    with open(p + ".tmp", "w") as f:
        f.write("".join(json.dumps(q) + "\n" for q in reqs))
    os.replace(p + ".tmp", p)
    fl = open(os.path.join(D, "fake.log"), "w")
    fp = subprocess.Popen([sys.executable, "-u", os.path.join(HERE, "fake_dispatch.py"), "--root", D, "--rate", "0", "--quiet"], stdout=fl, stderr=fl)
    try:
        a = gen.args(["--root", D, "--rows", os.path.join(D, "rows", "learn.jsonl"), "--run", os.path.join(D, "run"), "--slots", "6", "--box", "900", "--gens", "2",
                      "--no-versions", "--no-w", "--no-stack"])
        R = LabRun(a, lab_pattern=(1, 2), fast_poll=True)
        R.main()
    finally:
        fp.terminate()
        fp.wait()
    S = json.load(open(os.path.join(D, "run", "state.json")))
    check("e2e: two generations finish every step, lab included", sorted(S["gens"]) == ["1", "2"] and all(G["done"] == gen.STEPS for G in S["gens"].values()))
    res = {}
    for f in os.listdir(os.path.join(D, "q", "out")):
        if f.endswith(".jsonl"):
            for l in open(os.path.join(D, "q", "out", f)):
                r = json.loads(l)
                res[r["key"]] = r
    n1 = n2 = 0
    example = None
    for g, G in S["gens"].items():
        P = {p["slot"]: p for p in G["props"] if p["author"] == "P0"}
        recs = {}
        for l in open(os.path.join(D, "run", f"prompts_lab_g{int(g):03d}.jsonl")):
            r = json.loads(l)
            recs[(r["turn"], r["slot"])] = r
        check(f"e2e g{g}: lab_turns 1 and 2 on the P0 slots, a lab record for each", sorted({p["lab_turns"] for p in P.values()}) == [1, 2]
              and all(p.get("lab") for p in P.values()))
        for s, p in P.items():
            for t in range(1, p["lab_turns"] + 1):
                r = recs.get((t, s))
                if r and r.get("key"):
                    n1 += t == 1
                    n2 += t == 2
                    PR.guard_messages(r["req"]["messages"], rows)
                    if example is None and t == 1 and r["entry"]["cond"] != "always" and r["shown"]["routed"]:
                        example = r["req"]["messages"][-1]["content"]
        for c in G["cands"]:
            if c["author"] == "P0" and c.get("entry") and not c.get("repair_reason"):
                kf = LAB.final_key(P[c["slot"]])
                cand = PR.parse(res[kf]["content"])[0]
                assert c["key_final"] == kf and c["entry"]["condition"] == cand["condition"], (g, c["cid"])
        names = collections.Counter(re.sub(r"_\d+\.jsonl$", "", f) for f in os.listdir(os.path.join(D, "q", "in")) if f.startswith(f"g{int(g):03d}_lab"))
        check(f"e2e g{g}: one batch per lab turn and kind for all slots ({dict(sorted(names.items()))})", names and max(names.values()) == 1)
    labq = [json.loads(l) for f in os.listdir(os.path.join(D, "q", "in")) if "_labroute" in f or "_labread" in f for l in open(os.path.join(D, "q", "in", f))]
    check(f"e2e: all {len(labq)} lab routes / reads are on SHOW rows", labq and all(rows[q["row"]]["v4_role"] == "show" for q in labq))
    check(f"e2e: P0 candidates read the final key (turn 1 sent {n1}, turn 2 sent {n2}); every lab request passed the guard", n1 >= 6 and n2 >= 1)
    out = os.path.join(D, "example_lab_feedback.txt")
    with open(out, "w") as f:
        f.write(example + "\n")
    print(f"     example lab feedback: {out}")
    print(f"     e2e root: {D}")


def main():
    scenario_main()
    scenario_crash()
    scenario_guard()
    scenario_unanswered_and_context()
    e2e()
    print("ALL GREEN")


if __name__ == "__main__":
    main()
