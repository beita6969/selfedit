"""Tests of selfedit.py on the real learn rows. Run: python3 t_selfedit.py -> ALL GREEN; writes example_selfedit_prompt.txt."""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import proposer as P  # noqa: E402
import selfedit as S  # noqa: E402

ROWS = os.path.join(HERE, "..", "data", "learn.jsonl")
OUT = os.path.join(HERE, "example_selfedit_prompt.txt")


def check(name, cond):
    if not cond:
        raise AssertionError(name)
    print("ok  ", name)


def raises(fn, exc):
    try:
        fn()
    except exc:
        return True
    return False


def block(obj):
    return "I compared the outcomes.\n```json\n" + json.dumps(obj) + "\n```"


TABLE = [
    dict(gen=1, author="blind", kind="setting", condition="always", procedure=None, settings={"max_image_tokens": 8192},
         routed_share=1.0, rows_read=444, D=0.011, lfsr=0.06, accepted=True),
    dict(gen=1, author="you", kind="text", condition="The image contains a chart or a plot with labelled axes.",
         procedure="Read every label and number in the image first and write them down, then answer.", settings={},
         routed_share=0.17, rows_read=76, D=0.004, lfsr=0.21, accepted=False),
    dict(gen=1, author="you", kind="setting", condition="always", procedure=None, settings={"temperature": 0.8},
         routed_share=1.0, rows_read=111, D=-0.009, lfsr=0.83, accepted=False),
    dict(gen=1, author="another proposer", kind="text", condition="The question asks to count objects in the image.",
         procedure="Before answering, list every relevant item you can see and check the list once against the image.", settings={},
         routed_share=0.22, rows_read=98, D=0.007, lfsr=0.12, accepted=False),
    dict(gen=2, author="you", kind="text", condition="The question asks about lines, arrows or paths in the image.",
         procedure="Trace each line or path from its start to its end and note where it ends before answering.", settings={},
         routed_share=0.09, rows_read=40, D=0.003, lfsr=0.18, accepted=False),
    dict(gen=2, author="blind", kind="setting", condition="always", procedure=None, settings={"min_image_tokens": 1024},
         routed_share=1.0, rows_read=111, D=-0.002, lfsr=0.61, accepted=False),
]
EDITS = [dict(id="e1", gen=1, author="blind", kind="setting", condition="always", procedure=None, settings={"max_image_tokens": 8192}, D=0.011, lfsr=0.06)]


def main():
    rows = P.load_rows(ROWS)
    menu = P.PARAMS_MENU
    v0 = dict(S.VERSION_DEFAULT)
    show = [r for r in rows.values() if r["v4_role"] == "show"]
    judge = [r for r in rows.values() if r["v4_role"] == "judge"]
    show_txt = {P._ws(r["prompt"]) for r in show}
    jt = next(r for r in judge if P._ws(r["prompt"]) not in show_txt)

    # ---------------------------------------------------------------- parse
    e, err = S.parse_selfedit(block({"why": "w", "type": "memory", "memory": "trace"}))
    check("parse: memory edit", err is None and e == dict(why="w", type="memory", memory="trace"))
    e, err = S.parse_selfedit(block({"why": "w", "type": "memory", "memory": "lessons", "lessons": " Look at the table. "}))
    check("parse: lessons edit (text stripped)", err is None and e["lessons"] == "Look at the table.")
    e, err = S.parse_selfedit(block({"why": "w", "type": "param", "param": "n_shown", "value": "18"}))
    check("parse: param edit, numeric string normalised", err is None and e["value"] == 18)
    e, err = S.parse_selfedit(block({"why": "w", "type": "param", "param": "card_images", "value": "true"}))
    check("parse: param edit, 'true' -> True", err is None and e["value"] is True)
    e, err = S.parse_selfedit(block({"why": "w", "type": "transfer", "edit": "e1"}))
    check("parse: transfer edit", err is None and e == dict(why="w", type="transfer", edit="e1"))
    check("parse: missing block", S.parse_selfedit("I would switch to trace memory.")[1].startswith("no_json"))
    check("parse: broken JSON", S.parse_selfedit('```json\n{"type": "memory", "memory": "trace",}\n```')[1].startswith("bad_json"))
    check("parse: unknown type", S.parse_selfedit(block({"type": "weights"}))[1].startswith("field: type"))
    check("parse: lessons without text", S.parse_selfedit(block({"type": "memory", "memory": "lessons"}))[1].startswith("field: lessons"))
    check("parse: unknown memory kind", S.parse_selfedit(block({"type": "memory", "memory": "skills"}))[1].startswith("field: memory"))
    two = block({"type": "memory", "memory": "trace"}) + "\nOn reflection:\n" + block({"type": "param", "param": "drafts", "value": 2})
    check("parse: last block wins", S.parse_selfedit(two)[0]["type"] == "param")

    # ---------------------------------------------------------------- validate
    V = lambda obj, ver=v0, ids=("e1",): S.validate_selfedit(S.parse_selfedit(block(obj))[0], ver, menu, list(ids), rows)
    has = lambda probs, s: any(s in x for x in probs)
    check("validate accepts memory none -> trace", V({"type": "memory", "memory": "trace"}) == [])
    check("validate accepts good lessons",
          V({"type": "memory", "memory": "lessons", "lessons": "Entries with narrow conditions were read on few questions; settings changes on every question were read on all of them."}) == [])
    check("validate accepts param n_shown 18", V({"type": "param", "param": "n_shown", "value": 18}) == [])
    check("validate accepts param card_images true", V({"type": "param", "param": "card_images", "value": True}) == [])
    check("validate accepts transfer e1", V({"type": "transfer", "edit": "e1"}) == [])
    check("validate rejects no change (memory none -> none)", has(V({"type": "memory", "memory": "none"}), "no change"))
    check("validate rejects no change (param at its current value)", has(V({"type": "param", "param": "temperature", "value": 0.7}), "no change"))
    vl = dict(v0, memory="lessons", lessons="Same text.")
    check("validate rejects the same lessons text", has(V({"type": "memory", "memory": "lessons", "lessons": "Same  text."}, vl), "no change"))
    check("validate rejects lessons over 1500 chars", has(V({"type": "memory", "memory": "lessons", "lessons": "Write short entries. " * 80}), "not in 1..1500"))
    check("validate rejects lessons with a bench name", has(V({"type": "memory", "memory": "lessons", "lessons": "Charxiv questions need axis reading."}), "bench name"))
    check("validate rejects lessons with a 2+-word category name",
          has(V({"type": "memory", "memory": "lessons", "lessons": "Relative_Depth items gained most."}), "category name"))
    check("validate rejects lessons with a row id", has(V({"type": "memory", "memory": "lessons", "lessons": f"See {jt['id']}."}), "row id"))
    check("validate rejects lessons with an option letter as an answer",
          has(V({"type": "memory", "memory": "lessons", "lessons": "On mirrored figures the answer is B."}), "option letter"))
    check("validate rejects lessons that say think longer", has(V({"type": "memory", "memory": "lessons", "lessons": "Tell the executor to think longer."}), "effort is fixed"))
    check("validate rejects lessons holding a JUDGE question text",
          has(V({"type": "memory", "memory": "lessons", "lessons": jt["prompt"][:1400]}), "leak guard"))
    check("validate rejects a param value outside the menu", has(V({"type": "param", "param": "n_shown", "value": 10}), "not one of"))
    check("validate rejects an int for card_images", has(V({"type": "param", "param": "card_images", "value": 1}), "not one of"))
    check("validate rejects an unknown param and effort", has(V({"type": "param", "param": "top_p", "value": 0.9}), "unknown card parameter")
          and has(V({"type": "param", "param": "effort", "value": "high"}), "effort is fixed"))
    check("validate rejects a transfer id not in the list", has(V({"type": "transfer", "edit": "e9"}), "not an accepted"))
    check("validate rejects a transfer already applied", has(V({"type": "transfer", "edit": "e1"}, dict(v0, transfers=["e1"])), "already transferred"))
    check("validate rejects an unknown field", has(V({"type": "memory", "memory": "trace", "skills": ["x"]}), "unknown field"))
    ch = S.apply_selfedit(v0, S.parse_selfedit(block({"type": "param", "param": "n_shown", "value": 18}))[0], "P1")
    check("apply_selfedit builds the challenger, the incumbent unchanged",
          ch["params"]["n_shown"] == 18 and v0["params"]["n_shown"] == 12 and ch["parent"] == "P0" and ch["id"] == "P1")

    # ---------------------------------------------------------------- memory_text
    check("memory none -> None", S.memory_text(v0, TABLE, [], rows) is None)
    tr = S.memory_text(dict(v0, memory="trajectory"), TABLE, [], rows)
    check("memory trajectory: every outcome row, aggregates only, passes guard",
          isinstance(tr, str) and all(S.outcome_line(r) in tr for r in TABLE) and isinstance(P.guard("", tr, rows), dict))
    le = S.memory_text(dict(v0, memory="lessons", lessons="Narrow entries were read on few questions."), TABLE, [], rows)
    check("memory lessons: the version's own text", le.endswith("Narrow entries were read on few questions."))
    sr = show[3]
    items = [dict(TABLE[1], row_prompt=sr["prompt"], **{"with": "x " * 50 + "so the label reads 40. Answer: \\boxed{40}",
                                                         "without": "y " * 300 + "the value is 30. Answer: \\boxed{30}", "with_ok": True, "without_ok": False})] * 4
    tc = S.memory_text(dict(v0, memory="trace"), TABLE, items, rows)
    check("memory trace: own entries only, up to 3 examples, tails <= 400 chars, passes guard",
          "another proposer" not in tc and "| blind |" not in tc and tc.count("### Example") == 3 and sr["prompt"].strip() in tc
          and all(len(m) <= 403 for m in re.findall(r"end of its answer: (.*)", tc)) and isinstance(P.guard("", tc, rows), dict))
    bad = [dict(items[0], row_prompt=jt["prompt"])]
    check("memory trace: a JUDGE question text in trace_items raises LeakError",
          raises(lambda: S.memory_text(dict(v0, memory="trace"), TABLE, bad, rows), P.LeakError))
    bad2 = [dict(items[0], **{"with": "as in " + jt["id"]})]
    check("memory trace: a JUDGE row id in a transcript raises LeakError",
          raises(lambda: S.memory_text(dict(v0, memory="trace"), TABLE, bad2, rows), P.LeakError))
    mm, info = P.card_messages(rows, {r["id"]: [dict(ok=True, content="Answer: \\boxed{A}", reasoning="r", finish="stop")] for r in show},
                               [show[0]["id"], show[1]["id"]], None, json.load(open(os.path.join(HERE, "knobs.json"))), {}, author_memory=tc)
    check("trace memory goes into card_messages under '## Memory'", mm[1]["content"].endswith("## Memory\n" + tc))

    # ---------------------------------------------------------------- caps and card lines (coordinator 11-10 follow-ups)
    big = [dict(TABLE[k % len(TABLE)], gen=g) for g in range(1, 80) for k in range(4)]
    inf = {}
    tb = S.memory_text(dict(v0, memory="trajectory"), big, [], rows, info=inf)
    m_ = inf["memory"]
    check(f"trajectory memory capped at 6000 chars ({len(tb)}), newest rows kept, said in text and info ({m_['rows_left_out']} of {m_['rows_total']} left out)",
          len(tb) <= S.TRAJECTORY_CAP == 6000 and m_["cap"] == 6000 and m_["chars"] == len(tb) and m_["rows_left_out"] > 0
          and m_["rows_kept"] + m_["rows_left_out"] == len(big) == m_["rows_total"]
          and f"({m_['rows_left_out']} earlier rows left out for length)" in tb and "g79 |" in tb and "g1 |" not in tb)
    shw = [r["id"] for r in show[:12]]
    card = [dict(id=x, routed=(j % 3 != 0), tip_right=(j % 2 == 0)) for j, x in enumerate(shw)]
    own = [dict(TABLE[1], card=card), dict(TABLE[2], card=[dict(c, routed=None) for c in card[:4]]), dict(TABLE[3], card=card)]
    inf = {}
    tc2 = S.memory_text(dict(v0, memory="trace"), own, [], rows, info=inf)
    lines = [l for l in tc2.splitlines() if l.startswith('  - "')]
    q0 = P._ws(rows[shw[0]]["prompt"])
    check(f"trace memory: one line per shown question of each own entry's card ({len(lines)} = 12 + 4), other authors' cards not shown",
          len(lines) == 16 and tc2.count("condition held: always") == 4 and "another proposer" not in tc2
          and inf["memory"]["rows_total"] == 2 and isinstance(P.guard("", tc2, rows), dict))
    check("trace memory line: first 80 chars of the question text, condition held yes/no, tip right/wrong",
          lines[0] == f'  - "{q0[:80]}{"..." if len(q0) > 80 else ""}" | condition held: no | tip: right'
          and lines[1].endswith("| condition held: yes | tip: wrong"))
    check("trace memory: the only fixed sentence about cards is the column legend (no statement on how conditions route)",
          [l for l in tc2.splitlines() if "condition held" in l and not l.startswith('  - "')] == [S._legend(False), S.CARD_LEGEND]
          and not re.search(r"route[sd]? (well|badly|often|rarely)|works? (well|best)|better|worse", tc2, re.I))
    jid = jt["id"]
    check("trace memory: a JUDGE id on an entry's card raises LeakError",
          raises(lambda: S.memory_text(dict(v0, memory="trace"), [dict(TABLE[1], card=card + [dict(id=jid, routed=True, tip_right=False)])], [], rows), P.LeakError))
    mon = next(r["id"] for r in rows.values() if r["v4_role"] == "monitor")
    check("trace memory: a MONITOR id on an entry's card raises LeakError",
          raises(lambda: S.memory_text(dict(v0, memory="trace"), [dict(TABLE[1], card=[dict(id=mon, routed=False, tip_right=True)])], [], rows), P.LeakError))
    tj = S.memory_text(dict(v0, memory="trajectory"), own, [], rows)
    check("trajectory memory never shows card rows (aggregates only)",
          '  - "' not in tj and "condition held: " not in tj and q0[:40] not in tj and S.CARD_LEGEND not in tj)
    many = [dict(TABLE[1], gen=g, card=card) for g in range(1, 40)]
    inf = {}
    tm = S.memory_text(dict(v0, memory="trace"), many, [], rows, info=inf)
    check(f"trace memory entry list capped at {S.TRACE_ENTRIES_CAP} chars, newest kept ({inf['memory']['rows_left_out']} entries left out)",
          inf["memory"]["rows_left_out"] > 0 and "g39 |" in tm and "g1 |" not in tm
          and len(tm) <= S.TRACE_ENTRIES_CAP + 1000 and f"({inf['memory']['rows_left_out']} earlier entries left out for length)" in tm)
    inf = {}
    check("memory none / empty lessons: None, info says so",
          S.memory_text(v0, TABLE, [], rows, info=inf) is None and inf["memory"] == dict(kind="none", chars=0))

    # ---------------------------------------------------------------- the self-edit prompt
    msgs = S.selfedit_messages(dict(v0, memory="trajectory"), TABLE, menu, EDITS, rows)
    sys_, user = msgs[0]["content"], msgs[1]["content"]
    check("selfedit_messages: system + user, text only, passes guard",
          [m["role"] for m in msgs] == ["system", "user"] and isinstance(user, str) and isinstance(P.guard_messages(msgs, rows), dict))
    check("selfedit_messages: version, outcome table, transfer list, affordances, answer format",
          "- memory: trajectory" in user and all(S.outcome_line(r) in user for r in TABLE) and "- e1: generation 1" in user
          and all(f"  - {k}: one of" in user for k in menu) and '"type": "transfer"' in user)
    check("selfedit menu lists card_composition like the others",
          '  - card_composition: one of "single", "mixed" (now "single"). Single = all shown questions of a card come from one source; mixed = they come from all sources.' in user
          and V({"type": "param", "param": "card_composition", "value": "mixed"}) == []
          and any("no change" in x for x in V({"type": "param", "param": "card_composition", "value": "single"})))
    lab_line = ('  - lab_turns: one of 0, 1, 2 (now 0). Number of trial rounds per slot: in each round the entry is run 4 times (4 samples) on '
                "the card's questions and on other questions from the same sources, the results are shown as fractions, and the proposer "
                "keeps, revises or replaces the entry (0 = the first entry is submitted as it is).")
    check("selfedit menu lists lab_turns (0, 1, 2; now 0) in neutral words; the system text names trial rounds as a card parameter",
          lab_line in user and "how many trial rounds each slot gets" in sys_
          and V({"type": "param", "param": "lab_turns", "value": 2}) == [] and V({"type": "param", "param": "lab_turns", "value": "1"}) == []
          and any("no change" in x for x in V({"type": "param", "param": "lab_turns", "value": 0}))
          and any("not one of" in x for x in V({"type": "param", "param": "lab_turns", "value": 3}))
          and S.apply_selfedit(v0, S.parse_selfedit(block({"type": "param", "param": "lab_turns", "value": 1}))[0])["params"]["lab_turns"] == 1)
    ro = dict(TABLE[1], ops=[{"op": "check"}, {"op": "zoom"}])
    lo = S.outcome_line(ro)
    check("outcome rows: an ops column (engine order; '-' when none), named in the legend; transfer list and trace examples show ops too",
          '| ops: {"op": "zoom"}, {"op": "check"} |' in lo and "| ops: - |" in S.outcome_line(TABLE[1]) and "| settings | ops | routed share" in S.LEGEND
          and '; ops: {"op": "tiles", "n": 2}; D' in S._edits_block([dict(EDITS[0], ops=[{"op": "tiles", "n": 2}])])
          and '| ops: {"op": "check"}\n' in S.memory_text(dict(v0, memory="trace"), [ro], [dict(items[0], ops=[{"op": "check"}])], rows))
    benches = {r["bench"] for r in rows.values()}
    check("selfedit frame: no bench name, no row content, no 'effort'",
          not any(re.search(rf"\b{b}\b", sys_ + user, re.I) for b in benches) and "effort" not in (sys_ + user).lower()
          and not any(P._ws(r["prompt"]) in P._ws(user) for r in rows.values()))
    big = [dict(TABLE[1], gen=g) for g in range(1, 400)]
    mb = S.selfedit_messages(v0, big, menu, [], rows, max_table_chars=20000)
    check("selfedit_messages: a long table keeps the newest rows within max_table_chars",
          "earlier rows left out for length" in mb[1]["content"] and "g399 |" in mb[1]["content"] and "g1 |" not in mb[1]["content"]
          and "none yet" in mb[1]["content"].lower())
    with open(OUT, "w") as f:
        f.write("=== SYSTEM ===\n" + sys_ + "\n\n=== USER ===\n" + user + "\n")
    print(f"     example self-edit prompt: {OUT}: system {len(sys_)} + user {len(user)} = {len(sys_) + len(user)} chars")
    print("ALL GREEN")


if __name__ == "__main__":
    main()
