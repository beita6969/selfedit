"""Tests of proposer.py on the real learn rows and a fake parent cache. Run: python3 t_proposer.py -> ALL GREEN."""
import collections
import hashlib
import json
import os
import random
import re
import sys
import time
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import proposer as P  # noqa: E402

ROWS = os.path.join(HERE, "..", "data", "learn.jsonl")
KNOBS = os.path.join(HERE, "knobs.json")
OUT = os.path.join(HERE, "example_card_prompt.txt")
# knobs.json "settings" on 11-10: the frozen default prompt is built with these, so a later knobs.json edit does not move it
KNOBS_REF = {"settings": {
    "min_image_tokens": {"stock": None, "values": [None, 1024, 2048, 4096], "what": "every image is upscaled to at least this many visual tokens (null = the processor's own minimum)"},
    "max_image_tokens": {"stock": 16384, "values": [16384, 8192, 4096, 2048], "what": "larger images are downscaled to at most this many visual tokens"},
    "temperature": {"stock": 1.0, "values": [0.6, 0.8, 1.0, 1.2], "what": "sampling temperature of the answer"}}}
# S2-B default prompt frozen 11-10 (sha256[:16]); a change of the frame text or of the card text shows up here
REF_DIGESTS = {"system": "6946e7015342dead", "system_mixed": "c86b11a74e896fc9", "affordance": "aee315815d3a4650",
               "answer_format": "dee6c25411b302a3", "prompts": "961237e84ce172af"}
# gen.py's repairable problem classes (format limits, naming): ops problems must never fall in them
FORMAT_PROBLEMS = ("condition length", "procedure has", "procedure must be a string", "settings must be an object", "unknown field")
NAMING_PROBLEM = re.compile(r"^(condition|procedure): leakage precheck: (bench name|category name)")
ADVICE = re.compile(r"\b(should|tends?|better|worse|lessons?|hints?|recommend\w*|suggest\w*|consider|works?|effective|improv\w*|"
                    r"help\w*|hurt\w*|avoid|prefer\w*|best|good|bad|useful|cheap|costly)\b", re.I)     # t_lab.py's list, plus three
VOCAB = ("the left shape rotates so the count of marked cells changes and each row keeps its colour while the arrow "
         "points down so I compare the outline with option two then check the shading again").split()


def fake_parent(rows, seed=7, n=4):
    rng = random.Random(seed)
    par = {}
    for rid in sorted(rows):
        p = rng.choice((0.15, 0.5, 0.85))
        L = []
        for s in range(n):
            length = rng.random() < 0.05
            ok = (not length) and rng.random() < p
            ans = rows[rid]["answer"] if ok else "X"
            L.append(dict(key=f"p000|parent|{rid}|{s}", mode="read", row=rid, sample=s,
                          content="" if length else f"So the result follows. \\boxed{{{ans}}}",
                          reasoning=" ".join(rng.choice(VOCAB) for _ in range(rng.randint(40, 260))),
                          finish="length" if length else "stop", tokens=rng.randint(500, 9000), ok=ok))
        par[rid] = L
    return par


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


def main():
    rows = P.load_rows(ROWS)
    knobs = json.load(open(KNOBS))
    par = fake_parent(rows)
    show = {i for i, r in rows.items() if r["v4_role"] == "show"}
    judge = [i for i, r in rows.items() if r["v4_role"] == "judge"]
    check("768 rows, 180 show", len(rows) == 768 and len(show) == 180)

    # ---------------------------------------------------------------- cards
    right = lambda rid: P._is_right(P._samples(par, rid))
    for seed in (0, 1, 2026):
        for k in range(6):
            c = P.cards(rows, par, k, seed)
            assert len(c) == 12 and len(set(c)) == 12, (seed, k, len(c))
            assert all(i in show for i in c), "non-show row on a card"
            assert sum(not right(i) for i in c) == 8 and sum(right(i) for i in c) == 4, (seed, k)
            assert c == P.cards(rows, par, k, seed), "not deterministic"
            per = collections.Counter(rows[i]["bench"] for i in c)
            assert len(per) == 6 and max(per.values()) - min(per.values()) <= 1, per
            pw = collections.Counter(rows[i]["bench"] for i in c if not right(i))
            assert max(pw.values()) - min(pw.values()) <= 1 and len(pw) == 6, pw
    check("cards: SHOW only, 8 wrong + 4 right, deterministic per (seed, k), balanced over 6 benches", True)
    draws = {tuple(P.cards(rows, par, k, 3)) for k in range(10)}
    check("cards: each slot k gets its own draw (10 slots, 10 distinct cards)", len(draws) == 10)
    # majority rule, ties wrong, error lines unread
    tiny = {f"t/{j}": dict(id=f"t/{j}", bench="b", v4_role="show", prompt="q", options=[], answer="1", kind="num") for j in range(4)}
    tp = {"t/0": [dict(ok=True), dict(ok=True), dict(ok=False), dict(ok=False)],          # 2/4 tie -> wrong
          "t/1": [dict(ok=True), dict(ok=True), dict(ok=True), dict(ok=False)],           # 3/4 -> right
          "t/2": [dict(ok=True), dict(ok=None, error="timeout"), dict(ok=False, error="x")],  # 1/1 read -> right
          "t/3": [dict(error="timeout")]}                                                  # unread -> skipped
    c = P.cards(tiny, tp, 0, 0, n_wrong=8, n_right=4)
    check("cards: tie counts wrong, error lines unread, unread rows skipped, short pools filled",
          sorted(c) == ["t/0", "t/1", "t/2"] and not P._is_right(P._samples(tp, "t/0")) and P._is_right(P._samples(tp, "t/2")))

    # ---------------------------------------------------------------- card_prompt + guard
    cid = P.cards(rows, par, 0, 1)
    tip = {"settings": {}, "entries": []}
    system, user = P.card_prompt(rows, par, cid, tip, knobs)
    full = system + "\n" + user
    check("card_prompt passes guard", isinstance(P.guard(system, user, rows), dict))
    check("card_prompt: no row id of any row", not P._ids_in(full, set(rows)))
    norm = P._ws(full)
    others = [r for r in rows.values() if r["v4_role"] != "show"]
    show_txt = {P._ws(rows[i]["prompt"]) for i in show}
    check("card_prompt: no non-show question text (except texts identical to a show text)",
          not any(P._ws(r["prompt"]) in norm for r in others if P._ws(r["prompt"]) not in show_txt))
    check("card_prompt: every card question shown, with its mark",
          all(P._ws(rows[i]["prompt"]) in norm for i in cid) and user.count("currently wrong (") == 8 and user.count("currently right (") == 4)
    frame = system + "\n" + re.sub(r"## Shown questions.*?(?=## What an entry can contain)", "", user, flags=re.S)
    benches = {r["bench"] for r in rows.values()}
    check("frame text: no bench name, no category name (any length), no 'effort'",
          not any(re.search(rf"\b{b}\b", frame, re.I) for b in benches)
          and not any(re.search(r"(?<!\w)" + re.escape(P._cat_norm(c)) + r"(?!\w)", P._cat_norm(frame))
                      for c in {r["category"] for r in rows.values()} if len(c) >= 6)
          and "effort" not in frame.lower())
    check("card_prompt: no images, no file paths, no picture keys",
          "/data/" not in full and ".png" not in full and ".jpg" not in full and not (P.PIC_TOKEN.findall(full)))
    check("card_prompt: non-show card row -> ValueError", raises(lambda: P.card_prompt(rows, par, cid[:11] + [judge[0]], tip, knobs), ValueError))
    s2, u2 = P.card_prompt(rows, par, cid, tip, knobs, author_memory="MEMO line 1\nline 2")
    check("card_prompt: author_memory appended verbatim under its own heading", u2.endswith("## Memory\nMEMO line 1\nline 2") and u2.startswith(user))
    tip2 = {"settings": {"temperature": 0.8}, "entries": [dict(condition="the image shows a grid", procedure="Read every cell first.", settings={"max_image_tokens": 8192})]}
    _, u3 = P.card_prompt(rows, par, cid, tip2, knobs)
    check("card_prompt: tip entries and changed settings shown",
          "- temperature: 0.8\n" in u3 and "condition: the image shows a grid" in u3 and "max_image_tokens = 8192" in u3)

    # an injected judge question text, not identical to any show text
    jt = next(r for r in others if r["v4_role"] == "judge" and P._ws(r["prompt"]) not in show_txt)
    check("guard raises on an injected judge-row question text",
          raises(lambda: P.guard(system, user + "\n\n" + jt["prompt"], rows), P.LeakError))
    check("guard raises on the same text with other whitespace",
          raises(lambda: P.guard(system, user + "\n" + re.sub(r" ", "  \n ", jt["prompt"]), rows), P.LeakError))
    check("guard raises on an injected judge row id",
          raises(lambda: P.guard(system, user + f"\nsee {jt['id']}.", rows), P.LeakError))
    mon = next(r for r in others if r["v4_role"] == "monitor")
    check("guard raises on an injected monitor row id in the system text",
          raises(lambda: P.guard(system + f" ({mon['id']})", user, rows), P.LeakError))
    tmpl = next(r for r in others if P._ws(r["prompt"]) in show_txt)
    check("guard allows a non-show text identical to a show text (templated)",
          isinstance(P.guard(system, user + "\n" + tmpl["prompt"], rows), dict))
    sid = next(iter(sorted(show)))
    check("guard allows a show row id (only non-show ids are leaks)", isinstance(P.guard(system, user + " " + sid, rows), dict))
    # speed: 20 requests of ~30k chars, fresh index
    P._META.clear()
    reqs = []
    for k in range(20):
        s_, u_ = P.card_prompt(rows, par, P.cards(rows, par, k, 9), tip, knobs)
        pad = " ".join(VOCAB[j % len(VOCAB)] for j in range(6000))[:max(0, 30000 - len(s_) - len(u_))]
        reqs.append((s_, u_ + pad))
    P._META.clear()
    t0 = time.time()
    for s_, u_ in reqs:
        P.guard(s_, u_, rows)
    dt = time.time() - t0
    print(f"     guard: 20 requests of {min(len(a) + len(b) for a, b in reqs)}-{max(len(a) + len(b) for a, b in reqs)} chars in {dt * 1000:.0f} ms (index build included)")
    check("guard: 20 x ~30k-char requests in < 2 s", dt < 2.0)

    # ---------------------------------------------------------------- parse
    good = 'I looked at the cards.\n```json\n{"why": "w", "condition": "always", "procedure": null, "settings": {"temperature": 0.8}}\n```\n'
    c, e = P.parse(good)
    check("parse: good answer (ops [] when not given)", e is None and c == dict(why="w", condition="always", procedure=None, settings={"temperature": 0.8}, ops=[]))
    c, e = P.parse("I think the tip should change its temperature, but I give no block.")
    check("parse: missing block -> no_json error", c is None and e.startswith("no_json"))
    c, e = P.parse('```json\n{"why": "w", "condition": "always", "procedure": null,, }\n```')
    check("parse: broken JSON -> bad_json error", c is None and e.startswith("bad_json"))
    two = ('Draft:\n```json\n{"why": "a", "condition": "always", "procedure": null, "settings": {"temperature": 0.6}}\n```\n'
           'Better:\n```json\n{"why": "b", "condition": "The image contains a table.", "procedure": "Read the header row first.", "settings": {}}\n```')
    c, e = P.parse(two)
    check("parse: two blocks, last wins", e is None and c["why"] == "b" and c["procedure"] == "Read the header row first.")
    c, e = P.parse('Reasoning text with \\boxed{3} in it.\n{"why": "x", "condition": "Always", "procedure": "Trace each line.", "settings": {"max_image_tokens": "8192"}}')
    check("parse: bare JSON object at the end, normalised", e is None and c["condition"] == "always" and c["settings"] == {"max_image_tokens": 8192})
    c, e = P.parse('```json\n["always"]\n```')
    check("parse: non-object -> not_object error", c is None and e.startswith("not_object"))
    c, e = P.parse('```json\n{"why": "x", "condition": "always", "settings": {}, "effort": "high"}\n```')
    check("parse: extra keys kept for validate", e is None and c["extra"] == {"effort": "high"})

    # ---------------------------------------------------------------- validate
    V = lambda **kw: P.validate(dict(dict(why="", condition="always", procedure=None, settings={}), **kw), tip, knobs, rows)
    has = lambda probs, s: any(s in x for x in probs)
    check("validate rejects an effort key in settings", has(V(settings={"effort": "high"}), "effort is fixed"))
    check("validate rejects an effort field", has(P.validate(c, tip, knobs, rows), "effort is fixed"))
    check("validate rejects an unknown knob value", has(V(settings={"temperature": 0.7}), "not one of"))
    check("validate rejects a bool knob value", has(V(settings={"temperature": True}), "not one of"))
    check("validate rejects an unknown knob", has(V(settings={"top_p": 0.9}), "unknown knob"))
    check("validate rejects an empty payload", has(V(), "empty payload"))
    check("validate rejects a settings entry equal to the tip", has(V(settings={"max_image_tokens": 16384}), "no change vs the tip"))
    check("validate rejects a bench name in the condition", has(V(condition="The question is from BabyVision.", procedure="Count twice."), "bench name"))
    check("validate rejects a 2+-word category name in label form in the condition", has(V(condition="The task is about Relative_Depth of two points.", procedure="Find where each point meets the ground."), "category name 'Relative_Depth'"))
    check("validate allows the same words as an ordinary lower-case phrase (coordinator 11-10 b)", not has(V(condition="The task is about relative depth of two points.", procedure="Find where each point meets the ground."), "category name"))
    check("validate rejects a category phrase with other separators", has(V(condition="It is a spatial_perception / paper folding item.", procedure="Unfold the paper step by step."), "category name 'Spatial Perception/Paper Folding'"))
    check("validate allows single-word category names (counting, height, rotation, Jigsaw, position)",
          V(condition="The question asks for counting objects or comparing their height.",
            procedure="When counting, mark each item once. Note the position and rotation of each piece, as in a Jigsaw.") == [])
    check("validate rejects a row id in the procedure", has(V(procedure=f"Do what worked on {jt['id']}."), "row id"))
    check("validate rejects an option letter as an answer", has(V(procedure="If two figures are mirrored, the answer is B."), "option letter given as an answer"))
    check("validate rejects 'choose A'", has(V(procedure="When unsure, choose A."), "option letter given as an answer"))
    check("validate rejects a fixed value for a kind of object", has(V(procedure="A standard chess set has 32 pieces."), "fixed value"))
    check("validate rejects a procedure over 600 chars", has(V(procedure="Look at each part of the image in turn. " * 20), "> 600"))
    check("validate rejects a condition over 200 chars", has(V(condition="The image shows a figure with many parts " * 6, procedure="Name each part."), "condition length"))
    check("validate rejects 'think longer'", has(V(procedure="Think longer before you answer and check twice."), "effort is fixed"))
    check("validate rejects 'use more reasoning'", has(V(procedure="Use more reasoning tokens on hard items."), "effort is fixed"))
    check("validate rejects a duplicate of a tip entry",
          has(P.validate(dict(why="", condition="the image shows a grid", procedure="Read every cell first.", settings={"max_image_tokens": 8192}), tip2, knobs, rows), "same as tip entry 1"))
    ok1 = V(settings={"max_image_tokens": 8192, "min_image_tokens": 2048})
    check("validate accepts a good 'always' settings entry", ok1 == [])
    ok2 = V(condition="The image contains several thin lines that cross each other.",
            procedure="Before answering, trace each line from one end to the other and write down where it starts and ends; then compare these notes with the question.")
    check("validate accepts a good worded-condition procedure entry", ok2 == [])
    ok3 = V(condition="always", procedure="Describe the image in one sentence first.", settings={"temperature": 1.0})
    check("validate accepts a procedure with a no-op setting", ok3 == [])
    check("validate: parsed good answer is valid", P.validate(P.parse(good)[0], tip, knobs, rows) == [])

    # ---------------------------------------------------------------- Stage 2: a version's card parameters
    import t_ref_proposer_stage1 as R          # frozen copy of the Stage-1 proposer.py (sha256 f3247fcc...)
    n_same, n_mixed = 0, 0
    mixed = {"card_composition": "mixed"}
    tip_ops0 = {"settings": {}, "entries": [dict(condition="always", procedure=None, settings={}, ops=[{"op": "check"}]),
                                            dict(condition="the image shows a grid", procedure="Read every cell first.", settings={"temperature": 0.8},
                                                 ops=[{"op": "zoom"}, {"op": "tiles", "n": 3}])]}
    for seed in range(3):
        for k in range(4):
            c1 = R.cards(rows, par, k, seed)
            assert c1 == P.cards(rows, par, k, seed) == P.cards(rows, par, k, seed=seed) == P.cards(rows, par, k, seed, dict(P.PARAMS_DEFAULT)) \
                == P.cards(rows, par, k, seed, n_wrong=8, n_right=4) == P.cards(rows, par, k, seed, 8, 4) \
                == P.draw_card(rows, par, mixed, None, k, seed) == P.draw_card(rows, par, mixed, "lego", k, seed), (seed, k)
            n_mixed += 1
    def parts(u):
        i, j, k = u.index("## Shown questions"), u.index("## What an entry can contain"), u.index("## Answer format")
        m = u.find("\n\n## Memory\n", k)
        return dict(tip=u[:i], card=u[i:j], aff=u[j:k], fmt=u[k:m] if m >= 0 else u[k:], mem=u[m:] if m >= 0 else "")
    pay = lambda s: next(x for x in s.splitlines() if x.startswith("- The payload is"))
    check("SYSTEM: only the payload sentence differs from Stage 1 (it names the ops)",
          P.SYSTEM.replace(pay(P.SYSTEM), pay(R.SYSTEM)) == R.SYSTEM and "harness ops" in pay(P.SYSTEM))
    frozen = []
    for seed in ("g1:0", "g2:0"):
        for b in sorted(benches):
            for i in range(2):
                ids = P.draw_card(rows, par, {}, b, i, seed)
                assert ids == P.cards_source(rows, par, b, i, seed, n_wrong=8, n_right=4) and all(rows[x]["bench"] == b and x in show for x in ids)
                for tp, mem in ((tip, None), (tip2, "MEMO line"), (tip_ops0, None)):
                    s1, u1 = P.card_prompt(rows, par, ids, tp, knobs, mem)
                    m, info = P.card_messages(rows, par, ids, tp, knobs, {}, mem)
                    assert [x["content"] for x in m] == [s1, u1] and [x["role"] for x in m] == ["system", "user"] and info["dropped"] == 0
                    mm, _ = P.card_messages(rows, par, ids, tp, knobs, {"card_composition": "mixed"}, mem)
                    assert mm[1]["content"] == u1 and mm[0]["content"] == s1.replace(P.SOURCES_SINGLE, P.SOURCES_MIXED) != s1   # mixed: Stage-1 sentence
                    if tp is not tip_ops0:      # Stage-1 parts that S2-B leaves alone: card, memory, tip (minus the ops lines)
                        s0, u0 = R.card_prompt(rows, par, ids, tp, knobs, mem)
                        p1, p0 = parts(u1), parts(u0)
                        assert p1["card"] == p0["card"] and p1["mem"] == p0["mem"] and p1["fmt"] != p0["fmt"] and p1["aff"] != p0["aff"]
                        assert p1["tip"].replace("- ops: none\n", "").replace(" and their ops are applied together", "") == p0["tip"]
                        assert mm[0]["content"] == s0.replace(pay(s0), pay(s1))
                    frozen.append(P.card_prompt(rows, par, ids, tp, KNOBS_REF, mem))
                    n_same += 1
    check(f"defaults: card_messages == card_prompt on the same single-source ids, mixed differs only in the held-out sentence; card, "
          f"memory and tip text (minus ops lines) == frozen Stage-1 ({n_same} prompts)", True)
    dg = lambda x: hashlib.sha256(x.encode()).hexdigest()[:16]
    got = dict(system=dg(P._system_for(P.params_of({}))), system_mixed=dg(P._system_for(P.params_of({"card_composition": "mixed"}))),
               affordance=dg(P._affordance_block(KNOBS_REF)), answer_format=dg(P.ANSWER_FORMAT),
               prompts=dg(json.dumps(frozen)))
    print("     S2-B reference digests:", json.dumps(got))
    check(f"S2-B default prompt == the frozen reference ({len(frozen)} prompts at KNOBS_REF; changed parts: "
          f"{[k for k in got if REF_DIGESTS.get(k) != got[k]]})", got == REF_DIGESTS)
    check(f"mixed draws == frozen Stage-1 cards() ({n_mixed} draws; old cards() call forms too)", True)
    check("draw_card single: cards_source with split_counts as up-to counts; no source -> ValueError",
          P.draw_card(rows, par, {"n_shown": 6, "wrong_share": 0.5}, "blink", 0, "s") == P.cards_source(rows, par, "blink", 0, "s", n_wrong=3, n_right=3)
          and raises(lambda: P.draw_card(rows, par, {}, None, 0, "s"), ValueError))
    check("PARAMS_DEFAULT = False, 12, 0.67, 2, null, 16384, 0.7, 1, single, 0",
          P.PARAMS_DEFAULT == dict(card_images=False, n_shown=12, wrong_share=0.67, samples_shown=2, card_min_tokens=None,
                                   card_max_tokens=16384, temperature=0.7, drafts=1, card_composition="single", lab_turns=0)
          and P.PARAMS_MENU["lab_turns"]["values"] == [0, 1, 2] and P.params_of({"lab_turns": 2})["lab_turns"] == 2
          and raises(lambda: P.params_of({"lab_turns": 3}), ValueError) and raises(lambda: P.params_of({"lab_turns": True}), ValueError))
    check("params_of rejects a value outside the menu, an unknown name, and an int for a bool",
          raises(lambda: P.params_of({"n_shown": 7}), ValueError) and raises(lambda: P.params_of({"effort": "high"}), ValueError)
          and raises(lambda: P.params_of({"card_images": 1}), ValueError))
    for n in P.PARAMS_MENU["n_shown"]["values"]:
        for w in P.PARAMS_MENU["wrong_share"]["values"]:
            prm = dict(n_shown=n, wrong_share=w)
            c = P.cards(rows, par, 2, 11, prm)
            nw, nr = P.split_counts(prm)
            assert nw == round(n * w) and len(c) == n and sum(not right(i) for i in c) == nw and all(i in show for i in c), (n, w)
            per = collections.Counter(rows[i]["bench"] for i in c)
            assert max(per.values()) - min(per.values()) <= 1 and c == P.cards(rows, par, 2, 11, prm)
    check("cards(params): n_shown x wrong_share for every menu pair (n_wrong = round(n x share)), SHOW only, balanced, deterministic", True)
    c12 = P.cards(rows, par, 0, 1)
    for n in (1, 2, 3):
        m, _ = P.card_messages(rows, par, c12, tip, knobs, {"samples_shown": n})
        assert m[1]["content"].count("\n- Sample ") == 12 * n and f"For up to {n} sample{'' if n == 1 else 's'} of each" in m[1]["content"], n
    check("samples_shown 1 / 2 / 3: that many samples per question, stated in the card text", True)

    prm = {"card_images": True, "card_max_tokens": 1024}
    mi, ii = P.card_messages(rows, par, c12, tip, knobs, prm)
    mt, it = P.card_messages(rows, par, c12, tip, knobs, dict(prm, card_images=False))
    parts = mi[1]["content"]
    imgs = [x for x in parts if x["type"] == "image"]
    want = [pth for rid in ii["card_ids"] for pth in rows[rid]["image_paths"]]
    check(f"images: user content is a list of parts; {len(imgs)} image parts = the card rows' image_paths in order",
          isinstance(parts, list) and ii["dropped"] == 0 and [x["path"] for x in imgs] == want
          and all(x["min_image_tokens"] is None and x["max_image_tokens"] == 1024 for x in imgs))
    ok_place = True
    for j, x in enumerate(parts):
        if x["type"] == "image":
            nxt = next(y for y in parts[j + 1:] if y["type"] == "text")
            prv = next(y for y in reversed(parts[:j]) if y["type"] == "text")
            ok_place &= nxt["text"].startswith("Question text:") and bool(re.search(r"### Question \d+: [^\n]*\n$", prv["text"]))
    check("images: each question's images sit right after its header, right before its 'Question text'", ok_place)
    check("images: the text parts joined are exactly the text-only user content",
          "".join(x["text"] for x in parts if x["type"] == "text") == mt[1]["content"])
    check("images: the system sentence says the images are attached at the proposer's own limits",
          P.TEXT_ONLY_SENTENCE in mt[0]["content"] and P.TEXT_ONLY_SENTENCE not in mi[0]["content"]
          and "attached right before its question text and resized to between the processor's own minimum and 1024 visual tokens per image" in mi[0]["content"]
          and mi[0]["content"].replace(mi[0]["content"].splitlines()[-1], "") == mt[0]["content"].replace(mt[0]["content"].splitlines()[-1], ""))
    judge_img = next(pth for r in others for pth in r["image_paths"] if pth not in {q for i in show for q in rows[i]["image_paths"]})
    check("guard (list contents) passes the image card, raises on a JUDGE image part",
          isinstance(P.guard(mi[0]["content"], parts, rows), dict)
          and raises(lambda: P.guard(mi[0]["content"], parts + [dict(type="image", path=judge_img, min_image_tokens=None, max_image_tokens=1024)], rows), P.LeakError))
    check("guard (list contents) raises on a JUDGE question text in a text part, and on a JUDGE image path in text",
          raises(lambda: P.guard(mi[0]["content"], parts + [dict(type="text", text="\n" + jt["prompt"])], rows), P.LeakError)
          and raises(lambda: P.guard(mi[0]["content"], parts + [dict(type="text", text=" see " + judge_img)], rows), P.LeakError))
    check("guard_messages audits a whole message list", isinstance(P.guard_messages(mi, rows), dict)
          and raises(lambda: P.guard_messages(mi + [dict(role="user", content=[dict(type="image", path=judge_img)])], rows), P.LeakError))

    calls = []
    est = lambda row, p: calls.append((row["id"], p["n_shown"])) or 5000
    mf, inf = P.card_messages(rows, par, c12, tip, knobs, {}, max_prompt_tokens=40000, tok_est=est)
    check(f"fit: tok_est(row, params) is used; {inf['dropped']} dropped, estimate {inf['est_tokens']} <= 40000, mix {inf['n_wrong']}/{inf['n_shown']} vs 8/12",
          calls and all(r in c12 and n == 12 for r, n in calls) and inf["fits"] and inf["est_tokens"] <= 40000 and inf["dropped"] == 12 - inf["n_shown"] > 0
          and abs(inf["n_wrong"] / inf["n_shown"] - 8 / 12) <= 1 / inf["n_shown"] and inf["card_ids"] == [x for x in c12 if x not in set(inf["dropped_ids"])]
          and mf[1]["content"].count("### Question ") == inf["n_shown"])
    m16, i16 = P.card_messages(rows, par, c12, tip, knobs, {"card_images": True})
    check(f"fit: no image_tokens in the parent -> card_max per image (fallback): {i16['n_shown']} of 12 kept, estimate {i16['est_tokens']}",
          i16["fits"] and i16["est_tokens"] <= 40000 and 1 <= i16["n_shown"] < 12 and i16["images_at_card_max"] == i16["n_images"]
          and isinstance(P.guard_messages(m16, rows), dict))
    # parent records with stock image_tokens (engine: learn images are 64-450 tokens)
    itok = lambda pth: 64 + zlib.crc32(pth.encode()) % 387
    par_it = {rid: [dict(s, image_tokens=[itok(p_) for p_ in rows[rid]["image_paths"]]) for s in L] for rid, L in par.items()}
    cost = lambda prm_, ids: sum(min(prm_["card_max_tokens"], max(prm_["card_min_tokens"] or 0, itok(p_))) for i in ids for p_ in rows[i]["image_paths"])
    full = lambda m_: m_[0]["content"] + "".join(x["text"] for x in m_[1]["content"] if x["type"] == "text")
    for over in ({}, {"card_min_tokens": 2048}, {"card_max_tokens": 1024}, {"card_min_tokens": 1024, "card_max_tokens": 1024}):
        prm_ = P.params_of(dict({"card_images": True}, **over))
        ms, is_ = P.card_messages(rows, par_it, c12, tip, knobs, prm_)
        want_ = len(full(ms)) / P.CHARS_PER_TOKEN + cost(prm_, is_["card_ids"])
        assert is_["images_at_card_max"] == 0 and abs(is_["est_tokens"] - want_) <= 1, (over, is_["est_tokens"], want_)
        assert is_["fits"] and (is_["dropped"] == 0 or over.get("card_min_tokens") == 2048), over     # 22 images x 2048 > 40k: drops expected
    ms, is_ = P.card_messages(rows, par_it, c12, tip, knobs, {"card_images": True})
    img_part = cost(P.params_of({"card_images": True}), c12)
    check(f"fit: stock image_tokens from the parent -> min(card_max, max(card_min or 0, t)) per image; 12-question image card "
          f"= {is_['est_tokens']} tokens ({img_part} for {is_['n_images']} images), nothing dropped; checked at 4 min/max settings", True)
    par_bad = {rid: [dict(s, image_tokens=[100]) for s in L] for rid, L in par.items()}      # wrong length -> fallback for that row
    multi = next(i for i in c12 if len(rows[i]["image_paths"]) > 1)
    check("fit: an image_tokens list of the wrong length falls back to card_max for that question",
          P.image_cost(par_bad, multi, len(rows[multi]["image_paths"]), P.params_of({"card_images": True})) == (16384 * len(rows[multi]["image_paths"]), len(rows[multi]["image_paths"])))
    m1, i1 = P.card_messages(rows, par, c12, tip, knobs, {}, max_prompt_tokens=100)
    check("fit: one question always stays; fits False when even that is over", i1["n_shown"] == 1 and not i1["fits"])
    _, i2 = P.card_messages(rows, par, c12, tip, knobs, {}, max_prompt_tokens=4000)
    check(f"fit: the count follows the budget, not a constant (4000 tokens -> {i2['n_shown']} kept)", 1 < i2["n_shown"] < 12 and i2["fits"])

    # ---------------------------------------------------------------- harness ops (S2-B, engine_api.md §7)
    J = lambda **kw: "```json\n" + json.dumps(dict(dict(why="w", condition="always", procedure=None, settings={}), **kw)) + "\n```"
    c, e = P.parse(J(ops=[{"op": " Check"}, {"OP": "tiles", "n": "3"}, {"op": "zoom"}]))
    check("parse ops: normalised (case, spaces, '3' -> 3) and put in engine order zoom, tiles, check",
          e is None and c["ops"] == [{"op": "zoom"}, {"op": "tiles", "n": 3}, {"op": "check"}] and P.validate(c, tip, knobs, rows) == [])
    check("parse ops: null -> []; tiles n 2.0 -> 2", P.parse(J(ops=None))[0]["ops"] == [] and P.parse(J(ops=[{"op": "tiles", "n": 2.0}]))[0]["ops"] == [{"op": "tiles", "n": 2}])
    check("parse ops: not a list, or a list with a non-object -> a format error (repairable like other format errors)",
          P.parse(J(ops="zoom"))[1] == "field: ops must be a list of objects" and P.parse(J(ops=["zoom"]))[1] == "field: ops must be a list of objects"
          and P.parse(J(ops={"op": "zoom"}))[1] == "field: ops must be a list of objects")
    OV = lambda ops, **kw: P.validate(dict(dict(why="", condition="always", procedure=None, settings={}, ops=ops), **kw), tip, knobs, rows)
    bad_ops = {"unknown op": [{"op": "blur"}], "value": [{"op": "tiles", "n": 4}], "missing n": [{"op": "tiles"}],
               "bool n": [{"op": "tiles", "n": True}], "field": [{"op": "zoom", "n": 2}], "duplicate": [{"op": "check"}, {"op": "check"}],
               "two tiles": [{"op": "tiles", "n": 2}, {"op": "tiles", "n": 3}], "effort op": [{"op": "think_longer"}]}
    probs_ops = {k: OV(v) for k, v in bad_ops.items()}
    check("validate ops: unknown op, value outside 2 | 3, missing n, bool n, unknown field, an op twice, two tiles -> problems",
          all(probs_ops[k] for k in bad_ops) and has(probs_ops["two tiles"], "more than one tiles") and has(probs_ops["duplicate"], "check is given 2 times")
          and has(probs_ops["value"], "tiles n = 4 is not one of [2, 3]") and has(probs_ops["unknown op"], "unknown op 'blur'"))
    check("validate ops: every ops problem starts with 'ops:' and none is in gen.py's repairable classes (format / naming)",
          all(x.startswith("ops: ") and not x.startswith(FORMAT_PROBLEMS) and not NAMING_PROBLEM.match(x) for v in probs_ops.values() for x in v))
    check("validate ops: an ops-only entry is valid (always and worded condition), also with a procedure and settings",
          OV([{"op": "zoom"}]) == [] and OV([{"op": "check"}], condition="The image contains a chart or a plot with labelled axes.") == []
          and OV([{"op": "tiles", "n": 3}, {"op": "check"}], procedure="Read every label first.", settings={"temperature": 0.8}) == [])
    check("validate ops: empty payload names ops; ops given as [] is no payload", has(OV([]), "empty payload: needs a procedure, settings or ops"))
    tip_ops = {"settings": {}, "entries": [dict(condition="always", procedure=None, settings={}, ops=[{"op": "check"}]),
                                           dict(condition="the image shows a grid", procedure="Read every cell first.", settings={}, ops=[{"op": "tiles", "n": 2}])]}
    TV = lambda **kw: P.validate(dict(dict(why="", condition="always", procedure=None, settings={}, ops=[]), **kw), tip_ops, knobs, rows)
    check("validate ops: ops the tip already applies on every question are no change; a new op is",
          has(TV(ops=[{"op": "check"}], condition="The question shows a table."), "the tip applies these ops on every question already")
          and TV(ops=[{"op": "zoom"}]) == [] and TV(ops=[{"op": "check"}], settings={"temperature": 0.8}) == [])
    check("validate ops: a duplicate of a tip entry with ops (ops order does not matter)",
          has(TV(condition="the image shows a grid", procedure="Read every cell first.", ops=[{"op": "tiles", "n": 2}]), "same as tip entry 2")
          and not has(TV(condition="the image shows a grid", procedure="Read every cell first.", ops=[{"op": "tiles", "n": 3}]), "same as tip entry"))
    e0 = dict(condition="The image shows a grid.", procedure="Read every cell.", settings={"temperature": 0.8})
    check("_entry_key: without ops = the Stage-1 key (stored keys still match); with ops one more part, order-insensitive",
          P._entry_key(e0) == R._entry_key(e0) == P._entry_key(dict(e0, ops=[])) == P._entry_key(dict(e0, ops=None))
          and P._entry_key(dict(e0, ops=[{"op": "check"}, {"op": "zoom"}])) == P._entry_key(dict(e0, ops=[{"op": "zoom"}, {"op": "check"}]))
          != P._entry_key(e0) and len(P._entry_key(dict(e0, ops=[{"op": "zoom"}]))) == 4
          and json.loads(json.dumps(P._entry_key(dict(e0, ops=[{"op": "zoom"}])))) is not None)
    check("merge_ops: each op once, a later entry's tiles n wins, engine order",
          P.merge_ops([[{"op": "tiles", "n": 2}, {"op": "check"}], None, [{"op": "tiles", "n": 3}, {"op": "zoom"}]])
          == [{"op": "zoom"}, {"op": "tiles", "n": 3}, {"op": "check"}] and P.merge_ops([]) == [])
    _, u_ops = P.card_prompt(rows, par, cid, tip_ops, knobs)
    check("tip block: each entry's ops ('- ops: none' when empty)",
          '- ops: {"op": "check"}\n' in u_ops and '- ops: {"op": "tiles", "n": 2}\n' in u_ops
          and P.card_prompt(rows, par, cid, tip2, knobs)[1].count("- ops: none\n") == 1
          and "their procedures are joined in this order and their ops are applied together" in u_ops)
    aff = u_ops[u_ops.index("## What an entry can contain"):u_ops.index("## Answer format")]
    check("affordance list: one line per op as the engine runs it, the equal-compute rule, ops in the payload rule",
          all(x in aff for x in ['  - {"op": "zoom"}: a first short pass names one image region, then the question is answered with the original images plus that region cropped and enlarged as an extra image.',
                                 '  - {"op": "tiles", "n": 2 | 3}: each image is also given as n x n enlarged tiles.',
                                 '  - {"op": "check"}: after answering, one more pass shows the question with the proposed answer and asks to confirm or correct it, and the corrected answer is graded.',
                                 "c >= 1.5, its gain is measured against the plurality answer of ceil(c) tip samples on the same question instead of the tip's mean.",
                                 "- The entry needs a procedure, settings, ops, or a combination, and must differ from the tip."])
          and not ADVICE.search(aff) and '"ops": [{"op": "<op>", ...}, ...] | []}' in P.ANSWER_FORMAT)

    # ---------------------------------------------------------------- pick_draft (drafts > 1)
    g_a = '```json\n{"why": "a", "condition": "always", "procedure": null, "settings": {"temperature": 0.8}}\n```'
    g_b = '```json\n{"why": "b", "condition": "The image contains a table.", "procedure": "Read the header row first.", "settings": {}}\n```'
    broken = '```json\n{"why": "x", "condition": "always",, }\n```'
    leaky = '```json\n{"why": "x", "condition": "The question is from BLINK.", "procedure": "Look twice.", "settings": {}}\n```'
    k_, c_, e_ = P.pick_draft([broken, None, leaky, g_b, g_a], tip, knobs, rows)
    check("pick_draft: the first draft that parses and validates (broken, unanswered and leaky drafts skipped)",
          k_ == 3 and c_["why"] == "b" and e_ is None)
    check("pick_draft: selects on nothing else (two valid drafts -> the first)", P.pick_draft([g_a, g_b], tip, knobs, rows)[0] == 0)
    k_, c_, e_ = P.pick_draft([broken, leaky], tip, knobs, rows)
    check("pick_draft: no valid draft -> (None, None, error listing each draft)",
          k_ is None and c_ is None and e_.startswith("no_valid_draft: 0: bad_json") and "1: invalid:" in e_ and "bench name" in e_)
    wordy = '```json\n{"why": "x", "condition": "The picture shows a person about to blink.", "procedure": "Look at both eyes first.", "settings": {}}\n```'
    check("pick_draft: allowed_words is passed to validate (a bench word that is card content)",
          P.pick_draft([wordy], tip, knobs, rows)[0] is None and P.pick_draft([wordy], tip, knobs, rows, allowed_words={"blink"})[0] == 0)

    # ---------------------------------------------------------------- example prompt
    with open(OUT, "w") as f:
        f.write("=== SYSTEM ===\n" + system + "\n\n=== USER ===\n" + user + "\n")
    print(f"     example prompt: {OUT}: system {len(system)} + user {len(user)} = {len(system) + len(user)} chars")
    print("ALL GREEN")


if __name__ == "__main__":
    main()
