"""selfedit v4, proposer module (Stage 1 + Stage 2; design_1110_v4.md §3-§4, v4_build_spec_1110.md AM1-AM8).

One proposer slot = one card of SHOW rows -> one prompt (neutral task frame + facts) -> one candidate entry
(condition, payload). Functions:
  load_rows(path)                                   -> {row id: row}
  cards(rows, parent, k, seed, params=None)          -> [row ids]      (SHOW rows only, per-slot draw; old form
                                                                        cards(rows, parent, k, seed, n_wrong=8, n_right=4) works)
  cards_source(rows, parent, source, i, seed, n_wrong=8, n_right=4) -> [row ids]   (single-source card, coordinator 11-10)
  draw_card(rows, parent, params, source, i, seed)   -> [row ids]      (card_composition: single -> cards_source, mixed -> cards)
  card_prompt(rows, parent, card_ids, tip, knobs, author_memory=None) -> (system, user)   (Stage 1 text, guarded)
  card_messages(rows, parent, card_ids, tip, knobs, params, author_memory=None, max_prompt_tokens=40000, tok_est=None)
                                                     -> (messages, info)  (a proposer version's card parameters; guarded)
  parse(text)                                        -> (cand | None, error | None)   (cand has "ops": [] when none)
  validate(cand, tip, knobs, rows, allowed_words=None) -> [problems]   ([] = valid)
  pick_draft(texts, tip, knobs, rows, allowed_words=None) -> (index, cand, error)   (first draft that parses and validates)
  ops_problems(ops)                                  -> [problems]     (harness ops, engine_api.md §7)
  merge_ops(lists)                                   -> canonical ops of several entries holding on one question
  guard(system, user, rows)                          -> info dict, raises LeakError (str or list-of-parts contents)
  leak_precheck(text, rows)                          -> [hit reasons]  (v3 hw_round precheck, ported)
  params_of(params)                                  -> full card parameters (defaults filled, values checked)

parent: {row id: [result dicts]} with keys content, reasoning, finish, tokens, ok (engine_api.md §4; error lines are
unread and ignored). tip: None (stock) or {"settings": {knob: value}, "entries": [{"condition", "procedure",
"settings", "ops"}, ...]}. Effort is never a knob (AM1). Standard library only.
"""
from __future__ import annotations

import collections
import json
import random
import re

COND_MAX, PROC_MAX = 200, 600          # AM3 / builder brief
REASON_TAIL, CONTENT_TAIL = 600, 300    # chars of a shown sample's reasoning / final text (v3 card_cases idea)
SAMPLES_SHOWN = 2
MIN_TEXT = 16                           # B2Guard min_text_len
ANCHOR = 16                             # guard index key length (<= MIN_TEXT)
CATEGORY_MIN_WORDS = 2                  # coordinator 11-10: only category names of 2+ words (split on space, "_", "/") are rejected
CATEGORY_EXEMPT: set = set()            # normalised category phrases ("relative depth") the coordinator may exempt
CAT_SEP = re.compile(r"[\s_/]+")
CHARS_PER_TOKEN = 3.5                   # default text token estimate (coordinator 11-10)

# A proposer version's own card parameters (AM4 internal parameters; coordinator 11-10). The defaults reproduce Stage 1.
PARAMS_MENU = {
    "card_images": {"values": [False, True], "default": False,
                    "what": "whether each shown question's images are attached to the card"},
    "n_shown": {"values": [6, 9, 12, 18, 24], "default": 12,
                "what": "number of shown questions on each card"},
    "wrong_share": {"values": [0.5, 0.67, 0.83, 1.0], "default": 0.67,
                    "what": "share of the shown questions that the tip currently gets wrong (the rest it gets right)"},
    "samples_shown": {"values": [1, 2, 3], "default": 2,
                      "what": "number of the tip's samples shown for each question"},
    "card_min_tokens": {"values": [None, 1024, 2048], "default": None,
                        "what": "attached images are upscaled to at least this many visual tokens (null = the processor's own minimum)"},
    "card_max_tokens": {"values": [16384, 4096, 1024], "default": 16384,
                        "what": "attached images are downscaled to at most this many visual tokens"},
    "temperature": {"values": [0.4, 0.7, 1.0], "default": 0.7,
                    "what": "sampling temperature of the proposer's own answer"},
    "drafts": {"values": [1, 2, 3], "default": 1,
               "what": "number of drafts written per slot before one entry is submitted"},
    "card_composition": {"values": ["single", "mixed"], "default": "single",
                         "what": "single = all shown questions of a card come from one source; mixed = they come from all sources"},
    "lab_turns": {"values": [0, 1, 2], "default": 0,
                  "what": "number of trial rounds per slot: in each round the entry is run 4 times (4 samples) on the card's questions and "
                          "on other questions from the same sources, the results are shown as fractions, and the proposer keeps, "
                          "revises or replaces the "
                          "entry (0 = the first entry is submitted as it is)"},
}
PARAMS_DEFAULT = {k: v["default"] for k, v in PARAMS_MENU.items()}

# Harness ops on read (engine_api.md §7, dispatch.py run_ops): fields and allowed values; applied in OPS_ORDER (then no vote).
OPS = {"zoom": {}, "tiles": {"n": [2, 3]}, "check": {}}
OPS_ORDER = ("zoom", "tiles", "check")
OPS_IMAGE_LIMIT = 40                    # engine common.OPS_IMAGE_LIMIT: more images after zoom / tiles -> op_skipped


class LeakError(AssertionError):
    pass


def _menu_values(menu, name):
    v = menu[name]
    return v["values"] if isinstance(v, dict) else list(v)


def _in_menu(value, values):
    """Menu membership with types kept apart: bools only match bools, strings only strings, numbers match numbers
    (1 == 1.0), None matches None."""
    for x in values:
        if isinstance(x, str) or isinstance(value, str):
            if isinstance(x, str) and isinstance(value, str) and x == value:
                return True
        elif isinstance(x, bool) or isinstance(value, bool):
            if isinstance(x, bool) and isinstance(value, bool) and x == value:
                return True
        elif _same(value, x):
            return True
    return False


def params_of(params=None, menu=None):
    """Card parameters with defaults filled in; unknown names or values outside the menu raise ValueError."""
    menu = menu or PARAMS_MENU
    p = {k: (v["default"] if isinstance(v, dict) else None) for k, v in menu.items()}
    for k, v in (params or {}).items():
        if k not in menu:
            raise ValueError(f"unknown card parameter {k!r}")
        if not _in_menu(v, _menu_values(menu, k)):
            raise ValueError(f"card parameter {k} = {v!r} is not one of {_menu_values(menu, k)}")
        p[k] = v
    return p


def split_counts(params):
    """(n_wrong, n_right) of a card: n_wrong = round(n_shown x wrong_share) (Python round), the rest right."""
    p = params_of(params)
    nw = int(round(p["n_shown"] * p["wrong_share"]))
    return nw, p["n_shown"] - nw


# ------------------------------------------------------------------------------------------------ rows and samples
def load_rows(path):
    out = {}
    with open(path) as f:
        for ln in f:
            if ln.strip():
                r = json.loads(ln)
                if r["id"] in out:
                    raise ValueError(f"duplicate row id {r['id']!r} in {path}")
                out[r["id"]] = r
    return out


def _sidx(s):
    if isinstance(s.get("sample"), int):
        return s["sample"]
    try:
        return int(str(s.get("key", "")).rsplit("|", 1)[-1])
    except ValueError:
        return 0


def _samples(parent, rid):
    """Read samples of a row in sample order; error lines (no content, never graded) are unread, not wrong."""
    vs = [s for s in (parent.get(rid) or []) if isinstance(s, dict) and not s.get("error") and s.get("ok") is not None]
    return sorted(vs, key=_sidx)


def _is_right(vs):
    """Currently right iff a strict majority of the samples is ok (a tie counts as wrong)."""
    return 2 * sum(bool(s["ok"]) for s in vs) > len(vs)


def cards(rows, parent, k, seed, params=None, n_right=None, n_wrong=None):
    """SHOW rows for slot k: n_wrong currently wrong + n_right currently right, spread over benches round-robin in a
    per-slot shuffled bench order (one pointer for both draws, so per-bench totals differ by at most one). A short pool
    is filled from the other one. SHOW rows without a read sample are skipped. Returned in a shuffled order.
    params (dict): the card parameters, n_wrong / n_right from n_shown and wrong_share (split_counts). The Stage-1 forms
    cards(rows, parent, k, seed), cards(..., n_wrong=8, n_right=4) and cards(..., 8, 4) still work (default 8 + 4)."""
    if isinstance(params, dict):
        n_wrong, n_right = split_counts(params)
    else:
        if isinstance(params, int) and not isinstance(params, bool):
            n_wrong = params
        n_wrong = 8 if n_wrong is None else n_wrong
        n_right = 4 if n_right is None else n_right
    rng = random.Random(f"{seed}:{k}")
    pools = {False: collections.defaultdict(list), True: collections.defaultdict(list)}
    for rid in sorted(rows):
        r = rows[rid]
        if r.get("v4_role") != "show":
            continue
        vs = _samples(parent, rid)
        if vs:
            pools[_is_right(vs)][r["bench"]].append(rid)
    benches = sorted(set(pools[False]) | set(pools[True]))
    if not benches:
        return []
    rng.shuffle(benches)
    left = {ok: {b: rng.sample(v, len(v)) for b, v in sorted(pools[ok].items())} for ok in (False, True)}
    pos = [0]

    def take(ok, n):
        got = []
        while len(got) < n and any(left[ok].get(b) for b in benches):
            b = benches[pos[0] % len(benches)]
            pos[0] += 1
            if left[ok].get(b):
                got.append(left[ok][b].pop())
        return got

    wrong = take(False, n_wrong)
    want_right = n_right + (n_wrong - len(wrong))
    right = take(True, want_right)
    if len(right) < want_right:
        wrong += take(False, want_right - len(right))
    out = wrong + right
    rng.shuffle(out)
    return out


def source_pools(rows, parent, source):
    """(wrong ids, right ids) of the SHOW rows of one source bench that have a read sample, in id order."""
    pools = {False: [], True: []}
    for rid in sorted(rows):
        r = rows[rid]
        if r.get("v4_role") == "show" and r.get("bench") == source:
            vs = _samples(parent, rid)
            if vs:
                pools[_is_right(vs)].append(rid)
    return pools[False], pools[True]


def cards_source(rows, parent, source, i, seed, n_wrong=8, n_right=4):
    """Single-source card (coordinator 11-10): SHOW rows of ONE source bench for that source's i-th slot, up to n_wrong currently
    wrong + up to n_right currently right (no filling from the other pool). Each pool is shuffled once per (seed, source) and slot i
    takes the next window of it, so the slots of one source show different rows as long as the pool lasts. Shuffled order."""
    out = []
    for ok, (pool, n) in enumerate(zip(source_pools(rows, parent, source), (n_wrong, n_right))):
        P = list(pool)
        random.Random(f"{seed}:{source}:{ok}").shuffle(P)
        m = min(n, len(P))
        out += [P[(i * m + t) % len(P)] for t in range(m)]
    random.Random(f"{seed}:{source}:slot{i}").shuffle(out)
    return out


def draw_card(rows, parent, params, source, i, seed):
    """Card ids under a version's card_composition: "single" = cards_source(source, slot i) with split_counts(params) as
    up-to counts; "mixed" = cards(rows, parent, i, seed, params) over all sources (source ignored)."""
    p = params_of(params)
    if p["card_composition"] == "mixed":
        return cards(rows, parent, i, seed, p)
    if source is None:
        raise ValueError("a single-source card needs a source")
    n_wrong, n_right = split_counts(p)
    return cards_source(rows, parent, source, i, seed, n_wrong=n_wrong, n_right=n_right)


# ------------------------------------------------------------------------------------------------ tip
def _ws(s):
    return re.sub(r"\s+", " ", str(s)).strip()


def _is_always(c):
    return isinstance(c, str) and c.strip().strip(".\"'").lower() == "always"


def _same(v, x):
    if v is None or x is None:
        return v is None and x is None
    if isinstance(v, bool) or isinstance(x, bool) or not isinstance(v, (int, float)) or not isinstance(x, (int, float)):
        return False
    return float(v) == float(x)


def _tip_view(tip, knobs):
    """(settings on every question, entries). Base = stock, then tip['settings'], then settings of 'always' entries."""
    S = knobs["settings"]
    base = {k: S[k]["stock"] for k in S}
    tip = tip or {}
    base.update({k: v for k, v in (tip.get("settings") or {}).items() if k in S})
    entries = list(tip.get("entries") or [])
    for e in entries:
        if _is_always(e.get("condition")):
            base.update({k: v for k, v in (e.get("settings") or {}).items() if k in S})
    return base, entries


def _val(v):
    return json.dumps(v)


def _entry_key(e):
    """(condition, procedure, settings[, ops]): the ops part is there only when the entry has ops, so keys of entries without
    ops are the Stage-1 keys (stored keys still match)."""
    c = e.get("condition")
    c = "always" if _is_always(c) else _ws(c or "").lower()
    p = e.get("procedure")
    p = _ws(p) if isinstance(p, str) and p.strip() else None
    s = tuple(sorted((str(k), json.dumps(v)) for k, v in (e.get("settings") or {}).items()))
    o = _ops_key(e.get("ops"))
    return (c, p, s, o) if o else (c, p, s)


def _op_rank(o):
    name = o.get("op") if isinstance(o, dict) else None
    return OPS_ORDER.index(name) if name in OPS_ORDER else len(OPS_ORDER)


def _norm_op(o):
    """One op object: keys stripped and lower case, op name stripped and lower case, numbers from strings, 3.0 -> 3."""
    out = {}
    for k, v in o.items():
        k = str(k).strip().lower()
        if k == "op":
            v = v.strip().lower() if isinstance(v, str) else v
        else:
            v = _num(v)
            if isinstance(v, float) and v.is_integer():
                v = int(v)
        out[k] = v
    return out


def canon_ops(ops):
    """Ops (list of objects or None) in engine order (zoom, tiles, check; unknown ones last), each normalised."""
    if not ops:
        return []
    return sorted((_norm_op(o) if isinstance(o, dict) else o for o in ops), key=_op_rank)


def _ops_key(ops):
    if not ops:
        return ()
    if not isinstance(ops, list):
        return (json.dumps(ops, sort_keys=True, default=str),)
    return tuple(json.dumps(o, sort_keys=True, default=str) for o in canon_ops(ops))


def merge_ops(lists):
    """Ops of several entries that hold on one question (lists in entry order): each op once, a later entry's tiles n replacing an
    earlier one's (as a later entry's settings do); engine order."""
    by = {}
    for ops in lists:
        for o in canon_ops(ops or []):
            if isinstance(o, dict) and o.get("op") in OPS:
                by[o["op"]] = dict(o)
    return [by[k] for k in OPS_ORDER if k in by]


def _ops_text(ops):
    return ", ".join(json.dumps(o) for o in canon_ops(ops)) if ops else "none"


# ------------------------------------------------------------------------------------------------ prompt
SYSTEM = """You write one change to the harness of a vision-language model, called the executor.

The executor answers one question at a time. It receives the question's images and text, reasons, and then writes a final answer, which is graded against a gold answer.

The harness is an ordered list of entries. Each entry has a condition and a payload.
- The condition is either "always" or a sentence in words. For each question, a router reads that question's images and text and answers yes or no to the condition. "always" holds on every question.
- The payload is a procedure text, setting changes, harness ops, or a combination. On a question where the condition holds, the procedure is added to the executor's prompt right before the question text, the settings are used for that question, and the ops are applied when the executor answers that question. On a question where the condition does not hold, the entry has no effect.

You write exactly one new entry. It is added to the current harness, called the tip, and judged by its paired gain over the tip on held-out questions that you never see, drawn from the same sources as the questions shown to you. On each held-out question the tip with your entry and the tip alone both answer, and the gain is the difference in accuracy. Only questions on which your condition holds are read; on all other questions the two are identical and the difference is zero. The entry joins the harness only if its gain is credibly above zero.

The router decides the condition separately for each question, from that question's images and text alone, and the same router also runs on questions from benchmarks that are not shown here.

You see the questions as text only; the router and the executor see the images."""

ANSWER_FORMAT = """## Answer format
Reason first. Then end with exactly one json block:
```json
{"why": "...", "condition": "always" | "<words>", "procedure": "<text>" | null, "settings": {"<knob>": value, ...} | {}, "ops": [{"op": "<op>", ...}, ...] | []}
```"""


def _tail(t, n):
    t = _ws(t or "")
    if len(t) <= n:
        return t
    t = t[-n:]
    sp = t.find(" ")
    return "..." + (t[sp + 1:] if 0 <= sp < 30 else t)


def _gold_lines(r):
    opts, ans = list(r.get("options") or []), str(r.get("answer", "")).strip()
    if r.get("kind") == "mc" and opts:
        letters = [chr(65 + i) for i in range(len(opts))]
        L = ["Options:"] + [f"({a}) {o}" + ("   <- gold" if a == ans else "") for a, o in zip(letters, opts)]
        if ans not in letters:
            L.append(f"Gold answer: {ans}")
        return L
    return [f"Gold answer: {ans}"]


def _pick(vs, wrong_row, n=SAMPLES_SHOWN):
    """Up to n samples: first one of the row's majority outcome (a wrong one on a wrong row), then one of the other
    outcome where there is one, then the next samples in order."""
    first = next((s for s in vs if bool(s["ok"]) != wrong_row), None) or vs[0]
    rest = [s for s in vs if s is not first]
    other = next((s for s in rest if bool(s["ok"]) != bool(first["ok"])), None)
    out = [first] + ([other] if other is not None else [])
    out += [s for s in rest if all(s is not x for x in out)]
    return out[:n]


def _sample_lines(j, s):
    content = (s.get("content") or "").strip()
    if content:
        said = _tail(content, CONTENT_TAIL)
    else:
        said = "(none" + (", the sample reached the token limit" if s.get("finish") == "length" else "") + ")"
    reason = _tail(s.get("reasoning") or "", REASON_TAIL) or "(empty)"
    return [f"- Sample {j} ({'right' if s['ok'] else 'wrong'})", f"  End of reasoning: {reason}", f"  Final text: {said}"]


def _tip_block(tip, knobs):
    S = knobs["settings"]
    base, entries = _tip_view(tip, knobs)
    L = ["## The tip", "Settings on every question (an entry whose condition holds may change them for that question):"]
    L += [f"- {k}: {_val(base[k])}" + (" (stock)" if _same(base[k], S[k]["stock"]) else "") for k in S]
    if not entries:
        L.append("Entries: none (the tip is the executor as it is).")
    else:
        L.append(f"Entries ({len(entries)}, in order; where several conditions hold on one question, their procedures are joined in this order"
                 f" and their ops are applied together):")
        for j, e in enumerate(entries, 1):
            st = e.get("settings") or {}
            L += [f"Entry {j}", f"- condition: {e.get('condition')}",
                  f"- procedure: {e.get('procedure') if e.get('procedure') else 'none'}",
                  "- settings: " + (", ".join(f"{k} = {_val(v)}" for k, v in st.items()) if st else "none"),
                  f"- ops: {_ops_text(e.get('ops'))}"]
    return "\n".join(L)


def _card_lines(rows, parent, card_ids, p):
    """Card block as a list of lines; with p["card_images"] an image part dict sits between each question's header line
    and its "Question text:" line. Returns (lines, blocks): blocks[i] = (rid, text of question i's lines, n images)."""
    n_s = p["samples_shown"]
    marks = []
    for rid in card_ids:
        vs = _samples(parent, rid)
        marks.append((rid, vs, _is_right(vs)))
    n_w = sum(not ok for _, _, ok in marks)
    ns = sorted({len(vs) for _, vs, _ in marks})
    times = f"{ns[0]} times" if len(ns) == 1 else "several times"
    L = [f"## Shown questions ({len(marks)}: {n_w} currently wrong, {len(marks) - n_w} currently right)",
         f"The tip answered each question {times} (its samples). A question is currently right when more than half of "
         f"its samples are right, and currently wrong otherwise. For up to {n_s} sample{'' if n_s == 1 else 's'} of each question, the "
         f"end of the executor's reasoning and its final text are shown."]
    blocks = []
    for q, (rid, vs, ok) in enumerate(marks, 1):
        r = rows[rid]
        head = ["", f"### Question {q}: currently {'right' if ok else 'wrong'} ({sum(bool(s['ok']) for s in vs)} of {len(vs)} samples right)"]
        body = ["Question text:", str(r["prompt"]).strip()] + _gold_lines(r)
        for j, s in enumerate(_pick(vs, not ok, n_s), 1):
            body += _sample_lines(j, s)
        imgs = [dict(type="image", path=pth, min_image_tokens=p["card_min_tokens"], max_image_tokens=p["card_max_tokens"])
                for pth in (r.get("image_paths") or [])] if p["card_images"] else []
        L += head + imgs + body
        blocks.append((rid, "\n".join(head + body), len(imgs)))
    return L, blocks


def _card_block(rows, parent, card_ids):
    """Stage-1 card text (defaults, no images)."""
    return "\n".join(_card_lines(rows, parent, card_ids, PARAMS_DEFAULT)[0])


def _segments(lines):
    """Lines with image part dicts -> segments (str | image dict). The str segments concatenated equal "\n".join of the
    str lines, so the text of an image card is exactly the text-only card."""
    segs, buf = [], []
    for x in lines:
        if isinstance(x, str):
            buf.append(x)
        else:
            if buf:
                segs.append("\n".join(buf) + "\n")
                buf = []
            segs.append(x)
    if buf:
        segs.append("\n".join(buf))
    return segs


def _affordance_block(knobs):
    S = knobs["settings"]
    L = ["## What an entry can contain",
         f'- condition: "always", or one sentence of at most {COND_MAX} characters that can be answered yes or no from one question\'s images and text.',
         f"- procedure: a text of at most {PROC_MAX} characters, or null.",
         "- settings: an object with zero or more of these keys, each set to one of its listed values:"]
    for k, d in S.items():
        what = str(d.get("what", "")).strip()
        L.append(f"  - {k}: one of {', '.join(_val(v) for v in d['values'])} (stock {_val(d['stock'])}). {what[:1].upper() + what[1:]}.")
    L += ["- ops: a list of harness ops, each op at most once, or []. Whatever their order in the list, they are applied in the order "
          "zoom, tiles, answer, check:",
          '  - {"op": "zoom"}: a first short pass names one image region, then the question is answered with the original images plus '
          "that region cropped and enlarged as an extra image.",
          '  - {"op": "tiles", "n": 2 | 3}: each image is also given as n x n enlarged tiles.',
          '  - {"op": "check"}: after answering, one more pass shows the question with the proposed answer and asks to confirm or correct '
          "it, and the corrected answer is graded.",
          f"- On a question where the images with the zoom crop and the tiles would number more than {OPS_IMAGE_LIMIT}, or the prompt "
          "would not fit the context, the ops are not applied and the entry has no effect on that question.",
          "- Equal compute in judging: when an entry's generated tokens per question (all passes counted) exceed the tip's by a factor "
          "c >= 1.5, its gain is measured against the plurality answer of ceil(c) tip samples on the same question instead of the "
          "tip's mean.",
          "- The entry needs a procedure, settings, ops, or a combination, and must differ from the tip.",
          "- An entry is rejected without being run if its condition or procedure gives an option letter as an answer, "
          "states a fixed answer value for a kind of object or scene, names a dataset, a task category or a question id, "
          "or tells the executor to think longer or shorter."]
    return "\n".join(L)


def card_prompt(rows, parent, card_ids, tip, knobs, author_memory=None):
    """(system, user) text-only proposer request: (a) tip, (b) card, (c) affordance list, (d) answer format, then
    author_memory verbatim under '## Memory' when given. Fails closed: non-show card rows -> ValueError, leak -> LeakError."""
    _check_card(rows, parent, card_ids)
    system, segs, _ = _build(rows, parent, card_ids, tip, knobs, PARAMS_DEFAULT, author_memory)
    user = "".join(segs)
    guard(system, user, rows)
    return system, user


TEXT_ONLY_SENTENCE = "You see the questions as text only; the router and the executor see the images."


SOURCES_MIXED = "on held-out questions that you never see, drawn from the same sources as the questions shown to you."
SOURCES_SINGLE = ("on held-out questions that you never see: some come from the same source as the questions shown to you, "
                  "the others from different sources.")
assert SOURCES_MIXED in SYSTEM


def _system_for(p):
    """SYSTEM for a version's card parameters: the held-out sentence follows card_composition (coordinator 11-10: exact with
    single-source cards; mixed cards keep the Stage-1 sentence), the image sentence follows card_images."""
    base = SYSTEM if p.get("card_composition") == "mixed" else SYSTEM.replace(SOURCES_MIXED, SOURCES_SINGLE)
    if not p["card_images"]:
        return base
    lo = "the processor's own minimum" if p["card_min_tokens"] is None else f"{p['card_min_tokens']}"
    s = (f"You see each shown question's images, attached right before its question text and resized to between {lo} and "
         f"{p['card_max_tokens']} visual tokens per image; the router and the executor see the images at their own settings.")
    return base.replace(TEXT_ONLY_SENTENCE, s)


def _check_card(rows, parent, card_ids):
    bad = [i for i in card_ids if (rows.get(i) or {}).get("v4_role") != "show"]
    if bad:
        raise ValueError(f"card rows must be SHOW rows only: {len(bad)} others")
    miss = [i for i in card_ids if not _samples(parent, i)]
    if miss:
        raise ValueError(f"card rows without a read parent sample: {len(miss)}")


def _build(rows, parent, card_ids, tip, knobs, p, author_memory):
    """(system, user segments, question blocks). Text-only: one str segment equal to the Stage-1 user text."""
    lines, blocks = _card_lines(rows, parent, card_ids, p)
    segs = [_tip_block(tip, knobs) + "\n\n"] + _segments(lines)
    tail = [_affordance_block(knobs), ANSWER_FORMAT] + (["## Memory\n" + author_memory] if author_memory is not None else [])
    segs.append("\n\n" + "\n\n".join(tail))
    out = []
    for s in segs:
        if isinstance(s, str) and out and isinstance(out[-1], str):
            out[-1] += s
        else:
            out.append(s)
    return _system_for(p), out, blocks


def stock_image_tokens(parent, rid, n):
    """Image token counts of a row from its parent sample records (field image_tokens, a list in image order): the first
    record whose list has one number per image, else None."""
    for s in parent.get(rid) or []:
        it = s.get("image_tokens") if isinstance(s, dict) else None
        if isinstance(it, list) and len(it) == n and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in it):
            return it
    return None


def image_cost(parent, rid, n, p):
    """(tokens, images charged at card_max) of a question's n attached images: each image min(card_max, max(card_min or 0,
    its stock count)); card_max for every image when no parent record has the counts."""
    st = stock_image_tokens(parent, rid, n)
    if st is None:
        return n * p["card_max_tokens"], n
    return sum(min(p["card_max_tokens"], max(p["card_min_tokens"] or 0, t)) for t in st), 0


def _estimate(system, segs, blocks, rows, p, tok_est, parent=None):
    """(prompt tokens, images charged at card_max). Tokens: the text outside the question blocks at CHARS_PER_TOKEN plus
    each question's estimate: tok_est(row, params) when given, else its text at CHARS_PER_TOKEN plus image_cost."""
    text = system + "".join(s for s in segs if isinstance(s, str))
    inner = sum(len(t) for _, t, _ in blocks)
    q, fallback = [], 0
    for rid, t, n_img in blocks:
        if tok_est:
            q.append(tok_est(rows[rid], p))
        else:
            c, f = image_cost(parent or {}, rid, n_img, p) if n_img else (0, 0)
            q.append(len(t) / CHARS_PER_TOKEN + c)
            fallback += f
    return (len(text) - inner) / CHARS_PER_TOKEN + sum(q), fallback


def _drop_one(kept, wrong, target):
    """Index in kept to drop: the class (wrong / right) whose removal keeps the wrong share nearest the target, the last
    question of that class in card order; ties drop the class of the last question."""
    n, n_w = len(kept), sum(wrong[x] for x in kept)
    if n_w == 0 or n_w == n:
        return n - 1
    d_w = abs((n_w - 1) / (n - 1) - target)       # drop a wrong one
    d_r = abs(n_w / (n - 1) - target)             # drop a right one
    cls = wrong[kept[-1]] if abs(d_w - d_r) < 1e-12 else d_w < d_r
    return max(i for i, x in enumerate(kept) if wrong[x] == cls)


def card_messages(rows, parent, card_ids, tip, knobs, params, author_memory=None, max_prompt_tokens=40000, tok_est=None):
    """Chat messages of one proposer slot under a version's card parameters, and info.
    card_images False: [system, user] with str contents, identical to card_prompt (when nothing is dropped).
    card_images True: the user content is a list of parts ({"type": "text"} / {"type": "image", "path", "min_image_tokens",
    "max_image_tokens"}, engine_api.md §7), each question's images right before its "Question text".
    Questions are dropped from the end of the card (wrong / right mix kept nearest the card's own) until the token estimate
    fits max_prompt_tokens; at least one question stays (info["fits"] says whether it fit). Guarded (fails closed)."""
    p = params_of(params)
    _check_card(rows, parent, card_ids)
    wrong = {rid: not _is_right(_samples(parent, rid)) for rid in card_ids}
    target = sum(wrong.values()) / len(card_ids) if card_ids else 0.0
    kept = list(card_ids)
    while True:
        system, segs, blocks = _build(rows, parent, kept, tip, knobs, p, author_memory)
        est, fallback = _estimate(system, segs, blocks, rows, p, tok_est, parent)
        if est <= max_prompt_tokens or len(kept) <= 1:
            break
        kept.pop(_drop_one(kept, wrong, target))
    if p["card_images"]:
        user = [dict(type="text", text=s) if isinstance(s, str) else dict(s) for s in segs]
    else:
        user = "".join(segs)
    messages = [dict(role="system", content=system), dict(role="user", content=user)]
    g = guard(system, user, rows)
    kept_set = set(kept)
    info = dict(n_card=len(card_ids), n_shown=len(kept), dropped=len(card_ids) - len(kept),
                dropped_ids=[x for x in card_ids if x not in kept_set], card_ids=kept,
                n_wrong=sum(wrong[x] for x in kept), n_right=sum(not wrong[x] for x in kept),
                est_tokens=int(round(est)), max_prompt_tokens=max_prompt_tokens, fits=est <= max_prompt_tokens,
                n_images=sum(n for _, _, n in blocks), images_at_card_max=fallback, chars=g["chars"],
                temperature=p["temperature"], drafts=p["drafts"], params=p)
    return messages, info


# ------------------------------------------------------------------------------------------------ parse
FENCE = re.compile(r"```[ \t]*([A-Za-z]*)[ \t]*\n?(.*?)```", re.S)
TAIL_SCAN = 8000


def _tail_object(text):
    t = text.rstrip()
    if not t.endswith("}"):
        return None
    lo = max(0, len(t) - TAIL_SCAN)
    dec = json.JSONDecoder()
    for i in (m.start() for m in re.finditer(r"\{", t[lo:])):
        try:
            obj, end = dec.raw_decode(t, lo + i)
        except ValueError:
            continue
        if end == len(t) and isinstance(obj, dict):
            return t[lo + i:]
    return None


def _num(v):
    if isinstance(v, str):
        x = v.strip().lower()
        if x in ("null", "none"):
            return None
        if re.fullmatch(r"-?\d+", x):
            return int(x)
        if re.fullmatch(r"-?\d*\.\d+(e-?\d+)?|-?\d+e-?\d+", x):
            return float(x)
    return v


def extract_json(text):
    """(object, None) from the last ```json block (or a bare JSON object ending the text), or (None, 'kind: detail')."""
    if not isinstance(text, str) or not text.strip():
        return None, "empty: no text"
    blocks = [b for tag, b in FENCE.findall(text) if tag.lower() == "json" or (not tag and b.strip().startswith("{"))]
    src = blocks[-1] if blocks else _tail_object(text)
    if src is None:
        return None, "no_json: no ```json block and no JSON object at the end"
    try:
        obj = json.loads(src)
    except json.JSONDecodeError as e:
        return None, f"bad_json: {e.msg} at line {e.lineno} column {e.colno}"
    if not isinstance(obj, dict):
        return None, f"not_object: the json is a {type(obj).__name__}"
    return obj, None


def parse(text):
    """Last ```json block (or a bare JSON object ending the text) -> normalised candidate
    {why, condition, procedure (str | None), settings (dict), ops (list, engine order; [] when missing or null)[, extra]};
    errors are 'kind: detail'. A wrong type for ops is a format error here; unknown ops or values are validate problems."""
    obj, err = extract_json(text)
    if err:
        return None, err
    if "condition" not in obj:
        return None, "field: condition missing"
    c = obj["condition"]
    if not isinstance(c, str):
        return None, "field: condition must be a string"
    c = "always" if _is_always(c) else _ws(c)
    p = obj.get("procedure")
    if isinstance(p, str):
        p = p.strip()
        if p.lower() in ("", "null", "none"):
            p = None
    elif p is not None:
        return None, "field: procedure must be a string or null"
    s = obj.get("settings")
    if s is None:
        s = {}
    if not isinstance(s, dict):
        return None, "field: settings must be an object"
    o = obj.get("ops")
    if o is None:
        o = []
    if not isinstance(o, list) or not all(isinstance(x, dict) for x in o):
        return None, "field: ops must be a list of objects"
    why = obj.get("why")
    cand = dict(why=_ws(why) if why is not None else "", condition=c, procedure=p,
                settings={str(k).strip(): _num(v) for k, v in s.items()}, ops=canon_ops(o))
    extra = {k: v for k, v in obj.items() if k not in ("why", "condition", "procedure", "settings", "ops")}
    if extra:
        cand["extra"] = extra
    return cand, None


# ------------------------------------------------------------------------------------------------ validate
# v3 leakage precheck (harness/adapters/skill_store.DENY used by hw_round.critic + hboost.hardcoded_values), ported.
# The generic v3 row-id regex is replaced by the exact ids of the rows (it also hit fractions like 3/4).
DENY = [
    (r"\b(option|answer|letter|choice)\s*\(?(?-i:[A-H])\)?\s*(is|=|:)", "option letter given as an answer"),
    (r"\b(answer|option|choice|letter)\s*(is|=|:)\s*\(?(?-i:[A-H])\)?(?![A-Za-z])", "option letter given as an answer"),
    (r"\b(answer|choose|pick|select)\s+(?:option\s+)?\(?(?-i:[A-H])\)?(?![A-Za-z0-9])", "option letter given as an answer"),
    (r"\b(answer|count|is|equals|total)\s*(is\s*)?[:=]?\s*\d+\b", "numeral next to answer/count/is"),
    (r"\b(?:standard|always|usually|typically|normally|by default)\b[^.\n]{0,60}\d+", "fixed value for a kind of object or scene"),
    (r"\b[A-Za-z][A-Za-z ]{1,20}\s*=\s*\d+\b(?=[^\n]*[a-z])", "fixed value for a kind of object or scene"),
    (r"\b\d+\s+(?:rings|legs|columns|squares|pieces|sides|wheels|players)\b", "fixed value for a kind of object or scene"),
]
DENY_RE = [(re.compile(p, re.I), why) for p, why in DENY]

# AM1: effort is fixed. A procedure (or condition) that asks for longer / shorter / more / less thinking is rejected.
EFFORT = [
    r"\beffort\b",
    r"\b(think|reason|deliberate|ponder)\s+(?:much\s+|a\s+lot\s+|a\s+bit\s+|a\s+little\s+|even\s+|far\s+)?"
    r"(longer|shorter|harder|less|more|briefly|at\s+length|extensively|exhaustively|for\s+longer|deeper|deeply)\b",
    r"\b(more|less|longer|shorter|extended|extensive|extra|additional|minimal|maximum|maximal|brief|short|deeper)\s+"
    r"(thinking|reasoning|deliberation|thought)\b",
    r"\b(spend|take|use|allow|allocate)\s+(?:\w+\s+){0,2}?(more|less|extra|additional|minimal|little|longer|the\s+full)\s+"
    r"(time|tokens?|thinking|reasoning|thought)\b",
    r"\b(over|under)-?think\w*",
    r"\b(token|thinking|reasoning)\s+budget\b",
    r"\b(thinking|reasoning|thoughts?)\s+(short|brief|concise|minimal)\b",
]
EFFORT_RE = [re.compile(p, re.I) for p in EFFORT]
ID_TOKEN = re.compile(r"[A-Za-z0-9_.:\-]+(?:/[A-Za-z0-9_.:\-]+)+")
PIC_TOKEN = re.compile(r"(?<![0-9a-f])[0-9a-f]{24}(?![0-9a-f])")

_META = {}


def _meta(rows):
    """Per-rows cache: bench / category regexes, id sets and the guard's anchor indexes."""
    key = (id(rows), len(rows))
    m = _META.get(key)
    if m is not None and m["rows"] is rows:       # the cache holds the dict, so its id cannot be reused by another one
        return m
    R = list(rows.values())
    benches = sorted({str(r["bench"]) for r in R}, key=len, reverse=True)
    phr = {}
    for r in R:
        c = str(r.get("category") or "")
        p = _cat_norm(c)
        if len(p.split(" ")) >= CATEGORY_MIN_WORDS:
            phr.setdefault(p, c)
    cats = sorted(phr.items(), key=lambda x: len(x[0]), reverse=True)      # (normalised phrase, original name)
    show = [r for r in R if r.get("v4_role") == "show"]
    other = [r for r in R if r.get("v4_role") != "show"]
    show_txt = {_ws(r["prompt"]) for r in show}
    show_grp = {_ws(r["group"]) for r in show if r.get("group")}
    leak = {}
    for r in other:
        t = _ws(r["prompt"])
        if len(t) >= MIN_TEXT and t not in show_txt:
            leak.setdefault(t, ("question text", r["id"]))
        g = _ws(r.get("group") or "")
        if len(g) >= MIN_TEXT and g not in show_grp:
            leak.setdefault(g, ("group path", r["id"]))
    show_imgs = {pth for r in show for pth in (r.get("image_paths") or [])}
    for r in other:
        for pth in r.get("image_paths") or []:
            if len(pth) >= MIN_TEXT and pth not in show_imgs:
                leak.setdefault(_ws(pth), ("image path", r["id"]))
    allowed = {t for t in show_txt | show_grp if len(t) >= MIN_TEXT}

    def index(texts):
        ix = collections.defaultdict(list)
        for t in texts:
            ix[t[:ANCHOR]].append(t)
        return dict(ix)

    m = dict(
        rows=rows, show_imgs=show_imgs,
        bench_re=re.compile(r"\b(" + "|".join(map(re.escape, benches)) + r")\b", re.I) if benches else None,
        cats=cats,
        ids={r["id"] for r in R}, other_ids={r["id"] for r in other},
        other_pics={r["pic"] for r in other if r.get("pic")} - {r["pic"] for r in show if r.get("pic")},
        leak=leak, leak_ix=index(leak), allow_ix=index(allowed),
    )
    _META.clear()
    _META[key] = m
    return m


def _cat_norm(s):
    """Lower case, every run of space / '_' / '/' -> one space (category phrases and texts alike)."""
    return CAT_SEP.sub(" ", str(s).lower()).strip()


def _ids_in(text, idset):
    out = set()
    for tok in ID_TOKEN.findall(text):
        parts = tok.split("/")
        for a, b in zip(parts, parts[1:]):
            for x in {f"{a}/{b}", f"{a.lstrip('.:-')}/{b.rstrip('.:-')}", f"{a}/{b.rstrip('.:-')}"}:
                if x in idset:
                    out.add(x)
    return out


WORD = re.compile(r"[a-z0-9]+")


def card_words(rows, card_ids):
    """Words (lower case, [a-z0-9]+) that occur in at least half of the card's question texts (coordinator 11-10: a bench or
    category name that is also an ordinary word of the shown questions' own text is content, not a label)."""
    ids = list(card_ids or [])
    if not ids:
        return set()
    cnt = collections.Counter()
    for i in ids:
        cnt.update(set(WORD.findall(str(rows[i].get("prompt", "")).lower())))
    return {w for w, n in cnt.items() if 2 * n >= len(ids)}


def _label_form(t, c):
    """True if text t uses category name c as a label (coordinator 11-10): verbatim, or its words joined by _ or / (any case), or its
    words joined by spaces with the metadata's capitals (only when the name has capitals). An ordinary lower-case phrase is not a label."""
    ws = [w for w in CAT_SEP.split(str(c).strip()) if w]
    if not ws:
        return False
    b0, b1 = r"(?<![\w])", r"(?![\w])"
    if re.search(b0 + re.escape(str(c).strip()) + b1, t):
        return True
    if len(ws) > 1 and any(("_" in x.group(0) or "/" in x.group(0)) for x in
                           re.finditer(b0 + r"(?:\s*[_/]\s*|\s+)".join(map(re.escape, ws)) + b1, t, re.I)):
        return True                                   # words joined by separators of which at least one is _ or /
    if any(ch.isupper() for ch in str(c)) and re.search(b0 + r"\s+".join(map(re.escape, ws)) + b1, t):
        return True
    return False


def leak_precheck(text, rows, allowed_words=None):
    """Hit reasons for one condition or procedure text (empty list = clean). allowed_words (card_words of the writer's card): a bench
    name or a category phrase whose words all occur there is allowed; names that never occur in the questions' own text stay rejected."""
    t = str(text or "")
    m = _meta(rows)
    aw = allowed_words or set()
    hits = [why for rx, why in DENY_RE if rx.search(t)]
    if m["bench_re"] is not None and any(x.group(0).lower() not in aw for x in m["bench_re"].finditer(t)):
        hits.append("bench name")
    low = _cat_norm(t)
    for p, c in m["cats"]:
        if p not in CATEGORY_EXEMPT and p in low and _label_form(t, c):
            if aw and all(w in aw for w in WORD.findall(p)):
                continue
            hits.append(f"category name {c!r}")
    if _ids_in(t, m["ids"]):
        hits.append("row id")
    return list(dict.fromkeys(hits))


def effort_hits(text):
    return [f"asks for more or less thinking ({rx.search(text).group(0)!r}); effort is fixed" for rx in EFFORT_RE if rx.search(str(text or ""))]


def ops_problems(ops):
    """Problems of an ops list (engine_api.md §7): unknown op, a field or value outside OPS, an op given twice, more than one
    tiles. All start with "ops:" (menu problems, not format problems)."""
    if ops is None:
        return []
    if not isinstance(ops, list):
        return ["ops: must be a list of objects"]
    P, seen = [], collections.Counter()
    for o in ops:
        if not isinstance(o, dict):
            P.append(f"ops: {str(o)[:40]!r} is not an object")
            continue
        o = _norm_op(o)
        name = o.get("op")
        if name not in OPS:
            P.append(f"ops: unknown op {name!r} (the ops are {', '.join(OPS_ORDER)})")
            continue
        seen[name] += 1
        for k in sorted(set(o) - {"op"} - set(OPS[name])):
            P.append(f"ops: {name} has no field {k!r}")
        for k, vals in OPS[name].items():
            if k not in o:
                P.append(f"ops: {name} needs {k}, one of {vals}")
            elif not _in_menu(o[k], vals):
                P.append(f"ops: {name} {k} = {o[k]!r} is not one of {vals}")
    for name, n in seen.items():
        if n > 1:
            P.append("ops: more than one tiles" if name == "tiles" else f"ops: {name} is given {n} times")
    return P


def validate(cand, tip, knobs, rows, allowed_words=None):
    """Problems of a parsed candidate; [] = valid. allowed_words: see leak_precheck (None = no exemption)."""
    if not isinstance(cand, dict):
        return ["no candidate"]
    S = knobs["settings"]
    P = []
    for k in cand.get("extra") or {}:
        P.append("effort is fixed and is not a knob" if "effort" in str(k).lower() else f"unknown field {k!r}")
    c, p, s = cand.get("condition"), cand.get("procedure"), cand.get("settings")
    if not isinstance(c, str) or not 1 <= len(c.strip()) <= COND_MAX:
        P.append(f"condition length {len(c.strip()) if isinstance(c, str) else 'n/a'} not in 1..{COND_MAX}")
    if p is not None and not isinstance(p, str):
        P.append("procedure must be a string or null")
        p = None
    if isinstance(p, str) and not p.strip():
        p = None
    if p is not None and len(p) > PROC_MAX:
        P.append(f"procedure has {len(p)} chars > {PROC_MAX}")
    if s is None:
        s = {}
    if not isinstance(s, dict):
        P.append("settings must be an object")
        s = {}
    good = {}
    for k, v in s.items():
        kl = str(k).lower()
        if "effort" in kl or "thinking" in kl or "reasoning" in kl:
            P.append(f"setting {k!r}: effort is fixed and is not a knob")
        elif k not in S:
            P.append(f"unknown knob {k!r}")
        elif not any(_same(v, x) for x in S[k]["values"]):
            P.append(f"{k} = {v!r} is not one of {S[k]['values']}")
        else:
            good[k] = v
    ops = cand.get("ops")
    op_probs = ops_problems(ops)
    P += op_probs
    if p is None and not s and not ops:
        P.append("empty payload: needs a procedure, settings or ops")
    else:
        base, entries = _tip_view(tip, knobs)
        same_set = len(good) == len(s) and all(_same(v, base[k]) for k, v in good.items())
        if p is None and not ops and good and same_set:
            P.append("no change vs the tip: every setting equals the tip's")
        elif p is None and ops and not op_probs and same_set:
            always = merge_ops([e.get("ops") for e in entries if _is_always(e.get("condition"))])
            if merge_ops([always, ops]) == always:
                P.append("no change vs the tip: " + ("every setting equals the tip's and " if s else "")
                         + "the tip applies these ops on every question already")
        key = _entry_key(dict(condition=c, procedure=p, settings=s, ops=ops))
        P += [f"same as tip entry {j}" for j, e in enumerate(entries, 1) if _entry_key(e) == key]
    for what, t in (("condition", c), ("procedure", p)):
        if isinstance(t, str) and t.strip():
            P += [f"{what}: {h}" for h in effort_hits(t)]
            P += [f"{what}: leakage precheck: {h}" for h in leak_precheck(t, rows, allowed_words)]
    return P


def pick_draft(texts, tip, knobs, rows, allowed_words=None):
    """drafts > 1: (index, cand, None) of the first draft, in order, that parses and validates; selects on nothing else.
    None answered drafts count as failed. No valid draft: (None, None, 'no_valid_draft: <k>: <error> | ...').
    allowed_words: as in validate (card_words of the slot's card)."""
    errs = []
    for k, t in enumerate(texts or []):
        cand, err = parse(t) if isinstance(t, str) else (None, "unanswered")
        if err is None:
            probs = validate(cand, tip, knobs, rows, allowed_words)
            if not probs:
                return k, cand, None
            err = "invalid: " + "; ".join(probs)
        errs.append(f"{k}: {err}")
    return None, None, "no_valid_draft: " + (" | ".join(errs) if errs else "no drafts")


# ------------------------------------------------------------------------------------------------ guard
def _scan(req, ix):
    """(start, end, text) of every occurrence in req of a text of the anchor index ix."""
    get, K = ix.get, ANCHOR
    for i in range(len(req) - K + 1):
        lst = get(req[i:i + K])
        if lst:
            for t in lst:
                if req.startswith(t, i):
                    yield i, i + len(t), t


def _flatten(content, imgs):
    """Text of a message content (str or list of parts); image part paths go to imgs."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise ValueError(f"message content must be a str or a list of parts, not {type(content).__name__}")
    texts = []
    for x in content:
        t = x.get("type") if isinstance(x, dict) else None
        if t == "text":
            texts.append(str(x.get("text", "")))
        elif t == "image":
            imgs.append(x.get("path"))
        else:
            raise ValueError(f"unknown content part {str(x)[:80]}")
    return "".join(texts)


def guard_messages(messages, rows):
    """guard() over a chat message list (system and user contents in order)."""
    sys_ = "\n".join(m["content"] for m in messages if m.get("role") == "system" and isinstance(m["content"], str))
    rest = [m["content"] for m in messages if not (m.get("role") == "system" and isinstance(m["content"], str))]
    parts = []
    for c in rest:
        parts += [dict(type="text", text=c)] if isinstance(c, str) else list(c) + [dict(type="text", text="\n")]
    return guard(sys_, parts, rows)


def guard(system, user, rows):
    """Leak audit of a full proposer request (B2Guard.audit + assert_no_leak, ported): no id of a non-show row, no
    question text (>= 16 chars, whitespace-normalised) of a non-show row unless it is identical to a show row's text or
    the occurrence lies inside a shown show text; also no non-show-only picture key, group path or image path in the text.
    system / user: str or a list of content parts; every image part must be an image of a SHOW row. Raises LeakError."""
    m = _meta(rows)
    imgs = []
    raw = f"{_flatten(system, imgs)}\n{_flatten(user, imgs)}"
    badimg = [x for x in imgs if x not in m["show_imgs"]]
    if badimg:
        raise LeakError(f"image parts that are not images of SHOW rows: {len(badimg)} (e.g. {str(badimg[0])[-60:]})")
    ids = _ids_in(raw, m["other_ids"])
    if ids:
        raise LeakError(f"non-show row ids in request: {sorted(ids)[:5]} ({len(ids)})")
    pics = set(PIC_TOKEN.findall(raw)) & m["other_pics"]
    if pics:
        raise LeakError(f"non-show picture keys in request: {sorted(pics)[:3]} ({len(pics)})")
    req = _ws(raw)
    hits = list(_scan(req, m["leak_ix"]))
    if hits:
        spans = [(a, b) for a, b, _ in _scan(req, m["allow_ix"])]
        for a, b, t in hits:
            if not any(x <= a and b <= y for x, y in spans):
                what, rid = m["leak"][t]
                raise LeakError(f"{what} of non-show row {rid!r} in request (chars {a}-{b})")
    return dict(chars=len(raw), checked_texts=len(m["leak"]), excused=len(hits), images=len(imgs))
