"""selfedit v4, proposer module (Stage 1; design_1110_v4.md §3-§4, v4_build_spec_1110.md AM1-AM6).

One proposer slot = one card of SHOW rows -> one prompt (neutral task frame + facts) -> one candidate entry
(condition, payload). Functions:
  load_rows(path)                                   -> {row id: row}
  cards(rows, parent, k, seed, n_wrong=8, n_right=4) -> [row ids]      (SHOW rows only, per-slot draw)
  card_prompt(rows, parent, card_ids, tip, knobs, author_memory=None) -> (system, user)   (guarded)
  parse(text)                                        -> (cand | None, error | None)
  validate(cand, tip, knobs, rows)                   -> [problems]     ([] = valid)
  guard(system, user, rows)                          -> info dict, raises LeakError
  leak_precheck(text, rows)                          -> [hit reasons]  (v3 hw_round precheck, ported)

parent: {row id: [result dicts]} with keys content, reasoning, finish, tokens, ok (engine_api.md §4; error lines are
unread and ignored). tip: None (stock) or {"settings": {knob: value}, "entries": [{"condition", "procedure",
"settings"}, ...]}. Effort is never a knob (AM1). Standard library only.
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


class LeakError(AssertionError):
    pass


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


def cards(rows, parent, k, seed, n_wrong=8, n_right=4):
    """SHOW rows for slot k: n_wrong currently wrong + n_right currently right, spread over benches round-robin in a
    per-slot shuffled bench order (one pointer for both draws, so per-bench totals differ by at most one). A short pool
    is filled from the other one. SHOW rows without a read sample are skipped. Returned in a shuffled order."""
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
    c = e.get("condition")
    c = "always" if _is_always(c) else _ws(c or "").lower()
    p = e.get("procedure")
    p = _ws(p) if isinstance(p, str) and p.strip() else None
    s = tuple(sorted((str(k), json.dumps(v)) for k, v in (e.get("settings") or {}).items()))
    return c, p, s


# ------------------------------------------------------------------------------------------------ prompt
SYSTEM = """You write one change to the harness of a vision-language model, called the executor.

The executor answers one question at a time. It receives the question's images and text, reasons, and then writes a final answer, which is graded against a gold answer.

The harness is an ordered list of entries. Each entry has a condition and a payload.
- The condition is either "always" or a sentence in words. For each question, a router reads that question's images and text and answers yes or no to the condition. "always" holds on every question.
- The payload is a procedure text, setting changes, or both. On a question where the condition holds, the procedure is added to the executor's prompt right before the question text, and the settings are used for that question. On a question where the condition does not hold, the entry has no effect.

You write exactly one new entry. It is added to the current harness, called the tip, and judged by its paired gain over the tip on held-out questions that you never see, drawn from the same sources as the questions shown to you. On each held-out question the tip with your entry and the tip alone both answer, and the gain is the difference in accuracy. Only questions on which your condition holds are read; on all other questions the two are identical and the difference is zero. The entry joins the harness only if its gain is credibly above zero.

The router decides the condition separately for each question, from that question's images and text alone, and the same router also runs on questions from benchmarks that are not shown here.

You see the questions as text only; the router and the executor see the images."""

ANSWER_FORMAT = """## Answer format
Reason first. Then end with exactly one json block:
```json
{"why": "...", "condition": "always" | "<words>", "procedure": "<text>" | null, "settings": {"<knob>": value, ...} | {}}
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


def _pick(vs, wrong_row):
    """Up to SAMPLES_SHOWN samples: first one of the row's majority outcome (a wrong one on a wrong row), then one of
    the other outcome where there is one, else the next sample."""
    first = next((s for s in vs if bool(s["ok"]) != wrong_row), None) or vs[0]
    rest = [s for s in vs if s is not first]
    second = next((s for s in rest if bool(s["ok"]) != bool(first["ok"])), None) or (rest[0] if rest else None)
    return [s for s in (first, second) if s is not None][:SAMPLES_SHOWN]


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
        L.append(f"Entries ({len(entries)}, in order; where several conditions hold on one question, their procedures are joined in this order):")
        for j, e in enumerate(entries, 1):
            st = e.get("settings") or {}
            L += [f"Entry {j}", f"- condition: {e.get('condition')}",
                  f"- procedure: {e.get('procedure') if e.get('procedure') else 'none'}",
                  "- settings: " + (", ".join(f"{k} = {_val(v)}" for k, v in st.items()) if st else "none")]
    return "\n".join(L)


def _card_block(rows, parent, card_ids):
    marks = []
    for rid in card_ids:
        vs = _samples(parent, rid)
        marks.append((rid, vs, _is_right(vs)))
    n_w = sum(not ok for _, _, ok in marks)
    ns = sorted({len(vs) for _, vs, _ in marks})
    times = f"{ns[0]} times" if len(ns) == 1 else "several times"
    L = [f"## Shown questions ({len(marks)}: {n_w} currently wrong, {len(marks) - n_w} currently right)",
         f"The tip answered each question {times} (its samples). A question is currently right when more than half of "
         f"its samples are right, and currently wrong otherwise. For up to {SAMPLES_SHOWN} samples of each question, the "
         f"end of the executor's reasoning and its final text are shown."]
    for q, (rid, vs, ok) in enumerate(marks, 1):
        r = rows[rid]
        L += ["", f"### Question {q}: currently {'right' if ok else 'wrong'} ({sum(bool(s['ok']) for s in vs)} of {len(vs)} samples right)",
              "Question text:", str(r["prompt"]).strip()]
        L += _gold_lines(r)
        for j, s in enumerate(_pick(vs, not ok), 1):
            L += _sample_lines(j, s)
    return "\n".join(L)


def _affordance_block(knobs):
    S = knobs["settings"]
    L = ["## What an entry can contain",
         f'- condition: "always", or one sentence of at most {COND_MAX} characters that can be answered yes or no from one question\'s images and text.',
         f"- procedure: a text of at most {PROC_MAX} characters, or null.",
         "- settings: an object with zero or more of these keys, each set to one of its listed values:"]
    for k, d in S.items():
        what = str(d.get("what", "")).strip()
        L.append(f"  - {k}: one of {', '.join(_val(v) for v in d['values'])} (stock {_val(d['stock'])}). {what[:1].upper() + what[1:]}.")
    L += ["- The entry needs a procedure, settings, or both, and must differ from the tip.",
          "- An entry is rejected without being run if its condition or procedure gives an option letter as an answer, "
          "states a fixed answer value for a kind of object or scene, names a dataset, a task category or a question id, "
          "or tells the executor to think longer or shorter."]
    return "\n".join(L)


def card_prompt(rows, parent, card_ids, tip, knobs, author_memory=None):
    """(system, user) text-only proposer request: (a) tip, (b) card, (c) affordance list, (d) answer format, then
    author_memory verbatim under '## Memory' when given. Fails closed: non-show card rows -> ValueError, leak -> LeakError."""
    bad = [i for i in card_ids if (rows.get(i) or {}).get("v4_role") != "show"]
    if bad:
        raise ValueError(f"card rows must be SHOW rows only: {len(bad)} others")
    miss = [i for i in card_ids if not _samples(parent, i)]
    if miss:
        raise ValueError(f"card rows without a read parent sample: {len(miss)}")
    parts = [_tip_block(tip, knobs), _card_block(rows, parent, card_ids), _affordance_block(knobs), ANSWER_FORMAT]
    if author_memory is not None:
        parts.append("## Memory\n" + author_memory)
    user = "\n\n".join(parts)
    guard(SYSTEM, user, rows)
    return SYSTEM, user


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


def parse(text):
    """Last ```json block (or a bare JSON object ending the text) -> normalised candidate
    {why, condition, procedure (str | None), settings (dict)[, extra]}; errors are 'kind: detail'."""
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
    why = obj.get("why")
    cand = dict(why=_ws(why) if why is not None else "", condition=c, procedure=p,
                settings={str(k).strip(): _num(v) for k, v in s.items()})
    extra = {k: v for k, v in obj.items() if k not in ("why", "condition", "procedure", "settings")}
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
    allowed = {t for t in show_txt | show_grp if len(t) >= MIN_TEXT}

    def index(texts):
        ix = collections.defaultdict(list)
        for t in texts:
            ix[t[:ANCHOR]].append(t)
        return dict(ix)

    m = dict(
        rows=rows,
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


def leak_precheck(text, rows):
    """Hit reasons for one condition or procedure text (empty list = clean)."""
    t = str(text or "")
    m = _meta(rows)
    hits = [why for rx, why in DENY_RE if rx.search(t)]
    if m["bench_re"] is not None and m["bench_re"].search(t):
        hits.append("bench name")
    low = _cat_norm(t)
    for p, c in m["cats"]:
        if p not in CATEGORY_EXEMPT and p in low and re.search(r"(?<!\w)" + re.escape(p) + r"(?!\w)", low):
            hits.append(f"category name {c!r}")
    if _ids_in(t, m["ids"]):
        hits.append("row id")
    return list(dict.fromkeys(hits))


def effort_hits(text):
    return [f"asks for more or less thinking ({rx.search(text).group(0)!r}); effort is fixed" for rx in EFFORT_RE if rx.search(str(text or ""))]


def validate(cand, tip, knobs, rows):
    """Problems of a parsed candidate; [] = valid."""
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
    if p is None and not s:
        P.append("empty payload: needs a procedure or settings")
    else:
        base, entries = _tip_view(tip, knobs)
        if p is None and good and len(good) == len(s) and all(_same(v, base[k]) for k, v in good.items()):
            P.append("no change vs the tip: every setting equals the tip's")
        key = _entry_key(dict(condition=c, procedure=p, settings=s))
        P += [f"same as tip entry {j}" for j, e in enumerate(entries, 1) if _entry_key(e) == key]
    for what, t in (("condition", c), ("procedure", p)):
        if isinstance(t, str) and t.strip():
            P += [f"{what}: {h}" for h in effort_hits(t)]
            P += [f"{what}: leakage precheck: {h}" for h in leak_precheck(t, rows)]
    return P


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


def guard(system, user, rows):
    """Leak audit of a full proposer request (B2Guard.audit + assert_no_leak, ported): no id of a non-show row, no
    question text (>= 16 chars, whitespace-normalised) of a non-show row unless it is identical to a show row's text or
    the occurrence lies inside a shown show text; also no non-show-only picture key or group path. Raises LeakError."""
    m = _meta(rows)
    raw = f"{system}\n{user}"
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
    return dict(chars=len(raw), checked_texts=len(m["leak"]), excused=len(hits))
