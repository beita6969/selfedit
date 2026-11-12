"""selfedit v4, the incumbent proposer's one self-edit per generation (v4_build_spec_1110.md AM4, AM7, AM8).

A proposer version = {"id", "memory": none|trajectory|lessons|trace, "lessons": str | None, "params": card parameters
(proposer.PARAMS_MENU), "transfers": [ids of accepted executor edits]}. Each generation the incumbent writes ONE edit of
itself from a neutral frame plus the affordance list; the system builds the challenger (apply_selfedit) and the L1
contrast (yield_l1) decides. Functions:
  selfedit_messages(version, outcome_table, menu, accepted_executor_edits, rows=None, max_table_chars=40000) -> messages
  parse_selfedit(text)                                         -> (edit | None, error | None)
  validate_selfedit(edit, version, menu, accepted_ids, rows)   -> [problems]   ([] = valid)
  apply_selfedit(version, edit, new_id=None)                   -> challenger version
  memory_text(version, outcome_rows, trace_items, rows=None, info=None) -> str | None   (author_memory; guarded)
  card_lines(card, rows)                                       -> [lines]      (trace memory: an entry's card, SHOW rows only)
Outcome rows: dicts with gen, author ("you" | "another proposer" | "blind"), kind, condition, procedure, settings, ops
(optional list of harness ops), routed_share, rows_read, D, lfsr, accepted, and optionally card (trace memory only). The outcome table itself never
carries row content; only trace memory shows SHOW question texts (AM8). Standard library only.
"""
import copy
import json
import os
import re

import proposer as P

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROWS = os.path.join(HERE, "..", "data", "learn.jsonl")
MEMORY_KINDS = ("none", "trajectory", "lessons", "trace")
LESSONS_MAX = 1500
TRACE_MAX = 3            # trace examples shown (AM8 "up to a few")
TRACE_TAIL = 400         # chars of each transcript tail
TRAJECTORY_CAP = 6000    # chars of trajectory memory, newest rows kept (coordinator 11-10)
TRACE_ENTRIES_CAP = 12000  # chars of trace memory's entry list (entries + card lines), newest kept (builder's choice)
CARD_Q_CHARS = 80        # chars of a card question's text in trace memory (coordinator 11-10)
CARD_LEGEND = ("Under an entry: the shown questions of its card, each with the first 80 characters of its text, whether the "
               "entry's condition held on it (router yes / no), and whether the tip got it right.")
VERSION_DEFAULT = dict(id="P0", memory="none", lessons=None, params=dict(P.PARAMS_DEFAULT), transfers=[])

SYSTEM = """You are a proposer: a copy of a vision-language model that writes entries for the harness of another copy of the model, called the executor. An entry is a condition in words (or "always") with a procedure text, setting changes and/or harness ops; a router decides for each question whether the condition holds. Each entry is judged by its paired gain over the executor's current harness on held-out questions, and joins the harness only if that gain is credibly above zero. A slot is one entry written from one card of shown questions.

A proposer version has three parts:
- Memory, one of:
  - none: nothing is added to its prompt;
  - trajectory: the outcome table of earlier generations (aggregate results only) is added to its prompt;
  - lessons: a text of at most 1500 characters, written by the version itself, is added to its prompt;
  - trace: its own earlier entries with their aggregate results, and for up to 3 of them the end of the executor's answer with and without the entry on a shown question where the two answers differed, are added to its prompt.
- Card parameters: how its cards are built, how it samples its answers and how many trial rounds each slot gets (listed below with their allowed values).
- Transfers: edits that the executor has accepted, applied to the proposer as well.

Each generation you write exactly one edit of your own version. The edited version, the challenger, then writes entries beside your current version in the same generations. The two are compared on the expected accepted gain per slot of the entries each writes: if the challenger's is credibly higher, it replaces your current version; if it is credibly lower, the challenger is dropped; otherwise both keep writing."""

LEGEND = ("Columns: generation | author (you = your lineage, another proposer, or blind = one setting changed at random) | kind | "
          "condition | procedure | settings | ops | routed share (share of held-out questions on which the condition held) | "
          "held-out questions read | D (paired gain over the tip on the whole held-out set) | "
          "lfsr (posterior probability that the true gain is at most zero) | accepted")

ANSWER_FORMAT = """## Answer format
Reason first. Then end with exactly one json block holding one object in one of these forms:
- {"why": "...", "type": "memory", "memory": "none" | "trajectory" | "trace"}
- {"why": "...", "type": "memory", "memory": "lessons", "lessons": "<text, at most 1500 characters>"}
- {"why": "...", "type": "param", "param": "<name>", "value": <value from the menu>}
- {"why": "...", "type": "transfer", "edit": "<id of an accepted executor edit from the list>"}"""

_ROWS = {}


def _rows(rows):
    if rows is not None:
        return rows
    if "r" not in _ROWS:
        if not os.path.exists(DEFAULT_ROWS):
            raise FileNotFoundError(f"rows are needed for the leak guard and {DEFAULT_ROWS} does not exist: pass rows")
        _ROWS["r"] = P.load_rows(DEFAULT_ROWS)
    return _ROWS["r"]


# ------------------------------------------------------------------------------------------------ rendering
def _v(x):
    return json.dumps(x)


def _f(x, fmt):
    return "-" if x is None else format(x, fmt)


def _settings(s):
    return ", ".join(f"{k} = {_v(v)}" for k, v in s.items()) if s else "-"


def _ops(o):
    return ", ".join(_v(x) for x in P.canon_ops(o)) if o else "-"


def outcome_line(r, author=True):
    g = r.get("gen")
    parts = [g if isinstance(g, str) and g.startswith("g") else f"g{g}"]
    if author:
        parts.append(str(r.get("author") or "-"))
    acc = r.get("accepted")
    parts += [str(r.get("kind") or "-"), "cond: " + (P._ws(r.get("condition")) if r.get("condition") else "-"),
              "proc: " + (P._ws(r.get("procedure")) if r.get("procedure") else "-"), "set: " + _settings(r.get("settings") or {}),
              "ops: " + _ops(r.get("ops")), f"routed {_f(r.get('routed_share'), '.2f')}", f"read {r.get('rows_read') if r.get('rows_read') is not None else '-'}",
              f"D {_f(r.get('D'), '+.3f')}", f"lfsr {_f(r.get('lfsr'), '.2f')}", "accepted " + ("-" if acc is None else ("yes" if acc else "no"))]
    return " | ".join(parts)


def _gen_key(r):
    g = r.get("gen")
    try:
        return int(str(g).lstrip("g"))
    except ValueError:
        return 0


def _fit_groups(groups, budget):
    """Newest groups (strings, oldest first) whose lines fit in budget chars -> (kept, number left out)."""
    if budget is None:
        return list(groups), 0
    kept, used = [], 0
    for g in reversed(groups):
        if used + len(g) + 1 > budget:
            break
        kept.append(g)
        used += len(g) + 1
    return kept[::-1], len(groups) - len(kept)


def _legend(author):
    return LEGEND if author else LEGEND.replace(" | author (you = your lineage, another proposer, or blind = one setting changed at random)", "")


def outcome_block(table, author=True, max_chars=None, stats=None):
    """Compact table, oldest first. max_chars caps the whole block (legend and note included): the oldest rows are left out
    and counted. stats (dict, optional) receives rows_total, rows_kept, rows_left_out."""
    rows_ = sorted(table or [], key=_gen_key)
    lines = [outcome_line(r, author) for r in rows_]
    head = _legend(author)
    note = lambda n: f"({n} earlier rows left out for length)\n" if n else ""
    budget = None if max_chars is None else max_chars - len(head) - 1 - len(note(len(lines)))
    kept, omitted = _fit_groups(lines, budget)
    if stats is not None:
        stats.update(rows_total=len(lines), rows_kept=len(kept), rows_left_out=omitted)
    return head + "\n" + note(omitted) + "\n".join(kept or ["(no rows yet)"])


def _version_block(version, menu):
    p = P.params_of(version.get("params"), menu)
    L = ["## Your current version", f"- memory: {version.get('memory', 'none')}"]
    if version.get("memory") == "lessons" or version.get("lessons"):
        L.append("- lessons text:\n" + (version.get("lessons") or "(empty)"))
    L.append("- card parameters: " + ", ".join(f"{k} = {_v(v)}" for k, v in p.items()))
    tr = version.get("transfers") or []
    L.append("- transfers: " + (", ".join(map(str, tr)) if tr else "none"))
    return "\n".join(L)


def _edits_block(edits):
    L = ["## Accepted executor edits that can be transferred"]
    if not edits:
        return "\n".join(L + ["None yet."])
    for e in edits:
        L.append(f"- {e.get('id')}: generation {e.get('gen', '-')}, author {e.get('author', '-')}, {e.get('kind', '-')}; "
                 f"condition: {e.get('condition') or '-'}; procedure: {e.get('procedure') or '-'}; settings: {_settings(e.get('settings') or {})}; "
                 f"ops: {_ops(e.get('ops'))}; D {_f(e.get('D'), '+.3f')}, lfsr {_f(e.get('lfsr'), '.2f')}")
    return "\n".join(L)


def _affordance_block(version, menu, edits):
    p = P.params_of(version.get("params"), menu)
    L = ["## What the edit can be",
         f"- memory: set the memory to none, trajectory or trace; or to lessons with a new lessons text of at most {LESSONS_MAX} characters (the whole text is replaced).",
         "- param: set one card parameter to one of its values:"]
    for k, d in menu.items():
        vals = P._menu_values(menu, k)
        what = str(d.get("what", "")) if isinstance(d, dict) else ""
        L.append(f"  - {k}: one of {', '.join(_v(x) for x in vals)} (now {_v(p[k])}). {what[:1].upper() + what[1:]}.")
    L.append("- transfer: apply one accepted executor edit from the list above, by its id." if edits
             else "- transfer: possible once the executor has accepted an edit (none yet).")
    L += ["- The edit must change the version.",
          "- Lessons are rejected if they name a dataset, a task category or a question id, give an option letter as an answer, "
          "state a fixed answer value for a kind of object or scene, or tell the executor to think longer or shorter."]
    return "\n".join(L)


def selfedit_messages(version, outcome_table, menu, accepted_executor_edits, rows=None, max_table_chars=40000):
    """[system, user] text messages for the incumbent's one self-edit. Guarded when rows are given (or found)."""
    menu = menu or P.PARAMS_MENU
    edits = list(accepted_executor_edits or [])
    tab = outcome_block(outcome_table, True, max_table_chars)
    gens = sorted({_gen_key(r) for r in outcome_table or []})
    head = f"## Outcome table ({len(outcome_table or [])} entries" + (f", generations {gens[0]}-{gens[-1]})" if gens else ")")
    user = "\n\n".join([_version_block(version, menu), head + "\n" + tab, _edits_block(edits),
                        _affordance_block(version, menu, edits), ANSWER_FORMAT])
    messages = [dict(role="system", content=SYSTEM), dict(role="user", content=user)]
    P.guard_messages(messages, _rows(rows))
    return messages


# ------------------------------------------------------------------------------------------------ parse / validate / apply
def _norm_value(v):
    if isinstance(v, str):
        x = v.strip().lower()
        if x in ("true", "false"):
            return x == "true"
    return P._num(v)


def parse_selfedit(text):
    """Last ```json block -> {"why", "type", ...}; errors are 'kind: detail'."""
    obj, err = P.extract_json(text)
    if err:
        return None, err
    t = str(obj.get("type") or "").strip().lower()
    why = P._ws(obj.get("why")) if obj.get("why") is not None else ""
    used = {"why", "type"}
    if t == "memory":
        mem = str(obj.get("memory") or "").strip().lower()
        if mem not in MEMORY_KINDS:
            return None, f"field: memory must be one of {', '.join(MEMORY_KINDS)}"
        edit = dict(why=why, type="memory", memory=mem)
        used.add("memory")
        if mem == "lessons":
            if not isinstance(obj.get("lessons"), str):
                return None, "field: lessons text missing"
            edit["lessons"] = obj["lessons"].strip()
            used.add("lessons")
        elif obj.get("lessons") is None:
            used.add("lessons")
    elif t == "param":
        if not isinstance(obj.get("param"), str) or "value" not in obj:
            return None, "field: param needs a name and a value"
        edit = dict(why=why, type="param", param=obj["param"].strip(), value=_norm_value(obj["value"]))
        used |= {"param", "value"}
    elif t == "transfer":
        if obj.get("edit") is None or isinstance(obj.get("edit"), (dict, list)):
            return None, "field: transfer needs the id of an accepted executor edit"
        edit = dict(why=why, type="transfer", edit=str(obj["edit"]).strip())
        used.add("edit")
    else:
        return None, "field: type must be memory, param or transfer"
    extra = {k: v for k, v in obj.items() if k not in used}
    if extra:
        edit["extra"] = extra
    return edit, None


def validate_selfedit(edit, version, menu, accepted_ids, rows):
    """Problems of a parsed self-edit; [] = valid."""
    if not isinstance(edit, dict):
        return ["no edit"]
    menu = menu or P.PARAMS_MENU
    rows = _rows(rows)
    out = []
    for k in edit.get("extra") or {}:
        out.append("effort is fixed and is not a parameter" if "effort" in str(k).lower() else f"unknown field {k!r}")
    t = edit.get("type")
    if t == "memory":
        mem, cur = edit.get("memory"), version.get("memory", "none")
        if mem not in MEMORY_KINDS:
            out.append(f"memory {mem!r} is not one of {MEMORY_KINDS}")
        if mem == "lessons":
            les = edit.get("lessons")
            if not isinstance(les, str) or not 1 <= len(les.strip()) <= LESSONS_MAX:
                out.append(f"lessons length {len(les.strip()) if isinstance(les, str) else 'n/a'} not in 1..{LESSONS_MAX}")
            if isinstance(les, str) and les.strip():
                out += [f"lessons: {h}" for h in P.effort_hits(les)]
                out += [f"lessons: leakage precheck: {h}" for h in P.leak_precheck(les, rows)]
                try:
                    P.guard("", les, rows)
                except P.LeakError as e:
                    out.append(f"lessons: leak guard: {e}")
            if cur == "lessons" and isinstance(les, str) and P._ws(les) == P._ws(version.get("lessons") or ""):
                out.append("no change: the same lessons text")
        elif mem == cur:
            out.append(f"no change: memory is already {cur}")
    elif t == "param":
        name, val = edit.get("param"), edit.get("value")
        if name not in menu:
            out.append("effort is fixed and is not a parameter" if "effort" in str(name).lower() else f"unknown card parameter {name!r}")
        elif not P._in_menu(val, P._menu_values(menu, name)):
            out.append(f"{name} = {val!r} is not one of {P._menu_values(menu, name)}")
        elif P._in_menu(val, [P.params_of(version.get("params"), menu)[name]]):
            out.append(f"no change: {name} is already {val!r}")
    elif t == "transfer":
        eid = edit.get("edit")
        if eid not in {str(x) for x in accepted_ids or []}:
            out.append(f"transfer {eid!r} is not an accepted executor edit in the list")
        elif eid in {str(x) for x in version.get("transfers") or []}:
            out.append(f"no change: {eid!r} is already transferred")
    else:
        out.append(f"unknown edit type {t!r}")
    return out


def apply_selfedit(version, edit, new_id=None):
    """The challenger: a copy of the version with the edit applied (validate first)."""
    v = copy.deepcopy(version)
    v["params"] = P.params_of(v.get("params"))
    v.setdefault("transfers", [])
    if edit["type"] == "memory":
        v["memory"] = edit["memory"]
        if edit["memory"] == "lessons":
            v["lessons"] = edit["lessons"]
    elif edit["type"] == "param":
        v["params"][edit["param"]] = edit["value"]
    elif edit["type"] == "transfer":
        v["transfers"] = list(v["transfers"]) + [edit["edit"]]
    v["parent"], v["edit"] = version.get("id"), {k: x for k, x in edit.items() if k != "extra"}
    v["id"] = new_id or f"{version.get('id', 'P')}+{edit['type']}"
    return v


# ------------------------------------------------------------------------------------------------ memory
def memory_text(version, outcome_rows, trace_items, rows=None, info=None):
    """author_memory block for proposer.card_messages, by the version's memory kind; guarded (raises LeakError).
    trajectory: the outcome table, aggregates only, capped at TRAJECTORY_CAP chars (newest rows kept).
    trace: the version's own entries (author "you") with aggregates and, where an outcome row has "card" = [{"id",
    "routed": true | false | null, "tip_right"}], one line per shown question of that entry's card (card_lines); entries
    capped at TRACE_ENTRIES_CAP chars (newest kept); then up to TRACE_MAX with / without examples from trace_items.
    info (dict, optional) receives info["memory"] = {kind, chars, cap, rows_total, rows_kept, rows_left_out}."""
    kind = version.get("memory", "none")
    st = dict(kind=kind)
    if kind == "none":
        return _done(info, st, None)
    rows = _rows(rows)
    if kind == "lessons":
        if not (version.get("lessons") or "").strip():
            return _done(info, st, None)
        text = "Lessons written by this proposer version:\n" + version["lessons"].strip()
    elif kind == "trajectory":
        head = "Outcome table of earlier generations (aggregate results only):\n"
        text = head + outcome_block(outcome_rows, True, TRAJECTORY_CAP - len(head), st)
        st["cap"] = TRAJECTORY_CAP
    elif kind == "trace":
        own = sorted([r for r in outcome_rows or [] if r.get("author", "you") == "you"], key=_gen_key)
        groups = ["\n".join([outcome_line(r, False)] + card_lines(r.get("card"), rows)) for r in own]
        kept, omitted = _fit_groups(groups, TRACE_ENTRIES_CAP)
        st.update(rows_total=len(groups), rows_kept=len(kept), rows_left_out=omitted, cap=TRACE_ENTRIES_CAP)
        L = ["Your earlier entries and their aggregate results:", _legend(False)]
        if any(r.get("card") for r in own):
            L.append(CARD_LEGEND)
        if omitted:
            L.append(f"({omitted} earlier entries left out for length)")
        L += kept or ["(no entries yet)"]
        items = list(trace_items or [])[:TRACE_MAX]
        if items:
            L.append(f"\nShown questions on which the executor's answer changed with one of your entries ({len(items)}):")
        for j, it in enumerate(items, 1):
            mark = lambda k: "" if it.get(k) is None else (" (right)" if it[k] else " (wrong)")
            L += [f"\n### Example {j}",
                  f"Entry: condition: {it.get('condition') or '-'} | procedure: {it.get('procedure') or '-'} | settings: {_settings(it.get('settings') or {})} | ops: {_ops(it.get('ops'))}",
                  f"Result: routed {_f(it.get('routed_share'), '.2f')}, D {_f(it.get('D'), '+.3f')}, lfsr {_f(it.get('lfsr'), '.2f')}, "
                  f"accepted {'-' if it.get('accepted') is None else ('yes' if it['accepted'] else 'no')}",
                  "Shown question:", str(it.get("row_prompt") or "").strip(),
                  f"Executor with the entry{mark('with_ok')}, end of its answer: {P._tail(it.get('with') or '', TRACE_TAIL) or '(empty)'}",
                  f"Executor without the entry{mark('without_ok')}, end of its answer: {P._tail(it.get('without') or '', TRACE_TAIL) or '(empty)'}"]
        text = "\n".join(L)
    else:
        raise ValueError(f"unknown memory kind {kind!r}")
    P.guard("", text, rows)
    return _done(info, st, text)


def _done(info, st, text):
    if info is not None:
        st["chars"] = 0 if text is None else len(text)
        info["memory"] = st
    return text


def card_lines(card, rows):
    """One line per shown question of an entry's card: the first 80 characters of its question text, whether the entry's
    condition held on it (router yes / no; "always" for an always entry), and whether the tip got it right. SHOW rows only:
    any other row raises LeakError."""
    out = []
    for c in card or []:
        r = rows.get(c.get("id"))
        if r is None:
            raise ValueError(f"card row {c.get('id')!r} is not a learn row")
        if r.get("v4_role") != "show":
            raise P.LeakError(f"a {r.get('v4_role')} row on an entry's card in a memory text (SHOW rows only)")
        q = P._ws(r["prompt"])
        q = q[:CARD_Q_CHARS] + ("..." if len(q) > CARD_Q_CHARS else "")
        held = "always" if c.get("routed") is None else ("yes" if c["routed"] else "no")
        tip = "-" if c.get("tip_right") is None else ("right" if c["tip_right"] else "wrong")
        out.append(f'  - "{q}" | condition held: {held} | tip: {tip}')
    return out
