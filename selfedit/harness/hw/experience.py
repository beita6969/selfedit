"""Build a deterministic, machine-made ledger from the pool's shown traces.

The ledger deliberately has no model or filesystem dependencies.  It accepts
both full answer records and the compact records returned by ``hboost``'s
``load_member`` (the latter use ``tail``, ``said``, ``key`` and ``tok``).
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict

import numpy as np


_SENTENCE_RE = re.compile(r"[.!?\n]+")
_ROW_ID_RE = re.compile(r"[a-z0-9_]+/[A-Za-z0-9_.:-]*\d")
_DIGIT_RE = re.compile(r"\d")
_SPACE_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[a-z#]+")
_CANDIDATE_RE = re.compile(
    r"\b(?:means|corresponds|by definition|standard|typically|usually|"
    r"so they are|always|default|most common)\b",
    re.I,
)
_PRIOR_RE = re.compile(
    r"\b(?:standard|classic|by definition|typically|usually|illusion|normally|canonical)\b",
    re.I,
)
_RECALL_RE = re.compile(
    r"(?:seen this (?:exact )?image|answer key|typical answer|often [A-D] or [A-D]|"
    r"accepted answer|zhidao|from memory)",
    re.I,
)
def _as_text(value) -> str:
    return "" if value is None else str(value)


def _is_right(record: dict) -> bool:
    value = record.get("ok", False)
    return value is True or str(value) == "True"


def _record_content(record: dict) -> str:
    # ``said`` is the compact alias produced by hboost.load_member.
    return _as_text(record.get("content", record.get("said", "")))


def _record_reasoning(record: dict) -> str:
    # ``tail`` is the compact reasoning alias produced by hboost.load_member.
    return _as_text(record.get("reasoning", record.get("tail", "")))


def _record_answer(record: dict):
    # ``key`` is the parsed-answer alias produced by hboost.load_member.
    if "answer" in record:
        return record.get("answer")
    return record.get("key")


def _record_tokens(record: dict):
    value = record.get("tokens", record.get("tok"))
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _trace(record: dict) -> str:
    reasoning, content = _record_reasoning(record), _record_content(record)
    if reasoning and content:
        return reasoning + "\n" + content
    return reasoning or content


def _normal_sentence(sentence: str) -> str:
    sentence = _DIGIT_RE.sub("#", _as_text(sentence).lower())
    return _SPACE_RE.sub(" ", sentence).strip()


def _tokens(sentence: str) -> list[str]:
    return _TOKEN_RE.findall(sentence)


def _sixgrams(sentence: str) -> set[str]:
    words = _tokens(sentence)
    return {" ".join(words[j : j + 6]) for j in range(max(0, len(words) - 5))}


def _category(row: dict):
    value = row.get("category", "")
    return "" if value is None else str(value)


def _bench(data, row_id: str, row: dict) -> str:
    # Data.bench_of is the authoritative grouping rule when available.
    return str(data.bench_of(row_id))


def _key(value) -> str:
    return _as_text(value).strip().casefold()


def _display(value) -> str:
    text = _as_text(value).strip()
    return text if text else "(unparsed)"


def _option_value(value, row: dict) -> str:
    """Map an MC letter to its option text, preserving other answer values."""
    raw = _as_text(value).strip()
    options = row.get("options") or []
    if row.get("kind") == "mc" and options:
        match = re.fullmatch(r"([A-Za-z])(?:[.)]?)", raw)
        if match:
            index = ord(match.group(1).upper()) - ord("A")
            if 0 <= index < len(options):
                return _display(options[index])
        folded = raw.casefold()
        for option in options:
            if _key(option) == folded:
                return _display(option)
    if row.get("kind") == "yesno":
        return raw.casefold() if raw else "(unparsed)"
    return _display(value)


def _safe_line(line: str) -> str:
    """Remove any row-id-shaped substring before it reaches the card text."""
    line = _as_text(line)
    while True:
        # Repeat defensively in case a replacement leaves another match.
        cleaned, count = _ROW_ID_RE.subn("[redacted]", line)
        line = cleaned
        if not count:
            return line


def _safe_example(sentence: str, row: dict) -> str:
    """Make a wrong-rule example safe to show beside aggregate counts."""
    text = _normal_sentence(sentence)
    # Digits become '#'; standalone MC letters are not exposed as row answers.
    text = re.sub(r"\b[A-D]\b", "[choice]", text, flags=re.I)
    values = [row.get("answer")]
    values.extend(row.get("options") or [])
    for value in sorted({_as_text(x).strip() for x in values if _as_text(x).strip()}, key=len, reverse=True):
        text = re.sub(rf"(?<!\w){re.escape(value)}(?!\w)", "[answer]", text, flags=re.I)
    return _safe_line(text[:160])[:160]


def _safe_shingle(shingle: str, rows: list[dict]) -> str:
    """Keep row-specific answer labels out of the displayed rule pattern."""
    text = _normal_sentence(shingle)
    text = re.sub(r"\b[A-D]\b", "[choice]", text, flags=re.I)
    values = []
    for row in rows:
        values.append(row.get("answer"))
        values.extend(row.get("options") or [])
    for value in sorted({_as_text(x).strip() for x in values if _as_text(x).strip()}, key=len, reverse=True):
        text = re.sub(rf"(?<!\w){re.escape(value)}(?!\w)", "[answer]", text, flags=re.I)
    return _safe_line(text)


def _wrong_rules(group_records: list[tuple[str, str, dict, dict]], right_records: list[dict]) -> list[dict]:
    """Return top exact six-token shingles, shared across wrong rows."""
    right_shingles = set()
    for record in right_records:
        for sentence in _SENTENCE_RE.split(_trace(record)):
            right_shingles.update(_sixgrams(_normal_sentence(sentence)))

    occurrences = defaultdict(list)
    row_lookup = {}
    for row_id, _member, row, record in group_records:
        row_lookup.setdefault(row_id, row)
        if _is_right(record):
            continue
        seen_here = set()
        for sentence in _SENTENCE_RE.split(_trace(record)):
            normal = _normal_sentence(sentence)
            if not normal or not _CANDIDATE_RE.search(normal):
                continue
            for shingle in _sixgrams(normal):
                if shingle in seen_here:
                    continue
                seen_here.add(shingle)
                occurrences[shingle].append((row_id, _safe_example(sentence, row)))

    candidates = []
    for shingle, hits in occurrences.items():
        rows = {row_id for row_id, _example in hits}
        if len(hits) < 2 or len(rows) < 2 or shingle in right_shingles:
            continue
        example = sorted(example for _row_id, example in hits if example)[0] if any(example for _row_id, example in hits) else ""
        safe_pattern = _safe_shingle(shingle, [row_lookup[row_id] for row_id in sorted(rows, key=str)])
        candidates.append(
            dict(
                shingle=shingle,
                display_shingle=safe_pattern,
                count=len(hits),
                example=example[:160],
            )
        )
    candidates.sort(key=lambda item: (-item["count"], item["shingle"], item["example"]))
    return candidates[:3]


def _decision_words(rows: list[tuple[str, dict]], members_votes: dict) -> list[dict]:
    if not rows or any(row.get("kind") not in ("mc", "yesno") for _row_id, row in rows):
        return []
    golds = [(_option_value(row.get("answer"), row), row) for _row_id, row in rows]
    unique_golds = {_key(value) for value, _row in golds}
    if len(unique_golds) > 4:
        return []

    gold_count = Counter(_key(value) for value, _row in golds)
    gold_display = {}
    for value, _row in golds:
        gold_display.setdefault(_key(value), _display(value))
    gold_key, gold_rows = sorted(gold_count.items(), key=lambda item: (-item[1], item[0]))[0]
    out = []
    for member in sorted(members_votes):
        said_values = []
        for row_id, row in rows:
            for sample, record in sorted((members_votes.get(member, {}).get(row_id, {}) or {}).items(), key=lambda item: str(item[0])):
                said_values.append(_option_value(_record_answer(record), row))
        n = len(said_values)
        if n < 8:
            continue
        said_count = Counter(_key(value) for value in said_values)
        said_key, said_n = sorted(said_count.items(), key=lambda item: (-item[1], item[0]))[0]
        if said_key == gold_key or said_n / n < 0.75 or gold_rows / len(rows) < 0.60:
            continue
        out.append(
            dict(
                member=member,
                said=said_values[next(j for j, value in enumerate(said_values) if _key(value) == said_key)],
                said_count=int(said_n),
                samples=int(n),
                said_share=float(said_n / n),
                gold=gold_display[gold_key],
                gold_rows=int(gold_rows),
                rows=int(len(rows)),
                gold_share=float(gold_rows / len(rows)),
            )
        )
    return out


def _prior_census(records: list[dict]) -> dict:
    out = {}
    for name, pattern in (("prior", _PRIOR_RE), ("recall", _RECALL_RE)):
        matches = sum(bool(pattern.search(_trace(record))) for record in records)
        wrong = sum(bool(pattern.search(_trace(record))) and not _is_right(record) for record in records)
        out[name] = dict(
            matches=int(matches),
            wrong=int(wrong),
            wrong_share=float(wrong / matches) if matches else 0.0,
        )
    out["samples"] = int(len(records))
    return out


def _median_tokens(records: list[dict]):
    values = [value for value in (_record_tokens(record) for record in records) if value is not None]
    if not values:
        return None
    median = float(np.median(np.asarray(values, dtype=float)))
    return int(median) if median.is_integer() else median


def _right_methods(rows: list[tuple[str, dict]], members_votes: dict) -> list[dict]:
    qualified = []
    for row_id, row in rows:
        records = []
        for member in sorted(members_votes):
            for sample, record in sorted((members_votes.get(member, {}).get(row_id, {}) or {}).items(), key=lambda item: str(item[0])):
                records.append((member, str(sample), record))
        rights = [record for _member, _sample, record in records if _is_right(record)]
        wrongs = [record for _member, _sample, record in records if not _is_right(record)]
        if not rights or len(wrongs) <= len(rights):
            continue
        chosen = sorted(
            ((len(_record_reasoning(record)), member, sample, record) for member, sample, record in records if _is_right(record)),
            key=lambda item: (-item[0], item[1], item[2]),
        )[0][3]
        method = _DIGIT_RE.sub("#", _record_reasoning(chosen))[-400:]
        method = _safe_line(_SPACE_RE.sub(" ", method).strip())
        qualified.append(
            (
                _category(row),
                row_id,
                dict(
                    category=_category(row),
                    method=method,
                    right_tokens_median=_median_tokens(rights),
                    wrong_tokens_median=_median_tokens(wrongs),
                ),
            )
        )
    qualified.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in qualified[:4]]


def _fmt_value(value) -> str:
    return _as_text(value).replace("\n", " ").strip()


def _format_text(bench_fields: dict, max_chars: int) -> str:
    """1109c: every bench gets an equal share of max_chars (so no bench is cut off by the ones sorted before it); categories with nothing to say are left out."""
    head = "## Ledger of the pool's own traces on shown questions (machine-made counts)"
    share = max(200, (int(max_chars) - len(head)) // max(1, len(bench_fields)))
    blocks = [head]
    for bench in sorted(bench_fields):
        section = bench_fields[bench]
        lines = [f"\n### Bench: {_fmt_value(bench)}"]
        census = section["prior_census"]
        lines.append(
            "- prior/recall census: "
            f"prior {census['prior']['matches']} matches, {census['prior']['wrong_share']:.0%} wrong; "
            f"recall {census['recall']['matches']} matches, {census['recall']['wrong_share']:.0%} wrong"
        )
        for category in sorted(section["categories"]):
            values = section["categories"][category]
            body = []
            for item in values["wrong_rules"]:
                body.append(f"- wrong rule ({item['count']} samples): {item.get('display_shingle', item['shingle'])} — example: {item['example']}")
            for item in values["decision_words"]:
                body.append(f"- decision bias ({_fmt_value(item['member'])}): said '{_fmt_value(item['said'])}' in {item['said_count']} of {item['samples']} samples; "
                            f"gold is '{_fmt_value(item['gold'])}' on {item['gold_rows']} of {item['rows']} rows")
            for item in values["right_method"]:
                right_n, wrong_n = item["right_tokens_median"], item["wrong_tokens_median"]
                body.append(f"- right-sample method: {_fmt_value(item['method'])} (median tokens right {right_n if right_n is not None else 'n/a'} vs wrong {wrong_n if wrong_n is not None else 'n/a'})")
            if body:
                lines.append(f"#### Category: {_fmt_value(category)}")
                lines += body
        blocks.append("\n".join(_safe_line(l) for l in lines)[:share])
    return "\n".join(blocks)


def build_ledger(data, members_votes, show_ids, max_chars=4000) -> dict:
    """Build the card ledger from shown rows only.

    Every access to ``data.rows`` is guarded by the row's ``show`` role.  The
    returned ``fields`` retain structured counts; ``text`` is the bounded card
    representation consumed by the proposer.
    """
    row_ids = sorted(set(show_ids), key=str)
    rows = {}
    for row_id in row_ids:
        assert row_id in data.rows, f"unknown show row: {row_id}"
        assert data.role.get(row_id) == "show", f"ledger may read show rows only: {row_id}"
        rows[row_id] = data.rows[row_id]

    by_bench = defaultdict(list)
    for row_id in row_ids:
        row = rows[row_id]
        by_bench[_bench(data, row_id, row)].append((row_id, row))

    fields = {}
    for bench in sorted(by_bench):
        bench_rows = sorted(by_bench[bench], key=lambda item: str(item[0]))
        all_records = []
        by_category = defaultdict(list)
        for row_id, row in bench_rows:
            category = _category(row)
            by_category[category].append((row_id, row))
            for member in sorted(members_votes):
                for sample, record in sorted((members_votes.get(member, {}).get(row_id, {}) or {}).items(), key=lambda item: str(item[0])):
                    all_records.append(record)

        categories = {}
        for category in sorted(by_category):
            category_rows = by_category[category]
            group_records = []
            right_records = []
            for row_id, row in category_rows:
                for member in sorted(members_votes):
                    for sample, record in sorted((members_votes.get(member, {}).get(row_id, {}) or {}).items(), key=lambda item: str(item[0])):
                        group_records.append((row_id, member, row, record))
                        if _is_right(record):
                            right_records.append(record)
            categories[category] = dict(
                wrong_rules=_wrong_rules(group_records, right_records),
                decision_words=_decision_words(category_rows, members_votes),
                right_method=[],
            )

        # Right-method entries are computed per bench, then placed under the
        # corresponding category so the card never needs to print row IDs.
        for category, item in categories.items():
            item["right_method"] = [
                method for method in _right_methods(bench_rows, members_votes) if method["category"] == category
            ]
        census = _prior_census(all_records)
        # Keep the explicit ``categories`` map and expose category names at the
        # bench level as a small convenience for callers inspecting one card.
        bench_field = dict(categories=categories, prior_census=census)
        for category, value in categories.items():
            if category not in bench_field:
                bench_field[category] = value
        fields[bench] = bench_field

    text = _format_text(fields, max_chars)
    return dict(text=text, fields=fields)
