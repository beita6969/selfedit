"""selfedit v4, equal-compute reference (coordinator S2-B, engine_api.md §7 harness ops).

When an entry's generated tokens per question exceed the tip's by a factor c >= 1.5, its gain on a question is measured
against the plurality answer of ceil(c) tip samples on that question instead of the tip's mean. Functions:
  pred(rec, row)                         -> extracted, normalised answer = grade_mm.correct(r)[1] (engine definition), or None
  grade(rec, row)                        -> (ok, pred)
  answer_class(p, row)                   -> the class a pred votes for (None -> "none")
  plurality_ok(samples, row, m)          -> ok of the plurality pred among the first m samples, or None (fewer than m)
  cost_ratio(cand_tokens, parent_tokens_mean) -> c
  ref_m(c)                               -> ceil(c) when c >= 1.5, else None (plain mean reference)
  reference(samples, row, cand_tokens, parent_tokens_mean) -> (value, m, c)
rec: an engine read result (engine_api.md §4); rec["pred"] / rec["ok"] are used when present. row: the learn row (id, kind,
answer, options, category, precision). grade_mm is the unchanged copy in ../engine_v4 (code-1109c/mm/grade_mm.py; override with
the environment variable SELFEDIT_GRADE_MM). Standard library only.
"""
import importlib.util
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
C_MIN = 1.5                       # equal-compute threshold
NONE_CLASS = "none"               # every None pred votes for this one class, which is wrong
_EPS = 1e-9                       # ceil(2.0000000001) is 2, not 3 (float noise in a token ratio)


def _load_grader():
    paths = [os.environ.get("SELFEDIT_GRADE_MM"), os.path.join(HERE, "..", "engine_v4", "grade_mm.py"),
             os.path.join(HERE, "..", "..", "code-1109c", "mm", "grade_mm.py")]
    for p in paths:
        if p and os.path.exists(p):
            spec = importlib.util.spec_from_file_location("grade_mm_ecomp", p)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            mod.SOURCE = os.path.abspath(p)
            return mod
    raise ImportError(f"grade_mm.py not found in {[p for p in paths if p]}")


G = _load_grader()


# ------------------------------------------------------------------------------------------------ one sample
def _r(rec, row):
    """The record grade_mm.correct reads, built as engine_v4/common.py grade() builds it."""
    r = {"id": row["id"], "kind": row["kind"], "answer": row["answer"], "options": row.get("options") or [],
         "category": row.get("category"), "content": rec.get("content") or "", "finish": rec.get("finish")}
    if row.get("precision") is not None:
        r["precision"] = row["precision"]
    return r


def _graded(rec, row):
    """(ok, pred) from the content: pred None when finish != "stop", the content is empty, or the grader raises (as the engine)."""
    if rec.get("finish") != "stop" or not rec.get("content"):
        return False, None
    try:
        ok, p = G.correct(_r(rec, row))
    except Exception:
        return False, None
    if p is not None and not isinstance(p, (str, int, float)):
        p = str(p)
    return bool(ok), p


def pred(rec, row):
    """rec["pred"] when the record has the field (also when it is null), else grade_mm.correct(r)[1] on the record's content."""
    if "pred" in rec:
        return rec["pred"]
    return _graded(rec, row)[1]


def grade(rec, row):
    """(ok, pred): the record's own ok / pred where present, the grader otherwise; ok is False whenever pred is None."""
    ok, p = _graded(rec, row)
    if "pred" in rec:
        p = rec["pred"]
    if rec.get("ok") is not None:
        ok = bool(rec["ok"])
    return (False if p is None else ok), p


def answer_class(p, row):
    """The class a pred votes for. Classes follow the grader's normal forms, so answers it grades as the same answer are one class:
    mc letters and yes / no as they are; a number rounded to the row's precision (float otherwise); a list answer "a,b" as it is;
    a text answer by grade_mm.term_norm (CharXiv text categories) or grade_mm.canon. None -> "none"."""
    if p is None:
        return NONE_CLASS
    k = row.get("kind")
    if k in ("mc", "yesno"):
        return ("v", str(p))
    if k == "num":
        if isinstance(p, (int, float)) and not isinstance(p, bool):
            x = float(p)
            return ("num", round(x, int(row["precision"])) if row.get("precision") is not None else x)
        return ("v", str(p))
    s = str(p)
    if row.get("category") in G.CHARXIV_TEXT:
        return ("term", G.term_norm(s))
    try:
        return ("canon", G.canon(s))
    except Exception:
        return ("v", s)


# ------------------------------------------------------------------------------------------------ several samples
def _sidx(s, j):
    if isinstance(s.get("sample"), int) and not isinstance(s.get("sample"), bool):
        return s["sample"]
    try:
        return int(str(s.get("key", "")).rsplit("|", 1)[-1])
    except ValueError:
        return j


def read_samples(samples):
    """Read results in sample order: error lines and op_skipped lines are unread, not samples; a repeated sample index keeps
    its first record."""
    out, seen = [], set()
    vs = [(j, s) for j, s in enumerate(samples or []) if isinstance(s, dict) and not s.get("error") and not s.get("op_skipped")
          and (s.get("ok") is not None or s.get("finish") is not None)]
    for j, s in sorted(vs, key=lambda x: (_sidx(x[1], x[0]), x[0])):
        i = _sidx(s, j)
        if i not in seen:
            seen.add(i)
            out.append(s)
    return out


def plurality(samples, row, m):
    """(class, ok, index of its first sample) of the plurality answer among the first m samples in sample order, or None when
    fewer than m samples exist. Ties go to the class whose first sample is earliest. The "none" class is wrong."""
    vs = read_samples(samples)
    if m is None or m < 1 or len(vs) < m:
        return None
    count, first, ok = {}, {}, {}
    for j, s in enumerate(vs[:m]):
        o, p = grade(s, row)
        c = answer_class(p, row)
        count[c] = count.get(c, 0) + 1
        if c not in first:
            first[c], ok[c] = j, (False if c == NONE_CLASS else o)
    best = min(count, key=lambda c: (-count[c], first[c]))
    return best, ok[best], first[best]


def plurality_ok(samples, row, m):
    """Ok (True / False) of the plurality pred among the first m samples, or None when fewer than m samples exist."""
    r = plurality(samples, row, m)
    return None if r is None else r[1]


def cost_ratio(cand_tokens, parent_tokens_mean):
    """c = the entry's generated tokens per question / the tip's mean tokens per question."""
    if parent_tokens_mean is None or not parent_tokens_mean > 0:
        raise ValueError(f"parent_tokens_mean must be > 0, not {parent_tokens_mean!r}")
    if cand_tokens is None or cand_tokens < 0:
        raise ValueError(f"cand_tokens must be >= 0, not {cand_tokens!r}")
    return float(cand_tokens) / float(parent_tokens_mean)


def ref_m(c):
    """Tip samples in the plurality reference: ceil(c) when c >= 1.5, else None (the tip's mean is the reference)."""
    return max(2, math.ceil(c - _EPS)) if c >= C_MIN - _EPS else None


def ref_m_round(c):
    """AM16 item 3 (generations created after it): the plurality of round(c) cached parent samples when round(c) >= 2, else None (the
    parent's mean over its cached samples is the reference). round half up: c in [k - .5, k + .5) -> k; no trigger constant."""
    m = int(math.floor(c + 0.5 + _EPS))
    return m if m >= 2 else None


def reference(samples, row, cand_tokens, parent_tokens_mean):
    """(value, m, c): value = plurality_ok of ceil(c) tip samples (as 1.0 / 0.0) when c >= 1.5, else the mean ok of the tip's
    read samples; None when there are too few samples (fewer than m, or none)."""
    c = cost_ratio(cand_tokens, parent_tokens_mean)
    m = ref_m(c)
    if m is not None:
        v = plurality_ok(samples, row, m)
        return (None if v is None else float(v)), m, c
    vs = read_samples(samples)
    if not vs:
        return None, None, c
    return sum(grade(s, row)[0] for s in vs) / len(vs), None, c
