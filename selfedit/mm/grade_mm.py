"""Grade gen_mm.py sample files: pass@1, unbiased pass@k, token stats, truncation / loop / error / no-answer rates,
per-category pass@1 and (MMVP) pair accuracy. Mirrors ../grade.py; a sample counts only if finish == "stop".
Usage: python grade_mm.py OUT.jsonl [...] [--cats] [--md]
"""
import json, math, re, sys, zlib
from collections import defaultdict
from math import comb

WORDNUM = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
                                      "fifteen sixteen seventeen eighteen nineteen twenty".split())}
FILLER = {"row", "rows", "column", "columns", "col", "and", "the", "th", "st", "nd", "rd", "is", "are", "answer", "final"}


def last_boxed(s):
    i = s.rfind("\\boxed")
    if i < 0:
        i = s.rfind("\\fbox")
        if i < 0:
            return None
    j = s.find("{", i)
    if j < 0:
        return None
    depth, k = 0, j
    while k < len(s):
        if s[k] == "{":
            depth += 1
        elif s[k] == "}":
            depth -= 1
            if depth == 0:
                return s[j + 1:k]
        k += 1
    return None


def clean(a):
    a = re.sub(r"\\(?:text|textbf|mathrm|mathbf|operatorname)\{([^{}]*)\}", r"\1", a)
    a = a.replace("$", "").replace("**", "").replace("\\%", "%").replace("\\,", "").replace("\\!", "").replace("\\ ", " ")
    a = a.replace("\u2212", "-").replace("\u2013", "-").replace("\u2192", "-").replace("\\to", "-").replace("\\rightarrow", "-")
    return a.strip().strip(".").strip()


def final_answer(content):
    b = last_boxed(content)
    if b is not None:
        return clean(b)
    m = re.findall(r"(?:final answer|answer)\s*(?:is|:)\s*\**\s*([^\n]+)", content, re.I)
    return clean(m[-1]) if m else None


def norm_txt(s):
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def mc_letter(a, options, content):
    if a is not None:
        for i, o in enumerate(options):
            if norm_txt(a) == norm_txt(str(o)):
                return "ABCDEFGHIJ"[i]
        m = re.match(r"^(?:option\s*)?\(?([A-Ja-j])\)?(?=$|[\s.:)\],])", a.strip(), re.I)
        if m and (m.group(1).isupper() or len(a.strip()) <= 3):
            return m.group(1).upper()
        return None
    m = re.findall(r"answer is:?\s*\**\s*\(?([A-J])\)?\b", content)
    return m[-1] if m else None


def nums(a):
    """Every number written in an answer or a gold, in order. Thousands separators (1,234) are one number; 'a x 10^b',
    '10^-3', '1e+06' are one number; a unit power (m^2, s^{-1}) is not a number; a whole 'a/b' or \frac{a}{b} is a/b.
    A list such as '7,9,11' or '30, 40' gives several numbers."""
    if a is None:
        return []
    s = a.lower().replace("\u2212", "-").replace("%", "").replace("\\%", "").replace("$", "").strip()
    if "=" in s:
        s = s.split("=")[-1]
    s = re.sub(r"\\[dt]?frac\{(-?[\d.]+)\}\{(-?[\d.]+)\}", lambda m: _div(m.group(1), m.group(2)), s)
    m = re.fullmatch(r"\s*(-?\d+(?:\.\d+)?)\s*/\s*(-?\d+(?:\.\d+)?)\s*", s)
    if m:
        s = _div(m.group(1), m.group(2))
    s = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", s)                                   # 1,234,567 -> 1234567; 7,9,11 stays a list
    s = re.sub(r"([a-z])\s*\^\s*\{?-?\d+\}?", r"\1", s)                              # units: m^2, s^{-1}
    s = re.sub(r"(-?\d+(?:\.\d+)?)\s*(?:\\times|\\cdot|x|\*|×)\s*10\s*\^\s*\{?\s*(-?\d+)\s*\}?", lambda m: _pow10(m.group(1), m.group(2)), s)
    s = re.sub(r"(?<![\d.])10\s*\^\s*\{?\s*(-?\d+)\s*\}?", lambda m: _pow10("1", m.group(1)), s)
    if "nan_big" in s:                # an exponent past float range ('10^999'): no number at all, so it matches no gold (11-08)
        return []
    out = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?(?:e[-+]?\d+)?", s)]
    if not all(math.isfinite(x) for x in out):      # '1e309' reads as inf: no number either
        return []
    if out:
        return out
    w = re.findall(r"[a-z]+", s)
    return [float(WORDNUM[w[0]])] if w and w[0] in WORDNUM else []


def _pow10(a, e):
    """a x 10^e as text for nums(); 'nan_big' when the exponent is past float range (OverflowError before 11-08)."""
    try:
        e = int(e)
        if e > 400:                   # below float range is 0.0, as float("1e-999") reads
            return "nan_big"
        v = float(a) * 10.0 ** e
        return repr(v) if math.isfinite(v) else "nan_big"
    except (ValueError, OverflowError):
        return "nan_big"


def _div(p, q):
    try:
        return repr(float(p) / float(q))
    except (ValueError, ZeroDivisionError):
        return "nan_div"


def to_num(a):
    """The one number of an answer; None when it has none or several different ones (a scalar gold answered with a list is wrong)."""
    v = nums(a)
    return v[0] if v and len(set(v)) == 1 else None


def num_ok(pred, gold, precision=None):
    if pred is None:
        return False
    g = float(gold) if not isinstance(gold, str) or re.fullmatch(r"\s*-?\d+(\.\d+)?(e[-+]?\d+)?\s*", gold, re.I) else to_num(gold)
    if g is None or not math.isfinite(g) or not math.isfinite(pred):
        return False
    if precision is not None:
        return round(pred, int(precision)) == round(g, int(precision))
    return abs(pred - g) <= 1e-6 * max(1.0, abs(g))


def canon(s):
    """Structured short answers (BabyVision blanks, RealWorldQA words): set of pairs if 'x-y' pairs, else token tuple."""
    s = clean(s).lower()
    if "=" in s and s.split("=")[-1].strip():      # 'x = 5' reads 5; a bare symbol such as '=' (a comparison blank) stays itself (11-08: gold '=' read as empty)
        s = s.split("=")[-1]
    pairs = re.findall(r"([a-z0-9]+)\s*-\s*([a-z0-9]+)", s)
    rest = re.sub(r"([a-z0-9]+)\s*-\s*([a-z0-9]+)", " ", s)
    if pairs and not re.findall(r"[a-z0-9]", re.sub(r"\b(?:%s)\b" % "|".join(FILLER), " ", rest)):
        return ("pairs", frozenset(pairs))
    toks = [t for t in re.findall(r"[a-z]+|\d+(?:\.\d+)?", s) if t not in FILLER]
    toks = [str(WORDNUM[t]) if t in WORDNUM else t for t in toks]
    return ("toks", tuple(toks)) if toks else ("raw", s.replace(" ", ""))


CHARXIV_TEXT = ("text-in-chart", "text-in-general")
GREEK = {"α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "ε": "epsilon", "ϵ": "epsilon", "ζ": "zeta", "η": "eta", "θ": "theta",
         "ι": "iota", "κ": "kappa", "λ": "lambda", "μ": "mu", "ν": "nu", "ξ": "xi", "π": "pi", "ρ": "rho", "σ": "sigma", "τ": "tau",
         "υ": "upsilon", "φ": "phi", "ϕ": "phi", "χ": "chi", "ψ": "psi", "ω": "omega", "Γ": "gamma", "Δ": "delta", "Θ": "theta",
         "Λ": "lambda", "Ξ": "xi", "Π": "pi", "Σ": "sigma", "Φ": "phi", "Ψ": "psi", "Ω": "omega"}


def term_norm(s):
    """One term, for CharXiv's 'same term' rule: case, LaTeX markup, Greek spelling, spaces, brackets, quotes and a leading
    'the' do not count; letters, digits and the signs + - = < > . % / * do (ACC+ != ACC, P-101 != P=101)."""
    s = clean(s)
    for k, v in GREEK.items():
        s = s.replace(k, v)
    s = s.lower().replace("\u2010", "-").replace("\u2011", "-").replace("\u2012", "-").replace("\u2014", "-")
    s = re.sub(r"\\(?:left|right|mathcal|mathit|mathsf|mathbb|bm|boldsymbol|hat|bar|tilde|vec)\b", "", s)
    s = s.replace("\\", "")
    s = re.sub(r"^\s*the\s+", "", s)
    return re.sub(r"[^a-z0-9+\-=<>.%/*]", "", s).strip(".")


def term_eq(a, g):
    """Same term, or the same set of terms in any order (terms split at commas, semicolons, '&' and 'and')."""
    if term_norm(a) == term_norm(g) and term_norm(g):
        return True
    sp = lambda x: {t for t in (term_norm(y) for y in re.split(r",|;|&|\band\b", clean(x), flags=re.I)) if t}
    ta, tg = sp(a), sp(g)
    return len(tg) > 1 and ta == tg


def correct(r):
    if r["finish"] != "stop" or not r["content"]:
        return False, None
    a = final_answer(r["content"])
    k = r["kind"]
    if k == "mc":
        p = mc_letter(a, r.get("options") or [], r["content"])
        return p == r["answer"], p
    if k == "num":
        g = nums(str(r["answer"]))
        if len(g) > 1:                       # a list gold ('30, 40'): the same numbers in the same order
            v = nums(a)
            return len(v) == len(g) and all(num_ok(x, y) for x, y in zip(v, g)), (",".join(f"{x:g}" for x in v) if v else None)
        p = to_num(a)
        return num_ok(p, r["answer"], r.get("precision")), p
    if k == "yesno":
        p = None
        if a is not None:
            m = re.match(r"^\W*(yes|no)\b", a.lower())
            p = m.group(1) if m else None
        return p == r["answer"], p
    if a is None or not a.strip():         # an empty final answer is no answer (11-08: it matched an emptied gold)
        return False, None
    if r.get("category") in CHARXIV_TEXT:  # CharXiv text answers: the official rule "same term", made deterministic
        return term_eq(a, str(r["answer"])), a
    cg, cp = canon(r["answer"]), canon(a)
    if cg == cp:
        return True, a
    if cg[0] == cp[0] == "pairs":  # matching tasks: "8-2" == "2-8" when the two label sets are disjoint (unambiguous)
        left, right = {x for x, _ in cg[1]}, {y for _, y in cg[1]}
        if not left & right:
            return {frozenset(t) for t in cg[1]} == {frozenset(t) for t in cp[1]}, a
    return False, a


def looped(r):
    tail = ((r.get("reasoning") or "") + (r.get("content") or ""))[-20000:]      # a sample with no text (None) is no loop, not an error (11-08)
    return len(tail) > 5000 and len(zlib.compress(tail.encode())) / len(tail.encode()) < 0.08


def pass_at_k(n, c, k):
    return 1.0 if n - c < k else 1 - comb(n - c, k) / comb(n, k)


def grade(path, show_cats, rows_md):
    by = defaultdict(list)
    for line in open(path):
        r = json.loads(line)
        r["ok"], r["pred"] = correct(r)
        by[r.get("set") or r["id"].split("/")[0]].append(r)
    for bench, rs in sorted(by.items()):
        per = defaultdict(list)
        for r in rs:
            per[r["id"]].append(r)
        n = max(len(v) for v in per.values())
        full = {i: v for i, v in per.items() if len(v) == n}
        ks = [k for k in (1, 2, 4, 8, 16) if k <= n]
        pk = {k: sum(pass_at_k(len(v), sum(x["ok"] for x in v), k) for v in full.values()) / len(full) for k in ks}
        fr = [x for v in full.values() for x in v]
        toks = sorted(x["tokens"] for x in fr)
        res = {"file": path, "bench": bench, "problems": len(full), "partial_items": len(per) - len(full), "samples_per": n,
               **{f"pass@{k}": round(100 * v, 1) for k, v in pk.items()},
               "tok_mean": round(sum(toks) / len(toks)), "tok_p50": toks[len(toks) // 2], "tok_p90": toks[int(len(toks) * 0.9)],
               "prompt_tok_mean": round(sum(x.get("prompt_tokens", 0) for x in fr) / len(fr)),
               "truncated_%": round(100 * sum(x["finish"] == "length" for x in fr) / len(fr), 1),
               "looped_%": round(100 * sum(looped(x) for x in fr) / len(fr), 1),
               "error_%": round(100 * sum(x["finish"] == "error" for x in fr) / len(fr), 1),
               "no_answer_%": round(100 * sum(x["finish"] == "stop" and x["pred"] is None for x in fr) / len(fr), 1)}
        pairs = defaultdict(lambda: defaultdict(list))
        for v in full.values():
            for x in v:
                if x.get("pair") is not None:
                    pairs[x["pair"]][x["sample"]].append(x["ok"])
        if pairs:
            accs = [all(oks) for p in pairs.values() for oks in p.values() if len(oks) == 2]
            res["pair_acc"] = round(100 * sum(accs) / max(1, len(accs)), 1)
        cats = defaultdict(list)
        for v in full.values():
            cats[v[0].get("category")].append(sum(x["ok"] for x in v) / len(v))
        if len(cats) > 1:
            res["cat_pass@1"] = {str(c): [round(100 * sum(a) / len(a), 1), len(a)] for c, a in sorted(cats.items(), key=lambda kv: str(kv[0]))}
            res["cat_mean_pass@1"] = round(sum(100 * sum(a) / len(a) for a in cats.values()) / len(cats), 1)
        if not show_cats:
            res.pop("cat_pass@1", None)
        print(json.dumps(res))
        rows_md.append(res)


if __name__ == "__main__":
    files = [a for a in sys.argv[1:] if not a.startswith("--")]
    rows = []
    for f in files:
        grade(f, "--cats" in sys.argv, rows)
    if "--md" in sys.argv:
        print("\n| set | items | pass@1 | pass@4 | cat-mean | tok mean | tok p90 | trunc % | loop % | no-ans % |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            print(f"| {r['bench']} | {r['problems']} | {r['pass@1']} | {r.get('pass@4', '')} | {r.get('cat_mean_pass@1', '')} | "
                  f"{r['tok_mean']} | {r['tok_p90']} | {r['truncated_%']} | {r['looped_%']} | {r['no_answer_%']} |")
