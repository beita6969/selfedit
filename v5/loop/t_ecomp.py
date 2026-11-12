"""Tests of ecomp.py on real learn rows and engine-format read records. Run: python3 t_ecomp.py -> ALL GREEN."""
import hashlib
import importlib.util
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ecomp as E  # noqa: E402

ROWS = os.path.join(HERE, "..", "data", "learn.jsonl")
ENGINE = os.path.join(HERE, "..", "engine_v4")


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


def rec(row, s, ans=None, finish="stop", content=None, **kw):
    """An engine read result line (engine_api.md §4) for row, sample s; content ends with \\boxed{ans} unless given."""
    if content is None:
        content = "" if ans is None else f"The marks line up with the second figure, so the result follows.\n\nAnswer: \\boxed{{{ans}}}"
    r = dict(key=f"p000|parent|{row['id']}|{s}", mode="read", row=row["id"], sample=s, content=content,
             reasoning="I compare the parts of the figure one by one.", finish=finish, tokens=1200 + 10 * s, prompt_tokens=900,
             image_tokens=[256] * len(row.get("image_paths") or []), loop=False, grade_error=None, server="127.0.0.1:18400",
             t_submit=1.0, t_done=2.0, seed=12345 + s, settings=None, prefix=None, id=row["id"], kind=row["kind"],
             answer=row["answer"], options=row.get("options") or [], category=row.get("category"), precision=row.get("precision"),
             bench=row["bench"])
    r.update(kw)
    return r


def engine_common():
    spec = importlib.util.spec_from_file_location("engine_common_t", os.path.join(ENGINE, "common.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    rows = {}
    for ln in open(ROWS):
        if ln.strip():
            r = json.loads(ln)
            rows[r["id"]] = r
    sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
    ref = os.path.join(HERE, "..", "..", "code-1109c", "mm", "grade_mm.py")
    check(f"grader: {os.path.relpath(E.G.SOURCE, HERE)}, unchanged copy of code-1109c/mm/grade_mm.py",
          E.G.SOURCE.endswith(os.path.join("engine_v4", "grade_mm.py")) and (not os.path.exists(ref) or sha(E.G.SOURCE) == sha(ref)))

    # ---------------------------------------------------------------- pred = the engine's definition
    C = engine_common()
    rng = random.Random(3)
    kinds = {}
    n = 0
    for rid in sorted(rows):
        row = rows[rid]
        kinds.setdefault(row["kind"], 0)
        gold = str(row["answer"])
        variants = [gold, gold.lower(), "B" if gold != "B" else "C", "7", "8.0", "eight", "3, 6, 2", "yes", "No.", "(A)", "", "Sep"]
        for s, a in enumerate(rng.sample(variants, 4)):
            for fin in ("stop", "length"):
                x = rec(row, s, a, finish=fin)
                ok_e, _, _, pred_e = C.grade(row, x["content"], x["reasoning"], fin, x["tokens"], s)
                assert E.pred(x, row) == pred_e and E.grade(x, row) == (ok_e and pred_e is not None, pred_e), (rid, a, fin)
                n += 1
        kinds[row["kind"]] += 1
    check(f"pred / ok == engine_v4/common.py grade() on {n} records over all {len(rows)} rows ({kinds})", True)
    mc = next(r for r in rows.values() if r["kind"] == "mc" and len(r.get("options") or []) >= 3 and r["answer"] == "A")
    num = next(r for r in rows.values() if r["kind"] == "num" and str(r["answer"]) == "8")
    txt = next(r for r in rows.values() if r.get("category") == "text-in-chart" and r["answer"] == "Sep")
    check("pred: mc letter, num float, text answer", E.pred(rec(mc, 0, "A"), mc) == "A" and E.pred(rec(num, 0, "8"), num) == 8.0
          and E.pred(rec(txt, 0, "Sep"), txt) == "Sep")
    check("pred: None when finish != stop or the content is empty",
          E.pred(rec(mc, 0, "A", finish="length"), mc) is None and E.pred(rec(mc, 0, None), mc) is None
          and E.grade(rec(mc, 0, "A", finish="length"), mc) == (False, None))
    check("pred: rec['pred'] is preferred when present (also null)",
          E.pred(rec(mc, 0, "A", pred="C"), mc) == "C" and E.pred(rec(mc, 0, "A", pred=None), mc) is None
          and E.grade(rec(mc, 0, "A", pred=None, ok=True), mc) == (False, None) and E.grade(rec(mc, 0, "B", pred="A", ok=True), mc) == (True, "A"))

    # ---------------------------------------------------------------- plurality_ok
    S = lambda row, answers, **kw: [rec(row, s, a, **kw) for s, a in enumerate(answers)]
    check("plurality mc: [B, A, A] m=3 -> A (right); m=1 -> B (wrong); m=2 tie B/A -> earliest B (wrong)",
          E.plurality_ok(S(mc, ["B", "A", "A"]), mc, 3) is True and E.plurality_ok(S(mc, ["B", "A", "A"]), mc, 1) is False
          and E.plurality_ok(S(mc, ["B", "A", "A"]), mc, 2) is False and E.plurality_ok(S(mc, ["A", "B"]), mc, 2) is True)
    nn = [rec(mc, 0, "A", finish="length"), rec(mc, 1, None), rec(mc, 2, "A")]
    check("plurality: None preds are one class 'none', which is wrong (2 x none beat 1 x right)",
          E.plurality_ok(nn, mc, 3) is False and E.plurality(nn, mc, 3)[0] == E.NONE_CLASS)
    check("plurality: tie none / right / wrong (1 each) -> the earliest sample's class",
          E.plurality_ok([rec(mc, 0, None), rec(mc, 1, "A"), rec(mc, 2, "B")], mc, 3) is False
          and E.plurality_ok([rec(mc, 0, "A"), rec(mc, 1, None), rec(mc, 2, "B")], mc, 3) is True)
    sh = S(mc, ["A", "B", "B", "A"])
    random.Random(5).shuffle(sh)
    check("plurality: samples are taken in sample order, not list order (first m of samples 0..)",
          E.plurality_ok(sh, mc, 3) is False and E.plurality_ok(sh, mc, 1) is True and E.plurality(sh, mc, 3)[2] == 1)
    check("plurality: fewer than m samples -> None", E.plurality_ok(S(mc, ["A", "A"]), mc, 3) is None and E.plurality_ok([], mc, 2) is None)
    noise = [dict(key=f"p000|parent|{mc['id']}|0", mode="read", row=mc["id"], sample=0, error="timeout"),
             dict(key=f"x|ab|{mc['id']}|1", mode="read", row=mc["id"], sample=1, op_skipped="image_limit: 50 images > 40"),
             rec(mc, 2, "B"), rec(mc, 3, "A"), rec(mc, 3, "B"), rec(mc, 4, "A")]
    check("plurality: error and op_skipped lines are unread (not samples); a repeated sample index keeps its first record",
          [s["sample"] for s in E.read_samples(noise)] == [2, 3, 4] and E.plurality_ok(noise, mc, 3) is True
          and E.plurality_ok(noise, mc, 4) is None)
    numv = S(num, ["7", "8", "8.0", "eight", "7"])
    check("plurality num: 8, 8.0 and 'eight' are one class (pred 8.0); [7, 8, 8.0, eight, 7] m=5 -> 8 (right), m=2 tie -> 7 (wrong)",
          E.plurality_ok(numv, num, 5) is True and E.plurality_ok(numv, num, 2) is False and E.plurality(numv, num, 5)[0] == ("num", 8.0))
    prec = dict(num, answer="3.14", precision=2)
    check("plurality num with precision: 3.141 and 3.139 are one class (rounded to 2 places), right",
          E.plurality_ok(S(prec, ["3.141", "2.5", "3.139"]), prec, 3) is True)
    check("plurality text: 'Sep' and 'sep' are one class (term_norm), beating 'Oct' once",
          E.plurality_ok(S(txt, ["Oct", "sep", "Sep"]), txt, 3) is True and E.plurality_ok(S(txt, ["Oct", "sep", "Sep"]), txt, 1) is False)
    yn = next(r for r in rows.values() if r["kind"] == "yesno" and r["answer"] == "yes")
    check("plurality yesno: [no, yes, Yes.] -> yes (right)", E.plurality_ok(S(yn, ["no", "yes", "Yes."]), yn, 3) is True)
    withp = [rec(mc, 0, "B", pred="A", ok=True), rec(mc, 1, "B", pred="A", ok=True), rec(mc, 2, "B")]
    check("plurality: the records' own pred / ok are used when present", E.plurality_ok(withp, mc, 3) is True)

    # ---------------------------------------------------------------- cost ratio and m
    cm = [(c, E.ref_m(c)) for c in (0.4, 1.0, 1.49, 1.5, 1.9, 2.0, 2.0 + 1e-12, 2.01, 2.5, 3.0, 4.2)]
    check(f"ref_m: ceil(c) when c >= 1.5, else None (plain mean): {cm}",
          [m for _, m in cm] == [None, None, None, 2, 2, 2, 2, 3, 3, 3, 5])
    check("cost_ratio: cand / parent mean; float noise does not lift ceil (0.3 / 0.1 -> m 3, 3000 / 1500 -> m 2)",
          E.cost_ratio(3000, 1500) == 2.0 and E.ref_m(E.cost_ratio(3000, 1500)) == 2 and E.ref_m(E.cost_ratio(0.3, 0.1)) == 3
          and abs(E.cost_ratio(1150 + 48, 1000) - 1.198) < 1e-12 and E.ref_m(E.cost_ratio(1800, 1000)) == 2)
    check("cost_ratio: a parent mean <= 0 or None, or negative cand tokens -> ValueError",
          raises(lambda: E.cost_ratio(10, 0), ValueError) and raises(lambda: E.cost_ratio(10, None), ValueError)
          and raises(lambda: E.cost_ratio(-1, 10), ValueError))
    four = S(mc, ["B", "A", "A", "B"])
    check("reference: c < 1.5 -> the tip's mean ok over its samples; c >= 1.5 -> plurality of ceil(c) samples (1.0 / 0.0)",
          E.reference(four, mc, 1200, 1000) == (0.5, None, 1.2) and E.reference(four, mc, 2600, 1000) == (1.0, 3, 2.6)
          and E.reference(four, mc, 1500, 1000) == (0.0, 2, 1.5) and E.reference(four, mc, 4500, 1000) == (None, 5, 4.5)
          and E.reference([], mc, 1000, 1000) == (None, None, 1.0))
    print("ALL GREEN")


if __name__ == "__main__":
    main()
