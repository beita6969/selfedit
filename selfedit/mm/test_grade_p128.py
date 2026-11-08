"""Tests for the paper128 grade_mm patch. Usage: python test_grade_p128.py PATH/grade_mm.py [PATH/orig_grade_mm.py]"""
import importlib.util, sys
def load(p, n):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
g = load(sys.argv[1], "g_new"); o = load(sys.argv[2], "g_old") if len(sys.argv) > 2 else None
def row(kind, gold, ans, cat=None, opts=None):
    return dict(kind=kind, answer=gold, options=opts or [], category=cat, finish="stop", content=f"reasoning ... \\boxed{{{ans}}}")
T1, T2, N3, N4 = "text-in-chart", "text-in-general", "number-in-chart", "number-in-general"
CASES = [  # (why, row, expected)
    # numbers: lists, scalars, notation
    ("scalar gold, list answer 7,9,11 (the audit's 720 s1)", row("num", "12", "7,9,11,13,19,22"), False),
    ("scalar gold 7911 must not match list 7,9,11", row("num", "7911", "7,9,11"), False),
    ("list gold 7,9,11 = same list", row("num", "7,9,11", "7, 9, 11"), True),
    ("list gold 30, 40 = 30,40", row("num", "30, 40", "30,40", N3), True),
    ("list gold 30, 40 in wrong order", row("num", "30, 40", "40, 30", N3), False),
    ("list gold 2006, 2008, only one given", row("num", "2006, 2008", "2006", N3), False),
    ("scalar gold, two different numbers", row("num", "5", "5 or 6"), False),
    ("scalar gold, range", row("num", "3", "2-3"), False),
    ("thousands separator", row("num", "1500", "1,500"), True),
    ("1500 vs 1.5e3 (CharXiv example)", row("num", "1500", "1.5e3", N4), True),
    ("1e+06 gold vs 10^6", row("num", "1e+06", "10^6", N3), True),
    ("1e+06 gold vs 1,000,000", row("num", "1e+06", "1,000,000", N3), True),
    ("10^-1 gold vs 0.1", row("num", "10^-1", "0.1", N3), True),
    ("10^-3 gold vs 10^{-3}", row("num", "10^-3", "10^{-3}", N3), True),
    ("10^-3 gold vs 1 \\times 10^{-3}", row("num", "10^-3", "1 \\times 10^{-3}", N3), True),
    ("10^-3 gold vs 10^-2", row("num", "10^-3", "10^{-2}", N3), False),
    ("0.01 vs 10^-2 (CharXiv example)", row("num", "0.01", "10^{-2}", N3), True),
    ("0.01 vs 0.012 (CharXiv example)", row("num", "0.01", "0.012", N3), False),
    ("percent gold 85.5% vs 85.5", row("num", "85.5%", "85.5\\%", N3), True),
    ("percent gold -15% vs -15", row("num", "-15%", "-15", N3), True),
    ("unit power does not count as a number", row("num", "3", "3 m^2"), True),
    ("fraction", row("num", "0.75", "\\frac{3}{4}"), True),
    ("plain number with unit word", row("num", "5", "5 legs"), True),
    ("word number", row("num", "5", "five"), True),
    ("x = 5", row("num", "5", "x = 5"), True),
    ("decimal places gold 0.70 vs 0.7 (value equal)", row("num", "0.70", "0.7", N4), True),
    # CharXiv type 1 / type 2 terms
    ("LaTeX vs plain (official example R^2)", row("short", "$R^2_{t,h,v,m}$", "R^2_t,h,v,m", T1), True),
    ("Greek letter vs name", row("short", "α", "alpha", T1), True),
    ("\\gamma vs γ", row("short", "\\gamma = 0.5", "γ=0.5", T1), True),
    ("(b) vs b (official example)", row("short", "(b)", "b", T1), True),
    ("P56962 vs P55762 (official example)", row("short", "P56962", "P55762", T1), False),
    ("ACC+ vs ACC", row("short", "ACC+", "ACC", T1), False),
    ("P-101 vs P=101", row("short", "P-101", "P=101", T1), False),
    ("case and spaces", row("short", "Dynamic PPMM", "dynamic  ppmm", T1), True),
    ("multi-term, other order, and/comma", row("short", "English-speaking, Protestant, Confucian, and Baltic", "Baltic, Confucian, Protestant and English-speaking", T2), True),
    ("multi-term, one missing", row("short", "MTTK, SCR", "SCR", T1), False),
    ("leading 'the' and \\text{}", row("short", "Kansas City", "\\text{the Kansas City}", T1), True),
    ("T2 yes/no", row("short", "Yes", "yes", T2), True),
    ("T2 wrong option", row("short", "downward", "upward", T2), False),
    # unchanged paths
    ("BabyVision coordinate", row("short", "(4,7)", "(4, 7)", "Fine-grained Discrimination/Find the different"), True),
    ("BabyVision list blank keeps order rule", row("short", "2,5", "5,2", "Fine-grained Discrimination/Find the same"), False),
    ("MC letter", row("mc", "B", "B", None, ["x", "y"]), True),
    ("MC option text", row("mc", "B", "y", None, ["x", "y"]), True),
    ("yes/no", row("yesno", "no", "No"), True),
]
bad = 0
for why, r, want in CASES:
    got = bool(g.correct(r)[0]); old = None
    if o:
        try: old = bool(o.correct(r)[0])
        except Exception: old = "error"
    flag = "ok  " if got == want else "FAIL"
    bad += got != want
    print(f"{flag} want {str(want):5s} got {str(got):5s}" + (f" old {str(old):5s}" if o else "") + f" | {why}")
print(f"{len(CASES) - bad}/{len(CASES)} pass"); sys.exit(1 if bad else 0)
