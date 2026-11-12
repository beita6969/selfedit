"""Read-only: per-row correct-sample count k (8 server samples) for a parent cache; JUDGE and MONITOR rows; per source + overall.
usage: python ktable.py   (t002 parent: per-row key hash from g004_parentbg, samples 0-7; stock: p000|parent(x)|row|0-7)"""
import json, glob, collections, statistics as st, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import grade
REGR = collections.Counter()

V4 = "/data/fd/mingdazhang/selfedit/v4"
rows = {}
for l in open(f"{V4}/rows/learn.jsonl"):
    d = json.loads(l)
    rows[d["id"]] = d
h002 = {}
for l in open(glob.glob(f"{V4}/q/in/g004_parentbg_*.jsonl")[0]):
    d = json.loads(l)
    h002[d["row"]] = d["key"].split("|")[1]
want = {}
for r in rows:
    for s in range(8):
        want[f"p000|{'parent' if s < 4 else 'parentx'}|{r}|{s}"] = ("stock", r, s)
        if r in h002:
            want[f"x|{h002[r]}|{r}|{s}"] = ("t002", r, s)
got = collections.defaultdict(dict)
for f in glob.glob(f"{V4}/q/out/*.jsonl"):
    for l in open(f):
        if '"p000|' not in l and '"x|' not in l:
            continue
        d = json.loads(l)
        w = want.get(d.get("key"))
        if not w or "error" in d or d.get("cancelled") or not d.get("server"):
            continue
        if "pred" not in d:                    # older reads (stock cache) carry no pred: regrade the stored content with the same grader
            ok2, _, _, d["pred"] = grade(rows[w[1]], d.get("content"), d.get("reasoning"), d.get("finish"), d.get("tokens"), w[2])
            REGR[bool(ok2) == bool(d["ok"])] += 1
        got[(w[0], w[1])][w[2]] = d


def norm(p):
    return None if p is None else str(p).strip().lower()


def vote(xs):
    preds = [norm(x.get("pred")) for x in xs]
    c = collections.Counter(p for p in preds if p is not None)
    if not c:
        return 0.0
    top = max(c.values())
    win = [p for p in c if c[p] == top]
    okp = {norm(x.get("pred")): bool(x["ok"]) for x in xs if x.get("pred") is not None}
    return st.mean(okp[p] for p in win)          # tie: expected value over a random tie-break


G = lambda k: "k<=1" if k <= 1 else ("k>=7" if k >= 7 else "k2-6")
K = {}


def table(tip, role):
    by = collections.defaultdict(list)
    miss = 0
    for r, d in rows.items():
        if d["v4_role"] != role:
            continue
        xs = got.get((tip, r), {})
        if len(xs) < 8:
            miss += 1
            continue
        xs = [xs[s] for s in range(8)]
        k = sum(bool(x["ok"]) for x in xs)
        K[(tip, r)] = k
        rec = (k, st.mean(x["tokens"] for x in xs), vote(xs))
        by[d["bench"]].append(rec)
        by["ALL"].append(rec)
    print(f"\n== {tip} parent, {role.upper()} rows (8 server samples; rows missing a sample: {miss})")
    print(f"{'source':11s} {'n':>4s} | " + " ".join(f"{i:>3d}" for i in range(9)) + f" | {'mean':>5s} {'vote':>5s} {'any':>5s} | {'err%k<=1':>8s} {'k2-6':>5s} {'k>=7':>5s} | {'tok k<=1':>8s} {'k2-6':>5s} {'k>=7':>5s}")
    for s in sorted(by, key=lambda x: (x == "ALL", x)):
        v = by[s]
        hist = collections.Counter(x[0] for x in v)
        err = {g: sum(8 - x[0] for x in v if G(x[0]) == g) for g in ("k<=1", "k2-6", "k>=7")}
        E = sum(err.values()) or 1
        tk = {g: [x[1] for x in v if G(x[0]) == g] for g in err}
        print(f"{s:11s} {len(v):4d} | " + " ".join(f"{hist.get(i, 0):3d}" for i in range(9)) +
              f" | {100 * st.mean(x[0] / 8 for x in v):5.1f} {100 * st.mean(x[2] for x in v):5.1f} {100 * st.mean(x[0] > 0 for x in v):5.1f} | "
              f"{100 * err['k<=1'] / E:8.1f} {100 * err['k2-6'] / E:5.1f} {100 * err['k>=7'] / E:5.1f} | " +
              " ".join(f"{(st.mean(tk[g]) if tk[g] else 0):{w}.0f}" for g, w in (("k<=1", 8), ("k2-6", 5), ("k>=7", 5))))


print("regraded reads (ok agrees with stored ok):", dict(REGR))
for tip in ("t002", "stock"):
    for role in ("judge", "monitor"):
        table(tip, role)
for role in ("judge", "monitor"):
    m = collections.Counter()
    for r, d in rows.items():
        if d["v4_role"] == role and ("stock", r) in K and ("t002", r) in K:
            m[(G(K[("stock", r)]), G(K[("t002", r)]))] += 1
    gs = ("k<=1", "k2-6", "k>=7")
    print(f"\n== group moves stock -> t002, {role.upper()} rows (rows: stock group, columns: t002 group)")
    print(f"{'':6s} " + " ".join(f"{g:>6s}" for g in gs))
    for a in gs:
        print(f"{a:6s} " + " ".join(f"{m[(a, b)]:6d}" for b in gs))
