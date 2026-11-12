"""W-null A/A and weight cells re-expressed against the W reference.
Sides: P = t002's cached parent samples (servers, pcAA_t002_refkeys.jsonl), N = W-null reads (q/out/wnull_t002_1.jsonl, stock weights on a W worker),
cells = the g004 rung-1 W reads per edit_sha (q/out/g004_rung1_W_*.jsonl). Per row: mean ok / tokens per side; d = side - reference per row;
se over rows (rows = clusters); per source and macro (mean of source means, se = sqrt(sum se_s^2)/S).
usage: python wnull_report.py"""
import glob, json, math, statistics as st, collections

V4 = "/data/fd/mingdazhang/selfedit/v4"


def load(path, pred=lambda d: True):
    out = collections.defaultdict(list)
    for f in glob.glob(path):
        for l in open(f):
            d = json.loads(l)
            if "error" in d or not pred(d):
                continue
            out[d["row"]].append(d)
    return out


ref = [json.loads(l) for l in open(f"{V4}/pcAA/pcAA_t002_refkeys.jsonl")]
want = {k: r["row"] for r in ref for k in r["cached"]}
P = collections.defaultdict(list)
for f in glob.glob(f"{V4}/q/out/*.jsonl"):
    if "/pcAA" in f or "/wnull" in f:
        continue
    for l in open(f):
        if '"x|' not in l:
            continue
        d = json.loads(l)
        if d.get("key") in want and "error" not in d:
            P[want[d["key"]]].append(d)
N = load(f"{V4}/q/out/wnull_t002_1.jsonl")
cells = collections.defaultdict(lambda: collections.defaultdict(list))
for f in glob.glob(f"{V4}/q/out/g004_rung1_W_*.jsonl"):
    for l in open(f):
        d = json.loads(l)
        if "error" not in d:
            cells[d["edit_sha"]][d["row"]].append(d)
names = {}
for f in glob.glob(f"{V4}/q/in/g004_rung1_W_*.jsonl"):
    for l in open(f):
        d = json.loads(l)
        e = d.get("edit")
        import hashlib
        h = hashlib.sha256(json.dumps(e, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]
        names[h] = (e.get("file", "").split("/")[-1][:-5] if isinstance(e, dict) and e.get("op") == "surgeon" else json.dumps(e)[:60])

acc = lambda xs: st.mean(bool(x["ok"]) for x in xs)
tok = lambda xs: st.mean(x["tokens"] for x in xs)
unf = lambda xs: st.mean(x.get("finish") == "length" for x in xs)


def compare(A, B, label):
    rows = [r for r in A if r in B]
    by = collections.defaultdict(list)
    for r in rows:
        by[r.split("/")[0]].append((acc(A[r]) - acc(B[r]), tok(A[r]), tok(B[r]), A[r], B[r]))
    print(f"\n== {label}: rows {len(rows)}; reads {sum(len(A[r]) for r in rows)} vs {sum(len(B[r]) for r in rows)}")
    print(f"{'source':12s} {'n':>4s} {'acc_A':>6s} {'acc_B':>6s} {'d_pp':>7s} {'se':>5s} {'z':>6s} {'tok_A':>6s} {'tok_B':>6s} {'tok%':>6s} {'unf_A':>6s} {'unf_B':>6s}")
    ms, ses = [], []
    for s in sorted(by):
        v = by[s]
        ds = [x[0] for x in v]
        m, se = st.mean(ds), st.stdev(ds) / math.sqrt(len(ds))
        ms.append(m); ses.append(se)
        ta, tb = st.mean(x[1] for x in v), st.mean(x[2] for x in v)
        print(f"{s:12s} {len(v):4d} {100 * st.mean(acc(x[3]) for x in v):6.1f} {100 * st.mean(acc(x[4]) for x in v):6.1f} {100 * m:+7.2f} {100 * se:5.2f} {m / se if se else 0:+6.2f} {ta:6.0f} {tb:6.0f} {100 * (ta / tb - 1):+6.1f} "
              f"{100 * st.mean(unf(x[3]) for x in v):5.1f}% {100 * st.mean(unf(x[4]) for x in v):5.1f}%")
    mm, mse = st.mean(ms), math.sqrt(sum(x * x for x in ses)) / len(ses)
    ta = st.mean(tok(A[r]) for r in rows); tb = st.mean(tok(B[r]) for r in rows)
    print(f"{'MACRO':12s} {len(rows):4d} {'':6s} {'':6s} {100 * mm:+7.2f} {100 * mse:5.2f} {mm / mse:+6.2f} {ta:6.0f} {tb:6.0f} {100 * (ta / tb - 1):+6.1f}")
    return mm, mse


compare(N, P, "W-null (stock weights on W) vs parent (servers)")
for h in cells:
    compare(cells[h], P, f"cell {names.get(h, h)} [{h}] on W vs parent (servers)")
    compare(cells[h], N, f"cell {names.get(h, h)} [{h}] on W vs W-null")
