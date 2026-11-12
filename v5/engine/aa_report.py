"""Zero A/A reports (czero | wn32) re-expressed against the W reference.
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



import sys
mode = sys.argv[1]
if mode == "czero":
    C = load(f"{V4}/q/out/czero_t004_2.jsonl", lambda d: not d.get("cancelled"))
    T = load(f"{V4}/q/out/g005_parent*_W_*.jsonl", lambda d: not d.get("cancelled") and str(d.get("server", "")).startswith("127.0.0.1:820"))
    et = collections.Counter(x.get("engine_type") for v in T.values() for x in v)
    print("T-side engine_type counts", dict(et))
    compare(C, T, "cell server (t004 edit, own export) vs T servers (t004)")
elif mode == "wvt":
    C = load(f"{V4}/q/out/g004_rung1_W_*.jsonl", lambda d: d.get("edit_sha") == "dae1772730d84bfd")
    T = load(f"{V4}/q/out/g005_parent*_W_*.jsonl", lambda d: not d.get("cancelled") and str(d.get("server", "")).startswith("127.0.0.1:820"))
    print("T rows", len(T), "reads", sum(map(len, T.values())))
    compare(C, T, "t004 edit on W (g004 c30) vs t004 on T servers (g005 parent)")
elif mode == "wrep":
    A = load(f"{V4}/q/out/wrep_c30_1.jsonl", lambda d: not d.get("cancelled"))
    ref = [json.loads(l) for l in open(f"{V4}/pcAA/pcAA_t002_refkeys.jsonl")]
    want = {k: r["row"] for r in ref for k in r["cached"]}
    P = collections.defaultdict(list)
    for f in glob.glob(f"{V4}/q/out/*.jsonl"):
        if "/pcAA" in f or "/wn" in f or "/wrep" in f:
            continue
        for l in open(f):
            if '"x|' not in l:
                continue
            d = json.loads(l)
            if d.get("key") in want and "error" not in d:
                P[want[d["key"]]].append(d)
    T = load(f"{V4}/q/out/g005_parent*_W_*.jsonl", lambda d: not d.get("cancelled") and str(d.get("server", "")).startswith("127.0.0.1:820"))
    G4 = load(f"{V4}/q/out/g004_rung1_W_*.jsonl", lambda d: d.get("edit_sha") == "dae1772730d84bfd")
    compare(A, P, "c30 on W, 2nd read (sample 920) vs t002 parent (servers)")
    compare(A, T, "c30 on W, 2nd read vs t004 on T servers")
    compare(G4, P, "c30 on W, 1st read (g004) vs t002 parent (servers), for reference")
    compare(A, G4, "c30 on W, 2nd read vs 1st read")
elif mode == "pcsalt":
    import urllib.request
    S_ = load(f"{V4}/q/out/pcsalt_t002_1.jsonl", lambda d: d["key"].startswith("pcS|"))
    P_ = load(f"{V4}/q/out/pcsalt_t002_1.jsonl", lambda d: d["key"].startswith("pcP|"))
    ref = [json.loads(l) for l in open(f"{V4}/pcAA/pcAA_t002_refkeys.jsonl")]
    want = {k: r["row"] for r in ref for k in r["cached"]}
    C0 = collections.defaultdict(list)
    for f in glob.glob(f"{V4}/q/out/*.jsonl"):
        if "/pc" in f or "/wn" in f or "/wrep" in f or "/carry" in f:
            continue
        for l in open(f):
            if '"x|' not in l:
                continue
            d = json.loads(l)
            if d.get("key") in want and "error" not in d:
                C0[want[d["key"]]].append(d)
    compare(S_, P_, "salted pairs (shared prefix) vs plain (unique salt), same seeds, caching-on servers")
    compare(P_, C0, "plain on caching-on servers vs t002 parent cache (caching off)")
    compare(S_, C0, "salted pairs vs t002 parent cache (caching off)")
    for port in (8206, 8207):
        try:
            m = urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=10).read().decode()
            print(port, [l for l in m.splitlines() if "prefix_cache" in l and not l.startswith("#")][:6])
        except Exception as e:
            print(port, "metrics", e)
elif mode == "wn32":
    A = load(f"{V4}/q/out/wn32_t002_1.jsonl")
    N = load(f"{V4}/q/out/wnull_t002_1.jsonl")
    ref = [json.loads(l) for l in open(f"{V4}/pcAA/pcAA_t002_refkeys.jsonl")]
    want = {k: r["row"] for r in ref for k in r["cached"]}
    P = collections.defaultdict(list)
    for f in glob.glob(f"{V4}/q/out/*.jsonl"):
        if "/pcAA" in f or "/wn" in f:
            continue
        for l in open(f):
            if '"x|' not in l:
                continue
            d = json.loads(l)
            if d.get("key") in want and "error" not in d:
                P[want[d["key"]]].append(d)
    compare(A, P, "W at 32 running vs parent (servers)")
    compare(A, N, "W at 32 running vs W at 96 (same seeds)")
