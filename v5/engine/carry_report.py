"""Carry-in reads report: per candidate, d = mean ok(candidate samples) - mean ok(t002 server parent samples 0-7) per routed JUDGE row;
pooled (stored step-1 sample 0 + new 1-3) and new-only. Cluster-robust se (clusters = learn row "pic", CR1), se floored at 1/rows,
one-sided p with t(nu = clusters - 1). usage: python carry_report.py [cid ...]"""
import json, glob, sys, math, collections, statistics as st
from scipy import stats as SS
V4 = "/data/fd/mingdazhang/selfedit/v4"
rows = {json.loads(l)["id"]: json.loads(l) for l in open(f"{V4}/rows/learn.jsonl")}
cands = sys.argv[1:] or ["g003c31", "g004c12", "g003c00", "g003c16", "g004c11"]
need = {}
for c in cands:
    try:
        for l in open(f"{V4}/q/in/carry_{c}.jsonl"):
            q = json.loads(l)
            k0 = q["key"][:-2] + "|0"
            need[k0] = (c, q["row"], 0); need[q["key"]] = (c, q["row"], q["sample"])
    except FileNotFoundError:
        pass
par = {}
for l in open(glob.glob(f"{V4}/q/in/g004_parentbg_*.jsonl")[0]):
    q = json.loads(l)
    if q["sample"] == 4:
        sg = q["key"].split("|")[1]
        for s in range(8):
            need[f"x|{sg}|{q['row']}|{s}"] = ("P", q["row"], s)
got = collections.defaultdict(dict); skipped = collections.Counter(); srv = collections.Counter()
for f in glob.glob(f"{V4}/q/out/*.jsonl"):
    for l in open(f):
        if '"x|' not in l:
            continue
        d = json.loads(l)
        w = need.get(d.get("key"))
        if not w or "error" in d or d.get("cancelled"):
            continue
        if w[0] != "P" and d.get("op_skipped"):
            skipped[w[0]] += 1
        if w[0] == "P" and not d.get("server"):
            continue
        srv[(w[0], "server" if d.get("server", "").startswith("127.") else str(d.get("server")))] += 1
        got[(w[0], w[1])][w[2]] = d


def judge(c, which):
    recs = []
    for (cc, r), xs in got.items():
        if cc != c:
            continue
        use = [xs[s] for s in xs if (s > 0 if which == "new" else True)]
        P = list(got.get(("P", r), {}).values())
        if not use or len(P) < 8:
            continue
        recs.append(dict(r=r, src=rows[r]["bench"], cl=rows[r].get("pic") or r, n=len(use),
                         d=st.mean(bool(x["ok"]) for x in use) - st.mean(bool(x["ok"]) for x in P),
                         tc=st.mean(x["tokens"] for x in use), tp=st.mean(x["tokens"] for x in P)))
    if not recs:
        print(f"  {which}: no rows yet"); return
    n = len(recs); dbar = st.mean(x["d"] for x in recs)
    cl = collections.defaultdict(float)
    for x in recs:
        cl[x["cl"]] += x["d"] - dbar
    C = len(cl)
    se = math.sqrt(sum(v * v for v in cl.values()) * C / max(C - 1, 1)) / n if C > 1 else float("nan")
    se_u = max(se, 1.0 / n)
    p = 1 - SS.t.cdf(dbar / se_u, C - 1) if C > 1 else float("nan")
    tr = st.mean(x["tc"] for x in recs) / st.mean(x["tp"] for x in recs)
    print(f"  {which:6s}: rows {n} (samples/row {st.mean(x['n'] for x in recs):.2f}), clusters {C} | d {100 * dbar:+.2f} pp, se {100 * se:.2f} (used {100 * se_u:.2f}), "
          f"one-sided p {p:.4f}, nu {C - 1} | tokens x{tr:.3f}")
    by = collections.defaultdict(list)
    for x in recs:
        by[x["src"]].append(x)
    print("          per source: " + "; ".join(f"{s} n{len(v)} d{100 * st.mean(x['d'] for x in v):+.1f} sum{sum(x['d'] for x in v):+.2f}" for s, v in sorted(by.items())))
    if len(by) > 1:
        tot = sum(x["d"] for x in recs)
        print("          LOBO (leave source out: mean d pp): " + "; ".join(f"-{s} {100 * (tot - sum(x['d'] for x in v)) / (n - len(v)):+.2f}" for s, v in sorted(by.items())))


for c in cands:
    have = collections.Counter(len(xs) for (cc, r), xs in got.items() if cc == c)
    print(f"\n== {c}: rows by samples held {dict(sorted(have.items()))}; op_skipped reads {skipped[c]}; read by {dict((k[1], v) for k, v in srv.items() if k[0] == c)}")
    judge(c, "pooled"); judge(c, "new")
