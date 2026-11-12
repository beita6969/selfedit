"""Engine table for a hand bundle (AM23): per row candidate mean ok over its keys vs parent mean ok over parent_keys (table.json),
clusters = picture (CR1), se floored at 1/rows, one-sided p with t(nu = clusters - 1), tokens ratio; per source, by member count,
per member (rows where it holds). usage: python bundle_report.py TABLE_JSON"""
import json, glob, sys, math, collections, statistics as st
from scipy import stats as SS
V4 = "/data/fd/mingdazhang/selfedit/v4"
T = json.load(open(sys.argv[1]))
need = set()
for r in T["rows"]:
    need.update(r["keys"]); need.update(r["parent_keys"])
got = {}
for f in glob.glob(f"{V4}/q/out/*.jsonl"):
    for l in open(f):
        if '"x|' not in l and '"p000|' not in l:
            continue
        d = json.loads(l)
        if d.get("key") in need and "error" not in d and not d.get("cancelled"):
            got[d["key"]] = d
recs, miss = [], 0
for r in T["rows"]:
    c = [got[k] for k in r["keys"] if k in got]; p = [got[k] for k in r["parent_keys"] if k in got]
    if len(c) < len(r["keys"]) or len(p) < len(r["parent_keys"]):
        miss += 1
        if not c or not p:
            continue
    recs.append(dict(r=r["row"], src=r["bench"], cl=r["pic"], m=r["members"], n=len(c), skip=sum(bool(x.get("op_skipped")) for x in c),
                     d=st.mean(bool(x["ok"]) for x in c) - st.mean(bool(x["ok"]) for x in p),
                     ac=st.mean(bool(x["ok"]) for x in c), ap=st.mean(bool(x["ok"]) for x in p),
                     kc=sum(bool(x["ok"]) for x in c), kp=sum(bool(x["ok"]) for x in p), nc=len(c), np=len(p),
                     tc=st.mean(x["tokens"] for x in c), tp=st.mean(x["tokens"] for x in p)))


def line(label, v):
    if not v:
        return
    n = len(v); dbar = st.mean(x["d"] for x in v)
    cl = collections.defaultdict(float)
    for x in v:
        cl[x["cl"]] += x["d"] - dbar
    C = len(cl)
    se = math.sqrt(sum(a * a for a in cl.values()) * C / (C - 1)) / n if C > 1 else float("nan")
    su = max(se, 1 / n) if C > 1 else float("nan")
    p = 1 - SS.t.cdf(dbar / su, C - 1) if C > 1 else float("nan")
    tr = st.mean(x["tc"] for x in v) / st.mean(x["tp"] for x in v)
    print(f"{label:24s} rows {n:3d} cl {C:3d} | cand {100 * st.mean(x['ac'] for x in v):5.1f} parent {100 * st.mean(x['ap'] for x in v):5.1f} | "
          f"d {100 * dbar:+6.2f} pp se {100 * se:5.2f} (used {100 * su:5.2f}) p1 {p:.4f} | sum {sum(x['d'] for x in v):+6.2f} | tok x{tr:.3f}")


print(f"== {T['name']} on {T['tip']}: rows {len(T['rows'])}, rows with reads {len(recs)}, rows incomplete {miss}; reads held {sum(x['n'] for x in recs)}/{T['reads']}; "
      f"op_skipped {sum(x['skip'] for x in recs)}")
line("ALL", recs)
for s in sorted({x["src"] for x in recs}):
    line(f"  {s}", [x for x in recs if x["src"] == s])
tot = sum(x["d"] for x in recs)
print("LOBO mean d pp: " + "; ".join(f"-{s} {100 * (tot - sum(x['d'] for x in recs if x['src'] == s)) / max(1, len(recs) - sum(x['src'] == s for x in recs)):+.2f}"
                                    for s in sorted({x["src"] for x in recs})))
line("1 member", [x for x in recs if len(x["m"]) == 1])
line("2 members", [x for x in recs if len(x["m"]) == 2])
line("3+ members", [x for x in recs if len(x["m"]) >= 3])
for mb in T["members"]:
    line(f"  holds {mb['cid']}", [x for x in recs if mb["cid"] in x["m"]])
print("rows where 5+ members hold:")
for x in sorted((x for x in recs if len(x["m"]) >= 5), key=lambda x: x["r"]):
    print(f"  {x['r']:40s} members {len(x['m'])} | parent k {x['kp']}/{x['np']} | bundle k {x['kc']}/{x['nc']} | tok {x['tc']:.0f} vs {x['tp']:.0f}")
