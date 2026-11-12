"""Prefix-cache A/A: pcAA reads of t002 (caching ON, samples 900/901) vs t002's cached parent samples 0-7 (caching OFF).
Per row d = mean(ok, new) - mean(ok, cached); se over rows (rows are the clusters: one d per row); per source and macro
(macro = mean of source means, se = sqrt(sum se_s^2) / S). Also mean generated tokens and unfinished share (finish == "length") per side.
usage: python pcaa_report.py [out_batch ...]"""
import glob, json, math, statistics as st, sys, collections

V4 = "/data/fd/mingdazhang/selfedit/v4"
outs = sys.argv[1:] or ["pcAA_t002_1"]
new = collections.defaultdict(list)
for b in outs:
    for l in open(f"{V4}/q/out/{b}.jsonl"):
        d = json.loads(l)
        if "error" in d:
            continue
        new[d["row"]].append(d)
ref = [json.loads(l) for l in open(f"{V4}/pcAA/pcAA_t002_refkeys.jsonl")]
want = {k for r in ref for k in r["cached"]}
old = {}
for f in glob.glob(f"{V4}/q/out/*.jsonl"):
    if "/pcAA" in f:
        continue
    for l in open(f):
        if '"x|' not in l:
            continue
        d = json.loads(l)
        if d.get("key") in want and "error" not in d:
            old[d["key"]] = d
rows = []
for r in ref:
    o = [old[k] for k in r["cached"] if k in old]
    n = new.get(r["row"], [])
    if not o or not n:
        continue
    src = r["row"].split("/")[0]
    rows.append(dict(src=src, d=st.mean(bool(x["ok"]) for x in n) - st.mean(bool(x["ok"]) for x in o),
                     a_new=st.mean(bool(x["ok"]) for x in n), a_old=st.mean(bool(x["ok"]) for x in o),
                     t_new=[x["tokens"] for x in n], t_old=[x["tokens"] for x in o],
                     u_new=[x.get("finish") == "length" for x in n], u_old=[x.get("finish") == "length" for x in o]))
print(f"rows with both sides: {len(rows)} / {len(ref)}; new reads {sum(len(v) for v in new.values())}; cached reads found {len(old)} / {len(want)}")


def summ(rs):
    ds = [r["d"] for r in rs]
    m = st.mean(ds)
    se = st.stdev(ds) / math.sqrt(len(ds)) if len(ds) > 1 else float("nan")
    return m, se


by = collections.defaultdict(list)
for r in rows:
    by[r["src"]].append(r)
print(f"{'source':12s} {'n':>4s} {'acc_off':>8s} {'acc_on':>8s} {'diff_pp':>8s} {'se_pp':>6s} {'z':>6s} {'tok_off':>8s} {'tok_on':>8s} {'unf_off':>8s} {'unf_on':>7s}")
ms, ses = [], []
for s in sorted(by):
    rs = by[s]
    m, se = summ(rs)
    ms.append(m); ses.append(se)
    t_o = st.mean(t for r in rs for t in r["t_old"]); t_n = st.mean(t for r in rs for t in r["t_new"])
    u_o = st.mean(u for r in rs for u in r["u_old"]); u_n = st.mean(u for r in rs for u in r["u_new"])
    print(f"{s:12s} {len(rs):4d} {100 * st.mean(r['a_old'] for r in rs):8.1f} {100 * st.mean(r['a_new'] for r in rs):8.1f} {100 * m:+8.2f} {100 * se:6.2f} {m / se if se else 0:+6.2f} {t_o:8.0f} {t_n:8.0f} {100 * u_o:7.1f}% {100 * u_n:6.1f}%")
mm = st.mean(ms); mse = math.sqrt(sum(x * x for x in ses)) / len(ses)
t_o = st.mean(t for r in rows for t in r["t_old"]); t_n = st.mean(t for r in rows for t in r["t_new"])
u_o = st.mean(u for r in rows for u in r["u_old"]); u_n = st.mean(u for r in rows for u in r["u_new"])
print(f"{'MACRO':12s} {len(rows):4d} {'':8s} {'':8s} {100 * mm:+8.2f} {100 * mse:6.2f} {mm / mse:+6.2f} {t_o:8.0f} {t_n:8.0f} {100 * u_o:7.1f}% {100 * u_n:6.1f}%")
pm, pse = summ(rows)
print(f"pooled (row-weighted): diff {100 * pm:+.2f} pp, se {100 * pse:.2f}, z {pm / pse:+.2f}; tokens on/off {t_n / t_o - 1:+.1%}")
worst = max(abs(m / s) for m, s in zip(ms, ses) if s)
print(f"rule: |macro z| <= 2: {abs(mm / mse) <= 2}; max |source z| = {worst:.2f} <= 2.5: {worst <= 2.5}; tokens within 3 %: {abs(t_n / t_o - 1) <= 0.03}")
