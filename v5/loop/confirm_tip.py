"""Server-vs-server confirmation of a tip change (coordinator 11-10 ~21:10 CST, for t004 = t002 + g004c30): on the JUDGE rows, the paired
difference per row (mean right of NEW's server-read samples) - (mean right of OLD's cached samples), with the cluster-robust se (picture
clusters), the one-sided p (judge_bh.pval, nu = clusters - 1), per source, pooled and macro, LOBO on the per-source sums, and the token
ratio. Samples read by a W worker (server name W*) are left out of NEW unless --all; a row counts when it has >= MIN NEW samples and
>= 1 OLD sample. Read only (never writes state).
usage: confirm_tip.py RUN_DIR NEW_TIP OLD_TIP [MIN] [--all | --wonly] [--json OUT]"""
import collections, json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(os.path.dirname(HERE), "harness")]
import gen  # noqa: E402
import judge_bh as BH  # noqa: E402

args = [a for a in sys.argv[1:] if not a.startswith("--")]
RUN, NEW, OLD = args[0], args[1], args[2]
MIN = int(args[3]) if len(args) > 3 else 2
ALL = "--all" in sys.argv
WONLY = "--wonly" in sys.argv
OUT = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
R = gen.Run(gen.args(["--run", RUN]))
tips = {t["id"]: t for t in R.state["tips"] + [R.state["tip"]]}
EN, EO = tips[NEW]["entries"], tips[OLD]["entries"]
R.load_routes(EN, R.judge_rows)
R.load_routes(EO, R.judge_rows)
UP = 8


def is_w(x):
    return str(x.get("server") or "").startswith("W")


rows, srv = [], collections.Counter()
have = collections.Counter()
for r in R.judge_rows:
    ns = R.samples(EN, r, UP)
    os_ = R.samples(EO, r, UP)
    for x in ns:
        srv["W" if is_w(x) else "server"] += 1
    nk = [x for x in ns if is_w(x)] if WONLY else [x for x in ns if ALL or not is_w(x)]
    have[min(len(nk), UP)] += 1
    if len(nk) < MIN or not os_:
        continue
    a = sum(bool(x.get("ok")) for x in nk) / len(nk)
    b = sum(bool(x.get("ok")) for x in os_) / len(os_)
    ta = sum(x.get("tokens") or 0 for x in nk) / len(nk)
    tb = sum(x.get("tokens") or 0 for x in os_) / len(os_)
    rows.append(dict(r=r, bench=R.rows[r]["bench"], cl=R.rows[r].get("pic") or r, d=a - b, a=a, b=b, ta=ta, tb=tb, nn=len(nk), no=len(os_)))


def stats(rs):
    if not rs:
        return None
    N = len(rs)
    d = sum(x["d"] for x in rs) / N
    cl = collections.defaultdict(list)
    for x in rs:
        cl[x["cl"]].append(x["d"])
    C = len(cl)
    v = sum((sum(v) - len(v) * d) ** 2 for v in cl.values()) / N ** 2 * (C / (C - 1)) if C > 1 else float("nan")
    se = math.sqrt(v) if C > 1 else 1.0
    p, su = BH.pval(d, se, N, max(C - 1, 1))
    tr = sum(x["ta"] for x in rs) / max(sum(x["tb"] for x in rs), 1e-9)
    return dict(n=N, clusters=C, d=d, se=se, se_used=su, p=p, acc_new=sum(x["a"] for x in rs) / N, acc_old=sum(x["b"] for x in rs) / N,
                tok_ratio=tr, new_samples=sum(x["nn"] for x in rs), old_samples=sum(x["no"] for x in rs))


by = collections.defaultdict(list)
for x in rows:
    by[x["bench"]].append(x)
S = {b: stats(v) for b, v in sorted(by.items())}
pool = stats(rows)
mac = None
if S:
    ds = [s["d"] for s in S.values()]
    md = sum(ds) / len(ds)
    mse = math.sqrt(sum(s["se_used"] ** 2 for s in S.values())) / len(S)
    nu = sum(s["clusters"] - 1 for s in S.values())
    mp = float(BH._t.sf(md / mse, max(nu, 1)))
    mac = dict(d=md, se=mse, nu=nu, p=mp)
lobo = BH.lobo_ok({b: (sum(x["d"] for x in v), len(v)) for b, v in by.items()}) if len(by) >= 2 else False
print(f"CONFIRM {NEW} vs {OLD} on JUDGE ({len(R.judge_rows)} rows), NEW samples {'all engines' if ALL else 'server-read only (W left out)'}, "
      f"row needs >= {MIN} NEW and >= 1 OLD sample")
print(f"NEW samples by engine: {dict(srv)}; JUDGE rows by NEW sample count (after the filter): {dict(sorted(have.items()))}")
print(f"rows used {len(rows)} of {len(R.judge_rows)}")
fmt = lambda s: (f"n {s['n']:3d} cl {s['clusters']:3d} acc {s['acc_new']:.3f} vs {s['acc_old']:.3f} d {s['d']:+.4f} se {s['se']:.4f} "
                 f"p {s['p']:.4f} tok x{s['tok_ratio']:.3f} (samples {s['new_samples']}/{s['old_samples']})")
for b, s in S.items():
    print(f"  {b:11s} {fmt(s)}")
if pool:
    print(f"  {'POOLED':11s} {fmt(pool)}")
if mac:
    print(f"  {'MACRO':11s} d {mac['d']:+.4f} se {mac['se']:.4f} nu {mac['nu']} p {mac['p']:.4f}")
print(f"  LOBO {'holds' if lobo else 'FAILS'}; rule: pooled p <= .025 and LOBO -> {NEW} stands, else revert to {OLD}: "
      f"{'STANDS' if pool and pool['p'] <= .025 and lobo else 'REVERT'} (on {len(rows)} rows)")
if OUT:
    json.dump(dict(new=NEW, old=OLD, min=MIN, all=ALL, engines=dict(srv), have=dict(have), sources=S, pooled=pool, macro=mac, lobo=lobo,
                   rows=len(rows)), open(OUT, "w"), indent=1)
