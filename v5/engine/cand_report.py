"""Candidate vs tip-base table: python3 cand_report.py NAME BATCH[,BATCH...] [--gate 1500] [--json OUT]
Each result's base = the tip's own read under its "base_ref" key (latest result line). Paired per row (row-clustered se over rows with >= 1 pair).
gated = base reasoning >= gate tokens (extend: the op's own gate decision; W: base reasoning tokens from the extend trace if any, else base tokens).
Repetition: share of the candidate's ADDED reasoning lines (>= 20 chars) that repeat an earlier line of the same trace; 'looping' = share > 0.3."""
import argparse, collections, glob, json, math, statistics, sys
V4 = "/data/fd/mingdazhang/selfedit/v4"
ap = argparse.ArgumentParser(); ap.add_argument("name"); ap.add_argument("batches"); ap.add_argument("--gate", type=int, default=1500)
ap.add_argument("--json", default="")
a = ap.parse_args()
res = {}
for b in a.batches.split(","):
    for f in glob.glob(f"{V4}/q/out/{b}.jsonl"):
        for l in open(f):
            x = json.loads(l); res[x["key"]] = x
reqs = {}
for b in a.batches.split(","):
    for f in glob.glob(f"{V4}/q/in/{b}.jsonl"):
        for l in open(f):
            q = json.loads(l); reqs[q["key"]] = q
need = {(q.get("base_ref") or q.get("base_key")) for q in reqs.values()}
base = {}
for f in glob.glob(f"{V4}/q/out/*.jsonl"):
    for l in open(f, "rb"):
        if l.startswith(b'{"key": "x|') or l.startswith(b'{"key": "p000|'):
            k = l[9:l.index(b'"', 9)].decode()
            if k in need:
                x = json.loads(l)
                if not x.get("error") and x.get("content") is not None:
                    base[k] = x
def rep_share(prev, added):
    seen = set(s.strip() for s in prev.split("\n") if len(s.strip()) >= 20); n = r = 0
    for s in added.split("\n"):
        s = s.strip()
        if len(s) < 20:
            continue
        n += 1; r += s in seen; seen.add(s)
    return r / n if n else 0.0
P = []
for k, q in reqs.items():
    x = res.get(k)
    if not x or x.get("error") or x.get("op_skipped") or (q.get("base_ref") or q.get("base_key")) not in base:
        continue
    b = base[(q.get("base_ref") or q.get("base_key"))]
    tr = (x.get("op_trace") or [{}])[0]
    if tr.get("op") == "extend":
        g = bool(tr.get("extended")); added = int(x.get("tokens") or 0) - int(b.get("tokens") or 0)
        R0 = b.get("reasoning") or ""; addtxt = (x.get("reasoning") or "")[len(R0.rstrip()):] if g else ""
        rs = tr.get("rounds") or []
        close30 = bool(rs and rs[0].get("closed_at_once"))
    else:
        bt = int(b.get("tokens") or 0)
        g = bt > a.gate                                             # W cells: the tip read (base) is longer than gate tokens
        added = int(x.get("tokens") or 0) - bt; addtxt = x.get("reasoning") or ""; close30 = None; R0 = ""
    P.append(dict(row=q["row"], s=q["sample"], bench=q["row"].split("/")[0], g=g, ok=float(bool(x.get("ok"))), bok=float(bool(b.get("ok"))),
                  tok=int(x.get("tokens") or 0), btok=int(b.get("tokens") or 0), added=added, cut=(x.get("finish") == "length"), loop=bool(x.get("loop")),
                  rep=rep_share(R0, addtxt) if addtxt else 0.0, close30=close30, fb=bool(tr.get("fallback"))))
def table(ps, label):
    if not ps:
        print(f"  {label:22s} n 0"); return None
    byr = collections.defaultdict(list)
    for p in ps:
        byr[p["row"]].append(p["ok"] - p["bok"])
    d = [sum(v) / len(v) for v in byr.values()]; m = sum(d) / len(d)
    se = (statistics.pstdev(d) / math.sqrt(len(d) - 1)) if len(d) > 1 else float("nan")
    z = m / se if se and se == se and se > 0 else 0.0; pv = math.erfc(abs(z) / math.sqrt(2))
    wr = sum(1 for p in ps if p["ok"] > p["bok"]); rw = sum(1 for p in ps if p["ok"] < p["bok"])
    ad = sorted(p["added"] for p in ps)
    mult = sum(p["tok"] for p in ps) / max(1, sum(p["btok"] for p in ps))
    cut = sum(p["cut"] for p in ps) / len(ps); lp = sum(p["loop"] or p["rep"] > 0.3 for p in ps) / len(ps)
    c30 = [p["close30"] for p in ps if p["close30"] is not None]
    print(f"  {label:22s} reads {len(ps):4d} rows {len(byr):3d}  acc {sum(p['ok'] for p in ps)/len(ps):.3f} vs base {sum(p['bok'] for p in ps)/len(ps):.3f}  "
          f"d {m*100:+.2f} pp se {se*100:.2f} p {pv:.3f}  w->r {wr} r->w {rw}  added med {ad[len(ad)//2]} p90 {ad[min(len(ad)-1, int(.9*len(ad)))]}  "
          f"tok x{mult:.2f}  cut {cut:.1%} loop/rep {lp:.1%}" + (f"  close<=30 {sum(c30)/len(c30):.1%}" if c30 else ""))
    return dict(label=label, reads=len(ps), rows=len(byr), d=m, se=se, p=pv, wr=wr, rw=rw, mult=mult)
print(f"{a.name}: results {sum(1 for k in reqs if k in res)}/{len(reqs)} (errors {sum(1 for k in reqs if k in res and res[k].get('error'))}), paired {len(P)}")
out = {}
out["gated"] = table([p for p in P if p["g"]], "gated")
out["other"] = table([p for p in P if not p["g"]], "other")
out["all"] = table(P, "all")
for bn in sorted({p["bench"] for p in P}):
    table([p for p in P if p["bench"] == bn and p["g"]], f"gated {bn}")
if a.json:
    json.dump(out, open(a.json, "w"))
