"""tok_tail.py ARM [S1]: generated-token medians / p90 / reads >16k for a served arm vs the cached VP2M512 E reads on the 249 unrouted SIGNAL rows, per bench.
neither is shown both on s0-3 (the table's reference) and on s0-1 (same sample count as the arm)."""
import glob, json, sys, collections, statistics
V4 = "/data/fd/mingdazhang/selfedit/v4"
ROWS = {r["id"]: r for r in map(json.loads, open(f"{V4}/rows/learn.jsonl")) if r.get("v4_role") in ("show", "monitor")}
ROUTED = {l.strip() for l in open(f"{V4}/scratch/e1_routed_signal_rows.txt") if l.strip()}
arm = sys.argv[1]; S1 = int(sys.argv[2]) if len(sys.argv) > 2 else 1
want = {}
for r in ROWS:
    if r in ROUTED: continue
    for s in range(4): want[f"vp2m512|stock|{r}|{s}"] = ("n03" if s > S1 else "n01", r)
    for s in range(S1 + 1): want[f"{arm}|stock|{r}|{s}"] = (arm, r)
T = collections.defaultdict(lambda: collections.defaultdict(list)); seen = set()
for f in glob.glob(f"{V4}/q/out/vp_*.jsonl"):
    for l in open(f, "rb"):
        if not l.startswith(b'{"key": "'): continue
        k = l[9:l.index(b'"', 9)].decode()
        if k in want and k not in seen:
            x = json.loads(l)
            if x.get("error") or x.get("content") is None or x.get("finish") == "abort": continue
            seen.add(k); a, r = want[k]; b = ROWS[r].get("bench") or r.split("/")[0]
            T[a][b].append(x.get("tokens") or 0); T[a]["ALL"].append(x.get("tokens") or 0)
            if a == "n01": T["n03"][b].append(x.get("tokens") or 0); T["n03"]["ALL"].append(x.get("tokens") or 0)
def st(v):
    if not v: return "n/a"
    v = sorted(v); p90 = v[min(len(v) - 1, int(round(0.9 * (len(v) - 1))))]
    return f"n {len(v):3d} mean {sum(v)/len(v):6.0f} med {statistics.median(v):6.0f} p90 {p90:6d} >16k {sum(t > 16384 for t in v):2d}"
for b in sorted(set(T[arm]) - {"ALL"}) + ["ALL"]:
    print(f"== {b}")
    for a, lbl in ((arm, arm), ("n01", "neither s0-1"), ("n03", "neither s0-3")): print(f"    {lbl:13s} {st(T[a][b])}")
