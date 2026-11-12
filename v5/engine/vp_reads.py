"""Edited vision front-ends: signal-row reads and table (coordinator 11-12).
q ARM VIEW S0 S1   queue ARM (vp2|vpm|vp11) x VIEW (stock|f2048|f4096) on the 324 signal rows, samples S0..S1 (engine ARM upper-case)
t ARMS... [--s S0 S1] [--views ...]  table vs (a) stock p000|parent (cached) and (b) floor-only v5t000 x|2988ed164be18b6a (cached), on the same (row, s)"""
import glob, json, math, os, sys, collections
V4 = "/data/fd/mingdazhang/selfedit/v4"
sys.path[:0] = ["/workspace/fudan/mingdazhang/selfedit-v4/engine_v4", "/workspace/fudan/mingdazhang/selfedit-v4/loop"]
from common import smart_resize
ALL = [json.loads(l) for l in open(f"{V4}/rows/learn.jsonl")]
ROWS = {r["id"]: r for r in ALL if r.get("v4_role") in ("show", "monitor")}
JROWS = {r["id"]: r for r in ALL if r.get("v4_role") == "judge"}
assert len(ROWS) == 324 and len(JROWS) == 444
VIEWS = {"stock": None, "f2048": 2048, "f4096": 4096, "f1024": 1024, "f512": 512}
w = sys.argv[1]
if w == "q":
    arm, view, s0, s1 = sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
    judge = "--judge" in sys.argv
    st = {"min_image_tokens": VIEWS[view], "max_image_tokens": None, "temperature": None}
    Q = [dict(key=f"{arm}|{view}|{r}|{s}", mode="read", engine=arm.upper(), priority=-6, row=r, sample=s, prefix=None, settings=st)
         for s in range(s0, s1 + 1) for r in sorted(JROWS if judge else ROWS)]
    b = f"vp_{arm}_{view}{'_judge' if judge else ''}_s{s0}{s1}"; p = f"{V4}/q/in/{b}.jsonl"; assert not os.path.exists(p), p
    open(p + ".tmp", "w").write("".join(json.dumps(q) + "\n" for q in Q)); os.replace(p + ".tmp", p); print("queued", b, len(Q))
    sys.exit()
args = sys.argv[2:]; S = range(0, 2); views = ["stock"]
if "--s" in args: i = args.index("--s"); S = range(int(args[i + 1]), int(args[i + 2]) + 1); args = args[:i] + args[i + 3:]
if "--views" in args: i = args.index("--views"); views = args[i + 1:]; args = args[:i]
arms = args
from PIL import Image
def img_tokens(r):
    ps = r["image_paths"] if isinstance(r["image_paths"], list) else [r["image_paths"]]
    out = []
    for p in ps:
        with Image.open(p) as im: wd, ht = im.size
        hb, wb = smart_resize(ht, wd); out.append(dict(tok=(hb // 32) * (wb // 32), px=ht * wd, wh=(wd, ht)))
    return out
IT = {r: img_tokens(ROWS[r]) for r in ROWS}
def stratum(r):
    t = sum(x["tok"] for x in IT[r]) / len(IT[r])
    return "<400" if t < 400 else ("400-1000" if t <= 1000 else ">1000")
want = {}
for r in ROWS:
    for s in S:
        want[f"p000|parent|{r}|{s}"] = ("stock", r, s); want[f"x|2988ed164be18b6a|{r}|{s}"] = ("floor", r, s)
        for a in arms:
            for v in views: want[f"{a}|{v}|{r}|{s}"] = (f"{a}:{v}", r, s)
got, err = {}, collections.Counter()
for f in glob.glob(f"{V4}/q/out/*.jsonl"):
    for l in open(f, "rb"):
        if not l.startswith(b'{"key": "'): continue
        k = l[9:l.index(b'"', 9)].decode()
        if k in want and k not in got:
            x = json.loads(l)
            if x.get("error") or x.get("op_skipped"): err[want[k][0]] += 1; continue
            if x.get("content") is None or x.get("finish") == "abort": continue
            got[k] = x
D = collections.defaultdict(dict)
for k, x in got.items():
    a, r, s = want[k]; D[a][(r, s)] = x
def m_se(v):
    n = len(v); mu = sum(v) / n if n else float("nan")
    sd = math.sqrt(sum((y - mu) ** 2 for y in v) / (n - 1)) if n > 1 else float("nan")
    return mu, sd / math.sqrt(n) if n > 1 else float("nan")
benches = sorted({ROWS[r]["bench"] for r in ROWS})
def table(arm, keyset_filter=None, label="ALL rows"):
    A = D[arm]; lines = []
    for ref in ("stock", "floor"):
        R = D[ref]
        cells = [(r, s) for (r, s) in A if (r, s) in R and (keyset_filter is None or keyset_filter(r))]
        per_b = {}
        for b in benches + ["ALL"]:
            cb = [c for c in cells if b == "ALL" or ROWS[c[0]]["bench"] == b]
            if not cb: continue
            byrow = collections.defaultdict(list)
            for c in cb: byrow[c[0]].append(int(bool(A[c].get("ok"))) - int(bool(R[c].get("ok"))))
            d, se = m_se([sum(v) / len(v) for v in byrow.values()])
            acc = sum(bool(A[c].get("ok")) for c in cb) / len(cb); racc = sum(bool(R[c].get("ok")) for c in cb) / len(cb)
            gt = sum(A[c].get("tokens") or 0 for c in cb) / len(cb); rgt = sum(R[c].get("tokens") or 0 for c in cb) / len(cb)
            pt = sum(A[c].get("prompt_tokens") or 0 for c in cb) / len(cb); rpt = sum(R[c].get("prompt_tokens") or 0 for c in cb) / len(cb)
            per_b[b] = (len(cb), acc, racc, d, se, gt, rgt, pt, rpt)
        mac = [per_b[b][3] for b in benches if b in per_b]
        lines.append(f"  vs {ref:5s} ({label}):")
        for b in benches + ["ALL"]:
            if b not in per_b: continue
            n, acc, racc, d, se, gt, rgt, pt, rpt = per_b[b]
            lines.append(f"    {b:11s} n {n:4d} | acc {100*acc:5.1f} vs {100*racc:5.1f} | d {100*d:+5.1f}+-{100*se:4.1f} | gen {gt:6.0f} vs {rgt:6.0f} | prompt {pt:6.0f} vs {rpt:6.0f} | total {gt+pt:6.0f} vs {rgt+rpt:6.0f} (x{(gt+pt)/max(1,rgt+rpt):.2f})")
        if mac: lines.append(f"    MACRO d {100*sum(mac)/len(mac):+5.2f} pp over {len(mac)} benchmarks")
    return lines
print("reads per arm:", {a: len(D[a]) for a in D}, "not-fit / error lines:", dict(err))
for a in [f"{x}:{v}" for x in arms for v in views]:
    if not D[a]: print(a, "no reads yet"); continue
    print(f"=== {a}  samples {list(S)}"); print("\n".join(table(a)))
    for st in ("<400", "400-1000", ">1000"):
        print("\n".join(table(a, lambda r, st=st: stratum(r) == st, label=f"stock tokens/image {st}")))
cap = [(r, ROWS[r]["bench"], max(x["px"] for x in IT[r]), max(x["tok"] for x in IT[r])) for r in ROWS if any(x["px"] > 1048576 for x in IT[r])]
print(f"VP2 1-MP cap shrinks {len(cap)} rows (an image > 1,048,576 px):", collections.Counter(b for _, b, _, _ in cap))
print("  ", [f"{r} ({px/1e6:.1f} MP, stock {t} tok)" for r, b, px, t in sorted(cap, key=lambda z: -z[2])[:40]])
print("strata (rows):", collections.Counter(stratum(r) for r in ROWS), "by bench:", {b: dict(collections.Counter(stratum(r) for r in ROWS if ROWS[r]["bench"] == b)) for b in benches})
