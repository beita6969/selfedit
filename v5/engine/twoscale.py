"""TWOSCALE pilot (coordinator 11-12 02:22): the single image of a SIGNAL row sent twice in one user turn, stock weights, stock E server.
order a = floor-4096 copy (the v5t000 floor view) then the stock view; order b = stock view then floor copy. No added text.
q            queue twoscale_a on samples 0,1 and twoscale_b on samples 2,3 (one-image SIGNAL rows), engine E, priority -6
est          prompt-size estimate (image tokens of both copies) before queueing
t            table vs floor-only base x|2988ed164be18b6a (cached 8) and stock p000|parent (cached 4): per-row means, per bench + macro,
             strata by stock tokens/image and by cached floor effect; 10 reads that comment on two images"""
import glob, json, math, os, re, sys, collections
V4 = "/data/fd/mingdazhang/selfedit/v4"
sys.path[:0] = ["/workspace/fudan/mingdazhang/selfedit-v4/engine_v4", "/workspace/fudan/mingdazhang/selfedit-v4/loop"]
from common import image_url_for_tokens
ROWS = {r["id"]: r for r in map(json.loads, open(f"{V4}/rows/learn.jsonl")) if r.get("v4_role") in ("show", "monitor")}
assert len(ROWS) == 324
def paths(r):
    p = ROWS[r].get("image_paths") or []
    return p if isinstance(p, list) else [p]
ONE = sorted(r for r in ROWS if len(paths(r)) == 1)
w = sys.argv[1]
if w == "est":
    big = [(r, image_url_for_tokens(paths(r)[0], 4096, None), image_url_for_tokens(paths(r)[0], None, None)) for r in ONE]
    tot = sorted(b + o for _, b, o in big)
    print("one-image rows", len(ONE), "of", len(ROWS), "| image tokens both copies: median", tot[len(tot) // 2], "max", tot[-1],
          "| rows > 20k", sum(t > 20000 for t in tot), "> 30k", sum(t > 30000 for t in tot), "> 41,904", sum(t > 41904 for t in tot))
    sys.exit()
if w == "q":
    Q = []
    for order, ss in (("a", (0, 1)), ("b", (2, 3))):
        st = {"min_image_tokens": None, "max_image_tokens": None, "temperature": None, "twoscale": order, "twoscale_floor": 4096}
        Q += [dict(key=f"twoscale_{order}|{r}|{s}", mode="read", engine="E", priority=-6, row=r, sample=s, prefix=None, settings=st)
              for s in ss for r in ONE]
    b = "twoscale_pilot"; p = f"{V4}/q/in/{b}.jsonl"; assert not os.path.exists(p), p
    open(p + ".tmp", "w").write("".join(json.dumps(q) + "\n" for q in Q)); os.replace(p + ".tmp", p); print("queued", b, len(Q))
    sys.exit()
# ---- table
from PIL import Image
from common import smart_resize
def stock_tok(r):
    t = []
    for p in paths(r):
        with Image.open(p) as im: wd, ht = im.size
        hb, wb = smart_resize(ht, wd); t.append((hb // 32) * (wb // 32))
    return sum(t) / len(t)
want = {}
for r in ONE:
    for s in range(8): want[f"x|2988ed164be18b6a|{r}|{s}"] = ("floor", r, s)
    for s in range(4): want[f"p000|parent|{r}|{s}"] = ("stock", r, s)
    for s in (0, 1): want[f"twoscale_a|{r}|{s}"] = ("A", r, s)
    for s in (2, 3): want[f"twoscale_b|{r}|{s}"] = ("B", r, s)
got, err = {}, collections.Counter()
for f in glob.glob(f"{V4}/q/out/*.jsonl"):
    for l in open(f, "rb"):
        if not l.startswith(b'{"key": "'): continue
        k = l[9:l.index(b'"', 9)].decode()
        if k in want and k not in got:
            x = json.loads(l)
            if x.get("error"): err[want[k][0]] += 1; continue
            if x.get("content") is None or x.get("finish") == "abort": continue
            got[k] = x
D = collections.defaultdict(lambda: collections.defaultdict(list))   # arm -> row -> [records]
for k, x in got.items():
    a, r, s = want[k]; D[a][r].append(x)
for r in ONE:
    D["AB"][r] = D["A"][r] + D["B"][r]
def mean(v): return sum(v) / len(v) if v else float("nan")
def acc(a, r): return mean([float(bool(x.get("ok"))) for x in D[a][r]])
def tok(a, r, f): return mean([x.get(f) or 0 for x in D[a][r]])
FE = {r: acc("floor", r) - acc("stock", r) for r in ONE if D["floor"][r] and D["stock"][r]}
def fstrat(r):
    v = FE.get(r)
    return "n/a" if v is None else ("floor helped" if v >= 0.25 else ("floor hurt" if v <= -0.25 else "between"))
def tstrat(r):
    t = stock_tok(r); return "<400" if t < 400 else ("400-1000" if t <= 1000 else ">1000")
benches = sorted({ROWS[r]["bench"] for r in ONE})
print("reads:", {a: sum(len(v) for v in D[a].values()) for a in ("A", "B", "floor", "stock")}, "errors (not-fit etc.):", dict(err))
def table(arm, filt=None, label="all one-image rows"):
    rows = [r for r in ONE if D[arm][r] and D["floor"][r] and D["stock"][r] and (filt is None or filt(r))]
    print(f"  [{arm}] {label}: {len(rows)} rows")
    mac = {"floor": [], "stock": []}
    for b in benches + ["ALL"]:
        rb = [r for r in rows if b == "ALL" or ROWS[r]["bench"] == b]
        if not rb: continue
        A = mean([acc(arm, r) for r in rb]); Fl = mean([acc("floor", r) for r in rb]); St = mean([acc("stock", r) for r in rb])
        def dse(ref):
            v = [acc(arm, r) - acc(ref, r) for r in rb]; m = mean(v)
            se = math.sqrt(sum((y - m) ** 2 for y in v) / (len(v) - 1) / len(v)) if len(v) > 1 else float("nan")
            return m, se
        (df, sf), (ds, ss) = dse("floor"), dse("stock")
        if b != "ALL": mac["floor"].append(df); mac["stock"].append(ds)
        g = [mean([tok(a, r, "tokens") for r in rb]) for a in (arm, "floor", "stock")]
        p = [mean([tok(a, r, "prompt_tokens") for r in rb]) for a in (arm, "floor", "stock")]
        print(f"    {b:11s} n {len(rb):3d} | acc {100*A:5.1f} floor {100*Fl:5.1f} stock {100*St:5.1f} | d_floor {100*df:+5.1f}+-{100*sf:4.1f} d_stock {100*ds:+5.1f}+-{100*ss:4.1f}"
              f" | gen {g[0]:5.0f}/{g[1]:5.0f}/{g[2]:5.0f} prompt {p[0]:5.0f}/{p[1]:5.0f}/{p[2]:5.0f} total {g[0]+p[0]:5.0f}/{g[1]+p[1]:5.0f}/{g[2]+p[2]:5.0f}")
    if mac["floor"]:
        print(f"    MACRO d_floor {100*mean(mac['floor']):+5.2f}  d_stock {100*mean(mac['stock']):+5.2f} pp over {len(mac['floor'])} benchmarks  (token columns: arm/floor/stock)")
for arm in ("AB", "A", "B"):
    print(f"=== twoscale {arm} (A = floor copy first, s0-1; B = stock view first, s2-3)")
    table(arm)
    if arm == "AB":
        for st in ("<400", "400-1000", ">1000"): table(arm, lambda r, st=st: tstrat(r) == st, f"stock tokens/image {st}")
        for st in ("floor helped", "between", "floor hurt"): table(arm, lambda r, st=st: fstrat(r) == st, f"cached {st}")
print("strata rows:", collections.Counter(tstrat(r) for r in ONE), collections.Counter(fstrat(r) for r in ONE))
pat = re.compile(r"(two images|both images|second image|first image|the other image|same (image|picture|photo)|identical|higher[- ]resolution|zoomed|duplicate|two versions|two copies)", re.I)
ex = []
for a in ("A", "B"):
    for r in ONE:
        for x in D[a][r]:
            txt = (x.get("reasoning") or "") + "\n" + (x.get("content") or "")
            m = pat.search(txt)
            if m: ex.append((a, r, x.get("sample"), bool(x.get("ok")), txt[max(0, m.start() - 220): m.end() + 260].replace("\n", " ")))
print(f"reads that mention the two images: {len(ex)} of {sum(len(D[a][r]) for a in ('A', 'B') for r in ONE)}")
for e in ex[:: max(1, len(ex) // 10)][:10]:
    print(f"  [{e[0]} {e[1]} s{e[2]} ok={e[3]}] ...{e[4]}...")
