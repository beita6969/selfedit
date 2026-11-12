"""Smoke of the fl075 T servers (coordinator 11-11): 48 signal rows (8 per benchmark) x sample 0, engine W with the FLs8_L24-34_+0.75
edit -> the dispatcher routes them to the T servers whose edit_sha matches (same request build and CRN seeds as the e7 W reads).
Compared on the same (row, s) with the e7 W reads of FL +0.75 (fls8_L24_34_p075_s0) and the stock floor reads (x|2988ed164be18b6a).
usage: fl_smoke.py q | t"""
import glob, json, os, sys, collections
V4 = "/data/fd/mingdazhang/selfedit/v4"; sys.path.insert(0, "/workspace/fudan/mingdazhang/selfedit-v4/loop")
import ecomp
EDIT = {"op": "surgeon", "file": f"{V4}/cells/fable_int/edits/FLs8_L24-34_+0.75.json"}
ROWS = {r["id"]: r for r in map(json.loads, open(f"{V4}/rows/learn.jsonl")) if r.get("v4_role") in ("show", "monitor")}
assert len(ROWS) == 324
B = "fl075_smoke"
by = collections.defaultdict(list)
for r in sorted(ROWS): by[ROWS[r]["bench"]].append(r)
PICK = [rs[i * len(rs) // 8] for b in sorted(by) for rs in [by[b]] for i in range(8)]
ST = {"min_image_tokens": 4096, "max_image_tokens": None, "temperature": None}
w = sys.argv[1]
if w == "q":
    Q = [dict(key=f"FL075T|{r}|0", mode="read", engine="W", priority=-9, row=r, sample=0, prefix=None, settings=ST, edit=EDIT) for r in PICK]
    p = f"{V4}/q/in/{B}.jsonl"; assert not os.path.exists(p)
    open(p + ".tmp", "w").write("".join(json.dumps(q) + "\n" for q in Q)); os.replace(p + ".tmp", p); print("queued", B, len(Q))
else:
    def outs(b):
        o = {}
        for l in open(f"{V4}/q/out/{b}.jsonl"):
            x = json.loads(l)
            if not x.get("error") and x.get("content") is not None: o[x["key"]] = x
        return o
    T = {k.split("|")[1]: x for k, x in outs(B).items()}
    W = {x["key"].split("|")[2]: x for x in outs("fls8_L24_34_p075_s0").values() if x["key"].endswith("|0")}
    want = {f"x|2988ed164be18b6a|{r}|0" for r in PICK}; F = {}
    for f in glob.glob(f"{V4}/q/out/*.jsonl"):
        for l in open(f, "rb"):
            if l.startswith(b'{"key": "x|2988ed164be18b6a|'):
                k = l[9:l.index(b'"', 9)].decode()
                if k in want and k not in F:
                    x = json.loads(l)
                    if not x.get("error") and x.get("content") is not None: F[k.split("|")[2]] = x
    rows = [r for r in PICK if r in T and r in W and r in F]
    srv = collections.Counter(T[r].get("server") for r in T)
    ok = lambda x: bool(x.get("ok")); pred = lambda x, r: ecomp.pred(x, ROWS[r])
    n = len(rows)
    acc_t, acc_w, acc_f = (sum(ok(D[r]) for r in rows) / n for D in (T, W, F))
    agree_tw = sum(pred(T[r], r) == pred(W[r], r) for r in rows) / n
    agree_tf = sum(pred(T[r], r) == pred(F[r], r) for r in rows) / n
    tok = lambda D: sum(D[r].get("tokens") or 0 for r in rows) / n
    fin = collections.Counter(T[r].get("finish") for r in T)
    print(json.dumps(dict(reads=len(T), of=len(PICK), compared=n, servers=dict(srv), finish=dict(fin), errors=len(PICK) - len(T),
                          acc_fl075=round(acc_t, 3), acc_W_fl075_e7=round(acc_w, 3), acc_stock_floor=round(acc_f, 3),
                          agree_fl075_vs_W=round(agree_tw, 3), agree_fl075_vs_stock=round(agree_tf, 3),
                          tokens_fl075=round(tok(T)), tokens_W=round(tok(W)), tokens_stock=round(tok(F)),
                          tok_ratio_vs_stock=round(tok(T) / max(1, tok(F)), 2), tok_ratio_W_vs_stock=round(tok(W) / max(1, tok(F)), 2))))
