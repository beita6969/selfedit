"""PROBE hdr_probe table: hdr - tip (ood_tipE / sig_tipE), per bench + pooled, picture-clustered se; by routed entry; tokens; hint/trap regex share."""
import collections, glob, json, math, os, re
V4 = "/data/fd/mingdazhang/selfedit/v4"
rows = {r["id"]: r for r in map(json.loads, open(f"{V4}/rows/learn.jsonl"))}
tip = json.load(open(f"{V4}/runs/main_1110/state.json"))["tip"]["entries"]
GR = {"E1": [1], "E2/E5": [2, 5], "E9": [9], "E11": [11], "E6": [6], "H": [3, 4, 7, 8, 10]}
PY = {}
for f in glob.glob(f"{V4}/q/out/*.jsonl"):
    for l in open(f, "rb"):
        if l.startswith(b'{"key": "rt1|'):
            x = json.loads(l); _, c, r = x["key"].split("|", 2)
            if not x.get("error"): PY[(c, r)] = x.get("p", 0) >= 0.5 if isinstance(x.get("p"), (int, float)) else bool(x.get("yes"))
HINT = re.compile(r"\bhint|\btrap|suggests that|red herring|boilerplate|generic instruction", re.I)
def load(b):
    o = {}
    for l in open(f"{V4}/q/out/{b}.jsonl"):
        x = json.loads(l)
        if not x.get("error") and x.get("content") is not None: k = x["key"].split("|"); o[(k[-2], int(k[-1]))] = x
    return o
ok = lambda x: float(bool(x.get("ok")))
def cl(pairs):
    byr = collections.defaultdict(list)
    for (r, s), d in pairs: byr[r].append(d)
    if len(byr) < 2: return None
    v = {r: sum(x) / len(x) for r, x in byr.items()}; n = len(v); m = sum(v.values()) / n; byc = collections.defaultdict(float)
    for r, d in v.items(): byc[rows[r].get("pic") or r] += d - m
    G = len(byc); return m, math.sqrt(sum(x * x for x in byc.values()) * G / max(G - 1, 1)) / n, n
fm = lambda t: "n/a" if not t else f"{100*t[0]:+.1f}±{100*t[1]:.1f} (n={t[2]})"
for side in ("ood", "sig"):
    H, T = load(f"{side}_hdr_probe"), load(f"{side}_tipE"); keys = sorted(set(H) & set(T))
    print(f"\n== {side}: hdr {len(H)} tip {len(T)} paired {len(keys)}")
    def line(lab, ks):
        if not ks: return
        t = cl([(k, ok(H[k]) - ok(T[k])) for k in ks]); tk = lambda D, f: sum(D[k].get(f) or 0 for k in ks) / len(ks)
        hs = lambda D: 100 * sum(bool(HINT.search((D[k].get("reasoning") or "") + " " + (D[k].get("content") or ""))) for k in ks) / len(ks)
        print(f"{lab}: tip {100*sum(ok(T[k]) for k in ks)/len(ks):.1f} hdr {100*sum(ok(H[k]) for k in ks)/len(ks):.1f} | hdr-tip {fm(t)} | prompt tok {tk(T,'prompt_tokens'):.0f}->{tk(H,'prompt_tokens'):.0f}"
              f" | gen tok {tk(T,'tokens'):.0f}->{tk(H,'tokens'):.0f} | hint regex {hs(T):.1f}% -> {hs(H):.1f}% | cuts {sum(T[k].get('finish')=='length' for k in ks)}->{sum(H[k].get('finish')=='length' for k in ks)}")
    for b in sorted({rows[k[0]]["bench"] for k in keys}): line(b, [k for k in keys if rows[k[0]]["bench"] == b])
    line("POOLED", keys)
    for g, ix in GR.items(): line(f"  routed {g}", [k for k in keys if any(PY.get((tip[i]["cond"], k[0])) for i in ix)])
