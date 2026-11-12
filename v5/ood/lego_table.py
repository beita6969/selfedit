"""Lego layout pilot table: il - std for stock and for tip (paired by row, sample), tip - stock under each layout; picture-clustered se;
split single vs multi image and by number of images; prompt / generated tokens; cuts."""
import collections, json, math, os
V4 = "/data/fd/mingdazhang/selfedit/v4"
rows = {r["id"]: r for r in map(json.loads, open(f"{V4}/rows/learn.jsonl"))}
lst = lambda v: v if isinstance(v, list) else ([v] if v else [])
def load(b):
    o = {}
    for l in open(f"{V4}/q/out/{b}.jsonl"):
        x = json.loads(l)
        if not x.get("error") and x.get("content") is not None:
            k = x["key"].split("|"); o[(k[-2].replace("legoil/", ""), int(k[-1]))] = x
    return o
D = {a: load(f"lego_{a}") for a in ("std_stock", "il_stock", "std_tip", "il_tip")}
ok = lambda x: float(bool(x.get("ok")))
def cl(pairs):
    byr = collections.defaultdict(list)
    for (r, s), d in pairs: byr[r].append(d)
    if len(byr) < 2: return None
    v = {r: sum(x) / len(x) for r, x in byr.items()}; n = len(v); m = sum(v.values()) / n; byc = collections.defaultdict(float)
    for r, d in v.items(): byc[rows[r].get("pic") or r] += d - m
    G = len(byc); return m, math.sqrt(sum(x * x for x in byc.values()) * G / max(G - 1, 1)) / n, n
fm = lambda t: "n/a" if not t else f"{100*t[0]:+.1f}±{100*t[1]:.1f} (n={t[2]})"
print("reads", {a: len(v) for a, v in D.items()})
nim = lambda r: len(lst(rows[r]["image_paths"]))
def line(lab, f):
    ks = sorted(k for k in set.intersection(*[set(v) for v in D.values()]) if f(k))
    if not ks: return
    a = {n: 100 * sum(ok(D[n][k]) for k in ks) / len(ks) for n in D}
    d = lambda x, y: fm(cl([(k, ok(D[x][k]) - ok(D[y][k])) for k in ks]))
    t = lambda n, f_: sum(D[n][k].get(f_) or 0 for k in ks) / len(ks)
    print(f"{lab} [{len(ks)} reads]: acc std/il stock {a['std_stock']:.1f}/{a['il_stock']:.1f} tip {a['std_tip']:.1f}/{a['il_tip']:.1f} | il-std stock {d('il_stock','std_stock')}"
          f" | il-std tip {d('il_tip','std_tip')} | tip-stock std {d('std_tip','std_stock')} il {d('il_tip','il_stock')}"
          f" | prompt tok stock {t('std_stock','prompt_tokens'):.0f}/{t('il_stock','prompt_tokens'):.0f} | gen tok stock {t('std_stock','tokens'):.0f}/{t('il_stock','tokens'):.0f}"
          f" tip {t('std_tip','tokens'):.0f}/{t('il_tip','tokens'):.0f} | cuts {[sum(D[n][k].get('finish') == 'length' for k in ks) for n in D]}")
line("ALL", lambda k: True)
line("single-image", lambda k: nim(k[0]) == 1); line("multi-image", lambda k: nim(k[0]) > 1)
for n in sorted({nim(r) for r in rows if "lego" in rows[r]["bench"] and not r.startswith("legoil/")}): line(f"  {n} images", lambda k, n=n: nim(k[0]) == n)
for c in sorted({rows[r]["category"] for r in rows if rows[r]["bench"] == "lego" and not r.startswith("legoil/")}): line(f"  cat {c}", lambda k, c=c: rows[k[0]]["category"] == c)
