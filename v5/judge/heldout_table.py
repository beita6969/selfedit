#!/usr/bin/env python3
"""Held-out report table (REPORT ONLY; nothing is selected on these rows).
Systems: stock (8 samples), floor / tip (5), W1 = vp2m512 (5), W3 = vp2m512 + gated counting cell (as many samples as exist).
Per set (id / ood): one read; "2 reads + a third on disagreement" (exact expectation over ordered sample triples);
tokens as multiples of one stock read (generated / total with prompt); picture-clustered bootstrap intervals.
usage: python3 heldout_table.py [W3 compact files ...]"""
import gzip, itertools, json, random, sys
from collections import defaultdict

import os
X = os.environ.get("SE_EXPORTS", "v5_exports").rstrip("/") + "/"      # folder with the compact held-out read files
BASE = X + "heldout_base_reads_compact.jsonl.gz"
W1 = [X + "heldout_vp2m512_reads_compact.jsonl.gz", X + "heldout_vp2m512_s34_venus_compact.jsonl.gz"]
W3 = sys.argv[1:]

reads = defaultdict(dict)   # (system, set, row) -> {sample: (pred, ok, gen, prompt)}
meta = {}                   # (set, row) -> (bench, pic)


def load(path, system=None):
    # "file::+K" shifts the sample index by K (two machines may both call a sample "3")
    off = 0
    if "::" in path:
        path, o = path.split("::"); off = int(o)
    n = 0
    for l in gzip.open(path, "rt"):
        r = json.loads(l)
        s = system or r["system"]
        k = int(r["sample"]) + off
        assert k not in reads[(s, r["set"], r["row"])], ("duplicate sample index", path, k)
        reads[(s, r["set"], r["row"])][k] = (str(r["pred"]), bool(r["ok"]), int(r["tokens"]), int(r["prompt_tokens"]))
        meta[(r["set"], r["row"])] = (r["bench"], r["pic"])
        n += 1
    return n


load(BASE)
for p in W1:
    load(p, "w1")
for p in W3:
    print("W3 file", p.split("/")[-1], load(p, "w3"), "reads")


def one(samples):
    v = list(samples.values())
    k = len(v)
    return (sum(x[1] for x in v) / k, sum(x[2] for x in v) / k, sum(x[2] + x[3] for x in v) / k)


def pol3(samples):
    """2 reads; on disagreement a third read; majority; if all three differ the first read stands.
    Exact mean over ordered triples of distinct samples."""
    v = list(samples.values())
    if len(v) < 3:
        return None
    acc = gen = tot = 0.0
    n = 0
    for a, b, c in itertools.permutations(v, 3):
        n += 1
        if a[0] == b[0]:
            acc += a[1]; gen += a[2] + b[2]; tot += a[2] + a[3] + b[2] + b[3]
        else:
            gen += a[2] + b[2] + c[2]; tot += a[2] + a[3] + b[2] + b[3] + c[2] + c[3]
            if c[0] == a[0]:
                acc += a[1]
            elif c[0] == b[0]:
                acc += b[1]
            else:
                acc += a[1]
    return (acc / n, gen / n, tot / n)


def table(setname, system, fn, restrict=None):
    """per-row values {row: (acc, gen, tot)}"""
    out = {}
    for (s, st, row), sm in reads.items():
        if s != system or st != setname:
            continue
        if restrict is not None:
            sm = {k: v for k, v in sm.items() if k in restrict}
        if not sm:
            continue
        r = fn(sm)
        if r is not None:
            out[row] = r
    return out


def macro(setname, vals, idx=0, rows=None):
    by = defaultdict(list)
    for row, v in vals.items():
        if rows is not None and row not in rows:
            continue
        by[meta[(setname, row)][0]].append(v[idx])
    per = {b: sum(x) / len(x) for b, x in by.items()}
    return sum(per.values()) / len(per), per


def boot(setname, a, b, nb=2000, seed=7):
    """macro(a) - macro(b), paired by row, pictures resampled within benchmark."""
    rows = [r for r in a if r in b]
    cl = defaultdict(lambda: defaultdict(list))
    for r in rows:
        be, pic = meta[(setname, r)]
        cl[be][pic].append(a[r][0] - b[r][0])
    rng = random.Random(seed)
    point = sum(sum(sum(x) for x in pics.values()) / sum(len(x) for x in pics.values()) for pics in cl.values()) / len(cl)
    ds = []
    for _ in range(nb):
        m = 0.0
        for be, pics in cl.items():
            keys = list(pics)
            s = n = 0
            for _k in keys:
                x = pics[keys[rng.randrange(len(keys))]]
                s += sum(x); n += len(x)
            m += s / n
        ds.append(m / len(cl))
    ds.sort()
    return 100 * point, 100 * ds[int(0.025 * nb)], 100 * ds[int(0.975 * nb)]


def tokmult(setname, vals, ref):
    rows = [r for r in vals if r in ref]
    g = sum(vals[r][1] for r in rows) / sum(ref[r][1] for r in rows)
    t = sum(vals[r][2] for r in rows) / sum(ref[r][2] for r in rows)
    return g, t


OUT = {}
for setname in ("id", "ood"):
    print(f"\n===== {setname.upper()} =====")
    ref = table(setname, "stock", one)
    nW3 = sorted({k for (s, st, row), sm in reads.items() if s == "w3" and st == setname for k in sm})
    print("W3 samples present:", nW3, " rows:", len(table(setname, "w3", one)))
    systems = [("stock", "stock"), ("floor", "floor (request-level, harness)"), ("tip", "tip t007c (floor + 11 texts, harness)"), ("w1", "W1 = vp2m512 (weights, stock requests)"), ("w3", "W3 = W1 + gated counting cell (weights, stock requests)")]
    benches = sorted({meta[(setname, r)][0] for r in ref})
    print("system / policy".ljust(64), "macro   d vs stock 1 read [95%]     gen x  tot x   " + "  ".join(b[:6] for b in benches))
    keep = {}
    for sysname, label in systems:
        for pname, fn in (("1 read", one), ("2 reads + third", pol3)):
            vals = table(setname, sysname, fn)
            if len(vals) < 700:
                continue
            m, per = macro(setname, vals)
            d = boot(setname, vals, ref)
            g, t = tokmult(setname, vals, ref)
            keep[(sysname, pname)] = vals
            OUT[(setname, sysname, pname)] = (100 * m, d, g, t, {b_: 100 * per[b_] for b_ in benches})
            print(f"{(label + ' / ' + pname)[:64].ljust(64)} {100*m:6.2f}  {d[0]:+5.2f} [{d[1]:+5.1f},{d[2]:+5.1f}]   x{g:4.2f}  x{t:4.2f}   " + "  ".join(f"{100*per[b]:6.1f}" for b in benches))
    # paired contrasts
    def con(a, b, name):
        if a in keep and b in keep:
            d = boot(setname, keep[a], keep[b])
            OUT[(setname, "con", name)] = d
            print(f"  {name}: {d[0]:+5.2f} [{d[1]:+5.1f},{d[2]:+5.1f}]")
    con(("w1", "2 reads + third"), ("stock", "2 reads + third"), "W1 vs stock, same policy (2 + third)")
    con(("w3", "1 read"), ("w1", "1 read"), "W3 - W1, one read (all W1 samples)")
    con(("w3", "2 reads + third"), ("w1", "2 reads + third"), "W3 - W1, 2 + third")
    con(("w3", "2 reads + third"), ("stock", "2 reads + third"), "W3 vs stock, same policy (2 + third)")
    con(("w3", "1 read"), ("floor", "1 read"), "W3 - floor, one read")
    con(("w3", "1 read"), ("tip", "1 read"), "W3 - tip (harness: floor + 11 texts), one read")
    con(("w3", "2 reads + third"), ("tip", "2 reads + third"), "W3 - tip, 2 + third")
    # W1 restricted to W3's sample indices, both policies (like-for-like when W3 has fewer samples than W1)
    nW1 = sorted({k for (s_, st, row), sm in reads.items() if s_ == "w1" and st == setname for k in sm})
    nW3all = nW3
    nW3 = [k for k in nW3 if k in nW1]      # indices both systems have (shifted extra W3 samples are left out here)
    if nW3all != nW3:
        print("  (W3 extra samples without a W1 counterpart:", [k for k in nW3all if k not in nW1], ")")
    if nW3:
        for pname, fn in (("1 read", one), ("2 reads + third", pol3)):
            a = table(setname, "w3", fn, set(nW3)); b = table(setname, "w1", fn, set(nW3))
            if len(a) < 700 or len(b) < 700:
                continue
            ma, _ = macro(setname, a); mb, _ = macro(setname, b)
            d = boot(setname, a, b); db = boot(setname, b, ref)
            print(f"  same samples {nW3}, {pname}: W3 {100*ma:.2f}  W1 {100*mb:.2f} (W1 vs stock 1 read {db[0]:+.2f})  W3 - W1 {d[0]:+5.2f} [{d[1]:+5.1f},{d[2]:+5.1f}]")
    # same-sample contrast W3 - W1 (sample indices in common)
    if nW3:
        a = table(setname, "w3", one, set(nW3)); b = table(setname, "w1", one, set(nW3))
        d = boot(setname, a, b)
        print(f"  W3 - W1 on the same sample indices {nW3}: {d[0]:+5.2f} [{d[1]:+5.1f},{d[2]:+5.1f}]")
        for s in nW3:
            a = table(setname, "w3", one, {s}); b = table(setname, "w1", one, {s})
            ma, pa = macro(setname, a); mb, pb = macro(setname, b)
            print(f"    sample {s}: W3 {100*ma:.2f}  W1 {100*mb:.2f}   " + "  ".join(f"{bn[:6]} {100*(pa[bn]-pb[bn]):+5.1f}" for bn in benches))


import os
if os.environ.get("MD"):
    def cell(setname, sysname, pname):
        m, d, g, t, per = OUT[(setname, sysname, pname)]
        return m, d, g, t, per
    print("\n--- markdown ---")
    for sysname, lab in (("stock", "原始模型"), ("w1", "W1"), ("w3", "W3")):
        for pname, pl in (("1 read", "读 1 次"), ("2 reads + third", "读 2 次 + 不一致读第 3 次")):
            if ("id", sysname, pname) not in OUT:
                continue
            i = cell("id", sysname, pname); o = cell("ood", sysname, pname)
            def f(c):
                return f"{c[0]:.2f}" if sysname == "stock" and pname == "1 read" else f"{c[0]:.2f}（{c[1][0]:+.2f} [{c[1][1]:+.1f}, {c[1][2]:+.1f}]）"
            print(f"| {lab}，{pl} | {f(i)} | {f(o)} | ×{i[2]:.2f} / {o[2]:.2f} | ×{i[3]:.2f} / {o[3]:.2f} |")
    for setname in ("id", "ood"):
        benches = sorted(OUT[(setname, "stock", "1 read")][4])
        print("| " + setname.upper() + " | " + " | ".join(benches) + " | 宏平均 |")
        for sysname, lab in (("stock", "原始模型"), ("w1", "W1"), ("w3", "W3")):
            for pname, pl in (("1 read", "读 1 次"), ("2 reads + third", "读 2 次 + 第 3 次")):
                if (setname, sysname, pname) not in OUT or (sysname == "stock" and pname != "1 read"):
                    continue
                c = OUT[(setname, sysname, pname)]
                print(f"| {lab}，{pl} | " + " | ".join(f"{c[4][b_]:.1f}" for b_ in benches) + f" | {c[0]:.1f} |")
