#!/usr/bin/env python3
"""Per-benchmark held-out gains (REPORT ONLY). Same data and policies as heldout_table.py.
usage: python3 heldout_perbench.py"""
import gzip, itertools, json, random
from collections import defaultdict
import os
X = os.environ.get("SE_EXPORTS", "v5_exports").rstrip("/") + "/"      # folder with the compact held-out read files
FILES = [(X + "heldout_base_reads_compact.jsonl.gz", None, 0),
         (X + "heldout_vp2m512_reads_compact.jsonl.gz", "w1", 0), (X + "heldout_vp2m512_s34_venus_compact.jsonl.gz", "w1", 0),
         (X + "heldout_w3_s0_vesta_compact.jsonl.gz", "w3", 0), (X + "heldout_w3_s1_vesta_compact.jsonl.gz", "w3", 0),
         (X + "heldout_w3_s2_vesta_compact.jsonl.gz", "w3", 0), (X + "heldout_w3_s34_venus_compact.jsonl.gz", "w3", 0),
         (X + "heldout_w3_s3_vesta_compact.jsonl.gz", "w3", 2), (X + "heldout_w3_s4_vesta_compact.jsonl.gz", "w3", 2)]
reads = defaultdict(dict); meta = {}
for path, system, off in FILES:
    for l in gzip.open(path, "rt"):
        r = json.loads(l); s = system or r["system"]
        reads[(s, r["set"], r["row"])][int(r["sample"]) + off] = (str(r["pred"]), bool(r["ok"]))
        meta[(r["set"], r["row"])] = (r["bench"], r["pic"])

def one(v):
    return sum(x[1] for x in v) / len(v)

def pol3(v):
    acc = n = 0
    for a, b, c in itertools.permutations(v, 3):
        n += 1
        if a[0] == b[0]: acc += a[1]
        elif c[0] == a[0]: acc += a[1]
        elif c[0] == b[0]: acc += b[1]
        else: acc += a[1]
    return acc / n

def vals(setname, system, fn):
    return {row: fn(list(sm.values())) for (s, st, row), sm in reads.items() if s == system and st == setname}

def boot(setname, bench, a, b, nb=4000, seed=11):
    pics = defaultdict(list)
    for r in a:
        if meta[(setname, r)][0] == bench:
            pics[meta[(setname, r)][1]].append(a[r] - b[r])
    keys = list(pics); rng = random.Random(seed)
    point = sum(sum(x) for x in pics.values()) / sum(len(x) for x in pics.values())
    ds = []
    for _ in range(nb):
        s = n = 0
        for _k in keys:
            x = pics[keys[rng.randrange(len(keys))]]; s += sum(x); n += len(x)
        ds.append(s / n)
    ds.sort()
    return 100 * point, 100 * ds[int(0.025 * nb)], 100 * ds[int(0.975 * nb)]

for setname in ("id", "ood"):
    S1, S3 = vals(setname, "stock", one), vals(setname, "stock", pol3)
    A1, A3 = vals(setname, "w1", one), vals(setname, "w1", pol3)
    B1, B3 = vals(setname, "w3", one), vals(setname, "w3", pol3)
    F3, T3 = vals(setname, "floor", pol3), vals(setname, "tip", pol3)
    benches = sorted({meta[(setname, r)][0] for r in S1})
    print(f"\n===== {setname.upper()} (n = 128 rows per benchmark) =====")
    print("bench | stock 1 | W1 1 | W3 1 | W3 vote | W3 1 - stock 1 [95%] | W3 vote - stock 1 [95%] | W3 vote - stock vote [95%] | edit1: W1-stock (1 read) | edit2: W3-W1 (1 read) [95%] | W3 vote - harness tip vote [95%]")
    def m(v, b): 
        x = [v[r] for r in v if meta[(setname, r)][0] == b]; return 100 * sum(x) / len(x)
    def f(d): return f"{d[0]:+5.1f} [{d[1]:+5.1f},{d[2]:+5.1f}]"
    for b in benches:
        print(f"{b:16s} | {m(S1,b):5.1f} | {m(A1,b):5.1f} | {m(B1,b):5.1f} | {m(B3,b):5.1f} | {f(boot(setname,b,B1,S1))} | {f(boot(setname,b,B3,S1))} | {f(boot(setname,b,B3,S3))} | {f(boot(setname,b,A1,S1))} | {f(boot(setname,b,B1,A1))} | {f(boot(setname,b,B3,T3))}")


def macro_boot(setname, a, b, drop, nb=4000, seed=5):
    """macro over benchmarks not in drop of (a - b), pictures resampled within benchmark"""
    cl = defaultdict(lambda: defaultdict(list))
    for r in a:
        be, pic = meta[(setname, r)]
        if be in drop or r not in b:
            continue
        cl[be][pic].append(a[r] - b[r])
    rng = random.Random(seed)
    point = sum(sum(sum(x) for x in p.values()) / sum(len(x) for x in p.values()) for p in cl.values()) / len(cl)
    ds = []
    for _ in range(nb):
        m_ = 0.0
        for be, pics in cl.items():
            keys = list(pics); s_ = n_ = 0
            for _k in keys:
                x = pics[keys[rng.randrange(len(keys))]]; s_ += sum(x); n_ += len(x)
            m_ += s_ / n_
        ds.append(m_ / len(cl))
    ds.sort()
    return 100 * point, 100 * ds[int(0.025 * nb)], 100 * ds[int(0.975 * nb)]


def macro_level(setname, v, drop):
    by = defaultdict(list)
    for r, x in v.items():
        be = meta[(setname, r)][0]
        if be not in drop:
            by[be].append(x)
    return 100 * sum(sum(x) / len(x) for x in by.values()) / len(by)


import os
if os.environ.get("DROP"):
    print("\n######## macro with benchmarks removed (post-hoc; REPORT ONLY)")
    for setname, drops in (("id", [set(), {"lego"}]), ("ood", [set(), {"erqa"}])):
        S1, S3 = vals(setname, "stock", one), vals(setname, "stock", pol3)
        A1, A3 = vals(setname, "w1", one), vals(setname, "w1", pol3)
        B1, B3 = vals(setname, "w3", one), vals(setname, "w3", pol3)
        F1, F3 = vals(setname, "floor", one), vals(setname, "floor", pol3)
        T1, T3 = vals(setname, "tip", one), vals(setname, "tip", pol3)
        for drop in drops:
            f = lambda d: f"{d[0]:+5.2f} [{d[1]:+5.1f},{d[2]:+5.1f}]"
            print(f"\n== {setname.upper()} drop {sorted(drop) or 'nothing'} ({6 - len(drop)} benchmarks)")
            print(f"  levels: stock 1 read {macro_level(setname,S1,drop):.2f}, stock vote {macro_level(setname,S3,drop):.2f}, W1 1 read {macro_level(setname,A1,drop):.2f}, W1 vote {macro_level(setname,A3,drop):.2f}, W3 1 read {macro_level(setname,B1,drop):.2f}, W3 vote {macro_level(setname,B3,drop):.2f}, floor vote {macro_level(setname,F3,drop):.2f}, tip 1 read {macro_level(setname,T1,drop):.2f}, tip vote {macro_level(setname,T3,drop):.2f}")
            print(f"  W3 1 read - stock 1 read   {f(macro_boot(setname,B1,S1,drop))}")
            print(f"  W3 vote - stock 1 read     {f(macro_boot(setname,B3,S1,drop))}")
            print(f"  W3 vote - stock vote       {f(macro_boot(setname,B3,S3,drop))}")
            print(f"  W1 1 read - stock 1 read   {f(macro_boot(setname,A1,S1,drop))}")
            print(f"  W1 vote - stock 1 read     {f(macro_boot(setname,A3,S1,drop))}")
            print(f"  W3 - W1, 1 read            {f(macro_boot(setname,B1,A1,drop))}")
            print(f"  tip vote - stock 1 read    {f(macro_boot(setname,T3,S1,drop))}")
            print(f"  W3 vote - tip vote         {f(macro_boot(setname,B3,T3,drop))}")
            print(f"  W3 1 read - tip 1 read     {f(macro_boot(setname,B1,T1,drop))}")
