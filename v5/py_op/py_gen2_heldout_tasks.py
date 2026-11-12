"""PY generation 2, step 3 (only if PYA is accepted; REPORTING ONLY): the deployed escalation reads on held-out rows. Per row and system
(floor = general base, tip = full system) the fixed replicate pairs (s0,s1), (s2,s3) of the cached reads; a pair whose predicted answers
differ gets ONE PYA read carrying that pair's two answers (seeded order). Reads of the j-th disagreeing replicate use sample index j.
usage: py_gen2_heldout_tasks.py EXPORT.jsonl.gz OUT_PREFIX"""
import collections, gzip, json, os, random, sys, zlib
V5 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "v5")
sys.path.insert(0, V5)
import policy5 as P   # noqa: E402


def show(p):
    return str(int(p)) if isinstance(p, float) and p == int(p) else str(p)


by = collections.defaultdict(dict)
for l in gzip.open(sys.argv[1], "rt"):
    x = json.loads(l)
    if x["system"] in ("floor", "tip"):
        by[(x["system"], x["row"])][int(x["sample"])] = x
out = {}
for s in ("floor", "tip"):
    T, nohint = [], 0
    for (ss, r), d in sorted(by.items()):
        if ss != s:
            continue
        j = 0
        for k in range(2):
            a, b = d[2 * k], d[2 * k + 1]
            if P.cls_of(dict(pred=a.get("pred"))) == P.cls_of(dict(pred=b.get("pred"))):
                continue
            hint = None
            if a.get("pred") is not None and b.get("pred") is not None and show(a["pred"]) != show(b["pred"]):
                hint = [show(a["pred"]), show(b["pred"])]
                random.Random(zlib.crc32(f"pya|{s}|{r}|{k}".encode())).shuffle(hint)
            else:
                nohint += 1
            T.append(dict(row=r, sample=j, rep=k, hint=hint, system=s, set=a["set"]))
            j += 1
    json.dump(T, open(f"{sys.argv[2]}_{s}.json", "w"))
    print(s, "escalation reads", len(T), "rows", len({t["row"] for t in T}), "without hint", nohint, dict(collections.Counter(t["set"] for t in T)))
