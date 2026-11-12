"""mk_rows_h (fable_int, hindsight): reads for the hindsight contrast, signal rows only. Per benchmark up to NW rows with a read that ended WRONG (one wrong read each) and up to NR other rows
with a read that ended RIGHT (one each); reads of 150-3000 thinking tokens, prompt <= 6500 tokens, finish stop; seeded. Files: OUT_all_items.txt ("row sample"), OUT_cut_items.txt
(per benchmark 2 wrong + 1 right, for patching), OUT_meta.json ({"row sample": [bench, ok, held]}; held = crc32(row) % 5 == 0, the row-held-out evaluation rows).
python mk_rows_h.py TRACES OUT [NW NR]"""
import json, os, sys, zlib, collections
import numpy as np
D = json.load(open(sys.argv[1])); out = sys.argv[2]; NW = int(sys.argv[3]) if len(sys.argv) > 3 else 16; NR = int(sys.argv[4]) if len(sys.argv) > 4 else 8
rng = np.random.default_rng(2222); items, cut, meta = [], [], {}
for b in sorted({v["bench"] for v in D.values()}):
    rows = sorted(r for r, v in D.items() if v["bench"] == b and v.get("role") in ("show", "monitor"))
    ok_ = lambda s: 150 <= int(s["tokens"]) <= 3000 and int(s.get("prompt_tokens") or 0) <= 6500 and s.get("reasoning") and s.get("finish") == "stop"
    W = {r: sorted(int(k) for k, s in D[r]["samples"].items() if ok_(s) and not s["ok"]) for r in rows}; R = {r: sorted(int(k) for k, s in D[r]["samples"].items() if ok_(s) and s["ok"]) for r in rows}
    wr = [r for r in rows if W[r]]; wr = [wr[i] for i in rng.permutation(len(wr))][:NW]
    rr = [r for r in rows if R[r] and r not in set(wr)]; rr = [rr[i] for i in rng.permutation(len(rr))][:NR]
    wi = [(r, W[r][zlib.crc32(r.encode()) % len(W[r])]) for r in wr]; ri = [(r, R[r][zlib.crc32(r.encode()) % len(R[r])]) for r in rr]
    items += wi + ri; cut += wi[:2] + ri[:1]
    for (r, k), okv in [(x, False) for x in wi] + [(x, True) for x in ri]:
        meta[f"{r} {k}"] = [b, okv, zlib.crc32(r.encode()) % 5 == 0]
    print(b, "rows", len(rows), "| rows with a wrong read in the window", sum(bool(W[r]) for r in rows), "-> wrong reads", len(wi), "| right reads", len(ri))
items = [items[i] for i in rng.permutation(len(items))]            # shuffled, so that two shards hold the same mix
for name, L in (("all", items), ("cut", cut)):
    fn = f"{out}_{name}_items.txt"; assert not os.path.exists(fn); open(fn, "w").write("".join(f"{r} {k}\n" for r, k in L))
json.dump(meta, open(f"{out}_meta.json", "w"))
ev = {"rows": [k for k, v in meta.items() if v[2]]}                # evaluation lists per fold: the row-held-out reads, and per benchmark up to 12 wrong + 6 right reads
for b in sorted({v[0] for v in meta.values()}):
    ev[b] = [k for k, v in meta.items() if v[0] == b and not v[1]][:12] + [k for k, v in meta.items() if v[0] == b and v[1]][:6]
for f, L in ev.items():
    fn = f"{out}_eval_{f}_items.txt"; assert not os.path.exists(fn); open(fn, "w").write("".join(k + "\n" for k in L))
print("evaluation lists:", {f: len(L) for f, L in ev.items()})
print("reads", len(items), "wrong", sum(not m[1] for m in meta.values()), "right", sum(m[1] for m in meta.values()), "held-out rows", sum(m[2] for m in meta.values()), "| cut set", len(cut))
