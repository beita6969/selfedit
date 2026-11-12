"""mk_unr_bench (fable_int): unrouted signal rows by benchmark for the leak-by-benchmark table: up to N rows per benchmark that were used neither in the route-key fit nor as its held-out
rows, one cached t005r read each (300-6,000 generated tokens, finished), order by crc32. python mk_unr_bench.py TRACES OUT N EXCLUDE_ITEMS [EXCLUDE_ITEMS ...]"""
import json, sys, zlib
D = json.load(open(sys.argv[1])); N = int(sys.argv[3]); ex = {l.split()[0] for f in sys.argv[4:] for l in open(f) if l.strip()}
by = {}
for r in sorted(D):
    v = D[r]
    if not isinstance(v, dict) or v.get("role") not in ("show", "monitor") or v.get("prefix") or r in ex:
        continue
    ks = sorted(int(k) for k, s in v["samples"].items() if 300 <= int(s["tokens"]) <= 6000 and s.get("reasoning") and s.get("finish") == "stop" and int(s.get("prompt_tokens") or 0) <= 9000)
    if ks:
        by.setdefault(v["bench"], []).append((zlib.crc32(r.encode()), r, ks[zlib.crc32(r.encode()) % len(ks)]))
out = []
for b in sorted(by):
    L = sorted(by[b])[:N]; out += [(r, k) for _, r, k in L]; print(b, len(by[b]), "unrouted rows available; taken", len(L))
open(sys.argv[2], "x").write("".join(f"{r} {k}\n" for r, k in out)); print(len(out), "reads ->", sys.argv[2])
