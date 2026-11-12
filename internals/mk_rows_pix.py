"""mk_rows_pix (fable_int, C_pix): one cached read per single-image signal row (300-3,000 thinking tokens, finished), at most N per benchmark, order by crc32 of the row id.
python mk_rows_pix.py TRACES.json OUT_items.txt [N=10]"""
import json, sys, zlib
D = json.load(open(sys.argv[1])); N = int(sys.argv[3]) if len(sys.argv) > 3 else 10
by = {}
for r in sorted(D):
    v = D[r]
    if not isinstance(v, dict) or v.get("role") not in ("show", "monitor") or len(v.get("image_paths") or []) != 1:
        continue
    ks = sorted(int(k) for k, s in v["samples"].items() if 300 <= int(s["tokens"]) <= 3000 and s.get("reasoning") and s.get("finish") == "stop")
    if ks:
        by.setdefault(v["bench"], []).append((zlib.crc32(r.encode()), r, ks[zlib.crc32(r.encode()) % len(ks)]))
out = []
for b in sorted(by):
    L = sorted(by[b])[:N]; out += [(r, k) for _, r, k in L]
    print(b, len(by[b]), "rows with a read in the window; taken", len(L), "| ended wrong", sum(not D[r]["samples"][str(k)]["ok"] for _, r, k in L))
open(sys.argv[2], "x").write("".join(f"{r} {k}\n" for r, k in out)); print(len(out), "reads ->", sys.argv[2])
