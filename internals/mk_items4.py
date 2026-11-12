"""mk_items4 (fable_int): up to N cached reads per row (finished, with reasoning, LO <= tokens), for the rows named in the first column of a list; rows without reads are reported and skipped.
python mk_items4.py TRACES ROWS_OR_ITEMS OUT N [LO=150]"""
import json, sys
D = json.load(open(sys.argv[1])); n = int(sys.argv[4]); lo = int(sys.argv[5]) if len(sys.argv) > 5 else 150; out = []; miss = []
ids = []
for l in open(sys.argv[2]):
    if l.strip() and l.split()[0] not in ids:
        ids.append(l.split()[0])
for r in ids:
    if r not in D:
        miss.append(r); continue
    ks = sorted(int(k) for k, s in D[r]["samples"].items() if s.get("finish") == "stop" and s.get("reasoning") and int(s["tokens"]) >= lo)
    out += [(r, k) for k in ks[:n]]
open(sys.argv[3], "x").write("".join(f"{r} {k}\n" for r, k in out)); print(len(out), "reads on", len({r for r, _ in out}), "rows; rows without reads:", miss)
