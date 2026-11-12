"""mk_items3 (fable_int): up to N cached reads per row for the row lists of mk_proc_rows2 (rows stay the unit of holding out). python mk_items3.py TRACES IDS OUT N [PROMPT_HI]"""
import json, sys
D = json.load(open(sys.argv[1])); n = int(sys.argv[4]); hi = int(sys.argv[5]) if len(sys.argv) > 5 else 9000; out = []
for r in [l.strip() for l in open(sys.argv[2]) if l.strip()]:
    ks = sorted(int(k) for k, s in D[r]["samples"].items() if s.get("finish") == "stop" and s.get("reasoning") and int(s["tokens"]) >= 150 and int(s.get("prompt_tokens") or 0) <= hi)
    out += [(r, k) for k in ks[:n]]
open(sys.argv[3], "w").write("".join(f"{r} {k}\n" for r, k in out)); print(len(out), "reads on", len({r for r, _ in out}), "rows")
