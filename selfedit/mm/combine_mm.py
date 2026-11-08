"""Merge gen_mm.py outputs (e.g. two replicas, or a re-run of one set) into one file for grade_mm.py.
Later inputs win for the same (set, id, sample). --drop-sets removes superseded sets from the inputs listed BEFORE
--replace-from; --drop-error-items removes every sample of an item that has any finish == "error" record (e.g. vLLM
rejected the prompt), and lists them.
Usage: python combine_mm.py --out MERGED.jsonl IN1.jsonl [IN2.jsonl ...] [--drop-sets sets/mathvista,...] [--drop-error-items]
"""
import argparse, json

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("inputs", nargs="+")
ap.add_argument("--drop-sets", default="")        # applied to every input except the last one
ap.add_argument("--drop-error-items", action="store_true")
args = ap.parse_args()
drop = {s for s in args.drop_sets.split(",") if s}
recs = {}
for k, f in enumerate(args.inputs):
    for line in open(f):
        r = json.loads(line)
        if k < len(args.inputs) - 1 and r.get("set") in drop:
            continue
        recs[(r.get("set"), r["id"], r["sample"])] = r
bad = {(s, i) for (s, i, _), r in recs.items() if r["finish"] == "error"} if args.drop_error_items else set()
for s, i in sorted(bad):
    err = next(r.get("error", "") for (s2, i2, _), r in recs.items() if (s2, i2) == (s, i) and r["finish"] == "error")
    print("dropped error item", s, i, err[:160])
with open(args.out, "w") as f:
    for (s, i, _), r in recs.items():
        if (s, i) not in bad:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"wrote {sum((s, i) not in bad for (s, i, _) in recs)} records to {args.out}")
