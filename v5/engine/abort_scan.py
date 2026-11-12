"""List result lines that are aborted reads (finish == "abort", or no content, no reasoning and 0 tokens without an error), per batch,
and whether a later non-aborted line for the same key exists anywhere in q/out. usage: python3 abort_scan.py [--keys BATCH]"""
import glob, json, os, sys
V4 = "/data/fd/mingdazhang/selfedit/v4"


def aborted(r):
    if r.get("error") or r.get("op_skipped") or r.get("mode") == "chat":
        return False
    return r.get("finish") == "abort" or (not r.get("content") and not r.get("reasoning") and not (r.get("tokens") or 0))


bad, good = {}, set()
for f in sorted(glob.glob(f"{V4}/q/out/*.jsonl")):
    b = os.path.basename(f)[:-6]
    for l in open(f):
        try:
            r = json.loads(l)
        except ValueError:
            continue
        if "key" not in r:
            continue
        if aborted(r):
            bad.setdefault(b, []).append(r["key"])
        elif not r.get("error"):
            good.add(r["key"])
if len(sys.argv) > 2 and sys.argv[1] == "--keys":
    for k in bad.get(sys.argv[2], []):
        if k not in good:
            print(k)
    sys.exit()
tot = 0
for b, ks in sorted(bad.items()):
    open_ = [k for k in ks if k not in good]
    tot += len(ks)
    print(f"{b}: {len(ks)} aborted lines, {len(open_)} keys without a valid result")
print("total", tot)
