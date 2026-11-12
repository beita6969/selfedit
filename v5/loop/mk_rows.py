"""v4 rows (A1): learn rows of iid_paper128_v3 -> V4/rows/learn.jsonl with v4_role.
SHOW = show kept by exclude_96 (30/bench); MONITOR = select kept (24/bench); JUDGE = score kept (42) + every row exclude_96 removed (32) = 74/bench.
ID rows (hb_role id) are never written here."""
import collections, hashlib, json, os, sys
SRC = "/data/fd/mingdazhang/selfedit/rsi/mm/hb/iid_paper128_v3.jsonl"
EXC = "/data/fd/mingdazhang/selfedit/rsi/mm/hb/p1109c_final_c/exclude_96.json"
OUT = "/data/fd/mingdazhang/selfedit/v4/rows/learn.jsonl"
src = open(SRC, "rb").read()
assert hashlib.sha256(src).hexdigest().startswith("ff0d465b"), "pool v3 sha"
R = [json.loads(l) for l in src.decode().splitlines() if l.strip()]
E = json.load(open(EXC))
ex = set(E["rows"] if isinstance(E["rows"], list) else E["rows"].keys())
out, c = [], collections.Counter()
for r in R:
    h = r.get("hb_role")
    if h == "id":
        continue
    assert h in ("show", "score", "select"), (r["id"], h)
    role = "judge" if (r["id"] in ex or h == "score") else {"show": "show", "select": "monitor"}[h]
    out.append(dict(r, v4_role=role))
    c[(r["bench"], role)] += 1
for k in sorted(c):
    print(k, c[k])
benches = sorted({b for b, _ in c})
assert all(c[(b, "show")] == 30 and c[(b, "judge")] == 74 and c[(b, "monitor")] == 24 for b in benches), "role counts"
assert len(out) == 768 and len({r["id"] for r in out}) == 768
ids_id = {r["id"] for r in R if r.get("hb_role") == "id"}
assert not ids_id & {r["id"] for r in out}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
assert not os.path.exists(OUT), "learn.jsonl exists: never overwritten"
with open(OUT + ".tmp", "w") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
os.replace(OUT + ".tmp", OUT)
print("wrote", OUT, len(out), hashlib.sha256(open(OUT, "rb").read()).hexdigest()[:16])
