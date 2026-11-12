"""Append the 768 OOD v2 rows (data/ood128/ood128_v2.jsonl) to v4/rows/learn.jsonl with v4_role/hb_role "ood_v2".
Backup first (learn.jsonl.pre_ood_1111, cp -p semantics), checks: learn has 768 rows + sha b1263f83..., no id overlap, images exist; atomic replace."""
import hashlib, json, os, shutil, collections
V4 = "/data/fd/mingdazhang/selfedit/v4"; F = f"{V4}/rows/learn.jsonl"; SRC = "/data/fd/mingdazhang/selfedit/data/ood128/ood128_v2.jsonl"
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
s0 = sha(F); assert s0.startswith("b1263f834ed1d2db"), s0
L = open(F).read().splitlines(); assert len(L) == 768
B = f"{V4}/rows/learn.jsonl.pre_ood_1111"; assert not os.path.exists(B); shutil.copy2(F, B); assert sha(B) == s0
ids = {json.loads(l)["id"] for l in L}
O = [json.loads(l) for l in open(SRC)]; assert len(O) == 768 and not ({r["id"] for r in O} & ids)
for r in O:
    for p in (r["image_paths"] if isinstance(r["image_paths"], list) else [r["image_paths"]]): assert os.path.exists(p), p
    r["v4_role"] = r["hb_role"] = "ood_v2"
src_sha = sha(SRC)
with open(F + ".tmp", "w") as f:
    for l in L: f.write(l + "\n")
    for r in O: f.write(json.dumps(r) + "\n")
os.replace(F + ".tmp", F)
N = open(F).read().splitlines(); assert len(N) == 1536 and N[:768] == L
print(json.dumps(dict(src=SRC, src_sha=src_sha, before=s0, after=sha(F), rows=len(N), backup=B, per_bench=collections.Counter(r["bench"] for r in O))))
