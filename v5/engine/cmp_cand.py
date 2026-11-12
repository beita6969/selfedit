"""python3 cmp_cand.py CELL_BATCHES CAND_BATCHES: paired cell vs candidate on same (row, sample): tokens, cuts, >16k, answers identical by candidate length bin."""
import json, statistics as st, sys
sys.path.insert(0, "/workspace/fudan/mingdazhang/selfedit-v4/loop")
import ecomp
V4 = "/data/fd/mingdazhang/selfedit/v4"
rows = {json.loads(l)["id"]: json.loads(l) for l in open(f"{V4}/rows/learn.jsonl")}
def load(bs):
    d = {}
    for b in bs.split(","):
        for l in open(f"{V4}/q/out/{b}.jsonl"):
            x = json.loads(l); k = x["key"].split("|")
            if not x.get("error") and x.get("content") is not None: d[(k[2], int(k[3]))] = x
    return d
X, C = load(sys.argv[1]), load(sys.argv[2]); ks = sorted(set(X) & set(C)); n = len(ks)
norm = lambda p: str(p).strip().lower()
tx, tc = sum(X[k]["tokens"] for k in ks), sum(C[k]["tokens"] for k in ks)
print(f"paired reads {n}: tokens cell {tx} vs candidate {tc} (x{tx/tc:.2f}); per-read ratio median x{st.median(X[k]['tokens']/C[k]['tokens'] for k in ks):.2f}")
print(f"acc cell {sum(bool(X[k].get('ok')) for k in ks)/n:.3f} candidate {sum(bool(C[k].get('ok')) for k in ks)/n:.3f}; "
      f"w->r {sum(bool(X[k].get('ok')) and not C[k].get('ok') for k in ks)} r->w {sum(bool(C[k].get('ok')) and not X[k].get('ok') for k in ks)}")
print(f"cuts cell {sum(X[k].get('finish')=='length' for k in ks)} candidate {sum(C[k].get('finish')=='length' for k in ks)}; "
      f">16k cell {sum(X[k]['tokens']>16000 for k in ks)} candidate {sum(C[k]['tokens']>16000 for k in ks)}; loops cell {sum(bool(X[k].get('loop')) for k in ks)}")
for lo, hi in ((0, 5000), (5000, 12000), (12000, 10**9)):
    b = [k for k in ks if lo <= C[k]["tokens"] < hi]
    if not b: continue
    same = sum(norm(ecomp.pred(X[k], rows[k[0]])) == norm(ecomp.pred(C[k], rows[k[0]])) for k in b)
    tok_same = sum(X[k]["tokens"] == C[k]["tokens"] for k in b)
    print(f"  candidate {lo}-{hi if hi < 10**9 else 'inf'} tok: {len(b)} reads; answers identical {same}/{len(b)}; token count identical {tok_same}; "
          f"tokens x{sum(X[k]['tokens'] for k in b)/sum(C[k]['tokens'] for k in b):.2f}; acc cell {sum(bool(X[k].get('ok')) for k in b)} cand {sum(bool(C[k].get('ok')) for k in b)}")
