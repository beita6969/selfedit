"""ana_h (fable_int, hindsight): restoration R of the hindsight distribution by compiled cells, separately for reads that ended wrong and right (b_fn2 records).
Per cell: R per read = 1 - mean KL(H || cell) / mean KL(H || base), mean +- se over reads; pooled R over all positions; gamma (dose along the hindsight shift). python ana_h.py DIR [DIR ...]"""
import glob, os, sys
import numpy as np
for d in sys.argv[1:]:
    R = [np.load(f, allow_pickle=True) for f in sorted(glob.glob(os.path.join(d, "*.npz")))]
    if not R:
        continue
    names = [n for n in R[0]["names"].tolist() if n not in ("m", "x")]
    print(f"\n{d}: {len(R)} reads ({sum(not bool(r['ok']) for r in R)} ended wrong, {sum(bool(r['ok']) for r in R)} right); KL(H || base) per position: wrong reads {np.mean([r['m/kl_xc'].mean() for r in R if not bool(r['ok'])] or [np.nan]):.4f}, right reads {np.mean([r['m/kl_xc'].mean() for r in R if bool(r['ok'])] or [np.nan]):.4f}")
    print("cell | wrong reads: R mean +- se, pooled R, gamma | right reads: R mean +- se, pooled R, gamma | KL(base || cell) wrong / right")
    for n in names:
        out = []
        for okv in (False, True):
            S = [r for r in R if bool(r["ok"]) == okv]
            if not S:
                out.append("n/a"); continue
            r_ = np.array([1 - r[f"{n}/kl_xc"].mean() / max(1e-9, r["m/kl_xc"].mean()) for r in S]); pooled = 1 - sum(r[f"{n}/kl_xc"].sum() for r in S) / sum(r["m/kl_xc"].sum() for r in S)
            g = sum(r[f"{n}/ip_cx"].sum() for r in S) / max(1e-12, sum(r["x/ip_cx"].sum() for r in S))
            out.append(f"{r_.mean():+.3f} +- {r_.std(ddof=1) / np.sqrt(len(S)) if len(S) > 1 else float('nan'):.3f}, {pooled:+.3f}, {g:+.3f}")
        kw = np.mean([r[f"{n}/kl_mc"].mean() for r in R if not bool(r["ok"])] or [np.nan]); kr = np.mean([r[f"{n}/kl_mc"].mean() for r in R if bool(r["ok"])] or [np.nan])
        print(f"{n} | {out[0]} | {out[1]} | {kw:.4f} / {kr:.4f}")
