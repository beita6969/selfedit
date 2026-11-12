"""ana_leak (fable_int): leakage of compiled cells on rows where the procedure is not routed (b_xb records with mode proc on unrouted rows: reference x = medium).
Per spec: mean over traces of the per-position KL(medium || cell), of |log p(read token) cell - medium|, and the same on routed rows for scale (second folder, optional).
python ana_leak.py EV_UNROUTED [EV_ROUTED]"""
import glob, os, sys
import numpy as np
def load(d): return [np.load(f, allow_pickle=True) for f in sorted(glob.glob(os.path.join(d, "*.npz")))]
U = load(sys.argv[1]); R = load(sys.argv[2]) if len(sys.argv) > 2 else []
names = [n for n in U[0]["names"].tolist() if n not in ("m", "x")]
print(f"unrouted traces {len(U)}; routed traces {len(R)}")
if R:
    print(f"routed rows, for scale: KL(B || A) per position {np.mean([r['x/kl_mc'].mean() for r in R]):.4f}; mean |log p(read token) A - B| {np.mean([np.abs(r['x/lpy'] - r['m/lpy']).mean() for r in R]):.4f}")
print("cell | unrouted rows: KL(B || cell), mean |delta log p(read token)|, share of positions with KL > 0.05 | routed rows: KL(B || cell) | leakage ratio (unrouted KL / routed KL)")
for n in names:
    ku = np.mean([r[f"{n}/kl_mc"].mean() for r in U]); du = np.mean([np.abs(r[f"{n}/lpy"] - r["m/lpy"]).mean() for r in U]); su = np.mean([(r[f"{n}/kl_mc"] > 0.05).mean() for r in U])
    kr = np.mean([r[f"{n}/kl_mc"].mean() for r in R]) if R and f"{n}/kl_mc" in R[0] else float("nan")
    print(f"{n} | {ku:.4f}, {du:.4f}, {su:.3f} | {kr:.4f} | {ku / kr if kr == kr and kr > 0 else float('nan'):.2f}")
