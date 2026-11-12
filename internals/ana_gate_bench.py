"""ana_gate_bench (fable_int): a gated cell on ROUTED reads, by benchmark (b_fn5 records with the text as reference). Per cell and benchmark: reads, what the text does
KL(text || no text), the residual KL(text || cell), restoration R, the cell's activity KL(no text || cell) and its dose. python ana_gate_bench.py EV_DIR"""
import glob, os, sys
import numpy as np
R = [np.load(f, allow_pickle=True) for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.npz")))]
names = [n for n in R[0]["names"].tolist() if n not in ("m", "x")]; B = sorted({str(r["bench"]) for r in R}) + ["ALL"]
print(f"{len(R)} routed reads: " + ", ".join(f"{b} {sum(str(r['bench']) == b for r in R)}" for b in B[:-1]))
print("text alone, KL(text || no text) per position: " + " | ".join(f"{b} {np.mean([r['m/kl_xc'].mean() for r in R if b == 'ALL' or str(r['bench']) == b]):.4f}" for b in B))
print("cell | benchmark: residual KL(text || cell), R, activity KL(no text || cell), dose")
for n in names:
    line = []
    for b in B:
        S = [r for r in R if b == "ALL" or str(r["bench"]) == b]
        res = np.mean([r[n + "/kl_xc"].mean() for r in S]); Rr = np.mean([1 - r[n + "/kl_xc"].sum() / max(1e-12, r["m/kl_xc"].sum()) for r in S]); act = np.mean([r[n + "/kl_mc"].mean() for r in S])
        g = sum(r[n + "/ip_cx"].sum() for r in S) / max(1e-12, sum(r["x/ip_cx"].sum() for r in S))
        line.append(f"{b} {res:.4f}, {Rr:+.2f}, {act:.4f}, {g:.2f}")
    print(f"{n} | " + " | ".join(line))
