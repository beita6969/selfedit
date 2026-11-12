"""ana_fade (fable_int): dose along the xhigh shift (gamma) by thinking-token bin for each cell, on long reads (b_xb records with pidx). python ana_fade.py DIR"""
import glob, os, sys
import numpy as np
R = [np.load(f, allow_pickle=True) for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.npz")))]
B = [(0, 500), (500, 1500), (1500, 3000), (3000, 5000), (5000, 8000), (8000, 12000), (12000, 10 ** 9)]
names = [n for n in R[0]["names"].tolist() if n not in ("m", "x")]
print(f"{len(R)} reads; thinking tokens {[int(r['te']) for r in R]}")
print("cell | gamma by thinking-token bin " + " ".join(f"[{a},{b if b < 10 ** 8 else 'inf'})" for a, b in B) + " | whole read gamma, R, KL(m||c)")
for n in names:
    line = []
    for a, b in B:
        g = []
        for r in R:
            m = (r["pidx"] >= a) & (r["pidx"] < b)
            if m.sum() > 5 and r["x/ip_cx"][m].sum() > 0:
                g.append(r[f"{n}/ip_cx"][m].sum() / r["x/ip_cx"][m].sum())
        line.append(f"{np.mean(g):+.2f}±{np.std(g) / np.sqrt(len(g)):.2f}({len(g)})" if g else "n/a")
    gam = [r[f"{n}/ip_cx"].sum() / r["x/ip_cx"].sum() for r in R]; RR = [1 - r[f"{n}/kl_xc"].sum() / r["m/kl_xc"].sum() for r in R]
    print(f"{n} | " + " ".join(line) + f" | {np.mean(gam):+.3f}, {np.mean(RR):+.3f}, {np.mean([r[n + '/kl_mc'].mean() for r in R]):.4f}")
