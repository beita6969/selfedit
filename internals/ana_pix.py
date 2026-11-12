"""ana_pix (fable_int, C_pix): explained forcing energy on held reads from fit_lobo2 (ev_<tag>.npz), weighted over the blocks by their share of the forcing energy (a3 records).
Columns: bias only / rank 8 / rank 64 / full ridge / control (rank-64 read of another held position's state); reads that ended wrong and right pooled by their position counts are not
available per block, so the two outcome groups are printed separately. python ana_pix.py A3_DIR EV.npz"""
import glob, os, sys
import numpy as np
R = [np.load(f, allow_pickle=True) for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.npz")))]; EV = np.load(sys.argv[2])
nb = np.stack([r["nb"] for r in R]).astype(np.float64); jsq = np.stack([r["jsq"] for r in R]); J2 = (jsq * nb[:, None, :]).sum((0, 2)) / nb.sum(); w = J2 / J2.sum()
FB = [l for l in range(64) if l % 4 == 3]; LB = [l for l in range(64) if l % 4 != 3]
print(f"{len(R)} reads; image tokens base {np.mean([int(r['nimg_base']) for r in R if 'nimg_base' in r.files] or [0]):.0f} -> mode {np.mean([int(r['nimg_mode']) for r in R if 'nimg_mode' in r.files] or [0]):.0f}; KL(mode || base) per thinking position {np.mean([r['kl'].mean() for r in R]):.4f}")
print("share of the forcing energy: full-attention blocks %.3f; blocks 0-23 %.3f, 24-39 %.3f, 40-47 %.3f, 48-55 %.3f, 56-63 %.3f" % (w[FB].sum(), w[:24].sum(), w[24:40].sum(), w[40:48].sum(), w[48:56].sum(), w[56:].sum()))
print("fold | outcome | energy-weighted over all blocks: bias only / rank 8 / rank 64 / full ridge / control | full-attention blocks, rank 64 | linear blocks, rank 64 | blocks 48-63, rank 64")
for f in EV.files:
    E = EV[f]
    for j, nm in ((0, "wrong"), (1, "right")):
        if np.isnan(E[:, j]).all():
            continue
        e = np.nan_to_num(E[:, j]); ww = lambda bl: w[bl] / w[bl].sum()
        print(f"{f} | {nm} | " + " / ".join(f"{(w * e[:, c]).sum():+.3f}" for c in range(E.shape[2])) + f" | {(ww(FB) * e[FB, 2]).sum():+.3f} | {(ww(LB) * e[LB, 2]).sum():+.3f} | {(ww(list(range(48, 64))) * e[48:, 2]).sum():+.3f}")
print("rank-64 explained energy by block, fold rows, wrong reads: " + " ".join(f"{l}:{EV['rows'][l, 0, 2]:+.2f}" for l in (15, 23, 27, 31, 35, 39, 43, 47, 50, 51, 55, 58, 59, 62, 63)) if "rows" in EV.files else "")
