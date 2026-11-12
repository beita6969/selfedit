"""h_stats (fable_int, hindsight): size and place of the hindsight forcing from a3 records, reads that ended wrong against reads that ended right.
KL(H || base) per thinking position by decile of the read; rms forcing relative to the state and share of the summed forcing energy by block band; share of the block-level forcing already in the
mixer output; per benchmark. python h_stats.py A3_DIR"""
import glob, os, sys
import numpy as np
R = [np.load(f, allow_pickle=True) for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.npz")))]
print(f"{len(R)} reads; hint block length (tokens) {sorted({int(r['S']) for r in R})[:6]}")
for name, sel in (("ended WRONG", [r for r in R if not bool(r["ok"])]), ("ended right", [r for r in R if bool(r["ok"])])):
    if not sel:
        continue
    kl = [r["kl"] for r in sel]; dec = np.mean([[k[int(len(k) * i / 10): max(int(len(k) * i / 10) + 1, int(len(k) * (i + 1) / 10))].mean() for i in range(10)] for k in kl], 0)
    print(f"\nreads that {name}: {len(sel)}; KL(H || base) per thinking position: mean {np.mean([k.mean() for k in kl]):.4f}, median over positions {np.median(np.concatenate(kl)):.4f}; share of positions with KL >= 1: {np.mean(np.concatenate(kl) >= 1):.4f}")
    print("  by decile of the read: " + " ".join(f"{x:.3f}" for x in dec))
    print("  first 30 / 30-150 / 150-400 / 400+ thinking tokens: " + " / ".join(f"{np.mean([k[a_:b_].mean() for k in kl if len(k) > a_]):.3f}" for a_, b_ in ((0, 30), (30, 150), (150, 400), (400, 10 ** 9))))
    nb = np.stack([r["nb"] for r in sel]).astype(np.float64); jsq = np.stack([r["jsq"] for r in sel]); isq = np.stack([r["isq"] for r in sel]); xsq = np.stack([r["xsq"] for r in sel]); W = nb.sum()
    J2 = (jsq * nb[:, None, :]).sum((0, 2)) / W; I2 = (isq * nb[:, None, :]).sum((0, 2)) / W; X2 = (xsq * nb[:, None, :]).sum((0, 2)) / W
    FB = list(range(3, 64, 4)); LB = [l for l in range(64) if l % 4 != 3]
    print("  share of the summed forcing energy by band: " + ", ".join(f"L{lo}-{hi} {J2[lo:hi + 1].sum() / J2.sum():.3f}" for lo, hi in ((0, 23), (24, 31), (32, 39), (40, 47), (48, 55), (56, 63))) + f"; in the 16 full-attention blocks {J2[FB].sum() / J2.sum():.3f}")
    print("  rms forcing / rms state: full-attention blocks 27-59 " + " ".join(f"{np.sqrt(J2[l] / X2[l]):.3f}" for l in range(27, 60, 4)) + " | linear blocks 26, 34, 42, 50, 58: " + " ".join(f"{np.sqrt(J2[l] / X2[l]):.4f}" for l in (26, 34, 42, 50, 58)))
    print("  mixer share |I|^2 / |J|^2, full-attention blocks 27-59: " + " ".join(f"{I2[l] / J2[l]:.2f}" for l in range(27, 60, 4)))
    for b in sorted({str(r["bench"]) for r in sel}):
        kb = [r["kl"].mean() for r in sel if str(r["bench"]) == b]; print(f"  {b}: reads {len(kb)}, KL {np.mean(kb):.4f}", end=";")
    print()
