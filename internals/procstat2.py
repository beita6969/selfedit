"""procstat2 (fable_int, consolidation; spectra from the Gram matrix, fast): shape of the forcing term of a routed procedure from a3 records (mode proc).
Per block: rms |J_t| at thinking positions relative to rms |x_t|; share of the block-level forcing energy that is already in the mixer output (|I|^2 / |J|^2; the rest enters through the MLP);
constancy |mean J|^2 / mean |J|^2; from the sampled positions: participation ratio of J over positions and the number of principal directions holding 50 / 80 / 90 % of its energy,
at thinking positions and at the 5 tail positions; share of the total forcing energy (sum over blocks) per block band.
python procstat.py A3_DIR [--blocks 0-63:1]"""
import glob, os, sys
import numpy as np, torch
d = sys.argv[1]; F = sorted(glob.glob(os.path.join(d, "*.npz"))); R = [np.load(f, allow_pickle=True) for f in F]
n = len(R); NL = 64
nb = np.stack([r["nb"] for r in R]).astype(np.float64); jsq = np.stack([r["jsq"] for r in R]); isq = np.stack([r["isq"] for r in R]); xsq = np.stack([r["xsq"] for r in R]); dmsq = np.stack([r["dmsq"] for r in R])
te = np.array([int(r["te"]) for r in R]); JM = np.stack([r["Jm"] for r in R]); Jbar = (JM * te[:, None, None]).sum(0) / te.sum()
W = nb.sum()
J2 = (jsq * nb[:, None, :]).sum((0, 2)) / W; I2 = (isq * nb[:, None, :]).sum((0, 2)) / W; X2 = (xsq * nb[:, None, :]).sum((0, 2)) / W
print(f"{n} traces, {int(W)} thinking positions; thinking tokens median {int(np.median(te))}")
print("share of the summed forcing energy (sum over blocks of mean |J_t|^2) by band: " + ", ".join(f"L{lo}-{hi} {J2[lo:hi + 1].sum() / J2.sum():.3f}" for lo, hi in ((0, 7), (8, 15), (16, 23), (24, 31), (32, 39), (40, 47), (48, 55), (56, 63))))
EDGES = [0, 8, 32, 150, 400, 1000, 2000, 4000, 10 ** 9]
jb = np.sqrt((jsq * nb[:, None, :]).sum(0) / np.maximum(1, nb.sum(0))[None])      # [64, NB]
print("rms |J_t| by thinking-token bin [0,8) [8,32) [32,150) [150,400) [400,1000) [1000,2000) [2000,4000), mean over blocks 24-39 / 40-55 / 56-63: " +
      " | ".join(" ".join(f"{jb[lo:hi, b].mean():.3f}" for b in range(7)) for lo, hi in ((24, 40), (40, 56), (56, 64))))
S = [np.load(os.path.join(d, "samp", os.path.basename(f)))["J"] for f in F]      # [64, ns, H] f16, loaded once
print("\nL    rms|J|/rms|x|  mixer share |I|^2/|J|^2  constancy | thinking positions: participation ratio, directions for 50 / 80 / 90 % | tail positions: participation ratio")
bl = range(0, 64)
dev = "cuda" if torch.cuda.is_available() else "cpu"
for l in bl:
    J = torch.tensor(np.concatenate([s[l][5:].astype(np.float32) for s in S]), device=dev); Jt = torch.tensor(np.concatenate([s[l][:5].astype(np.float32) for s in S]), device=dev)
    def spec(M):
        M = M.double(); G = (M.T @ M) if M.shape[0] > M.shape[1] else (M @ M.T)
        sv = torch.linalg.eigvalsh(G).flip(0).clamp_min(0); pr = float(sv.sum() ** 2 / (sv ** 2).sum()); c = torch.cumsum(sv, 0) / sv.sum()
        return pr, [int((c < q).sum()) + 1 for q in (0.5, 0.8, 0.9)]
    pr, ks = spec(J); prt, _ = spec(Jt)
    const = float((Jbar[l] ** 2).sum()) / max(1e-12, J2[l])
    print(f"{l:2d}{'*' if l % 4 == 3 else ' '}  {np.sqrt(J2[l] / X2[l]):.4f}        {I2[l] / J2[l]:.2f}              {const:.3f}  | {pr:6.1f}  {ks[0]:3d} / {ks[1]:3d} / {ks[2]:3d} | {prt:5.1f}", flush=True)
