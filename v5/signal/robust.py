"""Robustness of the scalar features across two realisations (e.g. Vesta kernels vs node0015 torch fallback).
python robust.py DIR_A DIR_B   -> for traces in both dirs: per feature, correlation between realisations after row-centring
(each feature component minus its row mean over the row's traces present in BOTH dirs; rows with >= 2 common traces)."""
import glob, os, sys
import numpy as np
A, B = sys.argv[1:3]
common = sorted(set(map(os.path.basename, glob.glob(os.path.join(A, "*.npz")))) & set(map(os.path.basename, glob.glob(os.path.join(B, "*.npz")))))
rows = {}
for n in common:
    rows.setdefault(n.rsplit("__s", 1)[0], []).append(n)
rows = {r: ns for r, ns in rows.items() if len(ns) >= 2}
F = [("dT", lambda r: np.atleast_1d(r["dT"])), ("g_attn", lambda r: r["g_attn"]), ("g_mlp", lambda r: r["g_mlp"]), ("g_qtemp", lambda r: r["g_qtemp"]),
     ("g_merger", lambda r: r["g_merger"]), ("bias_rp_gen", lambda r: r["bias_rp"][0].ravel()), ("bias_rp_img", lambda r: r["bias_rp"][1].ravel()),
     ("bias_rp_txt", lambda r: r["bias_rp"][2].ravel())]
print(f"common traces {len(common)}; rows with >= 2 common traces {len(rows)} ({sum(map(len, rows.values()))} traces)")
print(f"{'feature':12s} {'pooled corr':>11s} {'median per-component corr':>26s}")
for f, get in F:
    xa, xb = [], []
    for ns in rows.values():
        a = np.stack([get(np.load(os.path.join(A, n))) for n in ns]).astype(np.float64); b = np.stack([get(np.load(os.path.join(B, n))) for n in ns]).astype(np.float64)
        xa.append(a - a.mean(0)); xb.append(b - b.mean(0))
    if not xa:
        break
    xa, xb = np.concatenate(xa), np.concatenate(xb)
    pooled = float((xa * xb).sum() / np.sqrt((xa ** 2).sum() * (xb ** 2).sum()))
    pc = [float(np.corrcoef(xa[:, j], xb[:, j])[0, 1]) for j in range(xa.shape[1]) if xa[:, j].std() > 0 and xb[:, j].std() > 0]
    print(f"{f:12s} {pooled:11.3f} {np.median(pc) if pc else float('nan'):26.3f}")
