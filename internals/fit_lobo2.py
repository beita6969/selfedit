"""fit_lobo2 (as fit_lobo, plus a control column: the rank-R read applied to the state of ANOTHER held position, drawn at random) (fable_int): reduced-rank ridge read of a forcing term (a3 samples) with a row-held-out fold and leave-one-benchmark-out folds. All samples are loaded once.
Folds: "rows" (held = reads of rows with crc32(row) % 5 == 0) and one per benchmark (held = every read of that benchmark; fitted on the other benchmarks only).
Per fold and block: bias = mean forcing at the fold's training thinking positions; features = MLP input minus its training mean, mean direction projected out; ridge (lambda x mean diagonal);
rank-R output directions = top right singular vectors of the fitted values; units scaled to unit activation variance (as fit3p).
Printed per fold and block: explained forcing energy 1 - |J - prediction|^2 / |J|^2 on the HELD positions at rank 8 / R / full ridge, for reads that ended wrong and for reads that ended right
(the bias-only value is given too). Arrays per fold: OUT/arrays_<TAG>_<fold>.npz with the keys of fit3p (k_l, ks_l from --base-arrays; down_FJm_l, up_FL_l_i, down_FL_l_i).
--tmax N: only the template tail and thinking positions < N are used (fit and evaluation): the opening-only read.
python fit_lobo.py --a3 DIR --out DIR --tag h --base-arrays keys.npz [--rmax 64] [--lam 0.3] [--tmax 0] [--blocks 0-63] [--folds rows,bench]"""
import argparse, glob, hashlib, json, os, zlib
import numpy as np, torch
ap = argparse.ArgumentParser()
ap.add_argument("--a3", required=True); ap.add_argument("--out", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--base-arrays", required=True)
ap.add_argument("--rmax", type=int, default=64); ap.add_argument("--lam", type=float, default=0.3); ap.add_argument("--tmax", type=int, default=0); ap.add_argument("--blocks", default="0-63"); ap.add_argument("--folds", default="rows,bench"); ap.add_argument("--no-save", action="store_true")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); dev = "cuda:0" if torch.cuda.is_available() else "cpu"
F = sorted(glob.glob(os.path.join(a.a3, "*.npz"))); meta = [np.load(f, allow_pickle=True) for f in F]
row = [str(m["row"]) for m in meta]; bench = [str(m["bench"]) for m in meta]; okv = np.array([bool(m["ok"]) for m in meta])
S = []
for f in F:
    z = np.load(os.path.join(a.a3, "samp", os.path.basename(f))); S.append((z["U"], z["J"], z["samp"]))
n = len(F); H = S[0][0].shape[-1]; BLO, BHI = (int(x) for x in a.blocks.split("-"))
tr_id = np.concatenate([np.full(len(s[2]), i) for i, s in enumerate(S)]); pos = np.concatenate([s[2] for s in S]); tail = pos < 5
use = np.ones(len(pos), bool) if not a.tmax else (pos < 5 + a.tmax)
folds = {}
if "rows" in a.folds:
    folds["rows"] = np.array([zlib.crc32(r.encode()) % 5 == 0 for r in row])
if "bench" in a.folds:
    for b in sorted(set(bench)):
        folds[b] = np.array([x == b for x in bench])
print(f"{n} reads ({int((~okv).sum())} ended wrong, {int(okv.sum())} right) of {len(set(row))} rows; sampled positions {len(pos)} (in use {int(use.sum())}); benchmarks {sorted(set(bench))}; lambda {a.lam}; rank {a.rmax}" + (f"; opening only: thinking tokens < {a.tmax}" if a.tmax else ""), flush=True)
BA = np.load(a.base_arrays); R = a.rmax
GEN = torch.Generator(device=dev); GEN.manual_seed(1112)
EV = {f: np.zeros((64, 2, 5)) for f in folds}             # [block, wrong / right, (bias only, rank 8, rank R, full)]
ARR = {f: {} for f in folds}
okp = okv[tr_id]
for l in range(BLO, BHI + 1):
    X = torch.tensor(np.concatenate([s[0][l].astype(np.float32) for s in S]), device=dev); Y = torch.tensor(np.concatenate([s[1][l].astype(np.float32) for s in S]), device=dev)
    for f, heldtr in folds.items():
        held = torch.tensor(heldtr[tr_id] & use, device=dev); trn = torch.tensor((~heldtr[tr_id]) & use, device=dev); tl = torch.tensor(tail, device=dev)
        Xt, Yt = X[trn], Y[trn]; xbar = Xt.mean(0); b = Yt[~tl[trn]].mean(0) if (~tl[trn]).any() else Yt.mean(0); xh = xbar / xbar.norm()
        proj = lambda Z: (Z - xbar) - ((Z - xbar) @ xh)[:, None] * xh[None]
        Xc, Yc = proj(Xt), Yt - b; G = Xc.T @ Xc / len(Xc); C = Xc.T @ Yc / len(Xc)
        B = torch.linalg.solve(G + a.lam * G.diagonal().mean() * torch.eye(H, device=dev), C)
        _, _, V = torch.svd_lowrank(Xc @ B, q=R + 16, niter=3); V = V[:, :R]; BV = B @ V
        for j, sel in enumerate((held & torch.tensor(~okp, device=dev), held & torch.tensor(okp, device=dev))):
            if int(sel.sum()) == 0:
                EV[f][l, j] = np.nan; continue
            Xv, Yv = proj(X[sel]), Y[sel]; tot = float(Yv.pow(2).sum()); Yvc = Yv - b
            pm_ = torch.randperm(len(Xv), device=dev, generator=GEN)
            EV[f][l, j] = [1 - float(Yvc.pow(2).sum()) / tot, 1 - float((Yvc - (Xv @ BV[:, :8]) @ V[:, :8].T).pow(2).sum()) / tot, 1 - float((Yvc - (Xv @ BV) @ V.T).pow(2).sum()) / tot, 1 - float((Yvc - Xv @ B).pow(2).sum()) / tot, 1 - float((Yvc - (Xv[pm_] @ BV) @ V.T).pow(2).sum()) / tot]
        if not a.no_save:
            sc = (Xc @ BV).std(0).clamp_min(1e-8); UP = (BV / sc[None]).T.cpu().numpy(); DN = (V * sc[None]).T.cpu().numpy()
            ARR[f][f"down_FJm_{l}"] = (b.cpu().numpy() / 8).astype(np.float32).reshape(-1, 1)
            for i in range(R):
                ARR[f][f"up_FL_{l}_{i}"] = UP[i].astype(np.float32); ARR[f][f"down_FL_{l}_{i}"] = (DN[i] / 8).astype(np.float32).reshape(-1, 1)
        del B, G, C, Xc, Yc, Xt, Yt
    print(f"block {l:2d}{'*' if l % 4 == 3 else ' '} | " + " | ".join(f"{f}: wrong {EV[f][l, 0, 1]:+.2f}/{EV[f][l, 0, 2]:+.2f}/{EV[f][l, 0, 3]:+.2f} right {EV[f][l, 1, 2]:+.2f}" for f in folds), flush=True)
    del X, Y
print("\nexplained forcing energy on HELD reads: fold | reads held (wrong, right) | wrong reads: bias only / rank 8 / rank R / full ridge, mean over the full-attention blocks 27-59, the linear blocks 24-62, all blocks | right reads: rank R, same three means")
FB = [l for l in range(27, 60, 4) if BLO <= l <= BHI]; LB = [l for l in range(24, 63) if l % 4 != 3 and BLO <= l <= BHI]; AB = list(range(BLO, BHI + 1))
for f, heldtr in folds.items():
    m = lambda j, c, bl: np.nanmean(EV[f][bl, j, c]) if bl else float("nan")
    print(f"{f} | {int((heldtr & ~okv).sum())}, {int((heldtr & okv).sum())} | " + " ; ".join(" / ".join(f"{m(0, c, bl):+.3f}" for c in range(4)) for bl in (FB, LB, AB)) + " | " + " ; ".join(f"{m(1, 2, bl):+.3f}" for bl in (FB, LB, AB)))
np.savez(os.path.join(a.out, f"ev_{a.tag}.npz"), **{f: EV[f] for f in folds})
if not a.no_save:
    for f in folds:
        arr = dict(ARR[f])
        for l in range(64):
            arr[f"k_{l}"] = BA[f"k_{l}"].astype(np.float32).reshape(-1); arr[f"ks_{l}"] = BA[f"ks_{l}"].astype(np.float32).reshape(-1)
        fn = os.path.join(a.out, f"arrays_{a.tag}_{f}.npz"); assert not os.path.exists(fn); np.savez(fn, **arr)
        print("wrote", fn, len(arr), "keys")
