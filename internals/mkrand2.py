"""mkrand2 (fable_int): RANDFL / ROTFL-type controls for any compiled cell built by mkproc.py (bias unit + linear units per block).
RAND (random read): the read directions of the linear units of each block are replaced by random directions orthogonal to the block's mean MLP input and uncorrelated with the
     original unit activations, mixed so that the covariance of their activations on the captured stream positions equals the original units'. Outputs, gates, neurons, dose unchanged.
ROT  (rotated write): all output vectors of the cell (bias + linear, all blocks) are carried by one random rotation into a random subspace orthogonal to the original; norms and angles kept.
Activation statistics from the a3 samples of the fit traces (MLP input of the hybrid run at thinking positions).
python mkrand2.py CELL.json ARRAYS.npz A3_DIR OUT_NPZ NPZ_PATH_IN_SPEC OUT_DIR"""
import glob, hashlib, json, os, sys
import numpy as np, torch
cellf, arrf, a3, out_npz, vpath, out_dir = sys.argv[1:7]; more = sys.argv[7:]      # further specs that use a subset of the same read / write keys (other gates, fewer blocks): same control arrays
assert not os.path.exists(out_npz), "arrays exist: choose a new name"
SEED = 20261111; rng = np.random.default_rng(SEED); dev = "cuda" if torch.cuda.is_available() else "cpu"
A = np.load(arrf); base = json.load(open(cellf)); name = os.path.basename(cellf)[:-5]
ups = {}; downs = []
for o in base:
    v = o.get("value")
    if isinstance(v, dict) and o["param"].endswith("up_proj") and v["key"].startswith("up_FL_"):
        l = int(o["param"].split(".")[1]); ups.setdefault(l, []).append(v["key"])
    if isinstance(v, dict) and o["param"].endswith("down_proj"):
        downs.append(v["key"])
S = [np.load(f)["U"] for f in sorted(glob.glob(os.path.join(a3, "samp", "*.npz")))]      # MLP inputs [64, ns, H] f16, loaded once
arr = {}
print(f"{name}: blocks {sorted(ups)}, linear units per block {len(next(iter(ups.values())))}, output vectors {len(downs)}; {len(S)} traces; seed {SEED}")
print("block | original activation variance (mean of units) | raw random reads: mean / max |corr| with the original activations | cond of the random covariance")
for l in sorted(ups):
    x = torch.tensor(np.concatenate([s[l][5:].astype(np.float32) for s in S]), device=dev).double(); n = len(x); xbar = x.mean(0); xc = x - xbar
    U = torch.tensor(np.stack([A[k].astype(np.float64) for k in ups[l]]), device=dev); R = len(U)
    A0 = xc @ U.T; CA = A0.T @ A0 / n
    G = torch.tensor(rng.standard_normal((R, x.shape[1])), device=dev)
    G -= torch.outer(G @ xbar, xbar) / (xbar @ xbar)
    B0 = xc @ G.T; rho = ((B0 / B0.std(0)).T @ (A0 / A0.std(0)) / n).abs()
    Q, _ = torch.linalg.qr(torch.cat([xbar[:, None], xc.T @ A0 / n], 1)); G -= (G @ Q) @ Q.T
    B = xc @ G.T; CB = B.T @ B / n
    Wm = torch.linalg.cholesky(CA + 1e-12 * torch.eye(R, device=dev, dtype=torch.float64)) @ torch.linalg.inv(torch.linalg.cholesky(CB)); Rn = Wm @ G
    chk = xc @ Rn.T; assert torch.allclose(chk.T @ chk / n, CA, rtol=1e-5, atol=1e-7)
    for k, r in zip(ups[l], Rn):
        arr[k.replace("up_FL_", "up_RF_")] = r.float().cpu().numpy()
    print(f"{l} | {float(CA.diagonal().mean()):.3f} | {float(rho.mean()):.3f} / {float(rho.max()):.3f} | {float(torch.linalg.cond(CB)):.1f}", flush=True)
Vo = np.stack([A[k].astype(np.float64).reshape(-1) for k in downs])
Us, Sv, Wt = np.linalg.svd(Vo, full_matrices=False); C = Us * Sv
G2 = rng.standard_normal(Wt.shape); G2 -= (G2 @ Wt.T) @ Wt; W2 = np.linalg.qr(G2.T)[0].T; Vn = C @ W2
assert np.allclose(Vn @ Vn.T, Vo @ Vo.T, rtol=1e-6, atol=1e-12)
for k, v in zip(downs, Vn):
    arr[k.replace("down_FJm_", "down_ROb_").replace("down_FL_", "down_RO_")] = v.astype(np.float32).reshape(A[k].shape)
np.savez(out_npz, **arr); os.makedirs(out_dir, exist_ok=True)


def swap(o, kind):
    v = o.get("value")
    if not isinstance(v, dict):
        return o
    k = v["key"]
    if kind == "read" and k.startswith("up_FL_"):
        return dict(o, value={"npz": vpath, "key": k.replace("up_FL_", "up_RF_")})
    if kind == "write" and o["param"].endswith("down_proj") and k.startswith("down_FL_"):
        return dict(o, value={"npz": vpath, "key": k.replace("down_FL_", "down_RO_")})
    if kind == "write" and o["param"].endswith("down_proj") and k.startswith("down_FJm_"):
        return dict(o, value={"npz": vpath, "key": k.replace("down_FJm_", "down_ROb_")})
    return o


for cf in [cellf] + more:
    base_ = json.load(open(cf)); nm = os.path.basename(cf)[:-5]
    for tag, kind in (("RAND", "read"), ("ROT", "write")):
        ops = [swap(o, kind) for o in base_]; s = json.dumps(ops); fn = os.path.join(out_dir, f"{tag}__{nm}.json"); assert not os.path.exists(fn)
        assert all(o["value"]["key"] in arr for o in ops if isinstance(o.get("value"), dict) and o["value"]["npz"] == vpath)
        open(fn, "w").write(s); print(f"{tag}__{nm}", len(ops) // 5, "units", sum(x is not y for x, y in zip(ops, base_)), "ops redirected", hashlib.sha256(s.encode()).hexdigest()[:16])
print("arrays", out_npz, len(arr), "keys", hashlib.sha256(open(out_npz, "rb").read()).hexdigest())
