"""fit3p (fable_int): fit3 for the consolidation task (ranks up to 64, the validation fold as an option). Forcing-term cells from a3 records.
1. tables: per block |mean J|, |J| rms by thinking-token bin (decay), constancy, cosine with the XS increment, forcing vs total increment.
2. per block: mean forcing b_l (thinking positions), early-level forcing (tokens 8-150), and a reduced-rank ridge map J ~ b + (x - xbar) B_r from the
   MLP input x of the hybrid run (the input the unit will see), features orthogonal to xbar; rows split train / val by crc32(row) % 5 == 0.
3. writes OUT/arrays_fj.npz and edit specs (Surgeon op vocabulary: set / scale on mlp.gate_proj | up_proj | down_proj, sel neurons, npz values).
   bias unit (XS's neuron): gate = 8 k_l, up = k_l, down = b_l / 8.   linear unit i: gate = 8 g_l (g = k_l or the stream key), up = u_i, down = v_i / 8.
"""
import argparse, glob, hashlib, json, os, sys, zlib
import numpy as np, torch
ap = argparse.ArgumentParser()
ap.add_argument("--a3", default="a3/fit"); ap.add_argument("--out", default="cells_fj"); ap.add_argument("--xs", default="arrays_xs.npz"); ap.add_argument("--ck", default="/dev/shm/fable_dir/xsig/ck.npz")
ap.add_argument("--xs-spec", default="xs_edits/XS_bias_all_+1.json"); ap.add_argument("--rmax", type=int, default=16); ap.add_argument("--device", default="cpu")
ap.add_argument("--npz-path", default="/data/fd/mingdazhang/selfedit/v4/cells/fable_int/arrays_fj.npz"); ap.add_argument("--no-fit", action="store_true")
ap.add_argument("--kcap", default="", help="dir of kcap records: fit the stream key and gate the linear units with it")
ap.add_argument("--exclude-bench", default="", help="leave this benchmark out of every fit (OOD check)"); ap.add_argument("--blocks", default="0-63")
ap.add_argument("--val-fold", type=int, default=0, help="rows with crc32(row) % 5 == fold are the validation rows of the fit")
ap.add_argument("--base-arrays", default="", help="copy k_l / ks_l and all other keys from this npz; only the FL / FJ arrays of the fitted blocks are replaced"); ap.add_argument("--arrays-out", default="arrays_fj.npz")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
torch.set_num_threads(32)
dev = a.device
F = sorted(glob.glob(os.path.join(a.a3, "*.npz")))
R = [np.load(f, allow_pickle=True) for f in F]
if a.exclude_bench:
    keep_ = [i for i, r in enumerate(R) if str(r["bench"]) != a.exclude_bench]
    F = [F[i] for i in keep_]; R = [R[i] for i in keep_]
BLO, BHI = (int(x) for x in a.blocks.split("-"))
n = len(R); NL, H = 64, 5120
EDGES = [0, 8, 32, 150, 400, 1000, 2000, 4000, 10 ** 9]; NB = len(EDGES) - 1
isval = np.array([zlib.crc32(str(r["row"]).encode()) % 5 == a.val_fold for r in R])
print(f"{n} traces ({isval.sum()} val rows)")
if a.xs and os.path.exists(a.xs):
    XS = np.load(a.xs)
    xs_inc = np.stack([XS[f"down_XS_{l}"].astype(np.float32).reshape(-1) * 8 for l in range(NL)]); KEY = np.stack([XS[f"k_{l}"].astype(np.float32) for l in range(NL)])
else:                                               # no XS arrays on this machine: the XS columns of the table are zero, the constant keys come from --base-arrays
    BA = np.load(a.base_arrays); xs_inc = np.zeros((NL, H), np.float32); KEY = np.stack([BA[f"k_{l}"].astype(np.float32).reshape(-1) for l in range(NL)])
cos = lambda x, y: float((x * y).sum() / max(1e-12, np.linalg.norm(x) * np.linalg.norm(y)))
te = np.array([int(r["te"]) for r in R])
NBv = np.stack([r["nb"] for r in R]).astype(np.float64)                 # [n, NB]
JM = np.stack([r["Jm"] for r in R])                                     # [n, 64, H] row means over thinking positions
Jbar = (JM * te[:, None, None]).sum(0) / te.sum()                      # position-weighted mean forcing
Jbar_row = JM.mean(0)                                                   # row-weighted
DM = np.stack([r["Dm"] for r in R]); Dbar = DM.mean(0); tinc = np.concatenate([Dbar[:1], np.diff(Dbar, axis=0)])        # total increment (row-weighted)
JB = np.stack([r["Jb"].astype(np.float32) for r in R])                 # [n, 64, NB, H]
w = NBv / np.maximum(1, NBv.sum(0, keepdims=True))                      # per bin, weight of each row
Jbin = np.einsum("nb,nlbh->lbh", w, JB)                                 # [64, NB, H] mean forcing per bin
jsq = np.stack([r["jsq"] for r in R]); dmsq = np.stack([r["dmsq"] for r in R]); xsq = np.stack([r["xsq"] for r in R])
jrms = np.sqrt((jsq * NBv[:, None, :]).sum(0) / np.maximum(1, NBv.sum(0))[None])        # [64, NB] rms |J_t|
xrms = np.sqrt((xsq * NBv[:, None, :]).sum(0) / np.maximum(1, NBv.sum(0))[None])
drms = np.sqrt((dmsq * NBv[:, None, :]).sum(0) / np.maximum(1, NBv.sum(0))[None])
hA = JM[::2].mean(0); hB = JM[1::2].mean(0)
print("\n== forcing term J_l (block output: xhigh context minus medium context on identical stream input), thinking positions")
print("bins (thinking-token index): " + " ".join(f"[{EDGES[b]},{EDGES[b + 1] if EDGES[b + 1] < 10 ** 8 else 'inf'})" for b in range(NB)) + " ; positions per bin: " + " ".join(str(int(x)) for x in NBv.sum(0)))
print("L   |Jbar|  halfcos |total inc| |XS inc| cos(Jbar,XSinc) cos(Jbar,totinc) const | rms|J_t| by bin ... | |Jbin| (mean vector norm) by bin ... | rms|J|/rms|x| bin 3")
for l in range(NL):
    const = float((Jbar[l] ** 2).sum()) / max(1e-12, float((jsq[:, l] * NBv).sum() / NBv.sum()))
    print(f"{l:2d}{'*' if l % 4 == 3 else ' '} {np.linalg.norm(Jbar[l]):6.3f}  {cos(hA[l], hB[l]):+.2f}   {np.linalg.norm(tinc[l]):6.3f}   {np.linalg.norm(xs_inc[l]):6.3f}   {cos(Jbar[l], xs_inc[l]):+.2f}   {cos(Jbar[l], tinc[l]):+.2f}   {const:.3f} | "
          + " ".join(f"{jrms[l, b]:.3f}" for b in range(NB)) + " | " + " ".join(f"{np.linalg.norm(Jbin[l, b]):.3f}" for b in range(NB)) + f" | {jrms[l, 3] / xrms[l, 3]:.4f}")
tot = (Jbar ** 2).sum(1)
print("share of sum_l |Jbar_l|^2 by band: " + ", ".join(f"L{lo}-{hi} {tot[lo:hi + 1].sum() / tot.sum():.3f}" for lo, hi in ((0, 23), (24, 31), (32, 39), (40, 47), (48, 55), (56, 63))))
print("sum_l Jbar_l vs final total offset: |sum J| %.2f, |d_final| %.2f, cos %.2f ; |sum XS inc| %.2f" % (np.linalg.norm(Jbar.sum(0)), np.linalg.norm(Dbar[63]), cos(Jbar.sum(0), Dbar[63]), np.linalg.norm(xs_inc.sum(0))))
# decay with error bars: per trace ratio rms|J| in bin b / bin 3 (150-400), layers pooled over 24-47 and 48-63
print("\ndecay of the forcing along the reasoning: per-trace rms|J_t| relative to tokens 150-400 (mean ± se over traces that have both bins)")
for name, ls in (("L24-39", range(24, 40)), ("L40-55", range(40, 56)), ("L56-63", range(56, 64))):
    line = []
    for b in range(NB):
        ok = (NBv[:, b] > 20) & (NBv[:, 3] > 20)
        if ok.sum() < 3:
            line.append("   n/a   "); continue
        ratio = np.sqrt(jsq[ok][:, ls, b].mean(1) / jsq[ok][:, ls, 3].mean(1))
        line.append(f"{ratio.mean():.2f}±{ratio.std() / np.sqrt(ok.sum()):.2f}({int(ok.sum())})")
    print(f"{name}: " + "  ".join(line))
print("same for the projection of the bin's mean forcing on the overall mean direction (signed, relative to bin 150-400):")
for name, ls in (("L24-39", range(24, 40)), ("L40-55", range(40, 56)), ("L56-63", range(56, 64))):
    line = []
    for b in range(NB):
        ok = (NBv[:, b] > 20) & (NBv[:, 3] > 20)
        if ok.sum() < 3:
            line.append("   n/a   "); continue
        u = Jbar[list(ls)] / np.linalg.norm(Jbar[list(ls)], axis=1, keepdims=True)
        pb = np.einsum("nlh,lh->n", JB[ok][:, list(ls), b], u); p3 = np.einsum("nlh,lh->n", JB[ok][:, list(ls), 3], u)
        line.append(f"{(pb / p3).mean():.2f}±{(pb / p3).std() / np.sqrt(ok.sum()):.2f}")
    print(f"{name}: " + "  ".join(line))
# instruction attention controls
if "mass59" in R[0]:
    print("\n== controls for the attention hypothesis (xhigh run)")
    for l in (47, 51, 55, 59, 63):
        r2 = np.array([r[f"r2_{l}"] for r in R]); dr = np.array([r[f"dir{l}"] for r in R])
        print(f"L{l}: R2 of the per-position residual difference on the 24 heads' instruction mass (in-sample, beyond the mean) {r2[:, 0].mean():.3f} ± {r2[:, 0].std() / np.sqrt(n):.3f}; "
              f"direct instruction term vs mixer difference: cos {dr[:, 0].mean():+.3f}, energy ratio {dr[:, 1].mean() / dr[:, 2].mean():.2f}, residual energy after subtracting it {dr[:, 3].mean() / dr[:, 2].mean():.2f}; vs forcing I: cos {dr[:, 4].mean():+.3f}, residual {dr[:, 5].mean() / dr[:, 6].mean():.2f}")
    tok = np.mean([r["tokm59"] for r in R], 0)       # [24, S]
    print("L59 mean attention mass by instruction token (sum over 24 heads / 24): " + " ".join(f"{x:.3f}" for x in tok.mean(0)))
    tok55 = np.mean([r["tokm55"] for r in R], 0)
    print("L55: " + " ".join(f"{x:.3f}" for x in tok55.mean(0)))
np.savez(os.path.join(a.out, "forcing_stats.npz"), Jbar=Jbar, Jbar_row=Jbar_row, Jbin=Jbin, jrms=jrms, tinc=tinc, Dbar=Dbar)
if a.no_fit:
    raise SystemExit(0)

# ------------------------------------------------------------------ reduced-rank map
S = []
for f in F:
    z = np.load(os.path.join(a.a3, "samp", os.path.basename(f)))
    S.append(dict(U=z["U"], J=z["J"]))
print("samples loaded", flush=True)
RANKS = [1, 2, 4, 8, 16, 32, 64]; RANKS = [r for r in RANKS if r <= a.rmax]
UP = np.zeros((NL, a.rmax, H), np.float32); DOWN = np.zeros((NL, a.rmax, H), np.float32); EV = np.zeros((NL, len(RANKS) + 2)); BIAS = np.zeros((NL, H), np.float32)
SKEY = None
if a.kcap:
    KF = sorted(glob.glob(os.path.join(a.kcap, "*.npz"))); KR = [{k_: v_ for k_, v_ in np.load(f, allow_pickle=True).items()} for f in KF]
    kval = np.array([zlib.crc32(str(r["row"]).encode()) % 5 == 0 for r in KR])
    SKEY = np.zeros((NL, H), np.float32); sk_stats = []
print("\n== reduced-rank ridge map of the forcing on the MLP input (val rows): explained energy 1 - |J - pred|^2 / |J|^2")
print("L   n_tr  n_va | mean only | " + " ".join(f"r={r}" for r in RANKS) + " | full ridge | lam | tail positions: mean-only, r=max" + (" | stream key: val mean on stream / context, sd" if a.kcap else ""))
for l in range(BLO, BHI + 1):
    X = torch.tensor(np.concatenate([s["U"][l].astype(np.float32) for s in S]), device=dev); Y = torch.tensor(np.concatenate([s["J"][l].astype(np.float32) for s in S]), device=dev)
    rowid = np.concatenate([np.full(s["U"].shape[1], i) for i, s in enumerate(S)]); istail = np.concatenate([np.arange(s["U"].shape[1]) < 5 for s in S])
    va = torch.tensor(isval[rowid], device=dev); tl = torch.tensor(istail, device=dev)
    Xtr, Ytr, Xva, Yva = X[~va], Y[~va], X[va], Y[va]
    xbar = Xtr.mean(0); b = Ytr[~tl[~va]].mean(0)                       # bias = mean forcing at thinking positions
    xh = xbar / xbar.norm()
    proj = lambda Z: (Z - xbar) - ((Z - xbar) @ xh)[:, None] * xh[None]
    Xc, Yc, Xvc, Yvc = proj(Xtr), Ytr - b, proj(Xva), Yva - b
    G = Xc.T @ Xc / len(Xc); C = Xc.T @ Yc / len(Xc)
    tot = float(Yva.pow(2).sum()); ev0 = 1 - float(Yvc.pow(2).sum()) / tot
    best = None
    for lam in (0.03, 0.3, 3.0):
        B = torch.linalg.solve(G + lam * G.diagonal().mean() * torch.eye(H, device=dev), C)
        _, _, V = torch.svd_lowrank(Xc @ B, q=a.rmax + 16, niter=6)      # principal output directions of the fitted values
        V = V[:, : a.rmax]                                                # [H, rmax]
        BV = B @ V                                                        # [H, rmax] input directions
        evs = [1 - float((Yvc - (Xvc @ BV[:, :r]) @ V[:, :r].T).pow(2).sum()) / tot for r in RANKS]
        evf = 1 - float((Yvc - Xvc @ B).pow(2).sum()) / tot
        if best is None or evs[-1] > best[0][-1]:
            best = (evs, evf, lam, BV.clone(), V.clone())
        del B
    evs, evf, lam, BV, V = best
    sc = (Xc @ BV).std(0).clamp_min(1e-8)                                 # unit activations ~ unit variance
    UP[l] = (BV / sc[None]).T.cpu().numpy(); DOWN[l] = (V * sc[None]).T.cpu().numpy(); BIAS[l] = b.cpu().numpy()
    EV[l] = [ev0] + evs + [evf]
    tv = tl[va]
    tl_tot = float(Yva[tv].pow(2).sum()); ev_t0 = 1 - float(Yvc[tv].pow(2).sum()) / tl_tot; ev_tr = 1 - float((Yvc[tv] - (Xvc[tv] @ BV) @ V.T).pow(2).sum()) / tl_tot
    extra = ""
    if a.kcap:
        Xk = torch.tensor(np.concatenate([r["X"][l].astype(np.float32) for r in KR], 0), device=dev); ck = np.concatenate([r["cls"] for r in KR]); rk = np.concatenate([np.full(len(r["cls"]), i) for i, r in enumerate(KR)])
        yk = torch.tensor((ck >= 3).astype(np.float32), device=dev); vk = torch.tensor(kval[rk], device=dev)
        wk = torch.where(yk > 0, 0.5 / yk[~vk].sum(), 0.5 / (1 - yk[~vk]).sum())[~vk]      # classes weighted equally
        Gk = (Xk[~vk] * wk[:, None]).T @ Xk[~vk]; sk = torch.linalg.solve(Gk + 1e-3 * Gk.diagonal().mean() * torch.eye(H, device=dev), (Xk[~vk] * wk[:, None]).T @ yk[~vk])
        z = Xk[vk] @ sk; ys = yk[vk] > 0
        SKEY[l] = sk.cpu().numpy(); sk_stats.append((float(z[ys].mean()), float(z[ys].std()), float(z[~ys].mean()), float(z[~ys].std())))
        extra = f" | {sk_stats[-1][0]:.3f} / {sk_stats[-1][2]:.3f}, sd {sk_stats[-1][1]:.3f} / {sk_stats[-1][3]:.3f}"
        del Xk, Gk
    print(f"{l:2d}{'*' if l % 4 == 3 else ' '} {len(Xc):5d} {len(Xvc):5d} | {ev0:+.3f} | " + " ".join(f"{e:+.3f}" for e in evs) + f" | {evf:+.3f} | {lam} | {ev_t0:+.3f} {ev_tr:+.3f}" + extra, flush=True)
    del X, Y, G, C, Xc, Yc, Xvc, Yvc
np.savez(os.path.join(a.out, "fit_ev.npz"), EV=EV, ranks=np.array(RANKS))
arr = dict(np.load(a.base_arrays)) if a.base_arrays else {f"k_{l}": KEY[l] for l in range(NL)}
for l in range(BLO, BHI + 1):
    arr[f"down_FJ_{l}"] = (Jbar[l] / 8).astype(np.float32)                                   # mean forcing, all thinking positions
    arr[f"down_FJe_{l}"] = ((Jbin[l, 2] * NBv[:, 2].sum() + Jbin[l, 1] * NBv[:, 1].sum()) / (NBv[:, 1].sum() + NBv[:, 2].sum()) / 8).astype(np.float32)   # tokens 8-150
    arr[f"down_FJm_{l}"] = (BIAS[l] / 8).astype(np.float32)                                  # bias of the linear map (train rows)
    for i in range(a.rmax):
        arr[f"up_FL_{l}_{i}"] = UP[l, i]; arr[f"down_FL_{l}_{i}"] = (DOWN[l, i] / 8).astype(np.float32)
    if SKEY is not None:
        arr[f"ks_{l}"] = SKEY[l]
arr = {k_: (np.asarray(v_, np.float32).reshape(-1, 1) if k_.startswith("down_") else np.asarray(v_, np.float32).reshape(-1)) for k_, v_ in arr.items()}
np.savez(os.path.join(a.out, a.arrays_out), **arr)
print("wrote", os.path.join(a.out, a.arrays_out), hashlib.sha256(open(os.path.join(a.out, a.arrays_out), "rb").read()).hexdigest())
