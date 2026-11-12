"""ana_round (fable_int, iterated compile): one cell against another on the SAME held-out reads (b_fn5 records). Per cell name: restoration R, dose gamma, residual KL to the teacher
KL(text || cell) per position (all positions and the first 400 thinking tokens), activity KL(no text || cell); paired differences new - old over reads with their standard error.
python ana_round.py OLD_DIR NEW_DIR [name_old=name_new ...]   (default pairs: r64_all, r64_full)"""
import glob, os, sys
import numpy as np
old, new = sys.argv[1:3]; pairs = [p.split("=") for p in sys.argv[3:]] or [["r64_all", "r64_all"], ["r64_full", "r64_full"]]
O = {os.path.basename(f): np.load(f, allow_pickle=True) for f in sorted(glob.glob(old + "/*.npz"))}; N = {os.path.basename(f): np.load(f, allow_pickle=True) for f in sorted(glob.glob(new + "/*.npz"))}
K = sorted(set(O) & set(N)); se = lambda v: np.std(v, ddof=1) / np.sqrt(len(v))
print(f"{len(K)} reads in both ({old} -> {new}); KL(text || no text) per position {np.mean([O[k]['m/kl_xc'].mean() for k in K]):.4f}, first 400 tokens {np.mean([O[k]['m/kl_xc'][:400].mean() for k in K]):.4f}")
for a, b in pairs:
    def st(D, n):
        R = np.array([1 - D[k][n + '/kl_xc'].sum() / D[k]['m/kl_xc'].sum() for k in K]); res = np.array([D[k][n + '/kl_xc'].mean() for k in K]); res4 = np.array([D[k][n + '/kl_xc'][:400].mean() for k in K])
        act = np.array([D[k][n + '/kl_mc'].mean() for k in K]); g = sum(D[k][n + '/ip_cx'].sum() for k in K) / max(1e-12, sum(D[k]['x/ip_cx'].sum() for k in K))
        return R, res, res4, act, g
    Ro, ro, ro4, ao, go = st(O, a); Rn, rn, rn4, an, gn = st(N, b)
    print(f"{a} -> {b}: R {Ro.mean():.3f} -> {Rn.mean():.3f} (paired {np.mean(Rn - Ro):+.3f} +- {se(Rn - Ro):.3f}) | dose {go:.3f} -> {gn:.3f} | residual KL(text || cell) {ro.mean():.4f} -> {rn.mean():.4f} (paired {np.mean(rn - ro):+.4f} +- {se(rn - ro):.4f}); "
          f"first 400 tokens {ro4.mean():.4f} -> {rn4.mean():.4f} (paired {np.mean(rn4 - ro4):+.4f} +- {se(rn4 - ro4):.4f}) | activity KL(no text || cell) {ao.mean():.4f} -> {an.mean():.4f}")
