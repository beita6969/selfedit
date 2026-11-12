"""Per field: relative difference (||x - ref|| / ||ref||, per trace) of each realisation to the f32 reference.
python f32cmp.py REF_DIR NAME=DIR [NAME=DIR ...]"""
import glob, os, sys
import numpy as np
ref = sys.argv[1]; others = [a.split("=", 1) for a in sys.argv[2:]]
names = sorted(os.path.basename(f) for f in glob.glob(os.path.join(ref, "*.npz")))
F = [("logp", lambda r: np.atleast_1d(r["logp"])), ("dT", lambda r: np.atleast_1d(r["dT"])), ("g_attn", lambda r: r["g_attn"]), ("g_mlp", lambda r: r["g_mlp"]),
     ("g_qtemp", lambda r: r["g_qtemp"]), ("g_merger", lambda r: r["g_merger"]), ("bias_rp_gen", lambda r: r["bias_rp"][0]), ("bias_rp_img", lambda r: r["bias_rp"][1]),
     ("bias_rp_txt", lambda r: r["bias_rp"][2]), ("bias_full_gen", lambda r: r["bias_full"][0]), ("bias_full_img", lambda r: r["bias_full"][1]), ("bias_full_txt", lambda r: r["bias_full"][2])]
print("traces:", ", ".join(n[:-4] for n in names))
print(f"{'field':14s} " + " ".join(f"{nm:>24s}" for nm, _ in others) + "   (rel diff to f32 per trace)")
for f, get in F:
    cells = []
    for nm, d in others:
        v = []
        for n in names:
            p = os.path.join(d, n)
            if not os.path.exists(p):
                v.append(float("nan")); continue
            x, y = get(np.load(p)).astype(np.float64).ravel(), get(np.load(os.path.join(ref, n))).astype(np.float64).ravel()
            v.append(np.linalg.norm(x - y) / max(np.linalg.norm(y), 1e-30))
        cells.append("/".join(f"{u:.3f}" if u < 10 else f"{u:.0f}" for u in v))
    print(f"{f:14s} " + " ".join(f"{c:>24s}" for c in cells))
