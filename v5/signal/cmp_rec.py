"""Compare score_worker records of the same traces in two dirs: python cmp_rec.py DIR_A DIR_B"""
import glob, os, sys
import numpy as np
A, B = sys.argv[1:3]
for fa in sorted(glob.glob(os.path.join(A, "*.npz"))):
    fb = os.path.join(B, os.path.basename(fa))
    if not os.path.exists(fb):
        continue
    a, b = np.load(fa), np.load(fb)
    print(f"{str(a['row'])}#{int(a['sample'])}: prompt {int(a['ptok'])}/{int(a['ptok_ref'])} gen {int(a['gtok'])} out-support {int(a['n_out'])}/{int(b['n_out'])}"
          f" | sec {float(a['sec']):.1f} vs {float(b['sec']):.1f} | peak {float(a['peak_gib']):.1f} vs {float(b['peak_gib']):.1f} GiB")
    for k in ("logp", "dT", "g_attn", "g_mlp", "g_qtemp", "g_merger", "bias_rp", "bias_full"):
        x, y = a[k].astype(np.float64).ravel(), b[k].astype(np.float64).ravel()
        rel = np.linalg.norm(x - y) / max(np.linalg.norm(y), 1e-30)
        cos = float(x @ y / max(np.linalg.norm(x) * np.linalg.norm(y), 1e-30))
        print(f"   {k:9s} rel {rel:.2e} cos {cos:.6f}" + (f"  ({float(x[0]):.4g} vs {float(y[0]):.4g})" if x.size == 1 else ""))
