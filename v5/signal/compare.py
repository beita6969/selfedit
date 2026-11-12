"""compare.py A B: loss and gradient agreement of two parity.py runs (A = fla, B = torch fallback)."""
import json, sys, numpy as np
A, B = sys.argv[1], sys.argv[2]
ja, jb = json.load(open(A + ".json")), json.load(open(B + ".json"))
ga, gb = np.load(A + ".npz"), np.load(B + ".npz")
print(f"loss  A {ja['loss']:.6f}  B {jb['loss']:.6f}  rel diff {abs(ja['loss'] - jb['loss']) / abs(jb['loss']):.2e}")
for k in ("chunk_impl", "fwd_s", "fwd_bwd_s", "weights_gb", "peak_gb", "reserved_gb", "n_prompt", "n_answer"):
    print(f"{k:12s} A {ja.get(k)}   B {jb.get(k)}")
for k in ga.files:
    x, y = ga[k].ravel().astype(np.float64), gb[k].ravel().astype(np.float64)
    rel = np.linalg.norm(x - y) / max(np.linalg.norm(y), 1e-30)
    cos = float(x @ y / max(np.linalg.norm(x) * np.linalg.norm(y), 1e-30)) if x.size > 1 else float("nan")
    print(f"{k:6s} |A| {np.linalg.norm(x):.4e} |B| {np.linalg.norm(y):.4e} rel diff {rel:.2e}" + (f" cos {cos:.6f}" if x.size > 1 else f"  A {x[0]:+.5e} B {y[0]:+.5e}"))
