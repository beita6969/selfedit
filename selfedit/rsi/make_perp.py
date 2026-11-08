"""verbose_perp_position: long_vs_short_correct directions with the late_vs_early_correct (position) direction removed per layer."""
import numpy as np
L = "/data/fd/mingdazhang/selfedit/probe/library"
a = np.load(f"{L}/long_vs_short_correct.npz")
b = np.load(f"{L}/late_vs_early_correct.npz")
out, lines = {}, []
for kind in ("resid", "mixer", "mlp"):
    A, B = a[f"{kind}/dir"].astype(np.float64), b[f"{kind}/dir"].astype(np.float64)
    Bn = B / np.linalg.norm(B, axis=1, keepdims=True)
    cos = (A * Bn).sum(1) / np.linalg.norm(A, axis=1)
    P = A - (A * Bn).sum(1, keepdims=True) * Bn
    keep = np.linalg.norm(P, axis=1) / np.linalg.norm(A, axis=1)
    out[f"{kind}/dir"] = (P / np.linalg.norm(P, axis=1, keepdims=True)).astype(np.float32)
    out[f"{kind}/cos_position"] = cos.astype(np.float32)
    out[f"{kind}/kept_norm"] = keep.astype(np.float32)
    if kind == "resid":
        best = sorted(range(16, 64), key=lambda l: -a["resid/loo_acc"][l] * keep[l])[:6]
        lines = [f"L{l}: cos with position {cos[l]:+.2f}, kept {keep[l]:.2f}, original loo {a['resid/loo_acc'][l]:.2f}, d {a['resid/d_tok'][l]:.2f}" for l in best]
np.savez(f"{L}/verbose_perp_position.npz", **out)
md = ("# verbose_perp_position (built by the mentor)\n"
      "long_vs_short_correct is ~85% a position-in-context direction (cos 0.82-0.90 with late_vs_early_correct at every layer). "
      "This contrast keeps only the part of the verbosity direction orthogonal to position, per layer and kind (resid/mixer/mlp). "
      "Only <kind>/dir exists (no svd_V). Layers >= 16 with the most verbosity signal left:\n- " + "\n- ".join(lines) + "\n")
open(f"{L}/verbose_perp_position_digest.md", "w").write(md)
print(md)
