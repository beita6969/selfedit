"""mk_krt (fable_int, route-keyed cells with a sharpened gate): adds to an arrays file of the counting cell
  krt<N>_l = (s / 8) (kr_l - theta k_l)   so that a neuron's gate 8 krt.x = s (z - theta), z = the route key's value: silu gives about 0 below theta and a ramp above it;
  up_FL_l_64 = k_l, down_FL_l_64 = down_FJm_l : the block's bias as a 65th gated unit (read = the constant key), so that the bias is switched by the same gate.
Cells then use rank 65, gate krt<N>, nobias. python mk_krt.py IN.npz OUT.npz NAME:s:theta [NAME:s:theta ...]"""
import hashlib, os, sys
import numpy as np
src, out = sys.argv[1:3]; assert not os.path.exists(out), "output exists: choose a new name"
A = dict(np.load(src)); n0 = len(A)
for v in sys.argv[3:]:
    name, s, th = v.split(":"); s, th = float(s), float(th)
    for l in range(64):
        A[f"krt{name}_{l}"] = ((s / 8.0) * (A[f"kr_{l}"].astype(np.float64).reshape(-1) - th * A[f"k_{l}"].astype(np.float64).reshape(-1))).astype(np.float32)
    print(f"gate krt{name}: slope {s}, threshold {th}: unit factor silu(s (z - theta)) / 8 at z = 0.2, 0.35, 0.5, 0.65, 0.8, 1.0: " + ", ".join(f"{(lambda g: g / (1 + np.exp(-g)) / 8)(s * (z - th)):+.3f}" for z in (0.2, 0.35, 0.5, 0.65, 0.8, 1.0)))
for l in range(64):
    assert f"up_FL_{l}_64" not in A
    A[f"up_FL_{l}_64"] = A[f"k_{l}"].astype(np.float32).reshape(-1); A[f"down_FL_{l}_64"] = A[f"down_FJm_{l}"].astype(np.float32).reshape(-1, 1)
np.savez(out, **A); print("wrote", out, len(A), "keys (was", n0, ")", hashlib.sha256(open(out, "rb").read()).hexdigest())
