"""keyz_bench (fable_int): the route key's value on stream positions, by benchmark, from the key-capture records (kcap: MLP inputs at sampled positions). Routed rows (label of the fit:
streams of text-generated reads run without the text) and unrouted rows; blocks 27, 43, 59; mean, sd, share above 0.35 and above 0.5, and the mean unit factor of a sharpened gate.
python keyz_bench.py ARRAYS.npz KCAP_ROUTED KCAP_UNROUTED [s:theta ...]"""
import glob, os, sys, collections
import numpy as np
A = np.load(sys.argv[1]); G = [tuple(float(x) for x in v.split(":")) for v in sys.argv[4:]]
silu = lambda g: g / (1 + np.exp(-np.clip(g, -60, 60)))
for name, d in (("routed rows", sys.argv[2]), ("unrouted rows", sys.argv[3])):
    by = collections.defaultdict(lambda: collections.defaultdict(list)); nr = collections.Counter()
    for f in sorted(glob.glob(os.path.join(d, "*.npz"))):
        z = np.load(f, allow_pickle=True); st = z["cls"] >= 3; b = str(z["row"]).split("/")[0]; nr[b] += 1
        for l in (27, 43, 59):
            by[b][l].append(z["X"][l][st].astype(np.float32) @ A[f"kr_{l}"].astype(np.float32).reshape(-1))
    print(f"\n{name}: reads per benchmark {dict(nr)}")
    print("benchmark | block: mean (sd) of the key, share > 0.35, share > 0.5, linear-gate factor silu(8 z) / 8" + "".join(f", sharpened s={s:g} theta={t:g}" for s, t in G))
    for b in sorted(by):
        for l in (27, 43, 59):
            v = np.concatenate(by[b][l])
            print(f"  {b} | {l}: {v.mean():+.2f} ({v.std():.2f}), {np.mean(v > 0.35):.2f}, {np.mean(v > 0.5):.2f}, {np.mean(silu(8 * v)) / 8:+.3f}" + "".join(f", {np.mean(silu(s * (v - t))) / 8:+.3f}" for s, t in G))
