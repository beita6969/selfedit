"""Noise ratio per field: mean sq diff between realisations (dirs) on the same trace / mean sq diff between samples of the same row within a dir.
python noise_ratio.py DIR1 DIR2 [DIR3 ...]   (records of the same traces in every dir)"""
import glob, itertools, os, sys
import numpy as np
dirs = sys.argv[1:]
names = sorted(set.intersection(*[set(map(os.path.basename, glob.glob(os.path.join(d, "*.npz")))) for d in dirs]))
R = {d: {n: np.load(os.path.join(d, n)) for n in names} for d in dirs}
rows = {}
for n in names:
    rows.setdefault(n.rsplit("__s", 1)[0], []).append(n)
F = [("dT", lambda r: np.atleast_1d(r["dT"])), ("g_attn", lambda r: r["g_attn"]), ("g_mlp", lambda r: r["g_mlp"]), ("g_qtemp", lambda r: r["g_qtemp"]),
     ("g_merger", lambda r: r["g_merger"]), ("bias_rp_gen", lambda r: r["bias_rp"][0]), ("bias_rp_img", lambda r: r["bias_rp"][1]), ("bias_rp_txt", lambda r: r["bias_rp"][2]),
     ("bias_full_gen", lambda r: r["bias_full"][0]), ("bias_full_img", lambda r: r["bias_full"][1]), ("bias_full_txt", lambda r: r["bias_full"][2]), ("logp", lambda r: np.atleast_1d(r["logp"]))]
print(f"{len(names)} traces x {len(dirs)} realisations; rows {len(rows)}")
print(f"{'field':14s} {'noise msd':>11s} {'between-sample msd':>19s} {'ratio':>7s}  (per-field ratio; noise/between of 1 = all noise)")
for f, get in F:
    noise = [np.mean((get(R[a][n]) - get(R[b][n])) ** 2) for n in names for a, b in itertools.combinations(dirs, 2)]
    betw = [np.mean((get(R[d][x]) - get(R[d][y])) ** 2) for d in dirs for ns in rows.values() for x, y in itertools.combinations(ns, 2)]
    # noise variance appears twice in both diffs, so ratio = noise var / (noise var + true between-sample var)
    print(f"{f:14s} {np.mean(noise):11.4g} {np.mean(betw):19.4g} {np.mean(noise) / np.mean(betw):7.3f}")
