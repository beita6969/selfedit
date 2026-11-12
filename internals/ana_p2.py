"""ana_p2 (fable_int, P2 step 1): does a late boost of attention to the image tokens create NEW reading positions, or does it re-amplify positions that already depended on the image?
From the records of p2_reopen: per knob, thinking positions at or after T_on. A reading event = KL(p_real || p_blank) >= 1. The events under the knob are split by the SAME position's value
without the knob: already an event (>= 1), near (0.5-1), weak (0.1-0.5), none (< 0.1). Also: rank correlation of the per-position image dependence with and without the knob, the mean
multiplier of the dependence by level, events lost, and what the knob does to the real-image run at each class of position (KL(p_X || p_X,knob), change of log p(read token)).
python ana_p2.py DIR [T_on]"""
import glob, os, sys
import numpy as np
R = [np.load(f, allow_pickle=True) for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.npz")))]; TON = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
names = R[0]["names"].tolist()


def rank(x):
    o = np.argsort(x, kind="stable"); r = np.empty(len(x)); r[o] = np.arange(len(x)); return r


print(f"{len(R)} reads ({sum(not bool(r['ok']) for r in R)} ended wrong): " + ", ".join(f"{str(r['bench'])} {int(r['te'])}" for r in R))
early = [1000 * (r["kl0"][:TON] >= 1).mean() for r in R]; late0 = [1000 * (r["kl0"][TON:] >= 1).mean() for r in R]
print(f"reading events per 1,000 thinking tokens without the knob: before token {TON}: {np.mean(early):.1f} (first 150 tokens: {np.mean([1000 * (r['kl0'][:150] >= 1).mean() for r in R]):.1f}); from token {TON}: {np.mean(late0):.1f}; mean KL before / after: {np.mean([r['kl0'][:TON].mean() for r in R]):.3f} / {np.mean([r['kl0'][TON:].mean() for r in R]):.3f}")
for n in names:
    tot = dict(pos=0, on=0, e0=0, eg=0, a=0, near=0, weak=0, none=0, lost=0)
    rho, mult, per = [], {b: [] for b in ("<0.01", "0.01-0.1", "0.1-0.5", "0.5-1", ">=1")}, []
    dist = {c: [] for c in ("new", "old", "rest")}; dlp = {c: [] for c in ("new", "old", "rest")}
    for r in R:
        k0, kg, on = r["kl0"][TON:], r[f"{n}/kl"][TON:], r[f"{n}/on"][TON:].astype(bool)
        d, l = r[f"{n}/kl_x0g"][TON:], r[f"{n}/lpy"][TON:] - r["lpy0"][TON:]
        k0, kg, d, l = k0[on], kg[on], d[on], l[on]                    # positions where the knob is on
        if len(k0) < 20:
            continue
        ev = kg >= 1; new = ev & (k0 < 0.5); old = ev & (k0 >= 1)
        tot["pos"] += len(r["kl0"][TON:]); tot["on"] += len(k0); tot["e0"] += int((k0 >= 1).sum()); tot["eg"] += int(ev.sum())
        tot["a"] += int(old.sum()); tot["near"] += int((ev & (k0 >= 0.5) & (k0 < 1)).sum()); tot["weak"] += int((ev & (k0 >= 0.1) & (k0 < 0.5)).sum()); tot["none"] += int((ev & (k0 < 0.1)).sum())
        tot["lost"] += int(((k0 >= 1) & (kg < 0.5)).sum())
        rho.append(np.corrcoef(rank(k0), rank(kg))[0, 1])
        for b, (lo, hi) in zip(mult, ((0, 0.01), (0.01, 0.1), (0.1, 0.5), (0.5, 1), (1, 1e9))):
            m = (k0 >= lo) & (k0 < hi)
            if m.sum() >= 5:
                mult[b].append(kg[m].mean() / max(1e-9, k0[m].mean()))
        for c, m in (("new", new), ("old", old), ("rest", ~ev)):
            if m.sum():
                dist[c].append(d[m].mean()); dlp[c].append(l[m].mean())
        per.append((1000 * (k0 >= 1).mean(), 1000 * ev.mean(), 1000 * new.mean(), 1000 * (ev & (k0 < 0.1)).mean()))
    if not tot["on"]:
        continue
    per = np.array(per); add = tot["eg"] - tot["e0"]
    print(f"\nknob {n}: positions from token {TON}: {tot['pos']}, knob on at {tot['on']} ({tot['on'] / tot['pos']:.2f})")
    print(f"  events per 1,000 knob-on positions (mean over reads): without {per[:, 0].mean():.1f} -> with {per[:, 1].mean():.1f}; new (>= 1 with, < 0.5 without) {per[:, 2].mean():.1f}; from nothing (< 0.1 without) {per[:, 3].mean():.1f}")
    print(f"  the {tot['eg']} events under the knob by the same position's value without it: already an event {tot['a']} ({tot['a'] / max(1, tot['eg']):.2f}), near 0.5-1 {tot['near']} ({tot['near'] / max(1, tot['eg']):.2f}), "
          f"weak 0.1-0.5 {tot['weak']} ({tot['weak'] / max(1, tot['eg']):.2f}), none < 0.1 {tot['none']} ({tot['none'] / max(1, tot['eg']):.2f}); events lost (>= 1 without, < 0.5 with) {tot['lost']}; net added {add}")
    print(f"  rank correlation of the image dependence per position, with against without the knob: {np.mean(rho):.3f} (min {np.min(rho):.3f}); mean multiplier of the dependence by its level without the knob: " + ", ".join(f"{b} x{np.mean(v):.2f}" for b, v in mult.items() if v))
    print("  effect on the real-image run, KL(p_X || p_X,knob) / change of log p(read token): " + "; ".join(f"{c} events {np.mean(dist[c]):.3f} / {np.mean(dlp[c]):+.3f}" if c != "rest" else f"non-events {np.mean(dist[c]):.4f} / {np.mean(dlp[c]):+.4f}" for c in dist if dist[c]))
print("done")
