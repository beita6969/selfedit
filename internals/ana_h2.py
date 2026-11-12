"""ana_h2 (fable_int, hindsight): b_fn3 records. Per cell, reads that ended wrong / right: R of the hindsight distribution (per read mean +- se, pooled), gamma along the gold-hint shift.
On wrong reads with a decoy reference (the same line with the read's own wrong answer): gamma along the decoy-hint shift, R toward the decoy, and gamma along the GOLD-SPECIFIC part of the
gold shift (the gold shift minus its projection on the decoy shift, per position): the share of "which answer is right" that the cell reproduces.
python ana_h2.py DIR [DIR ...]   (several folders are also pooled into one table at the end)"""
import glob, os, sys
import numpy as np


def table(R, title):
    names = [n for n in R[0]["names"].tolist() if n not in ("m", "x")]
    W = [r for r in R if not bool(r["ok"])]; G = [r for r in R if bool(r["ok"])]; WA = [r for r in W if "alt/ip_aa" in r.files]
    print(f"\n{title}: {len(R)} reads ({len(W)} ended wrong, {len(G)} right; {len(WA)} wrong reads with a decoy reference)")
    print(f"  KL(H || base) per position: wrong {np.mean([r['m/kl_xc'].mean() for r in W] or [np.nan]):.4f}, right {np.mean([r['m/kl_xc'].mean() for r in G] or [np.nan]):.4f}" +
          (f"; decoy on wrong reads: KL(decoy || base) {np.mean([r['alt/kl_am'].mean() for r in WA]):.4f}, KL(H || decoy) {np.mean([r['alt/kl_xa'].mean() for r in WA]):.4f}, cosine of the gold-hint and decoy-hint shifts {sum(r['alt/ip_xa'].sum() for r in WA) / np.sqrt(sum(r['x/ip_cx'].sum() for r in WA) * sum(r['alt/ip_aa'].sum() for r in WA)):+.3f}" if WA else ""))
    print("  cell | wrong: R mean +- se, pooled R, gamma gold | right: R mean +- se, pooled R, gamma | wrong reads with decoy: gamma gold, gamma decoy, R toward decoy (pooled), gamma on the gold-specific part | KL(base || cell) wrong / right")
    for n in names:
        o = []
        for S in (W, G):
            if not S:
                o.append("n/a"); continue
            r_ = np.array([1 - r[f"{n}/kl_xc"].mean() / max(1e-9, r["m/kl_xc"].mean()) for r in S]); pooled = 1 - sum(r[f"{n}/kl_xc"].sum() for r in S) / sum(r["m/kl_xc"].sum() for r in S)
            g = sum(r[f"{n}/ip_cx"].sum() for r in S) / max(1e-12, sum(r["x/ip_cx"].sum() for r in S))
            o.append(f"{r_.mean():+.3f} +- {(r_.std(ddof=1) / np.sqrt(len(S))) if len(S) > 1 else float('nan'):.3f}, {pooled:+.3f}, {g:+.3f}")
        if WA:
            gg = sum(r[f"{n}/ip_cx"].sum() for r in WA) / sum(r["x/ip_cx"].sum() for r in WA); gd = sum(r[f"{n}/ip_ca"].sum() for r in WA) / sum(r["alt/ip_aa"].sum() for r in WA)
            rd = 1 - sum(r[f"{n}/kl_ac"].sum() for r in WA) / sum(r["alt/kl_am"].sum() for r in WA)
            num = den = 0.0
            for r in WA:
                beta = r["alt/ip_xa"] / np.maximum(r["alt/ip_aa"], 1e-12); num += (r[f"{n}/ip_cx"] - beta * r[f"{n}/ip_ca"]).sum(); den += (r["x/ip_cx"] - beta * r["alt/ip_xa"]).sum()
            o.append(f"{gg:+.3f}, {gd:+.3f}, {rd:+.3f}, {num / max(den, 1e-12):+.3f}")
        else:
            o.append("n/a")
        print(f"  {n} | {o[0]} | {o[1]} | {o[2]} | {np.mean([r[f'{n}/kl_mc'].mean() for r in W] or [np.nan]):.4f} / {np.mean([r[f'{n}/kl_mc'].mean() for r in G] or [np.nan]):.4f}")


ALL = []
for d in sys.argv[1:]:
    R = [np.load(f, allow_pickle=True) for f in sorted(glob.glob(os.path.join(d, "*.npz")))]
    if R:
        table(R, d); ALL += R if os.path.basename(d.rstrip("/")) != "rows" else []
if len(sys.argv) > 2 and ALL:
    table(ALL, "POOLED over the leave-one-benchmark-out folds (every read evaluated with units fitted without its benchmark)")
