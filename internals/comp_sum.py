"""comp_sum (fable_int, law table): one component's row. From the patching records (a1): size of the contrast, attention-cut survival, mixers, depth profile; from the cell evaluation (ev):
restoration R by rank and band, the best cell with <= 64 units per block, the xhigh-form cell (rank 8, blocks 24-34), and the controls. python comp_sum.py NAME A1_DIR EV_DIR"""
import glob, os, sys
import numpy as np
name, a1, ev = sys.argv[1:4]
A = [np.load(f, allow_pickle=True) for f in sorted(glob.glob(os.path.join(a1, "*.npz")))]; E = [np.load(f, allow_pickle=True) for f in sorted(glob.glob(os.path.join(ev, "*.npz")))]


def ra(n):
    v = [1 - r[f"{n}/kl_xc"].sum() / r["kl_xm"].sum() for r in A if f"{n}/kl_xc" in r.files and r["kl_xm"].sum() > 0]
    return (np.mean(v), np.std(v) / np.sqrt(len(v))) if v else (np.nan, np.nan)


def re_(n):
    v = [1 - r[f"{n}/kl_xc"].sum() / r["m/kl_xc"].sum() for r in E if f"{n}/kl_xc" in r.files and r["m/kl_xc"].sum() > 0]
    g = sum(r[f"{n}/ip_cx"].sum() for r in E) / max(1e-12, sum(r["x/ip_cx"].sum() for r in E)) if E and f"{n}/ip_cx" in E[0].files else np.nan
    return (np.mean(v), np.std(v) / np.sqrt(len(v)), g) if v else (np.nan, np.nan, np.nan)


print(f"component {name}: patching reads {len(A)}, evaluation reads {len(E)}; inserted block {sorted({int(r['S']) for r in E})[:5] if E and 'S' in E[0].files else '?'} tokens")
if A:
    print(f"  KL(A || B) per thinking position {np.mean([r['kl_xm'].mean() for r in A]):.4f}")
    print("  survival of the attention cut: stream queries, all full-attention blocks %.3f +- %.3f | every later position %.3f +- %.3f | blocks 3-31 only %.3f | 35-63 only %.3f" % (*ra("xcut_all"), *ra("xcutP_all"), ra("xcut_lo")[0], ra("xcut_hi")[0]))
    print("  mixers alone: linear blocks %.3f +- %.3f | full-attention blocks %.3f +- %.3f" % (*ra("mixlin"), *ra("mixfull")))
    print("  all positions swapped after block 7 / 15 / 23 / 31 / 39 / 47 / 55: " + " / ".join(f"{ra(f'res{l}_all')[0]:.2f}" for l in (7, 15, 23, 31, 39, 47, 55)))
if E:
    names = [n for n in E[0]["names"].tolist() if n not in ("m", "x")]
    print("  restoration R on held-out reads (mean +- se over reads; gamma): " + "; ".join(f"{n} {re_(n)[0]:+.3f} +- {re_(n)[1]:.3f} ({re_(n)[2]:.2f})" for n in names))
    real = [n for n in names if "rot" not in n]; best = max(real, key=lambda n: re_(n)[0])
    print(f"  LAW ROW | {name} | survival {ra('xcut_all')[0]:.3f} | all-cut {ra('xcutP_all')[0]:.3f} | linear mixers {ra('mixlin')[0]:.3f} | full mixers {ra('mixfull')[0]:.3f} | best R (<= 64 units per block) {re_(best)[0]:.3f} [{best}] | "
          f"R rank 8 blocks 24-34 {re_('r8_L24-34')[0]:.3f} | R rank 64 all blocks {re_('r64_all')[0]:.3f} | controls: rotated write {re_('r64_all_wrot')[0]:+.3f}, random read {re_('r64_all_rrot')[0]:+.3f}")
