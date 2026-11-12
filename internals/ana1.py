"""ana1: restoration tables from a1 (or b) records. R = 1 - sum KL(x||cond) / sum KL(x||m) per trace, mean and se over traces; by thinking-token index bin and at paragraph starts."""
import argparse, glob, os, sys
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("dir"); ap.add_argument("--b", action="store_true", help="b_xb records"); ap.add_argument("--only", default="")
a = ap.parse_args()
R = [np.load(f, allow_pickle=True) for f in sorted(glob.glob(os.path.join(a.dir, "*.npz")))]
NN = 271  # "\n\n" id is resolved below from the data when possible
print(f"{len(R)} records")
EDG = [0, 8, 32, 150, 400, 1000, 10 ** 9] if os.environ.get("FI_FINE") else [0, 150, 400, 1000, 10 ** 9]
names = [n for n in R[0]["names"].tolist() if n not in ("m", "x")]
if a.only:
    names = [n for n in names if any(n.startswith(p) for p in a.only.split(","))]


def base(r):
    return r["m/kl_xc"] if a.b else r["kl_xm"]


def masks(r):
    n = len(base(r)); ix = np.arange(n)
    ms = {"all": np.ones(n, bool)}
    for j in range(len(EDG) - 1):
        ms[f"t{EDG[j]}-{EDG[j + 1] if EDG[j + 1] < 10 ** 8 else ''}"] = (ix >= EDG[j]) & (ix < EDG[j + 1])
    if a.b:
        ms["pstart"] = r["pstart"].astype(bool)
    else:
        y = r["y"]; ps = np.zeros(n, bool); ps[0] = True
        nn = int(y[r["bnd"]][np.argmax(np.bincount(y[r["bnd"]]).take(y[r["bnd"]]))]) if len(r["bnd"]) else -1     # the commoner of the two newline tokens = "\n\n"
        ps[1:] = y[:-1] == nn; ms["pstart"] = ps
    ms["other"] = ~ms["pstart"]
    return ms


keys = None; tab = {}
for r in R:
    ms = masks(r); keys = list(ms); b = base(r)
    for n in names:
        kc = r[f"{n}/kl_xc"]
        for k_, m_ in ms.items():
            if m_.sum() and b[m_].sum() > 0:
                tab.setdefault((n, k_), []).append(1 - kc[m_].sum() / b[m_].sum())
bk = {k_: np.mean([base(r)[masks(r)[k_]].mean() for r in R if masks(r)[k_].sum()]) for k_ in keys}
print("mean KL(x||m) by subset: " + "  ".join(f"{k_} {v:.4f}" for k_, v in bk.items()))
print(f"{'condition':18s} " + " ".join(f"{k_:>14s}" for k_ in keys))
for n in names:
    print(f"{n:18s} " + " ".join((f"{np.mean(tab[(n, k_)]):+.3f} ±{np.std(tab[(n, k_)]) / np.sqrt(len(tab[(n, k_)])):.3f}" if (n, k_) in tab else " " * 14) for k_ in keys))
if not a.b:
    print("\nfinal-residual projection (linear measure) and stop logit difference at paragraph ends:")
    for n in names:
        rp = [float(r[f"{n}/rproj"]) for r in R]
        ld = [(r[f"{n}/ld"] - r["ld_m"]).mean() for r in R if len(r["bnd"])]
        ldx = [(r["ld_x"] - r["ld_m"]).mean() for r in R if len(r["bnd"])]
        print(f"{n:18s} rproj {np.mean(rp):+.3f} ±{np.std(rp) / np.sqrt(len(rp)):.3f} | d(lp(nl)-lp(nlnl)) cond {np.mean(ld):+.3f}  xhigh {np.mean(ldx):+.3f}")

if a.b:
    print("\nxhigh-axis dose gamma = <c-m, x-m>/<x-m, x-m> and alignment cos (p_m-weighted log-prob shifts), KL(m||c), KL(c||m); all positions | paragraph starts")
    for n in names:
        if f"{n}/ip_cx" not in R[0]:
            continue
        rows = []
        for sub in ("all", "pstart"):
            g, c_, k1, k2 = [], [], [], []
            for r in R:
                m_ = masks(r)[sub]
                xx = r["x/ip_cx"][m_].sum(); cx_ = r[f"{n}/ip_cx"][m_].sum(); cc = r[f"{n}/ip_cc"][m_].sum()
                g.append(cx_ / xx); c_.append(cx_ / np.sqrt(xx * cc + 1e-30)); k1.append(r[f"{n}/kl_mc"][m_].mean()); k2.append(r[f"{n}/kl_cm"][m_].mean())
            rows.append(f"gamma {np.mean(g):+.3f} ±{np.std(g) / np.sqrt(len(g)):.3f} cos {np.mean(c_):+.3f} KL(m||c) {np.mean(k1):.4f} KL(c||m) {np.mean(k2):.4f}")
        print(f"{n:22s} " + " | ".join(rows))
