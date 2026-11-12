"""ana_same (fable_int): are the reads below the switch the candidate's reads? Per condition, difference to the reference condition on the same traces:
|delta log p(transcript token)| (mean, 99 %, max) and difference of KL(m||c).  python ana_same.py DIR REF_NAME"""
import glob, os, sys
import numpy as np
R = [np.load(f, allow_pickle=True) for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.npz")))]; ref = sys.argv[2]
print(f"{len(R)} traces; thinking tokens {[int(r['te']) for r in R]}; reference {ref}")
for n in [x for x in R[0]["names"].tolist() if x not in ("m", "x", ref)]:
    d = np.concatenate([np.abs(r[f"{n}/lpy"] - r[f"{ref}/lpy"]) for r in R]); e = np.concatenate([np.abs(r[f"{ref}/lpy"] - r["m/lpy"]) for r in R])
    g = [r[f"{n}/ip_cx"].sum() / r["x/ip_cx"].sum() for r in R]; g0 = [r[f"{ref}/ip_cx"].sum() / r["x/ip_cx"].sum() for r in R]
    print(f"{n}: |log p - log p_ref| mean {d.mean():.5f}, 99 % {np.quantile(d, .99):.4f}, max {d.max():.3f} (for scale: |ref - medium| mean {e.mean():.4f}); gamma {np.mean(g):.3f} vs ref {np.mean(g0):.3f}; R {np.mean([1 - r[n + '/kl_xc'].sum() / r['m/kl_xc'].sum() for r in R]):.3f} vs {np.mean([1 - r[ref + '/kl_xc'].sum() / r['m/kl_xc'].sum() for r in R]):.3f}")
