"""From a stop-read scan, the cheapest combinations of single-layer edits for a given stop level. Stop is about linear in alpha and additive across layers; drift about quadratic. Minimising the sum of q_i a_i^2 under the sum of s_i a_i = T gives a_i proportional to s_i / q_i.
Usage: python rsi/allocate.py RUN/hazard.jsonl OUT.json --stops 3,4,5 --layers 2,3,4,6 [--cap 1.5] [--dirs late_vs_early_correct,long_vs_short_correct]"""
import argparse, json
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("scan"); ap.add_argument("out")
ap.add_argument("--stops", default="3,4,5"); ap.add_argument("--layers", default="2,3,4,6")
ap.add_argument("--cap", type=float, default=1.5); ap.add_argument("--dirs", default=""); ap.add_argument("--min-z", type=float, default=3.0); ap.add_argument("--min-stop", type=float, default=0.25)
a = ap.parse_args()
best = {}
for l in open(a.scan):
    r = json.loads(l)
    e = r["edit"]
    if "u" not in e or "param" not in e or (a.dirs and e["u"].split(":")[1] not in a.dirs.split(",")):
        continue
    s, q = r["d_cut"] / e["alpha"], max(-r["d_all"], 1e-5) / e["alpha"] ** 2
    if abs(r["d_cut"]) < a.min_z * r["se_cut"] or abs(s) < a.min_stop:    # a tiny read with a tinier drift is noise, not efficiency
        continue
    L = int(e["param"].split(".")[1])
    if L not in best or s ** 2 / q > best[L]["eff"]:     # one direction per layer: those of a layer overlap (cos .82-.90)
        best[L] = {"L": L, "param": e["param"], "u": e["u"], "s": s, "q": q, "eff": s ** 2 / q}
rank = sorted(best.values(), key=lambda x: -x["eff"])
print("layers by stop^2 per drift:")
for x in rank[:14]:
    print(f"  L{x['L']} {x['u'].split(':')[1][:22]} stop per alpha {x['s']:+.2f}, drift per alpha^2 {x['q']:.4f}, efficiency {x['eff']:.0f}")
out = []
for T in map(float, a.stops.split(",")):
    for k in map(int, a.layers.split(",")):
        P = rank[:k]
        if len(P) < k:
            continue
        s, q = np.array([x["s"] for x in P]), np.array([x["q"] for x in P])
        lam = T / (s ** 2 / q).sum()
        al = np.clip(lam * s / q, -a.cap, a.cap)
        for _ in range(20):                               # clipped layers keep the cap; the others share what is left
            free = np.abs(al) < a.cap
            if not free.any():
                break
            al[free] = (T - (s * al)[~free].sum()) / (s[free] ** 2 / q[free]).sum() * (s / q)[free]
            al = np.clip(al, -a.cap, a.cap)
        pred, drift = float((s * al).sum()), float((q * al ** 2).sum())
        parts = [{"op": "amplify", "param": x["param"], "side": "out", "u": x["u"], "alpha": round(float(v), 3)} for x, v in zip(P, al) if abs(v) >= 0.05]
        label = f"stop {pred:.2f} over {len(parts)} layers: " + " ".join(f"L{p['param'].split('.')[1]} {p['alpha']:+.2f}" for p in parts)
        print(label, f"predicted drift {drift:.4f}")
        out.append({"label": label, "predicted": {"stop": pred, "drift": drift}, "combo": parts})
json.dump(out, open(a.out, "w"))
