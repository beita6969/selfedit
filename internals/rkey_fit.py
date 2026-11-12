"""rkey_fit (fable_int, consolidation): route key of a routed procedure, per block, from the state alone.
Ridge on the MLP input (kcap records): target 1 at stream positions (template tail, thinking, answer) of rows where the entry is routed, run WITHOUT the procedure text;
0 at stream positions of unrouted rows and at image / prompt positions of every row. The three groups are weighted equally. Rows with crc32(row) % 5 == 0 are never used in the
fit (they are the evaluation rows): the printed statistics are on those rows. Saved: kr_l = w + b * k_l (bias through the constant key), merged with the arrays of FIT.npz into OUT.npz.
python rkey_fit.py KCAP_ROUTED KCAP_UNROUTED FIT.npz OUT.npz"""
import glob, hashlib, os, sys, zlib
import numpy as np, torch
kr_dir, ku_dir, fitf, outf = sys.argv[1:5]
assert not os.path.exists(outf), "output exists: choose a new name"
dev = "cuda:0" if torch.cuda.is_available() else "cpu"
A = dict(np.load(fitf))


def load(d, lab):
    out = []
    for f in sorted(glob.glob(os.path.join(d, "*.npz"))):
        z = np.load(f, allow_pickle=True); out.append(dict(X=z["X"], cls=z["cls"], row=str(z["row"]), lab=lab))
    return out


R = load(kr_dir, 1) + load(ku_dir, 0)
cls = np.concatenate([r["cls"] for r in R]); lab = np.concatenate([np.full(len(r["cls"]), r["lab"]) for r in R]); rid = np.concatenate([np.full(len(r["cls"]), i) for i, r in enumerate(R)])
held = np.array([zlib.crc32(r["row"].encode()) % 5 == 0 for r in R])[rid]
stream = cls >= 3; grp = np.where(~stream, 2, np.where(lab == 1, 0, 1))          # 0 routed stream, 1 unrouted stream, 2 context
y = (grp == 0).astype(np.float32); tr = ~held
w = np.zeros(len(y), np.float32)
for g in range(3):
    m = tr & (grp == g); w[m] = 1.0 / 3 / max(1, m.sum())
print(f"traces: routed {sum(r['lab'] for r in R)}, unrouted {sum(1 - r['lab'] for r in R)}; held-out rows: routed {len({r['row'] for r in R if r['lab'] and zlib.crc32(r['row'].encode()) % 5 == 0})}, unrouted {len({r['row'] for r in R if not r['lab'] and zlib.crc32(r['row'].encode()) % 5 == 0})}; "
      f"training positions: routed stream {int((tr & (grp == 0)).sum())}, unrouted stream {int((tr & (grp == 1)).sum())}, context {int((tr & (grp == 2)).sum())}")
print("block | held-out rows: key on routed stream mean (sd) | on unrouted stream mean (sd), share > 0.5 | on context mean (sd) | AUC routed vs unrouted stream positions | per row (mean over the row's stream): routed min / median, unrouted median / max")
yt = torch.tensor(y, device=dev); wt = torch.tensor(w, device=dev)


def auc(a, b):
    v = np.concatenate([a, b]); r = np.argsort(np.argsort(v)) + 1.0
    return (r[: len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b))


for l in range(64):
    X = torch.tensor(np.concatenate([r["X"][l].astype(np.float32) for r in R]), device=dev)
    Xt = X[torch.tensor(tr, device=dev)]; wtt = wt[torch.tensor(tr, device=dev)]; ytt = yt[torch.tensor(tr, device=dev)]
    mu = (Xt * wtt[:, None]).sum(0); ym = (ytt * wtt).sum(); Xc = Xt - mu; G = (Xc * wtt[:, None]).T @ Xc
    wv = torch.linalg.solve(G + 1e-3 * G.diagonal().mean() * torch.eye(G.shape[0], device=dev), (Xc * wtt[:, None]).T @ (ytt - ym)); b = float(ym - mu @ wv)
    p = (X @ wv + b).cpu().numpy(); h = held
    a_, u_, c_ = p[h & (grp == 0)], p[h & (grp == 1)], p[h & (grp == 2)]
    pr = [p[(rid == i) & stream].mean() for i, r in enumerate(R) if r["lab"] == 1 and zlib.crc32(r["row"].encode()) % 5 == 0]; pu = [p[(rid == i) & stream].mean() for i, r in enumerate(R) if r["lab"] == 0 and zlib.crc32(r["row"].encode()) % 5 == 0]
    A[f"kr_{l}"] = (wv.cpu().numpy() + b * A[f"k_{l}"].reshape(-1)).astype(np.float32)
    print(f"{l:2d}{'*' if l % 4 == 3 else ' '} | {a_.mean():.2f} ({a_.std():.2f}) | {u_.mean():.2f} ({u_.std():.2f}), {(u_ > 0.5).mean():.3f} | {c_.mean():+.2f} ({c_.std():.2f}) | {auc(a_, u_):.3f} | {min(pr):.2f} / {np.median(pr):.2f}, {np.median(pu):.2f} / {max(pu):.2f}", flush=True)
    del X, Xt, G
np.savez(outf, **A); print("wrote", outf, len(A), "keys", hashlib.sha256(open(outf, "rb").read()).hexdigest())
