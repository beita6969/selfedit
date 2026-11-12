"""kcap (fable_int): MLP inputs (post-attention norm output) at sampled context and stream positions of medium-template runs, every block; for position keys.
OUT/<row>__s<k>.npz: X [64, m, H] f16, cls [m] (1 img, 2 txt, 3 tail, 4 thinking, 5 answer), tix (thinking-token index or -1)."""
import argparse, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import intlib
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True); ap.add_argument("--shard", default="0/1"); ap.add_argument("--per-bench", type=int, default=7); ap.add_argument("--seed", type=int, default=1111)
ap.add_argument("--f32", action="store_true"); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--row-ids", default=""); ap.add_argument("--stop", type=int, default=64); ap.add_argument("--rows", default="", help="file of \"row sample\" lines")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
lab = intlib.Lab(f32=a.f32)
ST, NL, H, dev = lab.ST, lab.NL, lab.H, lab.dev
items = [(l.split()[0], int(l.split()[1])) for l in open(a.rows) if l.strip()] if a.rows else (intlib.items_from_ids(lab.D, a.row_ids) if a.row_ids else intlib.pick_rows(lab.D, per_bench=a.per_bench, seed=a.seed))
si, sn = (int(x) for x in a.shard.split("/")); items = items[si::sn]
if a.limit:
    items = items[: a.limit]
for it, (rid, k) in enumerate(items):
    fn = os.path.join(a.out, f"{rid.replace('/', '__')}__s{k}.npz")
    if os.path.exists(fn):
        continue
    t0 = time.time()
    gen, te = lab.gen_ids(rid, k)
    c = lab.prep(rid, "medium", gen); P, T = c.P, c.T
    cl, cvec = lab.classes(c)
    g = torch.Generator().manual_seed(a.seed + it)
    pick = lambda idx, n: idx[torch.randperm(len(idx), generator=g)[:n].sort().values.to(idx.device)]
    think = torch.arange(P, P + te, device=dev); ans = torch.arange(P + te, T, device=dev)
    sel = torch.cat([pick(cl["img"], 128), pick(cl["txt"], 64), cl["tail"], pick(think, 128), pick(ans, 16)])
    cls = torch.cat([torch.full((min(128, len(cl["img"])),), 1), torch.full((min(64, len(cl["txt"])),), 2), torch.full((5,), 3), torch.full((min(128, te),), 4), torch.full((min(16, len(ans)),), 5)])
    tix = torch.where((sel >= P) & (sel < P + te), sel - P, torch.full_like(sel, -1))
    ST["cap"] = dict(mlpin=lambda l, o: o[0][sel].float().cpu()); ST["out"] = {}
    lab.run(c, stop=a.stop)
    X = np.zeros((NL, len(sel), H), np.float16); X[: a.stop] = torch.stack([ST["out"][("mlpin", l)] for l in range(a.stop)]).numpy().astype(np.float16)
    lab.clear()
    np.savez(fn, X=X, cls=cls.numpy(), tix=tix.cpu().numpy(), row=rid, sample=k)
    print(f"[{it + 1}/{len(items)}] {rid} s{k} T {T} sel {len(sel)} {time.time() - t0:.0f}s", flush=True)
print("done", flush=True)
