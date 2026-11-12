"""b (fable_int): what XB does to the next-token distribution under teacher forcing, layer 63 vs the rest, readout vs state.
Conditions on the MEDIUM template (identical tokens): edit specs applied to the weights exactly as deployed (--specs name=path ...), and
residual-add conditions from direction files (--dirs name=npz:key ... ; d [64, H] cumulative mean difference after each block):
  <name>_all   inc_L added after every block at every position       <name>_gen   only at tail + generated positions
  <name>_prm   only at prompt positions (image + text)               <name>_fin   the whole cumulative offset added after block 63 only (pure readout shift)
Per condition and prediction position: kl_xc, kl_mc, lpy (log p of the transcript's next token), lp_nl, lp_nn; dmass (sum of p_c - p_m) over all
positions and over paragraph starts. References m, x. float32.
"""
import argparse, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import intlib, edits
ap = argparse.ArgumentParser()
ap.add_argument("--mode", default="xhigh", help="template mode contrasted with medium (xhigh | low)"); ap.add_argument("--out", required=True); ap.add_argument("--shard", default="0/1"); ap.add_argument("--per-bench", type=int, default=4)
ap.add_argument("--bf16", action="store_true"); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--seed", type=int, default=2222)
ap.add_argument("--rows", default=""); ap.add_argument("--max-think", type=int, default=1800)
ap.add_argument("--specs", nargs="*", default=[]); ap.add_argument("--dirs", nargs="*", default=[]); ap.add_argument("--variants", default="all,gen,prm,fin")
ap.add_argument("--remap", nargs="*", default=[], help="npz path remaps old=new")
ap.add_argument("--lens", action="store_true"); ap.add_argument("--row-ids", default=""); ap.add_argument("--stride", type=int, default=1, help="evaluate every N-th prediction position (long reads)")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
lab = intlib.Lab(f32=not a.bf16)
ST, NL, H, dev = lab.ST, lab.NL, lab.H, lab.dev
ed = edits.Editor(lab, dict(x.split("=", 1) for x in a.remap))
SPECS = {x.split("=", 1)[0]: json.load(open(x.split("=", 1)[1])) for x in a.specs}
DIRS = {}
for x in a.dirs:
    name, src = x.split("=", 1); path, key = src.rsplit(":", 1)
    d = np.load(path)[key].astype(np.float32)                     # [64, H] cumulative offset after each block
    DIRS[name] = torch.tensor(np.concatenate([d[:1], np.diff(d, axis=0)]), device=dev)       # increments
if a.row_ids:
    items = intlib.items_from_ids(lab.D, a.row_ids, gen_hi=a.max_think)
elif a.rows:
    items = [(l.split()[0], int(l.split()[1])) for l in open(a.rows) if l.strip()]
else:
    excl = {r for r, _ in intlib.pick_rows(lab.D, per_bench=8, seed=1111)}
    items = intlib.pick_rows({r: v for r, v in lab.D.items() if r not in excl}, per_bench=a.per_bench, seed=a.seed, gen_hi=a.max_think)
    items = [items[i] for b in range(a.per_bench) for i in range(b, len(items), a.per_bench)]     # round-robin over benchmarks
si, sn = (int(x) for x in a.shard.split("/"))
items = items[si::sn]
if a.limit:
    items = items[: a.limit]
NLID = lab.tk("\n", add_special_tokens=False).input_ids[0]; NNID = lab.tk("\n\n", add_special_tokens=False).input_ids[0]
print(f"{len(items)} traces; specs {list(SPECS)}; dirs {list(DIRS)}", flush=True)
if a.lens:
    W = lab.model.lm_head.weight; g = 1.0 + lab.lm.norm.weight.float()
    for name, inc in DIRS.items():
        for tag, v in (("inc_63", inc[63]), ("d_final", inc.sum(0)), ("d_62 (all but L63)", inc[:63].sum(0))):
            z = (W.float() @ (g * v))
            top = z.topk(30).indices.tolist(); bot = (-z).topk(30).indices.tolist()
            print(f"[lens] {name} {tag} |v| {float(v.norm()):.2f}: UP " + " ".join(repr(lab.tk.decode([t])) for t in top), flush=True)
            print(f"[lens] {name} {tag}: DOWN " + " ".join(repr(lab.tk.decode([t])) for t in bot), flush=True)

for it, (rid, k) in enumerate(items):
    fn = os.path.join(a.out, f"{rid.replace('/', '__')}__s{k}.npz")
    if os.path.exists(fn):
        continue
    t0 = time.time()
    gen, te = lab.gen_ids(rid, k)
    cm = lab.prep(rid, "medium", gen); cx = lab.prep(rid, a.mode, gen)
    A, S = lab.align(cm, cx); P, T = cm.P, cm.T
    npred = min(te + 1, len(gen))
    pm = torch.arange(P - 1, P - 1 + npred, device=dev); y = torch.tensor(gen[:npred], device=dev)
    pstart = torch.zeros(npred, dtype=torch.bool, device=dev); pstart[1:] = y[:-1] == NNID; pstart[0] = True        # the last token read is "\n\n" (or <think>\n)
    pidx = torch.arange(npred, device=dev)
    if a.stride > 1:
        keep_ = pidx[:: a.stride]; pm, y, pstart, pidx = pm[keep_], y[keep_], pstart[keep_], keep_
    lpm = lab.logp(lab.run(cm)[0][pm]); lpx = lab.logp(lab.run(cx)[0][pm + S])
    p_m = lpm.exp(); dxm = lpx - lpm
    out = dict(row=rid, sample=k, ok=bool(lab.D[rid]["samples"][str(k)]["ok"]), bench=lab.D[rid]["bench"], y=y.cpu().numpy(), pstart=pstart.cpu().numpy(), te=te, P=P, T=T, pidx=pidx.cpu().numpy())

    def record(name, lpc):
        out[f"{name}/kl_xc"] = intlib.kl(lpx, lpc).cpu().numpy(); out[f"{name}/kl_mc"] = intlib.kl(lpm, lpc).cpu().numpy(); out[f"{name}/kl_cm"] = intlib.kl(lpc, lpm).cpu().numpy()
        out[f"{name}/lpy"] = lpc.gather(1, y[:, None])[:, 0].cpu().numpy(); out[f"{name}/lp_nl"] = lpc[:, NLID].cpu().numpy(); out[f"{name}/lp_nn"] = lpc[:, NNID].cpu().numpy()
        dp = lpc.exp() - p_m
        out[f"{name}/dmass"] = dp.sum(0).cpu().numpy().astype(np.float32); out[f"{name}/dmass_ps"] = dp[pstart].sum(0).cpu().numpy().astype(np.float32)
        out[f"{name}/ent"] = (-(lpc.exp() * lpc).sum(-1)).cpu().numpy()
        dc = lpc - lpm                                                   # p_m-weighted inner products with the xhigh shift: dose along the xhigh axis and alignment
        out[f"{name}/ip_cx"] = (p_m * dc * dxm).sum(-1).cpu().numpy(); out[f"{name}/ip_cc"] = (p_m * dc * dc).sum(-1).cpu().numpy()
    record("m", lpm); record("x", lpx)
    names = ["m", "x"]
    for name, spec in SPECS.items():
        ed.apply(spec)
        try:
            lpc = lab.logp(lab.run(cm)[0][pm])
        finally:
            ed.revert()
        record(name, lpc); names.append(name); del lpc
    for name, inc in DIRS.items():
        for var in a.variants.split(","):
            if var == "fin":
                rp = {63: (lambda h, v=inc.sum(0): h + v.to(h.dtype))}
            else:
                msk = torch.ones(T, 1, device=dev)
                if var == "gen":
                    msk[: P - 5] = 0
                elif var == "prm":
                    msk[P - 5:] = 0
                rp = {l: (lambda h, v=inc[l], m_=msk: h + (m_ * v).to(h.dtype)[None]) for l in range(NL)}
            lpc = lab.logp(lab.run(cm, res_patch=rp)[0][pm])
            record(f"{name}_{var}", lpc); names.append(f"{name}_{var}"); del lpc
    out["names"] = np.array(names); out["sec"] = time.time() - t0
    np.savez(fn, **out)
    b = out["x/kl_mc"].mean()      # KL(m || x)
    bx = out["m/kl_xc"].mean()     # KL(x || m)
    print(f"[{it + 1}/{len(items)}] {rid} s{k} think {te} KL(x||m) {bx:.4f} | R = 1 - KL(x||c)/KL(x||m), KL(m||c): " +
          " ".join(f"{n_} {1 - out[n_ + '/kl_xc'].mean() / bx:+.2f},{out[n_ + '/kl_mc'].mean():.3f}" for n_ in names[2:]) + f" | {time.time() - t0:.0f}s", flush=True)
    del lpm, lpx, p_m
    torch.cuda.empty_cache()
print("done", flush=True)
