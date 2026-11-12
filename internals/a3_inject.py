"""a3 (fable_int): the instruction's FORCING TERM on the thinking stream, separated from the model's own response.
Stream = template tail (5 tokens) + generated positions. For block l:  J_l(t) = block_l^{xhigh context}(xhigh stream input)(t) - block_l^{medium context}(xhigh stream input)(t)
(hybrid run: positions before the stream take the medium run's residual, stream positions take the xhigh run's residual, medium template / positions).
Adding J_l(t) after every block at stream positions of a medium run reproduces the xhigh run exactly at stream positions (checked on the first trace),
so {J_l} is the complete, non-double-counted description of what the sentence does to the thinking stream; I_l = the same at the mixer output.
Also: per-position attention mass of the L55 / L59 heads on the instruction tokens, which instruction tokens get it, how much of the mixer difference the
direct instruction term explains, and where (which positions) the mass and the KL are high.
Per trace: OUT/<row>__s<k>.npz (stats) and OUT/samp/<row>__s<k>.npz (U [64, ns, H] f16 = MLP input of the hybrid run, J [64, ns, H] f16, at sampled stream positions).
"""
import argparse, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import intlib
ap = argparse.ArgumentParser()
ap.add_argument("--mode", default="xhigh", help="template mode contrasted with medium (xhigh | low)"); ap.add_argument("--out", required=True); ap.add_argument("--shard", default="0/1"); ap.add_argument("--per-bench", type=int, default=8); ap.add_argument("--seed", type=int, default=1111)
ap.add_argument("--bf16", action="store_true"); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--ns", type=int, default=256); ap.add_argument("--rows", default="")
ap.add_argument("--no-samp", action="store_true"); ap.add_argument("--gen-lo", type=int, default=300); ap.add_argument("--gen-hi", type=int, default=3000); ap.add_argument("--exclude-a0", action="store_true"); ap.add_argument("--rev", action="store_true"); ap.add_argument("--row-ids", default=""); ap.add_argument("--stop-block", type=int, default=64, help="run blocks < this only (no next-token tables, no attention controls above it)")
a = ap.parse_args()
os.makedirs(os.path.join(a.out, "samp"), exist_ok=True)
lab = intlib.Lab(f32=not a.bf16)
ST, NL, H, dev = lab.ST, lab.NL, lab.H, lab.dev
if a.row_ids:
    items = intlib.items_from_ids(lab.D, a.row_ids, gen_lo=a.gen_lo, gen_hi=a.gen_hi)
elif a.rows:
    items = [(l.split()[0], int(l.split()[1])) for l in open(a.rows) if l.strip()]
else:
    D2 = lab.D
    if a.exclude_a0:
        ex = {r for r, _ in intlib.pick_rows(lab.D, per_bench=8, seed=1111)} | {r for r, _ in intlib.pick_rows({r: v for r, v in lab.D.items() if r not in {x for x, _ in intlib.pick_rows(lab.D, per_bench=8, seed=1111)}}, per_bench=4, seed=2222, gen_hi=1800)}
        D2 = {r: v for r, v in lab.D.items() if r not in ex}
    items = intlib.pick_rows(D2, per_bench=a.per_bench, seed=a.seed, gen_lo=a.gen_lo, gen_hi=a.gen_hi)
si, sn = (int(x) for x in a.shard.split("/"))
items = items[si::sn]
if a.rev:
    items = items[::-1]
if a.limit:
    items = items[: a.limit]
NNID = lab.tk("\n\n", add_special_tokens=False).input_ids[0]
EDGES = [0, 8, 32, 150, 400, 1000, 2000, 4000, 10 ** 9]; NB = len(EDGES) - 1
ALAY = [47, 51, 55, 59, 63]; PLAY = [55, 59]
print(f"{len(items)} traces", flush=True)

for it, (rid, k) in enumerate(items):
    fn = os.path.join(a.out, f"{rid.replace('/', '__')}__s{k}.npz")
    if os.path.exists(fn):
        continue
    t0 = time.time()
    gen, te = lab.gen_ids(rid, k)
    cm = lab.prep(rid, "medium", gen); cx = lab.prep(rid, a.mode, gen)
    A, S = lab.align(cm, cx); P, T = cm.P, cm.T
    tg = torch.arange(P - 5, T, device=dev); ntg = len(tg)
    th = slice(5, 5 + te)                                           # thinking positions inside tg
    tix = torch.arange(te, device=dev)
    bins = [((tix >= EDGES[b]) & (tix < EDGES[b + 1])) for b in range(NB)]
    g = torch.Generator().manual_seed(a.seed + 7 * it)
    ns = min(a.ns, te)
    samp = torch.cat([torch.arange(5), 5 + torch.randperm(te, generator=g)[:ns].sort().values]).to(dev)       # the 5 tail positions + sampled thinking positions
    Jm = np.zeros((NL, H), np.float32); Im = np.zeros((NL, H), np.float32); Jb = np.zeros((NL, NB, H), np.float16); Jtail = np.zeros((NL, 5, H), np.float16)
    Dm_ = np.zeros((NL, H), np.float32)                             # mean total residual difference after the block (thinking positions)
    jsq = np.zeros((NL, NB)); isq = np.zeros((NL, NB)); dmsq = np.zeros((NL, NB)); xsq = np.zeros((NL, NB)); nb = np.array([int(b.sum()) for b in bins])
    U = np.zeros((NL, len(samp), H), np.float16); Js = np.zeros((NL, len(samp), H), np.float16)
    att = {}; keepJ = [] if it == 0 else None
    hm, hx = cm.h0, cx.h0
    series = {}
    for l in range(min(NL, a.stop_block)):
        hh = torch.cat([hm[:, : P - 5], hx[:, P - 5 + S:]], 1)
        ST["cap"] = dict(mix=lambda l, o: o[0][tg], mlpin=lambda l, o: o[0][tg][samp]); ST["out"] = {}
        hh_out = lab.run(cm, start=l, stop=l + 1, h=hh); oh = ST["out"]
        ST["cap"] = dict(mix=lambda l, o: o[0][tg]); ST["out"] = {}
        if l in PLAY:
            ST["cap"]["qk"] = lambda l, o: (o[0][0][:, P: P + te].float(), o[1][0][:, :8].float())
        hm = lab.run(cm, start=l, stop=l + 1, h=hm); om = ST["out"]
        ST["cap"] = dict(mix=lambda l, o: o[0][tg + S]); ST["out"] = {}
        if l in ALAY:
            ST["cap"].update(qk=lambda l, o: (o[0][0][:, P + S: P + S + te].float(), o[1][0].float()), qproj=lambda l, o: o[0][P + S: P + S + te].float(), vproj=lambda l, o: o[0].float())
        hx = lab.run(cx, start=l, stop=l + 1, h=hx); ox = ST["out"]
        ST["cap"] = {}; ST["out"] = {}
        J = (hx[0][tg + S].float() - hh_out[0][tg].float())          # block-level forcing [ntg, H]
        I = (ox[("mix", l)].float() - oh[("mix", l)].float())        # mixer-level forcing
        Dx = (ox[("mix", l)].float() - om[("mix", l)].float())       # total mixer difference
        Jt, It, Dt = J[th], I[th], Dx[th]
        Jm[l] = Jt.mean(0).cpu().numpy(); Im[l] = It.mean(0).cpu().numpy(); Jtail[l] = J[:5].cpu().numpy()
        Dm_[l] = (hx[0][tg + S][th].float() - hm[0][tg][th].float()).mean(0).cpu().numpy()
        xn = hm[0][tg][th].float().pow(2).sum(-1)
        for b in range(NB):
            if nb[b]:
                Jb[l, b] = Jt[bins[b]].mean(0).cpu().numpy(); jsq[l, b] = float(Jt[bins[b]].pow(2).sum(-1).mean()); isq[l, b] = float(It[bins[b]].pow(2).sum(-1).mean())
                dmsq[l, b] = float(Dt[bins[b]].pow(2).sum(-1).mean()); xsq[l, b] = float(xn[bins[b]].mean())
        U[l] = oh[("mlpin", l)].float().cpu().numpy(); Js[l] = J[samp].cpu().numpy()
        if keepJ is not None:
            keepJ.append(J.clone())
        if l in ALAY:
            q, kk = ox[("qk", l)]                                    # q [24, te, 256], k [4, Tx, 256]
            gate = ox[("qproj", l)].view(te, 24, 512)[..., 256:]; v = ox[("vproj", l)].view(cx.T, 4, 256)
            Wo = lab.layers[l].self_attn.o_proj.weight.float()
            qpos = torch.arange(P + S, P + S + te, device=dev); ar = torch.arange(cx.T, device=dev)
            mass = torch.zeros(te, 24, device=dev); tokm = torch.zeros(24, S, device=dev); sw = torch.zeros(te, H, device=dev)
            for j in range(24):
                w = torch.softmax(((q[j] @ kk[j // 6].T) * (256 ** -0.5)).masked_fill(ar[None, :] > qpos[:, None], float("-inf")), -1)
                ws = w[:, A: A + S]
                mass[:, j] = ws.sum(-1); tokm[j] = ws.mean(0)
                sw += ((ws @ v[A: A + S, j // 6]) * torch.sigmoid(gate[:, j])) @ Wo[:, j * 256:(j + 1) * 256].T
                del w, ws
            att[f"mass{l}"] = mass.cpu().numpy().astype(np.float16); att[f"tokm{l}"] = tokm.cpu().numpy()
            cosd = torch.nn.functional.cosine_similarity(sw, Dt, dim=-1)
            att[f"dir{l}"] = np.array([float(cosd.mean()), float(sw.pow(2).sum(-1).mean()), float(Dt.pow(2).sum(-1).mean()), float((Dt - sw).pow(2).sum(-1).mean()),
                                       float(torch.nn.functional.cosine_similarity(sw, It, dim=-1).mean()), float((It - sw).pow(2).sum(-1).mean()), float(It.pow(2).sum(-1).mean())])
            # regression of the per-position residual difference after the block on the head masses (in-sample R2 beyond the mean)
            d_t = (hx[0][tg + S][th].float() - hm[0][tg][th].float())
            Xr = torch.cat([torch.ones(te, 1, device=dev), mass], 1)
            Bc = torch.linalg.lstsq(Xr, d_t).solution
            res_ = (d_t - Xr @ Bc).pow(2).sum(); tot_ = (d_t - d_t.mean(0)).pow(2).sum()
            att[f"r2_{l}"] = np.array([1 - float(res_ / tot_), float(d_t.mean(0).pow(2).sum() / d_t.pow(2).sum(-1).mean())])
            series[f"dn{l}"] = d_t.norm(dim=-1).cpu().numpy().astype(np.float32)
            del q, kk, gate, v, Wo, mass, sw, d_t
        if l in PLAY:                                                # medium run: where do the same heads look when there is no instruction (first 8 positions)
            q, k8 = om[("qk", l)]
            # mass on the first 8 keys needs the full softmax: recompute with all keys is costly; use the xhigh-run analogue instead (stored tokm) and
            # here only the unnormalised score of key 0 relative to the head's max over the first 8 keys (cheap sink indicator)
            sc = torch.einsum("hqd,hkd->hqk", q, k8[torch.arange(24, device=dev) // 6]) * (256 ** -0.5)
            att[f"msink{l}"] = sc.mean(1).cpu().numpy()
            del q, k8, sc
        del oh, om, ox, J, I, Dx, hh, hh_out
    # next-token distributions (for position tables)
    npred = min(te + 1, len(gen)); pm = torch.arange(P - 1, P - 1 + npred, device=dev)
    klv = np.zeros(npred, np.float32) if a.stop_block < NL else np.concatenate([intlib.kl(lab.logp(hx[0][pm[c0: c0 + 512] + S]), lab.logp(hm[0][pm[c0: c0 + 512]])).cpu().numpy() for c0 in range(0, npred, 512)])
    chk = -1.0
    if keepJ is not None and a.stop_block >= NL:                     # exactness: medium run + J at stream positions == xhigh run at stream positions
        rp = {l: (lambda h, Jl=keepJ[l]: torch.cat([h[:, : P - 5], h[:, P - 5:] + Jl.to(h.dtype)[None]], 1)) for l in range(NL)}
        hc = lab.run(cm, res_patch=rp)
        chk = float((hc[0][tg].float() - hx[0][tg + S].float()).abs().max()); ref = float((hm[0][tg].float() - hx[0][tg + S].float()).abs().max())
        klc = float(intlib.kl(lab.logp(hx[0][pm[:512] + S]), lab.logp(hc[0][pm[:512]])).mean())
        print(f"exactness check: max |medium+J - xhigh| at stream positions {chk:.2e} (medium vs xhigh {ref:.2e}); KL(x||medium+J) {klc:.2e} vs KL(x||m) {klv[:512].mean():.4f}", flush=True)
        del hc, keepJ
    s = lab.D[rid]["samples"][str(k)]
    np.savez(fn, row=rid, sample=k, ok=bool(s["ok"]), bench=lab.D[rid]["bench"], Jm=Jm, Im=Im, Dm=Dm_, Jb=Jb, Jtail=Jtail, jsq=jsq, isq=isq, dmsq=dmsq, xsq=xsq, nb=nb, kl=klv,
             gen=np.array(gen[:npred]), te=te, P=P, T=T, S=S, chk=chk, sec=time.time() - t0, **att, **series)
    if not a.no_samp:
        np.savez(os.path.join(a.out, "samp", os.path.basename(fn)), U=U, J=Js, samp=samp.cpu().numpy(), te=te)
    jn = np.linalg.norm(Jm, axis=1); dn = np.sqrt(dmsq.mean(1))
    print(f"[{it + 1}/{len(items)}] {rid} s{k} think {te} | |mean J| L15 {jn[15]:.3f} L31 {jn[31]:.3f} L47 {jn[47]:.3f} L59 {jn[59]:.3f} L63 {jn[63]:.3f} | const(J) L31 {jn[31] ** 2 / max(1e-12, (jsq[31] * nb).sum() / nb.sum()):.2f} L59 {jn[59] ** 2 / max(1e-12, (jsq[59] * nb).sum() / nb.sum()):.2f} "
          f"| {time.time() - t0:.0f}s peak {torch.cuda.max_memory_allocated() / 2 ** 30:.1f}G", flush=True)
    del hm, hx
    torch.cuda.empty_cache()
print("done", flush=True)
