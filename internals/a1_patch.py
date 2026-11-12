"""a1 (fable_int): causal localisation of the xhigh instruction by activation patching on identical tokens.
M = medium-template run, X = xhigh-template run of the same transcript. Conditions (name -> what is done):
  null                      M restarted from a cached residual (must reproduce M)
  res{L}_{cls}              M with the residual after block L replaced by X's at positions of cls in (img, txt, tail, gen, all)   [denoise]
  mixb{b}/mlpb{b}           M with the mixer / MLP OUTPUTS of blocks 8b..8b+7 at tail+gen positions replaced by X's                 [denoise]
  mixfull / mixlin          same for the 16 full-attention / the 48 linear-attention blocks
  mixall                    all 64 mixers (sanity: must reproduce X at tail+gen)
  headsK / headsRK          M with the per-head outputs of the top-K (random-K) full-attention heads at tail+gen replaced by X's     [denoise]
  xcut_all / xcut_lo / xcut_hi / xcut_63   X with attention from tail+gen queries to the sys keys blocked in full blocks (all 16 / 3..31 / 35..63 / 63)   [noise]
  xcutP_all                 X with attention from ALL later positions to the sys keys blocked in the 16 full blocks (only the linear blocks carry the instruction)
  xheadsK                   X with the top-K heads' outputs at tail+gen replaced by M's                                              [noise]
Per condition: kl_xc [n] = KL(p_X || p_cond) at the prediction positions, kl_mc [n] = KL(p_M || p_cond), rproj (projection of the final-residual
shift on X - M), stop logit difference lp("\\n") - lp("\\n\\n") at paragraph ends.
Output OUT/<row>__s<k>.npz ; --f32 / --both (every condition in float32 and in bf16).
"""
import argparse, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import intlib
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True); ap.add_argument("--shard", default="0/1"); ap.add_argument("--per-bench", type=int, default=4)
ap.add_argument("--f32", action="store_true"); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--rows", default=""); ap.add_argument("--seed", type=int, default=2222); ap.add_argument("--exclude", default="", help="dir of a0 records: rows to skip")
ap.add_argument("--heads", default="", help="json: ranked list of [layer, head] (from a0, other rows)")
ap.add_argument("--max-think", type=int, default=1800); ap.add_argument("--sets", default="res,res2,mix,mlp,heads,cut")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
lab = intlib.Lab(f32=a.f32)
ST, NL, H, dev = lab.ST, lab.NL, lab.H, lab.dev
SETS = set(a.sets.split(","))
if a.rows:
    items = [(l.split()[0], int(l.split()[1])) for l in open(a.rows) if l.strip()]
else:
    excl = set()
    if a.exclude:
        excl = {f.split("__s")[0].replace("__", "/") for f in os.listdir(a.exclude) if f.endswith(".npz")}
    excl |= {r for r, _ in intlib.pick_rows(lab.D, per_bench=8, seed=1111)}          # the a0 rows (head ranking, directions)
    D2 = {r: v for r, v in lab.D.items() if r not in excl}
    items = intlib.pick_rows(D2, per_bench=a.per_bench, seed=a.seed, gen_hi=a.max_think)
si, sn = (int(x) for x in a.shard.split("/"))
items = items[si::sn]
if a.limit:
    items = items[: a.limit]
print(f"{len(items)} traces; sets {sorted(SETS)}", flush=True)
HEADS = [tuple(x) for x in json.load(open(a.heads))] if a.heads else []
NLID = lab.tk("\n", add_special_tokens=False).input_ids; NNID = lab.tk("\n\n", add_special_tokens=False).input_ids
assert len(NLID) == 1 and len(NNID) == 1, (NLID, NNID)
NLID, NNID = NLID[0], NNID[0]
LSET = [7, 15, 23, 31, 39, 47, 55]; LSET2 = [3, 11, 19, 27, 35, 43, 51, 59]
LCACHE = sorted(set(LSET + LSET2))
FULL = lab.FULL

for it, (rid, k) in enumerate(items):
    fn = os.path.join(a.out, f"{rid.replace('/', '__')}__s{k}.npz")
    if os.path.exists(fn):
        continue
    t0 = time.time()
    gen, te = lab.gen_ids(rid, k)
    cm = lab.prep(rid, "medium", gen); cx = lab.prep(rid, os.environ.get("FI_MODE", "xhigh"), gen)
    A, S = lab.align(cm, cx); P, T = cm.P, cm.T
    if it == 0:
        am = cm.kw_full.get("attention_mask")
        print("full-attention mask passed by the model:", None if am is None else (am.dtype, tuple(am.shape)), "| linear:", cm.kw_lin.get("attention_mask"), flush=True)
    clsM, cvec = lab.classes(cm)
    pos = dict(img=clsM["img"][clsM["img"] >= A], txt=clsM["txt"][clsM["txt"] >= A], tail=clsM["tail"], gen=clsM["gen"])
    pos["all"] = torch.arange(A, T, device=dev)
    tg = torch.arange(P - 5, T, device=dev)                       # tail + gen (medium coordinates)
    npred = min(te + 1, len(gen))
    pm = torch.arange(P - 1, P - 1 + npred, device=dev)           # prediction positions (medium coordinates)
    y = torch.tensor(gen[:npred], device=dev)
    bnd = ((y == NNID) | (y == NLID)).nonzero()[:, 0]             # paragraph ends: the next token is "\n\n" or "\n"
    # ---- reference runs with caches (CPU)
    Mres, Xres, Xmix, Xmlp, Xhead, Mhead = {}, {}, {}, {}, {}, {}
    ST["cap"] = dict(head=lambda l, o: o[0][tg].cpu() if l in FULL else None); ST["out"] = {}
    hm = lab.run(cm, res_cap=lambda l, h: Mres.__setitem__(l, h.cpu()) if l in LCACHE else None)
    Mhead = {l: ST["out"][("head", l)] for l in FULL}
    ST["cap"] = dict(mix=lambda l, o: o[0][tg + S].cpu(), mlp=lambda l, o: o[0][tg + S].cpu(), head=lambda l, o: o[0][tg + S].cpu() if l in FULL else None); ST["out"] = {}
    hx = lab.run(cx, res_cap=lambda l, h: Xres.__setitem__(l, h.cpu()) if l in LCACHE else None)
    Xmix = {l: ST["out"][("mix", l)] for l in range(NL)}; Xmlp = {l: ST["out"][("mlp", l)] for l in range(NL)}; Xhead = {l: ST["out"][("head", l)] for l in FULL}
    lab.clear()
    hm_p, hx_p = hm[0][pm].float(), hx[0][pm + S].float()
    lpm, lpx = lab.logp(hm[0][pm]), lab.logp(hx[0][pm + S])
    dref = hx_p - hm_p; dref2 = float(dref.pow(2).sum())
    kl_xm = intlib.kl(lpx, lpm); kl_mx = intlib.kl(lpm, lpx)
    ld = lambda lp: (lp[bnd, NLID] - lp[bnd, NNID])
    res = {}

    def record(name, hfin, base):
        """hfin: final residual [1, T(or Tx), H] of the condition; base 'm' (medium coordinates) or 'x'"""
        hp = hfin[0][pm + (S if base == "x" else 0)]
        lpc = lab.logp(hp)
        res[name] = dict(kl_xc=intlib.kl(lpx, lpc).cpu().numpy(), kl_mc=intlib.kl(lpm, lpc).cpu().numpy(),
                         rproj=float(((hp.float() - hm_p) * dref).sum()) / dref2, ld=ld(lpc).cpu().numpy())
        del lpc

    def from_m(L):                      # residual after block L of M, on the card (a fresh copy)
        return Mres[L].to(dev).clone()

    # ---- sanity: restart reproduces M ; all mixers patched reproduces X
    record("null", lab.run(cm, start=LSET[2] + 1, h=from_m(LSET[2])), "m")
    # ---- residual patching
    todo = []
    if "res" in SETS:
        todo += [(L, c) for L in LSET for c in ("img", "txt", "tail", "gen", "all")]
    if "res2" in SETS:
        todo += [(L, c) for L in LSET2 for c in ("gen", "all")]
    for L, c in todo:
        h = from_m(L); xr = Xres[L]
        p_ = pos[c]
        h[0, p_] = xr[0][(p_ + S).cpu()].to(dev)
        record(f"res{L}_{c}", lab.run(cm, start=L + 1, h=h), "m")
        del h

    def put(cache, cols=None):
        def f(o):
            o = o.clone()
            if cols is None:
                o[0, tg] = cache.to(dev).to(o.dtype)
            else:
                o[0, tg[:, None], cols[None, :]] = cache[:, cols.cpu()].to(dev).to(o.dtype)
            return o
        return f

    def band_run(kind, cache, ls):
        ST["patch"] = {(kind, l): put(cache[l]) for l in ls}
        b0 = min(ls)
        st = max([L for L in LCACHE if L < b0], default=None)
        out = lab.run(cm) if st is None else lab.run(cm, start=st + 1, h=from_m(st))
        ST["patch"] = {}
        return out
    if "mix" in SETS:
        for b in range(8):
            record(f"mixb{b}", band_run("mix", Xmix, list(range(8 * b, 8 * b + 8))), "m")
        record("mixfull", band_run("mix", Xmix, FULL), "m")
        record("mixlin", band_run("mix", Xmix, [l for l in range(NL) if l not in FULL]), "m")
        record("mixall", band_run("mix", Xmix, list(range(NL))), "m")
    if "mlp" in SETS:
        for b in range(8):
            record(f"mlpb{b}", band_run("mlp", Xmlp, list(range(8 * b, 8 * b + 8))), "m")
        record("mlpall", band_run("mlp", Xmlp, list(range(NL))), "m")

    def head_patch(cache, hs):
        by = {}
        for (l, j) in hs:
            by.setdefault(l, []).append(j)
        pt = {}
        for l, js in by.items():
            cols = torch.cat([torch.arange(j * 256, (j + 1) * 256) for j in sorted(js)]).to(dev)
            pt[("head", l)] = put(cache[l], cols)
        return pt
    if "heads" in SETS and HEADS:
        rng = np.random.default_rng(a.seed + it)
        allh = [(l, j) for l in FULL for j in range(24)]
        for K in (4, 16, 64):
            ST["patch"] = head_patch(Xhead, HEADS[:K]); record(f"heads{K}", lab.run(cm), "m")
            rh = [allh[i] for i in rng.permutation(len(allh))[:K]]
            ST["patch"] = head_patch(Xhead, rh); record(f"headsR{K}", lab.run(cm), "m")
            # X run: positions are tg + S
            def putx(cache, cols):
                def f(o):
                    o = o.clone(); o[0, (tg + S)[:, None], cols[None, :]] = cache[:, cols.cpu()].to(dev).to(o.dtype); return o
                return f
            by = {}
            for (l, j) in HEADS[:K]:
                by.setdefault(l, []).append(j)
            ST["patch"] = {("head", l): putx(Mhead[l], torch.cat([torch.arange(j * 256, (j + 1) * 256) for j in sorted(js)]).to(dev)) for l, js in by.items()}
            record(f"xheads{K}", lab.run(cx), "x")
            ST["patch"] = {}
    if "cut" in SETS:
        Tx = cx.T
        base = torch.ones(Tx, Tx, dtype=torch.bool, device=dev).tril_()
        m1 = base.clone(); m1[P - 5 + S:, A: A + S] = False                 # tail+gen queries cannot see the sys keys
        m2 = base.clone(); m2[A + S:, A: A + S] = False                     # no later position can see the sys keys
        m1, m2 = m1[None, None], m2[None, None]
        am0 = cx.kw_full.get("attention_mask")
        if am0 is not None and am0.dtype != torch.bool:                     # the model passes an additive mask: same convention
            cvt = lambda m: torch.zeros(m.shape, dtype=am0.dtype, device=dev).masked_fill(~m, torch.finfo(am0.dtype).min)
            m1, m2 = cvt(m1), cvt(m2)
        for name, ls, mk in (("xcut_all", FULL, m1), ("xcut_lo", FULL[:8], m1), ("xcut_hi", FULL[8:], m1), ("xcut_63", [63], m1), ("xcutP_all", FULL, m2)):
            record(name, lab.run(cx, full_mask={l: mk for l in ls}), "x")
        del base, m1, m2
    s = lab.D[rid]["samples"][str(k)]
    out = dict(row=rid, sample=k, ok=bool(s["ok"]), bench=lab.D[rid]["bench"], kl_xm=kl_xm.cpu().numpy(), kl_mx=kl_mx.cpu().numpy(), ld_m=ld(lpm).cpu().numpy(), ld_x=ld(lpx).cpu().numpy(),
               bnd=bnd.cpu().numpy(), y=y.cpu().numpy(), te=te, P=P, T=T, S=S, f32=a.f32, names=np.array(sorted(res)), sec=time.time() - t0)
    for n_, r in res.items():
        for f_, v in r.items():
            out[f"{n_}/{f_}"] = v
    np.savez(fn, **out)
    base_kl = float(kl_xm.mean())
    R = lambda n_: 1 - float(res[n_]["kl_xc"].mean()) / base_kl if n_ in res else float("nan")
    print(f"[{it + 1}/{len(items)}] {rid} s{k}: T {T} think {te} KL(x||m) {base_kl:.4f} | null {R('null'):+.4f} mixall {R('mixall'):.3f} | res31 img {R('res31_img'):.2f} txt {R('res31_txt'):.2f} tail {R('res31_tail'):.2f} gen {R('res31_gen'):.2f} all {R('res31_all'):.2f} | "
          f"mixfull {R('mixfull'):.2f} mixlin {R('mixlin'):.2f} | xcut_all {R('xcut_all'):.2f} xcutP {R('xcutP_all'):.2f} | heads16 {R('heads16'):.2f} R16 {R('headsR16'):.2f} | {time.time() - t0:.0f}s peak {torch.cuda.max_memory_allocated() / 2 ** 30:.1f}G", flush=True)
    del Mres, Xres, Xmix, Xmlp, Xhead, Mhead, lpm, lpx, hm, hx
    torch.cuda.empty_cache()
print("done", flush=True)
