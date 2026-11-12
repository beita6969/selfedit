"""b_fn5 (fable_int; b_fn4 with the per-position statistics computed in chunks of 256 positions and only three log-prob tables kept: b_fn4 ran out of memory on reads above about 2,000 thinking tokens, a table is 2.5 GB there). b_fn4 = b_fn3 with one block's unit output held at a time: b_fn3 kept all 64 blocks' outputs until the end of the pass and ran out of memory on long reads). b_fn2 plus --alt: on reads that ended wrong, a second reference = the same hindsight line with the read's OWN answer instead of the gold ("decoy hint"); per cell the shift along the gold-hint shift, along the decoy-hint shift, and along the gold-specific part. b_fn plus controls built on the fly: var=wrot rotates every output vector by one random orthogonal map, var=rrot rotates the read directions and centres / rescales their activations on the evaluated read to the original units' variance): teacher-forced evaluation of compiled cells in FUNCTIONAL form (no weight edit: the units' output is added to the MLP output of their block,
computed from the MLP input exactly as the neurons would: silu(8 key.x) * (u.x) * v/8 * dose; the removal of the neurons they would occupy is not simulated).
For cells with thousands of units, where applying them through edits.py takes minutes per condition. Records in the format of b_xb.py (ana1.py --b, ana_leak.py read them).
Base run = "medium" (with FI_BASE_NOPREFIX=1: the prompt without the routed prefix), reference = --mode (proc: the prompt as read).
--cells name=lo-hi:rank:dose[:opt,...]   opts: gate=ks|k|kr (key of the linear units; default ks = stream key), blocks=all|full|lin, var=fl|rand|rot (rand / rot need --ctl), nobias
python b_fn.py --arrays A.npz [--ctl CTL.npz] --rows ITEMS --out DIR --cells ..."""
import argparse, json, os, sys, time
import numpy as np, torch
import torch.nn.functional as Fn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import intlib
ap = argparse.ArgumentParser()
ap.add_argument("--mode", default="proc"); ap.add_argument("--out", required=True); ap.add_argument("--rows", required=True); ap.add_argument("--arrays", required=True); ap.add_argument("--ctl", default="")
ap.add_argument("--bf16", action="store_true"); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--cells", nargs="+", required=True); ap.add_argument("--shard", default="0/1"); ap.add_argument("--alt", action="store_true")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
lab = intlib.Lab(f32=not a.bf16); ST, NL, H, dev = lab.ST, lab.NL, lab.H, lab.dev
A = np.load(a.arrays); C = np.load(a.ctl) if a.ctl else None
T_ = lambda z: torch.tensor(np.asarray(z, np.float32).reshape(-1), device=dev)
_Q = {}


def rot(tag):
    if tag not in _Q:
        g = torch.Generator().manual_seed(20261112 + len(tag)); _Q[tag] = torch.linalg.qr(torch.randn(H, H, generator=g).to(dev))[0]
    return _Q[tag]
CELLS = {}
for c in a.cells:
    name, spec = c.split("=", 1); f = spec.split(":"); (lo, hi), rank, dose = (int(x) for x in f[0].split("-")), int(f[1]), float(f[2])
    opt = dict(o.split("=") if "=" in o else (o, "1") for o in (f[3].split(",") if len(f) > 3 else []))
    gate, blocks, var, nobias = opt.get("gate", "ks"), opt.get("blocks", "all"), opt.get("var", "fl"), "nobias" in opt
    L = {}
    for l in range(lo, hi + 1):
        full = lab.layers[l].block_type == "full_attention"
        if (blocks == "full" and not full) or (blocks == "lin" and full):
            continue
        upk = "up_RF_" if var == "rand" else "up_FL_"; dnk = "down_RO_" if var == "rot" else "down_FL_"; bk = "down_ROb_" if var == "rot" else "down_FJm_"
        SU = C if var == "rand" else A; SD = C if var == "rot" else A
        if var in ("wrot", "rrot"):
            upk, dnk, bk, SU, SD = "up_FL_", "down_FL_", "down_FJm_", A, A
        L[l] = dict(g=T_(A[f"{gate}_{l}"]), k=T_(A[f"k_{l}"]), U=torch.stack([T_(SU[f"{upk}{l}_{i}"]) for i in range(rank)]), V=torch.stack([T_(SD[f"{dnk}{l}_{i}"]) for i in range(rank)]),
                    b=None if nobias else T_(SD[f"{bk}{l}"]))
        if var == "wrot":
            L[l]["V"] = L[l]["V"] @ rot("w").T
            if L[l]["b"] is not None:
                L[l]["b"] = L[l]["b"] @ rot("w").T
        if var == "rrot":
            L[l]["U0"] = L[l]["U"]; L[l]["U"] = L[l]["U"] @ rot("rd").T
    CELLS[name] = (L, dose); print(f"cell {name}: blocks {len(L)}, units {len(L) * (rank + (0 if nobias else 1))}, dose {dose}, gate {gate}, variant {var}", flush=True)
HINT = os.environ.get("FI_HIND", "The correct final answer to this question is: {gold}.")
items = [(l.split()[0], int(l.split()[1])) for l in open(a.rows) if l.strip()]
si, sn = (int(x) for x in a.shard.split("/")); items = items[si::sn]
if a.limit:
    items = items[: a.limit]
NLID = lab.tk("\n", add_special_tokens=False).input_ids[0]; NNID = lab.tk("\n\n", add_special_tokens=False).input_ids[0]


def run_cell(cm, L, dose):
    store = {}

    def cap(l, x):
        d = L.get(l)
        if d is None:
            return None
        xx = x[0].float()
        gs = xx @ d["g"]; acts = xx @ d["U"].T
        if "U0" in d:                                   # random-read control: centre and rescale on the stream positions of this read
            st = gs > 0.5
            if int(st.sum()) > 8:
                a0 = xx[st] @ d["U0"].T; acts = (acts - acts[st].mean(0)) * (a0.std(0) / acts[st].std(0).clamp_min(1e-6))
        out = (Fn.silu(8.0 * gs)[:, None] * acts) @ d["V"]
        if d["b"] is not None:
            out = out + (Fn.silu(8.0 * (xx @ d["k"])) * gs)[:, None] * d["b"][None]
        store[l] = (dose * out).to(x.dtype)
        return None
    ST["cap"] = dict(mlpin=cap); ST["out"] = {}
    ST["patch"] = {("mlp", l): (lambda o, l=l: o + store.pop(l)[None]) for l in L}
    try:
        return lab.run(cm)
    finally:
        lab.clear()


print(f"{len(items)} traces", flush=True)
for it, (rid, k) in enumerate(items):
    fn = os.path.join(a.out, f"{rid.replace('/', '__')}__s{k}.npz")
    if os.path.exists(fn):
        continue
    t0 = time.time(); gen, te = lab.gen_ids(rid, k)
    cm = lab.prep(rid, "medium", gen); cx = lab.prep(rid, a.mode, gen)
    Ax, S = lab.align(cm, cx); P, T = cm.P, cm.T
    npred = min(te + 1, len(gen)); pm = torch.arange(P - 1, P - 1 + npred, device=dev); y = torch.tensor(gen[:npred], device=dev)
    pstart = torch.zeros(npred, dtype=torch.bool, device=dev); pstart[1:] = y[:-1] == NNID; pstart[0] = True; pidx = torch.arange(npred, device=dev)
    lpm = lab.logp(lab.run(cm)[0][pm]); lpx = lab.logp(lab.run(cx)[0][pm + S]); CH = 256

    def chunked(fn, *T):                            # fn on chunks of positions -> concatenated numpy
        return torch.cat([fn(*[t[i: i + CH] for t in T]) for i in range(0, T[0].shape[0], CH)]).cpu().numpy()
    KL = lambda a_, b_: chunked(lambda x_, y_: (x_.exp() * (x_ - y_)).sum(-1), a_, b_)
    IP = lambda c_, r_: chunked(lambda m_, c__, r__: (m_.exp() * (c__ - m_) * (r__ - m_)).sum(-1), lpm, c_, r_)      # p_m-weighted <c - m, r - m>
    out = dict(row=rid, sample=k, ok=bool(lab.D[rid]["samples"][str(k)]["ok"]), bench=lab.D[rid]["bench"], y=y.cpu().numpy(), pstart=pstart.cpu().numpy(), te=te, P=P, T=T, pidx=pidx.cpu().numpy(), S=S)

    def record(name, lpc):
        out[f"{name}/kl_xc"] = KL(lpx, lpc); out[f"{name}/kl_mc"] = KL(lpm, lpc); out[f"{name}/kl_cm"] = KL(lpc, lpm)
        out[f"{name}/lpy"] = lpc.gather(1, y[:, None])[:, 0].cpu().numpy(); out[f"{name}/lp_nl"] = lpc[:, NLID].cpu().numpy(); out[f"{name}/lp_nn"] = lpc[:, NNID].cpu().numpy()
        out[f"{name}/ip_cx"] = IP(lpc, lpx); out[f"{name}/ip_cc"] = IP(lpc, lpc)
        out[f"{name}/ent"] = chunked(lambda c__: -(c__.exp() * c__).sum(-1), lpc)
    record("m", lpm); record("x", lpx); names = ["m", "x"]
    samp_ = lab.D[rid]["samples"][str(k)]; lpa = None
    if a.alt and not bool(samp_["ok"]) and str(samp_.get("pred") or "").strip() and str(samp_.get("pred")).strip() != str(lab.D[rid]["answer"]).strip():
        tf = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"decoy_{os.getpid()}_{it}.txt"); open(tf, "w").write(HINT.format(gold=str(samp_["pred"]).strip()))
        ca = lab.prep(rid, "sys@" + tf, gen); _, Sa = lab.align(cm, ca); lpa = lab.logp(lab.run(ca)[0][pm + Sa])
        out["alt/kl_am"] = KL(lpa, lpm); out["alt/ip_aa"] = IP(lpa, lpa); out["alt/ip_xa"] = IP(lpx, lpa); out["alt/kl_xa"] = KL(lpx, lpa); del ca
    for name, (L, dose) in CELLS.items():
        lpc = lab.logp(run_cell(cm, L, dose)[0][pm]); record(name, lpc); names.append(name)
        if lpa is not None:
            out[f"{name}/ip_ca"] = IP(lpc, lpa); out[f"{name}/kl_ac"] = KL(lpa, lpc)
        del lpc
    out["names"] = np.array(names); out["sec"] = time.time() - t0
    np.savez(fn, **out)
    bx = out["m/kl_xc"].mean()
    print(f"[{it + 1}/{len(items)}] {rid} s{k} think {te} KL(x||m) {bx:.4f} | R, KL(m||c): " + " ".join(f"{n_} {(1 - out[n_ + '/kl_xc'].mean() / bx) if bx > 0 else float('nan'):+.2f},{out[n_ + '/kl_mc'].mean():.3f}" for n_ in names[2:]) + f" | {time.time() - t0:.0f}s", flush=True)
    del lpm, lpx, lpa
    torch.cuda.empty_cache()
print("done", flush=True)
