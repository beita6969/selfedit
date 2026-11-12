"""a4 (fable_int): exact forcing map by block band. The thinking stream (tail + generated positions) is run with the xhigh CONTEXT in a chosen set of
blocks and the medium context in all others (context positions always come from the pure runs: they cannot see the stream).
  suf_<band>: xhigh context only in the band (what any edit confined to those blocks could at most reproduce)
  nec_<band>: xhigh context everywhere except the band (what is lost when the instruction cannot reach the stream there)
Sanity: all blocks -> the xhigh run (R = 1), no block -> the medium run (R = 0).
Per condition: kl_xc, kl_mc at the prediction positions; R = 1 - sum kl_xc / sum kl_xm. float32.
"""
import argparse, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import intlib
ap = argparse.ArgumentParser()
ap.add_argument("--mode", default="xhigh", help="template mode contrasted with medium (xhigh | low)"); ap.add_argument("--out", required=True); ap.add_argument("--shard", default="0/1"); ap.add_argument("--per-bench", type=int, default=2); ap.add_argument("--seed", type=int, default=2222)
ap.add_argument("--bf16", action="store_true"); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--max-think", type=int, default=1800)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
lab = intlib.Lab(f32=not a.bf16)
NL, dev = lab.NL, lab.dev
excl = {r for r, _ in intlib.pick_rows(lab.D, per_bench=8, seed=1111)}
items = intlib.pick_rows({r: v for r, v in lab.D.items() if r not in excl}, per_bench=a.per_bench, seed=a.seed, gen_hi=a.max_think)
items = [items[i] for b in range(a.per_bench) for i in range(b, len(items), a.per_bench)]
si, sn = (int(x) for x in a.shard.split("/")); items = items[si::sn]
if a.limit:
    items = items[: a.limit]
ALL = set(range(NL)); B = lambda lo, hi: set(range(lo, hi + 1))
COND = {"all": ALL, "none": set(), "suf_0-23": B(0, 23), "suf_24-34": B(24, 34), "suf_24-39": B(24, 39), "suf_26+30": {26, 30}, "suf_lin24-34": {l for l in range(24, 35) if l % 4 != 3},
        "suf_full24-35": {27, 31, 35}, "suf_35-47": B(35, 47), "suf_48-63": B(48, 63),
        "nec_0-23": ALL - B(0, 23), "nec_24-34": ALL - B(24, 34), "nec_24-39": ALL - B(24, 39), "nec_35-47": ALL - B(35, 47), "nec_48-63": ALL - B(48, 63), "nec_24-47": ALL - B(24, 47)}
print(f"{len(items)} traces; conditions {list(COND)}", flush=True)
for it, (rid, k) in enumerate(items):
    fn = os.path.join(a.out, f"{rid.replace('/', '__')}__s{k}.npz")
    if os.path.exists(fn):
        continue
    t0 = time.time()
    gen, te = lab.gen_ids(rid, k)
    cm = lab.prep(rid, "medium", gen); cx = lab.prep(rid, a.mode, gen)
    A, S = lab.align(cm, cx); P, T = cm.P, cm.T
    c0 = P - 5                                               # first stream position (medium coordinates)
    hm, hx = cm.h0, cx.h0
    hs = {n: hm[:, c0:].clone() for n in COND}               # stream state per condition (embeddings are identical in both templates)
    for l in range(NL):
        for n, xs in COND.items():
            if l in xs:
                o = lab.run(cx, start=l, stop=l + 1, h=torch.cat([hx[:, : c0 + S], hs[n]], 1)); hs[n] = o[:, c0 + S:]
            else:
                o = lab.run(cm, start=l, stop=l + 1, h=torch.cat([hm[:, :c0], hs[n]], 1)); hs[n] = o[:, c0:]
        hm = lab.run(cm, start=l, stop=l + 1, h=hm); hx = lab.run(cx, start=l, stop=l + 1, h=hx)
    npred = min(te + 1, len(gen)); pm = torch.arange(P - 1, P - 1 + npred, device=dev)
    lpm = lab.logp(hm[0][pm]); lpx = lab.logp(hx[0][pm + S])
    out = dict(row=rid, sample=k, bench=lab.D[rid]["bench"], te=te, kl_xm=intlib.kl(lpx, lpm).cpu().numpy(), names=np.array(list(COND)), y=np.array(gen[:npred]))
    for n in COND:
        lpc = lab.logp(hs[n][0][pm - c0])
        out[f"{n}/kl_xc"] = intlib.kl(lpx, lpc).cpu().numpy(); out[f"{n}/kl_mc"] = intlib.kl(lpm, lpc).cpu().numpy()
        del lpc
    np.savez(fn, **out)
    b = out["kl_xm"].sum()
    print(f"[{it + 1}/{len(items)}] {rid} s{k} think {te} | " + " ".join(f"{n} {1 - out[n + '/kl_xc'].sum() / b:+.3f}" for n in COND) + f" | {time.time() - t0:.0f}s", flush=True)
    del lpm, lpx, hs, hm, hx
    torch.cuda.empty_cache()
print("done", flush=True)
