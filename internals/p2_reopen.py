"""p2_reopen (fable_int, P2 step 1, activation level): does a LATE boost of the stream's attention to the image tokens create late reading events?
Cached long reads are teacher-forced with the real image (X) and with a blank image of the same size (M; identical token ids). Image dependence of a position = KL(p_X || p_M); a reading
event = a position with KL >= 1 (the vision-path study's definitions). Knob: +g on the attention logits from thinking positions at or after T_on to the image-token keys, in the
full-attention blocks LO..HI (default 35..55), in BOTH runs. Variants: "late" = every thinking position >= T_on; "ps" = only the W positions after each paragraph start >= T_on.
Per read and condition, by thinking-token bin: mean KL, events per 1,000 tokens, new events (KL >= 1 under the knob, < 0.5 without), and the disturbance of the real-image run:
KL(p_X || p_X,knob) and the change of log p(read token). Before T_on nothing can change (causal), which is checked.
python p2_reopen.py --out DIR [--n 10] [--ton 1000] [--gains 1.0,2.0] [--ps-gain 2.0] [--win 40]"""
import argparse, json, os, sys, time, zlib
import numpy as np, torch
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import intlib
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True); ap.add_argument("--n", type=int, default=10); ap.add_argument("--ton", type=int, default=1000); ap.add_argument("--gains", default="1.0,2.0"); ap.add_argument("--ps-gain", type=float, default=2.0)
ap.add_argument("--win", type=int, default=40); ap.add_argument("--lo", type=int, default=35); ap.add_argument("--hi", type=int, default=55); ap.add_argument("--tmin", type=int, default=2500); ap.add_argument("--tmax", type=int, default=6000)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
BLANK = [False]; _vi = intlib.view_image


def vi(path, mn=None, mx=None):
    im = _vi(path, mn, mx)
    if BLANK[0]:
        c = tuple(int(x) for x in np.asarray(im).reshape(-1, 3).mean(0)); return Image.new("RGB", im.size, c)
    return im


intlib.view_image = vi
lab = intlib.Lab(f32=False); dev = lab.dev; D = lab.D
cand = []
for r in sorted(D):
    v = D[r]
    if v.get("role") not in ("show", "monitor") or len(v["image_paths"]) != 1:
        continue
    for k, s in v["samples"].items():
        if a.tmin <= int(s["tokens"]) <= a.tmax and int(s.get("prompt_tokens") or 0) <= 6500 and s.get("reasoning") and s.get("finish") == "stop":
            cand.append((zlib.crc32(f"{r}#{k}".encode()), r, int(k), v["bench"], bool(s["ok"])))
cand.sort(); pick, per_row, per_b = [], set(), {}
for _, r, k, b, ok in cand:                                  # one read per row, benchmarks and outcomes spread
    if r in per_row or per_b.get((b, ok), 0) >= max(1, a.n // 8) or len(pick) >= a.n:
        continue
    pick.append((r, k)); per_row.add(r); per_b[(b, ok)] = per_b.get((b, ok), 0) + 1
print(f"{len(cand)} long single-image reads; picked {len(pick)}: {per_b}", flush=True)
NN = lab.tk("\n\n", add_special_tokens=False).input_ids[0]; FB = [l for l in lab.FULL if a.lo <= l <= a.hi]
GAINS = [float(x) for x in a.gains.split(",")]
EDGES = [0, a.ton, a.ton + 500, 2500, 10 ** 9] if a.ton + 500 < 2500 else [0, a.ton, a.ton + 500, 10 ** 9]
for it, (rid, k) in enumerate(pick):
    fn = os.path.join(a.out, f"{rid.replace('/', '__')}__s{k}.npz")
    if os.path.exists(fn):
        continue
    t0 = time.time(); gen, te = lab.gen_ids(rid, k)
    BLANK[0] = False; cX = lab.prep(rid, "medium", gen); BLANK[0] = True; cM = lab.prep(rid, "medium", gen); BLANK[0] = False
    assert torch.equal(cX.ids, cM.ids), "blank image changes the token ids"
    P, T = cX.P, cX.T; npred = min(te + 1, len(gen)); pm = torch.arange(P - 1, P - 1 + npred, device=dev); y = torch.tensor(gen[:npred], device=dev)
    img = cX.isimg.nonzero()[:, 0]; tix = torch.arange(npred, device=dev)
    lp = lambda c, fm=None: lab.logp(lab.run(c, full_mask=fm)[0][pm])
    lX0, lM0 = lp(cX), lp(cM); kl0 = intlib.kl(lX0, lM0)
    out = dict(row=rid, sample=k, ok=bool(D[rid]["samples"][str(k)]["ok"]), bench=D[rid]["bench"], te=te, P=P, nimg=len(img), kl0=kl0.cpu().numpy(), lpy0=lX0.gather(1, y[:, None])[:, 0].cpu().numpy())
    q_late = torch.zeros(T, dtype=torch.bool, device=dev); q_late[P + a.ton:] = True          # queries at thinking positions >= T_on
    gt = torch.tensor(gen, device=dev); ps = torch.zeros(T, dtype=torch.bool, device=dev)
    starts = (gt[: te] == NN).nonzero()[:, 0] + 1                                               # thinking index of the token after a paragraph break
    for s_ in starts.tolist():
        if s_ >= a.ton:
            ps[P + s_: P + min(te, s_ + a.win)] = True
    conds = [(f"late_{g:g}", q_late, g) for g in GAINS] + [(f"ps_{a.ps_gain:g}", ps, a.ps_gain)]
    names = []
    for name, qm, g in conds:
        mk = torch.zeros(T, T, dtype=torch.bfloat16, device=dev); mk.masked_fill_(~torch.ones(T, T, dtype=torch.bool, device=dev).tril_(), float("-inf"))
        rows_ = qm.nonzero()[:, 0]; mk[rows_[:, None], img[None, :]] += g                       # image keys lie before every thinking position: stays causal
        fm = {l: mk[None, None] for l in FB}
        lXg, lMg = lp(cX, fm), lp(cM, fm)
        out[f"{name}/kl"] = intlib.kl(lXg, lMg).cpu().numpy(); out[f"{name}/kl_x0g"] = intlib.kl(lX0, lXg).cpu().numpy(); out[f"{name}/lpy"] = lXg.gather(1, y[:, None])[:, 0].cpu().numpy(); out[f"{name}/on"] = qm[pm].cpu().numpy()
        names.append(name); del mk, fm, lXg, lMg
    out["names"] = np.array(names); np.savez(fn, **out)
    rel = np.arange(npred); late = rel >= a.ton
    ev = lambda x, m: 1000.0 * (x[m] >= 1).mean() if m.sum() else float("nan")
    print(f"[{it + 1}/{len(pick)}] {rid} s{k} think {te} img {len(img)} ok {out['ok']} | before T_on: KL {out['kl0'][~late].mean():.3f} events/1k {ev(out['kl0'], ~late):.0f} (max |change| under the knobs {max(float(np.abs(out[n + '/kl'][~late] - out['kl0'][~late]).max()) for n in names):.1e}) | "
          f"after T_on: base KL {out['kl0'][late].mean():.3f} ev {ev(out['kl0'], late):.0f} | " + " | ".join(f"{n}: KL {out[n + '/kl'][late].mean():.3f} ev {ev(out[n + '/kl'], late):.0f} dist {out[n + '/kl_x0g'][late].mean():.3f} dlp {(out[n + '/lpy'][late] - out['lpy0'][late]).mean():+.3f}" for n in names) + f" | {time.time() - t0:.0f}s", flush=True)
    del lX0, lM0
    torch.cuda.empty_cache()
# summary
R = [np.load(os.path.join(a.out, f), allow_pickle=True) for f in sorted(os.listdir(a.out)) if f.endswith(".npz")]
if R:
    names = R[0]["names"].tolist()
    print(f"\n{len(R)} reads ({sum(not bool(r['ok']) for r in R)} ended wrong); knob in the full-attention blocks {FB}, on from thinking token {a.ton}; mean over reads")
    print("condition | thinking-token bin | mean KL(p_X || p_M) base -> knob | events per 1,000 tokens base -> knob | new events per 1,000 (KL >= 1 with the knob, < 0.5 without) | disturbance KL(p_X || p_X,knob) | change of log p(read token) | share of positions with the knob on")
    for n in names:
        for lo_, hi_ in zip(EDGES[:-1], EDGES[1:]):
            row = []
            for r in R:
                rel = np.arange(len(r["kl0"])); m = (rel >= lo_) & (rel < hi_)
                if m.sum() < 20:
                    continue
                k0, kg = r["kl0"][m], r[f"{n}/kl"][m]
                row.append((k0.mean(), kg.mean(), 1000 * (k0 >= 1).mean(), 1000 * (kg >= 1).mean(), 1000 * ((kg >= 1) & (k0 < 0.5)).mean(), r[f"{n}/kl_x0g"][m].mean(), (r[f"{n}/lpy"][m] - r["lpy0"][m]).mean(), r[f"{n}/on"][m].mean()))
            if row:
                x = np.array(row).mean(0)
                print(f"{n} | [{lo_}, {hi_ if hi_ < 10 ** 8 else 'end'}) n {len(row)} | {x[0]:.3f} -> {x[1]:.3f} | {x[2]:.1f} -> {x[3]:.1f} | {x[4]:.1f} | {x[5]:.4f} | {x[6]:+.4f} | {x[7]:.2f}")
print("done", flush=True)
