"""chk_proc (fable_int, consolidation): the contrast A (prompt as read, routed prefix kept) against B (prefix removed) on identical tokens: alignment, the inserted block, size of the effect.
python chk_proc.py ITEMS [--limit n]   (env FI_BASE_NOPREFIX=1: "medium" = B, "proc" = A)"""
import os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import intlib
items = [(l.split()[0], int(l.split()[1])) for l in open(sys.argv[1]) if l.strip()]
if "--limit" in sys.argv:
    items = items[: int(sys.argv[sys.argv.index("--limit") + 1])]
lab = intlib.Lab(f32=False); dev = lab.dev
tot = []
for it, (rid, k) in enumerate(items):
    t0 = time.time(); gen, te = lab.gen_ids(rid, k)
    cb = lab.prep(rid, "medium", gen); ca = lab.prep(rid, "proc", gen)
    A, S = lab.align(cb, ca); P = cb.P
    npred = min(te + 1, len(gen)); pm = torch.arange(P - 1, P - 1 + npred, device=dev)
    lpb = lab.logp(lab.run(cb)[0][pm]); lpa = lab.logp(lab.run(ca)[0][pm + S])
    kl = intlib.kl(lpa, lpb).cpu().numpy(); y = torch.tensor(gen[:npred], device=dev)
    dlp = (lpa.gather(1, y[:, None]) - lpb.gather(1, y[:, None]))[:, 0].cpu().numpy()
    nimg = int(cb.isimg.sum()); ntxt_after = P - 5 - A
    if it < 2:
        print("inserted block:", repr(lab.tk.decode(ca.ids[A: A + S])[:90]), "...", repr(lab.tk.decode(ca.ids[A + S - 6: A + S + 6])), "| before:", repr(lab.tk.decode(ca.ids[A - 4: A])))
    tix = np.arange(npred)
    rec = int(lab.D[rid]["samples"][str(k)].get("prompt_tokens") or 0)
    print(f"[{it + 1}/{len(items)}] {rid} s{k} think {te} P(A) {P + S} recorded prompt tokens {rec} P {P} img {nimg} block at {A} len {S} question tokens after block {ntxt_after} | KL(A||B) mean {kl.mean():.4f} median {np.median(kl):.4f} "
          f"first 30 {kl[:30].mean():.3f} 30-150 {kl[30:150].mean():.3f} 150+ {(kl[150:].mean() if npred > 150 else float('nan')):.3f} | log p(read token) A - B mean {dlp.mean():+.4f} | {time.time() - t0:.0f}s", flush=True)
    tot.append((kl.mean(), dlp.mean(), te))
t = np.array(tot); print(f"{len(t)} traces: mean KL(A||B) {t[:, 0].mean():.4f} (xhigh against medium was 0.08-0.11); mean log p(read token) A - B {t[:, 1].mean():+.4f}")
