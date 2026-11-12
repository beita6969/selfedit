"""h_ans (fable_int, hindsight): is the hindsight line an effective teacher at the commit? For multiple-choice reads, at the position that predicts the option letter inside the final
\\boxed{...} of the cached read (the read's own text is teacher-forced up to there), compare base and H: p(gold letter), p(the read's letter), and which letter is the arg-max among the options.
Reads that ended wrong (the hint contradicts the read's text) and reads that ended right. Optional --decoy: the same with the read's own wrong letter stated as the correct answer.
python h_ans.py ITEMS OUT.txt [--limit n]"""
import os, re, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import intlib
items = [(l.split()[0], int(l.split()[1])) for l in open(sys.argv[1]) if l.strip()]
if "--limit" in sys.argv:
    items = items[: int(sys.argv[sys.argv.index("--limit") + 1])]
lab = intlib.Lab(f32=False); dev = lab.dev; D = lab.D; tk = lab.tk
rows = []
for it, (rid, k) in enumerate(items):
    v = D[rid]; s = v["samples"][str(k)]
    if v["kind"] != "mc" or not re.fullmatch(r"[A-H]", str(v["answer"]).strip()) or not re.fullmatch(r"[A-H]", str(s.get("pred") or "").strip()):
        continue
    gold, pred = str(v["answer"]).strip(), str(s["pred"]).strip()
    gen, te = lab.gen_ids(rid, k)
    txt = [tk.decode([t]) for t in gen]; j = None
    for i in range(len(gen) - 1, te, -1):                 # last token after the thinking block that is exactly the read's letter, preceded by "boxed{"
        if txt[i].strip() == pred and "boxed" in "".join(txt[max(0, i - 4): i]):
            j = i; break
    if j is None:
        continue
    ids = {L: tk(L, add_special_tokens=False).input_ids for L in "ABCDEFGH"}
    if tk.decode([gen[j]]) != pred or any(len(x) != 1 for x in ids.values()):
        continue
    cb = lab.prep(rid, "medium", gen); ch = lab.prep(rid, "hind", gen); _, S = lab.align(cb, ch); P = cb.P
    pos = torch.tensor([P - 1 + j], device=dev)
    lb = lab.logp(lab.run(cb)[0][pos])[0]; lh = lab.logp(lab.run(ch)[0][pos + S])[0]
    opts = [L for L in "ABCDEFGH" if L in (v.get("options_letters") or "ABCD") or L in (gold, pred)]
    g, p_ = ids[gold][0], ids[pred][0]; oi = [ids[L][0] for L in opts]
    r = dict(row=rid, ok=bool(s["ok"]), gold=gold, pred=pred, pb_gold=float(lb[g].exp()), ph_gold=float(lh[g].exp()), pb_pred=float(lb[p_].exp()), ph_pred=float(lh[p_].exp()),
             top_b=opts[int(torch.argmax(lb[oi]))], top_h=opts[int(torch.argmax(lh[oi]))])
    rows.append(r); print(f"[{it + 1}/{len(items)}] {rid} s{k} ok {r['ok']} gold {gold} read {pred} | p(gold) base {r['pb_gold']:.3f} H {r['ph_gold']:.3f} | p(read's letter) base {r['pb_pred']:.3f} H {r['ph_pred']:.3f} | top base {r['top_b']} H {r['top_h']}", flush=True)
with open(sys.argv[2], "w") as f:
    for name, sel in (("ended WRONG (the hint contradicts the read)", [r for r in rows if not r["ok"]]), ("ended right", [r for r in rows if r["ok"]])):
        if not sel:
            continue
        line = (f"multiple-choice reads that {name}: {len(sel)} | p(gold letter) at the boxed answer: base {np.mean([r['pb_gold'] for r in sel]):.3f} -> H {np.mean([r['ph_gold'] for r in sel]):.3f} | "
                f"p(the read's letter): base {np.mean([r['pb_pred'] for r in sel]):.3f} -> H {np.mean([r['ph_pred'] for r in sel]):.3f} | arg-max option = gold: base {np.mean([r['top_b'] == r['gold'] for r in sel]):.2f} -> H {np.mean([r['top_h'] == r['gold'] for r in sel]):.2f}")
        print(line); f.write(line + "\n")
print("done", flush=True)
