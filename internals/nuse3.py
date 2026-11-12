"""nuse3 (fable_int): per block (all 64) the neurons whose removal costs least: ranked by mean |activation| x |down_proj column| over all positions of 10 reads
(5 routed rows without the prefix, 5 unrouted rows). Neurons used by any existing cell (used_neurons.json) are excluded. Writes OUT: {block: [96 neuron ids, smallest write first]}.
python nuse3.py ITEMS_A ITEMS_B USED.json OUT.json"""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import intlib
ia, ib, usedf, outf = sys.argv[1:5]
assert not os.path.exists(outf)
rd = lambda p: [(l.split()[0], int(l.split()[1])) for l in open(p) if l.strip()]
items = rd(ia)[:5] + rd(ib)[::4][:5]
used = {int(k): set(v) for k, v in json.load(open(usedf)).items()}
lab = intlib.Lab(f32=False); ST, dev = lab.ST, lab.dev
use = {l: torch.zeros(lab.layers[l].mlp.down_proj.weight.shape[1], dtype=torch.float64, device=dev) for l in range(lab.NL)}; cnt = 0


def cap(l, x):
    use[l].add_(x[0].float().abs().sum(0).double())
    return None


for rid, k in items:
    gen, te = lab.gen_ids(rid, k); c = lab.prep(rid, "medium", gen)
    ST["cap"] = dict(neu=cap); lab.run(c); lab.clear(); cnt += c.T
out = {}
for l in range(lab.NL):
    act = (use[l] / cnt).cpu().numpy(); dn = lab.layers[l].mlp.down_proj.weight.float().norm(dim=0).cpu().numpy(); wr = act * dn
    order = np.argsort(wr); out[l] = [int(x) for x in order if int(x) not in used.get(l, set())][:96]
    if l % 8 == 7 or 24 <= l <= 34:
        print(f"block {l}: mean write (|act| x |down|): median {np.median(wr):.3e}; 96 smallest free: {wr[out[l]].mean():.3e} (first 16: {wr[out[l][:16]].mean():.3e})", flush=True)
json.dump(out, open(outf, "w")); print("wrote", outf, f"({len(items)} reads, {cnt} positions)")
