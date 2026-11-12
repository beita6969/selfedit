"""python3 reset_report.py NAME ARM_BATCHES(comma) [--tails OUT.txt] : reset arm vs candidate FLs8_L24-34_+0.75 vs W-null on the same (row, sample)."""
import json, statistics as st, sys
sys.path.insert(0, "/workspace/fudan/mingdazhang/selfedit-v4/loop")
import ecomp
V4 = "/data/fd/mingdazhang/selfedit/v4"
name, arm = sys.argv[1], sys.argv[2].split(","); tails = sys.argv[sys.argv.index("--tails") + 1] if "--tails" in sys.argv else ""
rows = {json.loads(l)["id"]: json.loads(l) for l in open(f"{V4}/rows/learn.jsonl")}
def load(bs):
    d = {}
    for b in bs:
        for l in open(f"{V4}/q/out/{b}.jsonl"):
            x = json.loads(l); k = x["key"].split("|")
            if not x.get("error") and x.get("content") is not None: d[(k[2], int(k[3]))] = x
    return d
C, W, R = load(["fls8_L24_34_p075_s0", "fls8_L24_34_p075_s1"]), load(["wnull_sig_s0", "wnull_sig_s1"]), load(arm)
L = json.load(open(f"{V4}/scratch/ext/reset_rows_fl075_12k.json"))
pr = lambda x, r: ecomp.pred(x, rows[r]) if x else None
out = []; miss = 0
for d in L:
    k = (d["row"], d["sample"]); c, w, r = C.get(k), W.get(k), R.get(k)
    if not r: miss += 1; continue
    pc, pr_ = pr(c, k[0]), pr(r, k[0])
    out.append(dict(k=k, ct=c["tokens"], rt=r["tokens"], wt=w.get("tokens") if w else None, co=bool(c.get("ok")), ro=bool(r.get("ok")), wo=bool(w and w.get("ok")),
                    rf=r.get("finish"), lp=bool(r.get("loop")), pt=(r.get("prompt_tokens"), c.get("prompt_tokens")), cf=c.get("finish"), same=(str(pc).strip().lower() == str(pr_).strip().lower()), pc=pc, prr=pr_, r=r))
n = len(out)
print(f"{name}: {n}/{len(L)} reads ({miss} missing)")
if not n: sys.exit()
print(f"tokens candidate -> reset: total {sum(o['ct'] for o in out)} -> {sum(o['rt'] for o in out)} (x{sum(o['rt'] for o in out)/sum(o['ct'] for o in out):.2f}); "
      f"median {st.median(o['ct'] for o in out):.0f} -> {st.median(o['rt'] for o in out):.0f}; per-read ratio median x{st.median(o['rt']/o['ct'] for o in out):.2f}; W-null total {sum(o['wt'] or 0 for o in out)}")
print(f"finish reset: stop {sum(o['rf']=='stop' for o in out)} / length(cut) {sum(o['rf']=='length' for o in out)}; candidate cuts {sum(o['cf']=='length' for o in out)}")
print(f"loops {sum(o['lp'] for o in out)}; prompt tokens identical to candidate {sum(o['pt'][0]==o['pt'][1] for o in out)}/{n}")
print(f"answers identical to candidate {sum(o['same'] for o in out)}/{n}, changed {n-sum(o['same'] for o in out)}")
print(f"accuracy: W-null {sum(o['wo'] for o in out)}/{n}  candidate {sum(o['co'] for o in out)}/{n}  reset {sum(o['ro'] for o in out)}/{n}  "
      f"(cand->reset w->r {sum((not o['co']) and o['ro'] for o in out)}, r->w {sum(o['co'] and not o['ro'] for o in out)})")
for o in out:
    print(f"  {o['k'][0]:34s} s{o['k'][1]} tok W {o['wt']} | cand {o['ct']} -> reset {o['rt']} ({o['rf']}{' LOOP' if o['lp'] else ''}) ok W{int(o['wo'])} C{int(o['co'])} R{int(o['ro'])} same {int(o['same'])} pred {str(o['pc'])[:12]!r}->{str(o['prr'])[:12]!r}")
if tails:
    big = [o for o in out if o["ct"] > 16000][:10]
    with open(tails, "w") as f:
        for o in big:
            f.write(f"===== {o['k'][0]} s{o['k'][1]} | candidate {o['ct']} tok ok {int(o['co'])} -> reset {o['rt']} tok finish {o['rf']} ok {int(o['ro'])} pred {o['prr']!r}\n")
            f.write("--- reset tail:\n" + (o["r"].get("reasoning") or "")[-1500:] + "\n--- candidate tail:\n" + (C[o["k"]].get("reasoning") or "")[-1500:] + "\n\n")
    print(f"tails: {len(big)} reads -> {tails}")
