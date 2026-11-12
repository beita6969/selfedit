"""pos_cap (fable_int): MLP inputs of blocks LO..HI at log-spaced thinking positions of LONG reads (for the length key). One medium-template forward per read (bf16, blocks <= HI).
OUT/<row>__s<k>.npz: X [npos, HI-LO+1, H] f16, t [npos] (thinking tokens read so far), te, row, bench."""
import argparse, json, os, sys, time, zlib
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import intlib
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--lo", type=int, default=24); ap.add_argument("--hi", type=int, default=34)
ap.add_argument("--n", type=int, default=30); ap.add_argument("--tmin", type=int, default=5000); ap.add_argument("--tmax", type=int, default=18000); ap.add_argument("--npos", type=int, default=24)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
lab = intlib.Lab(f32=False); ST, dev = lab.ST, lab.dev; BL = list(range(a.lo, a.hi + 1))
cand = []
for r in sorted(lab.D):
    for k, s in lab.D[r]["samples"].items():
        if a.tmin <= int(s["tokens"]) <= a.tmax and int(s.get("prompt_tokens") or 0) <= 6500 and s.get("reasoning") and s.get("finish") == "stop":
            cand.append((zlib.crc32(f"{r}#{k}".encode()), r, int(k)))
cand.sort(); pick, per_row, per_b = [], {}, {}
for _, r, k in cand:
    b = lab.D[r]["bench"]
    if per_row.get(r, 0) < 1 and per_b.get(b, 0) < max(4, a.n // 3) and len(pick) < a.n:
        pick.append((r, k)); per_row[r] = 1; per_b[b] = per_b.get(b, 0) + 1
print(f"{len(cand)} long reads; picked {len(pick)}: {per_b}", flush=True)
for it, (rid, k) in enumerate(pick):
    fn = os.path.join(a.out, f"{rid.replace('/', '__')}__s{k}.npz")
    if os.path.exists(fn):
        continue
    t0 = time.time()
    toks = lab.tk(lab.D[rid]["samples"][str(k)]["reasoning"], add_special_tokens=False).input_ids; te = len(toks)
    ts = np.unique(np.round(np.geomspace(64, te, a.npos)).astype(int))
    c = lab.prep(rid, "medium", toks); P = c.P
    pos = torch.tensor(P - 1 + ts, device=dev)
    ST["cap"] = dict(mlpin=lambda l, o: o[0][pos].float().cpu() if l in BL else None); ST["out"] = {}
    try:
        lab.run(c, stop=a.hi + 1)
    except torch.OutOfMemoryError:
        lab.clear(); torch.cuda.empty_cache(); print("OOM", rid, te, flush=True); continue
    X = torch.stack([ST["out"][("mlpin", l)] for l in BL], 1).numpy().astype(np.float16); lab.clear()
    np.savez(fn, X=X, t=ts, te=te, row=rid, bench=lab.D[rid]["bench"])
    print(f"[{it + 1}/{len(pick)}] {rid} s{k} think {te} T {c.T} {time.time() - t0:.0f}s", flush=True)
print("done", flush=True)
