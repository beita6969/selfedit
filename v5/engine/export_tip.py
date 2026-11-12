"""Materialise a tip's weights: stock + the tip's edit list baked into a HF checkpoint (Surgeon.export: edited tensors read back
from the live engine, unchanged shards hard-linked), plus a reference for verification: N learn rows, one request at a time,
temperature 0, max_tokens 1024, with the edit applied in this Surgeon (= what a W worker reads).
usage: python export_tip.py TIP_ID EDIT_JSON_FILE_OR_TIPS_JSON [--n 10]
writes /data/fd/mingdazhang/selfedit/v4/ckpt/<tip>/ (checkpoint) and ckpt/<tip>.export.json (edits, edit_sha, reference outputs)"""
import argparse, hashlib, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import MODEL, EFFORT, content_for
from wedit import surgeon_edit
from surgery import Surgeon

ap = argparse.ArgumentParser()
ap.add_argument("tip")
ap.add_argument("edits")
ap.add_argument("--n", type=int, default=10)
ap.add_argument("--util", type=float, default=0.85)
args = ap.parse_args()
V4 = "/data/fd/mingdazhang/selfedit/v4"
canon = lambda o: json.dumps(o, sort_keys=True, separators=(",", ":"))
j = json.load(open(args.edits))
edits = j["edits"] if isinstance(j, dict) and "edits" in j else j
es = hashlib.sha256(canon(edits).encode()).hexdigest()[:16]
out = f"{V4}/ckpt/{args.tip}"
if os.path.exists(out):
    sys.exit(f"{out} exists")
os.makedirs(f"{V4}/ckpt", exist_ok=True)
t0 = time.time()
S = Surgeon(MODEL, tp=1, max_model_len=46000 if args.n else 8192, max_num_seqs=16, max_num_batched_tokens=8192, gpu_memory_utilization=args.util,
            kv_cache_dtype="fp8", multimodal=True, enable_prefix_caching=False)
print(f"surgeon up {time.time() - t0:.0f}s", flush=True)
prims = surgeon_edit(edits)
t1 = time.time()
man = S.export(prims, out)
print(f"exported to {out} in {time.time() - t1:.0f}s", flush=True)
rows = [json.loads(l) for l in open(f"{V4}/rows/learn.jsonl")]
rows = [r for r in rows if r.get("image_paths")][3::37][:args.n]
ref = []
h = S.apply(prims) if rows else None
try:
    for r in rows:
        c, nt = content_for(r["prompt"], r["image_paths"])
        o = S.generate([[{"role": "user", "content": c}]], n=1, sampling=dict(temperature=0.0), max_tokens=1024, reasoning_effort=EFFORT, seeds=[0])[0]
        s = o["samples"][0]
        ref.append(dict(row=r["id"], prompt_tokens=o["prompt_tokens"], token_ids=s["token_ids"], finish=s["finish_reason"]))
        print(r["id"], o["prompt_tokens"], s["num_tokens"], flush=True)
finally:
    rep = S.revert(h) if h is not None else None
print("reverted", rep, flush=True)
json.dump(dict(tip=args.tip, edits=edits, edit_sha=es, ckpt=out, model=MODEL, manifest=man if isinstance(man, (dict, list)) else str(man),
               t=time.time(), reference=ref), open(f"{V4}/ckpt/{args.tip}.export.json", "w"))
print("done", es, flush=True)
