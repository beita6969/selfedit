"""E-path reads of tip t007c's view on OOD v2 rows, samples 0-1. Order: ood_tipE (tip view as is) is queued FIRST, then sub_e1.py
(ood_e1always, sig_e1always, sig_tipE), then ood_floor8k (single-image rows at min_image_tokens 8192; multi-image rows unchanged;
fall back to the tip's own floor where image tokens + text + prefix + 64 + 32768 > 46000; same rule as the XFER builder; text tokens
estimated as chars/3 + 200). Image token counts cached per path (samples share images)."""
import json, os, subprocess, sys, time
sys.path.insert(0, "/workspace/fudan/mingdazhang/selfedit-v4/engine_v4")
V4 = "/data/fd/mingdazhang/selfedit/v4"; MAXLEN, RMAX = 46000, 32768
def put(name, L):
    p = f"{V4}/q/in/{name}.jsonl"; open(p + ".tmp", "w").write("".join(json.dumps(x) + "\n" for x in L)); os.replace(p + ".tmp", p)
    open(f"{V4}/scratch/ext/submit_times.jsonl", "a").write(json.dumps(dict(batch=name, t_submit=time.time(), n=len(L))) + "\n")
    print(name, len(L), flush=True); time.sleep(1.1)
rows = {r["id"]: r for r in map(json.loads, open(f"{V4}/rows/learn.jsonl")) if r.get("v4_role") == "ood_v2"}
tq = [json.loads(l) for l in open(f"{V4}/scratch/ext/tip_ood.jsonl")]; tq = [q for q in tq if q["sample"] in (0, 1)]
A = [dict(q, key="otip|" + q["key"], engine="E", priority=-10) for q in tq]; assert len({a["key"] for a in A}) == len(A)
if "--no-tipE" not in sys.argv: put("ood_tipE", A)
if "--no-e1" not in sys.argv: subprocess.run(["python3", "/workspace/fudan/mingdazhang/selfedit-v4/ood/sub_e1.py"])
import common
B, fb, single, need_max, cache = [], [], 0, 0, {}
for q in tq:
    r = rows[q["row"]]; ims = r["image_paths"] if isinstance(r["image_paths"], list) else [r["image_paths"]]
    b = dict(q, key="of8k|" + q["key"], engine="E", priority=-9); s = dict(q.get("settings") or {})
    if len(ims) == 1:
        single += 1
        ck = (ims[0], s.get("max_image_tokens"))
        if ck not in cache: cache[ck] = common.image_url_for(ims[0], 8192, s.get("max_image_tokens"))[1]
        n8 = cache[ck]
        txt = len((r.get("prompt") or "") + json.dumps(r.get("options") or "")) // 3 + 200
        need = n8 + txt + len(q.get("prefix") or "") // 3 + 64 + RMAX; need_max = max(need_max, need)
        if need <= MAXLEN: s["min_image_tokens"] = 8192
        else: fb.append(q["row"])
    b["settings"] = s; B.append(b)
assert len({b["key"] for b in B}) == len(B)
put("ood_floor8k", B)
print(f"floor8k {len(B)}; single-image requests {single}; fell back {len(fb)} ({sorted(set(fb))[:10]}); max estimated need {need_max} (limit {MAXLEN})")
