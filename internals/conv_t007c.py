"""conv_t007c (fable_int, consolidation): the coordinator's export of tip t007c reads (signal rows routed to g005c23 or g005c29) -> traces files in intlib's layout, one per entry,
only rows where that entry is the ONLY routed procedure (fired = {g001c34, entry}), so that contrast B = the prompt with the whole prefix removed.
Each row keeps the message parts as sent ("parts": image parts with the node path, min / max image tokens; text parts verbatim); image paths mapped from Vesta to the node and checked by sha256.
python conv_t007c.py EXPORT.jsonl.gz OUT_PREFIX"""
import gzip, hashlib, json, os, sys, collections
import numpy as np
src, out = sys.argv[1], sys.argv[2]
PF, PT = "/data/fd/mingdazhang/selfedit/", "/home/bedicloud/sharestore2/zxc/selfedit/"
R = [json.loads(l) for l in gzip.open(src, "rt")]
rows = collections.defaultdict(list)
for r in R:
    rows[r["row"]].append(r)
sha_ok = {}
def node_path(p, sha):
    q = p.replace(PF, PT)
    if q not in sha_ok:
        sha_ok[q] = os.path.exists(q) and hashlib.sha256(open(q, "rb").read()).hexdigest() == sha
    return q
for ent in ("g005c23", "g005c29"):
    D = {}; bad = 0
    for rid, L in sorted(rows.items()):
        fired = {e["cid"] for e in L[0]["fired"]}
        if fired != {"g001c34", ent}:
            continue
        r0 = L[0]; msg = r0["messages"]; assert len(msg) == 1 and msg[0]["role"] == "user"
        parts = []
        for p in msg[0]["content"]:
            if p["type"] == "text":
                parts.append({"type": "text", "text": p["text"]})
            else:
                u = p["image_url"]; q = node_path(u["path"], u["sha256"]); bad += not sha_ok[q]
                parts.append({"type": "image", "path": q, "min_image_tokens": u.get("min_image_tokens"), "max_image_tokens": u.get("max_image_tokens"), "image_tokens": u.get("image_tokens")})
        pre = r0["request"]["prefix"]; tp = [p["text"] for p in parts if p["type"] == "text"]
        pidx = [i for i, p in enumerate(parts) if p["type"] == "text" and p["text"].strip() == pre.strip()]
        D[rid] = dict(bench=r0["bench"], role=r0["v4_role"], kind=r0["kind"], answer=r0["answer_gold"], options=r0.get("options"), prefix=pre, settings=r0["request"].get("settings"), parts=parts,
                      prefix_part=(pidx[0] if len(pidx) == 1 else None), entry=ent, image_paths=[p["path"] for p in parts if p["type"] == "image"], prompt="".join(t for i, t in enumerate(tp)),
                      samples={str(x["sample"]): dict(reasoning=x["reasoning"], content=x["content"], tokens=x["tokens"], prompt_tokens=x["prompt_tokens"], ok=x["ok"], finish=x["finish"], pred=x.get("pred")) for x in L})
    json.dump(D, open(f"{out}_{ent}.json", "w"))
    tk = [s["tokens"] for v in D.values() for s in v["samples"].values()]; pt = [next(iter(v["samples"].values()))["prompt_tokens"] for v in D.values()]
    print(ent, "rows", len(D), dict(collections.Counter(v["bench"] for v in D.values())), dict(collections.Counter(v["role"] for v in D.values())), "| images not found / sha mismatch:", bad,
          "| rows without a unique prefix part:", sum(v["prefix_part"] is None for v in D.values()), "| distinct prefixes", len({v["prefix"] for v in D.values()}),
          "| read tokens median", int(np.median(tk)), "p25", int(np.percentile(tk, 25)), "p75", int(np.percentile(tk, 75)), "| prompt tokens median", int(np.median(pt)), "max", max(pt), "<= 6500:", sum(p <= 6500 for p in pt), "<= 9000:", sum(p <= 9000 for p in pt),
          "| accuracy", round(float(np.mean([bool(s["ok"]) for v in D.values() for s in v["samples"].values()])), 3))
    v = next(iter(D.values())); print("  parts:", [(p["type"], (p["text"][:60] if p["type"] == "text" else p["image_tokens"])) for p in v["parts"]]); print("  prefix:", v["prefix"][:300])
