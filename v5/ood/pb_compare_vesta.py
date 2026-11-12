# Vesta copy of ~/selfedit/ood_cases/pb_compare.py: pinned upstream file (builder's input) and ood128_v2.jsonl; images popped, n_images = len(image).
import json, re
rows = {r["id"]: r for r in map(json.loads, open("/data/fd/mingdazhang/selfedit/data/ood128/ood128_v2.jsonl")) if r["bench"] == "perceptionbench"}
SUF = ["Answer with a number. Put the final number in \\boxed{}.", "Put your final answer in \\boxed{}, using the answer format the question asks for."]
RX = r"\s*(?:\(?\s*(?:Just |Only )?(?:answer|output|respond)[^.?]*?(?:number|digit|letter|word)[^.?]*(?:directly)?\.?\)?\s*)$"
n = same = cut = other = err = 0; cuts = []; others = []; ansdiff = []
for l in open("/data/fd/mingdazhang/selfedit/data/ood_cand/raw/moonshotai__PerceptionBench/PerceptionBench.jsonl"):
    o = json.loads(l); o["n_images"] = len(o.pop("image") or [])
    if "error" in o: err += 1; continue
    rid = f"perceptionbench/pb{int(o['index']):04d}"
    if rid not in rows: continue
    n += 1; r = rows[rid]; p = r["prompt"].rstrip()
    for s in SUF:
        if p.endswith(s): p = p[: -len(s)].rstrip()
    q0 = re.sub(r"<\|image_\d+\|>", "", o["problem"]).strip()
    if o.get("hint"): q0 = o["hint"].strip() + "\n\n" + q0
    if str(o["answer"]).strip() != str(r["answer"]).strip(): ansdiff.append((rid, o["answer"], r["answer"]))
    if o.get("n_images") != len(r["image_paths"]): others.append((rid, f"images {o.get('n_images')} vs {len(r['image_paths'])}"))
    if p == q0.rstrip(): same += 1
    elif q0.startswith(p):
        cut += 1; cuts.append((rid, q0[len(p):], p[-60:]))
    else:
        other += 1; others.append((rid, "text differs: " + repr(q0[-120:]) + " VS " + repr(p[-120:])))
print(f"compared {n} rows (fetch errors {err}): identical {same}, tail removed by our regex {cut}, other differences {other}, gold differs {len(ansdiff)}")
for rid, rem, tail in cuts:
    m = re.search(RX, rem + "", flags=re.I)
    kept = rows[rid]["prompt"]; unbal = kept.count("(") - kept.count(")")
    print(f"  CUT {rid}: kept ...{tail!r}  REMOVED {rem!r}  | regex-on-removed full match {bool(m and m.start() == 0)} | kept prompt open brackets {unbal}")
for x in others: print("  OTHER", x)
for x in ansdiff: print("  GOLD", x)
