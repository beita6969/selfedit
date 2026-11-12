"""conv_floor_parts (fable_int, law table): a traces file in the "parts" layout (t007c export rows of one routed entry) with its samples replaced by the engine's floor-only base reads
(no routed text, no cell; export e13): the same rows, reads generated WITHOUT the text. With FI_BASE_NOPREFIX=1, mode "medium" is the context these reads were generated under and mode
"proc" adds the routed text. python conv_floor_parts.py TRACES_IN.json EXPORT.jsonl.gz OUT.json"""
import gzip, json, sys, collections
src, exp, out = sys.argv[1:4]
D = json.load(open(src)); E = collections.defaultdict(dict)
for l in gzip.open(exp, "rt"):
    r = json.loads(l)
    if r["row"] in D and r.get("reasoning"):
        E[r["row"]][str(r["s"])] = dict(reasoning=r["reasoning"], content=r.get("content") or "", tokens=r["tokens"], prompt_tokens=r.get("prompt_tokens"), ok=bool(r["ok"]), finish=r.get("finish"), pred=r.get("pred"))
O = {}
for rid, v in D.items():
    if rid in E:
        w = {k: x for k, x in v.items() if k != "samples"}; w["samples"] = E[rid]; O[rid] = w
json.dump(O, open(out, "x"))
n0 = sum(len(v["samples"]) for v in D.values()); n1 = sum(len(v["samples"]) for v in O.values())
print(f"{len(O)} of {len(D)} rows; reads with the text {n0}, accuracy {sum(bool(s['ok']) for v in D.values() for s in v['samples'].values()) / max(1, n0):.3f}; floor-only reads {n1}, accuracy {sum(s['ok'] for v in O.values() for s in v['samples'].values()) / max(1, n1):.3f}")
