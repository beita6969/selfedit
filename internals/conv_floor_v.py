"""conv_floor_v (fable_int, E1 exposure check): the engine's export of floor-only base reads (no routed text, no cell; key x|2988ed164be18b6a|row|s) -> a traces file in intlib's v4 layout
for the rows where the tip t005r routes a prefix: each row keeps the t005r record (prompt, routed prefix, image paths, settings, answer) and gets the floor-only reads as its samples.
With FI_BASE_NOPREFIX=1, mode "medium" is then the context these reads were generated under, and mode "proc" adds the routed text.
python conv_floor_v.py EXPORT.jsonl.gz T005R_TRACES.json OUT.json"""
import gzip, json, sys, collections
src, tr, out = sys.argv[1:4]
D = json.load(open(tr)); E = collections.defaultdict(dict)
for l in gzip.open(src, "rt"):
    r = json.loads(l); s = r.get("s", r.get("sample")); rs = r.get("reasoning") or r.get("reasoning_full") or r.get("think")
    if rs is None or r["row"] not in D:
        continue
    E[r["row"]][str(s)] = dict(reasoning=rs, content=r.get("content") or "", tokens=r["tokens"], prompt_tokens=r.get("prompt_tokens"), ok=bool(r["ok"]), finish=r.get("finish"), pred=r.get("pred"))
O = {}
for rid, v in D.items():
    if isinstance(v, dict) and v.get("prefix") and rid in E:
        w = {k: x for k, x in v.items() if k != "samples"}; w["samples"] = E[rid]; O[rid] = w
json.dump(O, open(out, "x"))
tk = [int(s["tokens"]) for v in O.values() for s in v["samples"].values()]
print(f"{len(O)} routed rows with floor-only reads, {len(tk)} reads; accuracy {sum(s['ok'] for v in O.values() for s in v['samples'].values()) / max(1, len(tk)):.3f}; "
      f"tip t005r (with the text) on the same rows {sum(bool(s['ok']) for r in O for s in D[r]['samples'].values()) / max(1, sum(len(D[r]['samples']) for r in O)):.3f}; thinking+answer tokens median {sorted(tk)[len(tk) // 2] if tk else 0}")
by = collections.defaultdict(lambda: [0, 0, 0, 0])
for r, v in O.items():
    b = by[v["bench"]]; b[0] += sum(s["ok"] for s in v["samples"].values()); b[1] += len(v["samples"]); b[2] += sum(bool(s["ok"]) for s in D[r]["samples"].values()); b[3] += len(D[r]["samples"])
for b, x in sorted(by.items()):
    print(f"  {b}: floor-only {x[0] / max(1, x[1]):.3f} ({x[1]} reads) | t005r with the text {x[2] / max(1, x[3]):.3f} ({x[3]} reads)")
