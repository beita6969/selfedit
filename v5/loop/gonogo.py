"""Go/no-go extract of one v4 run generation (read only): the P0 entry, router share and two router prompts, one executor prompt as
sent (content layout of engine_v4/common.content_for, images shown as [image k: path]), D/se/lfsr per candidate, wall per step.
usage: gonogo.py RUN_DIR [GEN]"""
import glob, hashlib, json, os, sys

V4 = "/data/fd/mingdazhang/selfedit/v4"
RD = sys.argv[1]
g = sys.argv[2] if len(sys.argv) > 2 else "1"
S = json.load(open(os.path.join(RD, "state.json")))
G = S["gens"][g]
rows = {r["id"]: r for r in map(json.loads, open(f"{V4}/rows/learn.jsonl"))}
ROUTE_V1 = "Condition: {c}.\nDo not solve the question above. Does this condition hold for it? Reply with exactly one word: yes or no."


def results(prefix):
    out = {}
    for f in glob.glob(f"{V4}/q/out/*.jsonl"):
        for l in open(f):
            try:
                r = json.loads(l)
            except ValueError:
                continue
            if r.get("key", "").startswith(prefix) and "error" not in r:
                out[r["key"]] = r
    return out


def layout(prompt, imgs, prefix=None):
    parts = prompt.split("<image>")
    tags = [f"[image {k + 1}: {os.path.basename(p)}]" for k, p in enumerate(imgs)]
    if len(parts) - 1 == len(imgs) and imgs:
        out = ([prefix.rstrip() + "\n\n"] if prefix else [])
        for k, t in enumerate(parts):
            if t.strip():
                out.append(t.strip())
            if k < len(imgs):
                out.append(tags[k])
        return out
    return tags + ([prefix.rstrip() + "\n\n"] if prefix else []) + [prompt]


print(f"run {RD} gen {g}  slots {G.get('slots')}  sources {G.get('sources')}  repairs {G.get('repairs')}")
print("wall per step (s):", G.get("times"))
for c in G["cands"]:
    j = c.get("judge") or {}
    print(f"\n== {c['cid']} {c['author']} {c.get('kind')} source {c.get('card_source')} (card {c.get('card_wrong')} wrong + {c.get('card_right')} right) status: {c['status']}")
    e = c.get("entry") or {}
    print("   condition:", json.dumps(e.get("condition"), ensure_ascii=False))
    print("   procedure:", json.dumps(e.get("procedure"), ensure_ascii=False))
    print("   settings :", json.dumps(e.get("settings")))
    print("   why      :", json.dumps(c.get("why"), ensure_ascii=False))
    print(f"   routed {c.get('routed')}/{len([r for r in rows.values() if r['v4_role'] == 'judge'])}  route {c.get('route')}")
    print(f"   D {j.get('D')}  se {j.get('se')}  lfsr {j.get('lfsr')}  post_mean {j.get('post_mean')}  clusters {j.get('clusters')}  rows read {c.get('n_read')}  tok {c.get('tok_cand')}/{c.get('tok_parent')}")
    if e.get("cond") and e["cond"] != "always":
        R = results(f"rt1|{e['cond']}|")
        jud = [r for r in R.values() if rows[r["row"]]["v4_role"] == "judge"]
        print(f"   router on JUDGE: {sum(r['yes'] for r in jud)}/{len(jud)} yes; p_src {sorted(set(r.get('p_src') for r in jud))}; content {sorted(set(r.get('content') for r in jud))}")
        ex = [next((r for r in sorted(jud, key=lambda r: r['row']) if r["yes"]), None), next((r for r in sorted(jud, key=lambda r: r['row']) if not r["yes"]), None)]
        for r in ex:
            if r is None:
                continue
            row = rows[r["row"]]
            print(f"\n   -- router example {r['row']} ({row['bench']}) -> content {r.get('content')!r} p {r.get('p'):.3f} (py {r.get('py'):.3g} pn {r.get('pn'):.3g})")
            for part in layout(row["prompt"], row.get("image_paths") or []) + [ROUTE_V1.format(c=e["condition"].strip().rstrip("."))]:
                print("      | " + part.replace("\n", "\n      | "))
    if c["author"] == "P0" and e:
        # one executor prompt with the entry applied (a routed JUDGE row, as read)
        reads = results("x|")
        q = None
        for f in sorted(glob.glob(f"{V4}/q/in/g{int(g):03d}_rung1_*.jsonl")):
            for l in open(f):
                x = json.loads(l)
                hit = (e["procedure"] in (x.get("prefix") or "")) if e.get("procedure") else all((x.get("settings") or {}).get(k) == v for k, v in (e.get("settings") or {}).items())
                if hit:
                    q = x
                    break
            if q:
                break
        if q:
            row = rows[q["row"]]
            print(f"\n   -- executor request {q['key']} row {q['row']} ({row['bench']}) settings {q.get('settings')}; result ok {reads.get(q['key'], {}).get('ok')}")
            print("      (thinking on, effort medium, read sampling, max_tokens 32768; user message parts in order:)")
            for part in layout(row["prompt"], row.get("image_paths") or [], q.get("prefix")):
                print("      | " + part.replace("\n", "\n      | "))
