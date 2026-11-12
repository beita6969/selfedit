"""Export tip t007c cached reads of SIGNAL rows (show+monitor) where entry g005c23 or g005c29 is routed: request as sent (messages rebuilt with
common.content_for's layout; image payloads replaced by path + sha256 + image tokens), sampling, reasoning, answer, grade, tokens, fired entries."""
import glob, gzip, hashlib, json, sys
sys.path[:0] = ["/workspace/fudan/mingdazhang/selfedit-v4/engine_v4", "/workspace/fudan/mingdazhang/selfedit-v4/loop"]
import common, ecomp
V4 = "/data/fd/mingdazhang/selfedit/v4"; OUT = f"{V4}/scratch/t007c_signal_g005c23_c29_reads.jsonl.gz"
MAXS = int(sys.argv[1]) if len(sys.argv) > 1 else 8
tip = json.load(open(f"{V4}/runs/main_1110/state.json"))["tip"]
cid = {i: e.get("cid") for i, e in enumerate(tip["entries"])}
want = [i for i, c in cid.items() if c in ("g005c23", "g005c29")]; assert len(want) == 2, cid
rows = {json.loads(l)["id"]: json.loads(l) for l in open(f"{V4}/rows/learn.jsonl")}
fired = json.load(open(f"{V4}/scratch/ext/tip_fired.json"))
fired = fired.get("rows", fired) if isinstance(fired, dict) and "rows" in fired else fired
F = {r: v for r, v in fired.items() if r in rows}
reqs = [json.loads(l) for l in open(f"{V4}/scratch/ext/tip_signal8.jsonl")]
def fired_of(r):
    v = F.get(r) or []
    return [x if isinstance(x, int) else x.get("i") for x in v] if isinstance(v, list) else []
sel = [q for q in reqs if rows[q["row"]].get("v4_role") in ("show", "monitor") and set(fired_of(q["row"])) & set(want) and q["sample"] < MAXS]
need = {q["key"] for q in sel}; res = {}
for f in glob.glob(f"{V4}/q/out/*.jsonl"):
    for l in open(f, "rb"):
        if l.startswith(b'{"key": "x|') or l.startswith(b'{"key": "p000|'):
            k = l[9:l.index(b'"', 9)].decode()
            if k in need:
                x = json.loads(l)
                if not x.get("error") and x.get("content") is not None: res[k] = x
n = 0
with gzip.open(OUT, "wt") as fo:
    for q in sel:
        x = res.get(q["key"])
        if x is None: continue
        row = rows[q["row"]]; st = q.get("settings") or {}; imgs = []
        for p in row.get("image_paths") or []:
            _, nt = common.image_url_for(p, st.get("min_image_tokens"), st.get("max_image_tokens"))
            imgs.append({"type": "image_url", "image_url": {"path": p, "sha256": hashlib.sha256(open(p, "rb").read()).hexdigest(), "image_tokens": nt,
                         "min_image_tokens": st.get("min_image_tokens"), "max_image_tokens": st.get("max_image_tokens")}})
        prefix, prompt = q.get("prefix"), row["prompt"]; parts = prompt.split("<image>")
        if len(parts) - 1 == len(imgs) and imgs:
            content = ([{"type": "text", "text": prefix.rstrip() + "\n\n"}] if prefix else [])
            for k, t in enumerate(parts):
                if t.strip(): content.append({"type": "text", "text": t.strip()})
                if k < len(imgs): content.append(imgs[k])
        else:
            content = imgs + ([{"type": "text", "text": prefix.rstrip() + "\n\n"}] if prefix else []) + [{"type": "text", "text": prompt}]
        fo.write(json.dumps(dict(key=q["key"], row=q["row"], sample=q["sample"], bench=row["bench"], v4_role=row["v4_role"], answer_gold=row["answer"],
            options=row.get("options"), kind=row["kind"], request=q, messages=[{"role": "user", "content": content}],
            sampling=dict(common.SAMPLING, reasoning_effort=common.EFFORT, max_tokens=q.get("max_tokens") or 32768, seed=x.get("seed")),
            fired=[dict(i=i, cid=cid[i]) for i in fired_of(q["row"])], reasoning=x.get("reasoning"), content=x.get("content"),
            pred=ecomp.pred(x, row), ok=bool(x.get("ok")), tokens=x.get("tokens"), prompt_tokens=x.get("prompt_tokens"), finish=x.get("finish"),
            server=x.get("server"))) + "\n"); n += 1
print(f"entries {want} -> cids {[cid[i] for i in want]}; rows {len({q['row'] for q in sel})}; reads written {n} of {len(sel)} requested")
print(OUT, hashlib.sha256(open(OUT, "rb").read()).hexdigest(), round(__import__("os").path.getsize(OUT) / 1e6, 2), "MB")
