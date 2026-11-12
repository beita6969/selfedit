"""Verify a T server against export_tip.py's reference (Surgeon with the edit applied): the same rows one request at a time,
temperature 0, max_tokens 1024, seed 0; compare prompt token count and generated token ids.
usage: python verify_tip.py TIP PORT"""
import json, sys, urllib.request
sys.path.insert(0, "/workspace/fudan/mingdazhang/selfedit-v4/engine_v4")
from common import content_for, EFFORT

V4 = "/data/fd/mingdazhang/selfedit/v4"
tip, port = sys.argv[1], int(sys.argv[2])
ex = json.load(open(f"{V4}/ckpt/{tip}.export.json"))
rows = {r["id"]: r for r in map(json.loads, open(f"{V4}/rows/learn.jsonl"))}
same = 0
for ref in ex["reference"]:
    r = rows[ref["row"]]
    c, nt = content_for(r["prompt"], r["image_paths"])
    b = dict(model=tip, messages=[{"role": "user", "content": c}], max_tokens=1024, temperature=0, seed=0,
             chat_template_kwargs={"reasoning_effort": EFFORT, "enable_thinking": True}, return_token_ids=True)
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(b).encode(), headers={"Content-Type": "application/json"})
    o = json.loads(urllib.request.urlopen(req, timeout=1800).read())
    ids = o["choices"][0].get("token_ids")
    pt = o["usage"]["prompt_tokens"]
    k = next((i for i, (a, bb) in enumerate(zip(ids, ref["token_ids"])) if a != bb), min(len(ids), len(ref["token_ids"])))
    ok = ids == ref["token_ids"] and pt == ref["prompt_tokens"]
    same += ok
    print(json.dumps(dict(row=ref["row"], prompt_tokens=[pt, ref["prompt_tokens"]], n=[len(ids), len(ref["token_ids"])], identical=ok, first_diff=None if ok else k)), flush=True)
print(f"identical {same}/{len(ex['reference'])}")
