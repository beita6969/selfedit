"""Prefix-cache identity at batch size 1 on an otherwise idle (drained) pc server: 20 rows, temperature 0, max_tokens 512.
Pass 1 = cold (no cached prefix), pass 2 = same requests again (full-prefix cache hits), pass 3 = again (hits, repeatability).
usage: python pc_serial.py PORT"""
import json, sys, time, urllib.request
sys.path.insert(0, "/workspace/fudan/mingdazhang/selfedit-v4/engine_v4")
from common import content_for, EFFORT

PORT = int(sys.argv[1])
rows = [json.loads(l) for l in open("/data/fd/mingdazhang/selfedit/v4/rows/learn.jsonl")]
rows = [r for r in rows if r.get("image_paths")][5::13][:20]          # rows not used by pc_test (not in this server's cache)


def post(r):
    c, nt = content_for(r["prompt"], r["image_paths"])
    b = dict(model="stock", messages=[{"role": "user", "content": c}], max_tokens=512, temperature=0, seed=0,
             chat_template_kwargs={"reasoning_effort": EFFORT, "enable_thinking": True}, return_token_ids=True)
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=json.dumps(b).encode(), headers={"Content-Type": "application/json"})
    o = json.loads(urllib.request.urlopen(req, timeout=1800).read())
    return o["choices"][0].get("token_ids") or ((o["choices"][0]["message"].get("reasoning_content") or o["choices"][0]["message"].get("reasoning") or "") + "##" + (o["choices"][0]["message"].get("content") or ""))


def hits():
    m = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/metrics").read().decode()
    g = lambda k: sum(float(l.split()[-1]) for l in m.splitlines() if l.startswith(k))
    return g("vllm:prefix_cache_hits_total"), g("vllm:prefix_cache_queries_total")


P = []
for p in range(3):
    h0 = hits(); t = time.time()
    P.append([post(r) for r in rows])
    h1 = hits()
    print(f"pass {p + 1}: {time.time() - t:.0f}s, prefix hit rate {(h1[0] - h0[0]) / max(1, h1[1] - h0[1]):.2f}", flush=True)


def cmp(a, b):
    same = sum(1 for x, y in zip(P[a], P[b]) if x == y)
    fd = sorted(next((i for i, (u, v) in enumerate(zip(x, y)) if u != v), min(len(x), len(y))) for x, y in zip(P[a], P[b]))
    return f"identical {same}/20, median first-diff position {fd[10]}"


print("cold vs warm (pass1 vs pass2):", cmp(0, 1))
print("warm vs warm (pass2 vs pass3):", cmp(1, 2))
print("cold vs warm (pass1 vs pass3):", cmp(0, 2))
