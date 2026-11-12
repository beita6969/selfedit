"""Prefix-cache identity test at temperature 0: 20 learn rows (with images), medium effort, max_tokens 1024.
Passes: A = off-card, B = pc-card cold, C = pc-card warm (same requests again: full-prefix hits), D = off-card again (noise floor).
usage: python pc_test.py OFF_PORT PC_PORT"""
import json, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, "/workspace/fudan/mingdazhang/selfedit-v4/engine_v4")
from common import content_for, EFFORT

OFF, PC = int(sys.argv[1]), int(sys.argv[2])
rows = [json.loads(l) for l in open("/data/fd/mingdazhang/selfedit/v4/rows/learn.jsonl")]
rows = [r for r in rows if r.get("image_paths")][::13][:20]
bodies = []
for r in rows:
    c, nt = content_for(r["prompt"], r["image_paths"])
    bodies.append(dict(model="stock", messages=[{"role": "user", "content": c}], max_tokens=1024, temperature=0, seed=0,
                       chat_template_kwargs={"reasoning_effort": EFFORT, "enable_thinking": True}, logprobs=False))


def post(port, b):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(b).encode(), headers={"Content-Type": "application/json"})
    t = time.time()
    o = json.loads(urllib.request.urlopen(req, timeout=1800).read())
    m = o["choices"][0]["message"]
    return (m.get("reasoning_content") or m.get("reasoning") or "") + "\n##\n" + (m.get("content") or ""), time.time() - t, o["usage"]


def run(port):
    with ThreadPoolExecutor(20) as ex:
        return list(ex.map(lambda b: post(port, b), bodies))


P = {}
for name, port in (("A", OFF), ("B", PC), ("C", PC), ("D", OFF)):
    t = time.time()
    P[name] = run(port)
    print(name, port, f"{time.time() - t:.0f}s", "mean latency %.1fs" % (sum(x[1] for x in P[name]) / 20), flush=True)


def same(a, b):
    return sum(1 for x, y in zip(P[a], P[b]) if x[0] == y[0])


def prefix(a, b):
    out = []
    for x, y in zip(P[a], P[b]):
        s, t = x[0], y[0]
        k = next((i for i, (u, v) in enumerate(zip(s, t)) if u != v), min(len(s), len(t)))
        out.append(k)
    return sorted(out)[len(out) // 2]


for a, b, what in (("A", "D", "off vs off (noise floor)"), ("B", "C", "pc cold vs pc warm (cache hits)"), ("A", "B", "off vs pc cold"), ("A", "C", "off vs pc warm")):
    print(f"{what}: identical {same(a, b)}/20, median first-diff char {prefix(a, b)}")
