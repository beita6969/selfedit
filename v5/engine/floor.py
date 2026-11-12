"""Engine noise floor on STOCK weights: Surgeon (offline vllm.LLM, as in W workers / export_tip) vs a stock `vllm serve` server,
the t004 export's 10 rows one request at a time, temperature 0, max_tokens 1024. usage: python floor.py PORT"""
import json, sys, time, urllib.request
sys.path.insert(0, "/workspace/fudan/mingdazhang/selfedit-v4/engine_v4")
from common import MODEL, EFFORT, content_for
import wedit  # adds the code-1109c engine tree to sys.path
from surgery import Surgeon
V4 = "/data/fd/mingdazhang/selfedit/v4"
port = int(sys.argv[1])
ex = json.load(open(f"{V4}/ckpt/t004.export.json"))
rows = {r["id"]: r for r in map(json.loads, open(f"{V4}/rows/learn.jsonl"))}
S = Surgeon(MODEL, tp=1, max_model_len=46000, max_num_seqs=16, max_num_batched_tokens=8192, gpu_memory_utilization=0.85,
            kv_cache_dtype="fp8", multimodal=True, enable_prefix_caching=False)
same = 0
for ref in ex["reference"]:
    r = rows[ref["row"]]
    c, nt = content_for(r["prompt"], r["image_paths"])
    a = S.generate([[{"role": "user", "content": c}]], n=1, sampling=dict(temperature=0.0), max_tokens=1024, reasoning_effort=EFFORT, seeds=[0])[0]["samples"][0]["token_ids"]
    b = dict(model="stock", messages=[{"role": "user", "content": c}], max_tokens=1024, temperature=0, seed=0,
             chat_template_kwargs={"reasoning_effort": EFFORT, "enable_thinking": True}, return_token_ids=True)
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(b).encode(), headers={"Content-Type": "application/json"})
    s = json.loads(urllib.request.urlopen(req, timeout=1800).read())["choices"][0]["token_ids"]
    k = next((i for i, (x, y) in enumerate(zip(a, s)) if x != y), None if a == s else min(len(a), len(s)))
    same += a == s
    print(json.dumps(dict(row=ref["row"], identical=a == s, first_diff=k)), flush=True)
print(f"stock floor: identical {same}/{len(ex['reference'])}", flush=True)
