"""Rollout / eval server on its own card: the main system's Surgeon (same engine settings, message format, thinking split), driven by files in CTL.
Commands: cmd_<id>.json with {"op": "generate"|"reload"|"exit", ...}; answer in out_<id>.pkl. One server per card."""
import argparse, glob, json, os, pickle, time
from bl_common import *
from surgery import Surgeon
ap = argparse.ArgumentParser()
ap.add_argument("--ctl", required=True); ap.add_argument("--util", type=float, default=.85); ap.add_argument("--seed", type=int, default=101)
ap.add_argument("--model", default=STOCK); ap.add_argument("--num-seqs", type=int, default=64); ap.add_argument("--sleep-mode", type=int, default=0)
ap.add_argument("--prefix-cache", type=int, default=0); ap.add_argument("--max-batched", type=int, default=0)   # throughput only: shared prompt KV across the G samples; prefill chunk size
a = ap.parse_args()
os.makedirs(a.ctl, exist_ok=True)
E = dict(ENGINE, max_num_seqs=a.num_seqs, enable_prefix_caching=bool(a.prefix_cache))
if a.max_batched:
    E["max_num_batched_tokens"] = a.max_batched
if a.sleep_mode:
    E["enable_sleep_mode"] = True
S = Surgeon(a.model, gpu_memory_utilization=a.util, seed=a.seed, **E)
say("engine loaded", round(S.load_seconds, 1), "s", S.gpu_memory())
open(f"{a.ctl}/ready", "w").write(str(os.getpid()))
while True:
    cmds = sorted(glob.glob(f"{a.ctl}/cmd_*.json"))
    if not cmds:
        time.sleep(.2)
        continue
    f = cmds[0]
    c = json.load(open(f))
    cid = os.path.basename(f)[4:-5]
    t0 = time.time()
    res = dict(ok=True)
    try:
        if c["op"] == "exit":
            os.remove(f)
            break
        if c["op"] == "sleep":
            S.llm.sleep(level=c.get("level", 2))
        if c["op"] == "wake":
            S.llm.wake_up()
            if S.prefix_caching:
                S.llm.reset_prefix_cache()
        if c["op"] == "reload":
            S.llm.collective_rpc("reload_weights", kwargs=dict(weights_path=c["path"]))
            S.reset_encoder_cache()
            if S.prefix_caching:
                S.llm.reset_prefix_cache()      # cached prompt KV / SSM states were computed with the old weights
        if c["op"] == "generate":
            res["out"] = S.generate([dict(prompt=p, image_paths=i) for p, i in zip(c["prompts"], c["image_paths"])], n=c["n"], sampling=c["sampling"], max_tokens=c.get("max_tokens", BUDGET),
                                    reasoning_effort=c.get("effort", EFFORT), logprobs=c.get("logprobs"), seeds=c.get("seeds"))
            res["used"] = dict(max_tokens=c.get("max_tokens", BUDGET), effort=c.get("effort", EFFORT))
            res["engine"] = dict(max_num_seqs=E["max_num_seqs"], max_num_batched_tokens=E["max_num_batched_tokens"], enable_prefix_caching=E["enable_prefix_caching"],
                                 kv_cache_dtype=E["kv_cache_dtype"], max_model_len=E["max_model_len"], tp=E["tp"], util=a.util)
            res["stat"] = S.last_generate
            for o in res["out"]:
                for s in o["samples"]:
                    s.pop("top_logprobs", None)
        res["mem"] = S.gpu_memory() if c["op"] != "sleep" else None
    except Exception as e:
        import traceback
        res = dict(ok=False, error=traceback.format_exc()[-3000:])
    res["seconds"] = time.time() - t0
    say(c["op"], "seconds", round(res["seconds"], 1), res.get("stat"), res.get("error"))
    pickle.dump(res, open(f"{a.ctl}/.tmp_{cid}", "wb"))
    os.rename(f"{a.ctl}/.tmp_{cid}", f"{a.ctl}/out_{cid}.pkl")
    os.remove(f)
say("server exit")
