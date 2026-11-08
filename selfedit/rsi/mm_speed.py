"""Throughput of one card at given concurrency, on a fixed sample of real problems.
Usage: python -u rsi/mm_speed.py --model M --pool P --seqs 64 --batched 4096 [--problems 64] [--n 4] [--budget 8192]
Also checks that engine.step can be watched, so traces can reach the disk as they finish without chunking."""
import argparse, json, os, random, sys, time

def main():
    ap = argparse.ArgumentParser()
    for k, t, d in [("model", str, None), ("pool", str, None), ("seqs", int, 32), ("batched", int, 4096), ("problems", int, 64), ("n", int, 4), ("budget", int, 8192), ("util", float, .85), ("maxlen", int, 36864), ("seed", int, 101)]:
        ap.add_argument("--" + k, type=t, default=d)
    a = ap.parse_args()
    CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path[:0] = [CODE, os.path.join(CODE, "engine")]
    from surgery import Surgeon, image_message
    pool = [json.loads(l) for l in open(a.pool)]
    random.Random(0).shuffle(pool)
    probs = pool[:a.problems]
    S = Surgeon(a.model, tp=1, max_model_len=a.maxlen, max_num_seqs=a.seqs, max_num_batched_tokens=a.batched, gpu_memory_utilization=a.util, kv_cache_dtype="fp8", multimodal=True)
    prompts = [image_message(p["prompt"], p["image_paths"]) for p in probs]
    eng, orig = S.llm.llm_engine, S.llm.llm_engine.step
    seen, t0, last = [], time.time(), [time.time()]
    def step():
        out = orig()
        for o in out:
            if o.finished:
                seen.append((time.time() - t0, o.request_id, sum(len(c.token_ids) for c in o.outputs)))
        if time.time() - last[0] > 60:
            last[0] = time.time()
            print(f"t={time.time()-t0:.0f}s finished={len(seen)} tok={sum(s[2] for s in seen)}", flush=True)
        return out
    eng.step = step
    outs = S.generate(prompts, n=a.n, max_tokens=a.budget, reasoning_effort="medium", seeds=[a.seed + 1000 * i for i in range(len(prompts))])
    eng.step = orig
    g = S.last_generate
    toks = sorted(s["num_tokens"] for o in outs for s in o["samples"])
    pt = sorted(o["prompt_tokens"] for o in outs)
    print(f"SPEED seqs={a.seqs} batched={a.batched} seconds={g['seconds']:.0f} tokens={g['generated_tokens']} tok_per_s={g['tok_per_s']:.1f} median_gen={toks[len(toks)//2]} p90={toks[int(.9*len(toks))]} max={toks[-1]} prompt_median={pt[len(pt)//2]} prompt_max={pt[-1]}", flush=True)
    print("HOOK finished outputs", len(seen), "first ids", [s[1] for s in seen[:4]], "of", len(outs), flush=True)
    half = g["generated_tokens"] / 2
    c = 0
    for t, _, k in seen:
        c += k
        if c >= half:
            print(f"half of tokens by {t:.0f}s of {g['seconds']:.0f}s", flush=True)
            break

if __name__ == "__main__":
    main()
