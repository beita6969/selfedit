"""Where Qwen's stop decision sits: at each kind of point in its own traces, the log-probs of the two closing tokens separately ("\\n", then </think>), and the top next tokens.
Usage: python -u rsi/t_split.py RUN_DIR --base BASE.jsonl [--traces 40] [loop.py args]"""
import json, os, sys, time
from collections import Counter


def pop(flag, default=None):
    if flag not in sys.argv:
        return default
    i = sys.argv.index(flag)
    v = sys.argv[i + 1]
    del sys.argv[i: i + 2]
    return v


def main():
    RUN = sys.argv[1]
    BASE, K = pop("--base"), int(pop("--traces", 40))
    os.makedirs(RUN, exist_ok=True)
    sys.argv = ["loop.py", "--run-dir", RUN] + sys.argv[2:]
    CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path[:0] = [CODE, os.path.join(CODE, "engine")]
    import numpy as np
    import rsi.loop as L
    from rsi.hazard import answer_marks, family_tokens, pick
    from surgery import Surgeon
    from vllm import SamplingParams

    a = L.args
    S = Surgeon(a.model, tp=a.tp, max_model_len=a.max_model_len, max_num_seqs=a.max_num_seqs,
                max_num_batched_tokens=a.max_num_batched_tokens, gpu_memory_utilization=a.gpu_util, kv_cache_dtype=a.kv_cache_dtype)
    L._S = S
    tk = S.tokenizer
    lead, closing, _ = family_tokens(S)
    pool = {p["id"]: p for p in L.load_jsonl(a.pool)}
    jobs = {"cut": [], "ans": [], "end": []}
    for r in pick([json.loads(l) for l in open(BASE)], K, np.random.default_rng(a.seed)):
        head = tk(S.render(pool[r["id"]]["prompt"], reasoning_effort=a.effort), add_special_tokens=False).input_ids + lead
        th = tk(r["reasoning"], add_special_tokens=False).input_ids
        first, lines = answer_marks(tk, th, r["answer"])
        bounds = [k for k, t in enumerate(th) if "\n" in tk.decode([t])]
        if bounds:
            jobs["cut"].append(head + th[:bounds[int(0.8 * len(bounds))]])
        for k in [x for x in lines if x < len(th) - 2][:3]:
            jobs["ans"].append(head + th[:k])
        if r["finish"] != "length":
            jobs["end"].append(head + th)
    params = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20, detokenize=False)
    out = {}
    for kind, seqs in jobs.items():
        outs = S.llm.generate([{"prompt_token_ids": q} for q in seqs], [params] * len(seqs), use_tqdm=False)
        next1, top = [], Counter()
        for o in outs:
            lp = o.outputs[0].logprobs[0]
            next1.append({t: v.logprob for t, v in lp.items()})
            for t, v in lp.items():
                top[t] += float(np.exp(v.logprob))
        sc = S.score([q + closing for q in seqs], [len(q) for q in seqs])
        first_tok = [float(x["logprobs"][0]) for x in sc]
        second = [float(x["logprobs"][-1]) for x in sc]
        names = [(tk.decode([t]), round(p / len(seqs), 3)) for t, p in top.most_common(14)]
        out[kind] = dict(n=len(seqs), first_close_token=float(np.mean(first_tok)), then_stop=float(np.mean(second)),
                         then_stop_quantiles=[float(x) for x in np.percentile(second, [10, 50, 90])], next_tokens=names)
        L.say(f"t_split {kind}: n {len(seqs)}, log p(first closing token) {out[kind]['first_close_token']:.2f}, log p(stop | it) {out[kind]['then_stop']:.2f} {out[kind]['then_stop_quantiles']}; next {names}")
    json.dump(out, open(os.path.join(RUN, "split.json"), "w"), indent=1)
    L.say("t_hazard: done")


if __name__ == "__main__":
    main()
