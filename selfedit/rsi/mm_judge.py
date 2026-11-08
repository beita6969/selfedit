"""The judge's first read of one checkpoint: exam items and real problems, thinking off, one forward pass each, log-odds of the right option at "\\boxed{". Deterministic, about a minute. The model never sees this code or its result, except as a line in its ledger.

Usage: python -u rsi/mm_judge.py OUT.jsonl --exam DIR [--thin blink:3,cvbench:10] [loop.py args, with --mm and --pool]
"""
import json, os, re, sys


def pop(argv, flag, default=None):
    if flag not in argv:
        return default
    i = argv.index(flag)
    val = argv[i + 1]
    del argv[i:i + 2]
    return val


def main():                                                     # vLLM TP>1 workers re-import this file: nothing at top level
    out_path = sys.argv[1]
    argv = sys.argv[2:]
    exam = pop(argv, "--exam")
    thin = dict((kv.split(":")[0], int(kv.split(":")[1])) for kv in pop(argv, "--thin", "blink:3,cvbench:10").split(","))
    sys.argv = ["loop.py", "--run-dir", os.path.dirname(os.path.abspath(out_path))] + argv
    CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path[:0] = [CODE, os.path.join(CODE, "engine")]
    import numpy as np
    import rsi.loop as L
    from rsi.eye_exam import ENDS
    from surgery import Surgeon, image_message
    from vllm import SamplingParams

    a = L.args
    S = Surgeon(a.model, tp=a.tp, max_model_len=a.max_model_len, max_num_seqs=a.max_num_seqs, max_num_batched_tokens=a.max_num_batched_tokens,
                gpu_memory_utilization=a.gpu_util, kv_cache_dtype=a.kv_cache_dtype, multimodal=True)
    items = [dict(p, set="exam", group=p["task"]) for p in L.load_jsonl([os.path.join(exam, "items.jsonl")])]
    seen = {}
    for p in L.load_jsonl(a.pool):
        if not p.get("image_paths") or p["kind"] != "mc":
            continue
        seen[p["bench"]] = seen.get(p["bench"], 0) + 1
        if seen[p["bench"]] % thin.get(p["bench"], 1) == 0:
            items.append(dict(p, set="pool", group=p["bench"]))

    def tok(t):
        return S.tokenizer(t, add_special_tokens=False).input_ids

    heads = S.generate([image_message(p["prompt"], p["image_paths"]) for p in items], n=1, sampling=dict(L.SAMPLING), max_tokens=1, enable_thinking=False)
    prompts, cands, keep = [], [], []
    for p, o in zip(items, heads):
        opts = list(ENDS[: int(p["level"])]) if p.get("task") == "trace" else sorted(set(re.findall(r"\(([A-Z])\)", p["prompt"]) or re.findall(r"(?:^|\s)([A-Z])\.\s", p["prompt"])) | {p["answer"]})
        ids = [tok("\\boxed{" + x) for x in opts]
        n = next(i for i in range(len(ids[0]) + 1) if i == len(ids[0]) or any(i >= len(y) or y[i] != ids[0][i] for y in ids))
        if len(opts) < 2 or any(len(y) <= n for y in ids):
            continue
        prompts.append(S._token_prompt(list(o["prompt_token_ids"]) + ids[0][:n], p["image_paths"]))
        cands.append(([y[n] for y in ids], opts.index(p["answer"])))
        keep.append(p)
    params = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20, detokenize=False)
    tmp = out_path + ".tmp"
    with open(tmp, "w") as f:
        for p, (c, g), o in zip(keep, cands, S.llm.generate(prompts, [params] * len(prompts), use_tqdm=False)):
            top = o.outputs[0].logprobs[0]
            floor = min(v.logprob for v in top.values()) - 1.0
            lp = np.array([top[t].logprob if t in top else floor for t in c], dtype=np.float64)
            rest = np.delete(lp, g)
            f.write(json.dumps({"id": p["id"], "set": p["set"], "group": p["group"], "margin": float(lp[g] - np.logaddexp.reduce(rest)), "right": bool(lp[g] > rest.max())}) + "\n")
    os.replace(tmp, out_path)
    L.say(f"mm_judge: {len(keep)} items read")


if __name__ == "__main__":
    main()
