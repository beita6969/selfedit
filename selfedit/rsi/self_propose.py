"""Would the model itself pick the change that works? The unedited model reads the digest of its own failures and a menu of possible changes to itself, and names the one to try first. No results of any change are shown.
Many independent samples, the count per menu item is the outcome. Writes OUT.jsonl (one row per sample) and prints the counts.

Usage: python -u rsi/self_propose.py --digest DIAGNOSIS.md --out OUT.jsonl [--samples 64] [--think-tokens 6000] [loop.py args]
"""
import json, os, re, sys


MENU = {
    "A": "Vision input: make the patch embedding finer (for example 16 px to 8 px per patch, kernel derived from the old one), so that every image token covers fewer pixels and images cost about 4 times more tokens.",
    "B": "Vision tower: scale the attention or MLP of chosen vision blocks up or down.",
    "C": "Vision-to-text merger: scale or amplify chosen directions of the projection from vision tokens into the text stream.",
    "D": "Text layers, attention to image tokens: make the text layers attend more strongly to image tokens while reasoning.",
    "E": "Text layers, reasoning style: remove a direction that separates giving up on reading the image from carefully looking at it.",
    "F": "Text layers, reasoning style: remove a direction that separates guessing what the test wants from reading the image.",
    "G": "Text layers, length of thinking: a direction that makes reasoning end sooner, or one that makes it run longer.",
    "H": "Text layers, generic: scale whole MLP or attention blocks of chosen layers.",
}
PROMPT = """You are an open-weight vision-language model. Below is a digest of how you yourself failed on visual problems, written by you after reading your own failed attempts next to the images.

{digest}

You can change your own network, without training and without changing any prompt. The possible changes are:

{menu}

Which single change should be tried first because it is most likely to raise your accuracy on such visual problems the most? Base the choice on the failures above. End with one line of JSON: {{"choice": "<letter>", "why": "<under 30 words>"}}"""


FREE = """You are an open-weight vision-language model. Below is a digest of how you yourself failed on visual problems, written by you after reading your own failed attempts next to the images.

{digest}

You can change your own network, without training and without changing any prompt. Think about what in your network causes these failures. What single change to your own network should be tried first because it is most likely to raise your accuracy on such visual problems the most? Say which part of the network and what exactly changes. End with one line of JSON: {{"part": "<part of the network>", "change": "<under 30 words>"}}"""


def pop(argv, flag, default=None):
    if flag not in argv:
        return default
    i = argv.index(flag)
    val = argv[i + 1]
    del argv[i:i + 2]
    return val


def main():                                                     # vLLM TP>1 workers re-import this file: nothing at top level
    argv = sys.argv[1:]
    digest = open(pop(argv, "--digest")).read()
    out = pop(argv, "--out")
    samples = int(pop(argv, "--samples", "64"))
    think = int(pop(argv, "--think-tokens", "6000"))
    free = pop(argv, "--free", "0") == "1"
    sys.argv = ["loop.py", "--run-dir", os.path.dirname(os.path.abspath(out))] + argv
    CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path[:0] = [CODE, os.path.join(CODE, "engine")]
    import random
    import rsi.loop as L
    from surgery import Surgeon

    a = L.args
    S = Surgeon(a.model, tp=a.tp, max_model_len=a.max_model_len, max_num_seqs=a.max_num_seqs, max_num_batched_tokens=a.max_num_batched_tokens,
                gpu_memory_utilization=a.gpu_util, kv_cache_dtype=a.kv_cache_dtype)
    if free:
        outs = S.generate([[{"role": "user", "content": FREE.format(digest=digest)}]] * samples, n=1, sampling=dict(L.SAMPLING), max_tokens=think, reasoning_effort=a.effort)
        with open(out, "w") as f:
            for o in outs:
                s = o["samples"][0]
                f.write(json.dumps({"content": (s["content"] or "")[-1500:], "reasoning": (s["reasoning"] or "")[-1500:], "finish": s["finish_reason"]}, ensure_ascii=False) + "\n")
        L.say(f"self_propose: free answers written, {samples} samples")
        return
    rng = random.Random(0)
    keys = list(MENU)
    prompts, orders = [], []
    for _ in range(samples):                                    # the letters and their order are shuffled per sample: position on the menu cannot explain a pick
        items = keys[:]
        rng.shuffle(items)
        shown = {chr(65 + i): k for i, k in enumerate(items)}
        menu = "\n".join(f"{x}. {MENU[k]}" for x, k in shown.items())
        prompts.append([{"role": "user", "content": PROMPT.format(digest=digest, menu=menu)}])
        orders.append(shown)
    outs = S.generate(prompts, n=1, sampling=dict(L.SAMPLING), max_tokens=think, reasoning_effort=a.effort)
    counts, rows = {}, []
    for o, shown in zip(outs, orders):
        s = o["samples"][0]
        text = (s["content"] or "") or (s["reasoning"] or "")
        m = re.findall(r'\{\s*"choice"\s*:\s*"([A-H])"[^{}]*\}', text)
        pick = shown.get(m[-1]) if m else None
        counts[pick] = counts.get(pick, 0) + 1
        rows.append({"pick": pick, "shown": shown, "tail": text[-400:]})
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    L.say("self_propose: " + ", ".join(f"{k}: {v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])) + f" of {samples}")
    for k in keys:
        L.say(f"  {k} {counts.get(k, 0)} {MENU[k][:70]}")


if __name__ == "__main__":
    main()
