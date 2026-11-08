"""The improver reads its own record: the notes it wrote, by kind of problem, and what each did to its score. It writes rules for writing notes.
Usage: python -u rsi/meta_propose.py RECORD OUT.jsonl --proposals 24 --model M --mm [loop.py args]. RECORD: jsonl with cat, base, notes [{note, change}]."""
import json, os, random, re, sys

ASK = """You are an AI model. Earlier you read your own failures on several kinds of problems and wrote short notes to read before such problems. Each note was tried: below is how many percentage points it changed your score on problems of that kind.

{items}

Study this record. What separates the notes that raised your score from those that did nothing or lowered it? Write one rule of at most 40 words for your future self to follow when writing such a note after reading failures. It must apply to any kind of problem and must not mention any particular kind of problem or subject. End with exactly one line: {{"note": "..."}}"""


def main():
    rec, out = sys.argv[1], sys.argv[2]
    argv = sys.argv[3:]
    i = argv.index("--proposals")
    n = int(argv[i + 1])
    del argv[i:i + 2]
    sys.argv = ["loop.py", "--run-dir", os.path.dirname(out)] + argv
    CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path[:0] = [CODE, os.path.join(CODE, "engine")]
    import rsi.loop as L
    from surgery import Surgeon, image_message
    a = L.args
    S = Surgeon(a.model, tp=a.tp, max_model_len=a.max_model_len, max_num_seqs=a.max_num_seqs, max_num_batched_tokens=a.max_num_batched_tokens, gpu_memory_utilization=a.gpu_util, kv_cache_dtype=a.kv_cache_dtype, multimodal=True)
    R = [json.loads(l) for l in open(rec)]
    rng = random.Random(0)
    msgs = []
    for _ in range(n):                                  # every proposal sees its own draw of the record, so they differ
        items = []
        for r in rng.sample(R, min(8, len(R))):
            ns = sorted(r["notes"], key=lambda x: -x["change"])
            ns = ns[:2] + ns[-2:] if len(ns) > 4 else ns
            items.append(f"Kind of problem {len(items) + 1}, score without a note {100 * r['base']:.0f}%:\n" + "\n".join(f"  {x['change']:+.0f} points: {x['note']}" for x in ns))
        msgs.append(image_message(ASK.format(items="\n\n".join(items)), []))
    outs = S.generate(msgs, n=1, sampling=dict(L.SAMPLING), max_tokens=6000, reasoning_effort=a.effort)
    with open(out, "w") as f:
        for o in outs:
            m = re.findall(r'\{\s*"note"\s*:\s*"((?:[^"\\]|\\.)*)"\s*\}', o["samples"][0]["content"] or "")
            if m:
                f.write(json.dumps(dict(rule=m[-1].replace('\\"', '"')), ensure_ascii=False) + "\n")
    print("done", flush=True)


if __name__ == "__main__":
    main()
