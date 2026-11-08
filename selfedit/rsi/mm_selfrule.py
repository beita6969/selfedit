"""Can the model itself name what it gets wrong? It reads a few of its own failures on one kind of problem (images, the end of its reasoning, the right answer) and writes one note. Blind control: the same problems, no reasoning and no answers. Notes from half 0 of the problems are tried on half 0, the best of each arm is confirmed on half 1, which it never read. Notes are run-time probes only; they say whether the cause is within the model's reach, the edit comes after.

Usage: python -u rsi/mm_selfrule.py OUT --traces T.jsonl --cats A,B,C --pool POOL [--proposals 24] [--tries 6] [--confirm 8] [loop.py args, --samples is for the tries]
Writes OUT/<cat>.notes.jsonl, OUT/<cat>.tries.jsonl, OUT/report.txt.
"""
import json, os, random, re, sys, zlib

READ = """You are the AI model that answered the problems below. The images above belong to them, in order. For each one you see the question, the end of your own reasoning, what you answered and the right answer.

{items}

Study where your reasoning went wrong. Find the one general mistake you keep making on this kind of problem, a wrong rule or a wrong habit, not a slip on one case. Then write a note of at most 60 words that you could read before such a problem so that you would get it right. It must state general knowledge and must not mention any particular answer or problem. End with exactly one line: {{"note": "..."}}"""
BLIND = """You are an AI model. The images above belong to the problems below, in order.

{items}

Write a note of at most 60 words that you could read before such a problem so that you would answer it well. It must state general knowledge and must not mention any particular answer or problem. End with exactly one line: {{"note": "..."}}"""


def pop(argv, flag, default=None):
    if flag not in argv:
        return default
    i = argv.index(flag)
    val = argv[i + 1]
    del argv[i:i + 2]
    return val


def main():                                                     # vLLM TP>1 workers re-import this file: nothing at top level
    OUT = sys.argv[1]
    argv = sys.argv[2:]
    traces, cats = pop(argv, "--traces"), pop(argv, "--cats").replace("~", " ").split(",")
    proposals, tries, confirm = int(pop(argv, "--proposals", "24")), int(pop(argv, "--tries", "6")), int(pop(argv, "--confirm", "8"))
    engine = pop(argv, "--engine")      # this generation's edits, kept on for the whole run
    meta_file = pop(argv, "--meta")     # a rule for writing notes: the writer follows it, arm "meta"; "-" = the read arm alone, same seeds
    os.makedirs(OUT, exist_ok=True)
    sys.argv = ["loop.py", "--run-dir", OUT] + argv
    CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path[:0] = [CODE, os.path.join(CODE, "engine")]
    import rsi.loop as L
    from surgery import Surgeon, image_message

    a = L.args
    S = Surgeon(a.model, tp=a.tp, max_model_len=a.max_model_len, max_num_seqs=a.max_num_seqs, max_num_batched_tokens=a.max_num_batched_tokens, gpu_memory_utilization=a.gpu_util, kv_cache_dtype=a.kv_cache_dtype, multimodal=True)
    if engine:
        S.apply(json.load(open(engine))[0]["engine"])
    pool = [json.loads(l) for f in (a.pool if isinstance(a.pool, list) else [a.pool]) for l in open(f)]
    own = {}
    for l in open(traces):
        t = json.loads(l)
        if "id" in t:
            own.setdefault(t["id"], []).append(t)
    half = lambda p: zlib.crc32(p["id"].encode()) % 2
    rng = random.Random(0)

    def say(s):
        L.say(s)
        open(os.path.join(OUT, "report.txt"), "a").write(s + "\n")

    def question(p):
        return re.sub(r"<image>", "", p["prompt"].rsplit("\n\n", 1)[0])

    ARMS = (("read", True), ("blind", False))
    PROMPT = READ
    if meta_file == "-":
        ARMS = (("read", True),)
    elif meta_file:
        ARMS = (("meta", True),)
        rule = open(meta_file).read().strip().replace("{", "{{").replace("}", "}}")
        PROMPT = READ.replace("It must state general knowledge", "When you write the note, follow this: " + rule + " The note must state general knowledge", 1)
        assert PROMPT != READ

    def shown(ps, read):
        items, paths = [], []
        for n, p in enumerate(ps):
            paths += p["image_paths"]
            s = f"Problem {n + 1}: {question(p)}"
            if read:
                t = next((x for x in own[p["id"]] if not x["ok"]), own[p["id"]][0])
                r = (t["reasoning"] or "")[-700:]
                said = (t["content"] or "")[-120:]
                s += f"\nYour reasoning, end: ...{r}\nYou answered: {said}\nRight answer: {p['answer']}" + (f" ({p['options'][ord(str(p['answer'])) - 65]})" if p.get("options") and str(p["answer"]) in "ABCDEFGH" and ord(str(p["answer"])) - 65 < len(p["options"]) else "")
            items.append(s)
        return image_message((PROMPT if read else BLIND).format(items="\n\n".join(items)), paths)

    def with_note(p, note):
        i = p["prompt"].rfind("\n\n")
        return dict(p, prompt=p["prompt"][:i] + "\n\nNote: " + note + p["prompt"][i:])

    def rate(ps, n, note=None):
        L.mm_eval.PARTIAL = os.path.join(OUT, "partial.jsonl")
        R = L.evaluate(S, [with_note(p, note) if note else p for p in ps], n)
        by = {}
        for r in R:
            by.setdefault(r["id"], []).append(bool(r["ok"]) and r["finish"] != "length")
        return {k: sum(v) / len(v) for k, v in by.items()}, R

    def mean(d):
        return sum(d.values()) / max(1, len(d))

    def dump(name, R):
        with open(os.path.join(OUT, name), "w") as f:
            for r in R:
                f.write(json.dumps({k: v for k, v in r.items() if k != "seq"}, ensure_ascii=False) + "\n")

    for cat in cats:
        ps = [p for p in pool if p["category"] == cat and p.get("image_paths") and p["id"] in own]
        fit, test = [p for p in ps if half(p) == 0], [p for p in ps if half(p) == 1]
        wrong = [p for p in fit if any(not x["ok"] for x in own[p["id"]])]
        tag = re.sub(r"\W+", "_", cat)
        if len(wrong) < 4 or len(test) < 5:
            say(f"{cat}: too few, {len(wrong)} failed of half 0, {len(test)} in half 1")
            continue
        notes = {}
        for arm, read in ARMS:
            msgs = [shown(rng.sample(wrong, 4), read) for _ in range(proposals)]
            outs = S.generate(msgs, n=1, sampling=dict(L.SAMPLING), max_tokens=5000, reasoning_effort=a.effort)
            got = []
            for o in outs:
                m = re.findall(r'\{\s*"note"\s*:\s*"((?:[^"\\]|\\.)*)"\s*\}', o["samples"][0]["content"] or "")
                if m:
                    got.append(m[-1].replace('\\"', '"'))
            notes[arm] = got
            with open(os.path.join(OUT, f"{tag}.notes.jsonl"), "a") as f:
                for g in got:
                    f.write(json.dumps(dict(arm=arm, note=g), ensure_ascii=False) + "\n")
        base0, R0 = rate(fit, a.samples)
        base1, R1 = rate(test, confirm)
        dump(f"{tag}.base0.jsonl", R0)
        dump(f"{tag}.base1.jsonl", R1)
        say(f"{cat}: {len(fit)} + {len(test)} problems, base half 0 {mean(base0):.3f}, half 1 {mean(base1):.3f}; notes {' '.join(f'{k} {len(v)}' for k, v in notes.items())}")
        for arm, _ in ARMS:
            tried = []
            for note in rng.sample(notes[arm], min(tries, len(notes[arm]))):
                d, R = rate(fit, a.samples, note)
                tried.append((mean(d), note))
                with open(os.path.join(OUT, f"{tag}.tries.jsonl"), "a") as f:
                    f.write(json.dumps(dict(arm=arm, rate=mean(d), note=note), ensure_ascii=False) + "\n")
            if not tried:
                continue
            tried.sort(key=lambda x: -x[0])
            d, R = rate(test, confirm, tried[0][1])
            diffs = [d[k] - base1[k] for k in d if k in base1]
            dump(f"{tag}.{arm}.note1.jsonl", R)
            with open(os.path.join(OUT, "best.jsonl"), "a") as f:
                f.write(json.dumps(dict(cat=cat, tag=tag, arm=arm, note=tried[0][1], fit=tried[0][0], tries=[x[0] for x in tried], base0=mean(base0), base1=mean(base1), note1=mean(d), change1=sum(diffs) / max(1, len(diffs))), ensure_ascii=False) + "\n")
            say(f"  {arm}: half 0 tries " + " ".join(f"{x[0]:.3f}" for x in tried) + f"; best on half 1 {mean(d):.3f}, change {100 * sum(diffs) / max(1, len(diffs)):+.1f}. Note: {tried[0][1]}")
    say("done")


if __name__ == "__main__":
    main()
