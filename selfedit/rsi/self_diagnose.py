"""The model reads its own failed traces and says what went wrong, in its own words, with no cause suggested to it.
Each case: a failed trace (out of budget or wrong) and, where one exists, a right trace of the same problem. A second prompt merges the answers into failure modes.
Writes RUN/cases.jsonl (one reading per case), RUN/marks.jsonl (the lines it pointed at, located in the traces) and OUT (modes with counts and quotes, a digest for the surgeon).

Usage: python -u rsi/self_diagnose.py RUN --traces F1 [F2 ...] --cases 48 --out LIB/diagnosis.md [--diag-effort medium --diag-tokens 6000 --trace-tok 12000]
       [loop.py args: --model --pool --calib --dirlib --effort --budget --tp --max-model-len --max-num-seqs --max-num-batched-tokens --kv-cache-dtype --gpu-util; --mm shows the problem's images again]
"""
import json, os, random, re, sys, time

PROMPT = """Below is a problem{with_images} and your own reasoning on it from earlier runs. The correct answer is {answer}.

Problem:
{problem}

Attempt A {how}. Its reasoning:
<attempt_a>
{a}
</attempt_a>
{b_block}
Work out what actually went wrong in attempt A, in terms of what the reasoning was doing. Do not assume any particular cause; name what you see.
After your reasoning reply with one JSON object:
{{"mode": "<the failure in 2-5 words of your own>", "what_happened": "<under 50 words>", "first_bad_line": "<one line copied exactly from attempt A where it first goes astray>", "better_line": "<one line copied exactly from attempt B showing what it did instead, or empty>", "avoidable": <true if knowledge you already have could have avoided it, false if the problem was simply beyond you>}}"""

B_BLOCK = """
Attempt B reached the correct answer on the same problem in {tokens} tokens. Its reasoning:
<attempt_b>
{b}
</attempt_b>
"""

MERGE = """Below are your own readings of {n} of your failed attempts at problems, one per line: index | your name for the failure | what happened | avoidable.

{rows}

Merge them into at most 8 failure modes that are really different in what the reasoning does. Do not merge things only because they sound alike, and do not split them only by topic. After your reasoning reply with one JSON object:
{{"modes": [{{"name": "<2-5 words>", "definition": "<under 40 words: what the reasoning is doing>", "indices": [<the indices that belong to it>]}}]}}"""

ASSIGN = """Failure modes, one per line: index | name: definition.

{modes}

One of your own readings of a failed attempt: {row}

Which mode does it belong to? Reply with one JSON object: {{"mode_index": <index>}}"""

MERGE_N = 40                                                    # 120 readings in one prompt left no JSON inside the token limit (11-05, p8: empty modes.json), the assign step places the rest


def pop(argv, flag, default=None, many=False):
    if flag not in argv:
        return default
    i = argv.index(flag)
    if many:
        j = i + 1
        while j < len(argv) and not argv[j].startswith("--"):
            j += 1
        vals = argv[i + 1:j]
        del argv[i:j]
        return vals
    val = argv[i + 1]
    del argv[i:i + 2]
    return val


def last_json(text, key):
    """The last JSON object in the text that holds key."""
    dec = json.JSONDecoder()
    for i in reversed([m.start() for m in re.finditer(r"\{", text or "")]):
        try:
            obj, _ = dec.raw_decode(text, i)
        except ValueError:
            continue
        if isinstance(obj, dict) and key in obj:
            return obj
    return None


def main():                                                     # vLLM TP>1 workers re-import this file: nothing at top level
    RUN = sys.argv[1]
    argv = sys.argv[2:]
    files = pop(argv, "--traces", many=True)
    n_cases = int(pop(argv, "--cases", 48))
    out_path = pop(argv, "--out")
    effort = pop(argv, "--diag-effort", "medium")
    tokens = int(pop(argv, "--diag-tokens", 6000))
    trace_tok = int(pop(argv, "--trace-tok", 12000))
    os.makedirs(RUN, exist_ok=True)
    sys.argv = ["loop.py", "--run-dir", RUN] + argv
    CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path[:0] = [CODE, os.path.join(CODE, "engine")]
    import rsi.loop as L
    from surgery import Surgeon, image_message

    a = L.args
    S = Surgeon(a.model, tp=a.tp, max_model_len=a.max_model_len, max_num_seqs=a.max_num_seqs,
                max_num_batched_tokens=a.max_num_batched_tokens, gpu_memory_utilization=a.gpu_util,
                kv_cache_dtype=a.kv_cache_dtype, **({"multimodal": True} if a.mm else {}))
    L._S = S
    pool = {p["id"]: p for p in L.load_jsonl(a.pool)}
    by = {}
    for f in files:
        for i, line in enumerate(open(f)):
            r = json.loads(line)
            if r.get("reasoning") and r["id"] in pool:
                by.setdefault(r["id"], []).append(dict(r, file=f, line=i))
    rng = random.Random(a.seed)
    paired, single = [], []
    for k in sorted(by):
        fail = [r for r in by[k] if r["finish"] == "length" or not r.get("ok")]
        good = sorted((r for r in by[k] if r["finish"] != "length" and r.get("ok")), key=lambda r: r["tokens"])
        if fail:
            (paired if good else single).append((rng.choice(fail), good[0] if good else None))
    rng.shuffle(paired), rng.shuffle(single)
    cases = (paired[: n_cases] + single)[: n_cases]
    L.say(f"self_diagnose: {len(paired)} problems with a failed and a right trace, {len(single)} with only failed; reading {len(cases)}")

    def trim(text):
        ids = S.tokenizer(text, add_special_tokens=False).input_ids
        if len(ids) <= trace_tok:
            return text
        h, t = int(trace_tok * .6), int(trace_tok * .4)
        return S.tokenizer.decode(ids[:h]) + f"\n[... {len(ids) - h - t} tokens not shown ...]\n" + S.tokenizer.decode(ids[-t:])

    prompts = []
    for fail, good in cases:
        how = f"ran out of its {fail['tokens']}-token thinking budget before finishing" if fail["finish"] == "length" else f"finished after {fail['tokens']} tokens with a wrong answer"
        b = B_BLOCK.format(tokens=good["tokens"], b=trim(good["reasoning"])) if good else "\n"
        p = pool[fail["id"]]
        images = p.get("image_paths") if a.mm else None
        text = PROMPT.format(with_images=" with its image(s), shown again" if images else "", answer=fail["answer"], problem=p["prompt"], how=how, a=trim(fail["reasoning"]), b_block=b)
        prompts.append(image_message(text, images) if images else text)
    t0 = time.time()
    outs = S.generate(prompts, n=1, sampling=L.SAMPLING, max_tokens=tokens, reasoning_effort=effort)
    rows, marks = [], []

    def locate(trace, quote):
        """Index of the first line of the trace that holds the quote, ignoring surrounding space; None if the model did not copy it exactly."""
        q = (quote or "").strip()
        if len(q) < 8:
            return None
        for i, l in enumerate(trace.split("\n")):
            if q in l or (len(l.strip()) >= 8 and l.strip() in q):
                return i
        return None

    with open(os.path.join(RUN, "cases.jsonl"), "w") as f:
        for (fail, good), o in zip(cases, outs):
            s = o["samples"][0]
            obj = last_json(s["content"], "mode") or last_json(s["reasoning"], "mode") or {}
            row = {"id": fail["id"], "category": pool[fail["id"]].get("category"), "raw": s["content"], "fail_file": fail["file"], "fail_line": fail["line"], "kind": "truncated" if fail["finish"] == "length" else "wrong",
                   "good_file": good["file"] if good else None, "good_line": good["line"] if good else None, "tokens": s["num_tokens"], "finish": s["finish_reason"], **obj}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if obj.get("mode"):
                rows.append(row)
                bad, better = locate(fail["reasoning"], obj.get("first_bad_line")), locate(good["reasoning"], obj.get("better_line")) if good else None
                n = max(1, len(fail["reasoning"].split("\n")))
                marks.append({"case": len(rows) - 1, "id": fail["id"], "mode": obj["mode"], "fail_file": fail["file"], "fail_line": fail["line"], "bad_line": bad,
                              "bad_frac": None if bad is None else bad / n, "good_file": good["file"] if good else None, "good_line": good["line"] if good else None, "better_line": better})
    with open(os.path.join(RUN, "marks.jsonl"), "w") as f:
        for m in marks:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    found = sum(m["bad_line"] is not None for m in marks)
    L.say(f"self_diagnose: {len(rows)} of {len(cases)} readings parsed, {found} located the line ({time.time() - t0:.0f}s)")

    sample = rows if len(rows) <= MERGE_N else rng.sample(rows, MERGE_N)   # more readings than one prompt can merge: modes from a sample, then each reading is placed
    listing = "\n".join(f"{i} | {r['mode']} | {r.get('what_happened', '')} | {r.get('avoidable')}" for i, r in enumerate(sample))
    o = S.generate([MERGE.format(n=len(sample), rows=listing)], n=1, sampling=L.SAMPLING, max_tokens=tokens * 2, reasoning_effort=effort)[0]["samples"][0]
    merged = last_json(o["content"], "modes") or last_json(o["reasoning"], "modes") or {}
    json.dump({"finish": o["finish_reason"], "tokens": o["num_tokens"], "content": o["content"], "reasoning": (o["reasoning"] or "")[-4000:]}, open(os.path.join(RUN, "merge_raw.json"), "w"), ensure_ascii=False)
    if len(sample) < len(rows) and merged.get("modes"):
        names = "\n".join(f"{k} | {m.get('name')}: {m.get('definition')}" for k, m in enumerate(merged["modes"]))
        outs = S.generate([ASSIGN.format(modes=names, row=f"{r['mode']} | {r.get('what_happened', '')}") for r in rows], n=1, sampling=L.SAMPLING, max_tokens=2000, reasoning_effort="low")
        for m in merged["modes"]:
            m["indices"] = []
        for i, o in enumerate(outs):
            j = (last_json(o["samples"][0]["content"], "mode_index") or {}).get("mode_index")
            if isinstance(j, int) and 0 <= j < len(merged["modes"]):
                merged["modes"][j]["indices"].append(i)
    json.dump(merged, open(os.path.join(RUN, "modes.json"), "w"), indent=1, ensure_ascii=False)

    w = [f"# Diagnosis: your own reading of {len(rows)} of your own failures, no cause suggested", ""]
    used = set()
    for m in sorted(merged.get("modes", []), key=lambda m: -len(m.get("indices", []))):
        ix = [i for i in m.get("indices", []) if isinstance(i, int) and 0 <= i < len(rows)]
        used |= set(ix)
        share = len(ix) / max(1, len(rows))
        av = sum(rows[i].get("avoidable") is True for i in ix)
        trunc = sum(rows[i]["kind"] == "truncated" for i in ix)
        fr = sorted(marks[i]["bad_frac"] for i in ix if marks[i]["bad_frac"] is not None)
        where = f", first goes astray at a median {fr[len(fr) // 2]:.0%} of the trace" if fr else ""
        cats = {}
        for i in ix:
            if rows[i].get("category"):
                cats[rows[i]["category"]] = cats.get(rows[i]["category"], 0) + 1
        if cats:
            where += "; " + ", ".join(f"{k} {v}" for k, v in sorted(cats.items(), key=lambda kv: -kv[1])[:4])
        w.append(f"- **{m.get('name')}** ({len(ix)} cases, {share:.0%}; {trunc} out of budget, {len(ix) - trunc} wrong; avoidable in your reading {av}{where}): {m.get('definition')}")
        for i in ix[:2]:
            if rows[i].get("first_bad_line"):
                w.append(f"  - {rows[i]['id']}: \"{str(rows[i]['first_bad_line'])[:160]}\"")
    if len(used) < len(rows):
        w.append(f"- not placed in a mode: {len(rows) - len(used)} cases")
    text = "\n".join(w) + "\n"
    if out_path:
        open(out_path, "w").write(text)
    print(text, flush=True)
    L.say("self_diagnose: done")


if __name__ == "__main__":
    main()
