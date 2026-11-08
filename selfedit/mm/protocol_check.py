"""Official-protocol re-runs for BabyVision and ERQA (stage 2, step 1): is the gap to Qwen's reported numbers a protocol
artefact or a real weakness?

build:  python protocol_check.py build --base RUNS/q38_xhigh_n4.final.jsonl --out DATA/protocol
        Same items as our baseline (the BabyVision / ERQA test and pool items that have 4 samples there), re-prompted as
        the official harnesses do:
        - BabyVision (github.com/UniPat-AI/BabyVision, babyvision_eval/evaluate_model.py): image first, then
          question [+ "\\nChoices:\\n(A) ...\\n(B) ..."] + "\\nThink about the question and give your final answer in
          \\boxed{Answer} format."; no system prompt; reasoning on; graded by an LLM judge on the last \\boxed{}.
        - ERQA (github.com/embodiedreasoning/ERQA, eval_harness.py): the original question text, which ends with
          "Please answer directly with only the letter of the correct option and nothing else.", images placed by
          visual_indices (all first when empty); exact match of the stripped, '.'-free reply.
judge:  python protocol_check.py judge --server URL --in OUT.jsonl [...]  (needs the vLLM server)
        The official BabyVision judge prompt (utils.py LLM_JUDGE_PROMPT) answered by Qwen3.8 itself in non-thinking
        mode at temperature 0, as a stand-in for the official judge (GPT-5.2 or Qwen-Max; no API access here).
report: python protocol_check.py report --base ... --official OUT.jsonl
"""
import argparse, json, os, re, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import grade_mm as g  # noqa: E402

BV_SUFFIX = "\nThink about the question and give your final answer in \\boxed{Answer} format."
ERQA_TAIL = "Please answer directly with only the letter of the correct option and nothing else."
LLM_JUDGE_PROMPT = """You are a careful and strict evaluator. You will be given:

1. **Question**
2. **Ground Truth Answer** (correct answer)
3. **Model Output** (answer from another model)

**Your goal:** Determine if the Model Output **accurately matches** the Ground Truth Answer in meaning.

* Matching means: the facts, entities, and key details are equivalent, even if phrasing differs.
* Not matching means: the Model Output is wrong, incomplete, contains extra incorrect facts, or changes the meaning.

**Process (internal reasoning):**

1. Read and understand the Question, Ground Truth Answer, and Model Output.
2. Ignore small wording differences, formatting, or synonyms.
3. If all factual content matches, conclude `1`. Otherwise, conclude `0`.

**Important:**

* Think through your decision step-by-step **internally** before responding.
* In your final output, return **only** True or False, with no extra text or explanation.

**Output format:**

True

or

False

**Input:**

Question: {question},
Ground Truth Answer: {groundtruth},
Model Output: {modeloutput}
"""


def extract_boxed_official(text):
    """babyvision_eval/utils.py extract_boxed_answer (same regex, same fallback)."""
    import regex
    if text is None:
        return None
    m = regex.findall(r'\\boxed\{((?:[^{}]|{(?:[^{}]|{.*})*})*)\}', text)
    if m:
        return m[-1]
    m = regex.findall(r'<\|begin_of_box\|>(.*?)<\|end_of_box\|>', text)
    return m[-1].strip() if m else None


def erqa_official_ok(r):
    """eval_harness.py: response_text.replace(".", "").strip().lower() == answer.strip().lower()."""
    return r["finish"] == "stop" and (r["content"] or "").replace(".", "").strip().lower() == r["answer"].strip().lower()


def erqa_lenient_letter(r):
    """Our reading: a single option letter anywhere at the end of the reply (bold, parentheses, 'Answer: B')."""
    if r["finish"] != "stop" or not r["content"]:
        return None
    m = re.findall(r"(?<![A-Za-z])\(?([A-J])\)?(?![A-Za-z])", r["content"].strip()[-40:])
    return m[-1] if m else None


def load(path):
    return [json.loads(l) for l in open(path)]


def full_items(recs, sets):
    n = defaultdict(int)
    for r in recs:
        if r["set"] in sets and r.get("finish") != "error":
            n[(r["set"], r["id"])] += 1
    return {k for k, c in n.items() if c >= 4}


def build(a):
    import pandas as pd
    recs = load(a.base)
    keep = full_items(recs, {"sets/babyvision", "pool/babyvision", "sets/erqa", "pool/erqa"})
    items = {}
    for d in ("sets", "pool"):
        for b in ("babyvision", "erqa"):
            for l in open(f"{a.data}/{d}/{b}.jsonl"):
                it = json.loads(l)
                if (f"{d}/{b}", it["id"]) in keep:
                    items[(d, it["id"])] = it
    bv = pd.read_parquet(f"{a.data}/raw/UnipatAI__BabyVision/data/train-00000-of-00001.parquet")
    bv = {f"babyvision/{r.taskId}": r for r in bv.itertuples(index=False)}
    eq = pd.read_parquet(f"{a.data}/raw/FlagEval__ERQA/data/test-00000-of-00001.parquet")
    eq = {f"erqa/{r.question_id}": r for r in eq.itertuples(index=False)}
    os.makedirs(f"{a.out}/bvoff", exist_ok=True)
    out = defaultdict(list)
    for (d, iid), it in sorted(items.items()):
        new = dict(it)
        if iid.startswith("babyvision/"):
            r = bv[iid]
            q = r.question
            if r.ansType == "choice":
                q += "\nChoices:\n" + "\n".join(f"({chr(65 + i)}) {o}" for i, o in enumerate(list(r.options)))
                new["judge_gold"] = chr(65 + int(r.choiceAns))
            else:
                new["judge_gold"] = r.blankAns
            new["prompt"] = q + BV_SUFFIX
            new["judge_question"] = new["prompt"]
            out[f"bvoff/babyvision_{'test' if d == 'sets' else 'pool'}"].append(new)
        else:
            r = eq[iid]
            q = r.question
            assert q.rstrip().endswith(ERQA_TAIL), iid
            # our prompt = interleaved text with <image> placeholders + "\n\n" + boxed suffix; the placeholders sit at
            # the visual_indices positions of the stripped question, which is a prefix of the original question
            body = it["prompt"].rsplit("\n\n", 1)[0]
            plain = body.replace("<image>", "")
            assert q.startswith(plain.rstrip()), iid
            new["prompt"] = body + q[len(plain):] if q.startswith(plain) else body.rstrip() + q[len(plain.rstrip()):]
            assert new["prompt"].replace("<image>", "") == q, iid
            out[f"bvoff/erqa_{'test' if d == 'sets' else 'pool'}"].append(new)
    for name, rows in out.items():
        with open(f"{a.out}/{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(name, len(rows))


def judge(a):
    from openai import OpenAI
    from concurrent.futures import ThreadPoolExecutor
    cli = OpenAI(base_url=a.server, api_key="none", timeout=600)
    items = {}
    for f in a.items:
        for l in open(f):
            it = json.loads(l)
            items[it["id"]] = it

    def one(r):
        it = items[r["id"]]
        q = it.get("judge_question") or it["prompt"]
        gold = it.get("judge_gold") or it.get("gold_raw") or it["answer"]
        pred = extract_boxed_official(r.get("content") or "")
        c = cli.chat.completions.create(model="m", temperature=0.0, max_tokens=16, messages=[{"role": "user", "content":
                                        LLM_JUDGE_PROMPT.format(question=q, groundtruth=gold, modeloutput=pred)}],
                                        extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        t = (c.choices[0].message.content or "").strip().lower()
        return dict(id=r["id"], set=r["set"], sample=r["sample"], judge="true" in t, judge_raw=t[:40], extracted=pred,
                    rule=bool(g.correct(r)[0]))

    for path in a.inp:
        recs = [r for r in load(path) if r["id"].startswith("babyvision/") and r.get("finish") != "error" and r["id"] in items]
        with ThreadPoolExecutor(32) as ex:
            res = list(ex.map(one, recs))
        with open(path + ".judge.jsonl", "w") as f:
            for x in res:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        print(path, "judged", len(res))


def acc(xs):
    return 100.0 * sum(xs) / max(1, len(xs))


def report(a):
    base = load(a.base)
    off = load(a.official)
    lines = []
    for bench, sets in (("babyvision", ("sets/babyvision", "pool/babyvision")), ("erqa", ("sets/erqa", "pool/erqa"))):
        for split, s_base in zip(("test", "pool"), sets):
            s_off = f"bvoff/{bench}_{split}"
            ob = [r for r in off if r["set"] == s_off and r.get("finish") != "error"]
            ids = {r["id"] for r in ob}
            bb = [r for r in base if r["set"] == s_base and r["id"] in ids and r.get("finish") != "error"]
            row = dict(bench=bench, split=split, items=len(ids), base_n=len(bb), off_n=len(ob),
                       base_rule=acc([g.correct(r)[0] for r in bb]))
            if bench == "babyvision":
                row["off_rule"] = acc([g.correct(r)[0] for r in ob])
                for name, path in (("base", a.base), ("off", a.official)):
                    jp = path + ".judge.jsonl"
                    if os.path.exists(jp):
                        js = [x for x in load(jp) if x["id"] in ids and x["set"] == (s_base if name == "base" else s_off)]
                        row[f"{name}_judge"] = acc([x["judge"] for x in js])
                        row[f"{name}_judge_vs_rule_disagree"] = sum(x["judge"] != x["rule"] for x in js)
                row["off_trunc"] = acc([r["finish"] == "length" for r in ob])
                row["off_tok_mean"] = sum(r["tokens"] for r in ob) / max(1, len(ob))
                row["base_tok_mean"] = sum(r["tokens"] for r in bb) / max(1, len(bb))
                by = defaultdict(list)
                for r in ob:
                    by[r["category"].split("/")[0]].append(g.correct(r)[0])
                row["off_rule_by_type"] = {k: round(acc(v), 1) for k, v in sorted(by.items())}
            else:
                row["off_exact"] = acc([erqa_official_ok(r) for r in ob])
                row["off_lenient"] = acc([erqa_lenient_letter(r) == r["answer"] for r in ob])
                row["off_tok_mean"] = sum(r["tokens"] for r in ob) / max(1, len(ob))
                row["base_tok_mean"] = sum(r["tokens"] for r in bb) / max(1, len(bb))
            lines.append(row)
            print(json.dumps(row, ensure_ascii=False))
    json.dump(lines, open(a.official + ".report.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("build")
    b.add_argument("--base", required=True)
    b.add_argument("--data", default="/data/fd/mingdazhang/selfedit/data/mm")
    b.add_argument("--out", required=True)
    j = sp.add_parser("judge")
    j.add_argument("--server", required=True)
    j.add_argument("--items", nargs="+", required=True)
    j.add_argument("--in", dest="inp", nargs="+", required=True)
    r = sp.add_parser("report")
    r.add_argument("--base", required=True)
    r.add_argument("--official", required=True)
    a = ap.parse_args()
    {"build": build, "judge": judge, "report": report}[a.cmd](a)
