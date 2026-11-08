"""Direct dose-response of chosen edits at a lineage's setting, on many frontier problems (independent sampling): one
base eval, then each edit in turn, all on the same problems. Per edit: accuracy difference with its paired 90/98 %
bootstrap interval, tokens, truncation and finished-correct rates. Written after each edit to RUN/dose_result.json,
samples to RUN/base.jsonl and RUN/edit_NN.jsonl.
Usage: python -u rsi/t_dose.py RUN_DIR --edits JSON_FILE [loop.py args: --model --pool --calib --dirlib --effort
       --mb (problems; all frontier problems if larger) --samples ...]"""
import json, os, sys, time


def summary(recs):
    import numpy as np
    fin = [r for r in recs if r["finish"] != "length"]
    return {"acc": float(np.mean([r["ok"] for r in recs])), "tokens": float(np.mean([r["tokens"] for r in recs])),
            "trunc": 1 - len(fin) / len(recs), "fin_correct": float(np.mean([r["ok"] for r in fin])) if fin else None}


def dump(path, recs):
    with open(path, "w") as f:
        for r in recs:
            f.write(json.dumps({k: v for k, v in r.items() if k != "seq"}, ensure_ascii=False) + "\n")


def main():                                                     # vLLM TP>1 workers re-import this file: nothing at top level
    RUN = sys.argv[1]
    i = sys.argv.index("--edits")
    EDITS = json.load(open(sys.argv[i + 1]))
    os.makedirs(RUN, exist_ok=True)
    sys.argv = ["loop.py", "--run-dir", RUN] + sys.argv[2:i] + sys.argv[i + 2:]
    CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path[:0] = [CODE, os.path.join(CODE, "engine")]
    import numpy as np
    import rsi.loop as L
    from rsi.accept import per_problem
    from surgery import Surgeon

    a = L.args
    S = Surgeon(a.model, tp=a.tp, max_model_len=a.max_model_len, max_num_seqs=a.max_num_seqs,
                max_num_batched_tokens=a.max_num_batched_tokens, gpu_memory_utilization=a.gpu_util,
                kv_cache_dtype=a.kv_cache_dtype, **({"multimodal": True} if a.mm else {}))
    L._S = S
    pool, calib = L.load_jsonl(a.pool), L.load_jsonl([a.calib])
    acc = {k: float(np.mean(v)) for k, v in per_problem(calib).items()}
    trunc = {k: float(np.mean(v)) for k, v in per_problem([dict(r, ok=r["finish"] == "length") for r in calib]).items()}
    probs = [p for p in pool if 0 < acc.get(p["id"], 0) < 1 or (acc.get(p["id"]) == 0 and trunc.get(p["id"], 0) >= 0.5)]
    probs = probs[: a.mb]

    t0 = time.time()
    bpath = os.path.join(RUN, "base.jsonl")                     # a relaunch with a new edit list reuses the base
    if os.path.exists(bpath):
        base = [json.loads(l) for l in open(bpath)]
        assert sorted({r["id"] for r in base}) == sorted(p["id"] for p in probs), "cached base is on other problems"
    else:
        base = L.evaluate(S, probs, a.samples)
        dump(bpath, base)
    res = {"problems": len(probs), "samples": a.samples, "effort": a.effort, "budget": a.budget, "base": summary(base),
           "edits": []}
    L.say(f"t_dose: base on {len(probs)} problems x {a.samples}: {res['base']} ({time.time() - t0:.0f}s)")
    for n, raw in enumerate(EDITS):
        if "sampling" in raw:                                   # decoding baseline: temperature T = all logits x 1/T
            saved = dict(L.SAMPLING)
            L.SAMPLING.update(raw["sampling"])
            if "logit_bias" in L.SAMPLING:                      # JSON keys are strings, vLLM's sampler wants token ids
                L.SAMPLING["logit_bias"] = {int(k): v for k, v in L.SAMPLING["logit_bias"].items()}
            t0 = time.time()
            cand = L.evaluate(S, probs, a.samples)
            L.SAMPLING.clear()
            L.SAMPLING.update(saved)
            dump(os.path.join(RUN, f"edit_{n:02d}.jsonl"), cand)
            st = L.compare(base, cand)
            r = {"edit": raw, "summary": summary(cand), "diff": st["diff"], "lo90": st["lo90"], "hi90": st["hi90"],
                 "lo98": st.get("lo98"), "hi98": st.get("hi98"), "dtok": st["dtok"], "tok_rel": st.get("tok", {}).get("rel")}
            res["edits"].append(r)
            L.say(f"t_dose: sampling {raw.get('label', raw['sampling'])} alpha -: diff {r['diff']:+.3f} [{r['lo90']:+.3f},{r['hi90']:+.3f}] "
                  f"dtok {r['dtok']:+.0f} {r['summary']} ({time.time() - t0:.0f}s)")
            with open(os.path.join(RUN, "dose_result.json"), "w") as f:
                json.dump(res, f, indent=1)
            continue
        if "engine" in raw or "combo" in raw:                   # what the menu does not offer: an output row, several edits together
            t0 = time.time()
            parts = raw["engine"] if "engine" in raw else [x["engine"] if "engine" in x else L.engine_edit(dict(L.validate(dict(x, alpha=max(-2.0, min(2.0, x["alpha"]))))[0], alpha=x["alpha"])) for x in raw["combo"]]
            h = S.apply([y for z in parts for y in (z if isinstance(z, list) else [z])])    # a vision edit is a list (weight and bias)
            cand = L.evaluate(S, probs, a.samples)
            S.revert(h)
            dump(os.path.join(RUN, f"edit_{n:02d}.jsonl"), cand)
            st = L.compare(base, cand)
            r = {"edit": raw, "summary": summary(cand), "diff": st["diff"], "lo90": st["lo90"], "hi90": st["hi90"],
                 "lo98": st.get("lo98"), "hi98": st.get("hi98"), "dtok": st["dtok"], "tok_rel": st.get("tok", {}).get("rel")}
            res["edits"].append(r)
            L.say(f"t_dose: engine {raw.get('label', raw.get('engine', 'combo'))} alpha -: diff {r['diff']:+.3f} [{r['lo90']:+.3f},{r['hi90']:+.3f}] "
                  f"dtok {r['dtok']:+.0f} {r['summary']} ({time.time() - t0:.0f}s)")
            with open(os.path.join(RUN, "dose_result.json"), "w") as f:
                json.dump(res, f, indent=1)
            continue
        strong = raw.get("op") == "amplify" and abs(raw.get("alpha", 0)) > 2    # beyond the menu's bound (WSP's edit: alpha 6)
        e, why = L.validate(dict(raw, alpha=2.0 if raw["alpha"] > 0 else -2.0) if strong else raw)
        if strong and e is not None:
            e["alpha"] = raw["alpha"]
        if e is None:
            L.say(f"t_dose: invalid edit {raw}: {why}")
            continue
        t0 = time.time()
        h = S.apply(L.engine_edit(e))
        cand = L.evaluate(S, probs, a.samples)
        S.revert(h)
        dump(os.path.join(RUN, f"edit_{n:02d}.jsonl"), cand)
        st = L.compare(base, cand)
        r = {"edit": raw, "summary": summary(cand), "diff": st["diff"], "lo90": st["lo90"], "hi90": st["hi90"],
             "lo98": st.get("lo98"), "hi98": st.get("hi98"), "dtok": st["dtok"], "tok_rel": st.get("tok", {}).get("rel")}
        res["edits"].append(r)
        L.say(f"t_dose: {L.target(e)} alpha {raw.get('alpha')}: diff {r['diff']:+.3f} [{r['lo90']:+.3f},{r['hi90']:+.3f}] "
              f"dtok {r['dtok']:+.0f} {r['summary']} ({time.time() - t0:.0f}s)")
        with open(os.path.join(RUN, "dose_result.json"), "w") as f:
            json.dump(res, f, indent=1)
    L.say("t_dose: done")


if __name__ == "__main__":
    main()
