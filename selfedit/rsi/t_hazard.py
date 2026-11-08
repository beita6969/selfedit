"""The stop-decision read of rsi/hazard.py for a list of edits on a saved base. Entries {"random": SEED, "param": ..., "alpha": ...}
give the null line with a random unit direction: any perturbation raises a rare token's log-prob; {"engine": EDIT} goes to the engine as is
(what the menu does not offer, e.g. an output row); {"combo": [EDIT, ...], "label": ...} applies several together. One row per edit in RUN/hazard.jsonl.
Usage: python -u rsi/t_hazard.py RUN_DIR --edits JSON --base BASE.jsonl [--traces 24] [--end-base LOOSE.jsonl --end-k 30] [loop.py args]
--end-base adds d_end_right, d_end_wrong: the close at the natural end of finished right and wrong traces of that file.
--style-base FILE [--style-k 24 --style-tok 1536] adds d_style (rsi/hazard.py StyleRead), from same-problem correct traces of that file with tokens differing by at least 1.3.
Edits already in RUN/hazard.jsonl are skipped, so a stopped scan resumes."""
import json, os, sys, time, zlib


def pop(flag, default=None):
    if flag not in sys.argv:
        return default
    i = sys.argv.index(flag)
    v = sys.argv[i + 1]
    del sys.argv[i: i + 2]
    return v


def main():                                                     # vLLM TP>1 workers re-import this file: nothing at top level
    RUN = sys.argv[1]
    EDITS = json.load(open(pop("--edits")))
    BASE, K = pop("--base"), int(pop("--traces", 24))
    END_BASE, END_K = pop("--end-base"), int(pop("--end-k", 30))
    STYLE_BASE, STYLE_K, STYLE_TOK = pop("--style-base"), int(pop("--style-k", 24)), int(pop("--style-tok", 1536))
    os.makedirs(RUN, exist_ok=True)
    sys.argv = ["loop.py", "--run-dir", RUN] + sys.argv[2:]
    CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path[:0] = [CODE, os.path.join(CODE, "engine")]
    import numpy as np
    import rsi.loop as L
    from rsi.hazard import EndSplit, Hazard, StyleRead, answer_marks, family_tokens, pick
    from surgery import Surgeon

    a = L.args
    S = Surgeon(a.model, tp=a.tp, max_model_len=a.max_model_len, max_num_seqs=a.max_num_seqs,
                max_num_batched_tokens=a.max_num_batched_tokens, gpu_memory_utilization=a.gpu_util,
                kv_cache_dtype=a.kv_cache_dtype)
    L._S = S
    tk = S.tokenizer
    lead, _, _ = family_tokens(S)
    pool = {p["id"]: p for p in L.load_jsonl(a.pool)}
    picked = pick([json.loads(l) for l in open(BASE)], K, np.random.default_rng(a.seed))
    traces = []
    for r in picked:
        th = tk(r["reasoning"], add_special_tokens=False).input_ids
        traces.append((tk(S.render(pool[r["id"]]["prompt"], reasoning_effort=a.effort), add_special_tokens=False).input_ids + lead,
                       th, r["finish"] != "length", *answer_marks(tk, th, r["answer"])))

    t0 = time.time()
    hz = Hazard(S, traces)
    stops = hz.base_stop()
    cut, end, ans = stops[0], stops[1], stops[2] if len(stops) > 2 else float("nan")
    L.say(f"t_hazard: {len(picked)} traces, {len(hz.jobs)} sequences, {hz.tokens} tokens, base read {time.time() - t0:.0f}s; "
          f"base log p(stop): cuts {cut:.2f}, natural end {end:.2f}, after answer lines {ans:.2f} ({len(hz.base['ans'])} traces)")

    es = None
    if END_BASE:
        recs = [json.loads(l) for l in open(END_BASE)]
        rng = np.random.default_rng(a.seed)
        rng.shuffle(recs)

        def group(ok):
            rows = [r for r in recs if r["finish"] != "length" and bool(r["ok"]) == ok and r["reasoning"]][:END_K]
            return [(tk(S.render(pool[r["id"]]["prompt"], reasoning_effort=a.effort), add_special_tokens=False).input_ids + lead,
                     tk(r["reasoning"].rstrip(), add_special_tokens=False).input_ids) for r in rows]
        es = EndSplit(S, group(True), group(False))
        L.say(f"t_hazard: end split {es.kinds.count('right')} right, {es.kinds.count('wrong')} wrong; base close {es.base[np.array(es.kinds) == 'right'].mean():.2f}, {es.base[np.array(es.kinds) == 'wrong'].mean():.2f}")

    sr = None
    if STYLE_BASE:
        by = {}
        for l in open(STYLE_BASE):
            r = json.loads(l)
            if r["finish"] != "length" and r.get("ok") and r.get("reasoning") and r["id"] in pool:
                by.setdefault(r["id"], []).append(r)
        pairs = []
        for k in sorted(by, key=lambda k: zlib.crc32(k.encode())):
            rs = sorted(by[k], key=lambda r: r["tokens"])
            if len(rs) >= 2 and rs[-1]["tokens"] >= 1.3 * rs[0]["tokens"] and len(pairs) < STYLE_K:
                head = tk(S.render(pool[k]["prompt"], reasoning_effort=a.effort), add_special_tokens=False).input_ids + lead
                pairs.append((head, tk(rs[0]["reasoning"], add_special_tokens=False).input_ids[:STYLE_TOK], tk(rs[-1]["reasoning"], add_special_tokens=False).input_ids[:STYLE_TOK]))
        sr = StyleRead(S, pairs)
        L.say(f"t_hazard: style read on {len(pairs)} pairs")

    out_path = os.path.join(RUN, "hazard.jsonl")
    done = {json.dumps(json.loads(l)["edit"], sort_keys=True) for l in open(out_path)} if os.path.exists(out_path) else set()
    for raw in EDITS:
        if json.dumps(raw, sort_keys=True) in done:
            continue
        t1 = time.time()
        if "engine" in raw:
            eng = raw["engine"]
        elif "combo" in raw:
            eng = [x["engine"] if "engine" in x else L.engine_edit(dict(L.validate(dict(x, alpha=max(-2.0, min(2.0, x["alpha"]))))[0], alpha=x["alpha"])) for x in raw["combo"]]
        elif "random" in raw:
            v = np.random.default_rng(raw["random"]).standard_normal(int(S.spec(raw["param"])["shape"][0]))
            eng = {"op": "amplify", "param": raw["param"], "side": "out", "alpha": raw["alpha"], "u": v / np.linalg.norm(v)}
        else:
            strong = raw.get("op") == "amplify" and abs(raw.get("alpha", 0)) > 2
            e, why = L.validate(dict(raw, alpha=2.0 if raw["alpha"] > 0 else -2.0) if strong else raw)
            if e is None:
                L.say(f"t_hazard: invalid edit {raw}: {why}")
                continue
            if strong:
                e["alpha"] = raw["alpha"]
            eng = L.engine_edit(e)
        h = S.apply(eng)
        row = dict(hz.effect(), edit=raw, secs=time.time() - t1)
        if es is not None:
            row.update(es.effect())
        if sr is not None:
            row.update(sr.effect())
        S.revert(h)
        row["secs"] = time.time() - t1
        with open(out_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        L.say(f"t_hazard: {raw.get('param', 'engine')} {str(raw.get('label', raw.get('u', raw.get('engine', 'random'))))[:60]} alpha {raw.get('alpha')}: stop cuts {row['d_cut']:+.3f}+-{row['se_cut']:.3f} "
              f"ans {row['d_ans']:+.3f}+-{row['se_ans']:.3f} post {row['d_post']:+.3f} pre {row['d_pre']:+.3f} end {row['d_end']:+.3f}+-{row['se_end']:.3f} all {row['d_all']:+.4f} revisit {row['d_rev']:+.3f}+-{row['se_rev']:.3f} ({row['secs']:.0f}s)")
    L.say("t_hazard: done")


if __name__ == "__main__":
    main()
