"""How an edit changes what the model reads from the image, without generating: thinking off, one forward pass per problem,
log-odds of the right option at "\\boxed{". The read is deterministic, so small effects show and a whole vision-tower scan takes minutes.
Writes RUN/items.jsonl (base read per problem) and RUN/read.jsonl (one row per edit: change overall and per bench, with paired 90 % intervals).

Usage: python -u rsi/mm_read.py RUN --edits JSON_FILE [--acc CALIB.jsonl] [--thin blink:3,cvbench:10 keeps every k-th problem of a bench] [loop.py args, with --mm]
JSON_FILE: [{"label": .., "combo": [menu edits]}, {"label": .., "engine": [engine edits]} or a menu edit with "label"]. CALIB gives each problem's accuracy with thinking.
"""
import json, os, re, sys, time


def pop(argv, flag, default=None):
    if flag not in argv:
        return default
    i = argv.index(flag)
    val = argv[i + 1]
    del argv[i:i + 2]
    return val


def letters_of(p):
    """The option letters offered in the prompt, gold included."""
    found = re.findall(r"\(([A-Z])\)", p["prompt"]) or re.findall(r"(?:^|\s)([A-Z])\.\s", p["prompt"])
    return sorted(set(found) | {p["answer"]})


def main():                                                     # vLLM TP>1 workers re-import this file: nothing at top level
    RUN = sys.argv[1]
    argv = sys.argv[2:]
    edits = json.load(open(pop(argv, "--edits")))
    acc_path = pop(argv, "--acc")
    thin_spec = pop(argv, "--thin")
    os.makedirs(RUN, exist_ok=True)
    sys.argv = ["loop.py", "--run-dir", RUN] + argv
    CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path[:0] = [CODE, os.path.join(CODE, "engine")]
    import numpy as np
    import rsi.loop as L
    from surgery import Surgeon, image_message
    from vllm import SamplingParams

    a = L.args
    S = Surgeon(a.model, tp=a.tp, max_model_len=a.max_model_len, max_num_seqs=a.max_num_seqs,
                max_num_batched_tokens=a.max_num_batched_tokens, gpu_memory_utilization=a.gpu_util,
                kv_cache_dtype=a.kv_cache_dtype, multimodal=True)
    L._S = S
    pool = [p for p in L.load_jsonl(a.pool) if p.get("image_paths")]
    heads = S.generate([image_message(p["prompt"], p["image_paths"]) for p in pool], n=1, sampling=dict(L.SAMPLING), max_tokens=1, enable_thinking=False)
    L.say("mm_read: prompt ends " + repr(S.tokenizer.decode(heads[0]["prompt_token_ids"][-12:])))

    def tok(t):
        return S.tokenizer(t, add_special_tokens=False).input_ids

    thin = dict((kv.split(":")[0], int(kv.split(":")[1])) for kv in thin_spec.split(",")) if thin_spec else {}
    seen = {}
    prompts, cands, items = [], [], []
    for p, o in zip(pool, heads):
        seen[p["bench"]] = seen.get(p["bench"], 0) + 1
        if seen[p["bench"]] % thin.get(p["bench"], 1):
            continue
        head = list(o["prompt_token_ids"])
        if p["kind"] == "mc":
            opts = letters_of(p)
            if len(opts) < 2:
                continue
            ids = [tok("\\boxed{" + x) for x in opts]
            n = next(i for i in range(len(ids[0]) + 1) if i == len(ids[0]) or any(i >= len(y) or y[i] != ids[0][i] for y in ids))
            if any(len(y) <= n for y in ids):
                continue
            c = [y[n] for y in ids]
            g = opts.index(p["answer"])
        else:
            ids = [tok("\\boxed{" + p["answer"])]
            n = len(tok("\\boxed{"))
            if len(ids[0]) <= n:
                continue
            c, g = [ids[0][n]], 0
        prompts.append(S._token_prompt(head + ids[0][:n], p["image_paths"]))
        cands.append((c, g))
        items.append({"id": p["id"], "bench": p["bench"], "kind": p["kind"], "category": p.get("category"), "n_options": len(c)})
    params = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20, detokenize=False)
    L.say(f"mm_read: {len(items)} problems, one forward each")

    def read():
        outs = S.llm.generate(prompts, [params] * len(prompts), use_tqdm=False)
        out = np.zeros(len(items))
        right = np.full(len(items), np.nan)
        for k, (o, (c, g)) in enumerate(zip(outs, cands)):
            top = o.outputs[0].logprobs[0]                      # the 20 likeliest next tokens; below them a token counts as just under the last
            floor = min(v.logprob for v in top.values()) - 1.0
            lp = np.array([top[t].logprob if t in top else floor for t in c], dtype=np.float64)
            if len(c) == 1:
                out[k] = lp[0]
                continue
            rest = np.delete(lp, g)
            out[k] = lp[g] - np.logaddexp.reduce(rest)
            right[k] = lp[g] > rest.max()
        return out, right

    t0 = time.time()
    base, base_right = read()
    secs = time.time() - t0
    mc = np.array([i["kind"] == "mc" for i in items])
    benches = sorted({i["bench"] for i in items})
    with open(os.path.join(RUN, "items.jsonl"), "w") as f:
        for it, m, r in zip(items, base, base_right):
            f.write(json.dumps(dict(it, margin=float(m), right=None if np.isnan(r) else bool(r))) + "\n")
    line = f"mm_read: base read in {secs:.0f}s; no-thinking right {np.nanmean(base_right):.3f} on {mc.sum()} option problems"
    for b in benches:
        k = np.array([i["bench"] == b for i in items]) & mc
        if k.any():
            line += f"; {b} {np.nanmean(base_right[k]):.3f}"
    L.say(line)
    if acc_path:
        rows = {}
        for r in L.load_jsonl([acc_path]):
            rows.setdefault(r["id"], []).append(bool(r["ok"]))
        acc = np.array([np.mean(rows[i["id"]]) if i["id"] in rows else np.nan for i in items])
        known = mc & ~np.isnan(acc)
        L.say(f"mm_read: margin against accuracy with thinking, correlation {np.corrcoef(base[known], acc[known])[0, 1]:.3f}; right without thinking {base_right[known].mean():.3f}, with thinking {acc[known].mean():.3f}")
        for b in benches:
            k = known & np.array([i["bench"] == b for i in items])
            if k.sum() > 2:
                L.say(f"  {b}: correlation {np.corrcoef(base[k], acc[k])[0, 1]:.3f}; right without thinking {base_right[k].mean():.3f}, with thinking {acc[k].mean():.3f}")

    rng = np.random.default_rng(0)

    def stat(d):
        if not d.size:
            return None
        idx = rng.integers(0, d.size, (2000, d.size))
        m = d[idx].mean(1)
        return {"d": float(d.mean()), "lo": float(np.quantile(m, .05)), "hi": float(np.quantile(m, .95)), "n": int(d.size)}

    def flat(x):
        return [y for z in x for y in (z if isinstance(z, list) else [z])]

    def engine_of(raw):
        if "engine" in raw:
            return raw["engine"]
        if "combo" in raw:
            return flat([L.engine_edit(L.validate(x)[0]) for x in raw["combo"]])
        e, why = L.validate({k: v for k, v in raw.items() if k != "label"})
        if e is None:
            raise ValueError(why)
        return flat([L.engine_edit(e)])

    done = set()
    path = os.path.join(RUN, "read.jsonl")
    if os.path.exists(path):
        done = {json.loads(l)["label"] for l in open(path)}
    for raw in edits:
        if raw["label"] in done:
            continue
        t0 = time.time()
        try:
            h = S.apply(engine_of(raw))
        except Exception as ex:
            L.say(f"mm_read: {raw['label']}: {ex}")
            continue
        m, r = read()
        S.revert(h)
        d = m - base
        row = {"label": raw["label"], "edit": raw, "all": stat(d[mc]), "open": stat(d[~mc]), "right": float(np.nanmean(r) - np.nanmean(base_right)),
               "by": {b: stat(d[mc & np.array([i["bench"] == b for i in items])]) for b in benches}, "secs": time.time() - t0}
        with open(path, "a") as f:
            f.write(json.dumps(row) + "\n")
        L.say(f"mm_read: {raw['label']}: margin {row['all']['d']:+.3f} [{row['all']['lo']:+.3f},{row['all']['hi']:+.3f}] right {row['right']:+.3f} " +
              " ".join(f"{b} {row['by'][b]['d']:+.3f}" for b in benches if row["by"][b]) + (f" open {row['open']['d']:+.3f}" if row["open"] else "") + f" ({row['secs']:.0f}s)")
    m, _ = read()
    L.say(f"mm_read: done; base read again differs by {np.abs(m - base).max():.4f} at most")


if __name__ == "__main__":
    main()
