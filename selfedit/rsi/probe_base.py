"""Natural experiments on the unedited model: how it responds to its own settings, before any idea of what is wrong with it.
One generous budget per run gives the whole curve, since a trace that finished within b tokens is the same trace at budget b.
Writes RUN/<label>.jsonl (traces) and RUN/probes.jsonl (one row per probe and problem set, with the curve).

Usage: python -u rsi/probe_base.py RUN --probes PROBES.json --sets NAME:FILE[:N] ... [--curve 4096,8192,16384,32768] [loop.py args, --budget is the generous one]
PROBES.json: [{"label": "low", "effort": "low", "temperature": 1.0, "engine": [engine edits, optional]}, ...]. FILE holds problems, or traces whose ids select problems from --pool.
"""
import json, os, sys, time


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


def main():                                                     # vLLM TP>1 workers re-import this file: nothing at top level
    RUN = sys.argv[1]
    argv = sys.argv[2:]
    probes = json.load(open(pop(argv, "--probes")))
    sets = pop(argv, "--sets", many=True)
    curve = [int(x) for x in pop(argv, "--curve", "4096,8192,16384,32768").split(",")]
    os.makedirs(RUN, exist_ok=True)
    sys.argv = ["loop.py", "--run-dir", RUN] + argv
    CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path[:0] = [CODE, os.path.join(CODE, "engine")]
    import rsi.loop as L
    from surgery import Surgeon

    a = L.args
    S = Surgeon(a.model, tp=a.tp, max_model_len=a.max_model_len, max_num_seqs=a.max_num_seqs,
                max_num_batched_tokens=a.max_num_batched_tokens, gpu_memory_utilization=a.gpu_util,
                kv_cache_dtype=a.kv_cache_dtype, **({"multimodal": True} if a.mm else {}))
    L._S = S
    pool = {p["id"]: p for p in L.load_jsonl(a.pool)}
    problem_sets = {}
    for spec in sets:
        name, path, *n = spec.split(":")
        seen, ps = set(), []
        for r in L.load_jsonl([path]):
            if r["id"] not in seen:
                seen.add(r["id"])
                ps.append(r if "prompt" in r else pool[r["id"]])
        problem_sets[name] = ps[: int(n[0])] if n else ps
    base_sampling = dict(L.SAMPLING)
    rows_path = os.path.join(RUN, "probes.jsonl")
    done = {(r["label"], r["set"]) for r in map(json.loads, open(rows_path))} if os.path.exists(rows_path) else set()
    for name, probs in problem_sets.items():
        for p in probes:
            if (p["label"], name) in done or name not in p.get("sets", [name]):
                continue
            a.effort = p["effort"]
            L.SAMPLING.clear()
            L.SAMPLING.update(base_sampling)
            if "temperature" in p:
                L.SAMPLING["temperature"] = p["temperature"]
            t0 = time.time()
            h = S.apply(p["engine"]) if p.get("engine") else None
            L.mm_eval.PARTIAL = os.path.join(RUN, f"{p['label']}.{name}.partial")
            R = L.evaluate(S, probs, a.samples)
            if h is not None:
                S.revert(h)
            with open(os.path.join(RUN, f"{p['label']}.{name}.jsonl"), "w") as f:
                for r in R:
                    f.write(json.dumps({k: v for k, v in r.items() if k != "seq"}, ensure_ascii=False) + "\n")
            pts = {}
            for b in curve:
                fin = [r for r in R if r["finish"] != "length" and r["tokens"] <= b]
                pts[b] = {"acc": sum(bool(r["ok"]) for r in fin) / max(1, len(R)), "trunc": 1 - len(fin) / max(1, len(R)),
                          "fin_correct": sum(bool(r["ok"]) for r in fin) / max(1, len(fin))}
            row = dict(label=p["label"], set=name, effort=p["effort"], temperature=L.SAMPLING.get("temperature"), problems=len(probs), samples=a.samples,
                       tokens=sum(r["tokens"] for r in R) / max(1, len(R)), curve=pts, secs=time.time() - t0)
            with open(rows_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            L.say(f"probe {p['label']} {name}: tokens {row['tokens']:.0f} " + " ".join(f"{b}: acc {v['acc']:.3f} trunc {v['trunc']:.3f} fin {v['fin_correct']:.3f}" for b, v in pts.items()) + f" ({row['secs']:.0f}s)")
    L.say("probe_base: done")


if __name__ == "__main__":
    main()
