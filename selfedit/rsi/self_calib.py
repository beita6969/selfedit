"""Self-calibration of a model on the RSI selection pool, with the lineage's own decoding (rsi/loop.py evaluate():
the model's SAMPLING, thinking effort, budget; truncation counts as wrong).  Writes, in --out-dir:
  calib.jsonl     one record per sample, exactly what loop.py writes for its generation-0 calibration (no seq/start);
                  pass it to loop.py --calib
  traces.jsonl    the same records with the token sequence the model saw (seq) and the answer start (start), the
                  model's own traces for probe/build_library.py
  calib_summary.json  accuracy / truncation / token statistics and the frontier exactly as loop.py main() computes it
Runs in chunks and resumes (problems already in traces.jsonl are skipped).

    python rsi/self_calib.py --model M --pool bundle/rsi/pool.jsonl --out-dir /mnt/ssd/test/selfedit/xf/gemma4 --tp 2"""
import argparse, json, os, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--pool", required=True)
ap.add_argument("--out-dir", required=True)
ap.add_argument("--samples", type=int, default=4)              # loop.py --calib-samples default
ap.add_argument("--effort", default="medium")
ap.add_argument("--budget", type=int, default=8192)
ap.add_argument("--chunk", type=int, default=64)                # problems per generate() call
ap.add_argument("--tp", type=int, default=2)
ap.add_argument("--max-model-len", type=int, default=12288)
ap.add_argument("--max-num-seqs", type=int, default=64)
ap.add_argument("--max-num-batched-tokens", type=int, default=4096)
ap.add_argument("--kv-cache-dtype", default="fp8")
ap.add_argument("--gpu-util", type=float, default=None)
a = ap.parse_args()
os.makedirs(a.out_dir, exist_ok=True)

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [CODE, os.path.join(CODE, "engine")]
# loop.py's own evaluate() and decoding (its module scope parses these arguments; nothing else runs).  sys.argv is
# restored afterwards: vLLM's spawned workers get the parent's sys.argv and re-run this module's top level with it.
_argv = sys.argv
sys.argv = ["loop.py", "--run-dir", os.path.join(a.out_dir, "loop_import"), "--model", a.model, "--pool", a.pool,
            "--effort", a.effort, "--budget", str(a.budget), "--calib-samples", str(a.samples)]
from rsi import loop as LP
sys.argv = _argv
import numpy as np
from rsi.accept import per_problem
from rsi.dossier import pool_summary


def say(*x):
    print(time.strftime("[%H:%M:%S]"), *x, flush=True)


def main():
    from surgery import Surgeon
    pool = LP.load_jsonl([a.pool])
    tr_path, cal_path = os.path.join(a.out_dir, "traces.jsonl"), os.path.join(a.out_dir, "calib.jsonl")
    done = set()
    if os.path.exists(tr_path):
        for l in open(tr_path):
            done.add(json.loads(l)["id"])
    todo = [p for p in pool if p["id"] not in done]
    say(f"family {LP.FAMILY}; sampling {LP.SAMPLING}; effort {a.effort}; budget {a.budget}; "
        f"{len(pool)} problems, {len(done)} done, {len(todo)} to go x {a.samples}")
    if todo:
        S = Surgeon(a.model, tp=a.tp, max_model_len=a.max_model_len, max_num_seqs=a.max_num_seqs,
                    max_num_batched_tokens=a.max_num_batched_tokens, gpu_memory_utilization=a.gpu_util,
                    kv_cache_dtype=a.kv_cache_dtype)
        for i in range(0, len(todo), a.chunk):
            ch = todo[i: i + a.chunk]
            t0 = time.time()
            recs = LP.evaluate(S, ch, a.samples)
            dt = time.time() - t0
            with open(tr_path, "a") as f:
                for r in recs:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            ntok = sum(r["tokens"] for r in recs)
            say(f"chunk {i // a.chunk}: {len(ch)} problems, acc {np.mean([r['ok'] for r in recs]):.3f}, "
                f"trunc {np.mean([r['finish'] == 'length' for r in recs]):.3f}, {ntok} tokens in {dt:.0f}s "
                f"({ntok / dt:.0f} tok/s)")
    # calib.jsonl in loop.py's format, in pool order, from the traces
    recs = [json.loads(l) for l in open(tr_path)]
    order = {p["id"]: i for i, p in enumerate(pool)}
    recs.sort(key=lambda r: (order[r["id"]], r["sample"]))
    with open(cal_path, "w") as f:
        for r in recs:
            f.write(json.dumps({k: v for k, v in r.items() if k not in ("seq", "start")}, ensure_ascii=False) + "\n")
    calib = LP.load_jsonl([cal_path])
    acc = {k: float(np.mean(v)) for k, v in per_problem(calib).items()}
    trunc = {k: float(np.mean(v)) for k, v in per_problem([dict(r, ok=r["finish"] == "length") for r in calib]).items()}
    frontier = [p for p in pool if 0 < acc.get(p["id"], 0) < 1 or (acc.get(p["id"]) == 0 and trunc.get(p["id"], 0) >= 0.5)]
    toks = np.array([r["tokens"] for r in calib])
    summ = dict(family=LP.FAMILY, model=a.model, sampling=LP.SAMPLING, effort=a.effort, budget=a.budget,
                samples=a.samples, n_records=len(calib), n_problems=len(acc),
                accuracy=float(np.mean([r["ok"] for r in calib])), truncated=float(np.mean([r["finish"] == "length" for r in calib])),
                looped=float(np.mean([bool(r.get("loop")) for r in calib])),
                tokens=dict(mean=float(toks.mean()), p50=float(np.median(toks)), p90=float(np.percentile(toks, 90))),
                solved_always=sum(v == 1 for v in acc.values()), solved_never=sum(v == 0 for v in acc.values()),
                frontier=len(frontier), frontier_ids=[p["id"] for p in frontier], pool_summary=pool_summary(calib))
    json.dump(summ, open(os.path.join(a.out_dir, "calib_summary.json"), "w"), indent=1)
    say(f"calib: acc {summ['accuracy']:.3f}, truncated {summ['truncated']:.3f}, frontier {len(frontier)} / {len(pool)}\n"
        + summ["pool_summary"])


if __name__ == "__main__":
    main()
