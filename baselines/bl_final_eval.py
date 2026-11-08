"""Final evaluation of a finished baseline run, as its own step (coordinator 11-08: prepared, run only with the owner's go).
Grades one weights folder (default RUN/rollout_model, the final weights at the cap, bf16 as the engines served them) on IID or OOD rows, with the
trainer's evaluation exactly: main system's sampler (EVAL_SAMPLING), fixed seed 101+i per row, rows split across the engines, grade_rec.
Engines: bl_rollout.py servers already running and ready on the given ctl directories. Gold is never printed; per-row answers go to the results folder.
Usage: bl_final_eval.py --name grpo2 --set iid --rows BL_DATA/iid_eval.node.jsonl --ctl CTL... --owner-go "<who/when>" [--model DIR] [--tag final]"""
import argparse, json, os, pickle, time, uuid
from bl_common import *
ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True); ap.add_argument("--set", required=True, choices=["iid", "ood"]); ap.add_argument("--rows", required=True, nargs="+")
ap.add_argument("--ctl", nargs="+", required=True); ap.add_argument("--model", default=""); ap.add_argument("--tag", default="final")
ap.add_argument("--budget", type=int, required=True, choices=BUDGETS); ap.add_argument("--effort", default=EFFORT)   # must equal the main system's
ap.add_argument("--owner-go", required=True)      # who gave the go and when; written into the results
a = ap.parse_args()
RUN = f"{SHM}/{a.name}"; RES = f"{OUT}/results/{a.name}"; M = a.model or f"{RUN}/rollout_model"
assert os.path.isdir(M), M
out_f = f"{RES}/{a.set}_{a.tag}.jsonl"; assert not os.path.exists(out_f), f"{out_f} exists; not overwritten"


def ask(ops):
    ids = []
    for c, op in zip(a.ctl, ops):
        i = f"{time.time():.3f}-{uuid.uuid4().hex[:6]}"
        json.dump(op, open(f"{c}/.c{i}", "w")); os.rename(f"{c}/.c{i}", f"{c}/cmd_{i}.json"); ids.append(i)
    res = []
    for c, i in zip(a.ctl, ids):
        while not os.path.exists(f"{c}/out_{i}.pkl"):
            time.sleep(.2)
        res.append(pickle.load(open(f"{c}/out_{i}.pkl", "rb"))); os.remove(f"{c}/out_{i}.pkl")
        if not res[-1]["ok"]:
            raise RuntimeError(res[-1]["error"])
    return res


rows = rows_of(a.rows); k = len(a.ctl); t0 = time.time()
ask([dict(op="reload", path=M) for _ in a.ctl]); t1 = time.time()
seeds = crn_seeds(rows)      # the run's crn seeds (--crn 1 --seed 101, CRN_GEN 0; 11-08), were 101 + position
res = ask([dict(op="generate", prompts=[r["prompt"] for r in rows[i::k]], image_paths=[r["image_paths"] for r in rows[i::k]], n=1, sampling=EVAL_SAMPLING, logprobs=None, seeds=seeds[i::k], max_tokens=a.budget, effort=a.effort) for i in range(k)])
outs = [None] * len(rows); tok = 0
for i, x in enumerate(res):
    assert x.get("used") == dict(max_tokens=a.budget, effort=a.effort), f"engine {a.ctl[i]} used {x.get('used')}"
    assert x.get("engine") == RUN_ENGINE, f"engine {a.ctl[i]} is {x.get('engine')}, a measurement needs the run engine {RUN_ENGINE} (bl_rollout.py --num-seqs 64, no --prefix-cache, no --max-batched)"
    outs[i::k] = x["out"]; tok += x["stat"]["generated_tokens"]
recs = []
for r, o in zip(rows, outs):
    s = o["samples"][0]; ok, p = grade_rec(r, s)
    recs.append(dict(id=r["id"], bench=r["bench"], ok=bool(ok), pred=None if p is None else str(p), answer=r["answer"], tokens=s["num_tokens"], finish=s["finish_reason"], reasoning_head=(s.get("reasoning") or "")[:400]))
os.makedirs(RES, exist_ok=True)
with open(out_f, "w") as f:
    for x in recs:
        f.write(json.dumps(x, ensure_ascii=False) + "\n")
st = dict(event=a.set, tag=a.tag, model=M, rows_files=a.rows, owner_go=a.owner_go, budget=a.budget, effort=a.effort, engine_used=res[0].get("used"), engine=res[0].get("engine"), seeds=dict(crn=1, seed=CRN_SEED, gen=CRN_GEN), grader=GRADER, grade_errors=dict(GRADE_ERRORS), engines=k, s_reload=round(t1 - t0, 1), gen=dict(seconds=round(time.time() - t1, 1), tokens=tok), **score_table(recs))
open(f"{RES}/metrics_final.jsonl", "a").write(json.dumps(st) + "\n")
print(json.dumps({x: st[x] for x in st if x not in ("rows_files",)}))
