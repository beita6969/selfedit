"""SFT traces from the stock model's own output (STaR, ReST-EM), with no outside text. One sample per row per pass at the main system's sampler, correct ones kept. Later passes retry only rows with no correct trace yet.
Rows come in square-root order, so small benches are covered first if the time budget ends a pass. Traces are appended as chunks finish, so a stop loses nothing.
The STaR hint pass was dropped after the smoke test: of 52 correct hinted samples on 13 unsolved rows, 51 mentioned the hint, so it rescued 1 row."""
import argparse, json, os, pickle, time, uuid
from bl_common import *
ap = argparse.ArgumentParser()
ap.add_argument("--ctl", nargs="+", required=True); ap.add_argument("--out", required=True); ap.add_argument("--stat", default="")
ap.add_argument("--hours", type=float, default=7.0); ap.add_argument("--passes", type=int, default=4); ap.add_argument("--chunk", type=int, default=256); ap.add_argument("--rows", type=int, default=0); ap.add_argument("--seed", type=int, default=101)
ap.add_argument("--focus", default=""); ap.add_argument("--main-traces", default=""); ap.add_argument("--attempted", type=int, default=0); ap.add_argument("--inflight", type=int, default=0)   # top-up in select's categories beside a paused main trace process
ap.add_argument("--budget", type=int, default=BUDGET, choices=BUDGETS); ap.add_argument("--effort", default=EFFORT)
ap.add_argument("--per-row", type=int, default=1)   # data128: passes go on until a row has this many distinct correct traces (1 = the old rule: retry unsolved rows only)
a = ap.parse_args()


def put(c, op):
    i = f"{time.time():.3f}-{uuid.uuid4().hex[:6]}"
    json.dump(op, open(f"{c}/.c{i}", "w")); os.rename(f"{c}/.c{i}", f"{c}/cmd_{i}.json")
    return c, i


def got(c, i):
    while not os.path.exists(f"{c}/out_{i}.pkl"):
        time.sleep(.2)
    x = pickle.load(open(f"{c}/out_{i}.pkl", "rb")); os.remove(f"{c}/out_{i}.pkl")
    assert x["ok"], x["error"]
    return x


rows = rows_of(TRAIN_FILES)
g = sqrt_sampler(rows, a.seed); seen, order = set(), []
while len(order) < (a.rows or len(rows)):
    r = next(g)
    if r["id"] not in seen:
        seen.add(r["id"]); order.append(r)
solved0 = set()
if a.focus:      # rows the main process has tried come from its fixed order; rows it has in flight are left to it; untried rows go first, its failures after
    fo = set(json.load(open(a.focus))["train_ids"]); solved0 = {json.loads(l)["id"] for l in open(a.main_traces)}
    tried = {r["id"] for r in order[:a.attempted]}; flying = {r["id"] for r in order[a.attempted - a.inflight:a.attempted]}
    g = sqrt_sampler([r for r in rows if r["id"] in fo], a.seed + 1); seen, fx = set(), []
    while len(fx) < len(fo):
        r = next(g)
        if r["id"] not in seen:
            seen.add(r["id"]); fx.append(r)
    order = [r for r in fx if r["id"] not in tried] + [r for r in fx if r["id"] in tried and r["id"] not in solved0 and r["id"] not in flying]
    say("focus rows", len(fx), "untried", sum(r["id"] not in tried for r in fx), "retry", sum(r["id"] in tried and r["id"] not in solved0 and r["id"] not in flying for r in fx), "in flight", len(flying & fo), "by bench", dict(collections.Counter(r["bench"] for r in order)))
done_all = 0
t0 = time.time(); nk = collections.Counter(); hs = collections.defaultdict(set); solved = set(); kept = 0; tot = collections.Counter(); by = collections.Counter(); f = open(a.out, "a")
for p in range(a.passes):
    todo = [r for r in order if nk[r["id"]] < a.per_row]
    say("pass", p, "rows", len(todo))
    k = a.chunk * len(a.ctl)
    for s in range(0, len(todo), k):
        if time.time() - t0 > a.hours * 3600:
            break
        part = todo[s:s + k]; t1 = time.time()
        parts = [part[i::len(a.ctl)] for i in range(len(a.ctl))]
        reqs = [put(c, dict(op="generate", prompts=[r["prompt"] for r in q], image_paths=[r["image_paths"] for r in q], n=1, max_tokens=a.budget, effort=a.effort, sampling=EVAL_SAMPLING)) for c, q in zip(a.ctl, parts)]
        toks = 0
        for (c, i), q in zip(reqs, parts):
            x = got(c, i); toks += x["stat"]["generated_tokens"]
            assert x.get("used") == dict(max_tokens=a.budget, effort=a.effort), f"engine {c} used {x.get('used')}"
            for r, o in zip(q, x["out"]):
                z = o["samples"][0]; ok = bool(grade_rec(r, z)[0]); tot["samples"] += 1; tot["correct"] += ok; tot["truncated"] += z["finish_reason"] != "stop"; tot["tokens"] += z["num_tokens"]
                if ok and hash(tuple(z["token_ids"])) in hs[r["id"]]:
                    tot["duplicate"] += 1
                if ok and hash(tuple(z["token_ids"])) not in hs[r["id"]]:
                    hs[r["id"]].add(hash(tuple(z["token_ids"]))); nk[r["id"]] += 1
                    solved.add(r["id"]); kept += 1; by[r["bench"]] += 1
                    f.write(json.dumps(dict(id=r["id"], bench=r["bench"], source="self", prompt_token_ids=o["prompt_token_ids"], token_ids=z["token_ids"], image_paths=r["image_paths"], text_head=(z.get("reasoning") or "")[:300], text_tail=(z.get("content") or "")[-200:]), ensure_ascii=False) + "\n")
        f.flush()
        done_all += len(part)
        say(json.dumps(dict(event="traces", grade_errors=dict(GRADE_ERRORS), p=p, done_all=done_all, done=s + len(part), of=len(todo), kept=kept, tok_per_s=round(toks / (time.time() - t1)), s=round(time.time() - t1), hours=round((time.time() - t0) / 3600, 2), **tot, by=dict(by))))
    if time.time() - t0 > a.hours * 3600:
        break
f.close()
say("done kept", kept, "rows solved", len(solved), "of", len(order), dict(tot), dict(by), "hours", round((time.time() - t0) / 3600, 2))
