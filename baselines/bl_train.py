"""Baselines on stock Qwen3.8-27B: GRPO, PPO, SFT. One trainer, one model placement (bl_net), one rollout/eval engine (bl_rollout: the main system's Surgeon), one grader (the main system's).
Run under torch.distributed.run, one rank per training card. Rollout servers run separately on their own cards and are driven through --ctl directories."""
import argparse, gc, glob, json, math, os, pickle, random, threading, time, uuid
import numpy as np
import torch, torch.distributed as dist
from bl_common import *
from bl_net import *

ap = argparse.ArgumentParser()
ap.add_argument("--method", required=True, choices=["grpo", "ppo", "sft"])
ap.add_argument("--name", required=True)
ap.add_argument("--ctl", nargs="+", required=True)
ap.add_argument("--seed", type=int, default=101)
ap.add_argument("--iters", type=int, default=0)                  # rollout iterations (grpo, ppo) or updates (sft; 0 = from --epochs)
ap.add_argument("--q", type=int, default=128)                    # prompts per rollout
ap.add_argument("--g", type=int, default=8)                      # samples per prompt
ap.add_argument("--mb", type=int, default=4)                     # policy updates per rollout
ap.add_argument("--lr", type=float, default=1e-6)
ap.add_argument("--lr-critic", type=float, default=5e-6)
ap.add_argument("--warm", type=int, default=20)                  # linear warm-up, in updates
ap.add_argument("--clip", type=float, default=.2)
ap.add_argument("--vclip", type=float, default=.5)
ap.add_argument("--critic-mb", type=int, default=12)
ap.add_argument("--vwarm", type=int, default=0)                  # critic-only rollouts before the first policy update
ap.add_argument("--critic-offload", type=int, default=1)
ap.add_argument("--actor-offload", type=int, default=0)         # FSDP2 CPU offload for the actor as well (weights and grads in RAM, one block at a time on the card)
ap.add_argument("--colocate", type=int, default=0)              # engines and trainer share the cards: engines sleep during training (level 2), weights reload on wake
ap.add_argument("--freeze-vision", type=int, default=0)
ap.add_argument("--max-tok", type=int, default=16000)            # padded tokens per micro-batch per rank
ap.add_argument("--val-every", type=int, default=0)
ap.add_argument("--val-rows", type=int, default=0)               # 0 = all validation rows
ap.add_argument("--ckpt-at", type=int, nargs="*", default=[])
ap.add_argument("--ckpt-every", type=int, default=0)             # every so many iterations (grpo, ppo) or updates (sft)
ap.add_argument("--ckpt-adam", type=int, default=1)              # 0: weights only (PPO: full state does not fit next to the other users' RAM)
ap.add_argument("--resume", default="")
ap.add_argument("--rewarm", type=int, default=0)                # after a weights-only resume (Adam moments reset): linear LR ramp over this many updates
ap.add_argument("--stop-after-resume-check", type=int, default=0)
ap.add_argument("--shm-floor", type=float, default=SHM_FLOOR_GB)
ap.add_argument("--focus", default="")                           # data/focus.json: half-and-half sampling mix and the report-only validation view
ap.add_argument("--focus-share", type=float, default=0.5)
ap.add_argument("--reuse-stock", default="")                     # results directory that already holds this wrapper's val_stock.jsonl
ap.add_argument("--traces-extra", nargs="*", default=[])         # sft: more trace files, read after --traces
ap.add_argument("--traces", default="")                          # sft: rejection-sampled traces
ap.add_argument("--replay", default="")                          # bench only: samples from a saved rollout file, no servers
ap.add_argument("--bench-only", type=int, default=0)             # no publish, no eval
ap.add_argument("--replay-groups", type=int, default=0)          # keep only this many groups that have a nonzero advantage
ap.add_argument("--budget", type=int, default=BUDGET, choices=BUDGETS)   # answer budget (max_tokens) for rollouts and evaluations, the main system's
ap.add_argument("--effort", default=EFFORT)
ap.add_argument("--order", default="rr", choices=["rr", "lpt"])   # rr: rows[i::k] per engine; lpt: longest expected answers first, engines balanced by expected tokens (throughput only)
ap.add_argument("--cap-strict", type=int, default=0)              # 1: do not start an iteration (or SFT stretch to the next evaluation) that the last one's duration says would end past --max-hours
ap.add_argument("--max-hours", type=float, default=0)             # owner's budget: stop at the next boundary after this many hours; the best weights by validation are the result
ap.add_argument("--val-first", type=int, default=0)              # evaluate the stock weights first
ap.add_argument("--keep-best", type=int, default=1)             # 0: track the best validation macro for the report, keep no best_model copy (data128: no held-out selection set; saves 52 GB RAM)
ap.add_argument("--final-iid", type=int, default=0)             # 1: after the final evaluation, the IID evaluation rows (BL_DATA/iid_eval.node.jsonl), read only here
ap.add_argument("--patience", type=int, default=0)               # stop after this many evaluations without a better validation macro; 0 = never
ap.add_argument("--epochs", type=float, default=1.0)             # sft: natural passes over the traces
ap.add_argument("--batch", type=int, default=32)                 # sft: samples per update
ap.add_argument("--total-updates", type=int, default=0)          # sft: cosine horizon
a = ap.parse_args()
_ov = f"{OUT}/overrides/{a.name}.json"      # settings changed after a job's launcher already started (SFT's trainer starts hours after its traces)
if os.path.exists(_ov):
    vars(a).update(json.load(open(_ov)))
assert not a.colocate or (a.actor_offload and a.critic_offload), "colocated mode needs the trainer off the cards while engines run"

R, W, mesh = setup_dist()
torch.manual_seed(a.seed + R)
RUN = f"{SHM}/{a.name}"; RES = f"{OUT}/results/{a.name}"
if R == 0:
    os.makedirs(RUN, exist_ok=True); os.makedirs(RES, exist_ok=True)
    for f in glob.glob(f"{RUN}/.sync_*"):
        os.remove(f)
dist.barrier()
log = (lambda *x: say(*x)) if R == 0 else (lambda *x: None)
metrics = open(f"{RES}/metrics.jsonl", "a") if R == 0 else None


def rec(**k):
    if R == 0:
        metrics.write(json.dumps(dict(t=round(time.time(), 1), **k)) + "\n"); metrics.flush()
        say(json.dumps(k)[:1500])


guard = Guard(shm_floor=a.shm_floor)
stop_flag = [None]
if R == 0:
    def watch():
        while stop_flag[0] is None:
            stop_flag[0] = guard.stop()
            time.sleep(60)
    threading.Thread(target=watch, daemon=True).start()


_sync = [0]


def rank0_done():
    """Ranks other than 0 wait on a file, not in a collective. A rank waiting in an NCCL collective holds its card at 100 % and slowed the engine sharing that card 2.3 times in the colocated smoke test (card of rank 0: 148 tok/s, others: 55-72)."""
    _sync[0] += 1
    p = f"{RUN}/.sync_{_sync[0]}"
    if R == 0:
        open(p, "w").close()
        if os.path.exists(f"{RUN}/.sync_{_sync[0] - 4}"):      # not the previous one: rank 0 can pass a broadcast before rank 1 has seen that file (sft1 hung 14 min at u70, 11-08)
            os.remove(f"{RUN}/.sync_{_sync[0] - 4}")
    else:
        while not os.path.exists(p):
            time.sleep(.2)


T0 = time.time()


_LAST = [None, 0.0]     # time of the previous check, longest step seen


def over_budget():
    """Checked only between iterations or updates, so the last weights still get their export and evaluation.
    --cap-strict 1: also stop when the longest step so far (iteration or update, evaluations included) would end past the cap."""
    now = time.time()
    if _LAST[0] is not None:
        _LAST[1] = max(_LAST[1], now - _LAST[0])
    _LAST[0] = now
    left = a.max_hours * 3600 - (now - T0)
    o = [bool(a.max_hours and (left < 0 or (a.cap_strict and left < _LAST[1])))] if R == 0 else [None]
    dist.broadcast_object_list(o, 0)
    if o[0]:
        rec(event="budget reached", hours=a.max_hours, updates=u, elapsed_h=round((time.time() - T0) / 3600, 3), longest_step_s=round(_LAST[1], 1), strict=a.cap_strict)
    return o[0]


def should_stop(need=0):
    """All ranks leave at the same iteration boundary."""
    o = [None]
    if R == 0:
        o = [stop_flag[0] or guard.stop(need)]
    dist.broadcast_object_list(o, 0)
    if o[0]:
        rec(event="stop", why=o[0], state=guard.state() if R == 0 else None)
    return o[0]


# ---------------------------------------------------------------- rollout servers
def ask(ops):
    """ops: one command per server. Waits for all. Rank 0 only."""
    ids = []
    for c, op in zip(a.ctl, ops):
        i = f"{time.time():.3f}-{uuid.uuid4().hex[:6]}"
        json.dump(op, open(f"{c}/.c{i}", "w")); os.rename(f"{c}/.c{i}", f"{c}/cmd_{i}.json"); ids.append(i)
    out = []
    for c, i in zip(a.ctl, ids):
        while not os.path.exists(f"{c}/out_{i}.pkl"):
            time.sleep(.2)
        out.append(pickle.load(open(f"{c}/out_{i}.pkl", "rb")))
        os.remove(f"{c}/out_{i}.pkl")
        if not out[-1]["ok"]:
            raise RuntimeError(out[-1]["error"])
    return out


LEN, BLEN = {}, {}


def split_rows(rows, k):
    """Row indices per engine. rr: every k-th row (the old split). lpt: longest expected first (last seen length of the row, else its bench's),
    each to the engine with the least expected work, so no engine ends long after the others and long answers start first."""
    if a.order == "rr":
        return [list(range(i, len(rows), k)) for i in range(k)]
    est = [LEN.get(r["id"], BLEN.get(r["bench"], 4000.0)) for r in rows]
    parts, load = [[] for _ in range(k)], [0.0] * k
    for j in sorted(range(len(rows)), key=lambda j: -est[j]):
        e = min(range(k), key=lambda e: load[e]); parts[e].append(j); load[e] += est[j]
    return parts


def generate(rows, n, sampling, logprobs=None, seeds=None):
    """Split rows across the servers; results in row order. Rank 0 only."""
    k = len(a.ctl)
    parts = split_rows(rows, k)
    ops = [dict(op="generate", prompts=[rows[j]["prompt"] for j in ix], image_paths=[rows[j]["image_paths"] for j in ix], n=n, sampling=sampling, logprobs=logprobs, max_tokens=a.budget, effort=a.effort,
                seeds=None if seeds is None else [seeds[j] for j in ix]) for ix in parts]
    t0 = time.time()
    res = ask(ops)
    out = [None] * len(rows)
    tok = 0
    for i, x in enumerate(res):
        assert x.get("used") == dict(max_tokens=a.budget, effort=a.effort), f"engine {a.ctl[i]} used {x.get('used')}, want budget {a.budget} effort {a.effort}"
        for j, o in zip(parts[i], x["out"]):
            out[j] = o
        tok += x["stat"]["generated_tokens"]
    for r, o in zip(rows, out):      # expected answer length per row and per bench, for --order lpt
        m = float(np.mean([s["num_tokens"] for s in o["samples"]]))
        LEN[r["id"]] = m; BLEN[r["bench"]] = .8 * BLEN.get(r["bench"], m) + .2 * m
    return out, dict(seconds=round(time.time() - t0, 1), tokens=tok, tok_per_s=round(tok / max(1e-9, time.time() - t0)), mem=res[0].get("mem"))


def rollout(rows):
    """On-policy samples with the training sampler (T 1, no top-p/top-k) and logprobs. Rank 0 makes them; broadcast to all ranks."""
    samples, st = [], {}
    if a.replay:
        recs = json.load(open(a.replay))
        ids = {}
        for x in recs:
            g = ids.setdefault(x["id"], len(ids))
            samples.append(dict(id=x["id"], bench=x["bench"], group=g, reward=float(x["ok"]), finish=x["finish"], prompt_token_ids=x["prompt_token_ids"], token_ids=x["token_ids"], logprobs=x["logprobs"], image_paths=x["image_paths"]))
        if a.replay_groups:
            keep = [g for g in sorted({s["group"] for s in samples}) if len({s["reward"] for s in samples if s["group"] == g}) > 1][:a.replay_groups]
            samples = [s for s in samples if s["group"] in keep]
        return samples, {}
    if R == 0:
        out, st = generate(rows, a.g, ROLLOUT_SAMPLING, logprobs=0)
        for gi, (r, o) in enumerate(zip(rows, out)):
            for s in o["samples"]:
                ok, p = grade_rec(r, s)
                samples.append(dict(id=r["id"], bench=r["bench"], group=gi, reward=float(bool(ok)), pred=None if p is None else str(p), finish=s["finish_reason"], prompt_token_ids=o["prompt_token_ids"], token_ids=s["token_ids"],
                                    logprobs=s["logprobs"], image_paths=r["image_paths"], tail=(s.get("content") or "")[-160:], head=(s.get("reasoning") or "")[:200]))
    rank0_done()
    o = [samples, st]
    dist.broadcast_object_list(o, 0)
    return o


def val_eval(tag, rows, name="val"):
    """Main system's sampler, budget, effort and grader. Fixed seeds per row. Answers go to the results directory."""
    stat = {}
    if R == 0:
        out, st = generate(rows, 1, EVAL_SAMPLING, seeds=crn_seeds(rows))   # the run's crn seeds (11-08), were 101 + position
        recs = []
        for r, o in zip(rows, out):
            s = o["samples"][0]; ok, p = grade_rec(r, s)
            recs.append(dict(id=r["id"], bench=r["bench"], ok=bool(ok), pred=None if p is None else str(p), answer=r["answer"], tokens=s["num_tokens"], finish=s["finish_reason"], reasoning_head=(s.get("reasoning") or "")[:400], content=(s.get("content") or "")[-400:]))
        with open(f"{RES}/{name}_{tag}.jsonl", "w") as f:
            for x in recs:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        stat = dict(score_table(recs), gen=st, tok_mean=round(np.mean([x["tokens"] for x in recs])), truncated=round(np.mean([x["finish"] != "stop" for x in recs]), 4))
        rec(event=name, tag=tag, measurement="diagnostic: tuned training engine, not compared (11-08)", seeds=dict(crn=1, seed=CRN_SEED, gen=CRN_GEN), grade_errors=dict(GRADE_ERRORS), **stat)
        if name == "val" and FOCUS_VAL:      # report only: never used for best weights or stopping
            sub = [x for x in recs if x["id"] in FOCUS_VAL]
            rec(event="val_focus", tag=tag, rows=len(sub), right=sum(x["ok"] for x in sub), **score_table(sub))
    rank0_done()
    o = [stat]
    dist.broadcast_object_list(o, 0)
    return o[0]


best = dict(macro=-1.0, tag=None, stale=0)
FOCUS = json.load(open(a.focus)) if a.focus else None
FOCUS_IDS, FOCUS_VAL = (set(FOCUS["train_ids"]), set(FOCUS["val_ids"])) if FOCUS else (set(), set())


def evaluate(tag):
    """Validation files (early stopping) and the main system's select rows (reported). The best weights by validation macro stay as RUN/best_model. Returns True when patience has run out."""
    v = val_eval(tag, val, "val")
    if {r["id"] for r in select} != {r["id"] for r in val}:      # data128 default: val is the select rows, one evaluation
        val_eval(tag, select, "select")
    if v["macro"] > best["macro"]:
        best.update(macro=v["macro"], tag=tag, stale=0)
        if R == 0 and tag != "stock" and a.keep_best:
            for d in (f"{RUN}/best_model.tmp", f"{RUN}/best_model"):
                if os.path.exists(d):
                    shutil.rmtree(d)
            shutil.copytree(f"{RUN}/rollout_model", f"{RUN}/best_model.tmp", copy_function=os.link); os.rename(f"{RUN}/best_model.tmp", f"{RUN}/best_model")   # hard links: no second copy until the next export replaces rollout_model
    else:
        best["stale"] += 1
    rec(event="best", **best)
    rank0_done()
    dist.barrier()
    return bool(a.patience and best["stale"] >= a.patience)


# ---------------------------------------------------------------- training steps
def wholes(n_tokens_each, values):
    return torch.cat([torch.full((k,), float(v)) for k, v in zip(n_tokens_each, values)])


def plan(samples):
    rs = rounds(samples, W, a.max_tok)
    small = min(range(len(samples)), key=lambda i: len(samples[i]["prompt_token_ids"]) + len(samples[i]["token_ids"]))
    return rs, small


def policy_update(samples, lr, old_key="logprobs"):
    """One optimizer step on the clip objective, token mean over the kept samples. Per-token advantage in s["adv"] (scalar or list)."""
    t0 = time.time()
    rs, small = plan(samples)
    T = sum(len(s["token_ids"]) for s in samples)
    agg = torch.zeros(9, device="cuda", dtype=torch.float64)
    peak = 0
    for rd in rs:
        idx = rd[R]; wt = 1.0
        if not idx:
            idx, wt = [small], 0.0
        b = [samples[i] for i in idx]
        batch = collate(b)
        lp = net(**batch)
        old = logp_targets(b, old_key)
        adv = torch.cat([(torch.as_tensor(s["adv"], dtype=torch.float32) if isinstance(s["adv"], list) else torch.full((len(s["token_ids"]),), float(s["adv"]))) for s in b]).to("cuda")
        lr_ = lp - old
        ratio = lr_.exp()
        l = -torch.min(ratio * adv, ratio.clamp(1 - a.clip, 1 + a.clip) * adv)
        loss = l.sum() / T * W * wt
        loss.backward()
        with torch.no_grad():
            agg += torch.stack([l.sum().double() * wt, torch.tensor(float(lp.numel()) * wt, device="cuda", dtype=torch.float64), lr_.abs().sum().double() * wt, (lr_.abs() > math.log(1 + a.clip)).sum().double() * wt,
                                lr_.sum().double() * wt, lp.sum().double() * wt, ratio.max().double(), torch.tensor(float(batch["input_ids"].numel()) * wt, device="cuda", dtype=torch.float64), torch.tensor(1.0, device="cuda", dtype=torch.float64)])
        peak = max(peak, torch.cuda.max_memory_allocated())
        del batch, lp, old, adv, lr_, ratio, l, loss
    dist.all_reduce(agg)
    t1 = time.time()
    gn = opt.step(lr)
    n = max(1.0, float(agg[1]))
    return dict(loss=float(agg[0]) / n, tokens=int(agg[1]), abs_logratio=float(agg[2]) / n, clipfrac=float(agg[3]) / n, mean_logratio=float(agg[4]) / n, mean_logp=float(agg[5]) / n, grad_norm=gn, lr=lr,
                micro=len(rs), padded_tokens=int(agg[7]), s_fwdbwd=round(t1 - t0, 1), s_opt=round(time.time() - t1, 1), peak_gb=round(peak / 2**30, 1))


@torch.no_grad()
def forward_values(samples):
    """Critic values at every response token, for all samples, on all ranks."""
    rs, small = plan(samples)
    got = {}
    for rd in rs:
        idx = rd[R] or [small]
        b = [samples[i] for i in idx]
        v = critic(**collate(b)).cpu()
        k = 0
        for i, s in zip(idx, b):
            n = len(s["token_ids"])
            got[i] = v[k:k + n].tolist(); k += n
    critic.reshard() if hasattr(critic, "reshard") else None
    allg = [None] * W
    dist.all_gather_object(allg, got)
    for g in allg:
        for i, v in g.items():
            samples[i]["vold"] = v


def critic_update(samples, lr):
    t0 = time.time()
    rs, small = plan(samples)
    T = sum(len(s["token_ids"]) for s in samples)
    agg = torch.zeros(4, device="cuda", dtype=torch.float64)
    for rd in rs:
        idx = rd[R]; wt = 1.0
        if not idx:
            idx, wt = [small], 0.0
        b = [samples[i] for i in idx]
        v = critic(**collate(b))
        tgt = torch.cat([torch.full((len(s["token_ids"]),), s["reward"]) for s in b]).to("cuda")
        vo = logp_targets(b, "vold")
        l = .5 * torch.max((v - tgt) ** 2, (vo + (v - vo).clamp(-a.vclip, a.vclip) - tgt) ** 2)
        (l.sum() / T * W * wt).backward()
        with torch.no_grad():
            agg += torch.stack([l.sum().double() * wt, torch.tensor(float(v.numel()) * wt, device="cuda", dtype=torch.float64), v.sum().double() * wt, ((v - tgt) ** 2).sum().double() * wt])
        del v, tgt, vo, l
    dist.all_reduce(agg)
    t1 = time.time()
    gn = copt.step(lr)
    n = max(1.0, float(agg[1]))
    return dict(vloss=float(agg[0]) / n, vmean=float(agg[2]) / n, vmse=float(agg[3]) / n, grad_norm=gn, s_fwdbwd=round(t1 - t0, 1), s_opt=round(time.time() - t1, 1), peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 1))


@torch.no_grad()
def probe(samples):
    """Fixed-batch forward sum, for the resume check."""
    rs, small = plan(samples)
    s = torch.zeros((), device="cuda", dtype=torch.float64)
    for rd in rs:
        idx = rd[R] or [small]
        s += net(**collate([samples[i] for i in idx])).double().sum()
    net.reshard()     # FSDP reshards the root group only after a backward; without one its parameters stay whole
    dist.all_reduce(s)
    return float(s)


fixed = []


@torch.no_grad()
def fixed_batch(tag):
    """Learning signal on one fixed set of kept rollout samples from the first iteration: the clip objective at ratio from the saved logprobs, and mean log prob of right and of wrong samples. Forward only."""
    rs, small = plan(fixed)
    agg = torch.zeros(5, device="cuda", dtype=torch.float64)
    for rd in rs:
        idx = rd[R]; wt = 1.0
        if not idx:
            idx, wt = [small], 0.0
        b = [fixed[i] for i in idx]
        lp = net(**collate(b))
        old = logp_targets(b, "logprobs")
        adv = torch.cat([torch.full((len(s["token_ids"]),), float(s["adv"])) for s in b]).to("cuda")
        rw = torch.cat([torch.full((len(s["token_ids"]),), float(s["reward"])) for s in b]).to("cuda")
        agg += torch.stack([((lp - old).exp() * adv).sum().double() * wt, torch.tensor(float(lp.numel()) * wt, device="cuda", dtype=torch.float64), (lp * rw).sum().double() * wt, rw.sum().double() * wt, lp.sum().double() * wt])
    net.reshard()
    dist.all_reduce(agg)
    a0, n, rl, rn, al = [float(x) for x in agg]
    rec(event="fixed batch", tag=tag, surrogate=a0 / max(1, n), logp_right=rl / max(1, rn), logp_wrong=(al - rl) / max(1, n - rn), tokens=int(n))


U_RES = [None]      # first update after a resume


def rewarm(u):
    if not a.rewarm or U_RES[0] is None:
        return 1.0
    return min(1.0, (u - U_RES[0] + 1) / a.rewarm)


def lr_at(u, base):
    return base * min(1.0, (u + 1) / max(1, a.warm)) * rewarm(u)


def lr_cos(u, base):
    if u < a.warm:
        return base * (u + 1) / a.warm * rewarm(u)
    H = max(a.warm + 1, a.total_updates or a.iters)
    return rewarm(u) * base * .5 * (1 + math.cos(math.pi * min(1.0, (u - a.warm) / (H - a.warm))))


def rss_gb():
    import resource
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 1)


_eng = ["wake"]


def engines(op):
    """Colocated mode: put the engines to sleep before training, wake them after. The trainer frees its cached GPU memory first."""
    if not a.colocate or _eng[0] == op:
        return
    _eng[0] = op
    torch.cuda.empty_cache()
    dist.barrier()
    t0 = time.time()
    if R == 0:
        ask([dict(op=op) for _ in a.ctl])
        rec(event=op, s=round(time.time() - t0, 1), smi=smi())
    rank0_done()
    dist.barrier()


def publish(tag):
    """Current weights to the rollout servers. Lands in RUN/rollout_model (replaced each time)."""
    st = {}
    # export_bf16 deletes the previous export before writing, so it needs new room only when best_model shares those files (hard links):
    # then the old copy stays. A fixed 60 GB stopped grpo1 at 140 GB available, where the export needed nothing (11-08 13:34).
    d = f"{RUN}/rollout_model"
    shared = os.path.isdir(d) and any(os.stat(f"{d}/{f}").st_nlink > 1 for f in os.listdir(d))
    need = 60 if shared or not os.path.isdir(d) else 0
    if should_stop(need):
        return None
    engines("sleep")      # the export gathers whole tensors on the card; a card with an awake engine has no room for that (smoke test: 80.3 of 81.6 GB)
    t0 = time.time()
    export_bf16(net, f"{RUN}/rollout_model", R, log)
    t1 = time.time()
    engines("wake")
    if R == 0:
        ask([dict(op="reload", path=f"{RUN}/rollout_model") for _ in a.ctl])
        st = dict(s_export=round(t1 - t0, 1), s_reload=round(time.time() - t1, 1))
        rec(event="publish", tag=tag, **st)
    rank0_done()
    dist.barrier()
    return st or True


def checkpoint_now(tag, meta):
    """One resume slot, replaced safely: write new, verify, swap, then remove the old (bl_net.save_ckpt_safe). The space check asks for the new
    checkpoint's real size, since the old one stays until the new one is verified. No room: skip this checkpoint, keep the old one, keep training."""
    nets = dict(actor=net); opts = dict(actor=opt)
    if a.method == "ppo":
        nets["critic"] = critic; opts["critic"] = copt
    x = save_ckpt_safe(nets, opts, f"{RUN}/ckpt", meta, R, log, adam=bool(a.ckpt_adam), room=lambda gb: guard.stop(gb))
    rec(event="ckpt write" if x["ok"] else "ckpt skipped", tag=tag, **x, state=guard.state() if R == 0 else None)
    return x["ok"]


# ---------------------------------------------------------------- main
torch.cuda.reset_peak_memory_stats()
rec(event="start", args=vars(a), world=W, grader=GRADER, disk=guard.state() if R == 0 else None)
if a.colocate:      # building needs room on the card: 9.7 GB plus one 2.4 GB tensor at two ranks did not fit beside an awake engine. Level 2: level 1 kept 73 GB more per engine in RAM for good.
    if R == 0:
        ask([dict(op="sleep", level=2) for _ in a.ctl])
    rank0_done()
    dist.barrier()
    _eng[0] = "sleep"
net = build(mesh, offload=bool(a.actor_offload), freeze_vision=bool(a.freeze_vision), log=log)
gc.collect(); torch.cuda.empty_cache()     # build's own locals hold 4.8 GB until it returns; an engine sharing the card needs that room
rec(event="built", gpu_alloc_gb=round(torch.cuda.memory_allocated() / 2**30, 1), smi=smi() if R == 0 else None)
opt = OffloadAdamW(net.parameters())
critic = copt = None
if a.method == "ppo":
    critic = build(mesh, critic=True, offload=bool(a.critic_offload), freeze_vision=bool(a.freeze_vision), log=log)
    copt = OffloadAdamW(critic.parameters(), betas=(0.9, 0.95))
    gc.collect(); torch.cuda.empty_cache()
    rec(event="critic built", gpu_alloc_gb=round(torch.cuda.memory_allocated() / 2**30, 1))

train = rows_of(TRAIN_FILES)
val = rows_of(VAL_FILES)
random.Random(5).shuffle(val)
select = rows_of([PAPER_SELECT])
if a.val_rows:
    val = val[:a.val_rows]
gen = focus_sampler(train, FOCUS_IDS, a.seed, a.focus_share) if FOCUS_IDS else sqrt_sampler(train, a.seed)
rec(event="sampling", seed_whole=a.seed, seed_focus=a.seed + 1 if FOCUS_IDS else None, focus_share=a.focus_share if FOCUS_IDS else 0, focus_rows=sum(r["id"] in FOCUS_IDS for r in train), rows=len(train), by_bench=dict(collections.Counter(r["bench"] for r in train if r["id"] in FOCUS_IDS)), bench_q=.5, val_focus_rows=len(FOCUS_VAL))
u0, it0, u, cu = 0, 0, 0, 0
if a.colocate and not a.resume and (a.method != "sft" or a.val_first):
    engines("wake")      # level 2 dropped the weights
    if R == 0:
        t0 = time.time()
        ask([dict(op="reload", path=STOCK) for _ in a.ctl])
        rec(event="stock reloaded", s=round(time.time() - t0, 1))
    rank0_done()
    dist.barrier()
if a.val_first and not a.resume:
    evaluate("stock")
if a.reuse_stock and not a.resume:      # same stock weights, wrapper, rows and seeds: the file from the run this one replaced
    _r = [json.loads(l) for l in open(f"{a.reuse_stock}/val_stock.jsonl")]
    assert {x["id"] for x in _r} == {x["id"] for x in val}, "val_stock.jsonl is not these rows"
    _s = score_table(_r); best.update(macro=_s["macro"], tag="stock", stale=0)
    _f = [x for x in _r if x["id"] in FOCUS_VAL]
    rec(event="stock reused", file=f"{a.reuse_stock}/val_stock.jsonl", macro=_s["macro"], rows=len(_r), val_focus=dict(rows=len(_f), right=sum(x["ok"] for x in _f), **score_table(_f)) if _f else None)
    rec(event="best", **best)
if a.resume:
    engines("sleep")
    meta = load_ckpt(dict(actor=net, **({"critic": critic} if critic else {})), dict(actor=opt, **({"critic": copt} if copt else {})), a.resume, R, log)
    cs = checksum(net)
    psum = probe(meta["probe_samples"]) if "probe_samples" in meta else None
    rec(event="resume check", checksum_saved=meta.get("checksum"), checksum_loaded=cs, checksum_equal=meta.get("checksum") == cs, probe_saved=meta.get("probe"), probe_loaded=psum, probe_diff=None if psum is None else abs(psum - meta["probe"]), adam_t=opt.t)
    u, it0 = meta["updates"], meta["iter"] + 1
    U_RES[0] = u
    if os.path.exists(f"{RUN}/fixed_batch.pkl"):
        fixed = pickle.load(open(f"{RUN}/fixed_batch.pkl", "rb"))
    rec(event="resumed", updates=u, next_iter=it0, rewarm=a.rewarm, fixed_batch_kept=bool(fixed), patience=a.patience, max_hours=a.max_hours)
    for _ in range(meta.get("rows_used", 0)):
        next(gen)
    if a.stop_after_resume_check:
        dist.barrier(); rec(event="end"); dist.destroy_process_group(); raise SystemExit(0)
    if a.colocate and a.method != "sft":
        publish("resume")                    # engines start on stock weights
rows_used = meta.get("rows_used", 0) if a.resume else 0

if a.method in ("grpo", "ppo"):
    for it in range(it0, a.iters):
        if should_stop() or over_budget():
            break
        t0 = time.time()
        rows = [next(gen) for _ in range(a.q)]; rows_used += a.q
        samples, st = rollout(rows)
        engines("sleep")
        t1 = time.time()
        rew = np.array([s["reward"] for s in samples]); toks = np.array([len(s["token_ids"]) for s in samples])
        by = collections.defaultdict(list)
        for s in samples:
            by[s["bench"]].append(s["reward"])
        rec(event="rollout", it=it, q=len(rows), samples=len(samples), focus_q=sum(r["id"] in FOCUS_IDS for r in rows), reward=float(rew.mean()), per_bench={b: round(float(np.mean(v)), 3) for b, v in sorted(by.items())}, out_tok_mean=float(toks.mean()), out_tok_max=int(toks.max()), truncated=float(np.mean([s["finish"] != "stop" for s in samples])), prompt_tok_mean=float(np.mean([len(s["prompt_token_ids"]) for s in samples])), gen=st, s_rollout=round(t1 - t0, 1))
        if a.method == "grpo":
            kept = []
            for gi in sorted({s["group"] for s in samples}):
                g = [s for s in samples if s["group"] == gi]
                rs = np.array([s["reward"] for s in g])
                if rs.std() == 0:             # zero advantage in the whole group: no gradient without a KL term (DAPO Sec 3.2, MM-Eureka Sec 4)
                    continue
                std = rs.std(ddof=1)
                for s in g:
                    s["adv"] = float((s["reward"] - rs.mean()) / (std + 1e-6))
                kept.append(g)
            gs = collections.defaultdict(collections.Counter); gt = collections.Counter()
            for gi in sorted({s["group"] for s in samples}):
                g = [s for s in samples if s["group"] == gi]
                c = sum(s["reward"] for s in g)
                kind = "all_right" if c == len(g) else "all_wrong" if c == 0 else "mixed"
                gs[g[0]["bench"]][kind] += 1; gt[kind] += sum(len(s["token_ids"]) for s in g); gt[g[0]["bench"] + "_kept_tok"] += sum(len(s["token_ids"]) for s in g) if kind == "mixed" else 0
            rec(event="groups", kept=len(kept), of=len(rows), by_bench={b: dict(c) for b, c in sorted(gs.items())}, tokens=dict(gt))
            if R == 0:
                with open(f"{RES}/samples_it{it}.jsonl", "w") as f:     # a few real outputs from every rollout, for reading
                    for gi in sorted({s["group"] for s in samples})[:8]:
                        for s in [s for s in samples if s["group"] == gi][:2]:
                            f.write(json.dumps({k: s.get(k) for k in ("id", "bench", "reward", "pred", "finish", "head", "tail")}, ensure_ascii=False) + "\n")
            r_ = random.Random(a.seed * 1000 + it); r_.shuffle(kept)
            mbs = [[s for g in kept[i::a.mb] for s in g] for i in range(a.mb)] if kept else []
        else:
            forward_values(samples)
            rec(event="values", s=round(time.time() - t1, 1), vmean=float(np.mean([np.mean(s["vold"]) for s in samples])), corr=float(np.corrcoef([np.mean(s["vold"]) for s in samples], [s["reward"] for s in samples])[0, 1]) if rew.std() else None)
            if it >= a.vwarm:
                adv = np.concatenate([s["reward"] - np.array(s["vold"]) for s in samples])
                mu, sd = adv.mean(), adv.std() + 1e-8                   # batch whitening over tokens (ORZ App. B, verl masked_whiten)
                for s in samples:
                    s["adv"] = ((s["reward"] - np.array(s["vold"]) - mu) / sd).tolist()
            r_ = random.Random(a.seed * 1000 + it); idx = list(range(len(samples))); r_.shuffle(idx)
            mbs = [[samples[i] for i in idx[k::a.mb]] for k in range(a.mb)]
        if it < a.vwarm:
            mbs = []
        if a.method == "grpo" and not fixed and any(mbs):
            fixed = [x for x in next(m for m in mbs if m)][:8]
            if R == 0:
                pickle.dump(fixed, open(f"{RUN}/fixed_batch.pkl.tmp", "wb")); os.rename(f"{RUN}/fixed_batch.pkl.tmp", f"{RUN}/fixed_batch.pkl")
            fixed_batch(f"it{it}_before")
        for mb in mbs:
            if not mb:
                continue
            x = policy_update(mb, lr_at(u, a.lr)); u += 1
            rec(event="update", it=it, u=u, n=len(mb), **x)
        if a.method == "ppo":
            r_ = random.Random(a.seed * 2000 + it); idx = list(range(len(samples))); r_.shuffle(idx)
            for k in range(a.critic_mb):
                mb = [samples[i] for i in idx[k::a.critic_mb]]
                if mb:
                    x = critic_update(mb, lr_at(cu, a.lr_critic)); cu += 1
                    rec(event="critic", it=it, cu=cu, n=len(mb), **x)
        if fixed:
            fixed_batch(f"it{it}")
        if it in a.ckpt_at or (a.ckpt_every and (it + 1) % a.ckpt_every == 0):
            ps = (mbs[0] if any(mbs) else samples)[:4]
            ps = [dict(prompt_token_ids=s["prompt_token_ids"], token_ids=s["token_ids"], image_paths=s["image_paths"]) for s in ps]
            pv, cs = probe(ps), checksum(net)
            ok = checkpoint_now(f"it{it}", dict(iter=it, updates=u, rows_used=rows_used, probe=pv, checksum=cs, probe_samples=ps))
            rec(event="ckpt", it=it, probe=pv, checksum=cs, saved=ok)
        changed = any(mbs) or a.method == "ppo"
        if not a.bench_only and (changed or a.colocate) and not publish(f"it{it}"):   # colocated: engines dropped their weights on sleep, so they always reload
            break
        rec(event="iter", it=it, updates=u, grade_errors=dict(GRADE_ERRORS), s_iter=round(time.time() - t0, 1), smi=smi() if R == 0 else None, state=guard.state() if R == 0 else None, rss_gb=rss_gb())
        if a.val_every and (it + 1) % a.val_every == 0 and evaluate(f"it{it}"):
            rec(event="early stop", **best)
            break
else:
    engines("sleep")
    tr = [json.loads(l) for f_ in [a.traces] + list(a.traces_extra) for l in open(f_)]
    a.iters = a.iters or math.ceil(a.epochs * len(tr) / a.batch)
    a.total_updates = a.total_updates or a.iters
    a.warm = a.warm if a.warm >= 0 else round(.1 * a.iters)      # warm-up .1 (OpenVLThinker App. C Table 13, MMR1 App. C Table 5)
    rec(event="sft plan", traces=len(tr), iters=a.iters, warm=a.warm, by_bench=dict(collections.Counter(x["bench"] for x in tr)))
    tgen = focus_sampler(tr, FOCUS_IDS, a.seed, a.focus_share) if FOCUS_IDS else sqrt_sampler(tr, a.seed)
    _ft = [x for x in tr if x["id"] in FOCUS_IDS]
    rec(event="sft mix", traces=len(tr), distinct_rows=len({x["id"] for x in tr}), focus_traces=len(_ft), focus_rows=len({x["id"] for x in _ft}), focus_share_of_traces=round(len(_ft) / max(1, len(tr)), 4), focus_share_of_draws=a.focus_share if FOCUS_IDS else 0, focus_by_bench=dict(collections.Counter(x["bench"] for x in _ft)), seed_whole=a.seed, seed_focus=a.seed + 1 if FOCUS_IDS else None)
    if a.resume:
        for _ in range(u * a.batch):
            next(tgen)
    while u < a.iters:
        if should_stop() or over_budget():
            break
        t0 = time.time()
        b = [next(tgen) for _ in range(a.batch)]
        rs, small = plan(b)
        T = sum(len(s["token_ids"]) for s in b)
        agg = torch.zeros(3, device="cuda", dtype=torch.float64)
        for rd in rs:
            idx = rd[R]; wt = 1.0
            if not idx:
                idx, wt = [small], 0.0
            mb = [b[i] for i in idx]
            lp = net(**collate(mb))
            (-(lp.sum()) / T * W * wt).backward()
            agg += torch.stack([-lp.detach().sum().double() * wt, torch.tensor(float(lp.numel()) * wt, device="cuda", dtype=torch.float64), torch.tensor(1.0, device="cuda", dtype=torch.float64)])
            del lp
        dist.all_reduce(agg)
        t1 = time.time()
        lr = lr_cos(u, a.lr)
        gn = opt.step(lr); u += 1
        rec(event="update", u=u, n=len(b), nll=float(agg[0]) / max(1, float(agg[1])), tokens=int(agg[1]), grad_norm=gn, lr=lr, micro=len(rs), s_fwdbwd=round(t1 - t0, 1), s_opt=round(time.time() - t1, 1), peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 1))
        if a.val_every and u % a.val_every == 0:
            if publish(f"u{u}"):
                done = evaluate(f"u{u}")
                engines("sleep")
                if done:
                    rec(event="early stop", **best)
                    break
        if u - 1 in a.ckpt_at or (a.ckpt_every and u % a.ckpt_every == 0):
            ps = [dict(prompt_token_ids=s["prompt_token_ids"], token_ids=s["token_ids"], image_paths=s["image_paths"]) for s in b[:4]]
            pv, cs = probe(ps), checksum(net)
            ok = checkpoint_now(f"u{u}", dict(iter=u, updates=u, rows_used=0, probe=pv, checksum=cs, probe_samples=ps))
            rec(event="ckpt", u=u, probe=pv, checksum=cs, saved=ok)

if not a.bench_only and publish("final"):     # for grpo and ppo this repeats the last iteration's weights, which costs one export and reload
    evaluate("final")
    if a.final_iid:        # primary result: final weights at the cap on the IID rows; then the best-on-validation copy, if one is kept, labelled as selected
        iid = rows_of([IID_EVAL])
        rec(event="iid rows", n=len(iid), by_bench=dict(collections.Counter(r["bench"] for r in iid)), shared_ids_with_train=len({r["id"] for r in iid} & {r["id"] for r in train}))
        val_eval("final", iid, "iid")
        if a.keep_best and best["tag"] not in (None, "stock", "final") and os.path.isdir(f"{RUN}/best_model"):
            if R == 0:
                ask([dict(op="reload", path=f"{RUN}/best_model") for _ in a.ctl])
            rank0_done(); dist.barrier()
            val_eval("best_" + best["tag"], iid, "iid")
rec(event="end", peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 1), smi=smi() if R == 0 else None, rss_gb=rss_gb())
dist.barrier()
dist.destroy_process_group()
