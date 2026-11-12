"""v4 W worker: in-process Surgeon (code-1109c engine/surgery.py, tree res_driver_1109c_final2) on one card behind the q/in protocol (engine "W").
Owns batches whose requests are all "engine": "W" (the dispatcher skips them). Per batch: requests grouped by canonical edit JSON; per group
apply (vharness.validate of the knob form, scope all, stock genome) -> generate (same content / resize / sampling / CRN seeds / grading as E) -> revert
(bit-exact check). "edit": null = stock weights. Results append to q/out/<batch>.jsonl; <batch>.done when the batch is complete.
usage: python w_worker.py [--card-name W3]"""
import argparse, collections, glob, hashlib, json, os, sys, time, traceback, zlib

TREE = "/workspace/fudan/mingdazhang/selfedit/res_driver_1109c_final2"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(TREE, "engine"), os.path.join(TREE, "rsi"), os.path.join(TREE, "mm")]
from common import SAMPLING, EFFORT, MODEL, content_for, grade, chat_messages

ap = argparse.ArgumentParser()
ap.add_argument("--v4", default="/data/fd/mingdazhang/selfedit/v4")
ap.add_argument("--name", default="W3")
ap.add_argument("--model", default=None, help="base model dir (default common.MODEL = stock); e.g. v5/wckpt/vp2m512 for edit-on-edit stacks")
ap.add_argument("--max-num-seqs", type=int, default=64)
ap.add_argument("--util", type=float, default=0.85)
args = ap.parse_args()
V4 = args.v4
QIN, QOUT, QCAN = f"{V4}/q/in", f"{V4}/q/out", f"{V4}/q/cancel"
READ_MAX, CHAT_MAX, CHAT_TEMP, MAXLEN = 32768, 12000, 0.7, 46000
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
canon = lambda o: json.dumps(o, sort_keys=True, separators=(",", ":"))
esha = lambda e: hashlib.sha256(canon(e).encode()).hexdigest()[:16]


def crn_seed(row, sample):
    return zlib.crc32(f"101|0|{row}".encode()) % (2 ** 31 - 1024) + int(sample)


import vharness as vh
from surgery import Surgeon
from wedit import surgeon_edit


t0 = time.time()
S = Surgeon(args.model or MODEL, tp=1, max_model_len=MAXLEN, max_num_seqs=args.max_num_seqs, max_num_batched_tokens=8192, gpu_memory_utilization=args.util,
            kv_cache_dtype="fp8", multimodal=True, enable_prefix_caching=False)
log(f"W engine up in {time.time() - t0:.0f}s (load {S.load_seconds:.0f}s)")
os.makedirs(f"{V4}/w", exist_ok=True)
json.dump(dict(name=args.name, card=os.environ.get("CUDA_VISIBLE_DEVICES"), t=time.time(), up_s=time.time() - t0, pid=os.getpid()), open(f"{V4}/w/{args.name}.up.json", "w"))

rows, rows_m = {}, 0
done = set()
for f in glob.glob(f"{QOUT}/*.jsonl"):
    for l in open(f):
        try:
            r = json.loads(l)
        except ValueError:
            continue
        if "error" not in r or r.get("cancelled"):
            done.add(r["key"])
log(f"index: {len(done)} keys done")
seen = set()


_ts = [0.0, set()]


def t_served():
    """edit_shas of up T servers (engines.json): their reads go to the dispatcher, never to a W worker."""
    if time.time() - _ts[0] > 3:
        try:
            e = json.load(open(f"{V4}/engines.json"))
            _ts[1] = {x["edit_sha"] for x in e["servers"] if x.get("role") == "T" and x.get("edit_sha") and x.get("state", "up") == "up"}
        except Exception:
            pass
        _ts[0] = time.time()
    return _ts[1]


def cancelled(batch, es):
    """q/cancel/<batch> cancels the whole batch; q/cancel/<batch>@<edit_sha> cancels only that edit's requests in the batch."""
    return os.path.exists(f"{QCAN}/{batch}") or os.path.exists(f"{QCAN}/{batch}@{es}")


def write_cancel(r, es, edit, why):
    write(r["_batch"], dict(key=r["key"], mode=r.get("mode", "read"), error=f"cancelled: {why}", cancelled=True, edit=edit, edit_sha=es, server=args.name, engine_type="offline"))
    done.add(r["key"])


CUR = {}            # running sub-generate: base request counter, keep (index -> request), edit_sha; read by the step wrapper
_eng = S.llm.llm_engine
_step0 = _eng.step
_last_chk = [0.0]


def _step_with_cancel():
    """Every 5 s while a group generates: abort the unfinished requests whose (batch, edit) was cancelled; vLLM then drops them
    and llm.chat returns as soon as nothing else is unfinished (the group's revert follows at once)."""
    got = _step0()
    if CUR and time.time() - _last_chk[0] > 5:
        _last_chk[0] = time.time()
        try:
            handover = CUR["es"] in t_served() and not CUR.get("pinned")
            bad = set(CUR["batches"]) if handover else {b for b in CUR["batches"] if cancelled(b, CUR["es"])}
            if bad:
                ids = []
                for st in list(_eng.output_processor.request_states.values()):
                    rid = getattr(st, "external_req_id", None) or st.request_id
                    i = int(str(rid).split("-")[0]) - CUR["base"]
                    if 0 <= i < len(CUR["keep"]) and CUR["keep"][i]["_batch"] in bad and CUR["keep"][i]["key"] not in done:
                        ids.append(rid)
                if ids:
                    _eng.abort_request(sorted(set(ids)))
                    log(f"{'handover to T servers' if handover else 'cancel'}: aborted {len(ids)} running requests of {sorted(bad)} edit {CUR['es']}")
        except Exception as ex:
            log(f"cancel check failed: {ex!r}")
    return got


_eng.step = _step_with_cancel


def load_rows():
    global rows, rows_m
    f = f"{V4}/rows/learn.jsonl"
    if os.path.getmtime(f) != rows_m:
        rows_m = os.path.getmtime(f)
        rows = {r["id"]: r for r in map(json.loads, open(f))}


def write(batch, rec):
    with open(f"{QOUT}/{batch}.jsonl", "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def run_group(batch, edit, reqs):
    """One edit: apply, generate every request (sub-calls per (mode, temperature, max_tokens)), revert."""
    h, es = None, (esha(edit) if edit is not None else None)
    t_sub = time.time()
    try:
        if edit is not None:
            h = S.apply(surgeon_edit(edit))
            log(f"applied {canon(edit)}: {[(i['hf'], i['changed']) for i in h.info]}")
    except Exception as e:
        log(f"apply failed {canon(edit)}: {e!r}")
        for r in reqs:
            write(r["_batch"], dict(key=r["key"], mode=r.get("mode", "read"), error=f"apply: {e!r}"[:300], edit=edit, edit_sha=es, server=args.name, engine_type="offline"))
        return 0
    n_ok, gen_tok, t1 = 0, 0, time.time()
    try:
        sub = collections.defaultdict(list)
        for r in reqs:
            mode = r.get("mode", "read")
            if mode == "chat":
                key = ("chat", CHAT_TEMP if r.get("temperature") is None else float(r["temperature"]), int(r.get("max_tokens") or CHAT_MAX))
            else:
                st = r.get("settings") or {}
                key = ("read", SAMPLING["temperature"] if st.get("temperature") is None else float(st["temperature"]), int(r.get("max_tokens") or READ_MAX))
            sub[key].append(r)
        for (mode, temp, mt), rs in sub.items():
            prompts, seeds, metas, keep = [], [], [], []
            for r in rs:
                try:
                    if mode == "chat":
                        msgs, nt = chat_messages(r["messages"])
                        prompts.append(msgs)
                        seeds.append(int(r.get("seed") or zlib.crc32(r["key"].encode()) % (2 ** 31 - 1024)))
                    else:
                        row = rows[r["row"]]
                        st = r.get("settings") or {}
                        content, nt = content_for(row["prompt"], row.get("image_paths") or [], st.get("min_image_tokens"), st.get("max_image_tokens"), r.get("prefix"))
                        prompts.append([{"role": "user", "content": content}])
                        seeds.append(int(r["seed"]) if r.get("seed") is not None else crn_seed(r["row"], r.get("sample", 0)))
                    if sum(nt) > MAXLEN - 2000:
                        raise ValueError(f"about {sum(nt)} image tokens > max_model_len {MAXLEN}")
                    metas.append(dict(image_tokens=nt, seed=seeds[-1]))
                    keep.append(r)
                except Exception as e:
                    if len(seeds) > len(metas):
                        seeds.pop()
                    if len(prompts) > len(metas):
                        prompts.pop()
                    write(r["_batch"], dict(key=r["key"], mode=r.get("mode", "read"), error=f"build: {e!r}"[:300], edit=edit, edit_sha=es, server=args.name, engine_type="offline"))
            if not keep:
                continue
            sampling = dict(SAMPLING, temperature=temp)
            ts = time.time()

            def hook(i, o, keep=keep, metas=metas, mode=mode):
                nonlocal n_ok, gen_tok
                r, m, s = keep[i], metas[i], o["samples"][0]
                rec = dict(key=r["key"], mode=r.get("mode", "read"), content=s["content"], reasoning=s["reasoning"], finish=s["finish_reason"],
                           tokens=s["num_tokens"], prompt_tokens=o["prompt_tokens"], image_tokens=m["image_tokens"], seed=m["seed"],
                           edit=edit, edit_sha=es, server=args.name, engine_type="offline", t_submit=ts, t_done=time.time())
                if mode == "read":
                    row = rows[r["row"]]
                    ok, loop, gerr, pred = grade(row, s["content"], s["reasoning"], s["finish_reason"], s["num_tokens"], r.get("sample", 0))
                    rec.update(row=r["row"], sample=r.get("sample", 0), settings=r.get("settings"), prefix=r.get("prefix"), ok=ok, loop=loop, grade_error=gerr, pred=pred,
                               id=row["id"], kind=row["kind"], answer=row["answer"], options=row.get("options") or [], category=row.get("category"),
                               precision=row.get("precision"), bench=row.get("bench") or row["id"].split("/")[0])
                write(r["_batch"], rec)
                done.add(r["key"])
                n_ok += 1
                gen_tok += s["num_tokens"]

            CUR.update(base=S.llm.request_counter.counter, keep=keep, es=es, batches={r["_batch"] for r in keep}, pinned=any(r.get("pin") == "W" for r in keep))
            try:
                S.generate(prompts, n=1, sampling=sampling, max_tokens=mt, reasoning_effort=EFFORT, seeds=seeds, on_output=hook)
            finally:
                CUR.clear()
            for r in keep:
                if r["key"] not in done and cancelled(r["_batch"], es):
                    write_cancel(r, es, edit, "aborted while running")
    except Exception as e:
        traceback.print_exc()
        for r in reqs:
            if r["key"] not in done:
                write(r["_batch"], dict(key=r["key"], mode=r.get("mode", "read"), error=f"generate: {e!r}"[:300], edit=edit, edit_sha=es, server=args.name, engine_type="offline"))
    finally:
        if h is not None:
            rep = S.revert(h)          # raises if not bit-exact: the job dies and must be restarted (stock weights cannot be trusted)
            log(f"reverted: {rep}")
    dt = time.time() - t1
    with open(f"{V4}/stats.jsonl", "a") as f:
        f.write(json.dumps(dict(t=round(time.time(), 1), server=args.name, engine_type="offline", role="W", batch=batch, edit_sha=es, requests=len(reqs), finished=n_ok,
                                seconds=round(dt, 1), gen_tok_s=round(gen_tok / max(dt, 1e-9), 1), apply_s=round(t1 - t_sub, 1))) + "\n")
    log(f"group {batch} edit {es}: {n_ok}/{len(reqs)} in {dt:.0f}s, {gen_tok / max(dt, 1e-9):.0f} gen tok/s")
    return n_ok


def pending():
    """All W requests not yet answered, from batches without .done: [(priority, mtime, batch, req)], plus {batch: reqs}."""
    out, B = [], {}
    for f in glob.glob(f"{QIN}/*.jsonl"):
        b = os.path.basename(f)[:-6]
        if b in seen or os.path.exists(f"{QOUT}/{b}.done"):
            continue
        reqs = [json.loads(l) for l in open(f) if l.strip()]
        if not reqs or not all(r.get("engine") in ("W", "T") for r in reqs):
            seen.add(b)
            continue
        B[b] = reqs
        if os.path.exists(f"{QCAN}/{b}"):
            continue
        m = os.path.getmtime(f)
        ts = t_served()
        for r in reqs:
            if r["key"] not in done and r["key"] not in failed:
                es_r = r.get("edit_sha") or esha(r.get("edit"))
                if es_r in ts and r.get("pin") != "W":       # a T server reads it (dispatcher); pinned W-null reads stay here
                    continue
                if os.path.exists(f"{QCAN}/{b}@{es_r}"):
                    write_cancel(dict(r, _batch=b), es_r, r.get("edit"), "before start")
                    continue
                out.append((int(r.get("priority", 5)), m, b, dict(r, _batch=b)))
    return sorted(out, key=lambda x: x[:2]), B


def close(B):
    """Write .done for batches whose every request has a line (result or error) or was cancelled."""
    for b, reqs in B.items():
        have = set()
        if os.path.exists(f"{QOUT}/{b}.jsonl"):
            for l in open(f"{QOUT}/{b}.jsonl"):
                have.add(json.loads(l)["key"])
        can = os.path.exists(f"{QCAN}/{b}")
        if all(r["key"] in have for r in reqs) or can:
            fin = sum(1 for r in reqs if r["key"] in have)
            with open(f"{QOUT}/{b}.done", "w") as f:
                json.dump(dict(total=len(reqs), finished=fin, cancelled=len(reqs) - fin if can else 0,
                               errors=sum(1 for r in reqs if r["key"] in have and r["key"] not in done), t=time.time(), server=args.name, engine_type="offline"), f)
            seen.add(b)
            log(f"batch {b} done: {fin}/{len(reqs)}")


def claim(batch, es):
    p = f"{V4}/w/claims/{batch}.{es}"
    try:
        fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, args.name.encode())
        os.close(fd)
        return True
    except FileExistsError:
        if open(p).read().strip() == args.name:
            return True
        if os.path.exists(p + ".failed"):           # a cell server could not read it: the first W worker to claim p.w takes it
            try:
                fd = os.open(p + ".w", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, args.name.encode())
                os.close(fd)
                return True
            except FileExistsError:
                return open(p + ".w").read().strip() == args.name
        return False


DEFER = float(os.environ.get("W_DEFER", "0"))      # seconds a weight-cell group waits for a cell server before this worker may take it
os.makedirs(f"{V4}/w/claims", exist_ok=True)
failed = set()
MAX_GROUP = int(os.environ.get("W_MAX_GROUP", "4000"))
while True:
    load_rows()
    todo, B = pending()
    if not todo:
        close(B)
        time.sleep(3)
        continue
    # the most urgent edit this worker can claim; every pending request with that edit (any claimed batch) joins the group: one apply, one long generate.
    # Several W workers: a (batch, edit) pair is owned by the worker that first creates w/claims/<batch>.<edit_sha> (O_EXCL); a restarted
    # worker keeps its own claims (same --name), so nothing is left without a result or an error line.
    grp, ek = [], None
    for k in dict.fromkeys(canon(x[3].get("edit")) for x in todo):
        cand = [x[3] for x in todo if canon(x[3].get("edit")) == k]
        es_k = esha(cand[0].get("edit"))
        if DEFER and cand[0].get("edit") is not None and time.time() - min(x[1] for x in todo if canon(x[3].get("edit")) == k) < DEFER:
            continue
        mine = set()
        for b in dict.fromkeys(r["_batch"] for r in cand):    # one batch per group, so several W workers share same-edit batches
            if claim(b, es_k):
                mine = {b}
                break
        grp = [r for r in cand if r["_batch"] in mine][:MAX_GROUP]
        if grp:
            ek = k
            break
    if not grp:
        close(B)
        time.sleep(3)
        continue
    log(f"group: edit {ek[:80]}, {len(grp)} requests from {len({r['_batch'] for r in grp})} batches")
    run_group("+".join(sorted({r["_batch"] for r in grp}))[:120], grp[0].get("edit"), grp)
    handed = esha(grp[0].get("edit")) in t_served() and grp[0].get("pin") != "W"
    for r in grp:
        if r["key"] not in done and not handed:
            failed.add(r["key"])          # error line written; resent only after a worker restart (handed-over reads: T servers answer)
    close(B)
