"""v4 dispatcher daemon (engine_api.md §2-§5). One process inside the Vesta container, CPU only.
Reads request batches from V4/q/in, sends them to the loopback vLLM servers listed in V4/engines.json (role E / P / W), mixes all batches
by priority (0 first, then submission order), keeps up to `cap` requests in flight per server, appends graded results to V4/q/out/<batch>.jsonl,
writes <batch>.done, honours V4/q/cancel/<batch>, caches router answers in V4/route_cache_v1.jsonl, and writes V4/stats.jsonl.
Idempotent by key: a key with a non-error line in any q/out file is never sent again (index built at start, kept live).
engines.json (re-read every 5 s): servers [{name, port, card, role | roles: [..], state: up | drain, cap, convertible, [weights, edit_sha, model]}].
T servers (role "T", "edit_sha", "model" = served name) hold a tip's weights baked in: a read with engine "T", or engine "W", whose edit_sha
(field, else sha256 of the canonical "edit")[:16] equals an up T server's goes there (the edit is not re-applied). W/T reads that no up T server
serves are parked (the W worker reads them); they are released to T servers as soon as one with their edit_sha is up (keys already in the batch's
q/out file are skipped). Batches made only of W/T requests are closed (.done) by the W worker, every other batch by the dispatcher.
Drain a server: set its state to "drain"; nothing new goes to it, running requests finish, then V4/q/drained/<name>.json appears and it may be
stopped (gpu stop) and its card used for an in-process engine. A request that cannot reach its server (connection refused) is put back in the queue.
usage: python dispatch.py [--v4 DIR] [--stats-every 60]"""
import argparse, asyncio, glob, hashlib, heapq, itertools, json, math, os, re, sys, time, traceback, zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SAMPLING, EFFORT, content_for, grade, chat_messages, image_url_for
import common as C

import aiohttp
from concurrent.futures import ThreadPoolExecutor
POOL = ThreadPoolExecutor(8)          # image read / resize / base64 off the event loop (PIL releases the GIL)

ap = argparse.ArgumentParser()
ap.add_argument("--v4", default="/data/fd/mingdazhang/selfedit/v4")
ap.add_argument("--stats-every", type=float, default=60)
ap.add_argument("--tries", type=int, default=3)
args = ap.parse_args()
V4 = args.v4
QIN, QOUT, QCAN = f"{V4}/q/in", f"{V4}/q/out", f"{V4}/q/cancel"
for d in (QIN, QOUT, QCAN):
    os.makedirs(d, exist_ok=True)
READ_MAX, CHAT_MAX, CHAT_TEMP = 32768, 12000, 0.7
CRN_SEED = 101                     # v3: crn_seeds = crc32(f"{seed}|{CRN_GEN}|{id}") % (2**31-1024), sample k gets seed + k
ROUTE_V1 = "Condition: {c}.\nDo not solve the question above. Does this condition hold for it? Reply with exactly one word: yes or no."   # Venus router pilot v1 wording; decoding v2 (choice yes/no + first-token logprobs)
RV = "v1"
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)


def crn_seed(row, sample):
    return zlib.crc32(f"{CRN_SEED}|0|{row}".encode()) % (2 ** 31 - 1024) + int(sample)


canon = lambda o: json.dumps(o, sort_keys=True, separators=(",", ":"))


def esha_of(r):
    return r.get("edit_sha") or hashlib.sha256(canon(r.get("edit")).encode()).hexdigest()[:16]


def t_served():
    return {i["cfg"]["edit_sha"] for i in S.servers.values() if i["cfg"].get("role") == "T" and i["cfg"].get("edit_sha") and i["cfg"].get("state", "up") == "up"}


def role_of(r, ts):
    """Queue for a request: E / P / route roles as given; W / T reads -> "T@<edit_sha>" if an up T server has their weights, else None (W worker)."""
    eng = r.get("engine", "E")
    if r.get("pin") == "W":                    # W-null reads: always the offline W worker (AM20)
        return None
    if eng in ("W", "T"):
        h = esha_of(r)
        return f"T@{h}" if h in ts else None
    return eng


class State:
    def __init__(self):
        self.done = set()                     # keys with a result line (not error)
        self.batches = {}                     # name -> dict(total, keys set, finished set, cancelled, mtime)
        self.heaps = {"E": [], "P": [], "W": []}
        self.cv = asyncio.Condition()
        self.seq = itertools.count()
        self.rows, self.rows_mtime = {}, 0
        self.route = {}
        self.servers = {}                     # "host:port" -> dict(cfg, inflight, tasks, finished, gen tokens...)
        self.queued_keys = set()
        self.salt_home = {}                   # salt -> (server, t of the first member)
        self.aborted, self.redo = {}, set()   # key -> batches holding an aborted line for it; keys listed in q/redo_keys.txt
        self.loc = {}                         # key -> (file, byte offset) of a plain read result line (extend op base reads)
        self.parked = {}                      # batch -> [requests] W/T reads with no up T server for their edit_sha
        self.engines_seen = False

    def load_rows(self):
        f = f"{V4}/rows/learn.jsonl"
        if os.path.exists(f) and os.path.getmtime(f) != self.rows_mtime:
            self.rows_mtime = os.path.getmtime(f)
            self.rows = {r["id"]: r for r in map(json.loads, open(f))}
            log(f"rows: {len(self.rows)} from {f}")


S = None


def aborted(r):
    """A read the server dropped (finish "abort", e.g. a card restart), or an empty answer with zero generated tokens: not an answer."""
    if r.get("error") or r.get("op_skipped") or r.get("mode") == "chat":
        return False
    return r.get("finish") == "abort" or (not r.get("content") and not r.get("reasoning") and not (r.get("tokens") or 0))


def index_out():
    n, ab = 0, set()
    for f in glob.glob(f"{QOUT}/*.jsonl"):
        b = os.path.basename(f)[:-6]
        off = 0
        with open(f, "rb") as fh:
            for lb in fh:
                lo, off = off, off + len(lb)
                try:
                    r = json.loads(lb)
                except ValueError:
                    continue
                if "error" not in r:
                    if aborted(r):                    # an aborted read is not an answer: never "done"; its own batch does not re-read it
                        S.aborted.setdefault(r["key"], set()).add(b)      # (legacy lines), unless the key is in q/redo_keys.txt
                        continue
                    S.done.add(r["key"]); n += 1
                    if r.get("mode", "read") == "read" and not r.get("ops") and r.get("content") is not None:
                        S.loc[r["key"]] = (f, lo)     # plain read results: the extend op reuses them as its base read
    S.redo = set()
    if os.path.exists(f"{V4}/q/redo_keys.txt"):
        S.redo = {l.strip() for l in open(f"{V4}/q/redo_keys.txt") if l.strip()}
    ab = set(S.aborted) - S.done
    log(f"index: {len(S.done)} keys done in q/out ({len(ab)} keys with only aborted lines: re-read when another batch asks for them; "
        f"{len(ab & S.redo)} listed for a redo in their own batch)")
    f = f"{V4}/route_cache_v1.jsonl"          # v0 answers in route_cache.jsonl are never read
    if os.path.exists(f):
        for l in open(f):
            try:
                r = json.loads(l)
                S.route[(r["row"], r["cond"], json.dumps(r.get("view")), r.get("rv", RV))] = r
            except ValueError:
                pass
    log(f"route cache: {len(S.route)}")


def out_write(batch, rec):
    fn = f"{QOUT}/{batch}.jsonl"
    lo = os.path.getsize(fn) if os.path.exists(fn) else 0
    with open(fn, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    if "error" not in rec and rec.get("mode", "read") == "read" and not rec.get("ops") and rec.get("content") is not None and not aborted(rec):
        S.loc[rec["key"]] = (fn, lo)


def load_loc(key):
    f, lo = S.loc[key]
    with open(f, "rb") as fh:
        fh.seek(lo)
        return json.loads(fh.readline())


def maybe_done(batch):
    b = S.batches[batch]
    if not b["closed"] and len(b["finished"]) + b["cancelled"] >= b["total"]:
        b["closed"] = True
        with open(f"{QOUT}/{batch}.done", "w") as f:
            json.dump(dict(total=b["total"], finished=len(b["finished"]), cancelled=b["cancelled"], errors=b["errors"], t=time.time()), f)
        log(f"batch {batch} done: {len(b['finished'])} results, {b['cancelled']} cancelled, {b['errors']} errors")


async def scan_in():
    while not S.engines_seen:
        await asyncio.sleep(1)
    while True:
        try:
            S.load_rows()
            for f in sorted(glob.glob(f"{QIN}/*.jsonl"), key=os.path.getmtime):
                batch = os.path.basename(f)[:-6]
                if batch in S.batches:
                    continue
                reqs = [json.loads(l) for l in open(f) if l.strip()]
                if os.path.exists(f"{QCAN}/{batch}"):          # cancelled before this scan: nothing queued, no key held
                    wb = bool(reqs) and all(r.get("engine") in ("W", "T") for r in reqs)
                    S.batches[batch] = dict(total=len(reqs), finished=set(), cancelled=len(reqs), errors=0, w=wb,
                                            closed=wb or os.path.exists(f"{QOUT}/{batch}.done"))
                    log(f"batch {batch}: cancelled ({len(reqs)} requests), not queued")
                    maybe_done(batch)
                    continue
                wb = bool(reqs) and all(r.get("engine") in ("W", "T") for r in reqs)     # .done written by the W worker
                b = dict(total=len(reqs), finished=set(), cancelled=0, errors=0, closed=wb or os.path.exists(f"{QOUT}/{batch}.done"), w=wb)
                S.batches[batch] = b
                new, park, ts = 0, [], t_served()
                pf = f"{V4}/q/prio/{batch}.json"              # optional {"rows": [...], "priority": p}: those rows of this batch get priority p
                prio = None
                if os.path.exists(pf):
                    prio = json.load(open(pf)); prio["rows"] = set(prio["rows"])
                for r in reqs:
                    k = r["key"]
                    if k in S.done:
                        b["finished"].add(k)
                        continue
                    if batch in S.aborted.get(k, ()) and k not in S.redo:   # this batch's own legacy aborted line: not re-read here
                        b["finished"].add(k)
                        continue
                    if k in S.queued_keys:      # same key in two batches: the first one answers; mirror below
                        b.setdefault("mirror", []).append(k)
                        continue
                    if prio and r.get("row") in prio["rows"]:
                        r = dict(r, priority=prio["priority"])
                    role = role_of(r, ts)
                    if role is None:
                        park.append(r)
                        continue
                    S.queued_keys.add(k)
                    heapq.heappush(S.heaps.setdefault(role, []), (int(r.get("priority", 5)), next(S.seq), batch, r))
                    new += 1
                    async with S.cv:
                        S.cv.notify_all()
                if park:
                    S.parked[batch] = park
                log(f"batch {batch}: {len(reqs)} requests, {len(b['finished'])} already done, {new} queued, {len(park)} left to the W worker")
                maybe_done(batch)
            # parked W/T reads whose weights a T server now serves
            ts = t_served()
            for batch in list(S.parked):
                if not any(esha_of(r) in ts for r in S.parked[batch]):
                    continue
                have = set()
                if os.path.exists(f"{QOUT}/{batch}.jsonl"):
                    for l in open(f"{QOUT}/{batch}.jsonl"):
                        try:
                            d = json.loads(l)
                        except ValueError:
                            continue
                        if ("error" not in d and not aborted(d)) or d.get("cancelled"):
                            have.add(d["key"])
                keep, new = [], 0
                for r in S.parked[batch]:
                    if r["key"] in have or r["key"] in S.done:
                        continue
                    role = role_of(r, ts)
                    if role is None:
                        keep.append(r)
                        continue
                    if r["key"] in S.queued_keys:
                        continue
                    S.queued_keys.add(r["key"])
                    heapq.heappush(S.heaps.setdefault(role, []), (int(r.get("priority", 5)), next(S.seq), batch, r))
                    new += 1
                if keep:
                    S.parked[batch] = keep
                else:
                    S.parked.pop(batch)
                if new:
                    log(f"batch {batch}: {new} parked W/T reads released to T servers ({len(have)} already answered)")
                    async with S.cv:
                        S.cv.notify_all()
            # mirrors: keys that another batch answered
            for batch, b in S.batches.items():
                if b.get("mirror"):
                    keep = []
                    for k in b["mirror"]:
                        if k in S.done:
                            b["finished"].add(k)
                        else:
                            keep.append(k)
                    b["mirror"] = keep
                    maybe_done(batch)
        except Exception:
            traceback.print_exc()
        await asyncio.sleep(2)


SALT_DELAY = float(os.environ.get("SALT_DELAY", "4"))   # s: a salted sibling waits this long after the first member started (prefill done)


def salt_of(r):
    """cache_salt for vLLM prefix caching: requests that carry "salt" share it (same row and prompt, sent to one server back to back);
    every other request gets a unique salt, so with caching switched on in a server it can never hit the cache (= caching off)."""
    return hashlib.sha256(("s|" + (r.get("salt") or "k|" + r["key"])).encode()).hexdigest()[:32]


def build(r):
    """-> (body, meta) for one request."""
    body, meta = _build(r)
    body["cache_salt"] = salt_of(r)
    return body, meta


def _build(r):
    mode = r.get("mode", "read")
    if mode == "chat":
        msgs, nt = chat_messages(r["messages"])
        body = dict(messages=msgs, max_tokens=int(r.get("max_tokens") or CHAT_MAX),
                    chat_template_kwargs={"reasoning_effort": EFFORT, "enable_thinking": True}, **SAMPLING)
        body["temperature"] = CHAT_TEMP if r.get("temperature") is None else float(r["temperature"])
        if r.get("seed") is not None:
            body["seed"] = int(r["seed"])
        return body, dict(image_tokens=nt)
    row = S.rows[r["row"]]
    if mode == "route":
        v = r.get("view") or {}
        content, nt = content_for(row["prompt"], row.get("image_paths") or [], v.get("min_image_tokens"), v.get("max_image_tokens"))
        content = list(content) + [{"type": "text", "text": ROUTE_V1.format(c=r["condition"].strip().rstrip("."))}]
        body = dict(messages=[{"role": "user", "content": content}], max_tokens=3, temperature=0.0, logprobs=True, top_logprobs=20,
                    chat_template_kwargs={"enable_thinking": False}, structured_outputs={"choice": ["yes", "no"]})
        return body, dict(image_tokens=nt)
    st = r.get("settings") or {}
    content, nt = content_for(row["prompt"], row.get("image_paths") or [], st.get("min_image_tokens"), st.get("max_image_tokens"), r.get("prefix"))
    if st.get("twoscale"):                     # TWOSCALE pilot (coord 11-12): the single image sent twice in one turn, "a" = floor-4096 copy then
        ps = row.get("image_paths") or []      # the stock view, "b" = stock view then floor-4096 copy; no added text
        ps = ps if isinstance(ps, list) else [ps]
        if len(ps) != 1:
            raise ValueError("twoscale needs exactly one image")
        u_big, n_big = image_url_for(ps[0], int(st.get("twoscale_floor") or 4096), None)
        u_orig, n_orig = image_url_for(ps[0], None, None)
        pair = [({"type": "image_url", "image_url": {"url": u_big}}, n_big), ({"type": "image_url", "image_url": {"url": u_orig}}, n_orig)]
        if st["twoscale"] == "b":
            pair = pair[::-1]
        k = next(i for i, c in enumerate(content) if c.get("type") == "image_url")
        content = content[:k] + [x for x, _ in pair] + content[k + 1:]
        nt = [n for _, n in pair]
    body = dict(messages=[{"role": "user", "content": content}], max_tokens=int(r.get("max_tokens") or READ_MAX),
                chat_template_kwargs={"reasoning_effort": EFFORT, "enable_thinking": True}, **SAMPLING)
    if st.get("temperature") is not None:
        body["temperature"] = float(st["temperature"])
    body["seed"] = int(r["seed"]) if r.get("seed") is not None else crn_seed(r["row"], r.get("sample", 0))
    return body, dict(image_tokens=nt, seed=body["seed"])


def cond_hash(c):
    return hashlib.sha256(c.encode()).hexdigest()[:16]


async def handle(srv, sess, batch, r):
    key, mode = r["key"], r.get("mode", "read")
    t_sub = time.time()
    rec = dict(key=key, mode=mode, engine_type="serve")
    cfg = S.servers[srv]["cfg"]
    if cfg.get("role") == "T":                 # tip weights baked in: echo the edit like the W worker does
        rec.update(edit=r.get("edit"), edit_sha=esha_of(r), weights=cfg.get("weights"))
    if mode in ("read", "route"):
        rec.update(row=r["row"])
    if mode == "read":
        rec["sample"] = r.get("sample", 0)
    if mode == "route":
        ch = cond_hash(r["condition"])
        rec["cond"] = ch
        rkey = (r["row"], ch, json.dumps(r.get("view")), RV)
        hit = S.route.get(rkey)
        if hit:
            rec.update({k: hit.get(k) for k in ("yes", "p", "py", "pn", "p_src", "content", "tokens", "prompt_tokens")}, rv=RV, view=r.get("view"),
                       cached=True, t_submit=t_sub, t_done=time.time())
            return rec
    if mode == "read" and r.get("ops"):
        if r["row"] not in S.rows:
            rec.update(error=f"KeyError: row {r['row']} not in rows", server=srv, t_submit=t_sub, t_done=time.time())
            return rec
        if any(o.get("op") == "extend" for o in r["ops"]):
            return await run_extend(srv, sess, batch, r, rec, t_sub)
        return await run_ops(srv, sess, r, rec, t_sub)
    err, o, meta = None, None, {}
    for k in range(args.tries):
        try:
            if mode != "chat" and r["row"] not in S.rows:
                raise KeyError(f"row {r['row']} not in rows/learn.jsonl")
            body, meta = await asyncio.get_running_loop().run_in_executor(POOL, build, r)
            body["model"] = S.servers[srv]["cfg"].get("model", "stock")
            async with sess.post(f"http://{srv}/v1/chat/completions", json=body) as resp:
                txt = await resp.text()
                if resp.status != 200:
                    raise RuntimeError(f"HTTP {resp.status}: {txt[:300]}")
                o = json.loads(txt)
            break
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:300]}"
            if isinstance(e, aiohttp.ClientConnectorError):
                rec.update(error=err, conn=True)
                return rec
            if isinstance(e, KeyError) or "maximum context length" in str(e) or "longer than the maximum" in str(e):
                break
            if k + 1 < args.tries:
                await asyncio.sleep(5 * 4 ** k)
    if o is None:
        rec.update(error=err, server=srv, t_submit=t_sub, t_done=time.time())
        return rec
    c = o["choices"][0]
    m = c["message"]
    content = (m.get("content") or "").strip()
    reasoning = (m.get("reasoning_content") or m.get("reasoning") or "").strip()
    toks, ptoks = o["usage"]["completion_tokens"], o["usage"]["prompt_tokens"]
    finish = c.get("finish_reason")
    rec.update(server=srv, t_submit=t_sub, t_done=time.time())
    if mode == "route":
        content = content or reasoning
        py = pn = 0.0
        try:
            for t in c["logprobs"]["content"][0]["top_logprobs"]:
                w = t["token"].strip().lower().strip(".,!")
                if w == "yes":
                    py += math.exp(t["logprob"])
                elif w == "no":
                    pn += math.exp(t["logprob"])
        except (KeyError, IndexError, TypeError):
            pass
        if py + pn > 0:
            p, src = py / (py + pn), "logprob"
        else:
            p, src = (1.0 if content.strip().lower() == "yes" else 0.0), "choice"
        yes = p >= 0.5
        res = dict(yes=yes, p=p, py=py, pn=pn, p_src=src, content=content, tokens=toks, prompt_tokens=ptoks)
        rec.update(res, rv=RV, view=r.get("view"), cached=False)
        cr = dict(row=r["row"], cond=rec["cond"], view=r.get("view"), rv=RV, condition=r["condition"], t=time.time(), **res)
        S.route[(r["row"], rec["cond"], json.dumps(r.get("view")), RV)] = cr
        with open(f"{V4}/route_cache_v1.jsonl", "a") as f:
            f.write(json.dumps(cr, ensure_ascii=False) + "\n")
        return rec
    rec.update(content=content, reasoning=reasoning, finish=finish, tokens=toks)
    if mode == "chat":
        rec.update(prompt_tokens=ptoks, image_tokens=meta.get("image_tokens"))
        return rec
    row = S.rows[r["row"]]
    ok, loop, gerr, pred = grade(row, content, reasoning, finish, toks, rec["sample"])
    rec.update(prompt_tokens=ptoks, image_tokens=meta["image_tokens"], seed=meta["seed"], settings=r.get("settings"), prefix=r.get("prefix"),
               ok=ok, loop=loop, grade_error=gerr, pred=pred, id=row["id"], kind=row["kind"], answer=row["answer"], options=row.get("options") or [],
               category=row.get("category"), precision=row.get("precision"), bench=row.get("bench") or row["id"].split("/")[0])
    return rec


def roles_of(cfg):
    """A server answers the roles in "roles" (list) or "role"; e.g. an E server that also takes P chat until the Surgeon P exists."""
    if cfg.get("role") == "T":
        return [f"T@{cfg.get('edit_sha')}"]
    return list(cfg.get("roles") or [cfg.get("role", "E")])


async def post(sess, srv, body):
    """One chat call with retries; returns (json, None) or (None, error). A refused connection raises (the caller requeues)."""
    err = None
    body = dict(body, model=S.servers[srv]["cfg"].get("model", "stock"))
    body.setdefault("cache_salt", os.urandom(16).hex())       # every call without a salt group is unique: never a prefix-cache hit
    for k in range(args.tries):
        try:
            async with sess.post(f"http://{srv}/v1/chat/completions", json=body) as resp:
                txt = await resp.text()
                if resp.status != 200:
                    raise RuntimeError(f"HTTP {resp.status}: {txt[:300]}")
                return json.loads(txt), None
        except aiohttp.ClientConnectorError:
            raise
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:300]}"
            if "maximum context length" in str(e) or "longer than the maximum" in str(e):
                break
            if k + 1 < args.tries:
                await asyncio.sleep(5 * 4 ** k)
    return None, err


def build_ops(r, ops, box):
    """Pass-2 read body: the read content with zoom crop / tiles appended after the row's last image."""
    row = S.rows[r["row"]]
    st = r.get("settings") or {}
    paths = row.get("image_paths") or []
    content, nt = content_for(row["prompt"], paths, st.get("min_image_tokens"), st.get("max_image_tokens"), r.get("prefix"))
    extra, trace = [], {}
    for o in ops:
        if o["op"] == "zoom" and box:
            p, n = C.crop_part(paths[box[0] - 1], box, nt[box[0] - 1])
            extra.append(p); trace["zoom_tokens"] = n
        elif o["op"] == "tiles":
            k = int(o.get("n", 2))
            for i, pth in enumerate(paths):
                ts = C.tile_parts(pth, k, nt[i])
                extra += [p for p, _ in ts]
            trace["tiles"] = len(extra) - (1 if (box and any(x["op"] == "zoom" for x in ops)) else 0)
    content = C.insert_after_last_image(content, extra)
    body = dict(messages=[{"role": "user", "content": content}], max_tokens=int(r.get("max_tokens") or READ_MAX),
                chat_template_kwargs={"reasoning_effort": EFFORT, "enable_thinking": True}, **SAMPLING)
    if st.get("temperature") is not None:
        body["temperature"] = float(st["temperature"])
    body["seed"] = int(r["seed"]) if r.get("seed") is not None else crn_seed(r["row"], r.get("sample", 0))
    return body, nt, trace


def build_plain(r, tail_text):
    row = S.rows[r["row"]]
    st = r.get("settings") or {}
    content, nt = content_for(row["prompt"], row.get("image_paths") or [], st.get("min_image_tokens"), st.get("max_image_tokens"), r.get("prefix"))
    return list(content) + [{"type": "text", "text": tail_text}]


def msg_of(o):
    c = o["choices"][0]
    m = c["message"]
    return ((m.get("content") or "").strip(), (m.get("reasoning_content") or m.get("reasoning") or "").strip(), c.get("finish_reason"),
            o["usage"]["completion_tokens"], o["usage"]["prompt_tokens"])


SV_ANSWER_MAX = 800


def cut_reasoning(t):
    """The reasoning cut back to its last sentence end (". " or a newline); unchanged when it has none."""
    i, j = t.rfind(". "), t.rfind("\n")
    if i < 0 and j < 0:
        return t.rstrip()
    return t[:i + 1].rstrip() if i > j else t[:j].rstrip()


async def short_vote(sess, srv, row, r, body, sv):
    """op short_vote {c, m}: m reads with the reasoning capped at c generated tokens, plurality of their graded preds (coordinator 11-11 03:40).
    Read k: seed = read seed + 1000 * (k + 1), max_tokens = c. Finished by itself (finish stop) -> that is the read. Otherwise the reasoning is
    cut back to its last sentence end and a second call pre-fills the assistant turn with it (content "", reasoning = cut; continue_final_message,
    no generation prompt: the prompt ends "<think>\n<reasoning>\n</think>") and lets the model answer (max_tokens 400).
    Returns (content, reasoning, finish, total generated tokens, prompt tokens, per-read trace)."""
    c, m = int(sv["c"]), int(sv["m"])
    amax, gate = int(sv.get("a", SV_ANSWER_MAX)), bool(sv.get("gate"))
    sample = r.get("sample", 0)

    async def first(k):
        sk = int(body["seed"]) + 1000 * (k + 1)
        o, err = await post(sess, srv, dict(body, max_tokens=c, seed=sk))
        if o is None:
            raise RuntimeError(f"short_vote read {k}: {err}")
        return (sk,) + msg_of(o)

    async def one(k, pre=None):
        sk, c1, r1, f1, t1, p1 = pre if pre is not None else await first(k)
        if f1 == "stop":
            text, reas, fin, toks, cut, t2 = c1, r1, f1, t1, False, 0
        else:
            cr = cut_reasoning(r1)
            b2 = dict(body, messages=list(body["messages"]) + [{"role": "assistant", "content": "", "reasoning": cr}], max_tokens=amax,
                      seed=sk, continue_final_message=True, add_generation_prompt=False)
            o2, err = await post(sess, srv, b2)
            if o2 is None:
                raise RuntimeError(f"short_vote answer {k}: {err}")
            c2, r2, fin, t2, _ = msg_of(o2)
            text, reas, toks, cut = (c2 or r2), cr, t1 + t2, True
        _, _, _, pred = grade(row, text, reas, fin, toks, sample)
        return dict(k=k, seed=sk, tokens=toks, tokens_reason=t1, tokens_answer=t2, cut=cut, finish=fin, pred=pred, tail=(text or "")[-160:]), p1

    gate_open = None
    if gate:                     # gated form: read 1 alone when it finishes within c; the ensemble only when it is cut
        pre = await first(0)
        if pre[3] == "stop":
            x, p1 = await one(0, pre)
            x["gate_open"] = False
            return pre[1], pre[2], "stop", x["tokens"], p1, [x]
        gate_open = True
        res = await asyncio.gather(one(0, pre), *(one(k) for k in range(1, m)))
    else:
        res = await asyncio.gather(*(one(k) for k in range(m)))
    reads = [x for x, _ in res]
    if gate_open:
        reads[0]["gate_open"] = True
    total = sum(x["tokens"] for x in reads)
    votes = {}
    for x in reads:
        p = x["pred"]
        if p is None or (isinstance(p, str) and not p.strip()):
            continue
        q = str(p)
        votes.setdefault(q, [0, x["k"]])[0] += 1
    if not votes:
        return reads[0]["tail"], "", reads[0]["finish"], total, res[0][1], reads
    best = max(votes.items(), key=lambda kv: (kv[1][0], -kv[1][1]))[0]
    return "\\boxed{" + best + "}", "", "stop", total, res[0][1], reads


EXT_CUE = "\n\nWait"
EXT_SEED = 7000


def ext_body(body0, acc, seed, max_tokens):
    """Venus's open-think form (template chat_template_open_think.jinja, coordinator 11-11): the final assistant message carries the thinking so
    far in "reasoning" (vLLM maps it to reasoning_content), content empty; with chat_template_kwargs open_final_think the template renders it as
    "<|im_start|>assistant\\n<think>\\n" + trim(acc) with no close and no im_end; continue_final_message and add_generation_prompt both false."""
    ctk = dict(body0.get("chat_template_kwargs") or {}, enable_thinking=True, open_final_think=True)
    ctk.setdefault("reasoning_effort", EFFORT)
    return dict(body0, messages=list(body0["messages"]) + [{"role": "assistant", "content": "", "reasoning": acc}],
                chat_template_kwargs=ctk, continue_final_message=False, add_generation_prompt=False, seed=seed, max_tokens=max_tokens)


_END_IDS = {}


async def end_ids(sess, srv):
    """token ids that would close the thinking block or end the turn: </think>, <|im_end|>, <|endoftext|> (looked up from the server tokenizer)"""
    if srv not in _END_IDS:
        ids = []
        for t in ("</think>", "<|im_end|>", "<|endoftext|>"):
            async with sess.post(f"http://{srv}/tokenize", json=dict(model=S.servers[srv]["cfg"].get("model", "stock"), prompt=t, add_special_tokens=False)) as resp:
                tk = json.loads(await resp.text())["tokens"]
            if len(tk) != 1:
                raise RuntimeError(f"{t!r} is not one token: {tk}")
            ids.append(int(tk[0]))
        _END_IDS[srv] = ids
    return _END_IDS[srv]


def drop_last_paragraph(t):
    t = t.rstrip()
    i = t.rfind("\n\n")
    return t[:i] if i > 0 else t


async def n_tokens(sess, srv, text):
    async with sess.post(f"http://{srv}/tokenize", json=dict(model=S.servers[srv]["cfg"].get("model", "stock"), prompt=text, add_special_tokens=False)) as resp:
        return int(json.loads(await resp.text())["count"])


async def run_extend(srv, sess, batch, r, rec, t_sub):
    """op extend {k, gate, mode} (budget forcing, coordinator 11-11): base read = the tip's own read of (row, sample) (request "base_key",
    cached in q/out or read fresh and written under that key). If its reasoning has >= gate tokens (and it finished), K continuation rounds stay
    inside the thinking block: assistant content "<think>\n" + R + "\n\nWait" (mode cutwait: R minus its last paragraph), continue_final_message,
    no generation prompt (the template adds an empty closed think block before it: a constant 4 tokens); the model thinks on, closes the block
    and answers. Round i: seed = base seed + 7000 i; the prefill grows by each round's added thinking + "\n\nWait". Result = last round's answer;
    a failed / cut continuation falls back to the base answer. tokens = base tokens + all added tokens."""
    ex = next(o for o in r["ops"] if o.get("op") == "extend")
    K, T, mode, MIN = int(ex.get("k", 1)), int(ex.get("gate", 0)), ex.get("mode", "wait"), int(ex.get("min", 0))
    row = S.rows[r["row"]]
    sample = r.get("sample", 0)
    bkey = r.get("base_key")
    trace = dict(op="extend", k=K, gate=T, mode=mode, min=MIN, base_key=bkey)
    try:
        if bkey and bkey in S.loc:
            base = load_loc(bkey); trace["base_cached"] = True
        else:
            rb = {k: v for k, v in r.items() if k not in ("ops", "base_key")}
            rb["key"] = bkey or (r["key"] + "|base")
            base = await handle(srv, sess, batch, rb)
            if base.get("conn"):
                return base
            if "error" in base:
                raise RuntimeError(f"base read: {base['error']}")
            if aborted(base):
                rec.update(error="base read aborted", conn=False, server=srv, t_submit=t_sub, t_done=time.time())
                return rec
            out_write(batch, base); S.done.add(base["key"]); trace["base_cached"] = False
        R0, C0 = base.get("reasoning") or "", base.get("content") or ""
        st = r.get("settings") or {}
        body0, meta = await asyncio.get_running_loop().run_in_executor(POOL, _build, {k: v for k, v in r.items() if k not in ("ops", "base_key")})
        bseed = int(base.get("seed") or body0["seed"])
        rt = await n_tokens(sess, srv, R0) if R0 else 0
        trace.update(base_reason_tokens=rt, base_tokens=base.get("tokens"), base_finish=base.get("finish"), base_pred=base.get("pred"), base_ok=base.get("ok"))
        content, reasoning, finish, total = C0, R0, base.get("finish"), int(base.get("tokens") or 0)
        trace["extended"] = False
        if rt >= T and base.get("finish") == "stop":
            trace["extended"] = True
            acc = (drop_last_paragraph(R0) if mode == "cutwait" else R0).rstrip() + EXT_CUE
            rounds = []
            for i in range(1, K + 1):
                room = 45900 - int(base.get("prompt_tokens") or 0) - (total - int(base.get("tokens") or 0)) - rt - 64
                seed_i = bseed + EXT_SEED * i
                rd = dict(round=i)
                p1, ct1 = "", 0
                if MIN > 0:        # part 1: the block cannot close (and the turn cannot end) for the first MIN added tokens
                    ids = await end_ids(sess, srv)
                    b1 = ext_body(body0, acc, seed_i, max(16, min(MIN, room)))
                    b1["logit_bias"] = {str(t): -100 for t in ids}
                    o, err = await post(sess, srv, b1)
                    if o is None:
                        rd["error"] = "part1: " + str(err)[:200]; rounds.append(rd); trace["fallback"] = f"round {i} part1: {str(err)[:120]}"; break
                    ch1 = o["choices"][0]; m1 = ch1["message"]
                    p1 = (m1.get("reasoning_content") or m1.get("reasoning") or "") + (m1.get("content") or "")
                    ct1 = o["usage"]["completion_tokens"]; total += ct1
                    rd.update(part1_tokens=ct1, part1_finish=ch1.get("finish_reason"), part1_prompt_tokens=o["usage"]["prompt_tokens"])
                    room -= ct1
                b = ext_body(body0, acc + p1, seed_i, max(256, min(int(body0["max_tokens"]), room)))
                o, err = await post(sess, srv, b)
                if o is None:
                    rd["error"] = str(err)[:200]; rounds.append(rd); trace["fallback"] = f"round {i}: {str(err)[:120]}"; break
                ch = o["choices"][0]; m = ch["message"]
                add_r = m.get("reasoning_content") or m.get("reasoning") or ""
                c2 = (m.get("content") or "").strip()
                ct = o["usage"]["completion_tokens"]
                rtk = ((o["usage"].get("completion_tokens_details") or {}).get("reasoning_tokens"))
                total += ct
                fin2, cta = ch.get("finish_reason"), 0
                if fin2 == "stop" and not c2:
                    # the block closed (or the turn ended) with no answer -- the answer was written inside the thinking: close the block and
                    # force the answer (s1 form: reasoning closed by the stock branch, continue_final_message, answer cap SV_ANSWER_MAX)
                    full = (acc + p1 + add_r).rstrip()
                    ba = dict(body0, messages=list(body0["messages"]) + [{"role": "assistant", "content": "", "reasoning": full}],
                              continue_final_message=True, add_generation_prompt=False, seed=seed_i + 1, max_tokens=SV_ANSWER_MAX)
                    oa, erra = await post(sess, srv, ba)
                    if oa is not None:
                        ma = oa["choices"][0]["message"]; c2 = ((ma.get("content") or "") + "").strip() or (ma.get("reasoning_content") or ma.get("reasoning") or "").strip()
                        cta = oa["usage"]["completion_tokens"]; total += cta; fin2 = oa["choices"][0].get("finish_reason")
                        rd.update(answer_forced=True, answer_tokens=cta, answer_finish=fin2)
                    else:
                        rd.update(answer_forced=True, answer_error=str(erra)[:200])
                _, _, _, p2 = grade(row, c2, add_r, fin2, ct, sample)
                ch = dict(ch, finish_reason=fin2)
                rd.update(tokens=ct1 + ct + cta, part2_tokens=ct, reason_tokens=rtk, prompt_tokens=o["usage"]["prompt_tokens"], finish=ch.get("finish_reason"),
                          closed_at_once=(rtk is not None and rtk < 30), pred=p2)
                rounds.append(rd)
                if ch.get("finish_reason") != "stop" or not c2:
                    trace["fallback"] = f"round {i}: finish {ch.get('finish_reason')}, content {'empty' if not c2 else 'cut'}"
                    break
                content, finish = c2, "stop"
                reasoning = acc + p1 + add_r
                acc = reasoning.rstrip() + EXT_CUE
            trace["rounds"] = rounds
            if trace.get("fallback"):
                content, reasoning, finish = C0, R0, base.get("finish")
    except aiohttp.ClientConnectorError as e:
        rec.update(error=f"{type(e).__name__}: {e}"[:300], conn=True)
        return rec
    except Exception as e:
        rec.update(error=f"extend: {e!r}"[:300], server=srv, t_submit=t_sub, t_done=time.time(), op_trace=[trace])
        return rec
    ok, loopf, gerr, pred = grade(row, content, reasoning, finish, total, sample)
    trace["answer_changed"] = (pred != base.get("pred"))
    rec.update(server=srv, t_submit=t_sub, t_done=time.time(), content=content, reasoning=reasoning, finish=finish, tokens=total,
               prompt_tokens=base.get("prompt_tokens"), image_tokens=base.get("image_tokens"), seed=bseed, settings=r.get("settings"), prefix=r.get("prefix"),
               ops=r["ops"], op_trace=[trace], ok=ok, loop=loopf, grade_error=gerr, pred=pred, id=row["id"], kind=row["kind"], answer=row["answer"],
               options=row.get("options") or [], category=row.get("category"), precision=row.get("precision"), bench=row.get("bench") or row["id"].split("/")[0])
    return rec


async def run_ops(srv, sess, r, rec, t_sub):
    """Harness ops on a read, fixed order zoom -> tiles -> (read) -> check (engine_api.md §7)."""
    ops = sorted(r["ops"], key=lambda o: ["zoom", "tiles", "check", "short_vote"].index(o["op"]))
    sv = next((o for o in ops if o["op"] == "short_vote"), None)
    row = S.rows[r["row"]]
    loop_ = asyncio.get_running_loop()
    trace, total = [], 0
    why = await loop_.run_in_executor(POOL, C.ops_skip_reason, row, ops, r.get("settings"))
    if why:        # coordinator rule: the op is not applied on this row; the loop treats the row as unrouted (parent prompt and samples)
        rec.update(op_skipped=why, ops=r["ops"], server=srv, t_submit=t_sub, t_done=time.time(), id=row["id"], bench=row.get("bench") or row["id"].split("/")[0])
        return rec
    n_img, _ = await loop_.run_in_executor(POOL, C.ops_need, row, ops, r.get("settings"))
    lim = int(S.servers[srv]["cfg"].get("image_limit", 16))
    if n_img > lim:        # fits the ops limit but not this server (still at 16 during the rolling restart): back to the queue
        rec.update(error=f"server image limit {lim} < {n_img}", conn=True)
        return rec
    try:
        box = None
        if any(o["op"] == "zoom" for o in ops):
            content = await loop_.run_in_executor(POOL, build_plain, r, C.ZOOM_Q)
            o, err = await post(sess, srv, dict(messages=[{"role": "user", "content": content}], max_tokens=48, temperature=0.0,
                                                chat_template_kwargs={"enable_thinking": False}))
            if o is None:
                raise RuntimeError(f"zoom pass: {err}")
            txt, _, _, tk, pt = msg_of(o)
            total += tk
            box = C.parse_box(txt, len(row.get("image_paths") or []))
            trace.append(dict(op="zoom", tokens=tk, prompt_tokens=pt, reply=txt[:120], box=box, zoom_fail=box is None))
        body, nt, tr = await loop_.run_in_executor(POOL, build_ops, r, ops, box)
        if sv is not None:
            if any(x["op"] == "check" for x in ops):
                raise RuntimeError("short_vote does not combine with check")
            content, reasoning, finish, tk, pt, svt = await short_vote(sess, srv, row, r, body, sv)
            total += tk
            trace.append(dict(op="short_vote", c=int(sv["c"]), m=int(sv["m"]), gate=bool(sv.get("gate")), a=int(sv.get("a", SV_ANSWER_MAX)),
                              gate_open=svt[0].get("gate_open"), tokens=tk, prompt_tokens=pt, finish=finish, reads=svt, **tr))
        else:
            o, err = await post(sess, srv, body)
            if o is None:
                raise RuntimeError(f"read pass: {err}")
            content, reasoning, finish, tk, pt = msg_of(o)
            total += tk
            trace.append(dict(op="read", tokens=tk, prompt_tokens=pt, finish=finish, **tr))
        if any(x["op"] == "check" for x in ops):
            prop = C.proposed_answer(content)
            cc = await loop_.run_in_executor(POOL, build_plain, r, C.CHECK_Q.format(a=prop))
            cb = dict(messages=[{"role": "user", "content": cc}], max_tokens=int(r.get("max_tokens") or READ_MAX),
                      chat_template_kwargs={"reasoning_effort": EFFORT, "enable_thinking": True}, **SAMPLING, seed=body["seed"])
            o, err = await post(sess, srv, cb)
            if o is None:
                raise RuntimeError(f"check pass: {err}")
            ok0, _, _, pred0 = grade(row, content, reasoning, finish, tk, r.get("sample", 0))
            content, reasoning, finish, tk, pt2 = msg_of(o)
            total += tk
            trace.append(dict(op="check", tokens=tk, prompt_tokens=pt2, finish=finish, proposed=str(prop)[:300], ok_before=ok0, pred_before=pred0))
    except aiohttp.ClientConnectorError as e:
        rec.update(error=f"{type(e).__name__}: {e}"[:300], conn=True)
        return rec
    except Exception as e:
        rec.update(error=f"ops: {e!r}"[:300], server=srv, t_submit=t_sub, t_done=time.time(), op_trace=trace)
        return rec
    ok, loopf, gerr, pred = grade(row, content, reasoning, finish, total, r.get("sample", 0))
    rec.update(server=srv, t_submit=t_sub, t_done=time.time(), content=content, reasoning=reasoning, finish=finish, tokens=total,
               prompt_tokens=pt, image_tokens=nt, seed=body["seed"], settings=r.get("settings"), prefix=r.get("prefix"), ops=r["ops"], op_trace=trace,
               ok=ok, loop=loopf, grade_error=gerr, pred=pred, id=row["id"], kind=row["kind"], answer=row["answer"], options=row.get("options") or [],
               category=row.get("category"), precision=row.get("precision"), bench=row.get("bench") or row["id"].split("/")[0])
    return rec


async def _stick(home, item, delay):
    await asyncio.sleep(delay)
    async with S.cv:
        heapq.heappush(S.heaps.setdefault(f"srv:{home}", []), item)
        S.cv.notify_all()


async def pop(info, srv=None):
    while True:
        async with S.cv:
            while True:
                if info["cfg"].get("state", "up") != "up":
                    return None, None
                heads = [(S.heaps[ro][0], ro) for ro in roles_of(info["cfg"]) + ([f"srv:{srv}"] if srv else []) if S.heaps.get(ro)]
                if heads:
                    break
                try:
                    await asyncio.wait_for(S.cv.wait(), 10)
                except asyncio.TimeoutError:
                    pass
            pr, sq, batch, r = heapq.heappop(S.heaps[min(heads)[1]])
        if os.path.exists(f"{QCAN}/{batch}"):
            S.batches[batch]["cancelled"] += 1
            S.queued_keys.discard(r["key"])
            maybe_done(batch)
            continue
        if r.get("salt") and srv:                      # salted siblings: the first one fixes the server, the others follow it there
            first = r["salt"] not in S.salt_home
            home = S.salt_home.setdefault(r["salt"], (srv, time.time()))
            if first:
                return batch, r
            if home[0] != srv and home[0] in S.servers and S.servers[home[0]]["cfg"].get("state", "up") == "up":
                asyncio.ensure_future(_stick(home[0], (-1000, next(S.seq), batch, r), max(0.0, home[1] + SALT_DELAY - time.time())))
                continue
            if home[0] == srv and time.time() < home[1] + SALT_DELAY and not r.get("_stuck"):
                asyncio.ensure_future(_stick(srv, (-1000, next(S.seq), batch, dict(r, _stuck=1)), home[1] + SALT_DELAY - time.time()))
                continue
        return batch, r


async def worker(srv, slot):
    info = S.servers[srv]
    while True:
        if slot >= info["cap"] or info["cfg"].get("state", "up") != "up":
            await asyncio.sleep(3)
            continue
        batch, r = await pop(info, srv)
        if batch is None:
            continue
        info["inflight"] += 1
        try:
            rec = await handle(srv, info["sess"], batch, r)
        except Exception as e:
            traceback.print_exc()
            rec = dict(key=r["key"], mode=r.get("mode", "read"), error=f"dispatcher: {e!r}"[:300], server=srv)
        info["inflight"] -= 1
        b = S.batches[batch]
        if rec.get("conn"):              # server unreachable: back to the queue, this slot pauses
            log(f"{srv} unreachable ({rec['error'][:80]}); {r['key']} requeued")
            async with S.cv:
                ro = role_of(r, t_served())
                if ro is None:             # its T server is gone: back to the W worker
                    S.queued_keys.discard(r["key"])
                    S.parked.setdefault(batch, []).append(r)
                else:
                    heapq.heappush(S.heaps.setdefault(ro, []), (int(r.get("priority", 5)), next(S.seq), batch, r))
                S.cv.notify_all()
            await asyncio.sleep(30)
            continue
        if aborted(rec):                 # dropped by the server: requeue up to 3 times, then an error line (the loop treats it as not read)
            na = int(r.get("_aborts", 0)) + 1
            if na <= 3:
                log(f"{srv}: {r['key']} aborted (finish {rec.get('finish')!r}, {rec.get('tokens')} tokens); requeued ({na}/3)")
                async with S.cv:
                    ro = role_of(r, t_served())
                    if ro is None:
                        S.queued_keys.discard(r["key"])
                        S.parked.setdefault(batch, []).append(dict(r, _aborts=na))
                    else:
                        heapq.heappush(S.heaps.setdefault(ro, []), (int(r.get("priority", 5)), next(S.seq), batch, dict(r, _aborts=na)))
                    S.cv.notify_all()
                await asyncio.sleep(5)
                continue
            rec = dict(key=r["key"], mode=r.get("mode", "read"), row=r.get("row"), sample=r.get("sample"), server=srv,
                       error=f"aborted by the server {na} times (finish {rec.get('finish')!r})", finish=rec.get("finish"))
        out_write(batch, rec)
        S.queued_keys.discard(r["key"])
        if "error" in rec:
            b["errors"] += 1
            info["errors"] += 1
            log(f"error {r['key']} on {srv}: {rec['error'][:200]}")
        else:
            S.done.add(r["key"])
            info["finished"] += 1
            info["gen_tokens"] += rec.get("tokens") or 0
        b["finished"].add(r["key"])
        maybe_done(batch)


async def engines_loop():
    while True:
        try:
            cfg = json.load(open(f"{V4}/engines.json"))
            for s in cfg["servers"]:
                srv = f"{s.get('host', '127.0.0.1')}:{s['port']}"
                cap = int(s.get("cap", 60))
                if srv not in S.servers:
                    sess = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=4 * 3600),
                                                 connector=aiohttp.TCPConnector(limit=0))
                    S.servers[srv] = dict(cfg=s, cap=cap, inflight=0, finished=0, errors=0, gen_tokens=0, sess=sess, slots=0)
                    log(f"server {srv} role {s.get('role', 'E')} card {s.get('card')} cap {cap}")
                info = S.servers[srv]
                info["cfg"], info["cap"] = s, cap
                st = s.get("state", "up")
                if st == "drain" and info["inflight"] == 0 and not info.get("drained"):
                    os.makedirs(f"{V4}/q/drained", exist_ok=True)
                    with open(f"{V4}/q/drained/{s.get('name', srv)}.json", "w") as f:
                        json.dump(dict(server=srv, card=s.get("card"), t=time.time(), finished=info["finished"]), f)
                    info["drained"] = True
                    log(f"server {srv} drained (card {s.get('card')}): no request in flight, nothing new sent; it may be stopped")
                if st == "up":
                    info["drained"] = False
                while info["slots"] < cap:
                    asyncio.ensure_future(worker(srv, info["slots"]))
                    info["slots"] += 1
        except Exception:
            traceback.print_exc()
        S.engines_seen = True
        await asyncio.sleep(5)


METRICS = {"vllm:num_requests_running": "running", "vllm:num_requests_waiting": "waiting", "vllm:generation_tokens_total": "gen_total",
           "vllm:prompt_tokens_total": "prompt_total", "vllm:num_preemptions_total": "preemptions", "vllm:kv_cache_usage_perc": "kv_usage",
           "vllm:gpu_cache_usage_perc": "kv_usage"}


async def scrape(sess, srv):
    out = {}
    async with sess.get(f"http://{srv}/metrics", timeout=aiohttp.ClientTimeout(total=20)) as resp:
        for l in (await resp.text()).splitlines():
            if l.startswith("#"):
                continue
            m = re.match(r"^([a-zA-Z_:]+)(\{[^}]*\})?\s+([0-9.eE+-]+)$", l)
            if m and m.group(1) in METRICS:
                out[METRICS[m.group(1)]] = out.get(METRICS[m.group(1)], 0) + float(m.group(3))
    return out


async def stats_loop():
    prev = {}
    peak = {}
    sess = aiohttp.ClientSession()
    while True:
        t = time.time()
        for srv, info in list(S.servers.items()):
            try:
                m = await scrape(sess, srv)
            except Exception as e:
                m = dict(scrape_error=repr(e)[:100])
            p = prev.get(srv)
            rec = dict(t=round(t, 1), server=srv, card=info["cfg"].get("card"), role=info["cfg"].get("role", "E"), cap=info["cap"],
                       state=info["cfg"].get("state", "up"), inflight=info["inflight"], finished=info["finished"], errors=info["errors"], **{k: m.get(k) for k in ("running", "waiting", "preemptions", "kv_usage")})
            peak[srv] = max(peak.get(srv, 0), m.get("running") or 0)
            rec["peak_running"] = peak[srv]
            if p and "gen_total" in m and "gen_total" in p[1]:
                dt = t - p[0]
                rec["gen_tok_s"] = round((m["gen_total"] - p[1]["gen_total"]) / dt, 1)
                rec["prompt_tok_s"] = round((m.get("prompt_total", 0) - p[1].get("prompt_total", 0)) / dt, 1)
                rec["finished_window"] = info["finished"] - p[2]
            prev[srv] = (t, m, info["finished"])
            with open(f"{V4}/stats.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
        q = {k: len(v) for k, v in S.heaps.items()}
        log(f"queues {q}; " + "; ".join(f"{s} if={i['inflight']} fin={i['finished']}" for s, i in S.servers.items()))
        await asyncio.sleep(max(1, args.stats_every - (time.time() - t)))


async def main():
    global S
    S = State()
    S.load_rows()
    index_out()
    await asyncio.gather(engines_loop(), scan_in(), stats_loop())


if __name__ == "__main__":
    asyncio.run(main())
