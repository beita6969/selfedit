"""Fake v4 engine for dry runs: engine_api.md §2-§4 on a local root directory, no HTTP, no GPU.
Mirrors engine_v4/dispatch.py: batches from <root>/q/in/*.jsonl (renamed files only; .tmp never matches), results appended to
q/out/<batch>.jsonl, <batch>.done ({total, finished, cancelled, errors, t}) once every request has a result or was cancelled,
q/cancel/<batch> drops not-yet-started requests (checked when a request leaves the queue), idempotent by key (index of
non-error q/out lines built at start; a key queued by an earlier batch is mirrored, not re-run), route cache in
<root>/route_cache.jsonl keyed by (row, sha256(condition)[:16]). One queue for all engines, ordered by (priority, arrival).

Simulated answers (deterministic, no wall clock in any outcome):
  read : ok ~ Bernoulli(p), p = base(row) + effect_scale * effect(prefix, settings); base = hash(row) in [0.2, 0.9];
         effect = hash(prefix, non-stock settings) in [-0.06, +0.08], 0 for a null prefix with stock settings.
         Draw rng = Random(f"{seed}|{effect hash}"), seed = request seed or crc32("101|0|<row>") % (2**31-1024) + sample.
         1 % finish "length" (ok False, empty content, tokens = max_tokens); else tokens 500-6000.
         pred = the boxed answer (None on "length"), on every read.
  ops  : harness ops on read (engine_api.md §7): echoed, op_trace per pass (zoom, read, check), tokens = all passes:
         read pass R = base x 1.15 (zoom) x (1 + 0.2 n) (tiles), capped at max_tokens; zoom pass 48; check pass 0.8 R
         (so zoom alone gives base x 1.15 + 48, tiles base x (1 + 0.2 n), check base x 1.8).
         p gains op_effect_scale x OP_EFFECT[op] + a per-(op, row) spread of +-OP_SPREAD (zoom gains nothing when zoom_fail,
         5 % of samples); check: ok_before ~ the p without check, ok ~ the p with check, on the same uniform.
         Draws use Random(f"{seed}|{effect hash}|{ops signature}"), so a read with ops is a different draw than without.
         More than 40 images after zoom / tiles -> an op_skipped line (no content, no ok), as dispatch.py run_ops.
         An unknown op -> an error line.
  route: yes iff hash(row, cond) < q(cond), q = hash(cond) in [0.1, 0.6].
  chat : canned proposer answer per key: ~40 % "always" setting change, ~50 % worded condition + procedure, ~10 % broken JSON;
         with --chat-ops x, that share of the parsable answers also carries ops (default 0: the answers above, unchanged).

usage: python3 fake_dispatch.py --root DIR [--rows learn.jsonl] [--knobs knobs.json] [--rate 200] [--jitter 0.0]
                                [--effect-scale 1.0] [--op-effect 1.0] [--chat-ops 0.0] [--once] [--max-results N] [--quiet]
"""
import argparse
import glob
import hashlib
import heapq
import itertools
import json
import os
import random
import re
import sys
import time
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
CRN_SEED = 101
READ_MAX = 32768
BASE_LO, BASE_HI = 0.2, 0.9
EFF_LO, EFF_HI = -0.06, 0.08
Q_LO, Q_HI = 0.1, 0.6
P_LENGTH = 0.01
POLL = 0.5
OPS_ORDER = ("zoom", "tiles", "check")
OPS_IMAGE_LIMIT = 40                    # engine common.OPS_IMAGE_LIMIT
OP_EFFECT = {"zoom": 0.02, "tiles2": 0.01, "tiles3": 0.015, "check": 0.03}   # mean effect on p at --op-effect 1
OP_SPREAD = 0.03                        # per-(op, row) spread around the mean
ZOOM_TOKENS, ZOOM_FAIL = 48, 0.05
SERVER = "fake:{}"
WORDS = ("look at the figure again and compare the two parts then check which option keeps the same order of marks while the "
         "colour and the outline stay the same so the count of items on each side matters here").split()


# ------------------------------------------------------------------------------------------------ deterministic draws
def h01(*parts):
    """Uniform [0, 1) from a sha256 of the parts."""
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "big") / 2 ** 64


def cond_hash(c):
    return hashlib.sha256(c.encode()).hexdigest()[:16]


def crn_seed(row, sample):
    return zlib.crc32(f"{CRN_SEED}|0|{row}".encode()) % (2 ** 31 - 1024) + int(sample)


def canon_settings(st, knobs):
    """Settings minus null and stock values (null = stock, engine_api §3)."""
    S = knobs["settings"]
    out = {}
    for k, v in sorted((st or {}).items()):
        if v is None or (k in S and v == S[k]["stock"]):
            continue
        out[k] = v
    return out


def effect_key(prefix, st, knobs):
    prefix = prefix if prefix else None
    cs = canon_settings(st, knobs)
    if prefix is None and not cs:
        return "stock"
    return hashlib.sha256(json.dumps([prefix, cs], sort_keys=True).encode()).hexdigest()[:16]


def base_p(row):
    return BASE_LO + (BASE_HI - BASE_LO) * h01("base", row)


def effect(prefix, st, knobs, scale=1.0):
    ek = effect_key(prefix, st, knobs)
    return 0.0 if ek == "stock" else scale * (EFF_LO + (EFF_HI - EFF_LO) * h01("effect", ek))


def p_ok(row, prefix, st, knobs, scale=1.0):
    return min(1.0, max(0.0, base_p(row) + effect(prefix, st, knobs, scale)))


def route_q(cond):
    return Q_LO + (Q_HI - Q_LO) * h01("q", cond_hash(cond))


def route_yes(row, cond):
    return h01("route", row, cond_hash(cond)) < route_q(cond)


def draw(seed, ek, ops, mx):
    """(rng, length, u, base tokens) of one read: Random(f"{seed}|{effect hash}[|{ops signature}]"); the caller goes on drawing
    the reasoning words from the same rng."""
    rng = random.Random(f"{seed}|{ek}" + (f"|{ops_sig(ops)}" if ops else ""))
    length = rng.random() < P_LENGTH
    u = rng.random()
    return rng, length, u, (mx if length else rng.randint(500, 6000))


def op_name(o):
    return f"tiles{int(o.get('n', 2))}" if o["op"] == "tiles" else o["op"]


def ops_sorted(ops):
    """Ops in engine order; an unknown op raises ValueError (the engine's sort fails the same way)."""
    for o in ops:
        if not isinstance(o, dict) or o.get("op") not in OPS_ORDER:
            raise ValueError(f"unknown op {o!r}")
    return sorted(ops, key=lambda o: OPS_ORDER.index(o["op"]))


def ops_sig(ops):
    return hashlib.sha256(json.dumps([op_name(o) for o in ops_sorted(ops)]).encode()).hexdigest()[:16]


def ops_images(row, ops):
    """Images a read with these ops sends (engine common.ops_need, image count): the row's images, + 1 zoom crop, + n x n tiles each."""
    k = len(row.get("image_paths") or [])
    n = k
    for o in ops:
        if o["op"] == "zoom" and k:
            n += 1
        elif o["op"] == "tiles":
            n += int(o.get("n", 2)) ** 2 * k
    return n


def op_effect(o, row, scale=1.0):
    nm = op_name(o)
    return scale * (OP_EFFECT.get(nm, 0.0) + OP_SPREAD * (2 * h01("op", nm, row) - 1))


# ------------------------------------------------------------------------------------------------ canned proposer answers
CONDS = [
    "The question asks to count objects in the image.",
    "The image contains a chart or a plot with labelled axes.",
    "The question asks which of several drawn options continues a pattern.",
    "The question shows more than one picture that must be compared.",
    "The question asks about lines, arrows or paths in the image.",
    "The question asks about the colour or shape of marked items.",
    "The image contains small text or numbers that must be read.",
    "The question asks where objects are placed relative to each other.",
]
PROCS = [
    "Before answering, list every relevant item you can see and check the list once against the image.",
    "First describe the parts of the image the question refers to, then answer from that description.",
    "Compare each option with the image one at a time and note one difference for each before choosing.",
    "Read every label and number in the image first and write them down, then answer.",
    "Trace each line or path from its start to its end and note where it ends before answering.",
    "Check the final answer against the wording of the question and the requested answer format.",
]


CHAT_OPS = [[{"op": "zoom"}], [{"op": "tiles", "n": 2}], [{"op": "tiles", "n": 3}], [{"op": "check"}], [{"op": "zoom"}, {"op": "check"}]]


def canned_chat(key, knobs, chat_ops=0.0):
    """(content, reasoning, tokens) of a deterministic proposer answer for this key. chat_ops: share of the parsable answers that
    also carry ops (drawn from their own hash, so the answers without ops are those of chat_ops = 0)."""
    rng = random.Random(f"chat|{key}")
    u = h01("chatkind", key)
    think = " ".join(rng.choice(WORDS) for _ in range(rng.randint(60, 200)))
    lead = ("The shown questions that are currently wrong share a step that the samples skip or get wrong. "
            "A change aimed at that step should move some of them without hurting the ones that are right.")
    if u < 0.40:
        S = knobs["settings"]
        k = rng.choice(sorted(S))
        v = rng.choice([x for x in S[k]["values"] if x != S[k]["stock"]])
        cand = dict(why=f"Try {k} = {json.dumps(v)} on every question.", condition="always", procedure=None, settings={k: v})
        block = _block(cand, key, chat_ops)
    elif u < 0.90:
        cand = dict(why="A procedure for one kind of question.", condition=rng.choice(CONDS), procedure=rng.choice(PROCS), settings={})
        block = _block(cand, key, chat_ops)
    else:
        block = rng.choice(['```json\n{"why": "broken", "condition": "always", "procedure": null, "settings": {"temperature": 0.8,}\n```',
                            '```json\n{"why": "broken", "condition": "The image contains a chart, "procedure": "Read the axes."}\n```'])
    return f"{lead}\n\n{block}", think, rng.randint(800, 4000)


def _block(cand, key, chat_ops):
    if chat_ops and h01("chatops", key) < chat_ops:
        cand = dict(cand, ops=CHAT_OPS[int(h01("chatopsk", key) * len(CHAT_OPS))])
    return "```json\n" + json.dumps(cand) + "\n```"


# ------------------------------------------------------------------------------------------------ the fake dispatcher
class Fake:
    def __init__(self, a):
        self.a = a
        self.root = a.root
        self.qin, self.qout, self.qcan = (os.path.join(a.root, "q", x) for x in ("in", "out", "cancel"))
        for d in (self.qin, self.qout, self.qcan):
            os.makedirs(d, exist_ok=True)
        rows = a.rows or (os.path.join(a.root, "rows", "learn.jsonl") if os.path.exists(os.path.join(a.root, "rows", "learn.jsonl"))
                          else os.path.join(HERE, "..", "data", "learn.jsonl"))
        self.rows = {r["id"]: r for r in map(json.loads, (l for l in open(rows) if l.strip()))}
        self.knobs = json.load(open(a.knobs or os.path.join(HERE, "knobs.json")))
        self.done, self.batches, self.heap, self.queued, self.mirror_of = set(), {}, [], set(), {}
        self.seq = itertools.count()
        self.route = {}
        self.n_results = 0
        self.t_next = time.time()
        self.jrng = random.Random()             # pacing only; never touches an outcome
        self._index()

    def log(self, *x):
        if not self.a.quiet:
            print(time.strftime("%H:%M:%S"), *x, file=sys.stderr, flush=True)

    def _index(self):
        for f in glob.glob(os.path.join(self.qout, "*.jsonl")):
            for l in open(f):
                try:
                    r = json.loads(l)
                except ValueError:
                    continue
                if "error" not in r:
                    self.done.add(r["key"])
        rc = os.path.join(self.root, "route_cache.jsonl")
        if os.path.exists(rc):
            for l in open(rc):
                try:
                    r = json.loads(l)
                    self.route[(r["row"], r["cond"])] = r
                except ValueError:
                    pass
        self.log(f"index: {len(self.done)} keys done, route cache {len(self.route)}")

    # -- batches
    def scan(self):
        files = sorted(glob.glob(os.path.join(self.qin, "*.jsonl")), key=lambda f: (os.path.getmtime(f), f))
        for f in files:
            batch = os.path.basename(f)[:-6]
            if batch in self.batches:
                continue
            reqs = [json.loads(l) for l in open(f) if l.strip()]
            b = dict(total=len(reqs), finished=set(), cancelled=0, errors=0,
                     closed=os.path.exists(os.path.join(self.qout, batch + ".done")))
            self.batches[batch] = b
            new = 0
            for r in reqs:
                k = r["key"]
                if k in self.done:
                    b["finished"].add(k)
                elif k in self.queued:                       # same key in two batches: the first one answers
                    self.mirror_of.setdefault(k, []).append(batch)
                else:
                    self.queued.add(k)
                    heapq.heappush(self.heap, (int(r.get("priority", 5)), next(self.seq), batch, r))
                    new += 1
            self.log(f"batch {batch}: {len(reqs)} requests, {len(b['finished'])} already done, {new} queued")
            self.maybe_done(batch)

    def maybe_done(self, batch):
        b = self.batches[batch]
        if not b["closed"] and len(b["finished"]) + b["cancelled"] >= b["total"]:
            b["closed"] = True
            tmp = os.path.join(self.qout, batch + ".done.tmp")
            with open(tmp, "w") as f:
                json.dump(dict(total=b["total"], finished=len(b["finished"]), cancelled=b["cancelled"], errors=b["errors"], t=time.time()), f)
            os.replace(tmp, os.path.join(self.qout, batch + ".done"))
            self.log(f"batch {batch} done: {len(b['finished'])} results, {b['cancelled']} cancelled, {b['errors']} errors")

    def pop(self):
        while self.heap:
            pr, _, batch, r = heapq.heappop(self.heap)
            if os.path.exists(os.path.join(self.qcan, batch)):
                self.batches[batch]["cancelled"] += 1
                ms = self.mirror_of.pop(r["key"], [])
                if ms:                  # another batch also asked for this key and was not cancelled: it runs there instead
                    if ms[1:]:
                        self.mirror_of[r["key"]] = ms[1:]
                    heapq.heappush(self.heap, (pr, next(self.seq), ms[0], r))
                else:
                    self.queued.discard(r["key"])
                self.maybe_done(batch)
                continue
            return batch, r
        return None, None

    # -- one request
    def handle(self, r):
        key, mode = r["key"], r.get("mode", "read")
        t_sub = time.time()
        rec = dict(key=key, mode=mode)
        if mode in ("read", "route"):
            rec["row"] = r.get("row")
        if mode == "read":
            rec["sample"] = r.get("sample", 0)
        srv = SERVER.format(r.get("engine", "E"))
        if mode not in ("read", "route", "chat"):
            rec.update(error=f"ValueError: unknown mode {mode!r}", server=srv, t_submit=t_sub, t_done=time.time())
            return rec
        if mode != "chat" and r.get("row") not in self.rows:
            rec.update(error=f"KeyError: row {r.get('row')} not in rows/learn.jsonl", server=srv, t_submit=t_sub, t_done=time.time())
            return rec
        if mode == "route":
            ch = cond_hash(r["condition"])
            rec["cond"] = ch
            hit = self.route.get((r["row"], ch))
            if hit:
                rec.update(yes=hit["yes"], p=hit.get("p"), p_src=hit.get("p_src"), content=hit["content"], cached=True, t_submit=t_sub, t_done=time.time())
                return rec
            yes = route_yes(r["row"], r["condition"])
            u = h01("route_p", f"{r['row']}|{ch}")            # v1 router: p(yes) renormalised over yes/no, route iff p >= .5
            pp = 0.5 + 0.5 * u if yes else 0.5 * u
            cr = dict(row=r["row"], cond=ch, yes=yes, p=pp, p_src="logprob", rv="v1", view=r.get("view"), content="yes" if yes else "no", condition=r["condition"], t=time.time())
            self.route[(r["row"], ch)] = cr
            with open(os.path.join(self.root, "route_cache.jsonl"), "a") as f:
                f.write(json.dumps(cr, ensure_ascii=False) + "\n")
            rec.update(server=srv, t_submit=t_sub, t_done=time.time(), yes=yes, p=pp, p_src="logprob", rv="v1", view=r.get("view"), content=cr["content"], cached=False)
            return rec
        if mode == "chat":
            content, reasoning, toks = canned_chat(key, self.knobs, getattr(self.a, "chat_ops", 0.0) or 0.0)
            rec.update(server=srv, t_submit=t_sub, t_done=time.time(), content=content, reasoning=reasoning, finish="stop", tokens=toks)
            return rec
        return self.read(r, rec, srv, t_sub)

    def read(self, r, rec, srv, t_sub):
        row = self.rows[r["row"]]
        s = int(r.get("sample", 0))
        st = r.get("settings") or {}
        ops = r.get("ops") or []
        if ops:
            try:
                ops_s = ops_sorted(ops)
            except ValueError as e:
                rec.update(error=f"ValueError: {e}"[:300], server=srv, t_submit=t_sub, t_done=time.time())
                return rec
            n_img = ops_images(row, ops_s)
            if n_img > OPS_IMAGE_LIMIT:          # dispatch.py run_ops: the ops are not applied, no content and no ok
                rec.update(op_skipped=f"image_limit: {n_img} images > {OPS_IMAGE_LIMIT}", ops=r["ops"], server=srv, t_submit=t_sub,
                           t_done=time.time(), id=row["id"], bench=row.get("bench") or row["id"].split("/")[0])
                return rec
        ek = effect_key(r.get("prefix"), st, self.knobs)
        p = p_ok(r["row"], r.get("prefix"), st, self.knobs, self.a.effect_scale)
        seed = int(r["seed"]) if r.get("seed") is not None else crn_seed(r["row"], s)
        mx = int(r.get("max_tokens") or READ_MAX)
        for m_ in re.finditer(r"\[fake:([+-]?\d*\.?\d+)@(\w+)\]", r.get("prefix") or ""):   # v5 tests (opt-in): a procedure text with the
            if (row.get("bench") or row["id"].split("/")[0]) == m_.group(2):                    # marker [fake:+0.30@lego] adds 0.30 to p on
                p = min(1.0, max(0.0, p + float(m_.group(1))))                                  # rows of that benchmark only
        ed = r.get("edit")                                   # v5 dry runs (opt-in, so every v4 dry world is unchanged): an edit OBJECT that
        fk = isinstance(ed, dict) and any(str(k_).startswith("fake_") for k_ in ed)   # carries a "fake_" key is its own draw (a null edit =
        if fk:                                               # a fresh sample of the same p); "fake_effect" is added to p, "fake_tokens"
            es = json.dumps(ed, sort_keys=True)              # multiplies the generated tokens
            ek = hashlib.sha256(f"{ek}|{es}".encode()).hexdigest()[:16]
            if ed.get("fake_effect") is not None:
                p = min(1.0, max(0.0, p + float(ed["fake_effect"])))
        rng, length, u, toks = draw(seed, ek, ops, mx)
        if fk and ed.get("fake_tokens") and not length:
            toks = min(mx, int(toks * float(ed["fake_tokens"])))
        if ops:
            sc = getattr(self.a, "op_effect", 1.0)
            sc = 1.0 if sc is None else sc
            zoom_fail = any(o["op"] == "zoom" for o in ops_s) and h01("zoomfail", seed, ek, ops_sig(ops)) < ZOOM_FAIL
            e_in = sum(op_effect(o, r["row"], sc) for o in ops_s if o["op"] == "tiles" or (o["op"] == "zoom" and not zoom_fail))
            e_ck = sum(op_effect(o, r["row"], sc) for o in ops_s if o["op"] == "check")
            p_before = min(1.0, max(0.0, p + e_in))           # the read pass
            p = min(1.0, max(0.0, p_before + e_ck))           # the graded answer (after check)
        ok = (u < p) and not length
        gold = str(row.get("answer", ""))
        ans = None if length else (gold if ok else self.wrong(row))
        content = "" if length else f"Answer: \\boxed{{{ans}}}"
        reasoning = " ".join(rng.choice(WORDS) for _ in range(rng.randint(40, 160)))
        lo = st.get("min_image_tokens") or 0
        hi = st.get("max_image_tokens") or self.knobs["settings"].get("max_image_tokens", {}).get("stock") or 16384
        img = [min(hi, max(lo, 256 + int(3000 * h01("img", pth)))) for pth in row.get("image_paths") or []]
        pt = 200 + len(row.get("prompt", "")) // 4 + sum(img)
        rec.update(server=srv, t_submit=t_sub, t_done=time.time(), content=content, reasoning=reasoning,
                   finish="length" if length else "stop", tokens=toks, prompt_tokens=pt,
                   image_tokens=img, seed=seed, settings=r.get("settings"), prefix=r.get("prefix"), ok=ok, loop=False, grade_error=None,
                   pred=ans, id=row["id"], kind=row.get("kind"), answer=row.get("answer"), options=row.get("options") or [],
                   category=row.get("category"), precision=row.get("precision"), bench=row.get("bench") or row["id"].split("/")[0])
        if ops:
            rec.update(self.op_passes(row, ops_s, toks, mx, pt, img, seed, ek, zoom_fail, u < p_before, gold, length), ops=r["ops"])
        return rec

    def op_passes(self, row, ops, base, mx, pt, img, seed, ek, zoom_fail, ok_before, gold, length):
        """tokens (all passes), prompt_tokens (read pass) and op_trace of a read with ops, in the engine's field layout."""
        has = {o["op"]: o for o in ops}
        trace, total = [], 0
        f = 1.0
        rd = {}
        pt_read = pt
        if "zoom" in has:
            if zoom_fail:
                reply, box = "The region is unclear.", None
            else:
                x0, y0 = round(0.6 * h01("zx", seed, ek), 3), round(0.6 * h01("zy", seed, ek), 3)
                box = [1, x0, y0, round(x0 + 0.3, 3), round(y0 + 0.3, 3)]
                reply = json.dumps(box)
                if img:
                    rd["zoom_tokens"] = max(256, img[0])
                    pt_read += rd["zoom_tokens"]
            trace.append(dict(op="zoom", tokens=ZOOM_TOKENS, prompt_tokens=pt + 60, reply=reply, box=box, zoom_fail=box is None))
            total += ZOOM_TOKENS
            f *= 1.15
        if "tiles" in has:
            n = int(has["tiles"].get("n", 2))
            rd["tiles"] = n * n * len(img)
            pt_read += sum(n * n * max(256, t) for t in img)
            f *= 1 + 0.2 * n
        R = mx if length else min(mx, int(round(base * f)))
        trace.append(dict(op="read", tokens=R, prompt_tokens=pt_read, finish="length" if length else "stop", **rd))
        total += R
        if "check" in has:
            prop = gold if ok_before else self.wrong(row)
            ck = int(round(0.8 * R))
            trace.append(dict(op="check", tokens=ck, prompt_tokens=pt + 80, finish="length" if length else "stop", proposed=prop,
                              ok_before=bool(ok_before) and not length, pred_before=None if length else prop))
            total += ck
        return dict(tokens=total, prompt_tokens=pt_read, op_trace=trace)

    @staticmethod
    def wrong(row):
        g = str(row.get("answer", "")).strip()
        if row.get("kind") == "mc":
            letters = [chr(65 + i) for i in range(max(2, len(row.get("options") or []) or 4))]
            return next((x for x in letters if x != g), "Z")
        if row.get("kind") == "yesno":
            return "no" if g.lower() == "yes" else "yes"
        return "X"

    # -- writing and pacing
    def finish(self, batch, r, rec):
        with open(os.path.join(self.qout, batch + ".jsonl"), "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        k = r["key"]
        self.queued.discard(k)
        b = self.batches[batch]
        if "error" in rec:
            b["errors"] += 1
        else:
            self.done.add(k)
        b["finished"].add(k)
        for mb in self.mirror_of.pop(k, []):          # the key's line is in the first batch's file; the mirror counts it
            self.batches[mb]["finished"].add(k)
            self.maybe_done(mb)
        self.maybe_done(batch)
        self.n_results += 1

    def pace(self):
        if self.a.rate and self.a.rate > 0:
            dt = (1.0 / self.a.rate) * (1.0 + (self.jrng.uniform(-self.a.jitter, self.a.jitter) if self.a.jitter else 0.0))
            self.t_next = max(self.t_next, time.time()) + dt        # no burst credit after an idle spell
            w = self.t_next - time.time()
            if w > 0:
                time.sleep(w)

    def capped(self):
        return self.a.max_results is not None and self.n_results >= self.a.max_results

    def step(self, until):
        """Process queued requests until the queue is empty, the time `until` passes, or --max-results is reached."""
        while not self.capped() and time.time() < until:
            batch, r = self.pop()
            if batch is None:
                return False
            self.pace()
            self.finish(batch, r, self.handle(r))
        return True

    def run(self):
        if self.a.once:
            self.scan()
            self.step(float("inf"))
            self.log(f"--once: {self.n_results} results written, {len(self.heap)} left queued")
            return
        while not self.capped():
            self.scan()
            if not self.step(time.time() + POLL):
                time.sleep(POLL)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True, help="local V4 root: q/in, q/out, q/cancel, route_cache.jsonl live here")
    ap.add_argument("--rows", default=None, help="learn rows (default <root>/rows/learn.jsonl, else ../data/learn.jsonl)")
    ap.add_argument("--knobs", default=None, help="knob menu (default ./knobs.json): stock values and the chat menu")
    ap.add_argument("--rate", type=float, default=200, help="results per second (0 = no pacing)")
    ap.add_argument("--jitter", type=float, default=0.0, help="relative jitter of the interval between results, e.g. 0.5")
    ap.add_argument("--effect-scale", type=float, default=1.0, help="multiplies every read effect (0 = null world)")
    ap.add_argument("--op-effect", type=float, default=1.0, help="multiplies every harness-op effect on p (0 = ops change tokens and draws only)")
    ap.add_argument("--chat-ops", type=float, default=0.0, help="share of parsable canned proposer answers that also carry ops")
    ap.add_argument("--once", action="store_true", help="process what is in q/in now, then exit")
    ap.add_argument("--max-results", type=int, default=None, help="stop after N results (tests: a crash mid-batch)")
    ap.add_argument("--quiet", action="store_true")
    Fake(ap.parse_args(argv)).run()


if __name__ == "__main__":
    main()
