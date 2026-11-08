#!/usr/bin/env python3
"""Resident engine agent: one per card, one Surgeon engine for one family, fed from rsi/res_queue.py. Contract: vesta/resident_contract_1107.md. Design: vesta/resident_design_1107.md.

Two processes per card. `run` (no torch, no GPU) stays up and is what gpu bg starts; it restarts `engine`, runs write units on the freed card and changes family. `engine` holds the Surgeon and does the loop:
heartbeat, claim, feed rows into the running engine, step, write answers. A process exit is the only way to get the card's memory back, which is why a write or a family change ends `engine` rather than freeing in place.

Engine settings: hboost.engine_up with vh.TAIL and "--samples 2 --model <family> --pool IID", the same words job-batch runs with. Seeds: L.crn_seeds. Answers: a.r<tag>.jsonl, .partial, .time as hboost.run_item, with one change:
`.time` secs are this unit's share of the engine's time (each step's seconds split over live requests), so sum(n) / sum(secs) stays rows per card-hour when units overlap; wall has the plain clock.

Units (res_queue fields, plus):
  eval      rows, out, samples, engine (edit list), size ({"max": pixels, "min": pixels} or null). run_item's item.
  generate  (also "gen") req (json list of {id, system, user}), out. Proposer prompts as run_item: n=1, temperature .7, max_tokens 12000, effort medium, no seeds. Shares the engine; editid must be the loaded one like any unit.
  write     cmd (shell command, cwd code-xf, e.g. "python -u rsi/mm_gate.py ..."), out (the file it must leave), optional outs (all files it must leave). Runs on the drained card with no engine, then the engine reloads.
  Optional on any unit: stage, a label for the card-minute tables (default "prio<n>").

Exit codes of `engine`: 0 stop or idle limit, 75 run the write in ctl/<card>.write.json, 76 reload to the family in ctl/<card>.next.json, 77 revert fingerprint mismatch, 78 engine error, 79 stuck. 77 to 79: units are given back and `run` restarts it.
Log: <res>/log/<card>.jsonl, one json per event, summed by `res_agent.py sum <res>`.
Stop: file <res>/STOP or <res>/ctl/<card>.stop: drains what it holds, then exits 0.
"""
import argparse, collections, json, math, os, random, shlex, subprocess, sys, threading, time, traceback

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODE not in sys.path:
    sys.path.insert(0, CODE)
from rsi.res_queue import Queue, jdump, jload, write_atomic   # noqa: E402

EX_WRITE, EX_RELOAD, EX_MISMATCH, EX_ENGINE, EX_STUCK = 75, 76, 77, 78, 79
BEAT_S = 30                  # heartbeat, contract
STUCK_S = 900                # no step finished in this long with requests in the engine: leave
CLAIM_EVERY = 2.             # seconds between claims that found nothing
EXPIRE_EVERY = 60.           # an idle agent also clears other agents' dead leases
RULE_EVERY = float(os.environ.get("RES_RULE_EVERY", 30))     # family rule, seconds
TIME_EVERY = 60.             # card-time record in the log
FEED_BURST = 4               # rows rendered between two steps while the engine runs: rendering is done on this thread and must not stall the others
EXIT_S = 15.                 # process exit until the card is free, part of the reload cost; overwritten by nothing yet
FRESH_S = 120                # a heartbeat younger than this counts as a card on its family
CARD_FREE_MB = 8000          # same mark as cw/card_worker.py
LAUNCH_FAILS = 3             # engines dying this many times in a row within CRASH_S: `run` gives up and leaves it to the supervisor
CRASH_S = 120


class Log:
    def __init__(self, res, card):
        os.makedirs(os.path.join(res, "log"), exist_ok=True)
        self.f, self.card = os.path.join(res, "log", f"{card}.jsonl"), card

    def __call__(self, ev, **kw):
        line = json.dumps({**dict(ev=ev, t=round(time.time(), 3), card=self.card, pid=os.getpid()), **kw}, ensure_ascii=False, default=str)
        fd = os.open(self.f, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            os.write(fd, (line + "\n").encode())
        finally:
            os.close(fd)
        print(time.strftime("%H:%M:%S"), line[:400], flush=True)


# -------------------------------------------------- family rule
def _span(backlog, cards):
    """(families with work and no card, time until the slowest family with a card is done)."""
    uncovered, m = 0, 0.
    for f, b in backlog.items():
        if b <= 0:
            continue
        c = cards.get(f, 0)
        if c <= 0:
            uncovered += 1
        else:
            m = max(m, b / c)
    return uncovered, m


def family_choice(me, backlog, cards, cost):
    """Which family this card should move to, by the design's rule: only when the backlog time saved exceeds the reload cost. Returns (family or None, seconds saved).
    Time saved is the fall in the time until all backlog is done, moving one card from `me` to f, the card's share of f starting `cost` late. Work with no card at all counts as saving more than any cost. A card never leaves work of its own family with no card."""
    un0, m0 = _span(backlog, cards)
    best, gain = None, 0.
    for f, b in sorted(backlog.items()):
        if f == me or b <= 0:
            continue
        after = dict(cards)
        after[me] = after.get(me, 0) - 1
        after[f] = after.get(f, 0) + 1
        if after[me] <= 0 and backlog.get(me, 0) > 0:
            continue
        un1, m1 = _span(dict(backlog, **{f: b + cost}), after)
        saved = math.inf if un1 < un0 else m0 - m1
        if saved > gain:
            best, gain = f, saved
    return (best, gain) if gain > cost else (None, gain)


def card_counts(q, me=None, target=None):
    """Cards per family from fresh heartbeats. An agent moving counts on its target. Only targets of lower-numbered cards count against this card, so two cards racing for one family do not both give way."""
    out = collections.Counter()
    for n in os.listdir(q.p("agents")):
        if not n.endswith(".json"):
            continue
        try:
            h = jload(q.p("agents", n))
        except (OSError, ValueError):
            continue
        if time.time() - h.get("t", 0) > FRESH_S or h.get("family") is None:
            continue
        other = n[:-5]
        moving = h.get("target") and (me is None or _rank(other) < _rank(me))
        out[h["target"] if moving else h["family"]] += 1
    return dict(out)


def _rank(card):
    try:
        return int(card)
    except (TypeError, ValueError):
        return 10 ** 6


def stopping(res, card):
    return os.path.exists(os.path.join(res, "STOP")) or os.path.exists(os.path.join(res, "ctl", f"{card}.stop"))


# -------------------------------------------------- one unit in the engine
class Run:
    def __init__(self, unit, items):
        self.u, self.items = unit, items      # items: (row dict or request dict) in file order
        self.todo = collections.deque(range(len(items)))
        self.live = {}                        # feed key -> index
        self.res = {}                         # index -> records (eval) or result dict (generate)
        self.card_s = self.render_s = 0.
        self.t_first = self.t_last = None
        self.joined_with = None               # units already in the engine when this one's first row went in
        self.feed_live0 = None
        self.kind = "eval" if unit.get("kind", "eval") == "eval" else "generate"
        self.stage = unit.get("stage") or f"prio{unit.get('prio', 9)}"
        self.t_claim = time.time()

    def finished(self):
        return not self.todo and not self.live


def load_items(u):
    if u.get("kind", "eval") == "eval":
        probs = [json.loads(l) for l in open(u["rows"]) if l.strip()]
        keys = u.get("keys")
        if keys is not None and [p["id"] for p in probs] != list(keys):
            raise ValueError(f"rows of {u['unit']} are not its keys")
        return probs
    return json.load(open(u["req"]))


# -------------------------------------------------- the engine process
class Engine:
    def __init__(self, a, q, log, up):
        self.a, self.q, self.log, self.up = a, q, log, up
        self.res, self.card, self.family = q.res, str(a.card), a.family
        self.S = self.L = self.args = None
        self.cold = None
        self.editid, self.handle = "none", None
        self.active, self.keys = [], {}        # Runs on the loaded edit; feed key -> Run
        self.held = []                         # claimed, waiting for the engine to drain
        self.target = None                     # family this card leaves for once drained
        self.state, self.last_step = "load", time.time()
        self.last_claim = self.last_expire = self.last_rule = 0.
        self.idle_t0 = None
        self.mark_t = time.time()
        self.window = collections.Counter()    # seconds since the last time record: idle, load, switch, drain, busy:<stage>
        self.window_t = time.time()
        self.occ = self.occ_s = 0.             # time-weighted requests in the engine
        self.ppid = os.getppid()
        self.lookahead = 96
        self.stop_beat = threading.Event()
        self.switches = 0

    # ---------------- heartbeat
    def beat_state(self):
        return dict(card=self.card, family=self.family, editid=self.editid, pid=os.getpid(), state=self.state, target=self.target, units=[r.u["unit"] for r in self.active], held=[u["unit"] for u in self.held],
                    rows_left=sum(len(r.todo) + len(r.live) for r in self.active), last_step=self.last_step)

    def _beat_loop(self):
        while not self.stop_beat.wait(BEAT_S):
            try:
                if self.state == "busy" and time.time() - self.last_step > STUCK_S:
                    self.log("stuck", since=round(time.time() - self.last_step), units=[r.u["unit"] for r in self.active])
                    os._exit(EX_STUCK)
                if os.getppid() != self.ppid:     # `run` is gone: nothing would restart this or reuse the card's lease
                    self.log("orphan")
                    os._exit(EX_ENGINE)
                self.q.heartbeat(self.card, self.beat_state())
            except Exception as e:
                print("heartbeat failed", repr(e), flush=True)

    # ---------------- card time
    def spend(self, what, s):
        self.window[what] += s

    def flush_time(self, force=False):
        now = time.time()
        if force or now - self.window_t >= TIME_EVERY:
            self.log("time", dt=round(now - self.window_t, 2), family=self.family, by={k: round(v, 2) for k, v in self.window.items()}, mean_live=round(self.occ / self.occ_s, 1) if self.occ_s else 0.)
            self.window, self.window_t, self.occ, self.occ_s = collections.Counter(), now, 0., 0.

    # ---------------- start
    def start(self):
        threading.Thread(target=self._beat_loop, daemon=True).start()
        self.q.heartbeat(self.card, self.beat_state())
        t0 = time.time()
        self.S, self.L, self.args, self.cold = self.up(self.family, self.a)
        self.args.effort = "medium"
        self.lookahead = self.a.lookahead or (int(self.args.max_num_seqs) + 32)
        self.spend("load", time.time() - t0)
        self.log("engine_up", family=self.family, cold=round(self.cold, 1), lookahead=self.lookahead, surgery=getattr(sys.modules.get("surgery"), "__file__", None))
        if self.a.write_t0:
            self.log("write_cost", unit=self.a.write_unit, total_s=round(time.time() - self.a.write_t0, 1), reload_s=round(self.cold, 1))
        if self.a.verify_full:
            t = time.time()
            self.fp_full = self.S.model_fingerprint()
            self.log("full_fingerprint", s=round(time.time() - t, 2))
        self.state, self.last_step = "idle", time.time()
        self.idle_t0 = time.time()

    # ---------------- leaving
    def give_back(self, why, units=None):
        for u in (units if units is not None else [r.u for r in self.active] + self.held):
            try:
                self.q.release(u, why)
            except Exception:
                traceback.print_exc()

    def leave(self, code, **kw):
        self.flush_time(True)
        self.log("engine_exit", code=code, family=self.family, **kw)
        self.stop_beat.set()
        sys.stdout.flush()
        os._exit(code)          # no vLLM shutdown path: the process ending is what frees the card

    def engine_error(self, e):
        why = f"{type(e).__name__}: {e}"
        self.log("engine_error", why=why[:400], units=[r.u["unit"] for r in self.active])
        traceback.print_exc()
        for r in self.active:      # one step ran them all: each goes through fail, which splits on OOM
            try:
                self.q.fail(r.u, why)
            except Exception:
                traceback.print_exc()
        self.give_back("engine error on this card", self.held)
        self.leave(EX_ENGINE, why=why[:200])

    # ---------------- claims
    def claim_more(self):
        if self.held or self.target or stopping(self.res, self.card):
            return
        if sum(len(r.todo) for r in self.active) >= self.a.claim_ahead:
            return
        if time.time() - self.last_claim < CLAIM_EVERY:
            return
        for u in self.q.claim(self.card, self.family, self.editid, n=1):
            self.end_idle(u)
            self.route(u)
            return
        self.last_claim = time.time()

    def end_idle(self, u):
        if self.idle_t0 is not None:
            s = time.time() - self.idle_t0
            self.log("idle", s=round(s, 2), t0=round(self.idle_t0, 2), next_unit=u["unit"], next_stage=u.get("stage") or f"prio{u.get('prio', 9)}", next_prio=u.get("prio"))
            self.spend("idle", s)
            self.idle_t0 = None

    def route(self, u):
        kind = u.get("kind", "eval")
        self.log("claim", unit=u["unit"], kind=kind, prio=u.get("prio"), stage=u.get("stage"), rows=u.get("n"), editid=u.get("editid", "")[:12], loaded=self.editid[:12], live=self.S.unfinished())
        if kind == "write":
            self.held.append(u)
        elif kind not in ("eval", "generate", "gen"):
            self.q.fail(u, f"unknown kind {kind}")
        elif u.get("editid", "none") != self.editid:
            self.held.append(u)
        else:
            self.begin(u)

    def begin(self, u):
        if u.get("kind", "eval") == "eval":
            if os.path.exists(u["out"]):        # as run_item: a copy another site finished first
                self.log("unit_skip", unit=u["unit"], out=u["out"])
                self.q.done(u, dict(skipped=True, card=self.card))
                return
            m = u.get("model")
            if m and not u.get("size") and os.path.realpath(m) != os.path.realpath(self.family):
                self.q.fail(u, f"model {m} is not the loaded family {self.family} and the unit gives no size")
                return
        try:
            self.active.append(Run(u, load_items(u)))
        except Exception as e:
            self.q.fail(u, f"{type(e).__name__}: {e}")
            return
        self.active.sort(key=lambda r: (r.u.get("prio", 9), r.t_claim))

    # ---------------- feeding
    def feed(self):
        burst = FEED_BURST if self.S.unfinished() else 10 ** 6
        for r in list(self.active):
            while r.todo and burst > 0 and self.S.unfinished() < self.lookahead:
                burst -= 1
                if not self.feed_row(r):
                    break

    def feed_row(self, r):
        i = r.todo[0]
        t0 = time.time()
        if r.t_first is None:
            r.t_first = t0
            r.joined_with = [x.u["unit"] for x in self.active if x is not r and x.live]
            r.feed_live0 = self.S.unfinished()
            self.log("feed_start", unit=r.u["unit"], rows=len(r.items), live=r.feed_live0, joined_with=r.joined_with, size=r.u.get("size"))
        try:
            key = self.feed_one(r, i)
        except Exception as e:
            self.log("feed_failed", unit=r.u["unit"], row=i, why=f"{type(e).__name__}: {e}"[:300])
            self.drop(r, f"{type(e).__name__}: {e}")
            return False
        r.todo.popleft()
        r.live[key] = i
        self.keys[key] = r
        dt = time.time() - t0
        r.render_s += dt
        self.spend("busy:" + r.stage, dt)
        return True

    def feed_one(self, r, i):
        S, L, a, x = self.S, self.L, self.args, r.items[i]
        if r.kind == "eval":
            p = L.mm_eval.image_message(x["prompt"], x["image_paths"]) if x.get("image_paths") else x["prompt"]
            return S.feed([p], n=int(r.u.get("samples") or a.samples), sampling=L.SAMPLING, max_tokens=a.budget, reasoning_effort=a.effort, seeds=L.crn_seeds([x]), size=r.u.get("size"))[0]
        m = [{"role": "system", "content": x["system"]}, {"role": "user", "content": x["user"]}]
        return S.feed([m], n=1, sampling=dict(L.SAMPLING, temperature=.7), max_tokens=12000, reasoning_effort="medium")[0]

    def drop(self, r, why):
        """A unit that can't go on while the engine can: its requests leave, it goes through fail."""
        self.S.abort(list(r.live))
        for k in r.live:
            self.keys.pop(k, None)
        r.live.clear()
        r.todo.clear()
        self.active.remove(r)
        self.q.fail(r.u, why)
        self.log("unit_failed", unit=r.u["unit"], why=str(why)[:300])

    # ---------------- stepping
    def step(self):
        live = collections.Counter()
        for r in self.active:
            live[r.u["unit"]] = len(r.live)
        n = sum(live.values())
        draining = bool(self.held or self.target)
        t0 = time.time()
        try:
            got = self.S.pump()
        except Exception as e:
            self.engine_error(e)
            return
        now = time.time()
        dt = now - t0
        self.last_step = now
        self.occ += self.S.unfinished() * dt
        self.occ_s += dt
        if draining:
            self.spend("drain", dt)
        for r in self.active:
            if n:
                s = dt * live[r.u["unit"]] / n
                r.card_s += s
                self.spend("busy:" + r.stage, s)
        for key, res in got:
            r = self.keys.pop(key, None)
            if r is None:
                continue
            i = r.live.pop(key)
            try:
                self.row_done(r, i, res)
            except Exception as e:
                traceback.print_exc()
                self.drop(r, f"{type(e).__name__}: {e}")
                continue
            if r.finished():
                self.unit_done(r)

    def row_done(self, r, i, res):
        if r.kind == "eval":
            one = []
            self.L.mm_eval._collect(one, [r.items[i]], [res])
            r.res[i] = [{k: v for k, v in x.items() if k != "seq"} for x in one]
            with open(r.u["out"] + ".partial", "a") as f:
                for x in r.res[i]:
                    f.write(json.dumps(x, ensure_ascii=False) + "\n")
        else:
            s = res["samples"][0]
            q = r.items[i]
            r.res[i] = dict(id=q["id"], content=s["content"], reasoning=s["reasoning"], finish=s["finish_reason"], tokens=s["num_tokens"])
        r.t_last = time.time()

    def unit_done(self, r):
        u, out = r.u, r.u["out"]
        try:
            wall = time.time() - (r.t_first or r.t_claim)
            secs = r.card_s + r.render_s
            if r.kind == "eval":
                R = [x for i in range(len(r.items)) for x in r.res[i]]
                tmp = f"{out}.{os.getpid()}.tmp"
                with open(tmp, "w") as f:
                    for x in R:
                        f.write(json.dumps(x, ensure_ascii=False) + "\n")
                first = not os.path.exists(out)
                os.replace(tmp, out if first else out + ".later")      # the first to finish stands; a later copy is kept beside it
                tokens = sum(x["tokens"] for x in R) / max(1, len(R))
                jdump(out + (".time" if first else ".later.time"), dict(secs=secs, n=len(r.items), model=u.get("model"), tokens=tokens, wall=wall, card=self.card, render=r.render_s, size=u.get("size")))
            else:
                R = [r.res[i] for i in range(len(r.items))]
                tmp = f"{out}.{os.getpid()}.tmp"
                json.dump(R, open(tmp, "w"), ensure_ascii=False)
                os.replace(tmp, out)
                tokens = sum(x["tokens"] for x in R) / max(1, len(R))
                jdump(out + ".time", dict(secs=secs, n=len(R), wall=wall, card=self.card))
        except Exception as e:
            traceback.print_exc()
            self.active.remove(r)
            self.q.fail(u, f"{type(e).__name__}: {e}")
            return
        n = len(r.items)
        stats = dict(rows=n, secs=round(secs, 2), wall=round(wall, 2), rows_h=round(3600 * n / max(secs, 1e-6)), rows_h_wall=round(3600 * n / max(wall, 1e-6)), render=round(r.render_s, 2), tokens=round(tokens, 1), joined_with=r.joined_with, card=self.card, stage=r.stage, family=self.family, editid=self.editid, kind=r.kind)
        self.active.remove(r)
        self.q.done(u, stats)
        self.log("unit", unit=u["unit"], prio=u.get("prio"), t_first=round(r.t_first or 0, 2), **stats)

    # ---------------- edit switch, write
    def after_drain(self):
        """The engine is empty and a held unit waits."""
        u = self.held.pop(0)
        t0 = time.time()
        write = u.get("kind") == "write"
        want = "none" if write else u.get("editid", "none")
        revert_s = apply_s = 0.
        if self.handle is not None:
            t = time.time()
            try:
                self.S.revert(self.handle)     # bit-exact blocks and fingerprint of what the edit touched; raises if not
            except Exception as e:
                self.log("revert_mismatch", why=f"{type(e).__name__}: {e}"[:400], editid=self.editid)
                self.give_back("revert fingerprint mismatch on this card", [u] + self.held)
                self.leave(EX_MISMATCH, why="revert")
            revert_s = time.time() - t
            self.handle, self.editid = None, "none"
        if write:
            self.spend("switch", time.time() - t0)
            self.leave_for_write(u)
        eng = u.get("engine")
        if (want == "none") != (not eng):
            self.q.fail(u, f"editid {want} and engine {'with' if eng else 'without'} edits do not fit")
            return
        if eng:
            t = time.time()
            try:
                self.handle = self.S.apply(eng)
            except Exception as e:
                self.log("apply_failed", unit=u["unit"], why=f"{type(e).__name__}: {e}"[:400])
                self.q.fail(u, f"apply: {type(e).__name__}: {e}")
                self.editid = "none"
                return
            apply_s = time.time() - t
        self.editid = want
        self.switches += 1
        fp_s = None
        if self.a.verify_full and self.handle is None:
            t = time.time()
            if self.S.model_fingerprint() != self.fp_full:
                self.log("full_fingerprint_differs")
                self.give_back("full fingerprint differs", [u] + self.held)
                self.leave(EX_MISMATCH, why="full fingerprint")
            fp_s = round(time.time() - t, 2)
        self.spend("switch", time.time() - t0)
        self.log("switch", to=want[:12], unit=u["unit"], revert_s=round(revert_s, 3), apply_s=round(apply_s, 3), parts=len(eng or []), full_fp_s=fp_s)
        self.begin(u)

    def leave_for_write(self, u):
        """The lease passes to `run` before this process ends: expire() would otherwise see a dead pid and requeue a write that is about to run."""
        self.stop_beat.set()
        os.makedirs(os.path.join(self.res, "ctl"), exist_ok=True)
        for x in self.held:
            self.q.release(x, "card leaves for a write")
        self.held = []
        u = dict(u, pid=self.ppid, lease_t=time.time())
        jdump(self.q.p("claimed", f"{u['unit']}.json"), u)
        self.q.heartbeat(self.card, dict(self.beat_state(), pid=self.ppid, state="write", units=[u["unit"]]))
        jdump(os.path.join(self.res, "ctl", f"{self.card}.write.json"), dict(unit=u, t0=time.time()))
        self.leave(EX_WRITE, unit=u["unit"])

    # ---------------- family rule
    def family_rule(self):
        if self.target or self.held or time.time() - self.last_rule < RULE_EVERY or stopping(self.res, self.card):
            return
        self.last_rule = time.time()
        backlog = self.q.backlog()
        cost = (self.cold or 180.) + EXIT_S
        cards = card_counts(self.q)
        cards.setdefault(self.family, 1)
        f, saved = family_choice(self.family, backlog, cards, cost)
        if not f:
            return
        self.target = f
        self.q.heartbeat(self.card, self.beat_state())      # others see the move, then look again
        time.sleep(1 + random.random())
        cards = card_counts(self.q, me=self.card)
        cards.setdefault(self.family, 1)
        again, saved = family_choice(self.family, self.q.backlog(), cards, cost)
        self.log("family_rule", to=f, again=again, saved=saved if saved != math.inf else "no card", cost=round(cost, 1), cards=cards, backlog={k: round(v) for k, v in backlog.items()})
        self.target = f if again == f else None
        self.q.heartbeat(self.card, self.beat_state())

    # ---------------- loop
    def loop(self):
        while True:
            self.claim_more()
            self.feed()
            if self.S.unfinished():
                if self.state != "busy":
                    self.state, self.last_step = "busy", time.time()
                self.step()
                self.flush_time()
                self.family_rule()
                continue
            if any(r.todo for r in self.active):
                continue
            for r in list(self.active):
                if r.finished():           # a unit with no rows, or one whose last row was aborted
                    self.unit_done(r)
            if self.active:
                continue
            if self.held:
                self.after_drain()
                continue
            if self.target:
                jdump(self.ctl("next"), dict(family=self.target, t0=time.time()))
                self.flush_idle()
                self.leave(EX_RELOAD, to=self.target)
            if stopping(self.res, self.card):
                self.flush_idle()
                self.leave(0, why="stop file")
            self.wait_idle()

    def ctl(self, what):
        os.makedirs(os.path.join(self.res, "ctl"), exist_ok=True)
        return os.path.join(self.res, "ctl", f"{self.card}.{what}.json")

    def flush_idle(self):
        if self.idle_t0 is not None:
            s = time.time() - self.idle_t0
            self.log("idle", s=round(s, 2), t0=round(self.idle_t0, 2), next_unit=None, next_stage=None, next_prio=None)
            self.spend("idle", s)
            self.idle_t0 = None

    def wait_idle(self):
        now = time.time()
        if self.idle_t0 is None:
            self.idle_t0 = now
            self.state = "idle"
        if now - self.last_expire > EXPIRE_EVERY:
            self.last_expire = now
            self.q.expire()
        self.family_rule()
        self.flush_time()
        if self.a.exit_idle and now - self.idle_t0 > self.a.exit_idle:
            self.flush_idle()
            self.leave(0, why="idle limit")
        time.sleep(.5)


def real_up(family, a):
    """hboost.engine_up with the words job-batch uses. Surgery may come from a test copy: imported first, every later `from surgery import` gets that module."""
    if a.surgery_dir:
        sys.path.insert(0, a.surgery_dir)
        import surgery  # noqa: F401
    from rsi import hboost as h
    from rsi import vharness as vh
    argv = f"--samples 2 --model {family} --pool {h.IID} {vh.TAIL.format(lib=vh.DIRLIB)}".split() + shlex.split(a.engine_extra)
    S, L, args, cold = h.engine_up(argv, os.path.join(a.res, "engines", str(a.card)))
    return S, L, args, cold


def load_up(spec):
    if not spec:
        return real_up
    mod, fn = spec.split(":")
    import importlib
    return getattr(importlib.import_module(mod), fn)


def cmd_engine(a):
    q = Queue(a.res, rates=json.loads(a.rates) if a.rates else None)
    log = Log(q.res, a.card)
    a.res = q.res
    e = Engine(a, q, log, load_up(a.up))
    e.start()
    try:
        e.loop()
    except SystemExit:
        raise
    except BaseException as x:
        log("engine_crash", why=f"{type(x).__name__}: {x}"[:400])
        traceback.print_exc()
        e.give_back("agent crashed")
        e.leave(EX_ENGINE, why="crash")


# -------------------------------------------------- the process gpu bg starts
def card_used(card):
    try:
        return int(subprocess.run(f"nvidia-smi -i {card} --query-gpu=memory.used --format=csv,noheader,nounits", shell=True, capture_output=True, text=True, timeout=60).stdout.strip())
    except Exception:
        return None


def wait_card_free(card, limit=120):
    t0 = time.time()
    while time.time() - t0 < limit:
        u = card_used(card)
        if u is None or u < CARD_FREE_MB:
            break
        time.sleep(2)
    return time.time() - t0


def run_write(a, q, log, card, family, stop_beat_state):
    x = jload(os.path.join(q.res, "ctl", f"{card}.write.json"))
    u, t0 = x["unit"], x["t0"]
    free_s = wait_card_free(card)
    t1 = time.time()
    lf = os.path.join(q.res, "log", f"write_{u['unit']}.log")
    done = threading.Event()

    def beat():
        while not done.wait(BEAT_S):
            q.heartbeat(card, dict(stop_beat_state, pid=os.getpid(), state="write", units=[u["unit"]]))
    threading.Thread(target=beat, daemon=True).start()
    q.heartbeat(card, dict(stop_beat_state, pid=os.getpid(), state="write", units=[u["unit"]]))
    with open(lf, "ab") as o:
        rc = subprocess.run(["bash", "-c", u["cmd"]], stdout=o, stderr=subprocess.STDOUT, cwd=a.cwd or CODE).returncode
    done.set()
    write_s = time.time() - t1
    ok = rc == 0 and all(os.path.exists(f) for f in u.get("outs") or [u["out"]])
    stats = dict(rc=rc, write_s=round(write_s, 1), card_free_s=round(free_s, 1), card=card, stage=u.get("stage") or f"prio{u.get('prio', 9)}", family=family, kind="write")
    if ok:
        jdump(u["out"] + ".time", dict(secs=write_s, n=1, card=card))
        q.done(u, stats)
    else:
        q.fail(u, f"write rc {rc}, out {'there' if os.path.exists(u['out']) else 'missing'}, log {lf}")
    log("write", unit=u["unit"], ok=ok, out=u["out"], log=lf, **stats)
    return u["unit"], t0


def cmd_run(a):
    a.card = str(a.card if a.card is not None else os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
    q = Queue(a.res, rates=json.loads(a.rates) if a.rates else None)
    log = Log(q.res, a.card)
    os.makedirs(q.p("ctl"), exist_ok=True)
    family, fails, write_unit, write_t0 = a.family, 0, None, None
    while True:
        cmd = [sys.executable, "-u", os.path.abspath(__file__), "engine", "--res", a.res, "--family", family, "--card", a.card, "--claim-ahead", str(a.claim_ahead), "--lookahead", str(a.lookahead), "--exit-idle", str(a.exit_idle),
               f"--engine-extra={a.engine_extra}", f"--surgery-dir={a.surgery_dir}", f"--rates={a.rates}", f"--up={a.up}", f"--write-unit={write_unit or ''}", "--write-t0", str(write_t0 or 0)]
        if a.verify_full:
            cmd.append("--verify-full")
        t0 = time.time()
        log("engine_start", family=family)
        rc = subprocess.run(cmd, cwd=a.cwd or CODE).returncode
        ran = time.time() - t0
        write_unit = write_t0 = None
        if rc == 0:
            log("run_end", ran=round(ran, 1))
            return
        state = dict(card=a.card, family=family, editid="none", state="between")
        if rc == EX_WRITE:
            write_unit, write_t0 = run_write(a, q, log, a.card, family, state)
            fails = 0
            continue
        if rc == EX_RELOAD:
            old, family = family, jload(os.path.join(q.res, "ctl", f"{a.card}.next.json"))["family"]
            log("reload", old=old, new=family, card_free_s=round(wait_card_free(a.card), 1))
            fails = 0
            continue
        # a crash, a mismatch or a stuck engine: what it still holds goes back, the next engine loads from the checkpoint
        q.heartbeat(a.card, dict(state, pid=os.getpid()))
        back = q.expire()
        fails = fails + 1 if ran < CRASH_S else 0
        log("engine_died", rc=rc, ran=round(ran, 1), requeued=back, fails=fails)
        if fails >= LAUNCH_FAILS:
            log("run_end", why="engine keeps dying")
            sys.exit(1)
        wait_card_free(a.card)


# -------------------------------------------------- reading the logs
def cmd_sum(res):
    """Card-minutes per stage, idle by the stage that came next, switches and writes, from <res>/log/*.jsonl."""
    res = res if os.path.basename(res.rstrip("/")) == "res" else os.path.join(res, "res")
    busy, other, idle = collections.Counter(), collections.Counter(), collections.Counter()
    units, sw, wr, cold = [], [], [], []
    for n in sorted(os.listdir(os.path.join(res, "log"))):
        if not n.endswith(".jsonl"):
            continue
        for l in open(os.path.join(res, "log", n)):
            try:
                r = json.loads(l)
            except ValueError:
                continue
            e = r["ev"]
            if e == "time":
                for k, v in r["by"].items():
                    if k.startswith("busy:"):
                        busy[k[5:]] += v
                    elif k != "idle":
                        other[k] += v
            elif e == "idle":
                idle[r.get("next_stage") or "(none after)"] += r["s"]
            elif e == "unit":
                units.append(r)
            elif e == "switch":
                sw.append(r)
            elif e == "write":
                wr.append(r)
            elif e == "engine_up":
                cold.append(r["cold"])
    m = lambda x: round(x / 60, 2)
    print("busy card-min per stage:", {k: m(v) for k, v in sorted(busy.items())})
    print("idle card-min, by the stage that came next:", {k: m(v) for k, v in sorted(idle.items())})
    print("other card-min:", {k: m(v) for k, v in sorted(other.items())}, "(drain is inside busy)")
    if units:
        rows, secs = sum(u["rows"] for u in units), sum(u["secs"] for u in units)
        print(f"units {len(units)}, rows {rows}, rows/card-hour {3600 * rows / max(secs, 1e-6):.0f}")
    if sw:
        print(f"switches {len(sw)}, apply s mean {sum(x['apply_s'] for x in sw) / len(sw):.3f} max {max(x['apply_s'] for x in sw):.3f}, revert s mean {sum(x['revert_s'] for x in sw) / len(sw):.3f} max {max(x['revert_s'] for x in sw):.3f}")
    if wr:
        print(f"writes {len(wr)}, seconds {[x['write_s'] for x in wr]}, ok {[x['ok'] for x in wr]}")
    if cold:
        print(f"engine loads {len(cold)}, seconds {cold}")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "sum":
        return cmd_sum(sys.argv[2])
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "engine"])
    ap.add_argument("--res", required=True, help="<run> or <run>/res")
    ap.add_argument("--family", required=True, help="model dir of the first engine")
    ap.add_argument("--card", default=None)
    ap.add_argument("--claim-ahead", type=int, default=32, help="claim when fewer rows than this are not yet in the engine")
    ap.add_argument("--lookahead", type=int, default=0, help="requests kept in the engine; default max_num_seqs + 32, so the waiting queue never runs dry while rows remain, as in a batch")
    ap.add_argument("--exit-idle", type=float, default=0., help="exit after this many idle seconds; 0 stays")
    ap.add_argument("--engine-extra", default="", help="more words for engine_up, after TAIL (argparse: the last value wins)")
    ap.add_argument("--surgery-dir", default=os.environ.get("RES_SURGERY_DIR", ""), help="a test copy of engine/surgery.py")
    ap.add_argument("--rates", default="", help="res_queue rates table as json, for backlog()")
    ap.add_argument("--up", default="", help="module:function in place of the real engine, for tests")
    ap.add_argument("--cwd", default="")
    ap.add_argument("--verify-full", action="store_true", help="fingerprint of all weights after load and after each switch back to no edit; logs its seconds")
    ap.add_argument("--write-unit", default="")
    ap.add_argument("--write-t0", type=float, default=0.)
    a = ap.parse_args()
    if a.cmd == "run":
        cmd_run(a)
    else:
        a.card = str(a.card)
        cmd_engine(a)


if __name__ == "__main__":
    main()
