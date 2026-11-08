#!/usr/bin/env python3
"""File queue, ledger and leases for the resident engines. Contract: vesta/resident_contract_1107.md. Standard library only.

Atomicity. A claim is rename(queue/<name>, claimed/<unit>.json): one process wins, the others get ENOENT. enqueue holds a short O_EXCL lock so "is this key new?" and "record it" are one step. The lock
is released by rename, and ledger lines are one O_APPEND write. Nothing is ever deleted: files only move or are replaced by a newer copy of themselves.

What an agent does with a unit from claim(): run unit["rows"], append traces to unit["out"] + ".partial", write unit["out"] and unit["out"] + ".time" by rename (as run_item does), then done(unit, stats) or
fail(unit, why). Never use a path other than the unit's. On a retry or a split unit["rows"] holds only the rows still missing and unit["out"] is <final>.a2, <final>.s0 and so on; done() builds <final> once
every row is there, so an agent treats all units alike. A batch from claim(n > 1) is always one editid and one kind.

Ledger key: (weights id, prompt hash, row, seed).
  weights id   family + "|" + editid.
  prompt hash  sha1 of json [prompt text, image ids, size], first 16 hex: what the model is sent, and nothing about grading. An image id is the path, except a program view (/views/ in the path), where it is
               the "pixels" sha from the note render_view leaves beside the png: a view's path holds the member id, so two members with the same crop would otherwise never match. img_sha256 in a rows line
               is not used: program_rows copies it from the base row, so it is stale once views are added. No image is opened; base rows cost no file read.
  seed         crn_seeds' value for the row, then "x" and the sample count (a 2-sample answer can't stand in for a 4-sample request).
A kind other than eval has one key per unit: (weights id, sha1 of the unit without unit and prio, "-", "-"), which includes its out.
"""
import collections, contextlib, hashlib, json, os, random, re, threading, time, zlib

MAX_TRIES = 3          # fail() calls on a line of work before <out>.failed; the children of a split inherit the count
MAX_STALLS = 3         # leases lost in a row with no new row finished: a unit that kills its agent each time ends in <out>.failed rather than looping
LOCK_STALE = 120       # seconds; a lock whose holder is dead goes at once
CLAIM_GRACE = 60       # seconds a half-done step of a live process gets before expire() finishes it; its start time is in the file name, because rename does not update ctime on Vesta's /data
DEFAULT_RATE = 1000    # rows/h for a family with no entry in the table
WRITE_S = 900          # seconds for a write unit with no "write_s" in the table. A placeholder, not measured
OOM = re.compile(r"out.?of.?memory|\boom\b|cuda.*memory|memoryerror", re.I)
NAME = re.compile(r"(\d)_([0-9a-f]{8})_([0-9a-f]{8})_(\d+)_([a-z])(\d+)_(.+)\.json")
SIBLING = re.compile(r"(\.[as]\d+)*(\.short\d+)?(\.partial)?")      # what may follow <final> in the name of a file holding its rows; .time .job .later .failed do not match
LEASE = ("agent", "card", "pid", "lease_t")
STEP = re.compile(r"\.(taking|retiring)\.(\d+)\.\d+\.(\d+)")       # a step in progress in claimed/: <unit>.json.<step>.<pid>.<thread>.<ms>


def sha1(s):
    return hashlib.sha1(s.encode()).hexdigest()


def crn_seed(row_id, seed=101, gen=0):
    """rsi/loop.py crn_seeds for one row. It depends on the row id only, so a smaller rows file keeps every row's seed."""
    return zlib.crc32(f"{seed}|{gen}|{row_id}".encode()) % (2 ** 31 - 1024)


_pixels = {}


def image_id(path):
    if "/views/" not in path:
        return path
    if path not in _pixels:
        try:
            _pixels[path] = json.load(open(path + ".json"))["pixels"]
        except (OSError, ValueError, KeyError):
            return path                # no note yet: don't cache, and at worst a duplicate runs
    return _pixels[path]


def prompt_hash(row, size=None):
    ims = [image_id(p) for p in row.get("image_paths") or []]
    return sha1(json.dumps([row["prompt"], ims, size], sort_keys=True, ensure_ascii=False))[:16]


def row_key(weights, row, size=None, seed=101, gen=0, samples=2):
    return weights, prompt_hash(row, size), row["id"], f"{crn_seed(row['id'], seed, gen)}x{samples}"


def alive(pid):
    """Single machine. /proc first: the container's PID 1 does not reap, so a dead process stays as a zombie and answers kill -0."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] not in ("Z", "X")
    except FileNotFoundError:
        if os.path.isdir("/proc/self"):
            return False
    except (OSError, IndexError):
        pass
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    return True


def tmp_name(f):
    return f"{f}.{os.getpid()}.{threading.get_ident()}.new"


def write_atomic(f, text):
    t = tmp_name(f)
    with open(t, "w") as o:
        o.write(text)
    os.replace(t, f)


def jdump(f, x):
    write_atomic(f, json.dumps(x, ensure_ascii=False))


def jload(f):
    with open(f) as i:
        return json.load(i)


def read_records(path, want, rec):
    """Trace lines of an answer or .partial file into rec[id][sample] = (line, tokens). The first copy stands. A killed writer can leave a torn line; a line without a newline or that doesn't parse is skipped."""
    try:
        f = open(path, errors="replace")
    except FileNotFoundError:
        return rec
    with f:
        for l in f:
            if not l.endswith("\n"):
                continue
            try:
                r = json.loads(l)
                i = r["id"]
            except (ValueError, KeyError, TypeError):
                continue
            if i in want:
                rec.setdefault(i, {}).setdefault(r.get("sample", 0), (l, r.get("tokens", 0)))
    return rec


def siblings(final):
    d, b = os.path.split(final)
    try:
        names = os.listdir(d or ".")
    except FileNotFoundError:
        return []
    return sorted(os.path.join(d, n) for n in names if n.startswith(b) and SIBLING.fullmatch(n[len(b):]))


def complete(rec, ids, samples):
    return [i for i in ids if i in rec and all(k in rec[i] for k in range(samples))]


class Queue:
    def __init__(self, root, rates=None, seed=101, gen=0):
        """root is <run> or <run>/res. rates: {family: rows/h, "default": rows/h, "generate": rows/h, "write_s": seconds}. seed and gen are the --seed 101 and CRN_GEN 0 the engines run with."""
        self.res = root if os.path.basename(root.rstrip("/")) == "res" else os.path.join(root, "res")
        self.rates, self.seed, self.gen = rates or {}, seed, gen
        self.grace = CLAIM_GRACE
        for d in ("queue", "claimed", "done", "failed", "agents", "waiting", "rows", "families"):
            os.makedirs(self.p(d), exist_ok=True)
        self.ledger, self.lock = self.p("ledger.jsonl"), self.p("ledger.lock")
        self._idx, self._units, self._off, self._fams = {}, set(), 0, {}

    def p(self, *a):
        return os.path.join(self.res, *a)

    # -------------------------------------------------- ledger and lock
    def _log(self, **x):
        line = (json.dumps(dict(x, t=round(time.time(), 3)), ensure_ascii=False) + "\n").encode()
        fd = os.open(self.ledger, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            os.write(fd, line)
        finally:
            os.close(fd)

    def _catchup(self):
        """Read new ledger lines into the key index. Only processes that enqueue pay for this."""
        if not os.path.exists(self.ledger):
            return
        with open(self.ledger, "rb") as f:
            f.seek(self._off)
            data = f.read()
        data = data[:data.rfind(b"\n") + 1]
        self._off += len(data)
        for l in data.splitlines():
            if b'"ev": "enq"' in l:
                r = json.loads(l)
                self._units.add(r["unit"])
                for k in r["keys"]:
                    self._idx[(r["weights"], *k)] = (r["unit"], r["out"])

    def _break_stale(self):
        try:
            age = time.time() - os.stat(self.lock).st_mtime
            words = open(self.lock).read().split()
        except FileNotFoundError:
            return
        dead = bool(words) and not alive(int(words[0]))
        if not (age > LOCK_STALE or dead):
            return
        aside = f"{self.lock}.stale.{os.getpid()}.{time.time():.3f}"
        try:
            os.rename(self.lock, aside)
        except FileNotFoundError:
            return
        if open(aside).read().split() != words:          # a new holder came between the read and the rename: its lock goes back
            with contextlib.suppress(FileExistsError):
                os.link(aside, self.lock)
        else:
            self._log(ev="lock_broken", holder=words, age=round(age, 1))

    @contextlib.contextmanager
    def _locked(self):
        while True:
            try:
                fd = os.open(self.lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                self._break_stale()
                time.sleep(random.uniform(.002, .02))
                continue
            os.write(fd, f"{os.getpid()} {time.time():.3f}".encode())
            os.close(fd)
            break
        try:
            yield
        finally:
            os.replace(self.lock, self.lock + ".free")      # release by rename: nothing deleted, and no pile of lock files

    # -------------------------------------------------- enqueue
    def _qname(self, u):
        return f"{int(u['prio'])}_{sha1(u['family'])[:8]}_{sha1(u['editid'])[:8]}_{u['seq']:016d}_{u.get('kind', 'eval')[0]}{u.get('n', 1)}_{u['unit']}.json"

    def _put(self, u):
        u["seq"] = u.get("seq") or int(time.time() * 1e6)
        name = self._qname(u)
        t = self.p("queue", f".{name}.{os.getpid()}.{threading.get_ident()}.new")
        with open(t, "w") as o:
            json.dump(u, o, ensure_ascii=False)
        os.rename(t, self.p("queue", name))

    def _note_family(self, family):
        h = sha1(family)[:8]
        if h not in self._fams:
            f = self.p("families", h)
            if not os.path.exists(f):
                write_atomic(f, family)
            self._fams[h] = family

    def _owner_failed(self, unit, cache):
        if unit not in cache:
            cache[unit] = os.path.exists(self.p("failed", f"{unit}.json"))
        return cache[unit]

    def _read_rows(self, u):
        """Rows lines of an eval unit: [(id, raw line, row)]."""
        with open(u["rows"]) as f:
            lines = [l if l.endswith("\n") else l + "\n" for l in f if l.strip()]
        rows = [json.loads(l) for l in lines]
        return [(r["id"], l, r) for r, l in zip(rows, lines)]

    def enqueue(self, unit):
        """"new": some of it was queued. "waiting": nothing to run, because every key is already asked for (or <out> or <out>.failed is already there, as plan_item treats it). A unit that overlaps others runs
        its new rows only; the rest are copied into its <out> from the answers of the units that own them, so a waiting unit's <out> still appears and load_member reads it as usual."""
        u = dict(unit)
        for f in ("unit", "prio", "family", "editid", "out"):
            if f not in u:
                raise ValueError(f"unit without {f}: {u.get('unit')}")
        if "/" in u["unit"] or not 0 <= int(u["prio"]) <= 9:
            raise ValueError(f"bad unit id or prio: {u['unit']} {u['prio']}")
        u.setdefault("kind", "eval")
        weights = f"{u['family']}|{u['editid']}"
        lines = None
        if u["kind"] == "eval":
            lines = self._read_rows(u)
            ids = [i for i, _, _ in lines]
            if not ids or len(set(ids)) != len(ids):
                raise ValueError(f"{u['unit']}: rows file empty or with a repeated id: {u['rows']}")
            if u.get("keys") not in (None, ids) or u.get("n") not in (None, len(ids)):
                raise ValueError(f"{u['unit']}: keys or n do not match {u['rows']}")
            u["keys"], u["n"] = ids, len(ids)
            u.setdefault("samples", 2)
            keys = [row_key(weights, r, u.get("size"), u.get("seed", self.seed), self.gen, u["samples"]) for _, _, r in lines]
        else:
            ids = [u["unit"]]
            keys = [(weights, sha1(json.dumps({k: v for k, v in u.items() if k not in ("unit", "prio")}, sort_keys=True, default=str))[:16], "-", "-")]
        if os.path.exists(u["out"]) or os.path.exists(u["out"] + ".failed"):
            self._log(ev="already", unit=u["unit"], out=u["out"])
            return "waiting"
        os.makedirs(os.path.dirname(u["out"]) or ".", exist_ok=True)
        self._note_family(u["family"])
        cache, added = {}, False
        with self._locked():
            self._catchup()
            own, deps = [], {}
            for i, k in zip(ids, keys):
                o = self._idx.get(k)
                if o is None or self._owner_failed(o[0], cache):
                    own.append((i, k))
                else:
                    deps.setdefault(o, []).append(i)
            if not own and all(o[1] == u["out"] for o in deps):
                self._log(ev="dup", unit=u["unit"], out=u["out"], owner=[o[0] for o in deps])
                return "waiting"
            if any(o[1] == u["out"] for o in deps):
                raise ValueError(f"{u['unit']}: {u['out']} is being made from other rows by {[o[0] for o in deps]}")
            if u["unit"] in self._units and not self._owner_failed(u["unit"], cache):
                raise ValueError(f"{u['unit']}: this unit id was enqueued with other keys")
            old = self.p("failed", f"{u['unit']}.json")
            if os.path.exists(old):                          # the driver retries a failed unit: the old record moves aside
                os.rename(old, f"{old}.{time.time():.3f}.old")
            enq = dict(ev="enq", unit=u["unit"], out=u["out"], weights=weights, keys=[k[1:] for _, k in own])
            if not deps:
                u.update(tries=0, attempt=1, base=u["unit"])
                self._put(u)
                self._log(**enq)                             # queue first: a crash here gives at worst a repeated run, never a key nobody will answer
                return "new"
            root = self._root(u)
            root["deps"] = [dict(unit=o[0], out=o[1], ids=x) for o, x in deps.items()]
            jdump(self.p("waiting", f"{u['unit']}.json"), dict(u, root=root))
            if own:
                by_id = {i: l for i, l, _ in lines}
                self._child(dict(u, root=root, base=u["unit"], tries=0, attempt=0, stall=0), f"{u['unit']}.a1", [i for i, _ in own], by_id, tries=0, attempt=1)
            self._log(**enq)
            self._log(ev="wait", unit=u["unit"], own=len(own), deps=root["deps"])
            added = True
        if added:
            self.settle()
        return "new" if own else "waiting"

    # -------------------------------------------------- claim
    def _card(self, agent):
        try:
            c = jload(self.p("agents", f"{agent}.json")).get("card")
            if c is not None:
                return c
        except (OSError, ValueError):
            pass
        m = re.search(r"\d+$", str(agent))
        return int(m.group()) if m else None

    def claim(self, agent, family, editid, n=1):
        """Units of this family, as many as n. Order: the loaded edit first, then priority (0 first), then age. One batch is one editid and one kind, because the agent must drain before it changes edit."""
        fam, ed = sha1(family)[:8], sha1(editid)[:8]
        cands = []
        for name in os.listdir(self.p("queue")):
            m = NAME.fullmatch(name)
            if m and m.group(2) == fam:
                cands.append((m.group(3) != ed, int(m.group(1)), int(m.group(4)), name, m))
        cands.sort(key=lambda c: c[:4])
        out, batch = [], None
        for _, _, _, name, m in cands:
            if len(out) >= n:
                break
            if batch and batch != (m.group(3), m.group(5)):
                continue
            f = self._step(self.p("claimed", f"{m.group(7)}.json"), "taking")
            try:
                os.rename(self.p("queue", name), f)          # the winner is whoever's rename succeeds
            except FileNotFoundError:
                continue
            u = jload(f)
            final = self._root(u)["out"]
            if os.path.exists(final) or os.path.exists(final + ".failed"):
                os.rename(f, self._free(self.p("done", f"{u['unit']}.skipped.json")))
                self._log(ev="skip", unit=u["unit"], out=final)
                continue
            u.update(agent=agent, card=self._card(agent), pid=os.getpid(), lease_t=time.time())
            jdump(f, u)
            os.rename(f, self.p("claimed", f"{u['unit']}.json"))
            self._log(ev="claim", unit=u["unit"], agent=agent, card=u["card"], pid=u["pid"])
            out.append(u)
            batch = batch or (m.group(3), m.group(5))
        return out

    # -------------------------------------------------- results
    def _root(self, u):
        return u.get("root") or {k: u.get(k) for k in ("unit", "out", "rows", "keys", "samples", "model", "kind")} | {"deps": []}

    def _free(self, f):
        if not os.path.exists(f):
            return f
        k = 1
        while os.path.exists(f"{f}.{k}"):
            k += 1
        return f"{f}.{k}"

    def _mine(self, unit):
        """Does this caller still hold the lease?"""
        try:
            c = jload(self.p("claimed", f"{unit['unit']}.json"))
        except (OSError, ValueError):
            return False
        return all(c.get(k) == unit.get(k) for k in LEASE)

    def _scan(self, root):
        ids = set(root["keys"])
        rec = {}
        for f in siblings(root["out"]):
            read_records(f, ids, rec)
        return rec

    def _finish(self, root):
        """Build <final> if every row exists somewhere. Returns True once <final> stands. A plain unit's agent has already written it."""
        final = root["out"]
        if root.get("kind", "eval") != "eval" or os.path.exists(final):
            return os.path.exists(final)
        for d in root.get("deps") or []:
            if not os.path.exists(d["out"]):
                return False
        rec, samples = self._scan(root), root.get("samples") or 2
        for d in root.get("deps") or []:
            read_records(d["out"], set(d["ids"]), rec)
        if len(complete(rec, root["keys"], samples)) < len(root["keys"]):
            return False
        lines, toks = [], []
        for i in root["keys"]:
            for k in range(samples):
                lines.append(rec[i][k][0])
                toks.append(rec[i][k][1])
        secs, parts = 0., 0
        for f in siblings(final):
            if not f.endswith(".partial") and os.path.exists(f + ".time"):
                with contextlib.suppress(ValueError, OSError, KeyError):
                    secs += jload(f + ".time")["secs"]
                    parts += 1
        t = tmp_name(final)
        with open(t, "w") as o:
            o.writelines(lines)
        first = not os.path.exists(final)
        os.replace(t, final if first else final + ".later")      # the first to finish stands, as in run_item
        jdump(final + (".time" if first else ".later.time"), dict(secs=secs, n=len(root["keys"]), model=root.get("model"), tokens=sum(toks) / max(1, len(toks)), parts=parts, copied=sum(len(d["ids"]) for d in root.get("deps") or [])))
        jdump(self._free(self.p("done", f"{root['unit']}.json")), dict(root, built=True, t=time.time()))
        self._log(ev="built", unit=root["unit"], out=final, n=len(root["keys"]), parts=parts)
        return True

    def done(self, unit, stats=None):
        """The agent has written unit["out"]. Checks it holds every row: a short file is moved aside and only its missing rows go back, and no file at all counts as a failure. A caller that lost its lease only
        adds its answers: the unit is no longer its own to move."""
        mine = self._mine(unit)
        kind = unit.get("kind", "eval")
        if kind == "eval":
            rec = read_records(unit["out"], set(unit["keys"]), {})
            miss = len(unit["keys"]) - len(complete(rec, unit["keys"], unit.get("samples") or 2))
            if miss:
                if not mine:
                    self._log(ev="late_short", unit=unit["unit"], agent=unit.get("agent"), missing=miss)
                    return
                if os.path.exists(unit["out"]):
                    os.rename(unit["out"], self._free(f"{unit['out']}.short{int(time.time())}"))
                self.fail(unit, f"done with {miss} of {len(unit['keys'])} rows missing")
                return
        elif not os.path.exists(unit["out"]):
            if mine:
                self.fail(unit, "done without output")
            return
        if mine:
            f = self.p("claimed", f"{unit['unit']}.json")
            jdump(f, dict(unit, stats=stats, done_t=time.time()))
            os.rename(f, self._free(self.p("done", f"{unit['unit']}.json")))
        self._log(ev="done" if mine else "late_done", unit=unit["unit"], agent=unit.get("agent"), stats=stats)
        self._finish(self._root(unit))
        self.settle()

    # -------------------------------------------------- retries
    def _step(self, f, step):
        return f"{f}.{step}.{os.getpid()}.{threading.get_ident()}.{int(time.time() * 1000)}"

    def _retire(self, uid):
        """The single winner of taking a claim off the board, whoever asks: fail, release, expire. Returns the retiring path, or None for a loser."""
        f = self.p("claimed", f"{uid}.json")
        t = self._step(f, "retiring")
        try:
            os.rename(f, t)
        except FileNotFoundError:
            return None
        return t

    def _bury(self, t, uid, tag):
        os.rename(t, self._free(self.p("failed", f"{uid}.{tag}.json")))

    def _child(self, u, uid, keys, by_id, tries, attempt, stall=0):
        root = u.get("root") or self._root(u)
        c = {k: v for k, v in u.items() if k not in LEASE and k not in ("stats", "done_t")}
        if u.get("kind", "eval") == "eval":
            rows = self.p("rows", f"{uid}.jsonl")
            write_atomic(rows, "".join(by_id[i] for i in keys))
            c.update(rows=rows, out=root["out"] + uid[len(root["unit"]):], n=len(keys), keys=keys)
        c.update(unit=uid, base=re.sub(r"\.a\d+$", "", uid), root=root, tries=tries, attempt=attempt, stall=stall)
        self._put(c)
        return uid

    def _by_id(self, root):
        by_id = {}
        with open(root["rows"]) as f:
            for l in f:
                if l.strip():
                    by_id[json.loads(l)["id"]] = l if l.endswith("\n") else l + "\n"
        return by_id

    def _missing(self, u):
        """Rows of this unit not yet finished in any file of its final."""
        root = self._root(u)
        return [i for i in u["keys"] if i not in set(complete(self._scan(root), u["keys"], root.get("samples") or 2))]

    def _terminal(self, u, why):
        root = self._root(u)
        final = root["out"]
        if not os.path.exists(final):
            write_atomic(final + ".failed", f"{why}\n")
        jdump(self.p("failed", f"{root['unit']}.json"), dict(unit=root["unit"], out=final, why=why, t=time.time(), last=u["unit"]))
        self._log(ev="failed", unit=root["unit"], last=u["unit"], out=final, why=why)

    def fail(self, unit, why):
        """A retry for the rows still missing. On OOM, two units of half those rows each. After MAX_TRIES fails on this line of work, <final>.failed. A write or generate unit is retried whole with the same out; it has no rows to split."""
        if not self._mine(unit):
            self._log(ev="late_fail", unit=unit["unit"], agent=unit.get("agent"), why=str(why)[:300])
            return
        t = self._retire(unit["unit"])
        if t is None:
            return
        why, tries, eval_ = str(why), int(unit.get("tries", 0)) + 1, unit.get("kind", "eval") == "eval"
        uid, base, root = unit["unit"], unit.get("base", unit["unit"]), self._root(unit)
        self._log(ev="fail", unit=uid, agent=unit.get("agent"), tries=tries, why=why[:300])
        if eval_:
            miss = self._missing(unit)
            if not miss:
                self._bury(t, uid, "allrows")
                self._finish(root)
                self.settle()
                return
        if tries >= MAX_TRIES:
            self._terminal(unit, f"{tries} failures, last: {why[:300]}")
            self._bury(t, uid, "final")
            self.settle()
            return
        attempt = int(unit.get("attempt", 1)) + 1
        if eval_:
            by_id = self._by_id(root)
            if OOM.search(why) and len(miss) >= 2:
                h = len(miss) // 2
                kids = [self._child(unit, f"{base}.s{k}", part, by_id, tries, 1) for k, part in enumerate((miss[:h], miss[h:]))]
                self._log(ev="split", unit=uid, into=kids, rows=[h, len(miss) - h], tries=tries)
                self._bury(t, uid, "oom")
                return
            nid = self._child(unit, f"{base}.a{attempt}", miss, by_id, tries, attempt)
        else:
            nid = self._child(unit, f"{base}.a{attempt}", None, None, tries, attempt)
        self._log(ev="retry", unit=uid, next=nid, tries=tries)
        self._bury(t, uid, "fail")

    def _requeue_lost(self, t, u, why, count=True):
        """Back to the queue with only its missing rows. No try is used, but a unit that loses its agent with no new row finished MAX_STALLS times in a row goes to <final>.failed."""
        uid, base, root = u["unit"], u.get("base", u["unit"]), self._root(u)
        eval_ = u.get("kind", "eval") == "eval"
        miss = self._missing(u) if eval_ else None
        if eval_ and not miss:
            self._bury(t, uid, "allrows")
            self._finish(root)
            self._log(ev="requeue", unit=uid, why=why, next=None, missing=0)
            return None
        stall = int(u.get("stall", 0)) + 1 if count and (not eval_ or len(miss) == len(u["keys"])) else 0
        if stall >= MAX_STALLS:
            self._terminal(u, f"lost its agent {stall} times with no new row: {why}")
            self._bury(t, uid, "stalled")
            return None
        attempt = int(u.get("attempt", 1)) + 1
        nid = f"{base}.a{attempt}"
        if not self._known(nid):
            if eval_:
                self._child(u, nid, miss, self._by_id(root), int(u.get("tries", 0)), attempt, stall)
            else:
                self._child(u, nid, None, None, int(u.get("tries", 0)), attempt, stall)
        self._log(ev="requeue", unit=uid, why=why, next=nid, kept=len(u["keys"]) - len(miss) if eval_ else None, missing=len(miss) if eval_ else None)
        self._bury(t, uid, "lost")
        return nid

    def _known(self, uid):
        if any(n.endswith(f"_{uid}.json") for n in os.listdir(self.p("queue"))):
            return True
        return any(os.path.exists(self.p(d, f"{uid}.json")) for d in ("claimed", "done"))

    def release(self, unit, why="released"):
        """Give a claimed unit back without using a try, for an agent leaving or changing family. Rows it finished stay."""
        if self._mine(unit):
            t = self._retire(unit["unit"])
            if t:
                self._requeue_lost(t, unit, why, count=False)
                self.settle()

    # -------------------------------------------------- liveness
    def heartbeat(self, agent, state=None):
        s = dict(state or {})
        s.update(agent=agent, pid=s.get("pid", os.getpid()), t=time.time())
        jdump(self.p("agents", f"{agent}.json"), s)

    def _dead_why(self, u, now, stale_s):
        if not alive(u["pid"]):
            return f"process {u['pid']} dead"
        try:
            hb = jload(self.p("agents", f"{u['agent']}.json"))
        except (OSError, ValueError):
            return f"no heartbeat, lease {now - u['lease_t']:.0f}s old" if now - u["lease_t"] > stale_s else None
        if hb.get("pid") != u["pid"] and hb.get("t", 0) >= u["lease_t"]:
            return f"agent {u['agent']} now runs as {hb.get('pid')}, not {u['pid']}"
        ref = max(hb.get("t", 0), u["lease_t"])
        return f"heartbeat {now - ref:.0f}s old" if now - ref > stale_s else None

    def expire(self, stale_s=300):
        """Requeue units of dead agents, each with only its missing rows. Also finishes steps a crash left half done. Any process may call it, and two at once are safe. Returns the new unit ids."""
        now, out = time.time(), []
        for n in os.listdir(self.p("claimed")):
            f = self.p("claimed", n)
            try:
                if n.endswith(".new"):
                    continue
                m = STEP.search(n)
                if m:
                    if alive(int(m.group(2))) and now - int(m.group(3)) / 1000 < self.grace:
                        continue
                    u = jload(f)
                    if m.group(1) == "taking":           # died between the rename and the lease: the unit was never given out
                        os.rename(f, self.p("queue", self._qname(u)))
                        self._log(ev="untake", unit=u["unit"])
                        out.append(u["unit"])
                    else:
                        t = f"{f}.sweep.{os.getpid()}"
                        os.rename(f, t)
                        out.append(self._requeue_lost(t, u, "retirement interrupted", count=False))
                    continue
                if not n.endswith(".json"):
                    continue
                u = jload(f)
                why = self._dead_why(u, now, stale_s)
                if why:
                    t = self._retire(u["unit"])
                    if t:
                        out.append(self._requeue_lost(t, u, why))
            except (FileNotFoundError, ValueError):
                continue                                 # another process moved it, or it is being replaced
        self.settle()
        return [x for x in out if x]

    def settle(self):
        """Units waiting on others' answers: build them when the answers exist, flag them when a source failed."""
        for n in os.listdir(self.p("waiting")):
            f = self.p("waiting", n)
            if not n.endswith(".json"):
                continue
            try:
                w = jload(f)
            except (FileNotFoundError, ValueError):
                continue
            root, uid = w["root"], w["unit"]
            if os.path.exists(self.p("failed", f"{uid}.json")):
                to = self.p("failed", f"{uid}.waiter.json")
            elif self._finish(root):
                to = self.p("done", f"{uid}.waiter.json")
            elif bad := [d["unit"] for d in root["deps"] if os.path.exists(self.p("failed", f"{d['unit']}.json")) and not os.path.exists(d["out"])]:
                self._terminal(w, f"source units failed: {bad}")
                to = self.p("failed", f"{uid}.waiter.json")
            else:
                continue
            with contextlib.suppress(FileNotFoundError):
                os.rename(f, self._free(to))

    # -------------------------------------------------- reading
    def _secs(self, kind, n, family, rates):
        if kind == "w":
            return rates.get("write_s", WRITE_S)
        rate = rates.get(family) or rates.get("default") or DEFAULT_RATE
        if kind == "g":
            rate = rates.get("generate") or rate
        return 3600. * n / rate

    def backlog(self, rates=None):
        """{family: seconds of work}: open units, plus the rows of claimed units not finished yet. rates as in __init__; here it overrides the table given there."""
        rates = self.rates if rates is None else rates
        out = collections.defaultdict(float)
        for name in os.listdir(self.p("queue")):
            m = NAME.fullmatch(name)
            if m:
                fam = self._fam_name(m.group(2))
                out[fam] += self._secs(m.group(5), int(m.group(6)), fam, rates)
        for name in os.listdir(self.p("claimed")):
            if not name.endswith(".json"):
                continue
            try:
                u = jload(self.p("claimed", name))
            except (FileNotFoundError, ValueError):
                continue
            n = u.get("n", 1)
            if u.get("kind", "eval") == "eval":
                with contextlib.suppress(FileNotFoundError):
                    with open(u["out"] + ".partial", "rb") as f:
                        n = max(0, n - f.read().count(b"\n") // (u.get("samples") or 2))
            out[u["family"]] += self._secs(u.get("kind", "eval")[0], n, u["family"], rates)
        return dict(out)

    def _fam_name(self, h):
        if h not in self._fams:
            self._fams[h] = open(self.p("families", h)).read()
        return self._fams[h]

    def status(self):
        return {d: sum(1 for n in os.listdir(self.p(d)) if n.endswith(".json")) for d in ("queue", "claimed", "done", "failed", "waiting")}
