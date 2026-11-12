"""v4 loop side of engine_api.md: write request batches, index result lines, wait with deadlines. No HTTP here."""
import glob, json, os, time

MAX_RR = 2           # an aborted read is re-sent at most this many times by the process that submitted it
KEEP = 1500          # characters of content / reasoning kept in memory per result (cards show the tails)


def now():
    return time.time()


class IO:
    def __init__(self, root, log=print, loc_rows=None):
        """loc_rows (B1, coordinator 11-11): row ids whose read results get their location (file, byte offset of the line) indexed in
        self.loc, so the signal export can read them whole (in memory content / reasoning are cut to their last KEEP characters)."""
        self.root, self.log = root, log
        self.loc_rows, self.loc = (set(loc_rows) if loc_rows is not None else None), {}
        self.aborted, self._reqf, self._rr = {}, {}, []        # finish "abort" (coordinator 11-11 01:45): not read; re-sent (MAX_RR times)
        for d in ("q/in", "q/out", "q/cancel"):
            os.makedirs(os.path.join(root, d), exist_ok=True)
        self.res, self.err, self.off, self.pending = {}, {}, {}, set()
        for f in sorted(glob.glob(os.path.join(root, "q/in/*.jsonl"))):
            if os.path.exists(os.path.join(root, "q/cancel", os.path.basename(f)[:-6])):
                continue                                  # a cancelled batch (AM17 b): its dropped keys are not pending
            with open(f) as fh:
                for l in fh:
                    if l.strip():
                        try:
                            self.pending.add(json.loads(l)["key"])
                        except (ValueError, KeyError):
                            pass
        self.poll()

    # ------------------------------------------------------------------ results
    def poll(self):
        n = 0
        for f in sorted(glob.glob(os.path.join(self.root, "q/out/*.jsonl"))):
            o = self.off.get(f, 0)
            try:
                if os.path.getsize(f) <= o:
                    continue
                with open(f, "rb") as fh:
                    fh.seek(o)
                    buf = fh.read()
            except OSError:
                continue
            end = buf.rfind(b"\n") + 1            # only complete lines; a partial last line is read on the next poll
            pos = 0
            for lb in buf[:end].split(b"\n"):     # JSONL lines end at b"\n" only (str.splitlines would also split at U+2028 etc.)
                lo, pos = o + pos, pos + len(lb) + 1
                l = lb.decode("utf-8", "replace")
                if not l.strip():
                    continue
                try:
                    r = json.loads(l)
                except ValueError:
                    continue
                k = r.get("key")
                if not k:
                    continue
                if r.get("error"):
                    if k not in self.res:
                        self.err[k] = r
                    continue
                if r.get("finish") == "abort":                    # a read the server dropped (card restart): NOT a result (never graded
                    self.aborted[k] = self.aborted.get(k, 0) + 1   # wrong); the key is pending again and is re-sent when this process
                    if k not in self.res:                         # submitted it (else at the next submit that asks for it)
                        self.pending.discard(k)
                        if k in self._reqf and self.aborted[k] <= MAX_RR:
                            self._rr.append(k)
                    continue
                if self.loc_rows is not None and r.get("mode") == "read" and r.get("row") in self.loc_rows:
                    self.loc[k] = (f, lo)
                for f2 in ("content", "reasoning"):
                    if f2 == "content" and r.get("mode") == "chat":
                        continue                                  # proposer answers are kept whole (the repair turn sends them back)
                    if isinstance(r.get(f2), str) and len(r[f2]) > KEEP:
                        r[f2] = r[f2][-KEEP:]
                self.res[k] = r
                self.err.pop(k, None)
                n += 1
            self.off[f] = o + end
        if self._rr:
            self.resend()
        return n

    def resend(self):
        """Aborted reads this process submitted: the same requests again, in a batch named after their first batch + "_rr"."""
        ks, self._rr = [k for k in dict.fromkeys(self._rr) if k not in self.res and k not in self.pending], []
        by = {}
        for k in ks:
            by.setdefault(self._reqf[k], []).append(k)
        for f, kk in by.items():
            want, qs = set(kk), []
            try:
                with open(f) as fh:
                    for l in fh:
                        if any(k in l for k in want):
                            q = json.loads(l)
                            if q.get("key") in want:
                                qs.append(q)
            except (OSError, ValueError):
                continue
            if qs:
                name = os.path.basename(f)[:-6].rsplit("_", 1)[0] + "_rr"
                n = self.submit(name, qs)
                self.log(f"  {len(qs)} aborted reads re-sent as {name} ({n} new)")

    def have(self, k):
        return k in self.res

    def get(self, k):
        return self.res.get(k)

    # ------------------------------------------------------------------ requests
    def submit(self, name, reqs):
        """Writes the requests not yet done and not already pending as one batch; returns how many were written."""
        self.poll()
        new, seen = [], set()
        for q in reqs:
            k = q["key"]
            if k in self.res or k in self.pending or k in seen:
                continue
            seen.add(k)
            new.append(q)
        if not new:
            return 0
        b = f"{name}_{int(now() * 1000)}"
        p = os.path.join(self.root, "q/in", b + ".jsonl")
        with open(p + ".tmp", "w") as fh:
            for q in new:
                fh.write(json.dumps(q, ensure_ascii=False) + "\n")
        os.replace(p + ".tmp", p)
        for q in new:
            self._reqf[q["key"]] = p
        self.pending.update(seen)
        return len(new)

    def cancel_prefix(self, name):
        for f in glob.glob(os.path.join(self.root, "q/in", name + "_*.jsonl")):
            b = os.path.basename(f)[:-6]
            open(os.path.join(self.root, "q/cancel", b), "w").close()

    def wait(self, keys, deadline=None, every=5.0, label="", tick=300):
        """Blocks until every key has a result (or an error line) or the deadline passes. Returns (done, missing)."""
        keys = list(dict.fromkeys(keys))
        t0, last = now(), now()
        while True:
            self.poll()
            miss = [k for k in keys if k not in self.res and k not in self.err]
            if not miss:
                return [k for k in keys if k in self.res], []
            if deadline is not None and now() >= deadline:
                return [k for k in keys if k in self.res], miss
            if now() - last >= tick:
                last = now()
                self.log(f"  waiting {label}: {len(keys) - len(miss)}/{len(keys)} done after {now() - t0:.0f}s")
            time.sleep(every)

    # ------------------------------------------------------------------ throughput
    def tok_rate(self, window=600):
        """Generated tok/s over all servers from stats.jsonl: the latest line of each server inside the window."""
        p = os.path.join(self.root, "stats.jsonl")
        if not os.path.exists(p):
            return None
        try:
            with open(p, "rb") as fh:
                fh.seek(max(0, os.path.getsize(p) - 200000))
                lines = fh.read().decode("utf-8", "replace").splitlines()[1:]
        except OSError:
            return None
        last = {}
        for l in lines:
            try:
                r = json.loads(l)
            except ValueError:
                continue
            last[r.get("server")] = r
        rates = [float(r.get("gen_tok_s") or 0) for r in last.values()]
        return sum(rates) if rates else None
