"""B1 (coordinator 11-11, Track B): the signal export. The signal worker reads nothing of the run but these files.

RUN/signal/<tip>/traces.json        {row id: {bench, role, kind, answer, prompt, image_paths, prefix, settings[, ops][, edit],
                                     samples: {str(index): {content, reasoning, tokens, prompt_tokens, image_tokens, ok, pred, finish}}}}
                                    (the format of the hand export gset_t005r_traces.json): every row of role show or monitor, the tip's
                                    view on it exactly as the engine read it (prefix, settings with null = stock, ops, edit), and every
                                    cached sample of the tip there, whole (read from the result line, not from the loop's cut copy).
RUN/signal/<tip>/traces_meta.json   tip, entries, rows by role, samples (total, min / max per row), complete (every row routed and
                                    >= P_FULL samples), rows not yet routed, why / when written.
RUN/signal/<tip>/traces_cand.jsonl  appended: one line per candidate read on a show / monitor row against this tip (lab drafts, the
                                    candidates' SHOW reads, ...), same row fields with the request's own prefix / settings / ops / edit,
                                    tagged with batch, generation, cid (when the batch or the view names it) and edit / edit_sha / weights;
                                    samples holds that one read. RUN/signal/<tip>/traces_cand.keys lists the keys written (restart safe).
JUDGE rows are never exported: the row set is roles show + monitor only, and every written row is checked again (trace_row raises).
Written at every tip change (write_tip), at the generation's start and report, and on a tick from save() whenever more samples are in.
Mixed into gen.Run (needs rows, role, state, io, rd, routes, flat, read_req, read_key, view, g)."""
import collections, glob, json, os, re, time

SIGNAL_ROLES = ("show", "monitor")
SAMPLE_FIELDS = ("content", "reasoning", "tokens", "prompt_tokens", "image_tokens", "ok", "pred", "finish")
MAX_S = 64                     # sample indices looked up per row (the loop reads 0..7 for a tip; a candidate's LITE samples are 0..7 too)
TICK_S = 120                   # at most one tick per this many seconds from save()
CAND_BATCH = re.compile(r"_(?:rung1|rung1b|rung1s2|carry|lcomp)_(g\d+c\d+)$")
LAB_BATCH = re.compile(r"_labread(\d+)$")


def _jdump(p, x):
    with open(p + ".tmp", "w") as f:
        json.dump(x, f, ensure_ascii=False)
    os.replace(p + ".tmp", p)


class SignalMixin:
    def signal_rows(self):
        return [r for r in self.role["show"] + self.role["monitor"] if self.rows[r].get("v4_role") in SIGNAL_ROLES]

    def signal_dir(self, tid):
        return os.path.join(self.rd, "signal", tid)

    def full_result(self, k):
        """The whole result line of a read (content / reasoning not cut): by the IO's location index; an IO without one (the test fakes)
        keeps whole results in memory. None if the read is not in or its line cannot be read."""
        loc = getattr(self.io, "loc", None)
        if loc is None:
            return self.io.get(k)
        p = loc.get(k)
        if not p:
            return None
        try:
            with open(p[0], "rb") as fh:
                fh.seek(p[1])
                r = json.loads(fh.readline())
            return r if r.get("key") == k else None
        except (OSError, ValueError):
            return None

    def trace_row(self, r, prefix, settings, ops=None, edit=None):
        row = self.rows[r]
        if row.get("v4_role") not in SIGNAL_ROLES:
            raise ValueError(f"signal export: row {r} has role {row.get('v4_role')!r}; only {SIGNAL_ROLES} may be exported")
        x = dict(bench=row.get("bench"), role=row["v4_role"], kind=row.get("kind"), answer=row.get("answer"), prompt=row.get("prompt"),
                 image_paths=list(row.get("image_paths") or []), prefix=prefix, settings=settings)
        if ops:
            x["ops"] = ops
        if edit:
            x["edit"] = edit
        return x

    @staticmethod
    def trace_sample(full):
        return {f: full.get(f) for f in SAMPLE_FIELDS}

    def route_known(self, entries, r):
        return all(e.get("cond") == "always" or (e.get("cond"), r) in self.routes for e in self.flat(entries))

    # ------------------------------------------------------------------ the tip
    def export_traces(self, why, force=False, record=True):
        """traces.json + traces_meta.json of the current tip; rewritten only when the sample count or completeness changed (or force)."""
        t = self.state["tip"]
        tid = t["id"]
        old = (self.state.get("signal") or {}).get(tid) or {}
        self.load_routes(t["entries"], self.signal_rows())       # a new process (one per generation) has no routes before its parent step
        out, unrouted, nofull, nabort = {}, [], 0, 0
        ab = getattr(self.io, "aborted", None) or {}
        for r in self.signal_rows():
            if not self.route_known(t["entries"], r):
                unrouted.append(r)
                continue
            q, sg = self.read_req(t["entries"], r, 0, 5)
            sm = {}
            for s in range(MAX_S):
                k = self.read_key(sg, r, s)
                if self.io.get(k) is None:
                    nabort += k in ab                            # a dropped read (finish "abort") is not a sample: left out, counted
                    continue
                full = self.full_result(k)
                if full is None:
                    nofull += 1
                    continue
                if full.get("finish") == "abort":
                    nabort += 1
                    continue
                sm[str(s)] = self.trace_sample(full)
            x = self.trace_row(r, q.get("prefix"), q.get("settings"), q.get("ops"), q.get("edit"))
            x["samples"] = sm
            out[r] = x
        per = [len(x["samples"]) for x in out.values()]
        n = sum(per)
        pfull = getattr(self, "P_FULL_SIGNAL", 8)
        complete = not unrouted and bool(per) and min(per) >= pfull
        if not force and old.get("samples") == n and old.get("complete") == complete and old.get("rows") == len(out) and old.get("abort", 0) == nabort:
            return old
        d = self.signal_dir(tid)
        prev = old.get("samples")
        if prev is None:                                         # nothing recorded in state (another process wrote it): the file's own meta
            try:
                m_ = json.load(open(os.path.join(d, "traces_meta.json")))
                prev = m_.get("samples") if m_.get("tip") == tid else None
            except (OSError, ValueError):
                prev = None
        if prev is not None and n < prev:                        # a tip's samples only grow: never replace an export by a smaller one
            if getattr(self, "_sig_kept", None) != (tid, n):
                self._sig_kept = (tid, n)
                (getattr(self, "_signal_log", None) or (lambda m: None))(
                    f"  signal export ({why}): {n} samples < {prev} already written for {tid} ({len(unrouted)} rows not routed); file kept")
            return old
        os.makedirs(d, exist_ok=True)
        _jdump(os.path.join(d, "traces.json"), out)
        fl = self.flat(t["entries"])
        meta = dict(tip=tid, parent=t.get("parent"), born=t.get("born"), entries=len(fl), edits=sum(1 for e in fl if e.get("edit")),
                    rows=len(out), roles=dict(collections.Counter(x["role"] for x in out.values())), samples=n,
                    per_row=dict(min=min(per) if per else 0, max=max(per) if per else 0), complete=complete, p_full=pfull,
                    unrouted=len(unrouted), not_whole=nofull, abort_dropped=nabort, why=why, gen=getattr(self, "g", None), t=time.time(),
                    format="row id -> {bench, role, kind, answer, prompt, image_paths, prefix, settings[, ops][, edit], samples{str(index): "
                           "{" + ", ".join(SAMPLE_FIELDS) + "}}}; settings null = stock; roles show + monitor only, never judge")
        _jdump(os.path.join(d, "traces_meta.json"), meta)
        rec = dict(samples=n, rows=len(out), complete=complete, t=meta["t"], abort=nabort)
        if record:
            self.state.setdefault("signal", {})[tid] = rec
        log_ = getattr(self, "_signal_log", None) or (lambda m: None)
        log_(f"  signal export ({why}): {d}/traces.json, {len(out)} rows ({meta['roles']}), {n} samples (per row {meta['per_row']['min']}-"
             f"{meta['per_row']['max']}){'' if complete else f', not complete ({len(unrouted)} rows not routed yet)'}"
             + (f"; {nofull} reads not readable whole" if nofull else ""))
        return rec

    # ------------------------------------------------------------------ candidates against the tip
    def _tip_of(self, tid):
        if self.state["tip"]["id"] == tid:
            return self.state["tip"]
        return next((t for t in reversed(self.state["tips"]) if t["id"] == tid), None)

    def _tip_keys(self, t):
        ks = set()
        self.load_routes(t["entries"], self.signal_rows())
        for r in self.signal_rows():
            if self.route_known(t["entries"], r):
                sg = self.read_req(t["entries"], r, 0, 5)[1]
                ks.update(self.read_key(sg, r, s) for s in range(MAX_S))
        return ks

    def _tip_gens(self, tid):
        """Generations whose reads were against tip tid: G["tip_at_start"] (set after the boundary merges); a generation created before
        B1 has none, then its candidates' parent names it."""
        return sorted(int(gk) for gk, G in self.state["gens"].items()
                      if G.get("tip_at_start", None) == tid or ("tip_at_start" not in G and any(c.get("parent") == tid for c in G.get("cands", []))))

    def cand_tips(self):
        """The tips whose candidate reads may still land: those of the current and the previous generation (late reads after a tip change)."""
        g = int(getattr(self, "g", 0) or 0)
        out = []
        for gk in (g - 1, g):
            G = self.state["gens"].get(str(gk)) or {}
            ts = [G["tip_at_start"]] if G.get("tip_at_start") else sorted({c.get("parent") for c in G.get("cands", []) if c.get("parent")})
            out += [t for t in ts if t not in out]
        return out

    def export_traces_cand(self, why="", tid=None):
        """Append the candidate reads on show / monitor rows against tip tid (default: the current tip) that are in and not yet written."""
        tid = tid or self.state["tip"]["id"]
        t = self._tip_of(tid)
        if t is None:
            return 0
        d = self.signal_dir(tid)
        kf = os.path.join(d, "traces_cand.keys")
        st = self.__dict__.setdefault("_cand_state", {})
        if tid not in st:                                        # request files are written once (atomic rename) and never appended:
            st[tid] = dict(done=set(l.strip() for l in open(kf)) if os.path.exists(kf) else set(), pend={})   # relevant requests per file
        done, pend = st[tid]["done"], st[tid]["pend"]
        sig = set(self.signal_rows())
        tipk = self._tip_keys(t)
        gens = self._tip_gens(tid)
        cands = {c["cid"]: c for gk in gens for c in self.state["gens"][str(gk)].get("cands", [])}
        sigmap = st[tid].setdefault("sigmap", {})

        def cids_of(gk, r, k):
            """Candidates of generation gk whose view over tip t on row r has the key's signature (a batch of several candidates)."""
            if (gk, r) not in sigmap:
                m_ = collections.defaultdict(list)
                for c in self.state["gens"][str(gk)].get("cands", []):
                    e = c.get("entry")
                    if not e or c.get("kind") == "bundle":
                        continue
                    miss = [x for x in self.flat([e]) if x.get("cond") != "always" and (x["cond"], r) not in self.routes]
                    if miss:
                        self.load_routes(miss, [r])
                    if self.route_known(t["entries"] + [e], r):
                        m_[self.view(t["entries"] + [e], r)[2]].append(c["cid"])
                sigmap[(gk, r)] = dict(m_)
            parts = k.split("|")
            return sigmap[(gk, r)].get(parts[1], []) if len(parts) == 4 and parts[0] == "x" else []
        lines, keys = [], []
        for gk in gens:
            for f in sorted(glob.glob(os.path.join(self.io.root, "q", "in", f"g{gk:03d}_*.jsonl"))):
                name = os.path.basename(f)[:-6].rsplit("_", 1)[0]
                if "_lab" not in name and "_show" not in name and "_rung" not in name and "_carry" not in name and "_lcomp" not in name:
                    continue
                m, ml = CAND_BATCH.search(name), LAB_BATCH.search(name)
                if f not in pend:
                    qs = []
                    try:
                        with open(f) as fh:
                            for l in fh:
                                try:
                                    q = json.loads(l)
                                except ValueError:
                                    continue
                                if q.get("mode") == "read" and q.get("row") in sig and q.get("key") not in tipk:
                                    qs.append(q)
                    except OSError:
                        continue
                    pend[f] = qs
                left = []
                for q in pend[f]:
                    k, r = q.get("key"), q.get("row")
                    if k in done or k in tipk:
                        continue
                    if self.rows[r].get("v4_role") not in SIGNAL_ROLES:          # (sig holds show + monitor only; checked again)
                        continue
                    left.append(q)
                pend[f] = left
                for q in left:
                    k, r = q["key"], q["row"]
                    if self.io.get(k) is None:
                        continue
                    full = self.full_result(k)
                    if full is None or full.get("finish") == "abort":
                        continue
                    cid = m.group(1) if m else None
                    cids = [cid] if cid else ([] if ml else cids_of(gk, r, k))
                    cid = cid or (cids[0] if len(cids) == 1 else None)
                    ed = q.get("edit") or ((cands.get(cid) or {}).get("entry") or {}).get("edit")
                    x = dict(key=k, row=r, sample=q.get("sample"), batch=name, gen=gk, tip=tid, cid=cid, cids=cids,
                             author=(cands.get(cid) or {}).get("author") if cid else None, lab_turn=int(ml.group(1)) if ml else None,
                             edit=ed, edit_sha=q.get("edit_sha"), weights=q.get("weights"), engine=q.get("engine"))
                    x.update(self.trace_row(r, q.get("prefix"), q.get("settings"), q.get("ops"), None))
                    x["samples"] = {str(q.get("sample", 0)): self.trace_sample(full)}
                    lines.append(x)
                    keys.append(k)
                    done.add(k)
        if lines:
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "traces_cand.jsonl"), "a") as fh:
                for x in lines:
                    fh.write(json.dumps(x, ensure_ascii=False) + "\n")
            with open(kf, "a") as fh:
                for k in keys:
                    fh.write(k + "\n")
            log_ = getattr(self, "_signal_log", None) or (lambda m: None)
            log_(f"  signal export ({why}): {len(lines)} candidate reads appended to {d}/traces_cand.jsonl "
                 f"({dict(collections.Counter(x['batch'].split('_', 1)[1].rstrip('0123456789').split('_g')[0] for x in lines))})")
        return len(lines)

    # ------------------------------------------------------------------ when
    def signal_tick(self, why="tick", force=False):
        """Never lets an export error reach the loop."""
        if getattr(self.a, "no_signal", False):
            return
        now = time.time()
        if not force and now - getattr(self, "_signal_t", 0.0) < TICK_S:
            return
        self._signal_t = now
        try:
            self.export_traces(why, force=force)
            for tid in self.cand_tips() or [self.state["tip"]["id"]]:
                self.export_traces_cand(why, tid)
        except Exception as ex:                                  # noqa: BLE001 (an export problem must never stop the loop)
            log_ = getattr(self, "_signal_log", None) or (lambda m: None)
            log_(f"  signal export ({why}) failed: {type(ex).__name__}: {ex}")
