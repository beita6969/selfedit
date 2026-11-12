"""selfedit v4, Stage 1 driver: one generation = parent -> propose -> validate -> route -> rung1 -> judge1 -> rung2 -> judge2 -> tip -> report.

Spec: ~/selfedit/v4_build_spec_1110.md (A1-A11 + AM1-AM6; Stage 1 per coordinator 11-10). Authors B (blind one-factor knob mutations) and
P0 (frozen stock proposer, plain card prompt); P_inc = P0 in Stage 1. An entry = (condition | "always", procedure and/or settings); a router
call decides each conditional entry per row; the procedure is added and the settings used only on routed rows, so D comes from routed rows
only and is scaled by the routed share. Every read is content-addressed (prompt signature, row, sample): a read that exists is never
repeated, and an accepted candidate's reads are the new tip's parent samples on those rows. Effort is medium everywhere (engine side).
state.json is written after every step; a restart resumes at the next step."""
import argparse, collections, glob, hashlib, json, math, os, random, re, shutil, sys, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(os.path.dirname(HERE), "harness")]
import judge_eb as EB          # noqa: E402
import judge_bh as BH          # noqa: E402  (AM15 decision rule, coordinator 11-10 18:45 CST)
import proposer as PR          # noqa: E402
import l1 as L1                # noqa: E402
import ecomp as ECOMP          # noqa: E402
from v4io import IO            # noqa: E402
from lab import LabMixin, final_key   # noqa: E402  (S2-A proposer lab)
from versions import VersionsMixin, boot_y, WINDOW    # noqa: E402  (Stage 2 authors, versions, L1 step)
from signal_export import SignalMixin, SIGNAL_ROLES   # noqa: E402  (B1 signal export, coordinator 11-11)
from author_g import AuthorGMixin        # noqa: E402  (B2 author G, signal-driven cells)
import signal_cells as SCELLS            # noqa: E402  (B4: calibration of G's predictions after the judge; numpy only)

ALPHA, CONT = 0.10, 0.5        # AM15: BH false discovery rate alpha inside each (author, generation); rung 2 iff lfsr <= CONT (A4, --two-rungs only)
RUNG1_SHARE = 0.25             # rung 1 = routed rows up to a quarter of JUDGE (AM3)
BOX, GRACE = 4500.0, 0.0       # AM16: ONE time box T = the old 3600 s box + 900 s grace (the old hard end); no grace and no reserve constant
                               # (the pre-read reserve is the previous generation's measured parent..route wall, slots_am16)
LITE_M, LITE_FIRST = 4, 2.3
BUNDLE_MIN, BUNDLE_BUILD_AT, BUNDLE_WAIT_TO = 2, 0.6, 0.8   # AM23 build points as shares of the READ window (rung1 start -> box end)   # AM23 (coordinator 11-10 23:30): a bundle needs >= 2 members (declared choice); built when every eligible
                                       # candidate's step 1 is complete, or at this share of the box with the ones complete then
CTX_ERR = "maximum context length"     # coordinator 11-10 23:30: a read over the context is handled like op_skipped (skipped_ctx)   # AM21 LITE (coordinator 11-10 22:45): decision on 4 samples per routed row (declared choice); AM16 reads per
                               # candidate row the first time (sim_am21b), then the measured value
P_TARGET, P_FULL = 2, 8        # parent samples: 2 on JUDGE + SHOW before decisions, topped up to 8 (+ MONITOR) in the background (A8; S2-B: >= 8)
PARENT_PRIO = 0                # the parent step's waited-for reads (coordinator 11-11 02:45: ahead of the LITE completion at priority 1)
PROP_TEMP, PROP_MAX = 0.7, 12000
ROUTE_KEY = "rt1"              # router v1 (engine_api §3, 11-10): new key prefix so no v0 answer is ever reused
UNSURE = (0.1, 0.9)            # router p band reported per entry
N_WRONG, N_RIGHT = 8, 4        # single-source card: up to 8 currently wrong + up to 4 currently right SHOW rows of one source. AM16 item 6: card
                               # numbers (shown wrong / right questions, samples per question, reasoning excerpt length) are the proposer's own
                               # options (version params); these start values are arbitrary
FORMAT_PROBLEMS = ("condition length", "procedure has", "procedure must be a string", "settings must be an object", "unknown field")
NAMING_PROBLEM = re.compile(r"^(condition|procedure): leakage precheck: (bench name|category name)")   # repairable too (coordinator 11-10)
STEPS = ["parent", "propose", "lab", "validate", "route", "rung1", "judge1", "rung2", "judge2", "tip", "report", "l1"]


def esha(edit):
    """engine_api §7 / §8 edit_sha: sha256 of the canonical JSON of the full edit (null -> 74234e98afe7498f), first 16 hex."""
    return hashlib.sha256(json.dumps(edit, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]


def sha16(x):
    return hashlib.sha256(x.encode()).hexdigest()[:16]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def jdump(p, x):
    with open(p + ".tmp", "w") as f:
        json.dump(x, f, ensure_ascii=False, indent=1)
    os.replace(p + ".tmp", p)


class Run(VersionsMixin, LabMixin, SignalMixin, AuthorGMixin):
    P_FULL_SIGNAL = P_FULL
    def __init__(self, a):
        self.a = a
        self.rd = a.run
        os.makedirs(self.rd, exist_ok=True)
        self.rows = PR.load_rows(a.rows)
        self.io = IO(a.root, log=log, loc_rows=[i for i, r in self.rows.items() if r.get("v4_role") in SIGNAL_ROLES])   # B1: whole traces
        self._signal_log = lambda m: log(m)
        self.knobs = json.load(open(a.knobs))
        self.S = self.knobs["settings"]
        self.stock = {k: self.S[k]["stock"] for k in self.S}
        self.role = collections.defaultdict(list)
        for i, r in self.rows.items():
            self.role[r["v4_role"]].append(i)
        for k in self.role:
            self.role[k].sort()
        self.judge_rows = self.role["judge"]
        self.NJ = len(self.judge_rows)
        sp = os.path.join(self.rd, "state.json")
        self.state = json.load(open(sp)) if os.path.exists(sp) else dict(
            created=time.time(), gen=0, tip=dict(id="t000", entries=[], born=0), tips=[], gens={}, authors=["B", "P0"], note="v4 stage 1, from stock")
        self.routes, self.route_p = {}, {}
        self._ecomp = {}
        sub = self.io.submit

        def submit(name, reqs):                                  # engine_api §7 / §8: W and T requests go in batches of their own
            w = [q for q in reqs if q.get("engine") == "W"]
            t = [q for q in reqs if q.get("engine") == "T"]
            if not w and not t:
                return sub(name, reqs)
            e = [q for q in reqs if q.get("engine") not in ("W", "T")]
            return (sub(name, e) if e else 0) + (sub(name + "_W", w) if w else 0) + (sub(name + "_T", t) if t else 0)
        self.io.submit = submit
        self.run_name = os.path.basename(os.path.normpath(self.rd))

    def save(self):
        done = self.apply_annotations()
        self.signal_tick()                                       # B1: signal export when more samples are in (at most every TICK_S)
        jdump(os.path.join(self.rd, "state.json"), self.state)
        for f, line in done:                                     # ledger line and rename only after the state holding the change is on disk
            with open(os.path.join(self.rd, "ledger.jsonl"), "a") as fh:
                fh.write(json.dumps(line, ensure_ascii=False) + "\n")
            os.rename(f, f + time.strftime(".applied_%H%M"))

    def apply_annotations(self):
        """Out-of-band annotation of one decided candidate (coordinator 11-10 22:08: g004c30's server replication lands while generation 5
        runs): RUN/annotate/<name>.json = {gen, cid, set: {field: value}, status_append, ledger: {...}} is applied by the loop itself, so no
        outside writer races state.json; save() then appends one ledger line and renames the file .applied_<HHMM>. Only that candidate's
        fields change; the tip, the running generation and every decision are untouched."""
        d = os.path.join(self.rd, "annotate")
        out = []
        if not os.path.isdir(d):
            return out
        for f in sorted(glob.glob(os.path.join(d, "*.json"))) + sorted(glob.glob(os.path.join(d, "add", "*.json"))):
            try:                                                 # annotate/add/: "add" files only (loops before 11-10 00:50 do not look there)
                A = json.load(open(f))
                if "add" in A:                                   # AM23 hand bundle (H1): a synthetic candidate kept in state["synthetic"], not in any
                    x = dict(A["add"])                           # generation's cands, so no step, report, L1 or carry-in iterates over it;
                    assert x.get("cid") and x.get("entry", {}).get("parts") and x.get("parent"), "add: cid, entry.parts and parent needed"
                    syn = self.state.setdefault("synthetic", [])  # carry_merges merges it at a boundary when it is accepted + carry_merge
                    if any(y["cid"] == x["cid"] for y in syn):
                        raise ValueError(f"synthetic {x['cid']} exists")
                    x.setdefault("kind", "synthetic")
                    x.setdefault("author", "system")
                    x.setdefault("status", A.get("status_append") or "synthetic candidate (annotation)")
                    syn.append(x)
                    line = dict(id=x["cid"], gen=A.get("gen"), annotation=True, synthetic=True, t=time.time(), status=x["status"],
                                accepted=bool(x.get("accepted")), members=[y.get("cid") for y in x["entry"]["parts"]], parent=x["parent"])
                    line.update(dict(A.get("ledger") or {}))
                    out.append((f, line))
                    log(f"  annotation applied: synthetic {x['cid']} ({len(x['entry']['parts'])} entries on {x['parent']}): {x['status']}")
                    continue
                c = next(c for c in self.state["gens"][str(A["gen"])]["cands"] if c["cid"] == A["cid"])
                led = dict(A.get("ledger") or {})                # checked before any change, so a bad file changes nothing
                if not isinstance(A.get("set") or {}, dict):
                    raise ValueError("set must be an object")
                c.update(A.get("set") or {})
                sa = A.get("status_append")
                if sa and not (c.get("status") or "").endswith(sa):
                    c["status"] = (c.get("status") or "") + " | " + sa
                line = dict(id=A["cid"], gen=A["gen"], annotation=True, t=time.time(), status=c.get("status"), accepted=bool(c.get("accepted")))
                line.update(led)                                 # the annotation's own ledger fields win (11-10 23:34: "accepted" twice)
                out.append((f, line))
                log(f"  annotation applied: {A['cid']} (g{A['gen']}): {sa}")
            except Exception as ex:                              # bookkeeping only: never stop the loop on it
                log(f"  annotation {os.path.basename(f)}: {type(ex).__name__}: {ex}")
                os.rename(f, f + ".failed")
        return out

    # ------------------------------------------------------------------ prompts, settings, keys
    def settings_of(self, entries):
        s = dict(self.stock)
        for e in entries:
            s.update(e.get("settings") or {})
        return s

    def sig(self, prefix, settings, ops=None, edit=None):
        """Prompt signature: prefix, non-stock settings and (only when there are any) the harness ops and the weight edit, so reads
        without them keep their keys."""
        s = {k: v for k, v in sorted(settings.items()) if v != self.stock.get(k)}
        return sha16(json.dumps([prefix or None, s] + ([PR.canon_ops(ops)] if ops else []) + ([{"edit": edit}] if edit else []), sort_keys=True))

    def fires(self, e, r):
        if e.get("parts"):                                       # stack candidate: holds where any of its entries holds
            return any(self.fires(x, r) for x in e["parts"])
        return e["cond"] == "always" or bool(self.routes.get((e["cond"], r)))

    @staticmethod
    def flat(entries):
        """Entry list with stack candidates expanded into their entries (in stack order)."""
        return [x for e in entries for x in (e.get("parts") or [e])]

    def view(self, entries, r):
        """(prefix, settings, sig) of the executor on row r under an ordered entry list (routes must be known)."""
        f = [e for e in self.flat(entries) if self.fires(e, r)]
        pre = "\n\n".join(e["procedure"] for e in f if e.get("procedure")) or None
        st = self.settings_of(f)
        return pre, st, self.sig(pre, st, self.ops_of(f), self.edit_of(f))

    @staticmethod
    def edit_of(fired):
        """Weight edit of the entries that hold on a row (engine W): None, the one edit, or the list of edits in entry order."""
        ed = [e["edit"] for e in fired if e.get("edit")]
        return None if not ed else (ed[0] if len(ed) == 1 else ed)

    @staticmethod
    def ops_of(fired):
        """Harness ops of the entries that hold on a row (S2-B): each op once, a later entry's tiles n wins, engine order (PR.merge_ops)."""
        ls = [e.get("ops") for e in fired if e.get("ops")]
        return PR.merge_ops(ls) if ls else []

    def read_key(self, sg, r, s):
        return f"p000|parent|{r}|{s}" if sg == self.stock_sig else f"x|{sg}|{r}|{s}"

    @property
    def stock_sig(self):
        if not hasattr(self, "_stock_sig"):
            self._stock_sig = self.sig(None, dict(self.stock))
        return self._stock_sig

    def read_req(self, entries, r, s, prio):
        pre, st, sg = self.view(entries, r)
        req_st = {k: (None if st[k] == self.stock[k] else st[k]) for k in self.S}      # None = stock (engine_api §3)
        q = dict(key=self.read_key(sg, r, s), mode="read", engine="E", priority=prio, row=r, sample=s, prefix=pre, settings=req_st)
        fired = [e for e in self.flat(entries) if self.fires(e, r)]
        ops, ed = self.ops_of(fired), self.edit_of(fired)
        if ops:
            q["ops"] = ops                                                               # engine_api §7: zoom -> tiles -> read -> check
        if ed:
            q.update(engine="W", edit=ed)                                                # engine_api §7: W worker, apply -> generate -> revert
            if self.a.tfields:                                                           # engine_api §8: exactly the tip's weights -> T servers
                eds, te = (ed if isinstance(ed, list) else [ed]), self.tip_edits()
                n = len(te) if te and eds[:len(te)] == te else 0
                q.update(engine="T" if n and n == len(eds) else "W", edit_sha=esha(ed), weights=self.state["tip"]["id"] if n else "stock", tip_n=n)
        return q, sg

    def tip_edits(self):
        """The current tip's weight edits in entry order (engine_api §8)."""
        return [e["edit"] for e in self.flat(self.state["tip"]["entries"]) if e.get("edit")]

    def samples(self, entries, r, upto=P_FULL):
        """Results of the executor (entries) on row r: the existing samples 0..upto-1."""
        _, _, sg = self.view(entries, r)
        out = []
        for s in range(upto):
            x = self.io.get(self.read_key(sg, r, s))
            if x is not None:
                out.append(x)
        return out

    # ------------------------------------------------------------------ router
    def route_key(self, cond, r):
        return f"{ROUTE_KEY}|{cond}|{r}"

    def route_reqs(self, e, rows):
        """One router call per (row, condition, view); view None = the stock image view (the only one now). The engine caches the
        decision and p; every read of the candidate and of its parent on the row uses that one decision."""
        if e.get("parts"):
            return [q for x in e["parts"] for q in self.route_reqs(x, rows)]
        if e["cond"] == "always":
            return []
        return [dict(key=self.route_key(e["cond"], r), mode="route", engine="E", priority=0, row=r, condition=e["condition"], view=None) for r in rows]

    def load_routes(self, entries, rows):
        for e in self.flat(entries):
            if e["cond"] == "always":
                continue
            for r in rows:
                x = self.io.get(self.route_key(e["cond"], r))
                if x is not None:
                    p = x.get("p")
                    self.routes[(e["cond"], r)] = bool(p >= 0.5) if isinstance(p, (int, float)) else bool(x.get("yes"))
                    self.route_p[(e["cond"], r)] = p

    def route_stats(self, e, rows):
        """routed share, share of routed rows with p in the unsure band, unanswered calls, decisions taken from the constrained choice."""
        if e["cond"] == "always":
            return dict(routed=len(rows), share=1.0, unsure=None, unanswered=0, from_choice=0)
        if e.get("parts"):
            R = [r for r in rows if self.fires(e, r)]
            return dict(routed=len(R), share=len(R) / max(len(rows), 1), unsure=None, unanswered=0, from_choice=0, parts=len(e["parts"]))
        R = [r for r in rows if self.routes.get((e["cond"], r))]
        ps = [self.route_p.get((e["cond"], r)) for r in R]
        un = sum(1 for p in ps if isinstance(p, (int, float)) and UNSURE[0] < p < UNSURE[1])
        miss = sum(1 for r in rows if (e["cond"], r) not in self.routes)
        ch = sum(1 for r in rows if (self.io.get(self.route_key(e["cond"], r)) or {}).get("p_src") == "choice")
        return dict(routed=len(R), share=len(R) / max(len(rows), 1), unsure=(un / len(R)) if R else None, unanswered=miss, from_choice=ch)

    def ensure_routes(self, entries, rows, deadline, label):
        reqs = [q for e in entries for q in self.route_reqs(e, rows)]
        if reqs:
            n = self.io.submit(f"g{self.g:03d}_route_{label}", reqs)
            done, miss = self.io.wait([q["key"] for q in reqs], deadline, label=f"routes {label}")
            if miss:
                log(f"  routes {label}: {len(miss)} unanswered at the deadline (treated as no)")
            log(f"  routes {label}: {len(reqs)} asked, {n} new")
        self.load_routes(entries, rows)

    # ------------------------------------------------------------------ generation steps
    # ------------------------------------------------------------------ cards (coordinator 11-10 00:55: W cards again without a restart)
    def read_cards(self):
        """RUN/cards.json {"w": bool, "e_cards": int, "b_cells": int, "T": seconds} (every key optional), read when a generation is created
        and fixed in G["cards"]; without the file (or a key): --no-w / --e-cards / no cap on arm B's setting cells / --box.
        b_cells (coordinator 11-11 01:40): at most this many arm-B setting cells (0 = none); T: the generation's time box."""
        out = dict(w=not self.a.no_w, e_cards=int(self.a.e_cards or 0), b_cells=None, T=float(self.a.box or BOX), src="arguments")
        p = os.path.join(self.rd, "cards.json")
        if os.path.exists(p):
            try:
                x = json.load(open(p))
                if "w" in x:
                    out["w"] = bool(x["w"])
                if "e_cards" in x:
                    out["e_cards"] = int(x["e_cards"])
                if x.get("b_cells") is not None:
                    out["b_cells"] = max(0, int(x["b_cells"]))
                if x.get("T"):
                    out["T"] = float(x["T"])
                    assert out["T"] > 0
                for k_ in ("g_cells", "g_settings"):                 # B2: caps on author G's weight / setting cells per generation
                    if x.get(k_) is not None:
                        out[k_] = max(0, int(x[k_]))
                for k_ in ("g_token_multiple", "g_token_multiple_source", "g_token_base"):   # B4: the signal cells' one stated budget (the measured
                    if x.get(k_) is not None:                                                # xhigh / medium token multiple; read by signal_cells.read_budget)
                        out[k_] = x[k_]
                if x.get("g_routed") is not None:                    # B4: author G may propose routed cells (a tip condition + the cell)
                    out["g_routed"] = bool(x["g_routed"])
                out["src"] = "cards.json"
            except (ValueError, TypeError, AttributeError, AssertionError) as ex:
                log(f"  cards.json: {type(ex).__name__}: {ex}; using the arguments")
                out = dict(w=not self.a.no_w, e_cards=int(self.a.e_cards or 0), b_cells=None, T=float(self.a.box or BOX), src="arguments")
        return out

    def stamp_cards(self, G):
        """Once per generation (at its creation; a generation created by older code gets them at its first resume)."""
        if "cards" not in G:
            G["cards"] = self.read_cards()
            if G["cards"]["src"] == "cards.json":
                log(f"  cards (cards.json): W {'on' if G['cards']['w'] else 'off'}, {G['cards']['e_cards']} E cards"
                    + (f", arm B setting cells <= {G['cards']['b_cells']}" if G["cards"].get("b_cells") is not None else "")
                    + f", box T {G['cards']['T']:.0f} s")
        return G["cards"]

    def _cards(self):
        return (self.state["gens"].get(str(getattr(self, "g", 0))) or {}).get("cards")

    def w_on(self):
        c = self._cards()
        return bool(c["w"]) if c else not self.a.no_w

    def e_cards_n(self):
        c = self._cards()
        return int(c["e_cards"]) if c else int(self.a.e_cards or 0)

    def b_cells_cap(self):
        c = self._cards()
        return None if not c or c.get("b_cells") is None else int(c["b_cells"])

    def box_T(self):
        """The generation's time box: G["cards"]["T"] (RUN/cards.json) when set at its creation, else --box (default BOX)."""
        c = self._cards()
        return float(c["T"]) if c and c.get("T") else float(self.a.box or BOX)

    def cards_src(self):
        c = self._cards()
        return f"--e-cards" if not c or c.get("src") == "arguments" else "cards.json"

    def G(self):
        return self.state["gens"][str(self.g)]

    def step_parent(self):
        """New tip's parent samples: P_TARGET on SHOW + JUDGE at priority PARENT_PRIO (waited for), top-up to P_FULL and MONITOR at priority 5
        (not waited). PARENT_PRIO 0 (coordinator 11-11 02:45): the step blocks the generation, so its reads go ahead of the LITE completion
        (priority 1, submitted just before); the read keys do not hold the priority, so cached samples are reused as before."""
        tip = self.state["tip"]["entries"]
        allr = self.role["show"] + self.role["judge"] + self.role["monitor"]
        self.ensure_routes(tip, allr, self.deadline, "tip")
        need, bg = [], []
        for r in self.role["show"] + self.role["judge"]:
            for s in range(P_FULL):
                q, _ = self.read_req(tip, r, s, PARENT_PRIO if s < P_TARGET else 5)
                (need if s < P_TARGET else bg).append(q)
        for r in self.role["monitor"]:
            for s in range(P_FULL):
                bg.append(self.read_req(tip, r, s, 5)[0])
        n1 = self.io.submit(f"g{self.g:03d}_parent", need)
        n2 = self.io.submit(f"g{self.g:03d}_parentbg", bg)
        log(f"  parent of {self.state['tip']['id']}: {n1} new reads at priority {PARENT_PRIO}, {n2} in the background")
        show_keys = [q["key"] for q in need if self.rows[q["row"]]["v4_role"] == "show"]
        done, miss = self.io.wait(show_keys, self.deadline, label="parent SHOW")
        self.G()["parent"] = dict(new=n1, bg=n2, show_missing=len(miss))

    # ------------------------------------------------------------------ AM16: sizes from measurement, arm B ranges (coordinator 11-10)
    def gen_window(self, G):
        """(t0, t1) of a generation: its start and its end (G["t_end"], else t0 + the sum of its step walls)."""
        t0 = float(G.get("t0") or 0.0)
        return t0, float(G.get("t_end") or (t0 + sum((G.get("times") or {}).values())))

    def e_capacity(self, t0, t1):
        """E capacity in generated tok/s over the window, measured while backlogged (stats.jsonl, one record per server every few s): per
        E or T server (engine_api §8) the mean gen_tok_s over its records with inflight >= cap; a server that was never backlogged counts with its peak rate.
        Servers count when their last record in the window says state up. None if the file has no E record in the window."""
        p = os.path.join(self.a.root, "stats.jsonl")
        if not os.path.exists(p):
            return None
        sat, peak, last = collections.defaultdict(list), collections.defaultdict(float), {}
        with open(p) as fh:
            for l in fh:
                if '"role": "E"' not in l and '"role": "T"' not in l:          # T servers (engine_api §8) are executor capacity too
                    continue
                try:
                    r = json.loads(l)
                except ValueError:
                    continue
                if not (t0 <= float(r.get("t") or 0) <= t1):
                    continue
                s, v = r.get("server"), r.get("gen_tok_s")
                last[s] = r.get("state")
                if v is None:
                    continue
                peak[s] = max(peak[s], float(v))
                if r.get("cap") and (r.get("inflight") or 0) >= r["cap"]:
                    sat[s].append(float(v))
        up = [s for s, st in last.items() if st == "up"]
        self._e_up = len(up)
        if not up:
            return None
        return sum((sum(sat[s]) / len(sat[s])) if sat[s] else peak[s] for s in up)

    def launch_capacity(self):
        """Generation 1 (no earlier generation): E capacity from the stock launch cache, its generated tokens over its span."""
        xs = [x for k, x in self.io.res.items() if k.startswith("p000|parent|") and x.get("t_done") and x.get("tokens")]
        if len(xs) < 2:
            return None
        span = max(x["t_done"] for x in xs) - min(x.get("t_submit") or x["t_done"] for x in xs)
        return sum(x["tokens"] for x in xs) / span if span > 0 else None

    def w_capacity(self, t0, t1):
        """W reads per second summed over the W cards, measured on the W results finished in the window: per W server its reads over the
        span from its first to its last result there (apply / revert time between cells is inside the span). Falls back to every W
        result the run knows (e.g. the engine's own W checks) when the window has none. (rate, cards)."""
        def rate(xs):
            by = collections.defaultdict(list)
            for x in xs:
                by[x.get("server")].append(float(x["t_done"]))
            r = 0.0
            for s, ts in by.items():
                if len(ts) >= 2 and max(ts) > min(ts):
                    r += (len(ts) - 1) / (max(ts) - min(ts))
            return r, len(by)
        allw = [x for x in self.io.res.values() if str(x.get("server", "")).startswith("W") and x.get("t_done")]
        runw = [x for x in allw if str(x.get("key", "")).startswith("x|")]      # this run's cell reads (not the engine's own W checks)
        r, n = rate([x for x in runw if t0 <= x["t_done"] <= t1])
        if not r:
            r, n = rate(runw)
        if not r:
            r, n = rate(allw)
        cards = len(glob.glob(os.path.join(self.a.root, "w", "W*.up.json"))) or n
        return (r / max(n, 1) * cards if r else None), cards

    def new_read_tokens(self, gk):
        """{cid: generated tokens} of each proposer candidate's JUDGE and SHOW reads in generation gk that were NEW there (result finished
        after the generation started; a read cached from earlier costs nothing; a pending read counts with the candidate's mean tokens per
        read). Routes and keys are rebuilt on that generation's parent tip, as in finalise. None when the parent tip is unknown."""
        G = self.state["gens"][str(gk)]
        V = [c for c in G.get("cands", []) if c.get("rung") and c.get("entry") and c["author"] not in ("B", "G", "system")]
        cur_g, cur_tip, cur_r = getattr(self, "g", None), self.state["tip"], (self.routes, self.route_p)
        tips = {t["id"]: t for t in self.state["tips"] + [cur_tip]}
        if not V or V[0]["parent"] not in tips:
            return None
        out, t0 = {}, float(G.get("t0") or 0.0)
        try:
            self.g, self.state["tip"] = int(gk), tips[V[0]["parent"]]
            self.routes, self.route_p = {}, {}
            rows = self.role["show"] + self.judge_rows
            self.load_routes(self.state["tip"]["entries"], rows)
            self.load_routes([c["entry"] for c in V], rows)
            for c in V:
                tot = 0.0
                for r, pr in [(r, 1) for r in self.rung_rows(c, 1)] + [(r, 5) for r in self.role["show"] if self.fires(c["entry"], r)]:
                    x = self.io.get(self.read_req(self.cand_entries(c), r, 0, pr)[0]["key"])
                    if x is None:
                        tot += float(c.get("tok_cand") or 0.0)
                    elif float(x.get("t_done") or 0.0) >= t0:
                        tot += float(x.get("tokens") or 0.0)
                out[c["cid"]] = tot
        finally:
            self.g, self.state["tip"] = cur_g, cur_tip
            self.routes, self.route_p = cur_r
        return out

    def slot_costs(self, gk):
        """Per proposer slot of a finished generation gk: new read tokens (its candidate's JUDGE + SHOW reads, new_read_tokens), and the
        pre-read wall per slot (propose + lab + validate + route walls over the slots: the time the proposer, lab and router work took).
        The proposer answer, lab chat and lab read tokens per slot are reported (they are what fills that pre-read time). Also tokens per
        read of arm B's setting cells and of the tip."""
        G = self.state["gens"][str(gk)]
        props = [p for p in G.get("props", []) if p.get("author") not in ("B", "G", "system")]
        n = len(props)
        nr = self.new_read_tokens(gk) or {}
        pt = lt = 0.0
        for p in props:
            x = self.io.get(p.get("key")) if p.get("key") else None
            pt += float((x or {}).get("tokens") or 0)
            for t in p.get("lab") or []:
                lt += float(t.get("answer_tokens") or 0) + float(t.get("read_tokens") or 0)
        bs = [c for c in G.get("cands", []) if c.get("author") == "B" and c.get("kind") == "setting" and c.get("tok_cand")]
        tp = [c["tok_parent"] for c in G.get("cands", []) if c.get("tok_parent")]
        pre = sum(v for k, v in (G.get("times") or {}).items() if k in ("propose", "lab", "validate", "route"))
        return dict(slots=n, read_tok=(sum(nr.values()) / n) if n else None, pre_s=(pre / n) if n else None, prop_tok=(pt / n) if n else None,
                    lab_tok=(lt / n) if n else None, tok_b=(sum(c["tok_cand"] for c in bs) / len(bs)) if bs else None,
                    tok_tip=(sum(tp) / len(tp)) if tp else None, pre_total_s=pre)

    def slots_am16(self):
        """AM16 item 1: slots from measurement under the one time box T. E capacity cap = generated tok/s while backlogged in the previous
        generation (generation 1: the stock launch cache). After this generation's parent step (its wall is known), the box holds
        cap x (T - parent wall) tokens. Fixed costs: the parent reads still to come (tip top-up), arm B's setting cells (all JUDGE rows),
        the stack (its parts' routed rows). A proposer slot costs its new read tokens plus cap x its measured pre-read seconds (the time
        the proposer, lab and router work of one slot took in the previous generation, so that more slots also lengthen the pre-read
        phase). Proposer slots = (cap x (T - parent wall) - fixed) / that, at least the L1 floors; generation 1 (nothing measured): an
        always entry's JUDGE reads per slot and no pre-read time. Weight cells = measured W reads per s x (T - parent wall - measured
        pre-read of the previous generation) / JUDGE rows. No cap, no assumed rate."""
        T = self.box_T()
        prev = [int(k) for k, G in self.state["gens"].items() if int(k) < self.g and "report" in (G.get("done") or [])]
        gp = max(prev) if prev else None
        Gp = self.state["gens"][str(gp)] if gp else None
        cap, src = None, "stock launch cache"
        if Gp:
            cap, src = self.e_capacity(*self.gen_window(Gp)), f"g{gp} backlogged"
            if cap and self.e_cards_n() and getattr(self, "_e_up", 0):   # coordinator 11-10 22:08: size for the cards that will read
                cap, src = (cap / self._e_up * self.e_cards_n(),
                            f"g{gp} backlogged, {cap / self._e_up:.0f} tok/s per card on {self._e_up} cards x {self.e_cards_n()} cards "
                            f"({self.cards_src()})")
        if not cap:
            cap, src = self.launch_capacity(), "stock launch cache"
        if not cap:
            cap, src = self.io.tok_rate() or 0.0, "dispatcher's current rate"
        sc = self.slot_costs(gp) if Gp else {}
        st = [x.get("tokens") for k, x in self.io.res.items() if k.startswith("p000|parent|") and x.get("tokens")]
        tok_read = sc.get("tok_tip") or ((sum(st) / len(st)) if st else 0.0)
        G = self.G()
        par_wall = float((G.get("times") or {}).get("parent") or 0.0)
        budget = cap * max(0.0, T - par_wall)
        par = G.get("parent") or {}
        bc = self.b_cells()
        if self.b_cells_cap() is not None:                       # RUN/cards.json b_cells: arm B's setting cells capped (0 = none)
            bc = bc[:self.b_cells_cap()]
        nb = len(bc)
        tip = self.state["tip"]["entries"]
        b_new = 0                                                # arm B's setting reads that are really new: a row where the tip + cell view
        for k, v in bc:                                          # was read before (content-addressed) costs nothing
            e = dict(condition="always", cond="always", procedure=None, settings={k: v}, ops=[])
            b_new += sum(1 for r in self.judge_rows if self.read_req(tip + [e], r, 0, 1)[0]["key"] not in self.io.res)
        fac, fac_from = self.lite_factor(gp) if self.lite_on() else (1.0, "one sample per candidate row")
        f_parent = (par.get("new", 0) + par.get("bg", 0)) * tok_read
        f_b = b_new * (sc.get("tok_b") or tok_read) * fac
        f_stack = sum(int(c.get("routed") or 0) for c in self._stack_cands()) * tok_read * fac
        allc = {c["cid"]: c for Gx in self.state["gens"].values() for c in Gx.get("cands", [])}
        f_carry = sum(1 for i in (G.get("carry") or []) if i in allc for k in self.cand_keys(allc[i]) if k not in self.io.res) * tok_read
        if sc.get("read_tok") is not None:
            per, per_from = sc["read_tok"] * fac + cap * (sc.get("pre_s") or 0.0), f"g{gp}"
        else:
            per, per_from = self.NJ * tok_read * fac, "generation-1 bound (an always entry's JUDGE reads)"
        floors = self.l1_floor_total() if hasattr(self, "l1_floor_total") else 1
        npz = max(floors, int((budget - f_parent - f_b - f_stack - f_carry) / per)) if per > 0 else floors
        wr, cards = self.w_capacity(*self.gen_window(Gp)) if Gp else self.w_capacity(0.0, 0.0)
        tw = max(0.0, T - par_wall - float(sc.get("pre_total_s") or 0.0))
        nw = int(wr * tw / self.NJ) if (wr and self.w_on()) else 0
        info = dict(T=T, capacity_tok_s=round(cap, 1), capacity_from=src, parent_wall_s=round(par_wall), budget_tok=round(budget), tok_read=round(tok_read),
                    per_slot_tok=round(per), per_slot_from=per_from, per_slot_parts={k: (round(v, 1) if isinstance(v, float) else v) for k, v in sc.items()},
                    fixed_tok=dict(parent=round(f_parent), b_settings=round(f_b), stack=round(f_stack), carry=round(f_carry)), b_new_reads=b_new, floors=floors, w_reads_s=round(wr or 0.0, 4),
                    w_cards=cards, w_time_s=round(tw), reads_per_cand_row=round(fac, 3), reads_per_cand_row_from=fac_from)
        log(f"  slots (AM16): capacity {cap:.0f} tok/s ({src}) x (T {T:.0f} s - parent {par_wall:.0f} s) = {budget / 1e6:.1f} M tok; fixed: parent "
            f"{f_parent / 1e6:.2f} M, B settings {nb} cells {b_new} new reads {f_b / 1e6:.2f} M, stack {f_stack / 1e6:.2f} M, carry-in {f_carry / 1e6:.2f} M; per proposer slot {per / 1e3:.0f} k tok "
            f"({per_from}: new reads {(sc.get('read_tok') or 0) / 1e3:.0f} k + pre-read {sc.get('pre_s') or 0:.0f} s x capacity; proposer {(sc.get('prop_tok') or 0) / 1e3:.0f} k, "
            f"lab {(sc.get('lab_tok') or 0) / 1e3:.0f} k tok; candidate reads x{fac:.2f} per routed row, {fac_from}) -> {npz} proposer slots (floors {floors}); W {wr or 0:.3f} reads/s on {cards} cards x {tw:.0f} s -> {nw} weight cells")
        return {"B": nb, "P0": npz, "BW": nw, "am16": info}

    def lite_factor(self, gp):
        """AM21 LITE: reads per candidate row (samples read / routed rows read) of the previous generation if it ran under LITE (cancelled
        reads not counted), else LITE_FIRST (sim_am21b: 2.3)."""
        Gp = self.state["gens"].get(str(gp)) if gp else None
        if Gp and Gp.get("am21") == "lite":
            V = [c for c in Gp.get("cands", []) if c.get("rung") and c.get("am21")]
            rows = sum(int((c.get("rows_read") or {}).get("1") or 0) for c in V)
            reads = sum(int((c.get("rows_read") or {}).get("1") or 0) * int(c["am21"]["n"]) - int(c.get("cancelled_reads") or 0) for c in V)
            if rows > 0:
                return reads / rows, f"measured in g{gp} ({reads} reads on {rows} candidate rows)"
        return LITE_FIRST, f"AM21 LITE first time ({LITE_FIRST})"

    def _stack_cands(self):
        st_ = self.stack_entry()
        if not st_:
            return []
        ids = {x.get("cid") for x in st_["parts"]}
        return [c for G in self.state["gens"].values() for c in G.get("cands", []) if c["cid"] in ids]

    # arm B ranges (AM16 item 4): knobs.json "ranges_B"; proposer menus ("settings") stay byte-identical
    def b_ranges(self):
        return (self.knobs.get("ranges_B") or {})

    def pow2(self, lo, hi):
        out, v = [], 1
        while v <= hi:
            if v >= lo:
                out.append(v)
            v *= 2
        return out

    def b_cells_am16(self):
        """Arm B setting cells from the declared ranges: per knob one blind draw above and one below the tip's current value where the range
        allows (powers of two uniformly among those above / below; an interval knob uniformly on that side, 2 decimals); never a value
        already tried on this tip. Seeded by (generation, seed), so a resumed step draws the same cells."""
        R = self.b_ranges().get("settings") or {}
        base = self.settings_of([e for e in self.state["tip"]["entries"] if e["cond"] == "always"])
        tried = {(k, json.dumps(v)) for c in self.tip_cands() if c["author"] == "B" for k, v in (c.get("entry") or {}).get("settings", {}).items()}
        rng = random.Random(f"am16b:{self.g}:{self.a.seed}")
        out = []
        for k in sorted(R):
            d = R[k]
            cur = base.get(k)
            if "pow2" in d:
                lo, hi = d["pow2"]
                vals = self.pow2(lo, hi)
                c = cur if cur is not None else d.get("null_is", lo)
                for side in (lambda v: v > c, lambda v: v < c):
                    xs = [v for v in vals if side(v) and (k, json.dumps(v)) not in tried]
                    if xs:
                        out.append((k, rng.choice(xs)))
            elif "interval" in d:
                lo, hi = d["interval"]
                c = cur if cur is not None else d.get("null_is", self.stock.get(k))
                for a_, b_ in ((c, hi), (lo, c)):
                    if b_ - a_ > 0.01:
                        for _ in range(20):
                            v = round(rng.uniform(a_, b_), 2)
                            if a_ < v < b_ or (v == b_ and b_ != c) or (v == a_ and a_ != c):
                                if (k, json.dumps(v)) not in tried:
                                    out.append((k, v))
                                    break
        return out

    def b_setting_problems(self, cand):
        """Arm B setting cells are checked against the declared ranges (not the proposers' value lists)."""
        R = self.b_ranges().get("settings") or {}
        P = []
        for k, v in (cand.get("settings") or {}).items():
            d = R.get(k)
            if d is None:
                P.append(f"settings: {k} is not a declared arm-B range")
            elif "pow2" in d and not (v is None or v in self.pow2(*d["pow2"])):
                P.append(f"settings: {k} = {v} is not a power of two in {d['pow2']}")
            elif "interval" in d and not (v is None or d["interval"][0] <= v <= d["interval"][1]):
                P.append(f"settings: {k} = {v} is outside {d['interval']}")
        return P

    def band_cell(self, rng):
        """One blind ranged weight cell (AM16 item 4): an op among the ranged ones, a contiguous band [a, b] uniform over all bands of the
        declared layer range, a dose log-uniform in the declared interval with a random sign (alpha = 1 + signed dose)."""
        W = self.b_ranges().get("weights") or {}
        L0, L1_ = W["layers"]
        lo, hi = W["dose"]
        bands = [(a, b) for a in range(L0, L1_ + 1) for b in range(a, L1_ + 1)]
        a, b = bands[rng.randrange(len(bands))]
        dose = math.exp(rng.uniform(math.log(lo), math.log(hi))) * rng.choice((-1, 1))
        al = round(1.0 + dose, 3)
        op = rng.choice(W["ops"])
        if op in ("scale_mlp", "scale_attn"):
            ed = dict(op="scale", layers=[a, b], writer=op.split("_")[1], alpha=al)
            name = f"scale L{a}-{b} {ed['writer']} x{al}"
        elif op == "attn_temp":
            ed = dict(op="attn_temp", first=a, last=b, alpha=al)
            name = f"attn_temp L{a}-{b} x{al}"
        else:
            ed = dict(op="vision_scale", alpha=al)
            name = f"vision_scale x{al}"
        return name, ed

    W_OPS = ("scale", "skip", "attn_temp", "vision_scale", "amplify", "project_out", "surgeon")

    def w_menu(self):
        """Arm B's weight cells (engine W): [(group, name, edit)]. Group "band" = knobs.json weights_band_cells; group "dir:<name>" = the
        cells in <root>/cells/<name>/cells.json. A cells.json entry is an edit form ({"op": ...}) or {"name", "edit"}; a file of raw
        Surgeon primitives (entries with "param") is ONE cell {"op": "surgeon", "file": path}."""
        out, K = [], self.knobs.get("weights_band_cells") or {}
        for L in (K.get("scale") or {}).get("layers", []):
            for w in K["scale"]["writer"]:
                for a in K["scale"]["alpha"]:
                    out.append(("band", f"scale L{L[0]}-{L[1]} {w} x{a}", dict(op="scale", layers=list(L), writer=w, alpha=a)))
        for L in (K.get("skip") or {}).get("layer", []):
            for w in K["skip"]["writer"]:
                out.append(("band", f"skip L{L} {w}", dict(op="skip", layer=L, writer=w)))
        for b in (K.get("attn_temp") or {}).get("band", []):
            for a in K["attn_temp"]["alpha"]:
                out.append(("band", f"attn_temp L{b[0]}-{b[1]} x{a}", dict(op="attn_temp", first=b[0], last=b[1], alpha=a)))
        for a in (K.get("vision_scale") or {}).get("alpha", []):
            out.append(("band", f"vision_scale x{a}", dict(op="vision_scale", alpha=a)))
        for f in sorted(glob.glob(os.path.join(self.a.root, "cells", "*", "cells.json"))):
            name = os.path.basename(os.path.dirname(f))
            try:
                xs = json.load(open(f))
                xs = xs.get("cells", xs) if isinstance(xs, dict) else xs
                if any(isinstance(x, dict) and "param" in x for x in xs):
                    out.append((f"dir:{name}", name, dict(op="surgeon", file=f)))
                    continue
                for i, x in enumerate(xs):
                    ed = x.get("edit", x) if isinstance(x, dict) else None
                    if isinstance(ed, dict) and ed.get("op") in self.W_OPS:
                        ed = {k: v for k, v in ed.items() if k != "name"}
                        if ed["op"] == "surgeon" and isinstance(ed.get("file"), str) and not os.path.isabs(ed["file"]):
                            ed["file"] = os.path.join(os.path.dirname(f), ed["file"])
                        out.append((f"dir:{name}", x.get("name") or x.get("cell") or f"{name}#{i}", ed))
            except (OSError, ValueError, AttributeError, TypeError) as ex:
                log(f"  weight cells: {f} unreadable ({ex}); skipped")
        return out

    def w_fresh(self):
        """Weight cells not yet read against the current tip (a tip that holds weight edits gets the cell on top: a list edit, engine_api §7)."""
        if not self.w_on():
            return []
        tried = {json.dumps(c["entry"]["edit"], sort_keys=True) for c in self.tip_cands() if (c.get("entry") or {}).get("edit")}
        return [x for x in self.w_menu() if json.dumps(x[2], sort_keys=True) not in tried]

    @classmethod
    def edit_problems(cls, ed):
        if not isinstance(ed, dict) or ed.get("op") not in cls.W_OPS:
            return [f"edit: op must be one of {', '.join(cls.W_OPS)}"]
        return []

    def b_cells(self):
        if self.a.am16:
            return self.b_cells_am16()
        base = self.settings_of([e for e in self.state["tip"]["entries"] if e["cond"] == "always"])
        tried = {(k, json.dumps(v)) for c in self.tip_cands() if c["author"] == "B" for k, v in (c.get("entry") or {}).get("settings", {}).items()}
        return [(k, v) for k in sorted(self.S) for v in self.S[k]["values"] if v != base[k] and (k, json.dumps(v)) not in tried]

    def slots(self):
        if self.a.mini:
            return {"B": 1, "P0": 1}
        if self.a.slots:
            return {"B": self.a.slots, "P0": self.a.slots}
        if self.a.am16:
            return self.slots_am16()
        rate = max(self.io.tok_rate() or 0.0, self.a.assume_rate)             # tok/s over all E servers (idle servers under-report capacity)
        sph = rate * 3600 / self.a.tok_per_read                               # samples per hour
        reads = sph * max(0.0, self.box_T() - 900) / 3600              # 15 min of the ceiling for proposing, routing and judging
        sh = self.a.p0_share * self.NJ
        if self.a.two_rungs:
            k1 = int(math.ceil(RUNG1_SHARE * self.NJ))
            rb = k1 + 0.5 * (self.NJ - k1)                                    # an "always" entry: rung 1, then half continue
            rp = min(sh, k1) + 0.5 * max(0.0, sh - k1)                         # a P0 entry at the assumed routed share
        else:
            rb, rp = float(self.NJ), sh                                        # one rung: every routed JUDGE row is read once
        nb = len(self.b_cells())
        npz = max(1, min(self.a.max_p0, int((reads - nb * rb) / rp)))
        est = nb * rb + npz * rp
        log(f"  slots: {rate:.0f} tok/s = {sph:.0f} samples/h -> {reads:.0f} reads under the ceiling; B {nb} cells x {rb:.0f} reads, P0 {rp:.0f} reads each -> P0 {npz}; "
            f"expected {est:.0f} reads = {60 * est / max(sph, 1):.0f} min of capacity")
        return {"B": nb, "P0": npz, "est_reads": round(est), "est_min": round(60 * est / max(sph, 1), 1)}

    def tip_cands(self):
        """Candidates of earlier generations judged against the current tip."""
        t = self.state["tip"]["id"]
        return [c for k, G in self.state["gens"].items() if int(k) < self.g for c in G.get("cands", []) if c.get("parent") == t]

    def tip_view(self):
        return {"settings": {}, "entries": [dict(condition=e["condition"], procedure=e.get("procedure"), settings=e.get("settings") or {}, ops=e.get("ops") or [])
                                            for e in self.state["tip"]["entries"]]}

    def residual(self, parent_show):
        """SHOW residual mass per source: sum over its SHOW rows of 1 - mean correctness of the tip's cached samples."""
        res = collections.defaultdict(float)
        for r, vs in parent_show.items():
            vs = [s for s in vs if s.get("ok") is not None]
            if vs:
                res[self.rows[r]["bench"]] += 1.0 - sum(bool(s["ok"]) for s in vs) / len(vs)
        return dict(res)

    @staticmethod
    def alloc_sources(n, resid):
        """Proposer slots per source in proportion to residual mass, largest-remainder rounding; every source with residual gets >= 1
        slot when n allows (taken from the source most above its quota)."""
        src = sorted(b for b, v in resid.items() if v > 0)
        if n <= 0 or not src:
            return {}
        tot = sum(resid[b] for b in src)
        q = {b: n * resid[b] / tot for b in src}
        a = {b: int(math.floor(q[b])) for b in src}
        for b in sorted(src, key=lambda b: (-(q[b] - a[b]), b))[: n - sum(a.values())]:
            a[b] += 1
        if n >= len(src):
            for b in sorted((b for b in src if a[b] == 0), key=lambda b: (-resid[b], b)):
                donor = max((d for d in src if a[d] >= 2), key=lambda d: (a[d] - q[d], d))
                a[donor] -= 1
                a[b] += 1
        return {b: a[b] for b in src if a[b] > 0}

    def step_propose(self):
        if self.a.no_versions:
            return self.step_propose_s1()
        return self.v_step_propose()                             # versions.py: P0, incumbent, challengers, B cells, weight cells, stack, drafts

    def step_l1(self):
        if self.a.no_versions:
            return None
        return self.v_step_l1()

    b_in_draw = False                                            # B's setting cells stay outside the Thompson draw (read cost 444 vs ~111)

    def w_kind_slots(self, kind):
        """Arm B weight slots of one kind ("band": ranged / band cells, "dir": direction cells) over the last WINDOW generations: per slot
        the accepted contribution (share x d, 0 when not accepted) and its se_used."""
        m, s_ = [], []
        for gk, G in self.state["gens"].items():
            if not (self.g - WINDOW <= int(gk) < self.g):
                continue
            for c in G.get("cands", []):
                if c.get("author") != "B" or not (c.get("entry") or {}).get("edit"):
                    continue
                k = "dir" if str(c.get("cell_group", "")).startswith("dir:") else "band"
                if k != kind:
                    continue
                j = c.get("final") or c.get("judge") or {}
                acc = bool(c.get("accepted"))
                m.append(float(c.get("share") or 1.0) * float(j.get("D") or 0.0) if acc else 0.0)
                s_.append(float(c.get("share") or 1.0) * float(j.get("se_used") or j.get("se") or 0.0) if acc else 0.0)
        return m, s_

    def b_pooled_mean(self):
        """Pooled arm-B mean accepted contribution per slot over the last WINDOW generations (settings and weight cells)."""
        v = []
        for gk, G in self.state["gens"].items():
            if self.g - WINDOW <= int(gk) < self.g:
                for c in G.get("cands", []):
                    if c.get("author") == "B" and c.get("rung"):
                        j = c.get("final") or c.get("judge") or {}
                        v.append(float(c.get("share") or 1.0) * float(j.get("D") or 0.0) if c.get("accepted") else 0.0)
        return sum(v) / len(v) if v else 0.0

    def add_w_props_am16(self, props, sl):
        """AM16 items 4-5: arm B's weight cells on W. Number = sl["BW"] (measured W capacity). Per cell a Thompson draw between the kinds
        "band" (ranged cells: a contiguous band and a log-uniform signed dose, band_cell) and "dir" (direction cells from cells/*/cells.json
        not yet read on this tip): one Bayesian-bootstrap draw of each kind's accepted contribution per slot over the last WINDOW
        generations, with one pseudo-slot at the pooled arm-B mean per kind (so a kind with no accepts keeps a chance), argmax."""
        nw = 0 if not self.w_on() else int(sl.get("BW") or 0)
        if not nw:
            return
        dirs = [x for x in self.w_fresh() if x[0].startswith("dir:")]
        tried = {json.dumps(c["entry"]["edit"], sort_keys=True) for c in self.tip_cands() if (c.get("entry") or {}).get("edit")}
        pr = random.Random(f"am16w:{self.g}:{self.a.seed}")
        nr = np.random.default_rng(int(sha16(f"am16wk:{self.g}:{self.a.seed}"), 16))
        pm = self.b_pooled_mean()
        data = {k: self.w_kind_slots(k) for k in ("band", "dir")}
        b0 = sum(1 for p in props if p["author"] == "B")
        picks = collections.Counter()
        for k in range(nw):
            kinds = ["band"] + (["dir"] if dirs else [])
            y = {kd: float(boot_y(np.array(data[kd][0] + [pm]), np.array(data[kd][1] + [0.0]), 1, nr)[0]) for kd in kinds}
            top = [kd for kd in kinds if y[kd] == max(y.values())]
            kd = top[int(nr.integers(len(top)))]
            if kd == "dir":
                x = dirs.pop(int(nr.integers(len(dirs))))
                g_, name, ed = x
            else:
                for _ in range(50):
                    name, ed = self.band_cell(pr)
                    if json.dumps(ed, sort_keys=True) not in tried:
                        break
                g_ = "range"
            tried.add(json.dumps(ed, sort_keys=True))
            picks[kd] += 1
            props.append(dict(author="B", slot=b0 + k, cand=dict(why=f"blind weight cell ({g_}): {name}", condition="always", procedure=None,
                                                                 settings={}, ops=[], edit=ed, cell=name, cell_group=g_)))
        log(f"  B weight cells (engine W, AM16): {nw} cells, Thompson by kind {dict(picks)} (window slots band {len(data['band'][0])}, dir {len(data['dir'][0])}; "
            f"pseudo-slot at the pooled arm-B mean {pm:+.4f}); {len(dirs)} fresh direction cells left")

    def w_budget_after_g(self, props, sl):
        """B2: author G's props first; arm B's weight cells get the W budget G left (coordinator 11-11: before AM19, G takes its ranked
        cells first, B the rest)."""
        used = self.add_g_props(props, sl)
        bw = int(sl.get("BW") or 0) if self.a.am16 else int(sl.get("BW", self.a.w_cells) or 0)
        return dict(sl, BW=max(0, bw - used))

    def add_w_props(self, props, rng, sl):
        """S2-C: arm B's weight cells on engine W, blind draw (a menu group first, then a cell in it), at most once per tip."""
        if self.a.am16:
            return self.add_w_props_am16(props, sl)
        wf = self.w_fresh()
        nw = 0 if not self.w_on() else sl.get("BW", self.a.w_cells)
        groups, nd = sorted({x[0] for x in wf}), min(nw, len(wf))
        b0 = sum(1 for p in props if p["author"] == "B")
        for k in range(nd):
            g_ = rng.choice(sorted({x[0] for x in wf}))
            x = rng.choice([y for y in wf if y[0] == g_])
            wf.remove(x)
            props.append(dict(author="B", slot=b0 + k, cand=dict(why=f"blind weight cell ({x[0]}): {x[1]}", condition="always", procedure=None,
                                                                 settings={}, ops=[], edit=x[2], cell=x[1], cell_group=x[0])))
        if nw:
            log(f"  B weight cells (engine W): {nd} of {nw} slots drawn; menu groups {groups}; {len(wf)} fresh cells left")

    def step_propose_s1(self):
        G = self.G()
        sl = G.setdefault("slots", self.slots())
        tip = self.state["tip"]["entries"]
        tv = self.tip_view()
        pf = os.path.join(self.rd, f"prompts_g{self.g:03d}.jsonl")
        if os.path.exists(pf):                                       # resume inside the step: the same prompts and keys as before
            reqs = [json.loads(l) for l in open(pf) if l.strip()]
            props = G["props_p0"]
        else:
            parent_show = {r: self.samples(tip, r) for r in self.role["show"]}
            resid = self.residual(parent_show)
            alloc = self.alloc_sources(sl["P0"], resid)
            G["sources"] = dict(residual={b: round(v, 2) for b, v in sorted(resid.items())}, alloc=alloc)
            log(f"  P0 slots by source (SHOW residual): " + ", ".join(f"{b} {alloc.get(b, 0)} ({resid[b]:.1f})" for b in sorted(resid)))
            reqs, props, k = [], [], 0
            for b in sorted(alloc):
                for i in range(alloc[b]):
                    ids = PR.cards_source(self.rows, parent_show, b, i, seed=f"g{self.g}:{self.a.seed}", n_wrong=N_WRONG, n_right=N_RIGHT)
                    system, user = PR.card_prompt(self.rows, parent_show, ids, tv, self.knobs)
                    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
                    key = f"chat|{self.run_name}|g{self.g:03d}|P0|{k}|{sha16(json.dumps(msgs))[:8]}"
                    reqs.append(dict(key=key, mode="chat", engine="E", priority=0, messages=msgs, temperature=PROP_TEMP, max_tokens=PROP_MAX))
                    nw = sum(1 for r in ids if not PR._is_right(PR._samples(parent_show, r)))
                    props.append(dict(author="P0", slot=k, key=key, card=ids, card_source=b, card_wrong=nw, card_right=len(ids) - nw,
                                      prompt_chars=len(system) + len(user)))
                    k += 1
            with open(pf + ".tmp", "w") as f:
                for q in reqs:
                    f.write(json.dumps(q, ensure_ascii=False) + "\n")
            os.replace(pf + ".tmp", pf)
            G["props_p0"] = props
            self.save()
        n = self.io.submit(f"g{self.g:03d}_chat", reqs)
        props = [dict(p) for p in props]
        log(f"  P0: {len(reqs)} proposer calls ({n} new), prompt chars {[p['prompt_chars'] for p in props]}")
        rng = random.Random(f"B:{self.g}:{self.a.seed}")
        base = self.settings_of([e for e in tip if e["cond"] == "always"])
        fresh = self.b_cells()
        rng.shuffle(fresh)
        nb1 = min(sl["B"], len(fresh)) if self.b_cells_cap() is None else min(sl["B"], len(fresh), self.b_cells_cap())
        for k in range(nb1):                          # each distinct one-factor cell at most once per tip (coordinator 11-10)
            knob, val = fresh.pop()
            props.append(dict(author="B", slot=k, cand=dict(why="blind one-factor mutation", condition="always", procedure=None, settings={knob: val})))
        self.add_w_props(props, rng, self.w_budget_after_g(props, sl))
        st = self.stack_entry()
        if st is not None:
            props.append(dict(author="system", slot=0, stack=st))
            log(f"  stack candidate (author system): {len(st['parts'])} accepted, unmerged entries on {self.state['tip']['id']}: "
                + ", ".join(x["cid"] for x in st["parts"]))
        G["props"] = props
        done, miss = self.io.wait([q["key"] for q in reqs], self.deadline, label="proposers")
        if miss:
            log(f"  proposers: {len(miss)} calls unanswered at the deadline")

    def stack_entry(self):
        """Stack candidate (Stage 2, a13c bundle test): the accepted (AM15), unmerged entries of earlier generations that are not in the tip,
        in contribution_lb order, as ONE candidate of author "system" read against the current tip. Left out: entries already in the tip,
        an always setting-only entry whose knob the tip already sets, a second always setting-only entry on the same knob (the one with
        the larger contribution stays), and the same stack on the same tip judged before."""
        if self.a.no_stack:
            return None
        tip = self.state["tip"]
        tkeys = {PR._entry_key(e) for e in self.tip_view()["entries"]}
        knobs = {k for e in tip["entries"] if e["cond"] == "always" for k in (e.get("settings") or {})}
        pool = []
        for gk, G in self.state["gens"].items():
            if int(gk) >= self.g:
                continue
            for c in G.get("cands", []):
                e = c.get("entry")
                if not e or not c.get("accepted") or c.get("merged") or c["author"] == "system" or e.get("parts") or PR._entry_key(e) in tkeys:
                    continue
                pool.append(c)
        pool.sort(key=lambda c: (-self.cl_of(c), c["cid"]))                     # AM15: contribution_lb order
        keep, edits = [], {json.dumps(e["edit"], sort_keys=True) for e in tip["entries"] if e.get("edit")}
        for c in pool:
            e = c["entry"]
            if e.get("edit"):                                    # weight edits stack as a list edit (applied in order, engine_api §7)
                k = json.dumps(e["edit"], sort_keys=True)
                if k not in edits:
                    edits.add(k)
                    keep.append(c)
                continue
            if e["cond"] == "always" and not e.get("procedure") and not e.get("ops"):
                ks = set(e.get("settings") or {})
                if ks & knobs:
                    continue
                knobs |= ks
            keep.append(c)
        if not keep:
            return None
        ids = [c["cid"] for c in keep]
        if any(c["author"] == "system" and c.get("parent") == tip["id"] and c.get("stack_of") == ids
               for G in self.state["gens"].values() for c in G.get("cands", [])):
            return None
        parts = [{k: v for k, v in c["entry"].items()} for c in keep]
        return dict(condition=f"stack of {len(parts)} accepted entries", cond="stack", procedure=None, settings={}, ops=[], parts=parts, author="system")

    def check(self, x, tv, card=None):
        """(cand, err, repairable) of one proposer answer. Repairable = fails only on a format limit (malformed or missing json, a
        field of the wrong type, condition or procedure too long, an unknown field) or on naming a bench / category; fixed-value,
        option-letter, effort and menu rejections are not. A bench or category word that occurs in at least half of the writer's card
        question texts is content, not a name (PR.card_words)."""
        if x is None:
            return None, "unanswered", False
        cand, err = PR.parse(x.get("content") or "")
        if cand is None:
            return None, err, True
        aw = PR.card_words(self.rows, card) if card else None
        probs = PR.validate(cand, tv, self.knobs, self.rows, allowed_words=aw)
        if probs:
            return cand, "invalid: " + "; ".join(probs), all(p.startswith(FORMAT_PROBLEMS) or NAMING_PROBLEM.match(p) for p in probs)
        return cand, None, False

    def repair(self, props, tv):
        """One repair round (coordinator 11-10): a format-only failure gets one follow-up turn with the validator message."""
        reqs = []
        for p in props:
            if p["author"] in ("B", "G", "system") or not p.get("key"):
                continue
            fk = final_key(p)                                       # S2-A: the last answered lab turn, else the turn-1 answer
            q0 = self.lab_request(fk)
            if q0 is None:
                continue
            x = self.io.get(fk)
            cand, err, fix = self.check(x, tv, p.get("card"))
            if err is None or not fix:
                continue
            msg = (f"Your answer could not be used: {err}.\nSend your entry again as exactly one json block in the answer format, "
                   f"within the limits listed above.")
            msgs = q0["messages"] + [{"role": "assistant", "content": (x or {}).get("content") or ""}, {"role": "user", "content": msg}]
            p["repair_key"] = fk + "|fix1"
            p["repair_reason"] = err[:300]
            reqs.append(dict(key=p["repair_key"], mode="chat", engine="E", priority=0, messages=msgs, temperature=PROP_TEMP, max_tokens=PROP_MAX))
        if reqs:
            n = self.io.submit(f"g{self.g:03d}_chatfix", reqs)
            log(f"  repair round: {len(reqs)} format- or naming-only failures get one follow-up turn ({n} new)")
            done, miss = self.io.wait([q["key"] for q in reqs], self.deadline, label="repairs")
        return len(reqs)

    def step_validate(self):
        G = self.G()
        tv, seen, cands = self.tip_view(), set(PR._entry_key(e) for e in self.tip_view()["entries"]), []
        seen_by = {}                                                    # entry key -> the first candidate with it (B2 duplicate_of)
        G["repairs"] = self.repair(G["props"], tv)
        labrecs = self.lab_load()                               # AM18: lab_tested compares the final entry with the entries the lab ran
        for j, p in enumerate(G["props"]):
            c = dict(cid=f"g{self.g:03d}c{j:02d}", gen=self.g, author=p["author"], slot=p["slot"], parent=self.state["tip"]["id"], target="E.H",
                     card_source=p.get("card_source"), card_wrong=p.get("card_wrong"), card_right=p.get("card_right"))
            if p["author"] == "system":                         # stack candidate (built in step_propose): no proposer text to check
                c.update(entry=dict(p["stack"], cid=c["cid"]), kind="stack", why="accepted, unmerged entries stacked on the tip",
                         stack_of=[x["cid"] for x in p["stack"]["parts"]], status="valid", valid0=True)
                cands.append(c)
                continue
            if p["author"] in ("B", "G"):
                cand = p["cand"]
                probs = (self.edit_problems(cand["edit"]) if cand.get("edit") else
                         self.b_setting_problems(cand) if (self.a.am16 or p["author"] == "G") else PR.validate(cand, tv, self.knobs, self.rows))
                if p["author"] == "G":
                    c["g"] = cand.get("g")                                      # B2: the signal's prediction travels with the candidate
                err = ("invalid: " + "; ".join(probs)) if probs else None
                if cand.get("edit"):
                    c.update(target="E.W", cell=cand.get("cell"), cell_group=cand.get("cell_group"))
            else:
                x = self.io.get(final_key(p))
                c["lab_turns"], c["lab"] = p.get("lab_turns", 0), p.get("lab")
                if p.get("lab_turns"):
                    c["key_final"] = final_key(p)
                c["prop_tokens"] = (x or {}).get("tokens")
                c["prop_finish"] = (x or {}).get("finish")
                cand, err, _ = self.check(x, tv, p.get("card"))
                if p.get("repair_key"):
                    x2 = self.io.get(p["repair_key"])
                    cand2, err2, _ = self.check(x2, tv, p.get("card"))
                    c.update(repaired=err2 is None, repair_reason=p.get("repair_reason"), first_error=err)
                    c["prop_tokens_fix"] = (x2 or {}).get("tokens")
                    cand, err = (cand2, None) if err2 is None else (cand2 if cand2 is not None else cand, f"after repair: {err2}")
                c["lab_tested"] = self.lab_tested(p, cand, labrecs)   # AM18: True / False ("final entry not lab-tested") / None (no lab)
            if cand is not None:
                ek = PR._entry_key(cand) + ((json.dumps(cand["edit"], sort_keys=True),) if cand.get("edit") else ())
                if err is None and ek in seen:
                    err = "invalid: duplicate of the tip or of an earlier candidate this generation"
                    if p["author"] == "G":                              # B2 (coordinator 11-11): realised from the candidate it duplicates
                        c["duplicate_of"] = seen_by.get(ek, "tip")
                seen.add(ek)
                seen_by.setdefault(ek, c["cid"])
            c["status"] = "valid" if err is None else err
            c["valid0"] = err is None
            if cand is not None:
                c.update(why=cand.get("why", ""), entry=dict(condition=cand["condition"], cond="always" if cand["condition"] == "always" else sha16(cand["condition"]),
                                                          procedure=cand.get("procedure"), settings=cand.get("settings") or {}, ops=PR.canon_ops(cand.get("ops")),
                                                          cid=c["cid"], author=p["author"], **({"edit": cand["edit"]} if cand.get("edit") else {})))
                c["kind"] = "weight" if cand.get("edit") else ("text" if cand.get("procedure") else ("ops" if cand.get("ops") else "setting"))
            cands.append(c)
        G["cands"] = cands
        log(f"  candidates: {sum(c['status'] == 'valid' for c in cands)} valid of {len(cands)}; repaired {sum(bool(c.get('repaired')) for c in cands)} of "
            f"{G['repairs']} sent back; " + "; ".join(f"{c['cid']} {c['status'][:80]}" for c in cands if c["status"] != "valid"))
        lt = collections.Counter(c["lab_tested"] for c in cands if c.get("lab_tested") is not None)
        if lt:
            log(f"  lab_tested (AM18): final entry lab-tested {lt[True]}, final entry not lab-tested {lt[False]}")

    def valid(self):
        return [c for c in self.G()["cands"] if c["status"] == "valid" or c.get("rung")]

    def step_route(self):
        V = self.valid()
        SV = [c for c in V if c["author"] not in ("B", "G") or self.a.show_b]                 # AM8: proposer candidates also routed on SHOW rows (one batch)
        self.ensure_routes([c["entry"] for c in V], self.judge_rows, self.deadline, "cands")
        if SV and not self.a.no_show:
            self.ensure_routes([c["entry"] for c in SV], self.role["show"], self.deadline, "show")
            self.show_reads(SV)
        for c in V:
            e = c["entry"]
            R = [r for r in self.judge_rows if self.fires(e, r)]
            c["routed"] = len(R)
            c["share"] = len(R) / self.NJ
            c["route"] = self.route_stats(e, self.judge_rows)
            if not R:
                c["status"] = "no routed JUDGE rows (D = 0 exactly)"
        for c in V:
            rs = c["route"]
            if c["entry"]["cond"] == "always":
                log(f"  {c['cid']} {c['author']}: always, {c['routed']}/{self.NJ} JUDGE rows")
            else:
                log(f"  {c['cid']} {c['author']}: routed {c['routed']}/{self.NJ} = {rs['share']:.2f}, unsure band {UNSURE[0]}<p<{UNSURE[1]}: "
                    f"{'n/a' if rs['unsure'] is None else format(rs['unsure'], '.2f')} of routed, unanswered {rs['unanswered']}, p from choice {rs['from_choice']} "
                    f"| {c['entry']['condition'][:90]}")

    def show_reads(self, cands):
        """AM8: every proposer candidate is read once (sample 0, priority 5, never waited for) on its routed SHOW rows. These reads never
        enter D; they feed trace memory (the executor's answer with and without the entry where the two differ) and the lab's tails."""
        tip, reqs = self.state["tip"]["entries"], []
        for c in cands:
            R = [r for r in self.role["show"] if self.fires(c["entry"], r)]
            c["show_routed"] = len(R)
            reqs += [self.read_req(self.cand_entries(c), r, 0, 5)[0] for r in R]
        n = self.io.submit(f"g{self.g:03d}_show", reqs)
        log(f"  SHOW reads (AM8): {len(reqs)} for {len(cands)} candidates ({n} new), background")

    def order(self):
        """Per-generation stratified order of JUDGE rows: benches interleaved, inside a bench one row per picture cluster in turn."""
        rng = random.Random(f"order:{self.g}:{self.a.seed}")
        by = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in self.judge_rows:
            by[self.rows[r]["bench"]][self.rows[r].get("pic") or r].append(r)
        per = {}
        for b, cl in by.items():
            keys = sorted(cl)
            rng.shuffle(keys)
            for k in keys:
                rng.shuffle(cl[k])
            seq, i = [], 0
            while any(cl[k] for k in keys):
                for k in keys:
                    if cl[k]:
                        seq.append(cl[k].pop())
            per[b] = seq
        out, bs = [], sorted(per)
        rng.shuffle(bs)
        while any(per[b] for b in bs):
            for b in bs:
                if per[b]:
                    out.append(per[b].pop(0))
        return out

    def rung_rows(self, c, rung):
        if c.get("rows_in_hand") is not None:            # AM17: a futile weight cell or a candidate whose reads were cancelled is judged on
            return list(c["rows_in_hand"]) if rung == 1 else []    # the rows it had in hand at that moment, never on imputed ones
        pos = {r: i for i, r in enumerate(self.G()["order"])}
        R = sorted((r for r in self.judge_rows if self.fires(c["entry"], r)), key=pos.get)
        if not self.a.two_rungs and not self.a.rung1_rows:
            return R if rung == 1 else []           # one rung (coordinator 11-10): all routed JUDGE rows at once, one decision, one tail
        k = self.a.rung1_rows or int(math.ceil(RUNG1_SHARE * self.NJ))
        return R[:k] if rung == 1 else R[k:]

    def cand_entries(self, c):
        return self.state["tip"]["entries"] + [c["entry"]]

    def read_rung(self, rung, cands):
        reqs, keys = [], {}
        for c in cands:
            rows = self.rung_rows(c, rung)
            ks = []
            for r in rows:
                q, _ = self.read_req(self.cand_entries(c), r, 0, 1)
                reqs.append(q)
                ks.append(q["key"])
            keys[c["cid"]] = ks
            c.setdefault("rows_read", {})[str(rung)] = len(rows)
        n = self.io.submit(f"g{self.g:03d}_rung{rung}", reqs)
        log(f"  rung {rung}: {len(cands)} candidates, {len(reqs)} reads ({n} new)")
        allk = [k for ks in keys.values() for k in ks]
        done, miss = self.io.wait(allk, self.deadline + GRACE, label=f"rung {rung}")
        if miss:
            log(f"  rung {rung}: {len(miss)} reads unfinished after the grace window: those rows stay unread")
        return len(miss)

    def box_extend(self):
        """Coordinator 11-10: the running generation's box can be extended without a restart: RUN/box_extend.json = {"g": N, "add_s": S}
        (S >= 0) moves generation N's deadline to t0 + T + S; re-read on every wait cycle of the read step; no file = no change."""
        p = os.path.join(self.rd, "box_extend.json")
        if not os.path.exists(p):
            return
        try:
            x = json.load(open(p))
            if int(x.get("g", -1)) != self.g:
                return
            add = max(0.0, float(x.get("add_s") or 0.0))
        except (ValueError, TypeError) as ex:
            log(f"  box_extend.json: {type(ex).__name__}: {ex}")
            return
        G = self.G()
        new = float(G["t0"]) + self.box_T() + add
        if abs(new - self.deadline) > 0.5:
            log(f"  box extended by {add:.0f} s (box_extend.json): the read step now closes at {time.strftime('%H:%M:%S', time.localtime(new + GRACE))}")
            self.deadline = new
            G["box_extended_s"] = add

    # ------------------------------------------------------------------ AM21 LITE (coordinator 11-10 22:45): discovery -> replication
    def lite_on(self):
        """AM21 LITE holds for a generation created with it (G["am21"]); a generation created earlier keeps its rule on resume."""
        return (self.G().get("am21") == "lite") and not (self.a.two_rungs or self.a.rung1_rows)

    @staticmethod
    def nsamp(c, mode):
        """Samples per routed row the candidate is judged on: obs = the samples requested so far (1 in step 1, LITE_M after step 2 is
        queued); pess / opt = LITE_M (every unread one counted wrong / right) unless the candidate has left (then 1). 1 without AM21."""
        a = c.get("am21") or c.get("carry")                      # carry: an earlier generation's candidate in the LITE carry-in (4 samples)
        if not a:
            return 1
        return int(a["M"]) if mode in ("pess", "opt") else int(a["n"])

    @staticmethod
    def s0(c):
        """First sample index of the candidate's reads: 0, or LITE_M for an AM23 bundle (its samples are new on every row: where one member
        holds alone the bundle's view is the member's, whose samples 0..LITE_M-1 are its selection data)."""
        return int((c.get("am21") or {}).get("s0") or 0)

    def cand_keys(self, c):
        """{key: (row, sample)} of every read requested for the candidate on its rung-1 rows."""
        s0 = self.s0(c)
        return {self.read_req(self.cand_entries(c), r, s, 1)[0]["key"]: (r, s) for r in self.rung_rows(c, 1) for s in range(s0, s0 + self.nsamp(c, "obs"))}

    def lite_step(self, V):
        """AM21 LITE step 2: a candidate whose step 1 is complete (every sample-0 read in, or a final error / cancelled line; the parent
        reference on every row; a weight cell past its AM17 quarter): pooled d <= 0 -> it has left (not accepted, p = 1 in its family's BH,
        judged on its step-1 rows); d > 0 -> samples 1..LITE_M-1 on the same routed rows in one batch per candidate, the candidates found in
        the same pass submitted in order of step-1 p. The step-1 numbers are kept (c["am21"]["step1"]) for the ledger."""
        todo = []
        for c in V:
            a = c.get("am21")
            if not a or a["n"] != 1 or a.get("left") or c.get("decided") or c.get("reads_cancelled") or c.get("futile"):
                continue
            if c.get("w_stage") == "quarter":
                continue
            ks = [self.read_req(self.cand_entries(c), r, 0, 1)[0]["key"] for r in self.rung_rows(c, 1)]
            if any(self.io.get(k) is None and k not in self.io.err for k in ks):
                continue
            o = self.am15_numbers("obs", [c])[c["cid"]]
            if o["pend"]:
                continue                                         # a parent reference is missing on a row: wait for it
            a["step1"] = dict(d=o["d"], se=o["se"], se_used=o["se_used"], p=o["p"], n_rows=o["n_rows"], clusters=o["clusters"], tok=o["tc"],
                              tok_parent=o["tp"], c=(o.get("ec") or {}).get("c"), lobo=o["lobo"], t=time.time())
            if o["n_read"] < 1 or o["d"] <= 0:
                a.update(left=True, M=1)
                c.update(futile=True, decided=True, decided_at=time.time(), accepted=False,
                         status=f"not accepted: left after step 1 (AM21 LITE), d {o['d']:+.4f} <= 0 on {o['n_rows']} rows at 1 sample; in BH with p = 1")
                c["decision"] = dict(basis="left after step 1 (AM21 LITE)", rule="am21-lite-left", d=o["d"], se=o["se"], se_used=o["se_used"], nu=o["nu"],
                                     n_rows=o["n_rows"], p=o["p"], p_in_bh=1.0, K=o["K"], group=o["group"], bh=False, lobo=o["lobo"],
                                     is_global=o["is_global"], lb=o["lb"], contribution_lb=o["contribution_lb"], share=o["share"], pend=0)
                log(f"  {c['cid']} {c['author']}: step 1 d {o['d']:+.4f} (p {o['p']:.4f}, {o['n_rows']} rows) <= 0 -> left (AM21 LITE), p = 1 in BH")
            else:
                todo.append((o["p"], c, o))
        for p, c, o in sorted(todo, key=lambda t: t[0]):
            rows = self.rung_rows(c, 1)
            rq = [self.read_req(self.cand_entries(c), r, s, 1)[0] for r in rows for s in range(1, LITE_M)]
            c["am21"]["n"] = LITE_M
            c["am21"]["step2_at"] = time.time()
            n = self.io.submit(f"g{self.g:03d}_rung1s2_{c['cid']}", rq)
            log(f"  {c['cid']} {c['author']}: step 1 d {o['d']:+.4f} (p {p:.4f}, {o['n_rows']} rows) > 0 -> samples 1-{LITE_M - 1} on its {len(rows)} rows "
                f"({len(rq)} reads, {n} new; AM21 LITE step 2)")

    def lite_box_end(self, V):
        """AM21 item 4 at the end of the box: a candidate still undecided is accepted only if BH (and LOBO) accept it under the
        pessimistic completion (every unread sample of the LITE_M counted wrong, every other candidate's too); otherwise not accepted, and
        its pending reads are cancelled when it is BH-insignificant there (AM17 b: p = 1 is then exact for the others)."""
        und = [c for c in V if not c.get("decided")]
        if not und:
            return
        P = self.am15_numbers("pess", V)
        for c in und:
            o = P[c["cid"]]
            got = sum(1 for k in self.cand_keys(c) if self.io.get(k) is not None)
            rows = max(len(self.rung_rows(c, 1)), 1)
            if c.get("am21") is not None:
                c["am21"]["box_at"] = round(got / rows, 2)
            self.decide_rule(c, dict(o), f"the box end at {got / rows:.2f} samples per row (AM21: unread samples of the {LITE_M} counted wrong)")
            if not o["accept"] and self.a.am17 and o["pend"] > 0 and not o["bh"]:
                self.cancel_reads(c, V)

    # ------------------------------------------------------------------ AM23 (coordinator 11-10 23:30): portfolio acceptance
    def bundle_eligible(self, c):
        """A routed (not global) proposer candidate of this generation without ops or weight edit (AM23: global ones, cells and candidates with
        ops are not bundled)."""
        e = c.get("entry") or {}
        return (c["author"] not in ("B", "G", "system") and int(c.get("gen", self.g)) == self.g and c.get("routed") and not self.is_global(c)
                and not e.get("ops") and not e.get("edit") and not e.get("parts") and not self.is_w(c))

    def bundle_carry(self):
        """AM23: members of an accepted, unmerged bundle of an earlier generation whose entries are not in the tip stay for this generation's
        bundle (once)."""
        tk = {PR._entry_key(e) for e in self.flat(self.state["tip"]["entries"])}
        out = []
        for gk, Gb in self.state["gens"].items():
            if int(gk) >= self.g:
                continue
            for b in Gb.get("cands", []):
                if b.get("kind") == "bundle" and b.get("accepted") and not b.get("merged") and not b.get("bundle_carried"):
                    out.append((b, [x for x in b["entry"]["parts"] if PR._entry_key(x) not in tk]))
        return out

    def bundle_rule(self):
        """AM23 member rule in force: RUN/bundle_rule.json {"members": "all" | "narrow"} if present (coordinator 11-10 00:20: switchable without a
        restart, read when the bundle is built), else --bundle-members."""
        p = os.path.join(self.rd, "bundle_rule.json")
        if os.path.exists(p):
            try:
                m = json.load(open(p)).get("members")
                if m in ("all", "narrow"):
                    return m, "bundle_rule.json"
                log(f"  bundle_rule.json: unknown members rule {m!r}, using --bundle-members {self.a.bundle_members}")
            except (ValueError, AttributeError) as ex:
                log(f"  bundle_rule.json: {type(ex).__name__}: {ex}")
        return self.a.bundle_members, "--bundle-members"

    def bundle_members(self, el):
        """AM23 members among the eligible candidates `el` (step 1 complete or not): step-1 d > 0 and not left; rule narrow: of those, the
        ones whose routed JUDGE rows number at most the median routed-row count of ALL eligible candidates (the narrow half, a relative rule)."""
        rule, src = self.bundle_rule()
        mem = [c for c in el if ((c.get("am21") or {}).get("step1") or {}).get("d", 0) > 0 and not (c.get("am21") or {}).get("left")]
        med = None
        if rule == "narrow" and el:
            rs = sorted(int(c.get("routed") or 0) for c in el)
            med = rs[len(rs) // 2] if len(rs) % 2 else (rs[len(rs) // 2 - 1] + rs[len(rs) // 2]) / 2
            mem = [c for c in mem if int(c.get("routed") or 0) <= med]
        return mem, dict(rule=rule, rule_from=src, median_routed=med)

    def lite_bundle(self, V):
        """AM23: once every eligible candidate's LITE step 1 is complete (or has left / been cancelled), or at BUNDLE_BUILD_AT of the box with
        the ones complete then: ONE bundle candidate (author system, kind bundle) = the eligible candidates with step-1 d > 0 (entry order =
        candidate id) plus the members an accepted, unmerged bundle of an earlier generation left over, applied together as the tip applies
        several entries; read on the union of their routed JUDGE rows with LITE_M NEW samples (indices LITE_M..2 LITE_M-1), judged by BH
        inside (system, generation) with the stack candidate, interval rule as for any candidate, no LOBO."""
        G = self.G()
        B = G.get("bundle") or {}
        if B.get("state") != "planned":
            return
        el = [c for c in V if c["cid"] in set(B["eligible"])]
        done = lambda c: bool((c.get("am21") or {}).get("step1")) or c.get("decided") or c.get("reads_cancelled") or c.get("futile")
        t_r = float(G.get("rung1_t0") or G["t0"])               # coordinator 11-11 01:50: shares of the READ window, not of the box
        win = float(self.deadline) - t_r
        frac = (time.time() - t_r) / win if win > 0 else 1.0
        n_done = sum(1 for c in el if done(c))
        if n_done < len(el) and frac < BUNDLE_BUILD_AT:
            return
        mem, rinfo = self.bundle_members(el)
        carried = self.bundle_carry()
        parts = [dict(c["entry"]) for c in sorted(mem, key=lambda c: c["cid"])]
        keys = {PR._entry_key(x) for x in parts}
        for b, xs in carried:
            for x in xs:
                if PR._entry_key(x) not in keys:
                    parts.append(dict(x))
                    keys.add(PR._entry_key(x))
        info = dict(members=[x.get("cid") for x in parts], from_step1=[c["cid"] for c in sorted(mem, key=lambda c: c["cid"])],
                    carried={b["cid"]: [x.get("cid") for x in xs] for b, xs in carried if xs}, incomplete=[c["cid"] for c in el if not done(c)],
                    t=time.time(), eligible_n=len(el), step1_complete=n_done, read_frac=round(frac, 3), **rinfo)
        if len(parts) < BUNDLE_MIN and n_done < len(el) and frac < BUNDLE_WAIT_TO:
            if not B.get("waiting"):
                B["waiting"] = dict(t=time.time(), read_frac=round(frac, 3), members=len(parts), step1_complete=n_done, eligible_n=len(el))
                log(f"  AM23: at {frac:.0%} of the read window {len(parts)} member(s) (< {BUNDLE_MIN}) with {n_done} of {len(el)} eligible step 1s "
                    f"complete: waiting for the rest until {BUNDLE_WAIT_TO:.0%}")
            return
        log(f"  AM23: build decision at {frac:.0%} of the read window: {len(el)} eligible, {n_done} with step 1 complete, "
            f"{len(mem)} with step-1 d > 0" + (f", {len(parts) - len(mem)} carried" if len(parts) > len(mem) else ""))
        if len(parts) < BUNDLE_MIN:
            B.update(state="none", why=f"{len(parts)} members with step-1 d > 0 (< {BUNDLE_MIN})", **info)
            log(f"  AM23: no bundle ({B['why']}); the system family is the stack candidate alone")
            return
        for b, xs in carried:
            b["bundle_carried"] = self.g
        self.load_routes(parts, self.judge_rows)
        e = dict(condition=f"bundle of {len(parts)} routed entries", cond="bundle", procedure=None, settings={}, ops=[], parts=parts, author="system")
        cid = f"g{self.g:03d}c{len(G['cands']):02d}"
        e["cid"] = cid
        R = [r for r in self.judge_rows if self.fires(e, r)]
        hold = collections.Counter(sum(1 for x in parts if self.fires(x, r)) for r in R)
        c = dict(cid=cid, gen=self.g, author="system", slot=None, parent=self.state["tip"]["id"], target="E.H", entry=e, kind="bundle",
                 why="AM23: this generation's routed step-1 survivors applied together", members=info["members"], status="valid", valid0=True,
                 routed=len(R), share=len(R) / self.NJ, rung=1, am21=dict(M=LITE_M, n=LITE_M, s0=LITE_M, bundle=True),
                 bundle=dict(info, union=len(R), holders=dict(sorted(hold.items())), rows_2plus=sum(v for k, v in hold.items() if k >= 2)))
        c["route"] = self.route_stats(e, self.judge_rows)
        c.setdefault("rows_read", {})["1"] = len(R)
        G["cands"].append(c)
        V.append(c)
        rq = [self.read_req(self.cand_entries(c), r, s_, 1)[0] for r in self.rung_rows(c, 1) for s_ in range(LITE_M, 2 * LITE_M)]
        n = self.io.submit(f"g{self.g:03d}_rung1_{cid}", rq)
        B.update(state="built", cid=cid, **info)
        log(f"  AM23 bundle {cid}: {len(parts)} members {info['members']} (rule {rinfo['rule']} from {rinfo['rule_from']}"
            + (f", median routed {rinfo['median_routed']}" if rinfo["median_routed"] is not None else "") + ")"
            + (f" (carried from {info['carried']})" if info["carried"] else "")
            + (f"; built at {BUNDLE_BUILD_AT:.0%} of the box with {len(info['incomplete'])} step 1s incomplete" if info["incomplete"] else "")
            + f"; union {len(R)} JUDGE rows, members holding per row {dict(sorted(hold.items()))}; samples {LITE_M}-{2 * LITE_M - 1}: {len(rq)} reads ({n} new)")
        self.save()

    def step_rung1(self):
        G = self.G()
        G.setdefault("order", self.order())
        V = [c for c in self.valid() if c.get("routed")]
        lite = self.lite_on()
        for c in V:
            c["rung"] = 1
            if lite:
                c.setdefault("am21", dict(M=LITE_M, n=1))
        if self.a.two_rungs or self.a.rung1_rows:
            G["rung1_missing"] = self.read_rung(1, V)
            return
        reqs, n = [], 0
        G.setdefault("rung1_t0", time.time())                   # AM23: the read window starts here (coordinator 11-11 01:50)
        qs = set(G.setdefault("w_quarter", self.w_quarter())) if self.a.am17 else set()
        if lite and "bundle" not in G:
            el = [c for c in V if self.bundle_eligible(c)]
            G["bundle"] = (dict(state="planned", eligible=[c["cid"] for c in el], t=time.time()) if self.a.bundle and len(el) >= BUNDLE_MIN
                           else dict(state="none", why="--no-bundle" if not self.a.bundle else f"{len(el)} eligible candidates (< {BUNDLE_MIN})"))
        for c in V:
            rows = self.rung_rows(c, 1)
            if c.get("kind") == "bundle":                                 # (resume) an AM23 bundle built before the restart: its own samples
                rq = [self.read_req(self.cand_entries(c), r, s_, 1)[0] for r in rows for s_ in range(self.s0(c), self.s0(c) + LITE_M)]
                reqs += rq
                n += self.io.submit(f"g{self.g:03d}_rung1_{c['cid']}", rq)
                continue
            rq = [self.read_req(self.cand_entries(c), r, 0, 1)[0] for r in rows]
            reqs += rq
            c.setdefault("rows_read", {})["1"] = len(rows)
            if not self.a.am17:
                continue
            if self.is_w(c) and not c.get("w_stage"):                    # AM17 a: futility rung for weight cells, the stratified quarter first
                c["w_stage"] = "quarter"
            rest = [q for q in rq if q["row"] not in qs] if c.get("w_stage") in ("quarter", "full") else []
            if rest:
                rq = [q for q in rq if q["row"] in qs]
            n += self.io.submit(f"g{self.g:03d}_rung1_{c['cid']}", rq)   # AM17 b: one batch per candidate, so a rejected one can be cancelled
            if c.get("w_stage") == "full" and rest:                       # (resume) the cell passed its quarter: the rest of JUDGE
                n += self.io.submit(f"g{self.g:03d}_rung1b_{c['cid']}", rest)
        if not self.a.am17:
            n = self.io.submit(f"g{self.g:03d}_rung1", reqs)
        log(f"  reads: {len(V)} candidates, {len(reqs)} reads ({n} new); AM15 interval decisions (BH per author) every 20 s"
            + (f"; weight cells read a stratified quarter of JUDGE ({len(qs)} rows) first (AM17 a); rejected candidates' reads are cancelled (AM17 b)"
               if self.a.am17 else ""))
        if lite:
            log(f"  AM21 LITE: a candidate with step-1 d > 0 gets samples 1-{LITE_M - 1} on its routed rows; d <= 0 -> left (p = 1); decisions on "
                f"the pooled {LITE_M} samples (interval rule: unread samples of the {LITE_M} counted wrong / right)")
        t0 = last = time.time()
        while True:
            self.io.poll()
            if self.a.am17:
                self.w_futility(V)
            if lite:
                self.lite_step(V)
                self.lite_bundle(V)
            wait = self.interval(V)
            if not wait:
                break
            self.box_extend()
            if time.time() >= self.deadline + GRACE:
                if (G.get("bundle") or {}).get("state") == "planned":
                    G["bundle"].update(state="dropped", why="the box closed before it was built")
                    log("  AM23: the box closed before the bundle was built -> no bundle; the system family is the stack candidate alone")
                if lite:
                    log(f"  box + grace closed with {len(wait)} candidates waited for: undecided ones are decided under the pessimistic completion (AM21)")
                    self.lite_box_end(V)
                else:
                    log(f"  box + grace closed with {len(wait)} candidates undecided: they are decided on the reads they have")
                break
            if time.time() - last >= 300:
                last = time.time()
                done = sum(1 for q in reqs if self.io.have(q["key"]))
                log(f"  waiting: {done}/{len(reqs)} reads after {time.time() - t0:.0f}s; decided {sum(bool(c.get('decided')) for c in V)}/{len(V)}; "
                    f"waiting on {', '.join(c['cid'] for c in wait[:8])}")
            time.sleep(20)
        if self.a.am17:                                                   # a weight cell still in its quarter (decided in that pass, or the box
            self.w_futility(V)                                            # closed) is read to the end like any accepted or undecided candidate
            for c in V:
                if c.get("w_stage") == "quarter" and not c.get("reads_cancelled"):
                    self.w_rest(c, "the rung closed while it was in its quarter")
        G["rung1_pending"] = sum(1 for c in V for k in self.cand_keys(c) if not self.io.have(k) and not c.get("reads_cancelled"))
        log(f"  all decided after {time.time() - t0:.0f}s with {G['rung1_pending']} reads still running (they finish in the background; the ledger is finalised later)")

    # ------------------------------------------------------------------ AM17 (coordinator 11-10): futility rung for W, cancel rejected reads
    def w_quarter(self):
        """The stratified quarter of JUDGE that every weight cell of this generation reads first: per source, the first ceil(n / 4) of its
        JUDGE rows in the generation's order (one row per picture cluster in turn)."""
        by = collections.defaultdict(list)
        for r in self.G()["order"]:
            by[self.rows[r]["bench"]].append(r)
        return [r for b in sorted(by) for r in by[b][:int(math.ceil(len(by[b]) / 4))]]

    @staticmethod
    def is_w(c):
        e = c.get("entry") or {}
        return bool(e.get("edit")) and not e.get("parts")

    def w_rest(self, c, why):
        qs = set(self.G()["w_quarter"])
        rq = [self.read_req(self.cand_entries(c), r, 0, 1)[0] for r in self.rung_rows(c, 1) if r not in qs]
        c["w_stage"] = "full"
        n = self.io.submit(f"g{self.g:03d}_rung1b_{c['cid']}", rq)
        log(f"  {c['cid']} {c.get('cell') or ''}: {why} -> the other {len(rq)} JUDGE rows ({n} new)")

    def w_futility(self, V):
        """AM17 a: a weight cell whose quarter has every read in (and the parent's reference on those rows): AM15 d on the quarter <= 0 ->
        it stops there, not accepted, judged on the quarter (rows_in_hand) and entered in its author's BH family with p = 1; d > 0 -> the
        rest of JUDGE is read. A cell the interval rule decides during its quarter is read to the end (accepted, or rejected without a
        cancel); one whose reads were cancelled stays on the rows in hand."""
        qs = set(self.G()["w_quarter"])
        for c in V:
            if c.get("w_stage") != "quarter":
                continue
            if c.get("reads_cancelled"):
                c["w_stage"] = "cancelled"
                continue
            if c.get("decided"):
                self.w_rest(c, f"decided by the interval rule during its quarter ({'accepted' if c.get('accepted') else 'not accepted, reads not cancelled'})")
                continue
            qr = [r for r in self.rung_rows(c, 1) if r in qs]
            ks = [self.read_req(self.cand_entries(c), r, 0, 1)[0]["key"] for r in qr]
            if any(self.io.get(k) is None and k not in self.io.err for k in ks):
                continue
            c["rows_in_hand"] = qr
            try:
                a = self.am15_numbers("obs", [c])[c["cid"]]
            finally:
                c.pop("rows_in_hand", None)
            if a["pend"]:
                continue                                                 # a parent reference is missing on a quarter row: wait for it
            if a["n_read"] >= 1 and a["d"] <= 0:
                c.update(w_stage="futile", futile=True, rows_in_hand=qr, decided=True, decided_at=time.time(), accepted=False,
                         status=f"not accepted: futility rung (AM17 a), d {a['d']:+.4f} <= 0 on the stratified quarter ({a['n_read']} rows); in BH with p = 1")
                c["decision"] = dict(basis="futility rung (AM17 a)", rule="am17-futility", d=a["d"], se=a["se"], se_used=a["se_used"], nu=a["nu"],
                                     n_rows=a["n_rows"], p=a["p"], p_in_bh=1.0, K=a["K"], group=a["group"], bh=False, lobo=a["lobo"],
                                     is_global=a["is_global"], lb=a["lb"], contribution_lb=a["contribution_lb"], share=a["share"], pend=0)
                log(f"  {c['cid']} {c.get('cell') or ''}: futility rung, d {a['d']:+.4f} on the quarter ({a['n_read']} rows) <= 0 -> stops, not accepted")
            else:
                self.w_rest(c, f"d {a['d']:+.4f} > 0 on the quarter ({a['n_read']} rows)")

    def cancel_reads(self, c, V):
        """AM17 b: a candidate rejected for good while BH-insignificant under its pessimistic completion: cancel its pending reads (its own
        batches via q/cancel/<batch>: E drops the queued ones, in-flight ones finish; W aborts the running cell) unless another live
        candidate needs one of those keys. From then on it is judged on the rows in hand (rows_in_hand, never imputed: a read landing later
        is not used) and enters its BH family with p = 1. AM21 LITE: every requested sample counts; the samples in hand per row are kept
        as a bit mask (c["hand"]), so the frozen numbers never change."""
        mine = self.cand_keys(c)
        pend = [k for k in mine if self.io.get(k) is None and k not in self.io.err]
        if not pend:
            return
        live = set()
        for o in V:
            if o is not c and not o.get("reads_cancelled") and not o.get("futile") and (not o.get("decided") or o.get("accepted")):
                live |= set(self.cand_keys(o))
        if live & set(pend):
            log(f"  {c['cid']}: rejected for good, {len(pend)} reads pending, not cancelled (another live candidate needs {len(live & set(pend))} of them)")
            return
        got = {(r, s_) for k, (r, s_) in mine.items() if self.io.get(k) is not None or k in self.io.err}
        hand = [r for r in self.rung_rows(c, 1) if (r, self.s0(c)) in got]
        for nm in (f"g{self.g:03d}_rung1_{c['cid']}", f"g{self.g:03d}_rung1b_{c['cid']}", f"g{self.g:03d}_rung1s2_{c['cid']}"):
            self.io.cancel_prefix(nm)
        self.io.pending.difference_update(pend)
        if c.get("am21") is not None:
            mask = collections.defaultdict(int)
            for r, s_ in got:
                mask[str(r)] |= 1 << s_
            c["hand"] = {r: mask[str(r)] for r in map(str, hand)}
        c.update(reads_cancelled=True, rows_in_hand=hand, cancelled_reads=len(pend), cancelled_at=time.time())
        if c.get("w_stage") == "quarter":
            c["w_stage"] = "cancelled"
        log(f"  {c['cid']}: rejected for good -> {len(pend)} pending reads cancelled (AM17 b); judged on the {len(hand)} rows in hand, p = 1 in BH")

    def ecomp_on(self, c):
        """Equal-compute reference (S2-B) for candidates of generations created with it on (G["ecomp"]); earlier generations keep the mean."""
        return bool(self.state["gens"].get(str(c.get("gen")), {}).get("ecomp"))

    def diffs(self, c, rungs, mode="obs"):
        """Per picture cluster: mean of (candidate right - parent reference) over the rows (per routed row, AM14); rows per cluster;
        tokens; reads still pending. Parent reference = mean right of the tip's cached samples, or, under the equal-compute rule (S2-B)
        when the entry's tokens per row exceed the tip's by c >= 1.5, the plurality answer of the first ceil(c) tip samples (capped at
        P_FULL; a row with fewer samples waits like a row without parent samples). mode "obs" uses read rows only; "pess" / "opt" are the
        AM11 bounds: a pending candidate read counts wrong / right and a row without a parent reference counts the parent right / wrong.
        A row whose ops were skipped by the engine (op_skipped, engine_api §7) is unrouted for this entry: zero difference.
        AM21 LITE: a candidate has several samples per row (nsamp); its row score is the mean over the samples it has (obs) or over all
        LITE_M with the unread ones counted wrong / right (pess / opt); a frozen candidate (reads cancelled) uses only the samples in hand
        (c["hand"]); sample 0 decides op_skipped; a later sample's final error line is neither read nor pending."""
        tip = self.state["tip"]["entries"]
        items, ctx = [], []
        S, hand, s0 = self.nsamp(c, mode), c.get("hand"), self.s0(c)
        for rung in rungs:
            for r in self.rung_rows(c, rung):
                xs, unread, skip = [], 0, False
                for s in range(s0, s0 + S):
                    if hand is not None and not (int(hand.get(str(r), 0)) >> s) & 1:
                        continue                                 # frozen: only the samples in hand count, a read landing later is not used
                    q, _ = self.read_req(self.cand_entries(c), r, s, 1)
                    x = self.io.get(q["key"])
                    fin_err = x is None and q["key"] in self.io.err and not self.io.err[q["key"]].get("cancelled")
                    if s == s0 and (fin_err or (x is not None and x.get("op_skipped"))):
                        skip = True                              # a final error line (e.g. the W worker's 16-image limit) or skipped ops: the
                        if fin_err and CTX_ERR in str(self.io.err[q["key"]].get("error") or ""):   # entry cannot run on this row, so it is
                            ctx.append(r)                        # unrouted there (zero difference); over the context: skipped_ctx
                        break
                    if fin_err or (x is not None and x.get("op_skipped")):
                        continue
                    if x is None:
                        unread += 1                              # pending, or a cancelled line ("not read")
                    else:
                        xs.append(x)
                items.append((r, xs, unread, skip, self.samples(tip, r)))
        both = [(xs, ps) for r, xs, u, sk, ps in items if xs and ps and not sk]
        tc = [sum(x.get("tokens") or 0 for x in xs) / len(xs) for xs, ps in both]
        tp = [sum(p.get("tokens") or 0 for p in ps) / len(ps) for xs, ps in both]
        tcm, tpm = (sum(tc) / len(tc) if tc else None), (sum(tp) / len(tp) if tp else None)
        m = None
        if self.ecomp_on(c) and tcm is not None and tpm:
            rnd = bool(self.state["gens"].get(str(c.get("gen")), {}).get("ecomp_round"))      # AM16 item 3: round(c) for generations created after it
            m = ECOMP.ref_m_round(tcm / tpm) if rnd else ECOMP.ref_m(tcm / tpm)
            m = min(m, P_FULL) if m is not None else None
        self._ecomp[c["cid"]] = dict(c=round(tcm / tpm, 3) if (tcm is not None and tpm) else None, m=m,
                                     skipped=sum(1 for it in items if it[3]), skipped_ctx_rows=ctx)
        d, w, n, pend = collections.defaultdict(float), collections.defaultdict(int), 0, 0
        for r, xs, unread, skip, ps in items:
            cl = self.rows[r].get("pic") or r
            if skip:
                w[cl] += 1
                d[cl] += 0.0
                continue
            ref = None
            if ps:
                if m is None:
                    ref = sum(bool(p.get("ok")) for p in ps) / len(ps)
                else:
                    v = ECOMP.plurality_ok(ps, self.rows[r], m)
                    ref = None if v is None else float(v)
            pend += unread + (ref is None)                       # unread samples (and a missing parent reference) are pending in every mode
            if not xs or ref is None or (unread and mode != "obs"):
                if mode == "obs" or (not xs and not unread):     # obs: a row without a sample in hand or without a reference waits;
                    continue                                     # every sample with a later final error: nothing to count
                fill = 0.0 if mode == "pess" else 1.0
                co = (sum(bool(x.get("ok")) for x in xs) + fill * unread) / (len(xs) + unread)
                pm = ref if ref is not None else (1.0 if mode == "pess" else 0.0)
            else:                                                # obs on the samples in hand, or every sample read
                co, pm = sum(bool(x.get("ok")) for x in xs) / len(xs), ref
                n += 1
            d[cl] += co - pm
            w[cl] += 1
        dd = {k: d[k] / w[k] for k in w}                 # AM14: per-ROUTED-row scale (no share factor); contribution = share x post_mean
        return dd, dict(w), n, tcm, tpm, pend

    def cl_bench(self, cl):
        if not hasattr(self, "_clb"):
            self._clb = {(r.get("pic") or i): r["bench"] for i, r in self.rows.items()}
        return self._clb.get(cl)

    def across_sources(self, dd, w):
        """AM10: per-source paired gain D_s with its cluster-robust se_s, and P(mu <= 0) for the across-source mean mu under the family model
        with the sources as the family (judge_eb.mu_posterior). Sources with fewer than 2 clusters are left out; se_s >= sqrt(.25 / n_s)."""
        by = collections.defaultdict(dict)
        for cl in dd:
            by[self.cl_bench(cl)][cl] = cl
        xs = [dict(name=b, d={cl: dd[cl] for cl in cls}, w={cl: w[cl] for cl in cls}) for b, cls in sorted(by.items()) if len(cls) >= 2]
        if len(xs) < 2:
            return None
        D, S, _ = EB.family(xs)
        n_s = np.array([float(sum(x["w"].values())) for x in xs])               # routed rows per source
        v = np.maximum(np.diag(S), 0.25 / n_s)                                  # coordinator 11-10: se_s floored at the worst-case Bernoulli
        Sd = np.diag(v)                                                         # error sqrt(.25 / n_s), so equal paired diffs are not certainty
        post = EB.mu_posterior(D, Sd)
        P = float(post[EB.MU_GRID <= 0].sum())
        return dict(P_mu_le0=P, mu_mean=float((post * EB.MU_GRID).sum()), n_sources=len(xs),
                    D_s={x["name"]: round(float(D[j]), 4) for j, x in enumerate(xs)}, se_s={x["name"]: round(float(math.sqrt(v[j])), 4) for j, x in enumerate(xs)},
                    se_s_raw={x["name"]: round(float(math.sqrt(S[j, j])), 4) for j, x in enumerate(xs)}, n_s={x["name"]: int(n_s[j]) for j, x in enumerate(xs)})

    def judge_numbers(self, mode, cands):
        """{cid: numbers} for the candidates under one mode (obs / pess / opt): judge_eb row (D, se, clusters, lfsr, post_mean) in its family
        (parent, author, kind), plus AM10 numbers for global candidates, n read, pending reads, tokens, cluster d and w."""
        fam, info = collections.defaultdict(list), {}
        for c in cands:
            rungs = [1] if c["rung"] == 1 else [1, 2]
            dd, w, n, tc, tp, pend = self.diffs(c, rungs, mode)
            info[c["cid"]] = dict(n=n, pend=pend, tc=tc, tp=tp, dd=dd, w=w, ec=self._ecomp.get(c["cid"]))
            if len(w) >= 2 and (mode != "obs" or n >= 2):
                fam[(c["parent"], c["author"], c["kind"])].append((c, dict(name=c["cid"], d=dd, w=w)))
        out = {}
        for key, xs in fam.items():
            rows_, hyper = EB.judge([x for _, x in xs], ALPHA)
            for (c, x), o in zip(xs, rows_):
                o = dict(o, family="|".join(key), K=len(xs), mu=hyper.get("mu"), tau=hyper.get("tau"), share=c["share"], contrib=c["share"] * o["post_mean"])
                o["am10"] = self.across_sources(x["d"], x["w"])     # decides global candidates; reported for routed ones
                out[c["cid"]] = dict(o, **info[c["cid"]])
        for cid, i in info.items():
            out.setdefault(cid, dict(i, too_few=True))
        return out

    # ------------------------------------------------------------------ AM15 decision rule (coordinator 11-10 18:45 CST; harness/judge_bh.py)
    def is_global(self, c):
        """A global candidate: condition always, or an entry that holds on every JUDGE row (as in the coordinator's replay)."""
        if c.get("kind") == "bundle":                            # AM23: members are routed entries; LOBO does not apply
            return False
        return c["entry"].get("cond") == "always" or (c.get("share") or 0.0) >= 1.0

    def am15_numbers(self, mode, cands, p1=()):
        """AM15 per candidate under one completion (obs / pess / opt of every candidate's own pending reads): d = mean paired difference on
        its routed judge rows (routed scale), se = cluster-robust (judge_eb.family), nu = clusters - 1, the routed share, global or not and
        LOBO on the per-source sums; then judge_bh.judge inside each (author, generation) over ALL of that author's read candidates in
        `cands` (BH at alpha). {cid: dict(x=judge_bh input, p, se_used, bh, lobo, accept, lb, contribution_lb, K, n_read, pend, tc, tp, dd,
        w, src)}. A candidate with no rows under the completion has d 0, se 1 (p 1/2): it counts in K and is never accepted."""
        grp, info = collections.defaultdict(list), {}
        for c in cands:
            rungs = [1] if c.get("rung", 1) == 1 else [1, 2]
            dd, w, n, tc, tp, pend = self.diffs(c, rungs, mode)
            nr = int(sum(w.values()))
            if nr >= 1:
                D, S, _ = EB.family([dict(name=c["cid"], d=dd, w=w)])
                d, se = float(D[0]), float(math.sqrt(max(float(S[0, 0]), 0.0)))
            else:
                d, se = 0.0, 1.0
            src = collections.defaultdict(lambda: [0.0, 0])
            for cl in w:
                v = src[self.cl_bench(cl)]
                v[0] += dd[cl] * w[cl]
                v[1] += w[cl]
            g_ = self.is_global(c)
            x = dict(name=c["cid"], d=d, se=se, n=max(nr, 1), nu=float(max(len(w) - 1, 1)), share=float(c.get("share") or 0.0), is_global=g_,
                     lobo=BH.lobo_ok({k: (v[0], v[1]) for k, v in src.items()}) if g_ else True)
            if c.get("reads_cancelled") or c.get("futile") or c["cid"] in p1:   # AM17: rejected / futile; carry-in: not carried -> p = 1 (exact for the
                x = dict(x, d=-1.0, se=0.0, hand=(d, se))         # others); its own d, se, p on the rows in hand are reported below
            info[c["cid"]] = dict(x=x, n_read=n, pend=pend, tc=tc, tp=tp, dd=dd, w=w, ec=self._ecomp.get(c["cid"]), clusters=len(w), n_rows=nr,
                                  src={k: [round(v[0], 3), v[1]] for k, v in src.items()})
            grp[(c["author"], int(c.get("gen", self.g)))].append(c["cid"])
        out = {}
        bycid = {c["cid"]: c for c in cands}
        for key, ids in grp.items():
            pad = self.g_prior_pad(key, ids, bycid)                 # B4: author G's family also holds its library's earlier cells (p = 1)
            for r in BH.judge([info[i]["x"] for i in ids] + pad, ALPHA):
                if r["name"] not in info:
                    continue
                i = info[r["name"]]
                o = dict(i, **{k: r[k] for k in ("p", "se_used", "bh", "lobo", "accept", "lb", "contribution_lb")},
                         d=i["x"]["d"], se=i["x"]["se"], nu=i["x"]["nu"], share=i["x"]["share"], is_global=i["x"]["is_global"],
                         K=len(ids) + len(pad), group=f"{key[0]}|g{key[1]}", share_d=i["x"]["share"] * i["x"]["d"], **({"K_prior": len(pad)} if pad else {}))
                if "hand" in i["x"]:                                # AM17: numbers on the rows in hand (report); BH used p = 1
                    d, se = i["x"]["hand"]
                    ph, su = BH.pval(d, se, i["x"]["n"], i["x"]["nu"])
                    lb = d - float(BH._t.ppf(1 - ALPHA, max(i["x"]["nu"], 1.0))) * su
                    o.update(d=d, se=se, p=ph, se_used=su, lb=lb, contribution_lb=i["x"]["share"] * lb, share_d=i["x"]["share"] * d, p_in_bh=1.0,
                             bh=False, accept=False)
                out[r["name"]] = o
        return out

    def accept_rule(self, c, o):
        """AM15: BH accept inside (author, generation), and LOBO for a global candidate (judge_bh.judge)."""
        return bool(o.get("accept"))

    def interval(self, V):
        """AM15 interval decisions (judge_bh.interval): per (author, generation), accepted for good = BH accepts under every candidate's
        pessimistic completion (and LOBO holds there); rejected for good = not accepted under every candidate's optimistic completion;
        otherwise wait. Returns the candidates still waited for (undecided ones, then accepted ones whose order for the tip could still
        change: the tip takes the accepted candidate with the largest contribution_lb)."""
        und = [c for c in V if not c.get("decided")]
        if und:
            P, O = self.am15_numbers("pess", V), self.am15_numbers("opt", V)
            grp = collections.defaultdict(list)
            for c in V:
                grp[(c["author"], int(c.get("gen", self.g)))].append(c)
            for key, cs in grp.items():
                if all(c.get("decided") for c in cs):
                    continue
                if key[0] == "system" and key[1] == self.g and (self.G().get("bundle") or {}).get("state") == "planned":
                    continue                                     # AM23: K of (system, g) is known once the bundle is built or dropped
                pad = self.g_prior_pad(key, [c["cid"] for c in cs], {c["cid"]: c for c in cs})        # B4 (as in am15_numbers)
                dec = BH.interval([P[c["cid"]]["x"] for c in cs] + pad, [O[c["cid"]]["x"] for c in cs] + pad, ALPHA)
                pend = sum(P[c["cid"]]["pend"] for c in cs)
                for c in cs:
                    if c.get("decided") or dec[c["cid"]] == "wait":
                        continue
                    acc = dec[c["cid"]] == "accept"
                    o = P[c["cid"]] if acc else O[c["cid"]]
                    basis = "all reads" if pend == 0 else (f"the pessimistic bound ({pend} reads pending in {key[0]} g{key[1]})" if acc
                                                           else f"the optimistic bound ({pend} reads pending in {key[0]} g{key[1]})")
                    self.decide_rule(c, dict(o, accept=acc), basis)
                    if not acc and self.a.am17 and P[c["cid"]]["pend"] > 0:
                        if P[c["cid"]]["bh"]:                    # BH-significant even on its pessimistic completion (LOBO fails): its p counts
                            log(f"  {c['cid']}: rejected for good by LOBO only; its p still counts in the BH of {key[0]} g{key[1]}, so it is read to the end")
                        else:                                    # p > k alpha / K under every completion: p = 1 leaves every other decision as it is
                            self.cancel_reads(c, V)
            und = [c for c in V if not c.get("decided")]
            self._bounds = (P, O)
        if und:
            return und
        acc = [c for c in V if c.get("accepted")]
        if len(acc) < 2:
            return []
        P, O = self.am15_numbers("pess", acc), self.am15_numbers("opt", acc)       # contribution_lb is the candidate's own (K is irrelevant)
        cl = lambda X, c: (X.get(c["cid"]) or {}).get("contribution_lb", -1.0)
        best = max(acc, key=lambda c: cl(P, c))
        rivals = [c for c in acc if c is not best and cl(O, c) > cl(P, best)]
        return [c for c in [best] + rivals if (P.get(c["cid"]) or {}).get("pend", 0) > 0] if rivals else []

    def decide_rule(self, c, o, basis):
        c["decided"] = True
        c["decided_at"] = time.time()
        num = (f"p {o['p']:.4f}, BH at {ALPHA} over K {o['K']} of {o['group']}" + (f", LOBO {'holds' if o['lobo'] else 'fails'}" if o.get("is_global") else "")
               + f", lb {o['lb']:+.4f}, contribution_lb {o['contribution_lb']:+.4f}")
        if self.accept_rule(c, o):
            c.update(accepted=True, status=f"accepted on {basis} ({num})")
        else:
            c.update(accepted=False, status=f"not accepted on {basis} ({num})")
        c["decision"] = dict(basis=basis, rule="am15", **{k: o.get(k) for k in ("p", "d", "se", "se_used", "nu", "n_rows", "K", "group", "bh", "lobo",
                                                                                   "is_global", "lb", "contribution_lb", "share", "pend")})

    @staticmethod
    def eng_type(x):
        """server (vllm serve: E / T / cell server) | offline (the in-process W worker, server name W*); the engine's engine_type field wins."""
        t = str(x.get("engine_type") or "").lower()
        if t:
            return "offline" if t in ("offline", "w") else "server"
        return "offline" if str(x.get("server") or "").startswith("W") else "server"

    def engine_types(self, c):
        """Reads of the candidate (its rung rows, sample 0) and of its reference (the tip's samples on those rows) by engine type; mixed =
        the two sides were not read by the same engine type, which makes the comparison invalid (coordinator 11-10, g004c30)."""
        rd, rf = collections.Counter(), collections.Counter()
        tip = self.state["tip"]["entries"]
        for r in self.rung_rows(c, 1):
            x = self.io.get(self.read_req(self.cand_entries(c), r, self.s0(c), 1)[0]["key"])
            if x is not None:
                rd[self.eng_type(x)] += 1
                for p in self.samples(tip, r):
                    rf[self.eng_type(p)] += 1
        return dict(read=dict(rd), ref=dict(rf), mixed=bool(rd) and bool(rf) and set(rd) != set(rf))

    def note_ctx(self, c):
        """Coordinator 11-10 23:30: rows whose read could not run because the prompt exceeds the context count as unrouted (zero difference,
        never waited on, never wrong; gen.diffs); the count and the rows are kept on the candidate; over a quarter of its routed rows ->
        "mostly unusable" in the report (it is still judged on the rest)."""
        rows = list((self._ecomp.get(c["cid"]) or {}).get("skipped_ctx_rows") or [])
        if rows or c.get("skipped_ctx"):
            c.update(skipped_ctx=len(rows), skipped_ctx_rows=rows, mostly_unusable=len(rows) > 0.25 * max(int(c.get("routed") or 0), 1))

    def judge_obs(self, V, basis):
        """Numbers on the reads there are (obs) for every read candidate: AM15 (decides any still-undecided candidate by BH on these
        reads; used at the box and in the migration), judge_eb lfsr / post_mean and AM10 as estimates for the ledger only."""
        A, B = self.am15_numbers("obs", V), self.judge_numbers("obs", V)
        for c in V:
            a, o = A[c["cid"]], B.get(c["cid"]) or {}
            if not c.get("decided"):
                if a["n_read"] < 1:
                    c.update(decided=True, accepted=False, status=f"too few rows read ({a['n_read']})")
                else:
                    self.decide_rule(c, a, basis.format(pend=a["pend"]))
            pm = o.get("post_mean")
            c["judge"] = dict(rung=1, scale="routed", rule="am15", D=a["d"], se=a["se"], se_used=a["se_used"], clusters=a["clusters"], nu=a["nu"], n=a["n_read"],
                              n_rows=a["n_rows"], p=a["p"], bh=a["bh"], lobo=a["lobo"], is_global=a["is_global"], K=a["K"], group=a["group"], lb=a["lb"],
                              contribution_lb=a["contribution_lb"], share_d=a["share_d"], src=a["src"], contrib=a["contribution_lb"],
                              lfsr=o.get("lfsr"), post_mean=pm, contrib_pm=(c["share"] * pm) if pm is not None else None, family=o.get("family"),
                              mu=o.get("mu"), tau=o.get("tau"), am10=o.get("am10"))
            c["engine"] = self.engine_types(c)              # coordinator 11-10: which engine type read the candidate and its reference
            self.note_ctx(c)
            c.update(n_read=a["n_read"], tok_cand=a["tc"], tok_parent=a["tp"], ecomp=a.get("ec"), dw=dict(d=a["dd"], w=a["w"], scale="routed"),
                     pending_reads=a["pend"], contrib=a["contribution_lb"], contrib_d=a["share_d"])
            j = c["judge"]
            a10 = j.get("am10")
            log(f"  {c['cid']} {c['author']} {c['kind']}: rows {a['n_read']} (+{a['pend']} pending) share {c['share']:.2f} d {j['D']:+.4f} se {j['se']:.4f} "
                f"(used {j['se_used']:.4f}) p {j['p']:.4f} BH {'yes' if j['bh'] else 'no'} (K {j['K']})" + (f" LOBO {'ok' if j['lobo'] else 'FAILS'}" if j["is_global"] else "")
                + f" lb {j['lb']:+.4f} contribution_lb {j['contribution_lb']:+.4f} | est. lfsr {j['lfsr'] if j['lfsr'] is None else format(j['lfsr'], '.3f')} "
                f"pm {pm if pm is None else format(pm, '+.4f')}" + (f" | AM10 P(mu<=0) {a10['P_mu_le0']:.3f} mu {a10['mu_mean']:+.4f}" if a10 else "")
                + f" tok {c.get('tok_cand') or 0:.0f}/{c.get('tok_parent') or 0:.0f} -> {c['status']}")
        return A

    def judge(self, rung):
        """Family judge (judge_eb) per family key (parent, author, kind); a candidate uses all rungs it was read on."""
        fam = collections.defaultdict(list)
        for c in self.valid():
            if not c.get("rung"):
                continue
            rungs = [1] if c["rung"] == 1 else [1, 2]
            dd, w, n, tc, tp, _ = self.diffs(c, rungs)
            if n < 2:
                c.update(judge=dict(rung=rung, n=n), status=f"too few rows read ({n})")
                continue
            c.update(n_read=n, tok_cand=tc, tok_parent=tp, dw=dict(d=dd, w=w))         # per-cluster d and rows: the L1 covariance S (AM7) needs them
            fam[(c["parent"], c["author"], c["kind"])].append((c, dict(name=c["cid"], d=dd, w=w)))
        for key, xs in fam.items():
            out, hyper = EB.judge([x for _, x in xs], ALPHA)
            for (c, _), o in zip(xs, out):
                c["judge"] = dict(rung=rung, family="|".join(key), K=len(xs), mu=hyper.get("mu"), tau=hyper.get("tau"), **{k: o[k] for k in ("D", "se", "clusters", "lfsr", "post_mean")})
                if c.get("decided"):
                    continue
                if rung == 1 and o["lfsr"] <= CONT and self.rung_rows(c, 2) and not self.a.mini:
                    c["status"] = "to rung 2"                 # rung 1 decides only whether rung 2 is read (coordinator 11-10: one decision per candidate)
                    continue
                if rung == 1 and c.get("rung") == 2:
                    continue
                self.decide(c, o, f"{'all ' if rung == 2 or not self.rung_rows(c, 2) else 'rung-1 '}reads")
        for c in self.valid():
            j = c.get("judge")
            if j and j.get("rung") == rung and "D" not in j:          # too few rows read (Venus island B, 11-10): log only, no numbers
                log(f"  {c['cid']} {c['author']} {c['kind']}: rows {j.get('n')} -> {c['status']}")
            elif j and j.get("rung") == rung:
                log(f"  {c['cid']} {c['author']} {c['kind']}: rows {c.get('n_read')} share {c['share']:.2f} D {j['D']:+.4f} se {j['se']:.4f} lfsr {j['lfsr']:.3f} "
                    f"pm {j['post_mean']:+.4f} tok {c.get('tok_cand') or 0:.0f}/{c.get('tok_parent') or 0:.0f} -> {c['status']}")

    def decide(self, c, o, what):
        c["decided"] = True
        if o["accept"]:
            c.update(accepted=True, status=f"accepted on {what} (lfsr {o['lfsr']:.3f})")
        else:
            c["status"] = f"not accepted on {what} (lfsr {o['lfsr']:.3f})"

    def step_judge1(self):
        self.carry_decide(final=False)
        if not (self.a.two_rungs or self.a.rung1_rows):
            self.judge_obs([c for c in self.valid() if c.get("rung")], "the reads it has at the box ({pend} pending; BH on these reads)")
            return
        tip = self.state["tip"]["entries"]
        keys = []
        for c in self.valid():
            if c.get("rung"):
                for r in self.rung_rows(c, 1):
                    _, _, sg = self.view(tip, r)
                    keys.append(self.read_key(sg, r, 0))
        done, miss = self.io.wait(list(dict.fromkeys(keys)), self.deadline + GRACE, label="parent on rung-1 rows")
        self.judge(1)

    def step_rung2(self):
        C = [c for c in self.valid() if c.get("status") == "to rung 2"]
        if not C:
            return
        reads = sum(len(self.rung_rows(c, 2)) for c in C)
        rate = self.io.tok_rate() or self.a.assume_rate
        est = reads * self.a.tok_per_read / max(rate, 1.0)
        left = self.deadline - time.time()
        if est > left + GRACE:
            for c in C:                                   # the box closed rung 2: decided on the reads it has (rung 1)
                o = {k: c["judge"][k] for k in ("lfsr",)}
                o["accept"] = c["judge"]["lfsr"] <= ALPHA
                self.decide(c, o, f"rung-1 reads (rung 2 not started: ~{est / 60:.0f} min needed, {left / 60:.0f} left)")
            log(f"  rung 2 not started: {reads} reads ~{est / 60:.0f} min > {left / 60:.0f} min left + grace")
            return
        for c in C:
            c["rung"] = 2
        self.G()["rung2_missing"] = self.read_rung(2, C)

    def step_judge2(self):
        if any(c.get("rung") == 2 for c in self.valid()):
            tip = self.state["tip"]["entries"]
            keys = [self.read_key(self.view(tip, r)[2], r, 0) for c in self.valid() if c.get("rung") == 2 for r in self.rung_rows(c, 2)]
            self.io.wait(list(dict.fromkeys(keys)), self.deadline + GRACE, label="parent on rung-2 rows")
            self.judge(2)

    def step_tip(self):
        self.carry_decide(final=True)
        acc = [c for c in self.valid() if c.get("accepted")] + self.carry_accepted()
        G = self.G()
        if not acc:
            G["tip_change"] = None
            log(f"  tip unchanged: {self.state['tip']['id']}")
            self.write_tip()
            return
        best = max(acc, key=lambda c: (self.cl_of(c), -(c.get("tok_cand") or 0)))      # AM15: the largest contribution_lb = share x lb
        old = self.state["tip"]
        new = dict(id=f"t{self.g:03d}", entries=old["entries"] + self.flat([best["entry"]]), born=self.g, parent=old["id"], from_cand=best["cid"], author=best["author"])
        self.state["tips"].append(old)
        self.state["tip"] = new
        best["merged"] = True
        via = set(self.bundle_members_merged(best))
        for c in acc:
            if c is not best and c["cid"] not in via:
                c["status"] += " (not merged: one entry per generation in Stage 1)"
        G["tip_change"] = dict(old=old["id"], new=new["id"], cand=best["cid"], author=best["author"])
        log(f"  NEW TIP {new['id']} = {old['id']} + {best['cid']} ({best['author']}, {best['kind']}): {json.dumps(best['entry'], ensure_ascii=False)[:300]}")
        self.write_tip()

    def bundle_members_merged(self, b):
        """AM23: a merged bundle's members are tip entries now, recorded on each member as merged through the bundle (merged_through; their
        own merged flag stays False, so author counts and L1 see the bundle as the merge)."""
        if not b.get("entry", {}).get("parts") or b.get("kind") not in ("bundle", "synthetic"):
            return []
        ids = [x.get("cid") for x in b["entry"]["parts"]]
        allc = {c["cid"]: c for G in self.state["gens"].values() for c in G.get("cands", [])}
        for i in ids:
            m = allc.get(i)
            if m is not None and not m.get("merged_through"):
                m["merged_through"] = b["cid"]
                m["status"] = (m.get("status") or "") + f" | merged through bundle {b['cid']} (g{self.g})"
        return ids

    @staticmethod
    def cl_of(c):
        """contribution_lb of a judged candidate (AM15: share x (d - t_{nu,1-alpha} se)); the final one once all its reads are in; a
        candidate re-judged by the LITE carry-in (automatic or by hand) uses its pooled numbers."""
        for x in ((c.get("carry") or {}) if (c.get("carry") or {}).get("decided") else {}, c.get("carry_in") or {}, c.get("final") or {}, c.get("judge") or {}):
            if x.get("contribution_lb") is not None:
                return float(x["contribution_lb"])
        return float(c["contrib"]) if c.get("contrib") is not None else -1.0

    def write_tip(self):
        """tip.json (current) and tips/<id>.json: the executor harness as the engine applies it, for ID / OOD reads elsewhere (Venus).
        On a row: route each conditional entry (router v1, engine_api §3); the procedures of the entries that hold, joined by a blank line in
        this order, are a text part right before the prompt text; their settings override the stock settings in order."""
        t = self.state["tip"]
        x = dict(id=t["id"], born=t.get("born"), parent=t.get("parent"), run=self.rd, stock_settings=self.stock,
                 router=dict(wording="Condition: <c>.\nDo not solve the question above. Does this condition hold for it? Reply with exactly one word: yes or no.",
                             note="engine strips a trailing period from <c>; row content first, condition as the last text part; thinking off, T 0, "
                                  "constrained choice yes/no, p = P(yes)/(P(yes)+P(no)) over first-token top-20 variants, route iff p >= .5"),
                 ops_note="ops of the entries that hold on a row are applied together (each op once, a later entry's tiles n wins), in the order "
                          "zoom -> tiles -> read -> check (engine_api §7); a row with more than 40 images incl. crops/tiles or over the context gets no ops",
                 entries=[dict(condition=e["condition"], procedure=e.get("procedure"), settings=e.get("settings") or {}, ops=e.get("ops") or [],
                               **({"edit": e["edit"]} if e.get("edit") else {}), from_cand=e.get("cid"), author=e.get("author")) for e in t["entries"]])
        te = [e["edit"] for e in self.flat(t["entries"]) if e.get("edit")]          # engine_api §8: the tip's weights for the T exporter
        x["edits"] = None if not te else (te[0] if len(te) == 1 else te)
        x["edit_sha"] = esha(x["edits"]) if te else None
        os.makedirs(os.path.join(self.rd, "tips"), exist_ok=True)
        jdump(os.path.join(self.rd, "tips", f"{t['id']}.json"), x)
        jdump(os.path.join(self.rd, "tip.json"), x)
        self.signal_tick("tip change", force=True)                # B1: the new tip's export at once (rows not yet routed follow later)

    def step_report(self):
        G = self.G()
        led = os.path.join(self.rd, "ledger.jsonl")
        have = set()
        if os.path.exists(led):
            for l in open(led):
                try:
                    have.add(json.loads(l)["id"])
                except (ValueError, KeyError):
                    pass
        with open(led, "a") as f:
            for c in G["cands"]:
                if c["cid"] in have:
                    continue
                j = c.get("judge") or {}
                e = c.get("entry") or {}
                f.write(json.dumps(dict(id=c["cid"], gen=self.g, author=c["author"], target=c.get("target", "E.H"), kind=c.get("kind"), payload=dict(condition=e.get("condition"), procedure=e.get("procedure"), settings=e.get("settings"), ops=e.get("ops") or [], parts=[x.get("cid") for x in e.get("parts") or []] or None, edit=e.get("edit")),
                                        parent=c["parent"], routed=c.get("routed"), share=c.get("share"), rows_read=c.get("rows_read"), D=j.get("D"), se=j.get("se"), clusters=j.get("clusters"),
                                        lfsr=j.get("lfsr"), post_mean=j.get("post_mean"), family=j.get("family"), tok_cand=c.get("tok_cand"), tok_parent=c.get("tok_parent"),
                                        accepted=bool(c.get("accepted")), merged=bool(c.get("merged")), status=c["status"], why=c.get("why"), prop_tokens=c.get("prop_tokens"),
                                        card_source=c.get("card_source"), card_wrong=c.get("card_wrong"), card_right=c.get("card_right"),
                                        repaired=c.get("repaired"), repair_reason=c.get("repair_reason"), lab_tested=c.get("lab_tested"), route=c.get("route"),
                                        engine=c.get("engine"),
                                        decision=c.get("decision"), am10=j.get("am10"), pending_reads=c.get("pending_reads"), contrib=c.get("contrib"), scale=j.get("scale"),
                                        reads_cancelled=bool(c.get("reads_cancelled")), cancelled_reads=c.get("cancelled_reads"), futile=bool(c.get("futile")),
                                        w_stage=c.get("w_stage"), am21={k: v for k, v in (c.get("am21") or {}).items()} or None,
                                        members=c.get("members"), bundle=c.get("bundle"), merged_through=c.get("merged_through"),
                                        skipped_ctx=c.get("skipped_ctx"), skipped_ctx_rows=c.get("skipped_ctx_rows"), mostly_unusable=c.get("mostly_unusable"),
                                        rule=j.get("rule"), am15={k: j.get(k) for k in ("p", "se_used", "nu", "n_rows", "K", "group", "bh", "lobo", "is_global", "lb",
                                                                                         "contribution_lb", "share_d", "contrib_pm", "src")},
                                        wall=time.time() - G["t0"]), ensure_ascii=False) + "\n")
        by = collections.defaultdict(lambda: dict(slots=0, valid=0, repaired=0, read=0, accepted=0, merged=0))
        for c in G["cands"]:
            b = by[c["author"]]
            b["slots"] += 1
            b["valid"] += bool(c.get("valid0"))
            b["repaired"] += bool(c.get("repaired"))
            b["read"] += bool(c.get("rung"))
            b["accepted"] += bool(c.get("accepted"))
            b["merged"] += bool(c.get("merged"))
        src = collections.defaultdict(lambda: dict(slots=0, valid=0, read=0, accepted=0))
        for c in G["cands"]:
            if c.get("card_source"):
                s = src[c["card_source"]]
                s["slots"] += 1
                s["valid"] += bool(c.get("valid0"))
                s["read"] += bool(c.get("rung"))
                s["accepted"] += bool(c.get("accepted"))
        rep_breeder = self.breeder()
        try:
            allc = [c for G2 in self.state["gens"].values() for c in G2.get("cands", [])]
            _, ysum = L1.posteriors(allc, sorted({c["author"] for c in allc}), ALPHA)
        except Exception as ex:                                  # report-only: never stop the loop on it
            ysum = {"error": f"{type(ex).__name__}: {ex}"}
        steps = dict(G.get("times") or {})
        try:
            dep = self.deployment()
        except Exception as ex:                                  # report-only
            dep = {"error": f"{type(ex).__name__}: {ex}"}
        rep = dict(gen=self.g, tip=self.state["tip"]["id"], tip_change=G.get("tip_change"), authors=by, sources=dict(src), source_alloc=G.get("sources"),
                   repairs_sent=G.get("repairs", 0), slots=G.get("slots"), wall_s=time.time() - G["t0"], steps=steps, rate_tok_s=self.io.tok_rate(),
                   routes={c["cid"]: c.get("route") for c in G["cands"] if c.get("route")}, breeder=rep_breeder, Y=ysum, deployment=dep, tip_file=os.path.join(self.rd, "tip.json"))
        jdump(os.path.join(self.rd, f"report_g{self.g:03d}.json"), rep)
        log(f"REPORT g{self.g}: tip {rep['tip']} change {rep['tip_change']} | " + " | ".join(f"{a}: slots {b['slots']} valid {b['valid']} (repaired {b['repaired']}) read {b['read']} acc {b['accepted']}" for a, b in by.items())
            + f" | wall {rep['wall_s'] / 60:.1f} min")
        b = rep_breeder
        log("  Y (AM7, expected accepted gain per slot, all generations so far): " + ", ".join(
            f"{a} {v['mean']:+.4f} (sd {v['sd']:.4f}, K {v['K']})" for a, v in ysum.items() if isinstance(v, dict)))
        log(f"  breeder's check (AM9, tip {b.get('tip')}): merged {b['merged']} {b.get('cands')}: sum contribution_lb {b['sum_contribution_lb']:+.4f}, "
            f"sum share x d {b['sum_share_d']:+.4f} (est. sum share x post_mean {b['sum_contrib_pm']:+.4f}) vs realised tip - stock on "
            f"MONITOR {'n/a' if b['realised'] is None else format(b['realised'], '+.4f')} ({b['monitor_rows']} rows with both)")
        if isinstance(dep, dict) and dep.get("k"):
            log(f"  deployment (k-sample plurality of the tip's cached samples, MONITOR, {dep['rows']} rows): "
                + ", ".join(f"k={k} {v:.4f}" for k, v in dep["k"].items()) + f"; mean right per sample {dep['mean']:.4f}")
        log("  wall per step (min): " + ", ".join(f"{k} {v / 60:.1f}" for k, v in steps.items()))

    def deployment(self, ks=(1, 3, 5, 8)):
        """S2-B: k-sample plurality is a deployment-time compute choice, reported per tip (never an edit): accuracy of the plurality answer
        of the first k cached samples of the current tip on MONITOR rows, over the rows that have at least max(ks) samples."""
        tip = self.state["tip"]["entries"]
        if not all(e["cond"] == "always" for e in tip):
            self.load_routes(tip, self.role["monitor"])
        acc, mean, n = collections.defaultdict(float), 0.0, 0
        for r in self.role["monitor"]:
            ps = ECOMP.read_samples(self.samples(tip, r))
            if len(ps) < max(ks):
                continue
            n += 1
            mean += sum(bool(p.get("ok")) for p in ps) / len(ps)
            for k in ks:
                acc[k] += float(bool(ECOMP.plurality_ok(ps, self.rows[r], k)))
        return dict(rows=n, k={k: acc[k] / n for k in ks} if n else {}, mean=mean / n if n else None, tip=self.state["tip"]["id"])

    def finalise(self):
        """Candidates decided while reads were pending get their final numbers (AM15 on all reads, judge_eb and AM10 estimates, dw for L1)
        once all reads of their generation have landed; one ledger line per candidate with final=true. Option D as an assertion only: a
        decision the final judgement contradicts is logged loudly; an accepted candidate that fails loses its acceptance (never stacked),
        and a merged one is reverted out of the tip at the next generation boundary (self.revert_pending)."""
        cur_g, cur_tip = self.g, self.state["tip"]
        tips = {t["id"]: t for t in self.state["tips"] + [cur_tip]}
        led = os.path.join(self.rd, "ledger.jsonl")
        try:
            for gk in sorted(self.state["gens"], key=int):
                G = self.state["gens"][gk]
                if int(gk) >= cur_g or not any(c.get("pending_reads") and not c.get("final") for c in G.get("cands", [])):
                    continue
                V = [c for c in G["cands"] if c.get("rung") and c.get("entry")]
                if not V or V[0]["parent"] not in tips:
                    continue
                self.g, self.state["tip"] = int(gk), tips[V[0]["parent"]]
                self.load_routes(self.state["tip"]["entries"], self.judge_rows)
                self.load_routes([c["entry"] for c in V], self.judge_rows)
                A = self.am15_numbers("obs", V)
                if any(A[c["cid"]]["pend"] for c in V):
                    continue                                             # not all landed yet: try at the next generation
                B = self.judge_numbers("obs", V)
                with open(led, "a") as f:
                    for c in V:
                        a, o = A[c["cid"]], B.get(c["cid"]) or {}
                        c["final"] = dict(rule="am15", D=a["d"], se=a["se"], se_used=a["se_used"], nu=a["nu"], n=a["n_read"], p=a["p"], bh=a["bh"], lobo=a["lobo"],
                                          accept=a["accept"], K=a["K"], lb=a["lb"], contribution_lb=a["contribution_lb"], share_d=a["share_d"],
                                          lfsr=o.get("lfsr"), post_mean=o.get("post_mean"), am10=o.get("am10"), clusters=a["clusters"])
                        c["dw"] = dict(d=a["dd"], w=a["w"], scale="routed")
                        c["engine"] = self.engine_types(c)
                        c["final"]["engine"] = c["engine"]
                        self.note_ctx(c)
                        c["pending_reads"] = 0
                        c["contrib"], c["contrib_d"] = a["contribution_lb"], a["share_d"]
                        if (bool(a["accept"]) != bool(c.get("accepted")) and c.get("decided") and a["n_read"] >= 1
                                and not c.get("withdrawn")):           # a withdrawn acceptance (g004c30, 11-10) is not a contradiction
                            c["final_contradiction"] = True
                            log(f"  !!! AM15 ASSERTION: {c['cid']} ({c['author']}, g{gk}) was decided {'accepted' if c.get('accepted') else 'not accepted'} "
                                f"({(c.get('decision') or {}).get('basis')}) but the final judgement on all reads says {'accept' if a['accept'] else 'reject'} "
                                f"(p {a['p']:.4f}, K {a['K']}, LOBO {a['lobo']})")
                            if c.get("accepted"):
                                c["accepted"] = False
                                c["status"] += " | FINAL JUDGEMENT FAILS (AM15 assertion): acceptance withdrawn"
                                if c.get("merged"):
                                    self.state.setdefault("revert_pending", []).append(c["cid"])
                                    log(f"  !!! {c['cid']} is merged: it is reverted out of the tip at the next generation boundary")
                        f.write(json.dumps(dict(id=c["cid"], gen=int(gk), final=True, accepted=bool(c.get("accepted")), **c["final"],
                                                **({"reads_cancelled": True, "cancelled_reads": c.get("cancelled_reads")} if c.get("reads_cancelled") else {}),
                                                **({"futile": True} if c.get("futile") else {}),
                                                **({"am21": c["am21"]} if c.get("am21") else {})), ensure_ascii=False) + "\n")
                log(f"  finalised generation {gk}: {len(V)} candidates (AM15 on all reads)")
        except Exception as ex:                                          # bookkeeping only: never stop the loop on it
            log(f"  finalise: {type(ex).__name__}: {ex}")
        finally:
            self.g, self.state["tip"] = cur_g, cur_tip
            self.routes, self.route_p = {}, {}
            self.load_routes(cur_tip["entries"], self.role["show"] + self.role["judge"] + self.role["monitor"])
            G = self.state["gens"].get(str(cur_g)) or {}
            if G.get("cands"):
                self.load_routes([c["entry"] for c in G["cands"] if c.get("entry")], self.judge_rows)

    def revert_merges(self):
        """AM15 Option D (assertion): at a generation boundary (before the generation's first step), take the entries a failed merged
        candidate added out of the tip; the result is a new tip t<g>r (parent = the current tip) written and logged as a tip change."""
        todo = [cid for cid in self.state.get("revert_pending", [])]
        if not todo:
            return
        allc = {c["cid"]: c for G in self.state["gens"].values() for c in G.get("cands", [])}
        hist = self.state["tips"] + [self.state["tip"]]
        ents = list(self.state["tip"]["entries"])
        for cid in todo:
            for i, t in enumerate(hist):
                if t.get("from_cand") == cid and i > 0:
                    par = next((u for u in hist if u["id"] == t.get("parent")), hist[i - 1])
                    for e in t["entries"][len(par["entries"]):]:
                        k = json.dumps(e, sort_keys=True)
                        j = next((n for n, x in enumerate(ents) if json.dumps(x, sort_keys=True) == k), None)
                        if j is not None:
                            ents.pop(j)
                    break
            c = allc.get(cid)
            if c is not None:
                c["merged"], c["reverted_g"] = False, self.g
                c["status"] += f" | reverted out of the tip at the start of g{self.g}"
        old = self.state["tip"]
        why = self.state.pop("revert_reason", None) or "AM15 assertion: merged candidate failed its final judgement"
        new = dict(id=f"t{self.g:03d}r", entries=ents, born=self.g, parent=old["id"], from_cand=None, author="system", reverted=todo, reason=why)
        self.state["tips"].append(old)
        self.state["tip"] = new
        self.state["revert_pending"] = []
        self.state.setdefault("reverts", []).append(dict(g=self.g, cands=todo, old=old["id"], new=new["id"], reason=why))
        log(f"  !!! NEW TIP {new['id']} = {old['id']} without {todo} ({why})")
        self.write_tip()
        self.save()

    def carry_merges(self):
        """AM21 LITE carry-in merge (coordinator 11-10): a candidate of an earlier generation that an annotation marked carry_merge
        (accepted on pooled samples by the carry-in re-judgement of its birth family) merges into the tip at a generation boundary when it
        was read against the tip's content: the tip's entries equal its parent's -> tip + entry; or the tip is the previous generation's
        merge on that content -> the larger contribution_lb of the two is merged (from the content both were read against), the other stays
        accepted and unmerged (the stack reads it on the new tip next). Otherwise it stays accepted and unmerged (stack). One merge per
        boundary; the new tip is t<g>c."""
        allc = {c["cid"]: c for G in self.state["gens"].values() for c in G.get("cands", [])}
        for x in self.state.get("synthetic", []):                # AM23 hand bundle (H1): a synthetic multi-entry candidate (entry.parts)
            allc.setdefault(x["cid"], x)
        C = [c for c in allc.values() if c.get("carry_merge") and c.get("accepted") and not c.get("merged") and not c.get("carry_merge_done")]
        if not C:
            return
        tips = {t["id"]: t for t in self.state["tips"] + [self.state["tip"]]}
        key = lambda es: json.dumps(es, sort_keys=True)
        cl = lambda c: float((c.get("carry_in") or {}).get("contribution_lb", -1.0))
        tip = self.state["tip"]
        led, notes = os.path.join(self.rd, "ledger.jsonl"), []
        for c in sorted(C, key=lambda c: (-cl(c), c["cid"])):
            c["carry_merge_done"] = self.g
            tk = {PR._entry_key(e) for e in self.flat(tip["entries"])}
            new_parts = [x for x in self.flat([c["entry"]]) if PR._entry_key(x) not in tk]
            if not new_parts:                                    # its entries are in the tip already (e.g. merged through a bundle): nothing to add
                c["status"] = (c.get("status") or "") + f" | carry-in: not merged at the g{self.g} boundary, its entries are in the tip {tip['id']} already"
                notes.append(c)
                log(f"  carry-in {c['cid']}: its entries are in the tip {tip['id']} already -> nothing to merge")
                continue
            base = (tips.get(c["parent"]) or {}).get("entries")
            X, on = None, None
            if base is not None and key(tip["entries"]) == key(base):
                on = tip
            elif (base is not None and tip.get("from_cand") and tip.get("born") == self.g - 1 and tip.get("parent") in tips
                  and key(tips[tip["parent"]]["entries"]) == key(base)):
                X = allc.get(tip["from_cand"])
                if X is not None and self.cl_of(X) >= cl(c):
                    c["status"] += (f" | carry-in: not merged at the g{self.g} boundary, {X['cid']} (merged in g{self.g - 1}) has the larger "
                                    f"contribution_lb ({self.cl_of(X):+.4f} >= {cl(c):+.4f}); stays accepted for the stack")
                    notes.append(c)
                    log(f"  carry-in {c['cid']}: {X['cid']} merged in g{self.g - 1} has the larger contribution_lb -> stays accepted for the stack")
                    continue
                on = tips[tip["parent"]]
            else:
                c["status"] += (f" | carry-in: not merged at the g{self.g} boundary, the tip {tip['id']} is not the content {c['parent']} it was "
                                f"read against; stays accepted for the stack")
                notes.append(c)
                log(f"  carry-in {c['cid']}: the tip {tip['id']} has other content than {c['parent']} -> stays accepted for the stack")
                continue
            add = [x for x in self.flat([c["entry"]]) if PR._entry_key(x) not in {PR._entry_key(e) for e in self.flat(on["entries"])}]
            new = dict(id=f"t{self.g:03d}c", entries=on["entries"] + add, born=self.g, parent=tip["id"], from_cand=c["cid"],
                       author=c["author"], carry_in=True)
            if X is not None:
                X["merged"] = False
                X["status"] += (f" | unmerged at the g{self.g} boundary: carry-in {c['cid']} has the larger contribution_lb ({cl(c):+.4f} > "
                                f"{self.cl_of(X):+.4f}); stays accepted for the stack")
                notes.append(X)
            self.state["tips"].append(tip)
            self.state["tip"] = new
            c["merged"] = True
            c["status"] += f" | merged at the g{self.g} boundary (AM21 LITE carry-in): NEW TIP {new['id']}"
            notes += [m for m in (allc.get(i) for i in self.bundle_members_merged(c)) if m is not None]
            notes.append(c)
            self.state.setdefault("carry_merges", []).append(dict(g=self.g, cand=c["cid"], old=tip["id"], new=new["id"], base=on["id"],
                                                                   unmerged=X["cid"] if X else None))
            self.G()["boundary_tip_change"] = dict(old=tip["id"], new=new["id"], cand=c["cid"], author=c["author"], carry_in=True,
                                                   unmerged=X["cid"] if X else None)
            log(f"  !!! NEW TIP {new['id']} = {on['id']} + {c['cid']} ({c['author']}, AM21 LITE carry-in, contribution_lb {cl(c):+.4f})"
                + (f"; {X['cid']} unmerged, stays accepted for the stack" if X else "") + f": {json.dumps(c['entry'], ensure_ascii=False)[:200]}")
            self.write_tip()
            break
        with open(led, "a") as f:
            for x in notes:
                f.write(json.dumps(dict(id=x["cid"], gen=int(x.get("gen") or 0), annotation=True, t=time.time(), status=x["status"],
                                        accepted=bool(x.get("accepted")), merged=bool(x.get("merged")), carry_in=x.get("carry_in")), ensure_ascii=False) + "\n")
        self.save()

    # ------------------------------------------------------------------ AM21 LITE carry-in (coordinator 11-10, automatic, one time)
    def carry_select(self):
        """At the first generation created under LITE with this code (once per run): every candidate of an earlier NON-LITE generation that
        (a) was read against a parent whose entries equal the tip's now, (b) is not accepted (nor merged, withdrawn, left, cancelled, a
        weight cell), (c) has one-sided p <= alpha on its routed rows (final numbers, else judge), or had reads missing at its decision and
        d > 0, enters step 2 once: samples 0-3 on its routed JUDGE rows (only the missing ones are new: content-addressed keys), submitted
        now, alongside propose / lab. Families re-judged by hand (a member carries carry_in, or listed in --carry-skip) are left out.
        Everything else has left (p = 1). Selecting on step-1 data is a futility rule, so the re-judgement stays exact."""
        if not self.a.carry or self.state.get("carry_in_run") or not self.lite_on():
            return
        tip = self.state["tip"]
        key = lambda es: json.dumps(es, sort_keys=True)
        tips = {t["id"]: t for t in self.state["tips"] + [tip]}
        skip = set(self.a.carry_skip or [])
        sf = os.path.join(self.rd, "carry_skip.json")                # families re-judged by hand, listed by the coordinator (no restart needed)
        if os.path.exists(sf):
            try:
                skip |= set(json.load(open(sf)))
            except ValueError as ex:
                log(f"  carry_skip.json: {ex}")
        G = self.G()
        G.setdefault("order", self.order())
        picked, fams, skipped = [], collections.defaultdict(list), []
        for gk in sorted(self.state["gens"], key=int):
            Gb = self.state["gens"][gk]
            if int(gk) >= self.g or Gb.get("am21") == "lite":
                continue
            byfam = collections.defaultdict(list)
            for c in Gb.get("cands", []):
                if c.get("rung") and c.get("entry"):
                    byfam[f"{c['author']}|g{int(gk)}"].append(c)
            for fam, cs in sorted(byfam.items()):
                if fam in skip or any(c.get("carry_in") for c in cs):
                    skipped.append(fam)
                    continue
                for c in cs:
                    if key((tips.get(c["parent"]) or {}).get("entries")) != key(tip["entries"]):
                        continue
                    if (c.get("accepted") or c.get("merged") or c.get("withdrawn") or c.get("futile") or c.get("reads_cancelled") or self.is_w(c)
                            or (c.get("entry") or {}).get("parts")):
                        continue
                    j = c.get("final") or c.get("judge") or {}
                    p, d, pend = j.get("p"), j.get("D", j.get("d")), int((c.get("decision") or {}).get("pend") or 0)
                    if p is None:
                        continue
                    why = ("p <= alpha" if p <= ALPHA else "d > 0 with reads missing at its decision" if (pend and d is not None and d > 0) else None)
                    if not why:
                        continue
                    c["carry"] = dict(g=self.g, M=LITE_M, n=LITE_M, why=why, family=fam, t=time.time(),
                                      step1=dict(d=d, se=j.get("se"), p=p, n_rows=j.get("n_rows") or j.get("n"), pend=pend))
                    picked.append(c)
                    fams[fam].append(c["cid"])
        self.state["carry_in_run"] = dict(g=self.g, picked=[c["cid"] for c in picked], families=dict(fams), skipped=skipped, t=time.time())
        G["carry"] = [c["cid"] for c in picked]
        if not picked:
            log(f"  AM21 LITE carry-in: nothing to carry (families re-judged by hand: {skipped})")
            self.save()
            return
        self.load_routes([c["entry"] for c in picked], self.judge_rows)
        n = 0
        for c in picked:
            rq = [self.read_req(self.cand_entries(c), r, s_, 1)[0] for r in self.rung_rows(c, 1) for s_ in range(LITE_M)]
            n += self.io.submit(f"g{self.g:03d}_carry_{c['cid']}", rq)
        log(f"  AM21 LITE carry-in: {len(picked)} candidates of {len(fams)} families ({', '.join(c['cid'] + ' ' + c['carry']['why'] for c in picked)}); "
            f"{n} new reads (samples 0-3 on their routed JUDGE rows); families re-judged by hand left out: {skipped}")
        self.save()

    def lite_complete(self):
        """LITE completion (coordinator 11-10 00:35), at every boundary: each candidate of the previous generation (created under LITE) that was
        decided at the box end with samples missing (am21 box_at, decision pend > 0), has pooled d > 0 on the samples in hand, and is not
        accepted, left or a weight cell, gets its missing samples of the LITE_M (content-addressed keys: only the missing ones are new),
        submitted now alongside propose / lab; it is re-judged in its birth (author, generation) family on complete data, BH over the whole
        family with the others at p = 1 (carry_decide). Only if the tip content is unchanged since its birth generation (its reads are against
        that content); otherwise it is dropped. A frozen candidate (reads cancelled at the box end) is unfrozen; the frozen state is kept
        in c["completion"]["frozen"]."""
        if not self.a.carry or not self.lite_on():
            return
        runs = self.state.setdefault("carry_runs", {})
        if str(self.g) in runs:
            return
        gp = self.g - 1
        Gp = self.state["gens"].get(str(gp)) or {}
        run = dict(g=self.g, kind="LITE completion", birth=gp, picked=[], families={}, dropped=[], t=time.time())
        runs[str(self.g)] = run
        if Gp.get("am21") != "lite":
            run["why"] = f"g{gp} was not created under LITE"
            self.save()
            return
        tip = self.state["tip"]
        tips = {t["id"]: t for t in self.state["tips"] + [tip]}
        key = lambda es: json.dumps(es, sort_keys=True)
        G = self.G()
        G.setdefault("order", self.order())
        picked, fams = [], collections.defaultdict(list)
        for c in Gp.get("cands", []):
            a = c.get("am21") or {}
            if not c.get("rung") or not c.get("entry") or c.get("accepted") or c.get("merged") or self.is_w(c) or a.get("left") or a.get("box_at") is None:
                continue
            pend, d = int((c.get("decision") or {}).get("pend") or 0), (c.get("judge") or {}).get("D")
            if pend <= 0 or d is None or d <= 0:
                continue
            if key((tips.get(c["parent"]) or {}).get("entries")) != key(tip["entries"]):
                run["dropped"].append(c["cid"])
                continue
            fam = f"{c['author']}|g{gp}"
            c["completion"] = dict(g=self.g, frozen={k: c.get(k) for k in ("hand", "rows_in_hand", "reads_cancelled", "cancelled_reads")},
                                   at_box=dict(d=d, pend=pend, p=(c.get("judge") or {}).get("p"), box_at=a.get("box_at")))
            c.pop("hand", None)
            c.pop("rows_in_hand", None)
            c["reads_cancelled"] = False
            a["n"] = LITE_M
            c["carry"] = dict(g=self.g, M=LITE_M, n=LITE_M, why="LITE completion: decided at the box end with samples missing, d > 0 on the samples in hand",
                              family=fam, t=time.time(), step1=dict(d=d, p=(c.get("judge") or {}).get("p"), pend=pend))
            picked.append(c)
            fams[fam].append(c["cid"])
        run.update(picked=[c["cid"] for c in picked], families=dict(fams))
        G["carry"] = list(G.get("carry") or []) + [c["cid"] for c in picked]
        if run["dropped"]:
            log(f"  LITE completion: {len(run['dropped'])} box-ended candidates of g{gp} dropped (the tip content changed since): {run['dropped']}")
        if not picked:
            log(f"  LITE completion: nothing to complete from g{gp}")
            self.save()
            return
        self.load_routes([c["entry"] for c in picked], self.judge_rows)
        n = 0
        for c in picked:
            s0 = self.s0(c)
            rq = [self.read_req(self.cand_entries(c), r, s_, 1)[0] for r in self.rung_rows(c, 1) for s_ in range(s0, s0 + LITE_M)]
            n += self.io.submit(f"g{self.g:03d}_lcomp_{c['cid']}", rq)
        log(f"  LITE completion: {len(picked)} candidates of g{gp} in {len(fams)} families ({', '.join(c['cid'] for c in picked)}); {n} new reads "
            f"(their missing samples); re-judged in their birth families on complete data")
        self.save()

    def carry_runs_now(self):
        """The re-judgement runs of this generation: the one-time AM21 LITE carry-in (state carry_in_run) and the LITE completion
        (state carry_runs[g])."""
        out = []
        r = self.state.get("carry_in_run") or {}
        if r.get("g") == self.g:
            out.append(r)
        r2 = (self.state.get("carry_runs") or {}).get(str(self.g)) or {}
        if r2.get("g") == self.g:
            out.append(r2)
        return out

    def carry_accepted(self):
        """Carry-in / completion candidates accepted in this generation and not merged: they join the generation's accepted ones at the tip step."""
        allc = {c["cid"]: c for G in self.state["gens"].values() for c in G.get("cands", [])}
        return [allc[i] for run in self.carry_runs_now() for i in run.get("picked", [])
                if i in allc and (allc[i].get("carry") or {}).get("accepted") and not allc[i].get("merged")]

    def carry_decide(self, final=False):
        """Re-judge each birth (author, generation) family of the carry-in on pooled samples: BH at alpha over ALL its original read
        candidates, the ones not carried at p = 1, LOBO for global ones; once every carried member's reads are in (obs), or at the tip step
        under the pessimistic completion (unread samples of the 4 counted wrong). Accepted ones become accepted (they compete at the tip
        step by contribution_lb; the other accepted stay for the stack). One ledger annotation per carried candidate, birth generation kept."""
        for run in self.carry_runs_now():
            if run.get("families"):
                self._carry_decide_run(run, final)

    def _carry_decide_run(self, run, final):
        what = run.get("kind") or "AM21 LITE carry-in"
        allc = {c["cid"]: c for G in self.state["gens"].values() for c in G.get("cands", [])}
        done = run.setdefault("decided", {})
        lines = []
        for fam, ids in sorted(run["families"].items()):
            if fam in done:
                continue
            au, gb = fam.rsplit("|g", 1)
            members = [c for c in self.state["gens"][str(int(gb))].get("cands", []) if c.get("rung") and c.get("entry") and c["author"] == au]
            self.load_routes([c["entry"] for c in members], self.judge_rows)
            p1 = {c["cid"] for c in members if c["cid"] not in ids}
            A = self.am15_numbers("obs", members, p1=p1)
            complete = all(A[i]["pend"] == 0 for i in ids)
            if not complete and not final:
                continue
            if not complete:
                A = self.am15_numbers("pess", members, p1=p1)
            for i in ids:
                c, a = allc[i], A[i]
                acc = bool(a["accept"])
                c["carry"].update(decided=True, mode="all reads" if complete else "the pessimistic completion", accepted=acc, d=a["d"], se=a["se"],
                                  se_used=a["se_used"], p=a["p"], bh=a["bh"], lobo=a["lobo"], K=a["K"], lb=a["lb"], contribution_lb=a["contribution_lb"],
                                  n_rows=a["n_rows"], pend=a["pend"], tok=a["tc"], tok_parent=a["tp"], t_decided=time.time())
                if acc:
                    c["accepted"] = True
                c["status"] += (f" | re-judged under {what} in g{self.g} (birth g{int(gb)}): pooled d {a['d']:+.4f}, se {a['se']:.4f}, "
                                f"p {a['p']:.4f} at {LITE_M} samples per row on {a['n_rows']} rows, BH at {ALPHA} over K {a['K']} of {fam} (not carried p = 1)"
                                + (f", LOBO {'holds' if a['lobo'] else 'fails'}" if a["is_global"] else "")
                                + f", contribution_lb {a['contribution_lb']:+.4f}, on {c['carry']['mode']} -> {'accepted' if acc else 'not accepted'}")
                lines.append(dict(id=i, gen=int(gb), annotation=True, carry=True, rule=what, t=time.time(), status=c["status"], accepted=acc, carry_numbers=c["carry"]))
                log(f"  {what} {i} ({fam}): d {a['d']:+.4f} se {a['se']:.4f} p {a['p']:.4f} BH {a['bh']} (K {a['K']})"
                    + (f" LOBO {a['lobo']}" if a["is_global"] else "") + f" on {c['carry']['mode']} -> {'ACCEPTED' if acc else 'not accepted'}")
            done[fam] = "all reads" if complete else "pessimistic"
        if lines:
            with open(os.path.join(self.rd, "ledger.jsonl"), "a") as f:
                for l in lines:
                    f.write(json.dumps(l, ensure_ascii=False) + "\n")
            self.save()

    def breeder(self):
        """AM9 with AM15: predicted gain of the lineage on the whole-set scale (sum of contribution_lb = share x lb, and of share x d, of the
        merged candidates; judge_eb share x post_mean as an estimate) vs the realised gain of the tip over stock on MONITOR rows from cached
        samples (rows where both have samples)."""
        t = self.state["tip"]
        if (self.G().get("tip_change") or {}).get("new") == t["id"] and self.state["tips"]:   # a tip made at this generation's tip step has no
            t = self.state["tips"][-1]                          # MONITOR samples on its new rows yet: the realised gain is that of the tip this
                                                                # generation started from (a boundary / reset tip IS that tip; 11-10: t005r)
        merged = [c for G in self.state["gens"].values() for c in G.get("cands", []) if c.get("merged") and int(c.get("gen", 0)) <= int(t.get("born") or 0)]
        tip = t["entries"]
        diffs = []
        if tip:
            self.load_routes(tip, self.role["monitor"])
            for r in self.role["monitor"]:
                if not all(e["cond"] == "always" or (e["cond"], r) in self.routes for e in tip):
                    continue                                     # the tip's route on this row is not known yet (a tip born this generation)
                ts, ss = self.samples(tip, r), self.samples([], r)
                if ts and ss:
                    diffs.append(sum(bool(x.get("ok")) for x in ts) / len(ts) - sum(bool(x.get("ok")) for x in ss) / len(ss))
        jj = lambda c: c.get("final") or c.get("judge") or {}
        return dict(tip=t["id"], merged=len(merged), cands=[c["cid"] for c in merged],
                    sum_contribution_lb=sum(self.cl_of(c) for c in merged),                                       # AM15, whole-set scale
                    sum_share_d=sum((c.get("share") or 0.0) * (jj(c).get("D") or 0.0) for c in merged),            # whole-set scale
                    sum_contrib_pm=sum((c.get("share") or 0.0) * (jj(c).get("post_mean") or 0.0) for c in merged),  # judge_eb estimate only
                    sum_D=sum(jj(c).get("D") or 0.0 for c in merged),
                    realised=(sum(diffs) / len(diffs)) if diffs else (0.0 if not tip else None), monitor_rows=len(diffs) if tip else len(self.role["monitor"]))

    def am15_migrate(self, dry=False):
        """AM15 migration (coordinator 11-10 18:45 CST), once per run (state["am15"]): re-judge every generation whose judge step ran under
        the old rule from the reads it has (the AM15 interval rule while reads are still pending, at most --am15-wait s, then BH on the
        reads there are), keeping the old fields as *_v1. Tips of finished older generations stay (coordinator: t002 kept); for the newest
        judged generation whose tip step already ran, the tip is set from the AM15 judgement (an unsupported merge is reverted, the old tip
        file is kept as tips/<id>.pre_am15.json). Ledger lines of those generations are rewritten (backups *.pre_am15). dry: print only."""
        if self.state.get("am15") and not dry:
            return
        cur_tip = self.state["tip"]
        tips = {t["id"]: t for t in self.state["tips"] + [cur_tip]}
        gens = sorted((int(k) for k, G in self.state["gens"].items() if "judge1" in G.get("done", [])))
        if not gens:
            if not dry:
                self.state["am15"] = dict(t=time.time(), gens=[])
                self.save()
            return
        newest = max(int(k) for k in self.state["gens"])
        summary, keep_g = {}, getattr(self, "g", None)
        sp, lp_ = os.path.join(self.rd, "state.json"), os.path.join(self.rd, "ledger.jsonl")
        if not dry:
            for p in (sp, lp_):
                if os.path.exists(p) and not os.path.exists(p + ".pre_am15"):
                    shutil.copy2(p, p + ".pre_am15")
        for g in gens:
            G = self.state["gens"][str(g)]
            V = [c for c in G["cands"] if c.get("rung") and c.get("entry")]
            if not V or V[0]["parent"] not in tips:
                continue
            self.g, self.state["tip"] = g, tips[V[0]["parent"]]
            self.routes, self.route_p = {}, {}
            self.load_routes(self.state["tip"]["entries"], self.role["show"] + self.role["judge"] + self.role["monitor"])
            self.load_routes([c["entry"] for c in V], self.judge_rows)
            for c in V:
                if "accepted_v1" not in c:
                    c.update(judge_v1=c.get("judge"), accepted_v1=bool(c.get("accepted")), status_v1=c.get("status"), decision_v1=c.get("decision"),
                             contrib_v1=c.get("contrib"), merged_v1=bool(c.get("merged")), final_v1=c.get("final"))
                c.update(decided=False, accepted=False)
            t0 = time.time()
            while True:
                self.io.poll()
                wait = self.interval(V)
                if not wait or dry or time.time() - t0 >= self.a.am15_wait:
                    break
                if int(time.time() - t0) % 300 < 20:
                    log(f"  AM15 migration g{g}: waiting on {len(wait)} candidates ({', '.join(c['cid'] for c in wait[:8])}) after {time.time() - t0:.0f}s")
                time.sleep(20)
            log(f"AM15 re-judge of generation {g} (parent {V[0]['parent']}):")
            A = self.judge_obs(V, "the reads there were at the AM15 migration ({pend} pending; BH on these reads)")
            for c in V:
                c["status"] += f" [re-judged under AM15; v1: {'accepted' if c['accepted_v1'] else 'not accepted'}]"
                if c.get("final_v1") is not None and not A[c["cid"]]["pend"]:
                    a = A[c["cid"]]
                    c["final"] = dict(c["final_v1"] or {}, rule="am15", D=a["d"], se=a["se"], se_used=a["se_used"], nu=a["nu"], n=a["n_read"], p=a["p"],
                                      bh=a["bh"], lobo=a["lobo"], accept=a["accept"], K=a["K"], lb=a["lb"], contribution_lb=a["contribution_lb"], share_d=a["share_d"])
            acc = [c for c in V if c.get("accepted")]
            best = max(acc, key=lambda c: (self.cl_of(c), -(c.get("tok_cand") or 0))) if acc else None
            old_m = [c for c in V if c.get("merged_v1")]
            tipnote = None
            if "tip" in G.get("done", []) and g == newest and (cur_tip["id"] == V[0]["parent"] or cur_tip.get("from_cand") in {c["cid"] for c in V}):
                if (best is None and not old_m) or (best is not None and old_m and best is old_m[0]):
                    tipnote = f"tip {cur_tip['id']} supported by AM15 ({best['cid'] if best else 'no change'})"
                else:
                    par = tips[V[0]["parent"]]
                    for c in old_m:
                        c["merged"] = False
                        c["status"] += " (merge reverted under AM15)"
                    for c in acc:
                        c["status"] = c["status"].replace(" (not merged: one entry per generation in Stage 1)", "")
                    if best is not None:
                        new = dict(id=f"t{g:03d}", entries=par["entries"] + self.flat([best["entry"]]), born=g, parent=par["id"], from_cand=best["cid"],
                                   author=best["author"], am15=True)
                        best["merged"] = True
                        for c in acc:
                            if c is not best:
                                c["status"] += " (not merged: one entry per generation in Stage 1)"
                        G["tip_change"] = dict(old=par["id"], new=new["id"], cand=best["cid"], author=best["author"], am15=True)
                    else:
                        new = par
                        G["tip_change"] = None
                    G["tip_change_v1"] = dict(old=par["id"], new=cur_tip["id"], cand=old_m[0]["cid"] if old_m else None)
                    tipnote = (f"tip set from AM15: {cur_tip['id']} ({old_m[0]['cid'] if old_m else 'no merge'}) -> {new['id']} "
                               f"({best['cid'] if best else 'no merge: ' + par['id'] + ' stays'})")
                    if not dry:
                        old_f = os.path.join(self.rd, "tips", f"{cur_tip['id']}.json")
                        if cur_tip["id"] != par["id"] and os.path.exists(old_f):
                            shutil.copy2(old_f, os.path.join(self.rd, "tips", f"{cur_tip['id']}.pre_am15.json"))
                        hist = [t for t in self.state["tips"] if t["id"] != par["id"]] + [par] if best is not None else [t for t in self.state["tips"] if t["id"] != par["id"]]
                        self.state["tips"] = hist
                    cur_tip = new
                    log(f"  !!! {tipnote}")
            else:
                for c in V:
                    if c.get("merged") and not c.get("accepted"):
                        log(f"  !!! AM15: {c['cid']} is merged (tip kept by the coordinator's decision) but not accepted under AM15")
                tipnote = "older generation: tip kept"
            summary[g] = dict(parent=V[0]["parent"], tip=tipnote, accepted=[c["cid"] for c in V if c.get("accepted")],
                              accepted_v1=[c["cid"] for c in V if c.get("accepted_v1")], merged=[c["cid"] for c in V if c.get("merged")],
                              undecided_at_wait=[c["cid"] for c in V if c.get("pending_reads")],
                              rows=[dict(id=c["cid"], author=c["author"], kind=c.get("kind"), share=round(c["share"], 3), rows=A[c["cid"]]["n_read"],
                                         pend=A[c["cid"]]["pend"], d=round(A[c["cid"]]["d"], 4), se=round(A[c["cid"]]["se_used"], 4), p=round(A[c["cid"]]["p"], 5),
                                         K=A[c["cid"]]["K"], bh=A[c["cid"]]["bh"], glob=A[c["cid"]]["is_global"], lobo=A[c["cid"]]["lobo"],
                                         lb=round(A[c["cid"]]["lb"], 4), contribution_lb=round(A[c["cid"]]["contribution_lb"], 4), accepted=bool(c.get("accepted")),
                                         accepted_v1=c["accepted_v1"], merged=bool(c.get("merged")), basis=(c.get("decision") or {}).get("basis"))
                                    for c in V])
        self.g = keep_g
        self.state["tip"] = cur_tip
        self.routes, self.route_p = {}, {}
        unm = [c["cid"] for G in self.state["gens"].values() for c in G.get("cands", []) if c.get("accepted") and not c.get("merged")]
        out = dict(t=time.time(), gens=gens, tip=cur_tip["id"], accepted_unmerged=unm, summary=summary)
        if dry:
            print(json.dumps(out, indent=1, default=float))
            return out
        self.state["am15"] = {k: v for k, v in out.items() if k != "summary"}
        self.save()
        by = {c["cid"]: c for g in gens for c in self.state["gens"][str(g)]["cands"]}
        lines = []
        if os.path.exists(lp_):
            for l in open(lp_):
                x = json.loads(l)
                c = by.get(x.get("id"))
                if c is not None and x.get("gen") in gens and c.get("judge"):
                    j = c.get("final") if x.get("final") else c["judge"]
                    if "accepted_v1" not in x:
                        x.update(accepted_v1=x.get("accepted"), lfsr_v1=x.get("lfsr"), contrib_v1=x.get("contrib"), status_v1=x.get("status"), merged_v1=x.get("merged"))
                    x.update(rejudged="AM15", rule="am15", accepted=bool(c.get("accepted")), merged=bool(c.get("merged")),
                             am15={k: (j or {}).get(k) for k in ("D", "se", "se_used", "nu", "p", "K", "group", "bh", "lobo", "is_global", "lb", "contribution_lb", "share_d")},
                             lfsr=(j or {}).get("lfsr"), post_mean=(j or {}).get("post_mean"))
                    if not x.get("final"):
                        x.update(status=c["status"], decision=c.get("decision"), contrib=c.get("contrib"), D=c["judge"].get("D"), se=c["judge"].get("se"))
                lines.append(json.dumps(x, ensure_ascii=False))
            with open(lp_ + ".tmp", "w") as f:
                f.write("\n".join(lines) + "\n")
            os.replace(lp_ + ".tmp", lp_)
        for g in gens:
            rp = os.path.join(self.rd, f"report_g{g:03d}.json")
            if os.path.exists(rp):
                rep = json.load(open(rp))
                if "tip_v1" not in rep:
                    rep.update(tip_v1=rep.get("tip"), tip_change_v1=rep.get("tip_change"))
                rep.update(am15=summary.get(g))
                if g == newest:
                    rep.update(tip=cur_tip["id"], tip_change=self.state["gens"][str(g)].get("tip_change"))
                jdump(rp, rep)
        jdump(os.path.join(self.rd, "am15_migration.json"), out)
        self.write_tip()
        log(f"AM15 migration done: generations {gens}; tip {cur_tip['id']}; accepted, unmerged: {unm}; written am15_migration.json "
            f"(backups state.json.pre_am15, ledger.jsonl.pre_am15)")
        return out

    # ------------------------------------------------------------------ loop
    def generation(self, g):
        self.g = g
        G = self.state["gens"].setdefault(str(g), dict(t0=time.time(), done=[], times={}, ecomp=not self.a.no_ecomp, ecomp_round=bool(self.a.am16),
                                                       am21="lite" if self.a.lite else None))     # the rule is fixed when the generation is created
        self.stamp_cards(G)                                     # W on / E cards fixed when the generation is created (RUN/cards.json)
        self.save()                                             # the generation exists (and its clock runs) from here on
        self.deadline = G["t0"] + self.box_T()
        tip = self.state["tip"]["entries"]                      # resume: routes live in result files, reload the ones this generation uses
        self.load_routes(tip, self.role["show"] + self.role["judge"] + self.role["monitor"])
        if G.get("cands"):
            self.load_routes([c["entry"] for c in G["cands"] if c.get("entry")], self.judge_rows)
        self.finalise()                                         # AM11 / AM15: earlier generations' late reads, final judgement
        if not G["done"]:
            self.revert_merges()                                # AM15 Option D assertion: a merged candidate that failed its final judgement
            try:
                self.carry_merges()                             # AM21 LITE carry-in re-judged by hand (coordinator 11-10): merge at the boundary
            except Exception as ex:                             # never stop the loop on it; the candidate stays accepted for the stack
                log(f"  carry_merges: {type(ex).__name__}: {ex}")
            self.carry_select()                                 # AM21 LITE carry-in (automatic, one time): earlier near misses get samples 2-4
            try:
                self.lite_complete()                            # LITE completion (coordinator 11-10 00:35): box-ended candidates, every boundary
            except Exception as ex:                             # never stop the loop on it
                log(f"  lite_complete: {type(ex).__name__}: {ex}")
        G.setdefault("tip_at_start", self.state["tip"]["id"])   # B1: the tip this generation's reads are against (after the boundary merges)
        self.save()
        for st in STEPS:
            if st in G["done"]:
                continue
            if self.a.stop_after and st == self.a.stop_after:
                log(f"stop before step {st} (--stop-after)")
                sys.exit(3)
            t = time.time()
            log(f"g{g} step {st}")
            getattr(self, "step_" + st)()
            G["done"].append(st)
            G["times"][st] = round(time.time() - t, 1)
            if st in ("parent", "report"):
                self.signal_tick(f"after step {st}", force=True)  # B1
            if st == "report":
                try:
                    self.g_realised()                           # B2: predicted vs measured for this generation's G candidates
                    sd_ = os.path.join(self.rd, "signal")       # B4: refit the shrinkage, write the signal-quality number
                    G["signal_quality"] = SCELLS.calibrate(sd_).get("quality")
                    log("  " + SCELLS.quality_line(sd_))
                except Exception as ex:                         # never stop the loop on it
                    log(f"  g_realised: {type(ex).__name__}: {ex}")
            self.save()
        self.state["gen"] = g
        G.setdefault("t_end", time.time())                      # AM16: the window in which this generation's capacity is measured
        self.save()

    def main(self):
        if not self.a.no_am15_migrate:
            self.am15_migrate(dry=self.a.am15_dry)              # once per run: generations judged under the old rule are re-judged under AM15
            if self.a.am15_dry or self.a.am15_only:
                return
        g = max([int(k) for k in self.state["gens"]] or [0])
        n = self.a.gens
        if g and len(self.state["gens"][str(g)]["done"]) < len(STEPS):
            self.generation(g)                                  # a resumed generation counts as one of --gens, so run.sh picks up new code
            n -= 1                                              # at the next generation boundary
        for _ in range(max(0, n)):
            if os.path.exists(os.path.join(self.rd, "STOP")):
                log("STOP file: stopping at the generation boundary")
                return
            g = max([int(k) for k in self.state["gens"]] or [0]) + 1
            self.generation(g)


def args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/data/fd/mingdazhang/selfedit/v4")
    p.add_argument("--rows", default="/data/fd/mingdazhang/selfedit/v4/rows/learn.jsonl")
    p.add_argument("--knobs", default=os.path.join(HERE, "knobs.json"))
    p.add_argument("--run", required=True)
    p.add_argument("--gens", type=int, default=1)
    p.add_argument("--slots", type=int, default=0, help="slots per author; 0 = from measured throughput")
    p.add_argument("--max-slots", type=int, default=12)
    p.add_argument("--assume-rate", type=float, default=10800.0, help="capacity floor in tok/s over all E servers (7 x ~1550)")
    p.add_argument("--tok-per-read", type=float, default=2700.0, help="launch cache: 2659 generated tokens per stock sample")
    p.add_argument("--rung1-rows", type=int, default=0, help="override rung-1 size (mini / dry runs)")
    p.add_argument("--box", type=float, default=0)
    p.add_argument("--mini", action="store_true", help="1 B + 1 P0 candidate, rung 1 only")
    p.add_argument("--seed", default="v4")
    p.add_argument("--max-p0", type=int, default=30)
    p.add_argument("--p0-share", type=float, default=0.25, help="assumed routed share of a P0 entry when sizing slots (~100 of 444 JUDGE rows)")
    p.add_argument("--no-tfields", dest="tfields", action="store_false", help="no engine T / edit_sha / weights / tip_n on reads (engine_api §8): every edit read goes to W")
    p.add_argument("--no-carry", dest="carry", action="store_false", help="no automatic AM21 LITE carry-in at the first LITE generation")
    p.add_argument("--carry-skip", nargs="*", default=[], help="families re-judged by hand (\"P1|g5\" ...), left out of the carry-in")
    p.add_argument("--no-lite", dest="lite", action="store_false", help="AM21 LITE off for generations created from now on (one sample per candidate row)")
    p.add_argument("--no-am17", dest="am17", action="store_false", help="no futility rung for weight cells, no cancel of rejected reads (one rung1 batch)")
    p.add_argument("--no-am16", dest="am16", action="store_false", help="old sizing (assumed rate, max-p0, w-cells), knob lists for arm B, ceil(c)")
    p.add_argument("--am15-wait", type=float, default=900.0, help="AM15 migration: at most this many s of interval decisions on pending reads")
    p.add_argument("--no-am15-migrate", action="store_true", help="do not re-judge earlier generations under AM15 at start")
    p.add_argument("--am15-dry", action="store_true", help="print the AM15 re-judge of the judged generations and exit (no writes)")
    p.add_argument("--am15-only", action="store_true", help="run the AM15 migration (writes) and exit")
    p.add_argument("--l1-switch", action="store_true", help="pre-AM15-L1 gate, ignored now that --l1-rule exists (kept so old command lines parse)")
    p.add_argument("--l1-rule", choices=["am15", "am15perm", "none"], default="am15perm",          # coordinator 11-10: permutation from g5
                   help="L1 (AM15): am15 = Bayesian bootstrap over slots; am15perm = permutation P for decisions; none = authors reported only")
    p.add_argument("--l1-from", type=int, default=5, help="the L1 rule decides and draws slots from this generation on (earlier: report only, uniform extra slots)")
    p.add_argument("--no-robust-bounds", dest="robust_bounds", action="store_false",
                   help="AM11 on the candidate's own pending reads only (as in generations 1-3)")
    p.add_argument("--no-versions", action="store_true", help="Stage 1 propose (P0 + B only), no L1 step")
    p.add_argument("--no-signal", action="store_true", help="B1 off: no RUN/signal/<tip>/ traces export")
    p.add_argument("--no-g", action="store_true", help="B2 off: author G proposes nothing even when RUN/signal/<tip>/cells.json exists")
    p.add_argument("--no-w", action="store_true", help="no weight cells (engine W), e.g. on Venus")
    p.add_argument("--e-cards", type=int, default=0, help="AM16: scale the measured E capacity to this many E/T cards (per-card rate x N); 0 = as measured")
    p.add_argument("--w-cells", type=int, default=4, help="arm B weight cells per generation (W measured 11-10: one 444-read cell = 15.6 min per W card; 2 W cards)")
    p.add_argument("--no-stack", action="store_true", help="no stack candidate of accepted, unmerged entries")
    p.add_argument("--no-bundle", dest="bundle", action="store_false", help="AM23 off: no bundle of the generation's routed step-1 survivors")
    p.add_argument("--bundle-members", choices=["all", "narrow"], default="all",
                   help="AM23 member rule: all = eligible with step-1 d > 0; narrow = of those, routed JUDGE rows <= the median of the generation's "
                        "eligible candidates; RUN/bundle_rule.json {\"members\": ...} overrides it without a restart (read when the bundle is built)")
    p.add_argument("--no-show", action="store_true", help="AM8 off: no SHOW-row routes or reads of candidates")
    p.add_argument("--show-b", action="store_true", help="AM8: also read arm B's cells on SHOW rows")
    p.add_argument("--no-ecomp", action="store_true", help="S2-B off: the parent mean stays the reference for every candidate of new generations")
    p.add_argument("--pooled-global", action="store_true", help="decide global candidates on the pooled lfsr (pre-AM10)")
    p.add_argument("--two-rungs", action="store_true", help="old two-rung read (rung 1 = a quarter of JUDGE, rung 2 if lfsr <= .5); default is one rung")
    p.add_argument("--stop-after", default=None, help="exit (code 3) before this step: resume tests")
    return p.parse_args(argv)


if __name__ == "__main__":
    Run(args()).main()
