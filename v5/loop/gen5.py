"""selfedit v5 generation driver (design_1111_v5.md; coordinator 11-11: stage 2 on the stage-1 judge).

One generation = signals -> candidates -> prescreen -> judge -> update -> report. It reuses the v4 machinery for requests, keys, routes
and the proposer (gen.Run: read_req / view / sig / ensure_routes / check; v4io.IO; proposer cards) and replaces the judge, the references
and the tip by the v5 objects (../v5: judge_v5, policy5, stats5, guess5).

Tip (state["tip"]):
  entries   the GENERAL layer: always-on settings, accepted cells (edit) and operators (ops), text entries that passed the off-source rule
  special   the SPECIALISATION layer: text entries that are in-distribution only (ID_SPECIFIC) or unproven, each with its verdict and its
            source benchmark(s). OFF in the general system; reported separately; an entry leaves it only by passing the off-source rule.
  policy    the read policy: frontier (vertices chosen on SIGNAL rows, valued on JUDGE rows, both token conventions), default budget
            vertex (2 reads + a third on disagreement, x2.42) and the report points (x1.00, default, the last vertex); extras = accepted
            escalation policies (2 tip reads, a read with an accepted component on disagreement); guess = the declared-guess detector
            derived on SHOW + MONITOR reads. The budget is a reporting choice, never an improvement.
Steps
  signals     routes and cached reads of the tip (P_FULL samples on all learn rows); per-benchmark signals on SHOW + MONITOR (accuracy,
              disagreement, declared guess, tokens); the guess detector; the tip's frontier.
  candidates  (i) cells of stage G (RUN/signal/<tip>/cells.json; a cell needs its random-edit null to be accepted), (ii) operators
              (knobs "operators"), (iii) proposer text entries, source benchmark = the benchmark of the card's SHOW rows, (iv) audit of
              general-layer text entries (leave-one-entry-out), (v) re-tests of specialisation entries (--retest), (vi) read policies with
              an extra component (the grammar; offline, built in prescreen from the component's reads).
  prescreen   SIGNAL rows only, S_SCREEN samples (an entry: its routed rows outside its source benchmark). A candidate goes on when its
              macro gain at equal compute is > 0 with p_macro <= SCREEN_P; the best policy per extra component; at most K_MAX to JUDGE.
  judge       JUDGE rows, S_JUDGE samples; judge_v5.Judge.decide (p_gen, Holm over the screened family, no negative benchmark, source
              benchmark not scored). A weight cell that would be accepted gets its random-edit null read on JUDGE rows and is decided
              again with the null audit. Audit family separately (GENERAL / ID_SPECIFIC / UNPROVEN). Signal-row vs JUDGE-row effect of
              every judged candidate is appended to the shrinkage history (RUN/shrinkage.jsonl).
  admission   (inside judge; coordinator 11-11) a NEW proposer entry that fails off-source (screened out on signal rows, or rejected by the
              judge) is read, on top of the SPECIALISATION system (general + layer), on the routed JUDGE rows and MONITOR rows of its
              SOURCE benchmark. It enters the specialisation layer as ID_SPECIFIC only if its effect on the source benchmark's JUDGE rows,
              at equal compute against the specialisation system's own frontier, is positive in its own family (Holm 0.10 over the
              entries tested this way in the generation; one-sided sign-flip of picture clusters). Its effect on the MONITOR rows of the
              source benchmark (cards come from SHOW rows only) is recorded with it and in the shrinkage history: a number from rows
              that did not select it. The layer stays OFF in the general system; an admitted entry is never a general gain.
  update      accepted components enter the general layer, accepted policies the policy extras; audited entries that are not GENERAL move
              to the specialisation layer; promoted re-tests move back. A changed tip gets a new id and its cache is requested.
  report      RUN/report5_g<g>.txt: frontier points, verdict lines (per-benchmark effects, scored set, reference and tokens, both token
              conventions), both systems (general; general + specialisation), shrinkage history.
The rows file must be the clean 768-row learn.jsonl (sha256 b1263f834ed1d2db...); any other role or sha is refused (--no-rows-check for tests).
usage: gen5.py --run RUN --root ROOT --rows learn.jsonl [--gens 1] [--offline] [--p0-slots 6] [--g-cells 2] [--retest] [--audit-groups FILE]
       gen5.py init --run RUN --tip tips/t007c.json [--sources e6.jsonl.gz] [--verdicts stage1/verdicts.json]
"""
import argparse
import collections
import gzip
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(os.path.dirname(HERE), "v5"), os.path.join(os.path.dirname(HERE), "harness")]
import gen as G4  # noqa: E402
import ecomp as ECOMP  # noqa: E402
import proposer as PR  # noqa: E402
import guess5 as GU  # noqa: E402
import judge_v5 as JV  # noqa: E402
import policy5 as P5  # noqa: E402
import reads5 as R5  # noqa: E402

STEPS5 = ["signals", "candidates", "prescreen", "judge", "update", "report"]
P_FULL = 8              # cached reads of the tip per learn row (the frontier's votes of 5 need at least 5)
P_MIN = 4               # a generation refuses to judge with fewer tip reads than this on any row
S_SCREEN, S_JUDGE, S_AUDIT = 2, 4, 2
SCREEN_P, K_MAX = 0.25, 6
WINDOWS = (600, 1500)   # tail windows of the guess flag (the loop keeps 1,500 characters of reasoning per read)
DEFAULT_POLICY = "n2:disagree->tip"   # 2 reads + a third on disagreement (the vote of 3's answers at x2.4 tokens)
# the one sentence about the judge that a proposer card carries (coordinator 11-11: feedback about the acceptance rule, no task hint)
RULE_NOTE = ("How entries are judged: an entry is kept in the general system only if it helps on benchmarks other than the one its "
             "failures came from; otherwise it can only enter that benchmark's own layer.")
LEARN_ROLES = ("show", "monitor", "judge")
log, jdump, sha16 = G4.log, G4.jdump, G4.sha16


class Refuse(Exception):
    pass


def args5(argv=None):
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--offline", action="store_true", help="no request is written: reads that are not in q/out count as missing (dry runs on recorded reads)")
    p.add_argument("--p0-slots", type=int, default=6, help="proposer cards per generation (0 = no proposer)")
    p.add_argument("--g-cells", type=int, default=2, help="cells of stage G per generation")
    p.add_argument("--cells", default=None, help="cells.json (default RUN/signal/<tip>/cells.json)")
    p.add_argument("--retest", action="store_true", help="re-test the specialisation entries this generation (off-source rule on the general tip)")
    p.add_argument("--audit-groups", default=None, help="json {name: [general-layer entry cids]}: entries audited together (default: one by one)")
    p.add_argument("--no-audit", action="store_true")
    p.add_argument("--no-admit", action="store_true", help="no admission test of failed proposer entries to the specialisation layer")
    p.add_argument("--wait", type=float, default=6 * 3600.0, help="seconds a step waits for its reads")
    p.add_argument("--no-rows-check", action="store_true", help="tests only: skip the sha256 check of the rows file")
    p.add_argument("--k-max", type=int, default=K_MAX)
    p.add_argument("--poll", type=float, default=5.0, help="seconds between polls of the result files")
    a5, rest = p.parse_known_args(argv)
    a = G4.args(rest + ["--no-versions", "--no-signal"])
    for k, v in vars(a5).items():
        setattr(a, k, v)
    return a


def init_state(tip_v4, sources=None, verdicts=None, note=""):
    """A v5 state from a v4 tip file. sources: {cid: [benchmark]} (ledger card_source). verdicts: {cid: {layer, ...}} of an audit that
    was done before (stage 1): entries whose layer is not GENERAL start in the specialisation layer; without verdicts every conditional
    text entry starts in the general layer with audit = "pending" and the first generation audits it."""
    ents, special = [], []
    for e in tip_v4["entries"]:
        cid = e.get("from_cand") or e.get("cid")
        x = dict(condition=e["condition"], cond="always" if e["condition"] == "always" else sha16(e["condition"]), procedure=e.get("procedure"),
                 settings=e.get("settings") or {}, ops=e.get("ops") or [], cid=cid, author=e.get("author"), source=list((sources or {}).get(cid) or []),
                 pos=len(ents) + len(special))
        if e.get("edit"):
            x["edit"] = e["edit"]
        if x["cond"] == "always":
            ents.append(x)
            continue
        v = (verdicts or {}).get(cid)
        if v is None:
            ents.append(dict(x, audit="pending"))
        elif v.get("layer") == "GENERAL":
            ents.append(dict(x, audit="GENERAL", verdict=v))
        else:
            special.append(dict(x, verdict=v))
    return dict(created=time.time(), v=5, gen=0, gens={}, tips=[], shrinkage=[], note=note or f"v5 from v4 tip {tip_v4.get('id')}",
                tip=dict(id="v5t000", born=0, parent=tip_v4.get("id"), entries=ents, special=special,
                         policy=dict(default=DEFAULT_POLICY, extras=[], frontier=None, guess=None)))


class Run5(G4.Run):
    def __init__(self, a):
        if not a.no_rows_check:
            sha = hashlib.sha256(open(a.rows, "rb").read()).hexdigest()
            if sha != R5.LEARN_SHA:
                raise Refuse(f"{a.rows}: sha256 {sha[:16]}... is not the clean 768-row learn.jsonl ({R5.LEARN_SHA[:16]}...)")
        super().__init__(a)
        bad = sorted({r.get("v4_role") for r in self.rows.values()} - set(LEARN_ROLES))
        if bad:
            raise Refuse(f"{a.rows} holds rows of roles {bad}: the v5 loop runs on show / monitor / judge rows only")
        if self.state.get("v") != 5:
            if self.state.get("gens"):
                raise Refuse(f"{self.rd}/state.json is a v4 state: start v5 in a new run directory (gen5.py init)")
            self.state = init_state(dict(id="t000", entries=[]), note="v5 from stock")
        self.rows5 = {i: dict(id=i, bench=r["bench"], role=r["v4_role"], pic=r.get("pic") or i, kind=r.get("kind"), category=r.get("category"))
                      for i, r in self.rows.items()}
        self.benches = sorted({r["bench"] for r in self.rows.values()})
        self.SIGR = self.role["show"] + self.role["monitor"]
        self.JR = self.judge_rows
        self.deadline, self._jg = None, None
        self.g = self.state.get("gen", 0)
        w = self.io.wait
        self.io.wait = lambda keys, deadline=None, every=None, label="", tick=300: w(keys, deadline, a.poll if every is None else every, label, tick)

    # ------------------------------------------------------------------ state, reads
    def save(self):
        jdump(os.path.join(self.rd, "state.json"), self.state)

    def tip(self):
        return self.state["tip"]

    def read(self, name, reqs, label=None):
        """submit + wait (or, offline, only count what is missing). -> number of reads still missing."""
        keys = list(dict.fromkeys(q["key"] for q in reqs))
        if not keys:
            return 0
        if self.a.offline:
            self.io.poll()
            miss = [k for k in keys if not self.io.have(k)]
            if miss:
                log(f"  {name}: OFFLINE, {len(miss)} of {len(keys)} reads are not recorded")
            return len(miss)
        n = self.io.submit(name, reqs)
        done, miss = self.io.wait(keys, time.time() + self.a.wait, label=label or name)
        log(f"  {name}: {len(keys)} reads, {n} new, {len(miss)} unanswered")
        return len(miss)

    def reqs(self, entries, rows, samples, prio=1):
        return [self.read_req(entries, r, s, prio)[0] for r in rows for s in samples]

    def arm(self, entries, rows, upto=P_FULL, tail=False):
        """the cached reads of a view as a judge arm: {row: [dict(ok, tokens, prompt_tokens, pred, cls, finish, s[, tail, guess])]}."""
        out = {}
        for r in rows:
            _, _, sg = self.view(entries, r)
            row, xs = self.rows[r], []
            for s in range(upto):
                x = self.io.get(self.read_key(sg, r, s))
                if x is None or x.get("op_skipped") or (x.get("ok") is None and x.get("finish") is None):
                    continue
                ok, p = ECOMP.grade(x, row)
                c = ECOMP.answer_class(p, row)
                d = dict(ok=bool(ok), tokens=float(x.get("tokens") or 0.0), prompt_tokens=float(x.get("prompt_tokens") or 0.0), pred=p,
                         cls=P5.NONE if c == ECOMP.NONE_CLASS else c, finish=x.get("finish"), s=s)
                if tail:
                    d["tail"] = x.get("reasoning") or ""
                xs.append(d)
            if xs:
                out[r] = xs
        return out

    def fired_rows(self, e, rows):
        return [r for r in rows if self.fires(e, r)]

    # ------------------------------------------------------------------ judges (rebuilt from the cache on resume)
    def judges(self):
        if self._jg is not None:
            return self._jg
        tip = self.tip()["entries"]
        tS, tJ = self.arm(tip, self.SIGR, tail=True), self.arm(tip, self.JR, tail=True)
        nmin = min([len(tS.get(r, ())) for r in self.SIGR] + [len(tJ.get(r, ())) for r in self.JR])
        if nmin < P_MIN:
            raise Refuse(f"the tip {self.tip()['id']} has {nmin} cached reads on some learn row (< {P_MIN}): fill its cache first")
        base = [dict(n0=1, trigger="none", extra="tip"), dict(n0=2, trigger="disagree", extra="tip", final="plur"), dict(n0=3, trigger="none", extra="tip")]
        F0 = P5.Frontier(tS, self.SIGR, self.rows5, policies=base)
        V = F0.vertices
        price = (V[1][1] - V[0][1]) / ((V[1][0] - V[0][0]) / V[0][0]) if len(V) > 1 else 0.0
        det = GU.derive([dict(row=r, ok=x["ok"], tokens=x["tokens"], tail=x["tail"]) for r in self.SIGR for x in tS[r]], self.rows5, price, windows=WINDOWS)
        for armx, cross in ((tS, True), (tJ, False)):          # signal rows: cross-fitted flags (the patterns picked WITHOUT the row's benchmark),
            for r, xs in armx.items():                         # so a guess policy's signal-row value is never in-sample for the detector
                pats = (det.get("picked") or {}).get(self.rows[r]["bench"], det["patterns"]) if cross else det["patterns"]
                for x in xs:
                    x["guess"] = {w: GU.flag(dict(seed=det["seed"], window=w, patterns=pats), x.get("tail")) for w in WINDOWS}
        pols = P5.tip_policies(max_vote=min(5, nmin), with_guess=True, windows=WINDOWS)
        Js = JV.Judge(self.rows5, tS, benches=self.benches, policies=pols, roles=("show", "monitor"))
        Jg = JV.Judge(self.rows5, tJ, benches=self.benches, policies=pols, hull_from={p[2] for p in Js.F.vertices},
                      hull_from_total={p[2] for p in Js.Ft.vertices})
        self._jg = dict(Js=Js, Jg=Jg, tS=tS, tJ=tJ, det=det, price=price, nmin=nmin)
        return self._jg

    def special_view(self, tip=None):
        """the entry list of the SPECIALISATION system: general + layer, in the entries' original order."""
        t = tip or self.tip()
        return sorted(t["entries"] + t["special"], key=lambda e: e.get("pos", 10 ** 6))

    def judges_special(self):
        """(signal judge, JUDGE judge) of the specialisation system (general + layer), each with that system's own frontier; the general
        judges when the layer is empty; None when the system's cache is not filled (fewer than P_MIN reads on some learn row)."""
        if not self.tip()["special"]:
            j = self.judges()
            return j["Js"], j["Jg"]
        if getattr(self, "_jsp", None) is not None:
            return self._jsp
        V = self.special_view()
        self.load_routes(V, self.SIGR + self.JR)
        tS, tJ = self.arm(V, self.SIGR), self.arm(V, self.JR)
        nmin = min([len(tS.get(r, ())) for r in self.SIGR] + [len(tJ.get(r, ())) for r in self.JR])
        if nmin < P_MIN:
            return None
        pols = P5.tip_policies(max_vote=min(5, nmin), with_guess=False)
        Js = JV.Judge(self.rows5, tS, benches=self.benches, policies=pols, roles=("show", "monitor"))
        Jg = JV.Judge(self.rows5, tJ, benches=self.benches, policies=pols, hull_from={p[2] for p in Js.F.vertices}, hull_from_total={p[2] for p in Js.Ft.vertices})
        self._jsp = (Js, Jg)
        return self._jsp

    def frontier_record(self):
        j = self.judges()
        Js, Jg = j["Js"], j["Jg"]
        V = [p[2] for p in sorted(Js.F.vertices)]
        pts = []
        for n in sorted(Jg.F.table, key=lambda n: P5.macro_mean(Jg.F.table[n], self.rows5, "tokens")):
            ts, tj = Js.F.table[n], Jg.F.table[n]
            pts.append(dict(name=n, vertex=n in V, multiple=P5.macro_mean(tj, self.rows5, "tokens") / Jg.F.base_tokens,
                            multiple_total=P5.macro_mean(tj, self.rows5, "total") / Jg.Ft.base_tokens,
                            acc_judge=P5.macro_mean(tj, self.rows5, "ok"), acc_signal=P5.macro_mean(ts, self.rows5, "ok")))
        default = self.tip()["policy"].get("default") or DEFAULT_POLICY
        if default not in Jg.F.table:
            default = "vote1"
        return dict(vertices=V, points=pts, default=default, report=["vote1", default, V[-1]], base_tokens=Jg.F.base_tokens, base_total=Jg.Ft.base_tokens)

    # ------------------------------------------------------------------ steps
    def G5(self):
        return self.state["gens"][str(self.g)]

    def step_signals(self):
        G, tip = self.G5(), self.tip()["entries"]
        allr = self.SIGR + self.JR
        if self.a.offline:
            self.load_routes(tip, allr)
        else:
            self.ensure_routes(tip, allr, time.time() + self.a.wait, "tip")
        miss = self.read(f"g{self.g:03d}_parent", self.reqs(tip, allr, range(P_FULL), prio=0), "tip cache")
        if self.tip()["special"] and not self.a.no_admit:      # the specialisation system's own cache (the reference of admission tests)
            V = self.special_view()
            if self.a.offline:
                self.load_routes(V, allr)
            else:
                self.ensure_routes(V, allr, time.time() + self.a.wait, "special")
            G["special_reads_missing"] = self.read(f"g{self.g:03d}_special", self.reqs(V, allr, range(P_FULL), prio=2), "specialisation system cache")
        j = self.judges()
        sig = {}
        for b in self.benches:
            rs = [r for r in self.SIGR if self.rows[r]["bench"] == b]
            xs = [x for r in rs for x in j["tS"][r]]
            dis = [len({P5.cls_of(x) for x in j["tS"][r][:2]}) > 1 for r in rs if len(j["tS"][r]) >= 2]
            sig[b] = dict(rows=len(rs), acc=sum(x["ok"] for x in xs) / len(xs), tokens=sum(x["tokens"] for x in xs) / len(xs),
                          disagree=sum(dis) / max(len(dis), 1), guess={str(w): sum(bool(x["guess"][w]) for x in xs) / len(xs) for w in WINDOWS})
        det = j["det"]
        fr = self.frontier_record()
        self.tip()["policy"].update(frontier=fr, default=fr["default"], guess=dict(seed=det["seed"], patterns=det["patterns"], window=det["window"], windows=list(WINDOWS),
                                                                               lobo_value=det["lobo_value"], price=j["price"]))
        G["signals"] = dict(by_bench=sig, tip_reads_missing=miss, tip_reads_min=j["nmin"])
        log(f"  signals: tip {self.tip()['id']} reads per row >= {j['nmin']}; frontier vertices {fr['vertices']}; guess detector seed '{det['seed']}' + {det['patterns']}")

    def read_cells(self):
        p = self.a.cells or os.path.join(self.rd, "signal", self.tip()["id"], "cells.json")
        if not os.path.exists(p):
            return [], f"no cells.json ({p})"
        x = json.load(open(p))
        if isinstance(x, dict) and x.get("tip") not in (None, self.tip()["id"]):
            return [], f"cells.json is for tip {x.get('tip')}, the tip is {self.tip()['id']}"
        items = [it for it in (x.get("cells") if isinstance(x, dict) else x) or [] if isinstance(it, dict) and it.get("kind") == "weight" and it.get("edit")]
        return items, f"{len(items)} weight cells in {p}"

    def step_candidates(self):
        G, tip = self.G5(), self.tip()
        cands, k = [], 0

        def cid():
            nonlocal k
            k += 1
            return f"v{self.g:03d}c{k - 1:02d}"
        base = dict(condition="always", cond="always", procedure=None, settings={}, ops=[])
        # (i) cells of stage G
        def payload(c):
            return json.dumps([c["entry"].get("edit"), c["entry"].get("ops"), c["entry"].get("settings") or None], sort_keys=True)
        # a cell / operator that was already screened or decided on THIS tip is not read again (a new tip is a new question)
        tried = {payload(c) for gk, g in self.state["gens"].items() if int(gk) != self.g for c in g.get("cands", [])
                 if c.get("kind") in ("cell", "operator", "setting") and c.get("tip") == tip["id"] and (c.get("decided") or (c.get("screen") or {}).get("judged"))}
        items, why = self.read_cells()
        G["cells_info"] = why
        for it in [it for it in items if json.dumps([it["edit"], [], None], sort_keys=True) not in tried][: self.a.g_cells]:
            c = dict(cid=cid(), kind="cell", name=str(it.get("name") or "cell"), source=[], entry=dict(base, ops=[], edit=it["edit"], author="G"),
                     pred=it.get("pred"), fitted_on="signal rows")
            if it.get("null_edit"):
                c["null_entry"] = dict(base, edit=it["null_edit"], author="G")
            cands.append(c)
        # (ii) operators (knobs "operators": [{"name", "ops": [...]}])
        for o in self.knobs.get("operators") or []:
            if json.dumps([None, PR.canon_ops(o["ops"]), None], sort_keys=True) not in tried:
                cands.append(dict(cid=cid(), kind="operator", name=str(o.get("name") or "operator"), source=[], entry=dict(base, ops=PR.canon_ops(o["ops"]), author="O")))
        # (vii) settings arm (stage 3 A; knobs "settings_arm": [{"name", "settings": {knob: value}}]): the tip with one request-level knob
        # changed, source = [] (all benchmarks scored). Effort is not a knob. Its reads are its own view: frontier against frontier.
        cur = self.settings_of([e for e in tip["entries"] if e["cond"] == "always"])
        for o in self.knobs.get("settings_arm") or []:
            st_ = dict(o["settings"])
            if "effort" in " ".join(st_).lower() or "reasoning" in " ".join(st_).lower():
                raise Refuse(f"settings_arm {o}: the reasoning effort is never a knob")
            if all(cur.get(k_) == v for k_, v in st_.items()) or json.dumps([None, [], st_], sort_keys=True) in tried:
                continue
            cands.append(dict(cid=cid(), kind="setting", name=str(o.get("name") or json.dumps(st_, sort_keys=True)), source=[], entry=dict(base, settings=st_, author="B"),
                              **({"view_only": True} if o.get("view_only") else {})))
        # (iii) proposer text entries, source = the benchmark of the card's SHOW rows
        if self.a.p0_slots > 0 and not self.a.offline:
            self._cands_so_far = cands
            cands += self.propose(cid)
        # (iv) audit of general-layer text entries
        if not self.a.no_audit:
            cond = [e for e in tip["entries"] if e["cond"] != "always" and e.get("audit") != "GENERAL"]
            groups = json.load(open(self.a.audit_groups)) if self.a.audit_groups else {}
            used = {c_ for v in groups.values() for c_ in v}
            units = [(n, [e for e in cond if e["cid"] in set(v)]) for n, v in groups.items()] + [(e["cid"], [e]) for e in cond if e["cid"] not in used]
            for n, es in units:
                if es:
                    cands.append(dict(cid=cid(), kind="audit", name=f"audit {n}", source=sorted({b for e in es for b in e.get("source") or []}), members=[e["cid"] for e in es]))
        # (v) re-tests of specialisation entries
        for e in tip["special"]:                               # --retest, or the rows file holds a benchmark the entry's verdict did not see
            seen_b = (e.get("verdict") or {}).get("benches")
            if self.a.retest or (seen_b and set(self.benches) - set(seen_b)):
                cands.append(dict(cid=cid(), kind="retest", name=f"retest {e['cid']}", source=list(e.get("source") or []),
                                  entry={k_: v for k_, v in e.items() if k_ not in ("verdict",)}, member=e["cid"]))
        for c in cands:
            c.update(tip=tip["id"], gen=self.g, status="candidate")
        G["cands"] = cands
        log(f"  candidates: " + ", ".join(f"{n} {v}" for n, v in sorted(collections.Counter(c["kind"] for c in cands).items())) + f" | cells: {why}")

    def propose(self, cid):
        """P0 cards. Half of the slots: single-benchmark cards (slots by SHOW residual), source = that benchmark. Half: MIXED cards (SHOW
        failures of all benchmarks), source = [] (all six scored; p_gen's leave-one-benchmark-out guards against a one-benchmark recipe).
        What a card may propose: an entry = a condition (or "always") with a text procedure and / or settings and / or harness ops, as the
        proposer's menu offers them -> kind "entry", scored off its source benchmark (a row where its op cannot be applied counts as the
        tip); a settings change alone on every question (knob menu of knobs.json; effort refused) -> the settings arm (kind "setting",
        source [], frontier against frontier). Every dropped proposal is kept with its reason and its first 200 characters."""
        G = self.G5()
        tip = self.tip()["entries"]
        tv = self.tip_view()
        parent_show = {r: self.samples(tip, r) for r in self.role["show"]}
        n_mixed = self.a.p0_slots // 2
        alloc = self.alloc_sources(self.a.p0_slots - n_mixed, self.residual(parent_show))
        cards = [(b, PR.cards_source(self.rows, parent_show, b, i, seed=f"v5g{self.g}:{self.a.seed}", n_wrong=G4.N_WRONG, n_right=G4.N_RIGHT))
                 for b in sorted(alloc) for i in range(alloc[b])]
        cards += [(None, PR.cards(self.rows, parent_show, k_, seed=f"v5g{self.g}:mixed:{self.a.seed}", n_wrong=G4.N_WRONG, n_right=G4.N_RIGHT)) for k_ in range(n_mixed)]
        reqs, props = [], []
        for b, ids in cards:
            if not ids:
                continue
            system, user = PR.card_prompt(self.rows, parent_show, ids, tv, self.knobs)
            msgs = [{"role": "system", "content": system}, {"role": "user", "content": user + "\n\n" + RULE_NOTE}]
            key = f"chat|{self.run_name}|v5g{self.g:03d}|P0|{len(reqs)}|{sha16(json.dumps(msgs))[:8]}"
            reqs.append(dict(key=key, mode="chat", engine="E", priority=0, messages=msgs, temperature=G4.PROP_TEMP, max_tokens=G4.PROP_MAX))
            props.append(dict(key=key, card=ids, source=[b] if b else [], mixed=b is None))
        self.io.submit(f"g{self.g:03d}_chat", reqs)
        self.io.wait([q["key"] for q in reqs], time.time() + self.a.wait, label="proposer")
        seen = {PR._entry_key(e) for e in tv["entries"]} | {PR._entry_key(e) for e in self.tip()["special"]}
        have_set = {json.dumps(c["entry"]["settings"], sort_keys=True) for c in getattr(self, "_cands_so_far", []) if c["kind"] == "setting"}
        have_set |= {json.dumps(c["entry"]["settings"], sort_keys=True) for gk, g_ in self.state["gens"].items() if int(gk) != self.g for c in g_.get("cands", [])
                     if c.get("kind") == "setting" and c.get("tip") == self.tip()["id"] and (c.get("screen") or {}).get("judged")}   # read on this tip before
        base = dict(condition="always", cond="always", procedure=None, settings={}, ops=[])
        out, dropped = [], []
        for p in props:
            x = self.io.get(p["key"])
            cand, err, _ = self.check(x, tv, p["card"])
            proc = (cand or {}).get("procedure") if isinstance((cand or {}).get("procedure"), str) and (cand or {}).get("procedure").strip() else None
            sett = (cand or {}).get("settings") or {}
            ops = PR.canon_ops((cand or {}).get("ops"))
            kind = None
            if cand is not None and err is None:
                always = PR._is_always(cand["condition"])
                if sett and not proc and not ops and always:
                    kind = "setting"                             # a request-level settings change on every question -> the settings arm
                elif proc or ops or sett:
                    kind = "entry"                               # one entry = its condition (or always) + its procedure / settings / ops, as proposed
                else:
                    err = "empty proposal"
            if kind == "entry" and PR._entry_key(cand) in seen:
                kind, err = None, "duplicate of a tip or specialisation entry or of an earlier candidate"
            if kind == "setting" and json.dumps(sett, sort_keys=True) in have_set:
                kind, err = None, "duplicate of a settings-arm cell (this generation, or read on this tip before)"
            if kind is None:
                dropped.append(dict(why=str(err)[:160], source=p["source"], mixed=p["mixed"], head=((x or {}).get("content") or "")[-200:] if cand is None else
                                    json.dumps({k_: cand.get(k_) for k_ in ("condition", "procedure", "settings", "ops") if cand.get(k_)}, ensure_ascii=False)[:200]))
                continue
            c = cid()
            if kind == "entry":
                seen.add(PR._entry_key(cand))
                always = PR._is_always(cand["condition"])
                what = "+".join(w for w, v_ in (("text", proc), ("settings", sett), ("ops", ops)) if v_)
                out.append(dict(cid=c, kind="entry", name=f"entry {c} ({'always-on' if always else 'routed'} {what})", source=list(p["source"]), card=p["card"], mixed=p["mixed"],
                                why=cand.get("why", ""), entry=dict(condition="always" if always else cand["condition"], cond="always" if always else sha16(cand["condition"]),
                                                                    procedure=proc, settings=dict(sett), ops=ops, cid=c, author="P0", source=list(p["source"]))))
            else:
                have_set.add(json.dumps(sett, sort_keys=True))
                out.append(dict(cid=c, kind="setting", name="P0 " + json.dumps(sett, sort_keys=True), source=[], proposed_from=list(p["source"]), card=p["card"],
                                why=cand.get("why", ""), entry=dict(base, settings=dict(sett), author="P0")))
        G["dropped"] = dropped
        G["cards"] = dict(single=len([p for p in props if not p["mixed"]]), mixed=len([p for p in props if p["mixed"]]), alloc=alloc)
        log(f"  proposer: {len(props)} cards ({G['cards']['single']} single-benchmark, {G['cards']['mixed']} mixed) -> {sum(c['kind'] == 'entry' for c in out)} entries "
            f"({sum(c['kind'] == 'entry' and c['entry']['cond'] == 'always' for c in out)} always-on), {sum(c['kind'] == 'setting' for c in out)} settings; "
            f"dropped {len(dropped)}: {dict(collections.Counter(d['why'][:50] for d in dropped))}")
        return out

    def cand_view(self, c):
        """the entry list a candidate's own reads are made with."""
        tip = self.tip()["entries"]
        if c["kind"] == "audit":
            return [e for e in tip if e["cid"] not in set(c["members"])]
        return tip + [c["entry"]]

    def cand_rows(self, c, rows, screen):
        """rows a candidate is read on: all rows for always-on components; the routed rows for entries (a pre-screen never reads the
        source benchmark); an audit: the rows where a member fires."""
        if c["kind"] in ("cell", "operator", "setting"):
            return list(rows)
        if c["kind"] == "audit":
            mem = [e for e in self.tip()["entries"] if e["cid"] in set(c["members"])]
            return [r for r in rows if any(self.fires(e, r) for e in mem)]
        rr = self.fired_rows(c["entry"], rows)
        card = set(c.get("card") or ()) if screen else set()   # a pre-screen reads neither the source benchmark nor the card's own SHOW rows
        return [r for r in rr if not (screen and self.rows[r]["bench"] in set(c.get("source") or [])) and r not in card]

    def to_judge(self, Jx, c, rows, upto):
        """the candidate as a judge_v5 candidate on the row set of Jx (None when reads are missing)."""
        rr = self.cand_rows(c, rows, screen=Jx.roles != ("judge",))
        arm = self.arm(self.cand_view(c), rr, upto)
        miss = [r for r in rr if r not in arm]
        if miss:                                               # a row whose reads ended as error / op_skipped lines (context or image limit) cannot be
            why = {r: self.unreadable(self.cand_view(c), r, upto) for r in miss}      # read under this candidate: it is the tip there (difference 0)
            if any(v is None for v in why.values()):
                return None                                    # reads really missing (pending)
            c.setdefault("skipped_rows", {})["judge" if Jx.roles == ("judge",) else "signal"] = dict(
                n=len(miss), why=dict(collections.Counter(why.values()).most_common(3)), by_bench=dict(collections.Counter(self.rows[r]["bench"] for r in miss)))
            rr = [r for r in rr if r in arm]
        kw = {}
        if c.get("null_entry"):
            na = self.arm(self.tip()["entries"] + [c["null_entry"]], rows, upto)
            if all(r in na for r in rows):
                kw["null"] = Jx.system_of_arm(na)
        if c["kind"] == "audit":                               # the members as a candidate on top of the tip without them
            tipa = {r: Jx.tip[r] for r in rr}
            return Jx.cand_from_arm(c["name"], tipa, "entry", source=c["source"], routed=rr, base=Jx.system_of_arm(arm, routed=rr))
        kind = "entry" if c["kind"] == "retest" else c["kind"]
        return Jx.cand_from_arm(c["name"], arm, kind, source=c["source"], routed=None if (c["kind"] in ("cell", "operator") and not miss) else rr, **kw)

    def unreadable(self, entries, r, upto):
        """why a row has no read under a view although its requests were answered: the text of an error line or of an op_skipped line of
        one of its samples (None when a sample is simply not there yet)."""
        _, _, sg = self.view(entries, r)
        for s_ in range(upto):
            k = self.read_key(sg, r, s_)
            x = self.io.get(k)
            if x is not None and x.get("op_skipped"):
                return "op_skipped: " + str(x.get("op_skipped"))[:60]
            if x is None and k in self.io.err:
                return "error: " + str(self.io.err[k].get("error"))[:60]
        return None

    def budgets(self, Jx, name, arm):
        """a view's three report points (one read, the default budget, the last vertex) against the tip's frontier at equal tokens."""
        out = {}
        n = min(len(arm[r]) for r in Jx.J)
        for lab, pol in P5.budget_policies(name):
            if len(pol["init"]) + len(pol["extra_views"]) > n:
                continue
            out[lab] = self.brief(Jx.score(Jx.cand_from_views(f"{name} {lab}", pol, {name: arm}, kind="setting")))
        return out

    @staticmethod
    def brief(r):
        """what is kept in the state of a judge record."""
        if not r.get("judged"):
            return dict(judged=False, error=r.get("error"))
        e = r["eq"]
        out = dict(judged=True, acc=r["acc"], multiple=r["multiple"], multiple_total=(r.get("total") or {}).get("multiple"), raw=r["raw"]["d"], d=e["d"], se=e["se"],
                   p_macro=e["p_macro"], p_gen=e["p_gen"], carrier=e["carrier"], npos=e["npos"], k=e["k"], scored=r["scored"], ref=r.get("ref"),
                   per_bench={b: round(v["d"], 5) for b, v in e["per_bench"].items()}, d_total=(r.get("total") or {}).get("d"), pooled=r["pooled"]["d"],
                   pooled_p=r["pooled"]["p_macro"])
        if r.get("null"):
            n = r["null"]
            out["null"] = dict(d=n["d"], se=n["se"], audit_ok=n["audit_ok"], vs_null_d=n["vs_null"]["d"], vs_null_p_gen=n["vs_null"]["p_gen"])
        for k_ in ("accept", "reason", "layer", "K", "holm_threshold", "gate"):
            if k_ in r:
                out[k_] = r[k_]
        return out

    def step_prescreen(self):
        G, j = self.G5(), self.judges()
        Js = j["Js"]
        tip = self.tip()["entries"]
        comp = [c for c in G["cands"] if c["kind"] in ("cell", "operator", "entry", "retest")]
        sets = [c for c in G["cands"] if c["kind"] == "setting"]
        text = [c for c in comp if c["kind"] in ("entry", "retest")]
        if text:
            if self.a.offline:
                self.load_routes([c["entry"] for c in text], self.SIGR + self.JR)
            else:
                self.ensure_routes([c["entry"] for c in text], self.SIGR + self.JR, time.time() + self.a.wait, "cands")
        reqs = []
        for c in comp:
            reqs += self.reqs(self.cand_view(c), self.cand_rows(c, self.SIGR, True), range(S_SCREEN))
            if c.get("null_entry"):
                reqs += self.reqs(tip + [c["null_entry"]], self.SIGR, range(S_SCREEN))
        for c in sets:                                         # a setting is its own view; first rung: S_SCREEN reads per signal row
            reqs += self.reqs(self.cand_view(c), self.SIGR, range(S_SCREEN))
        self.read(f"g{self.g:03d}_screen", reqs, "pre-screen")
        passed = []
        # settings, futility rung: with only the first-rung reads in the cache, one read of the arm against the tip's frontier; an arm that is
        # not above the tip there stops (2,592 reads per arm saved); the others get their full cache (P_FULL reads) for the three budgets
        more = []
        for c in sets:
            arm = self.arm(self.cand_view(c), self.SIGR, P_FULL)
            n_ = min([len(arm.get(r, ())) for r in self.SIGR])
            if 1 <= n_ < 3 and not c.get("view_only"):
                x1 = self.brief(Js.score(Js.cand_from_views(f"{c['name']} x1.00", P5.views_policy([c["name"]]), {c["name"]: arm}, kind="setting")))
                if not x1["d"] > 0:
                    c.update(status=f"screened out at the first rung ({n_} reads per signal row; one read of the arm against the tip)", screen=dict(x1, passed=False, rung=1))
                    continue
            if n_ < P_FULL:
                more += self.reqs(self.cand_view(c), self.SIGR, range(P_FULL))
        if more:
            self.read(f"g{self.g:03d}_screen2", more, "pre-screen, full cache of the settings arms that passed the first rung")
        # settings: the setting's own frontier points against the tip's frontier at equal tokens; the default budget screens
        views = {}
        for c in sets:
            if (c.get("screen") or {}).get("rung") == 1:
                continue
            arm = self.arm(self.cand_view(c), self.SIGR, P_FULL)
            if any(len(arm.get(r, ())) < 3 for r in self.SIGR):
                c.update(status="pre-screen reads missing", screen=dict(judged=False))
                continue
            views[c["name"]] = arm
            c["budgets_signal"] = self.budgets(Js, c["name"], arm)
            sc = dict(c["budgets_signal"]["default"])
            ok = bool(sc["judged"] and sc["d"] > 0 and sc["p_macro"] <= SCREEN_P) and not c.get("view_only")
            c["screen"] = dict(sc, passed=ok)
            c["status"] = ("a cached view for view-diverse votes only (not a settings candidate)" if c.get("view_only") else
                           "passed the pre-screen (default budget)" if ok else "screened out on signal rows (default budget)")
            if ok:
                passed.append((sc["p_macro"], c))
        # view-diverse votes (stage 3 B): reads through two different views; offline; the best policy per pair of views
        vn = ["tip"] + sorted(views)
        for i_, va in enumerate(vn):
            for vb in vn[i_ + 1:]:
                best = None
                for pol in P5.grammar_views([va, vb]):
                    r = Js.score(Js.cand_from_views(P5.views_name(pol), pol, views))
                    if r.get("judged") and r["eq"]["d"] > 0 and r["eq"]["p_macro"] <= SCREEN_P and (best is None or r["eq"]["p_macro"] < best[0]["eq"]["p_macro"]):
                        best = (r, pol)
                if best is not None:
                    pc = dict(cid=f"v{self.g:03d}w{len([c for c in G['cands'] if c['kind'] == 'viewpolicy']):02d}", kind="viewpolicy", name=P5.views_name(best[1]),
                              source=[], policy=best[1], views=sorted({v for v in best[1]["init"] + best[1]["extra_views"] if v != "tip"}), tip=self.tip()["id"], gen=self.g,
                              status="passed the pre-screen", screen=dict(self.brief(best[0]), passed=True))
                    G["cands"].append(pc)
                    passed.append((best[0]["eq"]["p_macro"], pc))
        for c in comp:
            if c["kind"] in ("entry", "retest") and not self.cand_rows(c, self.SIGR, True):
                # its condition matches no signal row outside its source benchmark: there is no off-source effect to measure (not "+0.00").
                # It cannot be a general gain; it goes straight to the admission test on its source benchmark.
                c.update(status="no off-source rows routed (its condition matches only its source benchmark)", screen=dict(judged=False, no_off_source=True, passed=False))
                continue
            jc = self.to_judge(Js, c, self.SIGR, P_FULL)
            if jc is None:
                c.update(status="pre-screen reads missing", screen=dict(judged=False))
                continue
            r = Js.score(jc)
            c["screen"] = self.brief(r)
            ok = bool(r.get("judged") and r["eq"]["d"] > 0 and r["eq"]["p_macro"] <= SCREEN_P)
            c["screen"]["passed"] = ok
            c["status"] = "passed the pre-screen" if ok else "screened out on signal rows"
            if ok:
                passed.append((r["eq"]["p_macro"], c))
        # read policies with an extra component: the grammar, offline; the best policy per component
        for c in [c for c in comp if c["kind"] in ("cell", "operator") and c.get("screen", {}).get("judged")]:
            ex = self.arm(self.cand_view(c), self.SIGR, P_FULL)
            best = None
            for pol in P5.grammar((c["cid"],), windows=WINDOWS):
                if pol["trigger"] == "none":
                    continue
                r = Js.score(Js.cand_from_policy(pol, extra_arm=ex))
                if r.get("judged") and r["eq"]["d"] > 0 and r["eq"]["p_macro"] <= SCREEN_P and (best is None or r["eq"]["p_macro"] < best[0]["eq"]["p_macro"]):
                    best = (r, pol)
            if best is not None:
                pc = dict(cid=f"{c['cid']}p", kind="policy", name=P5.name_of(best[1]), source=[], policy=best[1], extra=c["cid"], tip=c["tip"], gen=self.g,
                          status="passed the pre-screen", screen=dict(self.brief(best[0]), passed=True))
                G["cands"].append(pc)
                passed.append((best[0]["eq"]["p_macro"], pc))
        passed.sort(key=lambda x: x[0])
        for i, (_, c) in enumerate(passed):
            c["sent"] = i < self.a.k_max
            if not c["sent"]:
                c["status"] = f"passed the pre-screen, not sent (more than {self.a.k_max} candidates)"
        log(f"  pre-screen: {len(comp)} components read on signal rows; passed {[c['name'] for _, c in passed]}; sent {sum(bool(c.get('sent')) for _, c in passed)}")

    def step_judge(self):
        G, j = self.G5(), self.judges()
        Jg, tip = j["Jg"], self.tip()["entries"]
        by = {c["cid"]: c for c in G["cands"]}
        sent = [c for c in G["cands"] if c.get("sent")]
        audits = [c for c in G["cands"] if c["kind"] == "audit"]
        reqs = []
        byname = {c["name"]: c for c in G["cands"] if c["kind"] == "setting"}
        for c in sent:
            if c["kind"] in ("setting", "viewpolicy"):         # the views' full caches on JUDGE rows (only for what passed the pre-screen)
                for x in ([c] if c["kind"] == "setting" else [byname[v] for v in c["views"]]):
                    reqs += self.reqs(self.cand_view(x), self.JR, range(P_FULL))
                continue
            x = by[c["extra"]] if c["kind"] == "policy" else c
            reqs += self.reqs(self.cand_view(x), self.cand_rows(x, self.JR, False), range(S_JUDGE))
        for c in audits:
            reqs += self.reqs(self.cand_view(c), self.cand_rows(c, self.JR, False), range(S_AUDIT))
        self.read(f"g{self.g:03d}_judge", reqs, "JUDGE reads")

        def build(with_null=True):
            out, cs = [], []
            for c in sent:
                if c["kind"] in ("setting", "viewpolicy"):
                    va = {x["name"]: self.arm(self.cand_view(x), self.JR, P_FULL) for x in ([c] if c["kind"] == "setting" else [byname[v] for v in c["views"]])}
                    if any(len(a_.get(r, ())) < 3 for a_ in va.values() for r in self.JR):
                        jc = None
                    elif c["kind"] == "setting":                # accepted on the default budget; the three points are reported
                        c["budgets_judge"] = self.budgets(Jg, c["name"], va[c["name"]])
                        jc = Jg.cand_from_views(c["name"], P5.budget_policies(c["name"])[1][1], va, kind="setting")
                    else:
                        jc = Jg.cand_from_views(c["name"], c["policy"], va)
                elif c["kind"] == "policy":
                    ex = self.arm(self.cand_view(by[c["extra"]]), self.JR, P_FULL)
                    jc = Jg.cand_from_policy(c["policy"], extra_arm=ex, name=c["name"]) if all(r in ex for r in self.JR) else None
                else:
                    jc = self.to_judge(Jg, c, self.JR, P_FULL)
                    if jc is not None and not with_null:
                        jc.pop("null", None)
                if jc is None:
                    c.update(status="JUDGE reads missing", decided=False)
                    continue
                out.append(jc)
                cs.append(c)
            return out, cs
        jcs, cs = build()
        recs = Jg.decide(jcs, family=f"generation {self.g}") if jcs else []
        # a weight cell that would be accepted: its random-edit null on JUDGE rows, then the decision again (the null audit)
        need = [c for c, r in zip(cs, recs) if r.get("accept") and c["kind"] == "cell" and c.get("null_entry") and not r.get("null")]
        if need:
            nr = [q for c in need for q in self.reqs(tip + [c["null_entry"]], self.JR, range(S_JUDGE))]
            self.read(f"g{self.g:03d}_null", nr, "random-edit null on JUDGE rows")
            jcs, cs = build()
            recs = Jg.decide(jcs, family=f"generation {self.g}")
        for c, r in zip(cs, recs):
            if r.get("accept") and c["kind"] == "cell" and not r.get("null"):
                r["accept"] = False
                r["reason"] += " | NOT accepted: a weight cell needs its random-edit null read on the JUDGE rows" + ("" if c.get("null_entry") else " (cells.json gives no null_edit)")
            c.update(judge=self.brief(r), accept=bool(r["accept"]), decided=True, status=("ACCEPTED: " if r["accept"] else "rejected: ") + str(r["reason"]),
                     line=JV.line(r, self.benches))
        # audit family
        ja, ca = [], []
        for c in audits:
            jc = self.to_judge(Jg, c, self.JR, P_FULL)
            if jc is None:
                c.update(status="audit reads missing", decided=False)
                continue
            ja.append(jc)
            ca.append(c)
        for c, r in zip(ca, Jg.audit_entries(ja) if ja else []):
            c.update(judge=self.brief(r), layer=r["layer"], accept=r["layer"] == "GENERAL", decided=True, status=f"{r['layer']}: {r['reason']}", line=JV.line(r, self.benches))
        # shrinkage history: signal-row effect against JUDGE-row effect of everything that was judged on both
        for c in cs:
            s_, j_ = c.get("screen") or {}, c.get("judge") or {}
            if s_.get("judged") and j_.get("judged"):
                row = dict(gen=self.g, tip=self.tip()["id"], cid=c["cid"], kind=c["kind"], name=c["name"], fitted_on=c.get("fitted_on") or by.get(c.get("extra"), {}).get("fitted_on"),
                           signal_d=s_["d"], signal_se=s_["se"], judge_d=j_["d"], judge_se=j_["se"], signal_raw=s_["raw"], judge_raw=j_["raw"],
                           kept=(j_["d"] / s_["d"]) if s_["d"] > 0 else None, multiple=j_["multiple"], accepted=bool(c.get("accept")), t=time.time())
                self.state["shrinkage"].append(row)
                with open(os.path.join(self.rd, "shrinkage.jsonl"), "a") as fh:
                    fh.write(json.dumps(row) + "\n")
        self.admission()
        log(f"  judge: family K {len(recs)}: accepted {[c['name'] for c in cs if c.get('accept')]}; audit {[(c['name'], c.get('layer')) for c in ca]}"
            f"; admitted to the specialisation layer {[c['cid'] for c in G['cands'] if (c.get('admit') or {}).get('admitted')]}")

    def admission(self):
        """Admission of failed proposer entries to the specialisation layer (see the module text). Family = the entries tested here."""
        G = self.G5()
        # failed off-source = screened out on signal rows (read and not passed) or decided and rejected by the judge; an entry that passed
        # the pre-screen but was not sent, or whose reads are missing, has not failed and is not tested here
        adm = [c for c in G["cands"] if c["kind"] == "entry" and c.get("source") and not c.get("accept")
               and (c.get("decided") or (c.get("screen") or {}).get("no_off_source") or ((c.get("screen") or {}).get("judged") and not c["screen"].get("passed")))]
        if self.a.no_admit or not adm:
            return
        jj = self.judges_special()
        if jj is None:
            for c in adm:
                c["admit"] = dict(judged=False, admitted=False, why="the specialisation system's cache is not filled")
            log("  admission: the specialisation system's cache is not filled; no entry is tested this generation")
            return
        Jss, Jsg = jj
        V = self.special_view()
        mon = set(self.role["monitor"])
        reqs, rowsets = [], {}
        for c in adm:
            src = set(c["source"])
            fj = [r for r in self.fired_rows(c["entry"], self.JR) if self.rows[r]["bench"] in src]
            fm = [r for r in self.fired_rows(c["entry"], self.SIGR) if self.rows[r]["bench"] in src and r in mon]
            rowsets[c["cid"]] = (fj, fm)
            reqs += self.reqs(V + [c["entry"]], fj + fm, range(S_JUDGE))
        self.read(f"g{self.g:03d}_admit", reqs, "admission reads (source benchmark)")
        tested = []
        for c in adm:
            fj, fm = rowsets[c["cid"]]
            aj, am = self.arm(V + [c["entry"]], fj, P_FULL), self.arm(V + [c["entry"]], fm, P_FULL)
            if not fj or any(r not in aj for r in fj):
                c["admit"] = dict(judged=False, admitted=False, why="no routed JUDGE row in the source benchmark" if not fj else "admission reads missing")
                continue
            e = Jsg.source_effect(Jsg.cand_from_arm(c["name"], aj, "entry", routed=fj), c["source"])
            m = (Jss.source_effect(Jss.cand_from_arm(c["name"], am, "entry", routed=fm), c["source"], rows=mon)
                 if fm and all(r in am for r in fm) else dict(judged=False, error="no routed MONITOR row in the source benchmark"))
            c["admit"] = dict(judged=bool(e.get("judged")), admitted=False, source=c["source"], judge=e, monitor=m, routed_judge=len(fj), routed_monitor=len(fm))
            if e.get("judged"):
                tested.append(c)
        acc = JV.ST.holm([c["admit"]["judge"]["p"] for c in tested], JV.ALPHA) if tested else []
        order = sorted(range(len(tested)), key=lambda i: tested[i]["admit"]["judge"]["p"])
        for rank, i in enumerate(order):
            tested[i]["admit"]["holm_threshold"] = JV.ALPHA / (len(tested) - rank)
        for c, a_ in zip(tested, acc):
            e, m = c["admit"]["judge"], c["admit"]["monitor"]
            ok = bool(a_ and e["d"] > 0)
            c["admit"].update(admitted=ok, K=len(tested))
            c["status"] += (f" | admission on the source benchmark {c['source']}: {100 * e['d']:+.2f} +- {100 * e['se']:.2f} pp on its JUDGE rows (p {e['p']:.4f}, Holm {c['admit']['holm_threshold']:.4f}, K {len(tested)})"
                            + (f"; MONITOR rows {100 * m['d']:+.2f} +- {100 * m['se']:.2f}" if m.get("judged") else "; MONITOR rows: none routed")
                            + (" -> ID_SPECIFIC, specialisation layer (never a general gain)" if ok else " -> not admitted"))
            s_ = c.get("screen") or {}
            row = dict(gen=self.g, tip=self.tip()["id"], cid=c["cid"], kind="admission", name=c["name"], source=c["source"],
                       signal_d=s_.get("d"), signal_se=s_.get("se"), judge_d=(c.get("judge") or {}).get("d"), judge_se=(c.get("judge") or {}).get("se"),
                       source_judge_d=e["d"], source_judge_se=e["se"], source_judge_p=e["p"], source_monitor_d=m.get("d"), source_monitor_se=m.get("se"),
                       kept=(m["d"] / e["d"]) if (m.get("judged") and e["d"] > 0) else None, admitted=ok, accepted=False, t=time.time())
            self.state["shrinkage"].append(row)
            with open(os.path.join(self.rd, "shrinkage.jsonl"), "a") as fh:
                fh.write(json.dumps(row) + "\n")
        G["admission"] = dict(reference="the specialisation system (general + layer)" if self.tip()["special"] else "the general system (the layer is empty)",
                              layer_entries=len(self.tip()["special"]), tested=len(tested), admitted=[c["cid"] for c in tested if c["admit"]["admitted"]],
                              reference_one_read=sum(Jsg.tip_v.values()) / len(Jsg.tip_v))
        log(f"  admission: {len(tested)} failed entries tested on their source benchmark against {G['admission']['reference']}; admitted {G['admission']['admitted']}")

    def step_update(self):
        G, tip = self.G5(), self.tip()
        ents, special, pol = list(tip["entries"]), list(tip["special"]), json.loads(json.dumps(tip["policy"]))
        by = {c["cid"]: c for c in G["cands"]}
        change = []
        for c in G["cands"]:
            if (c.get("admit") or {}).get("admitted"):          # a failed proposer entry that passed the admission test on its source benchmark
                e, m = c["admit"]["judge"], c["admit"]["monitor"]
                off = c.get("judge") if (c.get("judge") or {}).get("judged") else (c.get("screen") or {})
                special.append(dict(c["entry"], cid=c["cid"], source=c["source"], pos=10 ** 5 + self.g * 100 + len(special),
                                    verdict=dict(layer="ID_SPECIFIC", gen=self.g, cid=c["cid"], benches=list(self.benches), basis="admission test on the JUDGE rows of the source benchmark",
                                                 d=off.get("d"), p_gen=off.get("p_gen"), off_source_rows="JUDGE" if off is c.get("judge") else "signal (screened out)",
                                                 source_d=e["d"], source_se=e["se"], source_p=e["p"], K=c["admit"].get("K"),
                                                 monitor_d=m.get("d"), monitor_se=m.get("se"))))
                change.append(f"+specialisation {c['cid']} (ID_SPECIFIC on {','.join(c['source'])})")
                continue
            if not c.get("decided"):
                continue
            v = dict(gen=self.g, cid=c["cid"], benches=list(self.benches), **{k: c["judge"].get(k) for k in ("d", "se", "p_gen", "pooled", "pooled_p", "scored", "multiple")})
            if c["kind"] == "viewpolicy" and c["accept"]:
                bn = {x["name"]: x["entry"] for x in G["cands"] if x["kind"] == "setting"}
                pol["extras"].append(dict(cid=c["cid"], policy=c["policy"], views={v_: bn[v_] for v_ in c["views"]}, name=c["name"], verdict=dict(v, layer="GENERAL")))
                change.append(f"+view policy {c['name']}")
            elif c["kind"] in ("cell", "operator", "entry", "setting") and c["accept"]:
                ents.append(dict(c["entry"], cid=c["cid"], source=c["source"], accepted=self.g, audit="GENERAL", verdict=dict(v, layer="GENERAL")))
                change.append(f"+{c['kind']} {c['name']}")
            elif c["kind"] == "policy" and c["accept"]:
                pol["extras"].append(dict(cid=c["cid"], policy=c["policy"], extra=by[c["extra"]]["entry"], name=c["name"], verdict=dict(v, layer="GENERAL")))
                change.append(f"+policy {c['name']}")
            elif c["kind"] == "retest" and c["accept"]:
                e = next(e for e in special if e["cid"] == c["member"])
                special.remove(e)
                ents.append(dict({k: x for k, x in e.items() if k != "verdict"}, audit="GENERAL", verdict=dict(v, layer="GENERAL")))
                change.append(f"promoted {c['member']}")
            elif c["kind"] == "retest":
                next(e for e in special if e["cid"] == c["member"]).setdefault("retests", []).append(dict(v, accept=False))
            elif c["kind"] == "audit":
                for e in [e for e in ents if e["cid"] in set(c["members"])]:
                    if c["layer"] == "GENERAL":
                        e.update(audit="GENERAL", verdict=dict(v, layer="GENERAL"))
                    else:
                        ents.remove(e)
                        special.append(dict({k: x for k, x in e.items() if k != "audit"}, verdict=dict(v, layer=c["layer"], audit=c["name"])))
                        change.append(f"{e['cid']} -> specialisation ({c['layer']})")
        G["tip_change"] = change
        if not change:
            log(f"  tip {tip['id']} unchanged")
            return
        self.state["tips"].append(dict(id=tip["id"], born=tip.get("born"), entries=len(tip["entries"]), special=len(tip["special"]), replaced=self.g))
        special.sort(key=lambda e: e.get("pos", 10 ** 6))      # the layer keeps the entries' original order (the prompt of the specialisation system)
        new = dict(id=f"v5t{self.g:03d}", born=self.g, parent=tip["id"], entries=ents, special=special, policy=dict(pol, frontier=None))
        self.state["tip"] = new
        self._jg, self._jsp = None, None
        os.makedirs(os.path.join(self.rd, "tips"), exist_ok=True)
        jdump(os.path.join(self.rd, "tips", new["id"] + ".json"), new)
        allr = self.SIGR + self.JR
        if not self.a.offline:                                 # caches of the new tip: requested now, read while the report is written
            self.ensure_routes(ents, allr, time.time() + self.a.wait, "newtip")
            n = self.io.submit(f"g{self.g:03d}_newtip", self.reqs(ents, allr, range(P_FULL), prio=5))
            if special and not self.a.no_admit:                # the cache of the new specialisation system (reference of the next admissions)
                V = self.special_view(new)
                self.ensure_routes(V, allr, time.time() + self.a.wait, "newspecial")
                G["new_special_cache_requested"] = self.io.submit(f"g{self.g:03d}_newspecial", self.reqs(V, allr, range(P_FULL), prio=6))
        else:
            self.load_routes(ents, allr)
            n = None
        have = collections.Counter(len(v) for v in self.arm(ents, allr).values())
        G["new_tip_cache"] = dict(requested=n, rows_by_reads={str(k_): v for k_, v in sorted(have.items())}, rows_without=len(allr) - sum(have.values()),
                                  missing_reads=sum(P_FULL - k_ for k_, v in have.items() for _ in range(v)) + P_FULL * (len(allr) - sum(have.values())))
        log(f"  tip {tip['id']} -> {new['id']}: {'; '.join(change)}; cache of the new tip: {G['new_tip_cache']}")

    def step_report(self):
        G = self.G5()
        tip0 = G.get("tip_at_start")
        L = [f"selfedit v5, generation {self.g}: tip at start {tip0} -> tip now {self.tip()['id']}", ""]
        fr = G.get("frontier") or {}
        L.append(f"READ POLICY of the tip at start (frontier vertices chosen on signal rows, valued on JUDGE rows; default = {fr.get('default')}; the budget is a reporting choice)")
        for p in fr.get("points", []):
            if p["name"] in fr.get("report", []) or p["vertex"]:
                L.append(f"  {p['name']:30s} JUDGE acc {100 * p['acc_judge']:6.2f} (signal {100 * p['acc_signal']:6.2f}) tokens x{p['multiple']:4.2f} (total x{p['multiple_total']:4.2f})"
                         + ("  <- default budget" if p["name"] == fr.get("default") else "") + ("  [vertex]" if p["vertex"] else ""))
        L.append("")
        L.append("SIGNALS (SHOW + MONITOR rows)")
        for b, s in (G.get("signals") or {}).get("by_bench", {}).items():
            L.append(f"  {b:12s} acc {100 * s['acc']:5.1f} disagreement of two reads {100 * s['disagree']:4.1f}% declared guess {100 * s['guess'][str(WINDOWS[-1])]:4.1f}% tokens {s['tokens']:6.0f}")
        g = self.tip()["policy"].get("guess") or (G.get("guess") or {})
        L += ["", f"CANDIDATES ({G.get('cells_info')})"]
        for c in G.get("cands", []):
            s = c.get("screen") or {}
            L.append(f"- {c['cid']} {c['kind']:8s} {c['name']} | source {c.get('source') or '-'}" + ("" if c.get("line") else f" | {c.get('status')}"))
            if c.get("skipped_rows"):
                L.append(f"    rows the candidate could not be read on (error / op_skipped lines; they count as the tip): {c['skipped_rows']}")
            if c.get("why"):
                L.append(f"    proposer: {str(c['why'])[:200]!r}" + (f" | {json.dumps({k_: c['entry'].get(k_) for k_ in ('condition', 'procedure', 'settings') if c['entry'].get(k_)}, ensure_ascii=False)[:300]}" if c.get("entry") else ""))
            if s.get("judged"):
                L.append(f"    signal rows: x{s['multiple']:.2f} raw {100 * s['raw']:+.2f} at equal compute {100 * s['d']:+.2f} +- {100 * s['se']:.2f} p_macro {s['p_macro']:.3f}"
                         + (f" | null {100 * s['null']['d']:+.2f} +- {100 * s['null']['se']:.2f} (audit {'ok' if s['null']['audit_ok'] else 'FAILED'})" if s.get("null") else ""))
            for lab_, key_ in (("signal rows", "budgets_signal"), ("JUDGE rows ", "budgets_judge")):
                if c.get(key_):
                    L.append(f"    {lab_}, frontier against frontier: " + " | ".join(
                        f"{b_} acc {100 * v_['acc']:.2f} (x{v_['multiple']:.2f}) vs the tip at equal tokens {100 * v_['d']:+.2f} +- {100 * v_['se']:.2f} p_gen {v_['p_gen']:.3f}"
                        for b_, v_ in c[key_].items() if v_.get("judged")))
            if c.get("line"):
                L += ["    " + x for x in c["line"].split("\n")]
        if G.get("cards") or G.get("dropped"):
            L += ["", f"PROPOSER: cards {G.get('cards')}; dropped proposals ({len(G.get('dropped') or [])}), reason and first 200 characters:"]
            for d in G.get("dropped") or []:
                L.append(f"  - [{'mixed card' if d.get('mixed') else 'card of ' + ','.join(d.get('source') or [])}] {d['why']} | {d['head']!r}")
        L += ["", "SIGNAL-ROW GAIN AGAINST JUDGE-ROW GAIN (equal compute; how much of a signal-row gain is fit)"]
        for r in self.state["shrinkage"][-20:]:
            if r.get("kind") == "admission":
                L.append(f"  g{r['gen']} admission {r['name'][:38]:38s} source {','.join(r['source'])}: JUDGE rows {100 * r['source_judge_d']:+5.2f} +- {100 * r['source_judge_se']:4.2f}"
                         + (f" | MONITOR rows {100 * r['source_monitor_d']:+5.2f} +- {100 * r['source_monitor_se']:4.2f}" if r.get("source_monitor_d") is not None else " | MONITOR rows: none")
                         + (" | ADMITTED (ID_SPECIFIC)" if r.get("admitted") else " | not admitted"))
                continue
            L.append(f"  g{r['gen']} {r['kind']:8s} {r['name'][:40]:40s} signal {100 * r['signal_d']:+5.2f} +- {100 * r['signal_se']:4.2f} | JUDGE {100 * r['judge_d']:+5.2f} +- {100 * r['judge_se']:4.2f}"
                     + (f" | kept {100 * r['kept']:.0f}%" if r.get("kept") is not None else "") + (f" | fitted on {r['fitted_on']}" if r.get("fitted_on") else ""))
        t = self.tip()
        L += ["", f"TIP {t['id']}",
              "  GENERAL system: " + "; ".join(f"{e['cid']} ({'settings ' + json.dumps(e['settings']) if e.get('settings') else 'cell' if e.get('edit') else 'operator' if e.get('ops') else 'entry'})"
                                              for e in t["entries"]) + f" + read policy {t['policy'].get('default')}" + "".join(f" + {x['name']}" for x in t["policy"].get("extras", []))]
        L.append(f"  SPECIALISATION layer (OFF in the general system; never counted as a general gain): {len(t['special'])} entries")
        for e in t["special"]:
            v = e.get("verdict") or {}
            L.append(f"    {e['cid']} source {e.get('source') or '?'}: {v.get('layer')}"
                     + (f", off-source {100 * v['d']:+.2f} pp" + (f" (p_gen {v['p_gen']:.3f})" if v.get("p_gen") is not None else "") if v.get("d") is not None else "")
                     + (f", all benchmarks {100 * v['pooled']:+.2f} pp" if v.get("pooled") is not None else "")
                     + (f", source benchmark JUDGE rows {100 * v['source_d']:+.2f} +- {100 * v['source_se']:.2f} pp (p {v['source_p']:.4f}, K {v.get('K')})" if v.get("source_d") is not None else "")
                     + (f", source benchmark MONITOR rows {100 * v['monitor_d']:+.2f} +- {100 * v['monitor_se']:.2f} pp" if v.get("monitor_d") is not None else ""))
        if G.get("tip_change"):
            L.append(f"  changes this generation: {'; '.join(G['tip_change'])}")
            L.append(f"  cache of the new tip: {G.get('new_tip_cache')}")
        both = self.both_systems()
        if both:
            L += ["", "BOTH SYSTEMS on JUDGE rows (one read; the specialisation system is reported, not optimised)"] + both
        open(os.path.join(self.rd, f"report5_g{self.g:03d}.txt"), "w").write("\n".join(L) + "\n")
        with open(os.path.join(self.rd, "ledger5.jsonl"), "a") as fh:
            for c in G.get("cands", []):
                fh.write(json.dumps(dict({k: v for k, v in c.items() if k not in ("line", "card")}, t=time.time()), ensure_ascii=False) + "\n")
        log(f"  report5_g{self.g:03d}.txt written")

    def both_systems(self):
        t, out = self.tip(), []
        self.load_routes(t["entries"] + t["special"], self.JR)
        both = sorted(t["entries"] + t["special"], key=lambda e: e.get("pos", 10 ** 6))
        for name, ents in (("general", t["entries"]), ("general + specialisation", both)):
            arm = self.arm(ents, self.JR)
            n = min([len(arm.get(r, ())) for r in self.JR])
            if n < 2:
                out.append(f"  {name:26s} not read (fewer than 2 cached reads on some JUDGE row)")
                continue
            byb = collections.defaultdict(list)
            for r in self.JR:
                byb[self.rows[r]["bench"]].append(sum(x["ok"] for x in arm[r]) / len(arm[r]))
            acc = sum(sum(v) / len(v) for v in byb.values()) / len(byb)
            out.append(f"  {name:26s} acc {100 * acc:6.2f} (>= {n} reads per row) | " + " ".join(f"{b[:5]} {100 * sum(v) / len(v):5.1f}" for b, v in sorted(byb.items())))
        return out if t["special"] else out[:1]

    # ------------------------------------------------------------------ loop
    def generation(self, g):
        self.g = g
        G = self.state["gens"].setdefault(str(g), dict(t0=time.time(), done=[], times={}, tip_at_start=self.tip()["id"], v=5))
        self.save()
        for st in STEPS5:
            if st in G["done"]:
                continue
            if self.a.stop_after and st == self.a.stop_after:
                log(f"stop before step {st} (--stop-after)")
                sys.exit(3)
            t = time.time()
            log(f"v5 g{g} step {st}")
            getattr(self, "step_" + st)()
            if st == "signals":
                G["frontier"] = self.tip()["policy"]["frontier"]
            G["done"].append(st)
            G["times"][st] = round(time.time() - t, 1)
            self.save()
        self.state["gen"] = g
        G["t_end"] = time.time()
        self.save()

    def main(self):
        g = max([int(k) for k in self.state["gens"]] or [0])
        n = self.a.gens
        if g and len(self.state["gens"][str(g)]["done"]) < len(STEPS5):
            self.generation(g)
            n -= 1
        for _ in range(max(0, n)):
            if os.path.exists(os.path.join(self.rd, "STOP")):
                log("STOP file: stopping at the generation boundary")
                return
            self.generation(max([int(k) for k in self.state["gens"]] or [0]) + 1)


def load_sources(path):
    """{cid: [benchmark]} from the ledger export (e6: lines with cid and card_source) or a json {cid: [bench]}."""
    if path.endswith(".json"):
        return json.load(open(path))
    src = {}
    for l in (gzip.open(path, "rt") if path.endswith(".gz") else open(path)):
        if l.strip():
            v = json.loads(l)
            if v.get("card_source"):
                src[v["cid"]] = [v["card_source"]]
    return src


def cli_init(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--tip", required=True, help="a v4 tip file (tips/<id>.json)")
    p.add_argument("--sources", default=None)
    p.add_argument("--verdicts", default=None, help="stage-1 verdicts.json: its entry_audit records give the layers (no re-read of the audit)")
    p.add_argument("--members", default=None, help="json {audit name: [cid]} naming the entries of each audit record of --verdicts")
    a = p.parse_args(argv)
    if os.path.exists(os.path.join(a.run, "state.json")):
        raise Refuse(f"{a.run}/state.json exists")
    verd = None
    if a.verdicts:
        mem = json.load(open(a.members))
        verd = {}
        for r in json.load(open(a.verdicts))["entry_audit"]:
            for cid in mem[r["name"]]:
                verd[cid] = dict(layer=r["layer"], d=r["eq"]["d"], se=r["eq"]["se"], p_gen=r["eq"]["p_gen"], pooled=r["pooled"]["d"], pooled_p=r["pooled"]["p_macro"],
                                 scored=r["scored"], benches=sorted(set(r["scored"]) | set(r.get("source") or [])), audit=r["name"], gen=0,
                                 basis="stage-1 verdict table 11-11")
    st = init_state(json.load(open(a.tip)), load_sources(a.sources) if a.sources else None, verd)
    os.makedirs(a.run, exist_ok=True)
    jdump(os.path.join(a.run, "state.json"), st)
    print(f"{a.run}/state.json: tip {st['tip']['id']}, general {len(st['tip']['entries'])} entries, specialisation {len(st['tip']['special'])}")


if __name__ == "__main__":
    try:
        if len(sys.argv) > 1 and sys.argv[1] == "init":
            cli_init(sys.argv[2:])
        else:
            Run5(args5()).main()
    except Refuse as ex:
        print(f"REFUSED: {ex}")
        sys.exit(2)
