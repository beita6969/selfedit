"""B4 (coordinator 11-11, Track B): the measurement planner. Which cells to read next on SIGNAL rows, under a read budget.

Library (a json list; nothing assumes a cell is "direction x alpha x band": an entry says what it is)
  {"family", "kind": "weight", "restrict": {"base_edit": <Surgeon primitive list file>, "base_alpha": 1.0, "layers": [0, 63],
   "server_dir": <where the engine reads the derived files>[, "dose_on": {"op": "scale", "param_suffix": "mlp.down_proj"}]},
   "dose": {"range": [lo, hi]}, "prior_family": n}        any alpha and any contiguous band = the same primitives restricted / rescaled
  {"family", "kind": "weight" | "ops" | "setting", "cells": [{"name", "edit" | "ops" | "settings"[, "dose", "continues"]}]}
                                                          explicit cells (position / state units, ops, ...): raced as they are
  {"family", "kind": "setting", "knob": <name>, "dose": {"range": [lo, hi]} | {"values": [...]}, "origin": <the tip's value>}
Schedule = successive halving with reuse, racing on row-paired statistics (all cells read the same rows in the same order):
  rounds   K = ceil(log2(coarse cells)); every round may spend (reads left) / (rounds left).
  round 0  the coarse grid (coarse_doses doses over the range x the thirds of the layer range and the full band; every explicit
           cell), one sample, on the longest prefix of the race order the round can pay for. The race order is fixed per plan: every
           prefix is balanced by benchmark and, inside a benchmark, by the tip's read length (short / long) and picture cluster.
  update   stats from signal_cells (the pairing with the smaller se). value v = judge-scale gain - lambda x added tokens, lambda = the
           current shadow price of the token budget. incumbent = the best lower bound of v (the tip itself counts: v = 0).
           A cell STOPS when the paired upper bound of (v_cell - v_incumbent) is < 0, when its gain's upper bound is < 0, or when the
           lower bound of its added tokens exceeds the budget (bounds at alpha / rounds). Of the rest the top half by upper bound of
           v continue (the others are parked, not dead); they go to the next longer prefix (then more samples) the round can pay for.
  refine   with what the round has left: per continuing dose arm the midpoint of the dose segment next to its best dose with the
           steeper gain per added token (bisection, at most dose_depth times), and for the leader one band move (the union with an
           adjacent band that gains, else its two halves). A new cell reads the whole prefix, so every comparison stays paired.
Files  PLAN/state.json (cells, statuses, levels, rounds), PLAN/batch_<k>.json (the requests: cell spec + rows + samples; row ids of
       show / monitor rows only), PLAN/reads.jsonl (appended by ingest: cell, row, sample, ok, tokens, pred), PLAN/edits/*.json.
usage: python3 signal_plan.py init --run RUN --tip T --rows learn.jsonl --library LIB.json --budget-reads N
       python3 signal_plan.py next --run RUN --tip T --rows learn.jsonl          (writes the next batch_<k>.json, or says why not)
       python3 signal_plan.py ingest --run RUN --tip T --rows learn.jsonl RESULT.jsonl ...
       python3 signal_plan.py lines --run RUN --tip T --batch K > requests.jsonl  (engine_api request lines from the tip's view)
Python standard library + numpy."""
import argparse, collections, hashlib, json, math, os, re, sys, time

import numpy as np

import signal_cells as SC

ETA = 2                        # successive halving: the top 1 / ETA continue (the standard choice)
COARSE_DOSES = 3               # doses of the coarse grid over a declared range (low / middle / high)
DOSE_DEPTH, BAND_DEPTH = 3, 4  # bisections of a coarse dose gap; a band is not split below layers / 2^BAND_DEPTH
MAX_SAMPLES = 2                # samples per row at the last level (tonight's hand measurements read 2)
KEY = "sc|{tip}|{sha}|{row}|{s}"


def _jdump(p, x):
    with open(p + ".tmp", "w") as f:
        json.dump(x, f, ensure_ascii=False, indent=1)
    os.replace(p + ".tmp", p)


def _h(*a):
    return hashlib.sha256("|".join(str(x) for x in a).encode()).hexdigest()


# ---------------------------------------------------------------------------------------------------------------- edits of a library entry
def restrict_edit(prims, band, alpha, base_alpha=1.0, dose_on=None):
    """A Surgeon primitive list restricted to the layers of the band [a, b] with the dose set to alpha: the primitives whose param is
    layers.<L>.* with a <= L <= b, the dose primitive (default: the "scale" of mlp.down_proj, make_cell's bias form) multiplied by
    alpha / base_alpha. Only for library entries that declare "restrict"."""
    d = dict(op="scale", param_suffix="mlp.down_proj")
    d.update(dose_on or {})
    out, n = [], 0
    for p in prims:
        m = re.match(r"(?:.*\.)?layers\.(\d+)\.", str(p.get("param", "")))
        if not m:
            raise ValueError(f"restrict_edit: primitive without a layer in its param: {json.dumps(p)[:120]}")
        if not band[0] <= int(m.group(1)) <= band[1]:
            continue
        q = dict(p)
        if q.get("op") == d["op"] and str(q.get("param", "")).endswith(d["param_suffix"]):
            q["alpha"] = float(q.get("alpha", 1.0)) * float(alpha) / float(base_alpha)
            n += 1
        out.append(q)
    if not out:
        raise ValueError(f"restrict_edit: no primitive in layers {band}")
    if n == 0 and float(alpha) != float(base_alpha):
        raise ValueError(f"restrict_edit: no dose primitive ({d}) in layers {band}; the entry must name it (dose_on)")
    return out


def thirds(layers):
    a, b = layers
    n = b - a + 1
    c1, c2 = a + int(round(n / 3.0)), a + int(round(2.0 * n / 3.0))
    return [[a, c1 - 1], [c1, c2 - 1], [c2, b]]


def _g(x):
    return f"{float(x):g}"


# ---------------------------------------------------------------------------------------------------------------- the plan
class Plan:
    def __init__(self, d, st):
        self.d, self.st = d, st

    # ------------------------------------------------------------------ creation
    @classmethod
    def create(cls, d, tip_id, tip, R, library, budget_reads, seed=0, coarse_doses=COARSE_DOSES, max_samples=MAX_SAMPLES, sample0=0,
               dose_depth=DOSE_DEPTH, band_depth=BAND_DEPTH, alpha=SC.ALPHA, rule="round"):
        os.makedirs(os.path.join(d, "edits"), exist_ok=True)
        for r in tip:
            R.check(r, "plan: tip table")
        bad = SC.prior_problems({e.get("family"): e for e in library})            # B4b: a weight entry states prior_family (no default)
        if bad:
            raise ValueError("plan: prior_family is required for every weight library entry (an integer >= 0): " + "; ".join(bad))
        st = dict(tip=tip_id, created=time.time(), library=library, budget=int(budget_reads), used=0, seed=seed, sample0=int(sample0),
                  params=dict(eta=ETA, coarse_doses=coarse_doses, max_samples=max_samples, dose_depth=dose_depth, band_depth=band_depth, alpha=alpha, ecomp=rule),
                  order=race_order(tip, R, seed), cells={}, rounds=[], level=None, finished=False)
        H = len(R.W)
        N = len(st["order"])
        sizes, n = [], N
        while n >= 2 * H:
            sizes.append(n)
            nn = int(math.ceil(n / 2.0 / H) * H)
            if nn >= n:
                break
            n = nn
        st["levels"] = [[n, 1] for n in sorted(sizes)] + [[N, s] for s in (2 ** k for k in range(1, 8)) if s <= max_samples]
        P = cls(d, st)
        for e in library:
            P._coarse(e)
        if not st["cells"]:
            raise ValueError("plan: the library gives no cell")
        st["K"] = max(1, int(math.ceil(math.log(len(st["cells"]), ETA))))
        P.save()
        return P

    @classmethod
    def load(cls, d):
        return cls(d, json.load(open(os.path.join(d, "state.json"))))

    def save(self):
        _jdump(os.path.join(self.d, "state.json"), self.st)

    # ------------------------------------------------------------------ cells of a library entry
    def _add(self, e, spec, arm=None, dose=None, band=None, why=""):
        spec = dict(spec, kind=e["kind"], family=e["family"], prior_family=int(e.get("prior_family") or 0), library=e.get("library") or e["family"])   # weight: create() refused a missing one
        spec.setdefault("engine", e.get("engine") or ("W" if e["kind"] == "weight" else "E"))
        cid = spec.setdefault("id", f"{e['family']}|{spec['name']}")
        if cid in self.st["cells"]:
            return None
        if dose is not None:
            spec["dose"] = dose
        self.st["cells"][cid] = dict(spec=spec, arm=arm, dose=dose, band=band, status="new", level=None, born=len(self.st["rounds"]), why=why,
                                     sha=SC.cell_sha(spec))
        return cid

    def _restricted(self, e, alpha, band, why):
        rs = e["restrict"]
        if not hasattr(self, "_base"):
            self._base = {}
        if rs["base_edit"] not in self._base:
            self._base[rs["base_edit"]] = json.load(open(rs["base_edit"]))
        prims = restrict_edit(self._base[rs["base_edit"]], band, alpha, rs.get("base_alpha", 1.0), rs.get("dose_on"))
        raw = json.dumps(prims)                                  # json.dump's default separators, as make_cell.py writes them
        name = f"{e['family']}_a{_g(alpha)}_L{band[0]}-{band[1]}_{hashlib.sha256(raw.encode()).hexdigest()[:8]}.json"   # edit_sha hashes the path:
        p = os.path.join(self.d, "edits", name)                  # the content hash in the name keeps "one path, one content"
        if not os.path.exists(p):
            with open(p + ".tmp", "w") as fh:
                fh.write(raw)
            os.replace(p + ".tmp", p)
        spec = dict(name=f"{e['family']} a{_g(alpha)} L{band[0]}-{band[1]}", id=f"{e['family']}|a{_g(alpha)}|L{band[0]}-{band[1]}", alpha=float(alpha),
                    band=list(band), edit=dict(op="surgeon", file=rs.get("server_dir", os.path.join(self.d, "edits")).rstrip("/") + "/" + name),
                    edit_local=p, base_edit=rs["base_edit"])
        return self._add(e, spec, arm=f"{e['family']}|L{band[0]}-{band[1]}", dose=float(alpha), band=list(band), why=why)

    def _knob(self, e, v, why):
        v = v if isinstance(e["dose"].get("values"), list) else round(float(v), 4)
        return self._add(e, dict(name=f"{e['knob']} {v}", id=f"{e['family']}|{v}", settings={e["knob"]: v}), arm=e["family"],
                         dose=float(v) if isinstance(v, (int, float)) else None, why=why)

    def _coarse(self, e):
        n = self.st["params"]["coarse_doses"]
        if e.get("cells"):
            for c in e["cells"]:                                  # explicit cells are coordinates of their own unless the entry says "exclusive"
                self._add(e, dict(c, **({} if e.get("exclusive") or c.get("menu") else {"menu": f"{e['family']}|{c['name']}"})), arm=None, dose=c.get("dose"), why="library cell")
        elif e.get("restrict"):
            lo, hi = e["dose"]["range"]
            lay = e["restrict"].get("layers", [0, 63])
            for band in thirds(lay) + [list(lay)]:
                for a in np.linspace(lo, hi, n):
                    self._restricted(e, float(round(a, 4)), band, "coarse grid")
        elif e.get("knob"):
            vals = e["dose"].get("values") or [float(v) for v in np.linspace(e["dose"]["range"][0], e["dose"]["range"][1], n)]
            for v in vals:
                if v != e.get("origin"):
                    self._knob(e, v, "coarse grid")
        else:
            raise ValueError(f"library entry {e.get('family')}: needs cells, restrict or knob")

    def entry(self, family):
        return next(e for e in self.st["library"] if e["family"] == family)

    # ------------------------------------------------------------------ reads
    def keys_of(self, c):
        if not c["level"]:
            return []
        n, S = c["level"]
        return [(r, s) for r in self.st["order"][:n] for s in range(self.st["sample0"], self.st["sample0"] + S)]

    def load_reads(self, R):
        """({cell: {row: {sample: read}}}, {cell: set of (row, sample) with a result or a final error})."""
        tb, done = {}, collections.defaultdict(set)
        p = os.path.join(self.d, "reads.jsonl")
        if os.path.exists(p):
            with open(p) as fh:
                for l in fh:
                    if l.strip():
                        x = json.loads(l)
                        SC.add_read(tb.setdefault(x["cell"], {}), R, x["row"], x.get("sample"), x, "plan reads")
                        done[x["cell"]].add((x["row"], int(x.get("sample") or 0)))
        return tb, done

    def ingest(self, lines, R):
        """Append finished reads (engine result lines with key "sc|tip|sha|row|s", or lines that name "cell") to reads.jsonl. A line
        whose row is not a show / monitor row raises before anything is written. Error lines are kept as "done, not read"."""
        bysha = {c["sha"]: cid for cid, c in self.st["cells"].items()}
        _, done = self.load_reads(R)
        out = []
        for x in lines:
            cid, row, s = x.get("cell"), x.get("row"), x.get("sample")
            k = str(x.get("key") or "").split("|")
            if len(k) == 5 and k[0] == "sc":
                if k[1] != self.st["tip"]:
                    continue
                cid, row, s = cid or bysha.get(k[2]), row or k[3], k[4] if s is None else s
            if cid not in self.st["cells"] or row is None:
                continue
            R.check(row, "plan ingest")
            s = int(s or 0)
            if (row, s) in done[cid]:
                continue
            done[cid].add((row, s))
            y = dict(cell=cid, row=row, sample=s)
            for f in ("ok", "tokens", "pred", "finish", "op_skipped", "error", "cancelled"):
                if x.get(f) is not None:
                    y[f] = x[f]
            out.append(y)
        if out:
            with open(os.path.join(self.d, "reads.jsonl"), "a") as fh:
                for y in out:
                    fh.write(json.dumps(y, ensure_ascii=False) + "\n")
        return len(out)

    # ------------------------------------------------------------------ statistics of the race
    def stats(self, tip, R, budget):
        """Per measured cell its global option (signal_cells.global_option), lambda_hat and the value vectors v = P - lambda A."""
        reads, done = self.load_reads(R)
        t0pop, allow, votes, _, _ = SC.tip_numbers(tip, R, budget, self.st["params"]["ecomp"])
        O = {}
        for cid, c in self.st["cells"].items():
            if reads.get(cid):
                _, o, est = SC.global_option(cid, reads[cid], tip, R, votes, self.st["params"]["ecomp"], t0pop)
                if o is not None:
                    o["est"] = est
                    O[cid] = o
        live = [O[cid] for cid, c in self.st["cells"].items() if cid in O and c["status"] != "stopped"]
        menu = collections.defaultdict(list)
        for o in live:
            menu[SC.menu_of(self.st["cells"][o["id"]]["spec"], o["id"])].append(dict(o))
        fr = SC.frontier(dict(menu), allow, self.st["params"]["alpha"], R)
        lam = fr["lam"] or 0.0
        for o in O.values():
            o["V"] = {r: o["P"][r] - lam * o["A"][r] for r in o["P"]}
            o["sV"] = SC.strat_stat(o["V"], R)
        return O, lam, allow, done, fr

    def pending(self, done):
        return sum(1 for cid, c in self.st["cells"].items() if c["status"] in ("active", "new") for k in self.keys_of(c) if k not in done[cid])

    # ------------------------------------------------------------------ one planning step
    def next_batch(self, tip, R, budget, force=False):
        """Ingested reads -> statuses -> the next batch (written to batch_<k>.json) or None (st["note"] says why: reads pending, or the
        race is finished)."""
        st = self.st
        P = st["params"]
        O, lam, allow, done, fr = self.stats(tip, R, budget)
        pend = self.pending(done)
        if pend and not force:
            st["note"] = f"waiting: {pend} requested reads have no result yet"
            self.save()
            return None
        k = len(st["rounds"])
        look = P["alpha"] / st["K"]
        act = [cid for cid, c in st["cells"].items() if c["status"] == "active"]
        n_prev, log = len(act), []
        # -- stops by bounds
        inc = max((cid for cid in act if cid in O), key=lambda cid: SC.bound(O[cid]["sV"], look), default=None)
        if inc is not None and not SC.bound(O[inc]["sV"], look) > 0:
            inc = None                                           # nothing is above the tip yet: the tip is the incumbent (v = 0)
        for cid in act:
            o, why = O.get(cid), None
            if o is None:
                continue
            if SC.bound(o["sP"], look, upper=True) < 0:
                why = f"gain upper bound {100 * SC.bound(o['sP'], look, upper=True):+.2f} pp < 0"
            elif SC.bound(o["sA"], look) > allow:
                why = f"added tokens lower bound {SC.bound(o['sA'], look):.0f} > budget {allow:.0f}"
            elif inc is None:
                if SC.bound(o["sV"], look, upper=True) < 0:
                    why = f"value upper bound {100 * SC.bound(o['sV'], look, upper=True):+.2f} pp < 0 (incumbent: the tip)"
            elif cid != inc:
                d = SC.strat_stat({r: o["V"][r] - O[inc]["V"][r] for r in o["V"] if r in O[inc]["V"]}, R)
                if SC.bound(d, look, upper=True) < 0:
                    why = f"paired upper bound of value against {inc} {100 * SC.bound(d, look, upper=True):+.2f} pp < 0"
            if why:
                st["cells"][cid].update(status="stopped", stop=dict(round=k, why=why))
                log.append(f"stop {cid}: {why}")
        act = [cid for cid in act if st["cells"][cid]["status"] == "active"]
        new = [cid for cid, c in st["cells"].items() if c["status"] == "new"]
        left = st["budget"] - st["used"]
        if k >= st["K"] or (k > 0 and not act) or left <= 0:
            st.update(finished=True, note=f"finished after {k} rounds: {len(act)} cells in the race at level {st['level']}, {st['used']} of {st['budget']} reads used")
            st["rounds"].append(dict(k=k, t=time.time(), final=True, lam=lam, incumbent=inc, log=log, standing=self._standing(O, lam)))
            self.save()
            return None
        b = left / (st["K"] - k)
        if k == st["K"] - 1:
            b = left
        reads_of = lambda lv: lv[0] * lv[1]                      # noqa: E731
        have = lambda cid: reads_of(st["cells"][cid]["level"]) if st["cells"][cid]["level"] else 0   # noqa: E731
        ucb = lambda cid: SC.bound(O[cid]["sV"], look, upper=True) if cid in O else SC.INF            # noqa: E731
        if k == 0:
            keep, refs = list(new), []
        else:
            act.sort(key=lambda cid: -ucb(cid))
            m = min(len(act), max(1, int(math.ceil(n_prev / float(P["eta"])))))   # halving counts the cells the bounds already stopped
            keep = act[:m]
            for cid in act[m:]:
                st["cells"][cid].update(status="parked", stop=dict(round=k, why=f"not in the top {m} by upper bound of value (halving)"))
                log.append(f"park {cid}")
            refs = self._refine(keep, O, lam) if keep else []
        cur = st["levels"].index(st["level"]) if st["level"] else -1
        cost = lambda li, cells, nref: sum(max(0, reads_of(st["levels"][li]) - have(c)) for c in cells) + nref * reads_of(st["levels"][li])   # noqa: E731
        top = len(st["levels"]) - 1
        li = None
        while keep:
            li = max((j for j in range(max(cur + (1 if k > 0 else 0), 0), top + 1) if cost(j, keep, 0) <= b), default=None)
            if li is not None or len(keep) == 1 or k == 0:
                break
            cid = keep.pop()                                     # the round cannot carry them all one level up: the weakest is parked
            st["cells"][cid].update(status="parked", stop=dict(round=k, why="the round's budget does not carry it to the next level"))
            log.append(f"park {cid} (budget)")
        if li is None:
            if k == 0:
                raise ValueError(f"plan: {len(keep)} coarse cells x {reads_of(st['levels'][0])} reads do not fit a round of {b:.0f} reads (budget {st['budget']}, {st['K']} rounds)")
            li = cur                                             # already at the top level (or nothing affordable): refinements only
        nref = 0
        while nref < len(refs) and cost(li, keep, nref + 1) <= b:
            nref += 1
        reqs, spent = [], 0
        lv = st["levels"][li]
        for why, spec_fn in refs[:nref]:
            cid = spec_fn()
            if cid:
                st["cells"][cid]["why"] = why
                keep.append(cid)
                log.append(f"new {cid}: {why}")
        for cid in keep:
            c = st["cells"][cid]
            old = set(self.keys_of(c))
            c.update(status="active", level=list(lv))
            ks = [x for x in self.keys_of(c) if x not in old]
            if not ks:
                continue
            rows = list(dict.fromkeys(r for r, _ in ks))
            samples = sorted({s for _, s in ks})
            full = len(ks) == len(rows) * len(samples)
            parts = [(rows, samples)] if full else [([r for r, s_ in ks if s_ == s], [s]) for s in samples]
            for rws, sms in parts:
                reqs.append(dict(cell=cid, sha=c["sha"], engine=c["spec"].get("engine"), spec={a: v for a, v in c["spec"].items() if a != "edit_local"},
                                 rows=rws, samples=sms, reads=len(rws) * len(sms), why=c.get("why") or ""))
            spent += len(ks)
        if not spent:
            st.update(finished=True, note=f"finished after {k} rounds: nothing affordable is left to read ({st['used']} of {st['budget']} reads used)")
            st["rounds"].append(dict(k=k, t=time.time(), final=True, lam=lam, incumbent=inc, log=log, standing=self._standing(O, lam)))
            self.save()
            return None
        st["used"] += spent
        st["level"] = list(lv)
        st["note"] = f"round {k}: {len(keep)} cells to level {lv} ({spent} reads; {st['budget'] - st['used']} left)"
        batch = dict(tip=st["tip"], plan=self.d, round=k, written=time.time(), reads=spent, budget_left=st["budget"] - st["used"], level=lv,
                     key=KEY, key_note="result lines must carry key (or cell), row, sample, ok, tokens[, pred, finish]", requests=reqs)
        st["rounds"].append(dict(k=k, t=time.time(), level=lv, cells=len(keep), reads=spent, lam=lam, incumbent=inc, log=log, standing=self._standing(O, lam)))
        _jdump(os.path.join(self.d, f"batch_{k}.json"), batch)
        self.save()
        return batch

    def _standing(self, O, lam):
        return {cid: dict(status=self.st["cells"][cid]["status"], rows=o["sP"]["n"], pred=SC._f(o["sP"]["d"]), se=SC._f(o["sP"]["se"]), tokens_multiple=SC._f(o["mult"], 3),
                          value=SC._f(o["sV"]["d"]), use=o["est"]["use"]) for cid, o in O.items()}

    # ------------------------------------------------------------------ refinement
    def _refine(self, keep, O, lam):
        """[(why, make the cell)] in priority order: one dose bisection per continuing dose arm (rank order), then one band move for the
        leader."""
        st, out, seen = self.st, [], set()
        P = st["params"]
        for cid in keep:
            c = st["cells"][cid]
            arm = c["arm"]
            if arm is None or c["dose"] is None or arm in seen:
                continue
            seen.add(arm)
            e = self.entry(c["spec"]["family"])
            if not e.get("dose"):
                continue
            pts = {x["dose"]: (O[i]["Pall"], O[i]["Aall"], O[i]["sV"]["d"]) for i, x in st["cells"].items() if x["arm"] == arm and x["dose"] is not None and i in O}
            org = 0.0 if e.get("restrict") else e.get("origin")
            if isinstance(org, (int, float)) and org not in pts:
                pts[float(org)] = (0.0, 0.0, 0.0)
            ds = sorted(pts)
            best = max(ds, key=lambda d_: pts[d_][2])
            i = ds.index(best)
            if e["dose"].get("values"):
                vals = sorted(v for v in e["dose"]["values"] if isinstance(v, (int, float)))
                near = [v for v in vals if v not in pts and (ds[i - 1] if i > 0 else -SC.INF) < v < (ds[i + 1] if i + 1 < len(ds) else SC.INF)]
                if near:
                    v = min(near, key=lambda v_: abs(v_ - best))
                    out.append((f"refine {arm}: the unread value {v} next to the best ({best})", lambda e=e, v=v: self._knob(e, v, "")))
                continue
            lo, hi = e["dose"]["range"]
            step = (hi - lo) / max(P["coarse_doses"] - 1, 1) / 2 ** P["dose_depth"]
            segs = []
            for seg in ([(ds[i - 1], best)] if i > 0 else []) + ([(best, ds[i + 1])] if i + 1 < len(ds) else []):
                a, b_ = seg
                mid = round((a + b_) / 2.0, 4)
                if b_ - a < 2 * step - 1e-9 or not lo <= mid <= hi or mid in pts:
                    continue
                dP, dA = pts[b_][0] - pts[a][0], pts[b_][1] - pts[a][1]
                segs.append((abs(dP / dA) if abs(dA) > 1e-9 else SC.INF, mid, a, b_))
            if segs:
                slope, mid, a, b_ = max(segs)
                why = f"refine {arm}: dose {_g(mid)}, the middle of [{_g(a)}, {_g(b_)}] next to the best dose {_g(best)} (the steeper side per added token)"
                if e.get("restrict"):
                    out.append((why, lambda e=e, mid=mid, band=c["band"], why=why: self._restricted(e, mid, band, why)))
                else:
                    out.append((why, lambda e=e, mid=mid, why=why: self._knob(e, mid, why)))
        lead = st["cells"][keep[0]] if keep else None
        if lead and lead.get("band") and self.entry(lead["spec"]["family"]).get("restrict"):
            e, (a, b_), al = self.entry(lead["spec"]["family"]), lead["band"], lead["dose"]
            lay = e["restrict"].get("layers", [0, 63])
            sib = {tuple(x["band"]): i for i, x in st["cells"].items() if x.get("band") and x["spec"]["family"] == e["family"] and x["dose"] == al}
            move = None
            for (a2, b2), i in sorted(sib.items(), key=lambda kv: -(O[kv[1]]["Pall"] if kv[1] in O else -1.0)):
                u = (min(a, a2), max(b_, b2))
                if i in O and O[i]["Pall"] > 0 and (a2 == b_ + 1 or b2 == a - 1) and u not in sib:
                    move = [(u, f"band move: the union L{u[0]}-{u[1]} of the leader's band and the adjacent band that gains, at dose {_g(al)}")]
                    break
            if move is None and (b_ - a + 1) >= 2 * max(1, (lay[1] - lay[0] + 1) // 2 ** P["band_depth"]):
                mid = (a + b_) // 2
                move = [(h, f"band move: half L{h[0]}-{h[1]} of the leader's band L{a}-{b_}, at dose {_g(al)}") for h in ((a, mid), (mid + 1, b_)) if h not in sib]
            for band, why in move or []:
                out.append((why, lambda e=e, al=al, band=list(band), why=why: self._restricted(e, al, band, why)))
        return out

    def summary(self):
        st = self.st
        L = [f"plan {self.d}: tip {st['tip']}, {len(st['cells'])} cells, {st['used']} / {st['budget']} reads, {sum(1 for r in st['rounds'] if not r.get('final'))} batches of {st['K']}, level {st['level']}"
             + ("; finished" if st["finished"] else ""), f"  {st.get('note', '')}"]
        cnt = collections.Counter(c["status"] for c in st["cells"].values())
        L.append("  cells: " + ", ".join(f"{k} {v}" for k, v in sorted(cnt.items())))
        sd = (st["rounds"][-1].get("standing") if st["rounds"] else None) or {}
        for cid, c in st["cells"].items():
            s = sd.get(cid) or {}
            L.append(f"  {c['status']:8s} {cid:28s} level {c['level']}  " + (f"pred {100 * s['pred']:+.2f} (se {100 * s['se']:.2f}) x{s['tokens_multiple']} value {100 * s['value']:+.2f} [{s['use']}]"
                                                                               if s.get("pred") is not None else "") + (f"  ({(c.get('stop') or {}).get('why')})" if c.get("stop") else ""))
        return "\n".join(L) + "\n"


def race_order(tip, R, seed=0):
    """The signal rows of the tip table in the plan's fixed order: benchmarks in turn; inside a benchmark short and long rows (the row's
    mean tip tokens against the benchmark's median) in turn; inside those, one row per picture cluster before any second one; the
    rest by a seeded hash. Every prefix of a multiple of the number of benchmarks is balanced by benchmark."""
    by = collections.defaultdict(list)
    for r in sorted(tip):
        by[R.bench(r)].append(r)
    seqs = []
    for b in sorted(by, key=str):
        rows = by[b]
        tok = {r: sum(q["tokens"] for q in tip[r].values()) / len(tip[r]) for r in rows}
        med = float(np.median(list(tok.values())))
        halves = []
        for part in ([r for r in rows if tok[r] <= med], [r for r in rows if tok[r] > med]):
            nth = collections.Counter()
            keyed = []
            for r in sorted(part, key=lambda r: _h(seed, r)):
                keyed.append((nth[R.cluster(r)], _h(seed, r), r))
                nth[R.cluster(r)] += 1
            halves.append([r for _, _, r in sorted(keyed)])
        seq, i = [], 0
        while any(halves):
            h = halves[i % 2] or halves[(i + 1) % 2]
            seq.append(h.pop(0))
            i += 1
        seqs.append(seq)
    out = []
    while any(seqs):
        for s in seqs:
            if s:
                out.append(s.pop(0))
    return out


# ---------------------------------------------------------------------------------------------------------------- engine request lines
def engine_lines(batch, traces, priority=5):
    """engine_api request lines of a batch from the tip's own view of each row in traces.json (prefix, settings with null = stock, ops,
    edit): the cell's settings replace the tip's, its ops are added (an op of the same name is replaced), its edit comes after the tip's
    (engine_api section 8: the full ordered edit list). W / T requests must go in batches of their own: {"E": [...], "W": [...]}."""
    out = {"E": [], "W": []}
    for q in batch["requests"]:
        sp = q["spec"]
        for r in q["rows"]:
            x = traces[r]
            if x.get("role") not in SC.SIGNAL_ROLES:
                raise ValueError(f"engine lines: row {r} has role {x.get('role')!r}; only {SC.SIGNAL_ROLES} may be read for the signal")
            st = dict(x.get("settings") or {})
            st.update(sp.get("settings") or {})
            ops = {o["op"]: o for o in (x.get("ops") or []) + (sp.get("ops") or []) if isinstance(o, dict)}
            te = x.get("edit")
            eds = ([] if not te else list(te) if isinstance(te, list) else [te]) + ([sp["edit"]] if sp.get("edit") else [])
            for s in q["samples"]:
                l = dict(key=batch["key"].format(tip=batch["tip"], sha=q["sha"], row=r, s=s), mode="read", engine="W" if sp.get("edit") else (q.get("engine") or "E"),
                         priority=priority, row=r, sample=s, prefix=x.get("prefix"), settings=st, cell=q["cell"])
                if ops:
                    l["ops"] = list(ops.values())
                if eds:
                    l["edit"] = eds[0] if len(eds) == 1 else eds
                    l["edit_sha"] = SC.esha(l["edit"])
                out["W" if l["engine"] in ("W", "T") else "E"].append(l)
    return out


# ---------------------------------------------------------------------------------------------------------------- command line
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("init", "next", "ingest", "lines", "status"))
    ap.add_argument("--run", required=True)
    ap.add_argument("--tip", required=True)
    ap.add_argument("--rows", help="learn.jsonl")
    ap.add_argument("--traces", help="the tip's traces.json (default RUN/signal/<tip>/traces.json)")
    ap.add_argument("--plan", help="plan directory (default RUN/signal/<tip>/plan)")
    ap.add_argument("--library")
    ap.add_argument("--budget-reads", type=int)
    ap.add_argument("--budget", type=float, help="token multiple M (default RUN/cards.json g_token_multiple)")
    ap.add_argument("--max-samples", type=int, default=MAX_SAMPLES)
    ap.add_argument("--sample0", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int)
    ap.add_argument("--force", action="store_true", help="plan although requested reads are still pending")
    ap.add_argument("files", nargs="*")
    a = ap.parse_args(argv)
    d = a.plan or os.path.join(a.run, "signal", a.tip, "plan")
    tp = a.traces or os.path.join(a.run, "signal", a.tip, "traces.json")
    if a.cmd == "status":
        sys.stdout.write(Plan.load(d).summary())
        return 0
    traces = json.load(open(tp))
    if a.cmd == "lines":
        k = a.batch if a.batch is not None else len(Plan.load(d).st["rounds"]) - 1
        L = engine_lines(json.load(open(os.path.join(d, f"batch_{k}.json"))), traces)
        for eng, ls in L.items():
            if ls:
                with open(os.path.join(d, f"batch_{k}_{eng}.jsonl"), "w") as fh:
                    for l in ls:
                        fh.write(json.dumps(l, ensure_ascii=False) + "\n")
                print(f"{len(ls)} {eng} request lines -> {os.path.join(d, f'batch_{k}_{eng}.jsonl')}")
        return 0
    R = SC.Rows.from_learn(a.rows) if a.rows else SC.Rows.from_traces(traces)
    tip = SC.tip_table(traces, R)
    if a.cmd == "init":
        P = Plan.create(d, a.tip, tip, R, json.load(open(a.library)), a.budget_reads, seed=a.seed, max_samples=a.max_samples, sample0=a.sample0)
        print(f"plan created: {len(P.st['cells'])} coarse cells, {P.st['K']} rounds, levels {P.st['levels']}, budget {P.st['budget']} reads")
    P = Plan.load(d)
    if a.cmd == "ingest":
        n = 0
        for f in a.files:
            with open(f) as fh:
                n += P.ingest((json.loads(l) for l in fh if l.strip()), R)
        print(f"{n} reads appended to {os.path.join(d, 'reads.jsonl')}")
        return 0
    b = P.next_batch(tip, R, SC.read_budget(a.run, a.budget), force=a.force)
    print(P.st.get("note"))
    if b:
        print(f"batch {b['round']}: {len(b['requests'])} requests, {b['reads']} reads -> {os.path.join(d, 'batch_%d.json' % b['round'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
