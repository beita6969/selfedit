"""L1 proposer versions for the v4 loop (spec A3, A7, AM4, AM5, AM7, AM8, AM12, AM14; coordinator brief 11-10 evening).

VersionsMixin is mixed into gen.Run (class Run(VersionsMixin, LabMixin)). It owns:
  - state["versions"] {vid: version} (selfedit format + meta born_gen, origin, base, window_start, status), state["inc"] (incumbent id,
    "P0" while it is identical to the frozen control P0, so the two share one author and one set of slots), state["challengers"] (active,
    creation order), state["challenger_queue"] (waiting, FIFO), state["authors"] (the authors with slots);
  - v_step_propose: the replacement of gen.Run.step_propose: Thompson slots per author (alloc_authors), prompts per author
    (build_prompts: per-source cards, memory, card parameters, drafts), one slot counter over all proposer authors, every draft key in
    prompts_g###.jsonl, and the draft pick after the answers (pick_drafts: p["key"] = the chosen draft's key);
  - outcome_rows / trace_items: the memory inputs of a reader version (SHOW rows only; JUDGE and MONITOR never appear);
  - v_step_l1 (step "l1" after "report"): challenger decisions (switch | drop | stay, re-base on a switch), the incumbent's self-edit
    (one chat, one repair turn on format-only errors), one blind self-mutation, the FIFO queue, the L1 report in G["l1"].
Author id = version id. Ledger rows keep that id; "you" in a reader's memory = the reader's lineage (parent chain)."""
import copy
import hashlib
import json
import math
import os
import random
import time

import numpy as np

import proposer as PR
import selfedit as SE
import l1 as L1                # kept importable (judged-candidate helpers); since AM15 no L1 decision or slot draw uses l1 / yield_l1
import ecomp as ECOMP

ALPHA = 0.10                 # = gen.ALPHA (t_versions checks)
GRACE = 0.0                  # = gen.GRACE (AM16: one time box, no grace)
PROP_TEMP, PROP_MAX = 0.7, 12000   # = gen.PROP_TEMP, gen.PROP_MAX (P0 requests byte-identical to gen's)
WINDOW = 4                   # Thompson window: the last 4 generations (each author from its window_start on)
MAX_CH = 3                   # active challengers at most (AM12)
def l1_floor(alpha):
    """AM16 item 2: the smallest K at which the permutation test of two authors with K slots each can reach alpha at all: the most extreme
    split has probability 1 / C(2K, K), so K = min{K : 1 / C(2K, K) <= alpha} (3 at alpha .10)."""
    K = 1
    while 1.0 / math.comb(2 * K, K) > alpha:
        K += 1
    return K


FLOOR_P = l1_floor(ALPHA)    # slot floor of every author in an L1 comparison: P0 (the control), the incumbent, every challenger; B: own sizing
L1_WAIT = 600.0              # the self-edit / repair chat may wait until max(deadline + GRACE, now + L1_WAIT)
BOOT = 4000                  # AM15: Bayesian bootstrap draws per comparison / report
P_RETIRE = 0.5               # AM15: a blocked queue retires the active challenger with the lowest P(better) if that P < 1/2
L1_FROM = 5                  # AM15 rule from this generation on when gen's args carry no --l1-from (coordinator 11-10: generation 5)
KNOB_PARAM = {"min_image_tokens": "card_min_tokens", "max_image_tokens": "card_max_tokens", "temperature": "temperature"}
FIX_MSG = ("Your answer could not be used: {err}.\nSend your edit again as exactly one json block in the answer format, "
           "within the limits listed above.")


def _sha16(x):
    return hashlib.sha256(x.encode()).hexdigest()[:16]


def _log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def canon(v):
    """Canonical content of a version (memory, lessons, card parameters with defaults, transfers as a set) for duplicate checks."""
    p = PR.params_of(v.get("params"))
    return json.dumps([v.get("memory", "none"), (v.get("lessons") or "").strip() if v.get("memory") == "lessons" else None,
                       sorted(p.items()), sorted(map(str, v.get("transfers") or []))], sort_keys=True, default=str)


def _chars(msgs):
    n = 0
    for m in msgs:
        c = m["content"]
        n += len(c) if isinstance(c, str) else sum(len(x.get("text", "")) for x in c if x.get("type") == "text")
    return n


def boot_y(m, s, T, rng):
    """AM15 Bayesian bootstrap of Y = mean accepted contribution per slot. m (n,) = share x d of each slot whose candidate was accepted
    (0 for every other slot: rejected, invalid, unread), s (n,) = share x se_used of the accepted slots (0 otherwise). Each draw: weights
    Dirichlet(1, ..., 1) over the n slots plus ONE zero pseudo-slot, every accepted slot's d redrawn from N(d, se^2). Returns T draws."""
    m, s = np.asarray(m, float), np.asarray(s, float)
    n = len(m)
    if n == 0:
        return np.zeros(T)
    W = rng.dirichlet(np.ones(n + 1), size=T)[:, :n]
    return (W * (m[None, :] + s[None, :] * rng.standard_normal((T, n)))).sum(1)


def p_greater(ya, yb):
    """P(Y_a > Y_b) from paired-by-index independent draws (ties count half: two authors without an accept tie at exactly 0)."""
    ya, yb = np.asarray(ya), np.asarray(yb)
    return float(np.mean(ya > yb) + 0.5 * np.mean(ya == yb))


def perm_better(mc, mi, T, rng):
    """Variant am15perm (builder's proposal, 11-10): P(better) = 1 - one-sided mid-p of a permutation test of the plug-in Y difference
    (challenger slots mc vs incumbent slots mi, slot labels permuted T times). Exact under exchangeable slots, so a null author pair is
    switched in <= alpha of cases also when accepts are sparse (where the Bayesian bootstrap switches 0.14-0.25 at 8-24 slots)."""
    mc, mi = np.asarray(mc, float), np.asarray(mi, float)
    if len(mc) == 0 or len(mi) == 0:
        return 0.5
    x = np.concatenate([mc, mi])
    n = len(mc)
    obs = mc.mean() - mi.mean()
    xs = x[np.argsort(rng.random((T, len(x))), 1)]
    dd = xs[:, :n].mean(1) - xs[:, n:].mean(1)
    tol = 1e-12 * max(1.0, float(np.abs(x).max()))
    return float(np.mean(dd < obs - tol) + 0.5 * np.mean(np.abs(dd - obs) <= tol))


def _rng(*parts):
    return np.random.default_rng(int(_sha16(":".join(map(str, parts))), 16))


def _format_only(err):
    """Repairable self-edit failure: a parse error (no / bad json, a field missing or of the wrong type) or only unknown fields."""
    if err is None:
        return False
    if isinstance(err, str):
        return True
    return bool(err) and all(str(p).startswith("unknown field") for p in err)


class VersionsMixin:
    # ------------------------------------------------------------------ state
    def v_state(self):
        st = self.state
        if "versions" not in st:
            p0 = copy.deepcopy(SE.VERSION_DEFAULT)
            p0.update(born_gen=0, origin="control", base=None, edit=None, window_start=1, status="frozen control")
            st["versions"] = {"P0": p0}
        st.setdefault("inc", "P0")
        st.setdefault("challengers", [])
        st.setdefault("challenger_queue", [])
        st.setdefault("v_next", 1)
        return st

    def proposer_authors(self):
        st = self.v_state()
        return ["P0"] + ([st["inc"]] if st["inc"] != "P0" else []) + list(st["challengers"])

    def lineage(self, vid):
        V, out, x = self.v_state()["versions"], set(), vid
        while x and x in V and x not in out:
            out.add(x)
            x = V[x].get("parent")
        return out

    def _ws(self, a):
        v = self.v_state()["versions"].get(a)
        return int(v.get("window_start", 1)) if v else 1

    def _all_cands(self, upto=None):
        upto = self.g if upto is None else upto
        return [c for gk, G in sorted(self.state["gens"].items(), key=lambda x: int(x[0])) if int(gk) <= upto for c in G.get("cands", [])]

    # ------------------------------------------------------------------ transfers (accepted executor setting edits usable as card params)
    @staticmethod
    def _transfer_of(c):
        """{param: value} of an accepted always / setting-only executor candidate whose every knob maps to a card-menu value; {} otherwise."""
        e = c.get("entry") or {}
        if not (c.get("accepted") and c.get("kind") == "setting" and e.get("cond") == "always" and e.get("settings")):
            return {}
        out = {}
        for k, v in e["settings"].items():
            name = KNOB_PARAM.get(k)
            if name is None or not PR._in_menu(v, PR._menu_values(PR.PARAMS_MENU, name)):
                return {}
            out[name] = v
        return out

    def transfer_params(self, cid):
        for c in self._all_cands(10 ** 9):
            if c.get("cid") == cid:
                return self._transfer_of(c)
        return {}

    def transfer_edits(self, reader):
        """The accepted executor edits a version can transfer (selfedit _edits_block format), oldest first."""
        lin, out = self.lineage(reader), []
        for c in self._all_cands():
            if self._transfer_of(c):
                e, j = c["entry"], c.get("final") or c.get("judge") or {}
                out.append(dict(id=c["cid"], gen=c.get("gen"), author=self._rel(c["author"], lin), kind=c.get("kind"), condition=e.get("condition"),
                                procedure=e.get("procedure"), settings=e.get("settings") or {}, ops=e.get("ops") or [], D=j.get("D"), lfsr=j.get("lfsr")))
        return out

    def eff_params(self, v):
        """Card parameters of a version: its own, then its transfers in order (a later one wins)."""
        p = PR.params_of(v.get("params"))
        if v.get("transfers"):
            ix = {c.get("cid"): c for c in self._all_cands(10 ** 9)}
            for t in v["transfers"]:
                p.update(self._transfer_of(ix[t]) if t in ix else {})
        return p

    # ------------------------------------------------------------------ slots per author
    def l1_on(self):
        """AM15 decisions and Thompson slots are on iff --l1-rule am15 (default) and the generation is >= --l1-from (default L1_FROM).
        Off: challengers are still made, admitted and slotted (floors, uniform extra slots); Y is reported; nobody is switched,
        dropped or retired."""
        if not hasattr(self.a, "l1_rule") and hasattr(self.a, "l1_switch"):   # gen.py before the --l1-rule fold: its --l1-switch gate holds
            on = bool(self.a.l1_switch)
        else:
            on = self.l1_rule_name() in ("am15", "am15perm")
        return on and self.g >= int(getattr(self.a, "l1_from", None) or L1_FROM)

    def l1_rule_name(self):
        return str(getattr(self.a, "l1_rule", None) or "am15")

    @staticmethod
    def _slot_value(c):
        """(share x d, share x se_used, accepted) of one slot's candidate; 0 for a slot without an accepted candidate. Final numbers when
        the candidate has them in AM15 form (d, se_used), else the judgement it was accepted on; pre-AM15 records fall back to D / se."""
        if not c or not c.get("accepted"):
            return 0.0, 0.0, False
        f = c.get("final") or {}
        j = f if f.get("d") is not None and f.get("se_used") is not None else (c.get("judge") or {})
        d = j.get("d", j.get("D"))
        se = j.get("se_used", j.get("se"))
        if d is None:
            return 0.0, 0.0, False
        sh = float(c.get("share") or 0.0)
        return sh * float(d), sh * float(se or 0.0), True

    def _gens(self, author, lo, hi):
        """Generations lo..hi (from the author's window_start on) that have candidates."""
        return [g for g in range(max(int(lo), self._ws(author), 1), int(hi) + 1) if "cands" in (self.state["gens"].get(str(g)) or {})]

    def bundle_credit(self):
        """AM23 L1 credit (coordinator 11-10): an accepted bundle's contribution (share x d, share x se_used, as _slot_value; a hand bundle
        such as H1 from its carry_in numbers) is credited to its member candidates in proportion to share_j x max(d_j, 0) at step 1 (their
        own first samples: am21 step1, else the single-sample judgement of a pre-LITE generation). A member accepted on its own keeps its own
        value (no double count; the weights still run over all members). {cid: (value, se)}; a member of several bundles keeps the largest."""
        allc = {c["cid"]: c for G in self.state["gens"].values() for c in G.get("cands", [])}
        bs = [c for c in allc.values() if c.get("kind") == "bundle" and c.get("accepted")]
        bs += [x for x in self.state.get("synthetic", []) if x.get("accepted") and (x.get("entry") or {}).get("parts")]
        out = {}
        for b in bs:
            if b["cid"] in allc:
                v, e, ok = self._slot_value(b)
            else:
                ci = b.get("carry_in") or {}
                sh = float(b.get("share") or 0.0)
                ok = ci.get("d") is not None
                v, e = (sh * float(ci["d"]), sh * float(ci.get("se_used", ci.get("se")) or 0.0)) if ok else (0.0, 0.0)
            if not ok:
                continue
            ms = [allc[x.get("cid")] for x in b["entry"]["parts"] if x.get("cid") in allc]
            d1 = lambda m: ((m.get("am21") or {}).get("step1") or {}).get("d", (m.get("judge") or {}).get("D", (m.get("judge") or {}).get("d"))) or 0.0
            wt = {m["cid"]: float(m.get("share") or 0.0) * max(float(d1(m)), 0.0) for m in ms}
            tot = sum(wt.values())
            if tot <= 0:
                continue
            for m in ms:
                if m.get("accepted") or wt[m["cid"]] <= 0:
                    continue
                x = (v * wt[m["cid"]] / tot, e * wt[m["cid"]] / tot)
                if m["cid"] not in out or x[0] > out[m["cid"]][0]:
                    out[m["cid"]] = x
        return out

    def slot_data(self, author, gens):
        """(m, s, n_accepted): one entry per prop (slot) of the author in the generations, valued by _slot_value of its candidate; a member
        of an accepted AM23 bundle that is not accepted itself is valued by its share of the bundle (bundle_credit)."""
        m, s, na = [], [], 0
        credit = self.bundle_credit()
        for g in gens:
            G = self.state["gens"].get(str(g)) or {}
            cm = {(c["author"], c.get("slot")): c for c in G.get("cands", [])}
            for p in G.get("props", []):
                if p["author"] == author:
                    c_ = cm.get((author, p["slot"]))
                    v, e, ok = self._slot_value(c_)
                    if not ok and c_ is not None and c_["cid"] in credit:
                        (v, e), ok = credit[c_["cid"]], True
                    m.append(v)
                    s.append(e)
                    na += ok
        return np.array(m), np.array(s), na

    def _has_slots(self, author, g):
        return any(p["author"] == author for p in (self.state["gens"].get(str(g)) or {}).get("props", []))

    def l1_floor_total(self):
        """Sum of the L1 floors of the authors that get slots this generation (P0, the incumbent, the active challengers)."""
        st = self.v_state()
        return FLOOR_P * (1 + (st["inc"] != "P0") + len(st["challengers"]))

    def alloc_authors(self, n_total, rng):
        """{author: slots}: floors (B 1 when B is in the draw; P0, the incumbent and every challenger FLOOR_P = l1_floor(alpha), AM16) in the order P0, incumbent, B,
        challengers (FIFO); a challenger whose floor does not fit waits (0 slots, not dropped). The rest (AM15, l1_on): per slot one
        Bayesian-bootstrap draw of Y per author over its slots of the last WINDOW generations (from its window_start), the slot to the
        argmax (exact ties uniform); rule off: uniform draws. K in self._alloc_info = window slots per author."""
        st = self.v_state()
        inc, chs = st["inc"], list(st["challengers"])
        b_on = getattr(self, "b_in_draw", True) and not (hasattr(self, "b_cells_cap") and self.b_cells_cap() == 0)   # cards.json b_cells 0: B out
        order = ["P0"] + ([inc] if inc != "P0" else []) + (["B"] if b_on else []) + chs
        got, left, waiting = {}, int(n_total), []
        for a in order:
            f = 1 if a == "B" else FLOOR_P                       # AM16: P0 too (it is the control every author is compared with)
            if a in chs:
                if f <= left:
                    got[a], left = f, left - f
                else:
                    waiting.append(a)
            else:
                got[a] = min(f, max(left, 0))
                left -= got[a]
        el = list(got)
        data = {a: self.slot_data(a, self._gens(a, self.g - WINDOW, self.g - 1)) for a in el}
        on = self.l1_on()
        mode = "am15" if on else "uniform"
        if left > 0 and el:
            if on:                                                   # AM15 Thompson: per slot one bootstrap draw of Y per author, argmax
                Y = np.stack([boot_y(data[a][0], data[a][1], left, rng) for a in el], 1)
                for row in Y:
                    top = np.flatnonzero(row == row.max())           # exact ties (e.g. no accepts anywhere) broken uniformly
                    got[el[int(top[rng.integers(len(top))])]] += 1
            else:
                for _ in range(left):
                    got[el[int(rng.integers(len(el)))]] += 1
        self._alloc_info = dict(n_total=int(n_total), mode=mode, waiting=waiting, K={a: len(data[a][0]) for a in el},
                                accepted={a: int(data[a][2]) for a in el}, slots=dict(got))
        return got

    # ------------------------------------------------------------------ prompts
    def build_prompts(self, author, n_slots, parent_show, tv, slot0=0):
        """(reqs, props) of one proposer author: per-source slots (alloc_sources over the SHOW residual; mixed cards: one pool), cards by
        PR.draw_card under the version's card parameters, the version's memory, PR.card_messages; key chat|run|g###|author|slot[|d j]|sha8.
        Cards use the common seed g<g>:<seed>, so the same source slot shows the same card to every author (paired cards)."""
        v = self.v_state()["versions"][author]
        params = self.eff_params(v)
        mem_info = {}
        kind = v.get("memory", "none")
        mem = None
        if kind != "none":
            orows = self.outcome_rows(author, cards=kind == "trace") if kind in ("trajectory", "trace") else []
            items = self.trace_items(author) if kind == "trace" else []
            mem = SE.memory_text(v, orows, items, self.rows, mem_info)
        if params["card_composition"] == "mixed":
            alloc = [(None, n_slots)] if n_slots > 0 else []
        else:
            alloc = sorted(self.alloc_sources(n_slots, self.residual(parent_show)).items())
        reqs, props, k = [], [], slot0
        seed = f"g{self.g}:{self.a.seed}"
        for b, n in alloc:
            for i in range(n):
                ids = PR.draw_card(self.rows, parent_show, params, b, i, seed)
                msgs, info = PR.card_messages(self.rows, parent_show, ids, tv, self.knobs, params, author_memory=mem)
                h = _sha16(json.dumps(msgs))[:8]
                base = f"chat|{self.run_name}|g{self.g:03d}|{author}|{k}"
                d = int(params["drafts"])
                keys = [f"{base}|{h}"] if d == 1 else [f"{base}|d{j}|{h}" for j in range(d)]
                for key in keys:
                    reqs.append(dict(key=key, mode="chat", engine="E", priority=0, messages=msgs, temperature=params["temperature"], max_tokens=PROP_MAX))
                p = dict(author=author, slot=k, key=keys[0], card=info["card_ids"], card_source=b, card_wrong=info["n_wrong"], card_right=info["n_right"],
                         prompt_chars=_chars(msgs), lab_turns=int(params["lab_turns"]))
                if d > 1:
                    p["draft_keys"] = keys
                if mem is not None:
                    p["memory"] = dict(mem_info.get("memory") or {}, chars=len(mem))
                if info.get("dropped"):
                    p["card_dropped"] = info["dropped"]
                props.append(p)
                k += 1
        return reqs, props

    def pick_drafts(self, props, tv):
        """After the answers: p["key"] = the first draft that parses and validates (PR.pick_draft), else the first draft's key;
        p["draft_pick"] = its index or the error. Slots with one draft are left as they are."""
        n = 0
        for p in props:
            ks = p.get("draft_keys")
            if not ks:
                continue
            texts = [(self.io.get(k) or {}).get("content") for k in ks]
            aw = PR.card_words(self.rows, p.get("card")) if p.get("card") else None
            idx, _, err = PR.pick_draft(texts, tv, self.knobs, self.rows, allowed_words=aw)
            p["key"] = ks[idx] if idx is not None else ks[0]
            p["draft_pick"] = idx if idx is not None else err[:300]
            n += idx is not None
        return n

    def v_step_propose(self):
        """gen.Run.step_propose with L1 authors: prompts of P0, the incumbent and the slotted challengers (one slot counter), B cells
        and the stack candidate as before, then the draft pick."""
        G = self.G()
        self.v_state()
        sl = G.setdefault("slots", self.slots())
        tip = self.state["tip"]["entries"]
        tv = self.tip_view()
        pf = os.path.join(self.rd, f"prompts_g{self.g:03d}.jsonl")
        if os.path.exists(pf) and "props_v" in G:                  # resume inside the step: the same prompts and keys as before
            reqs = [json.loads(l) for l in open(pf) if l.strip()]
            props, alloc = G["props_v"], G["l1_alloc"]["slots"]
        else:
            rng = np.random.default_rng(int(_sha16(f"alloc:{self.g}:{self.a.seed}"), 16))
            alloc = self.alloc_authors(sl["B"] + sl["P0"] if getattr(self, "b_in_draw", True) else sl["P0"], rng)
            parent_show = {r: self.samples(tip, r) for r in self.role["show"]}
            resid = self.residual(parent_show)
            G["sources"] = dict(residual={b: round(v, 2) for b, v in sorted(resid.items())},
                                alloc={a: self.alloc_sources(alloc.get(a, 0), resid) for a in self.proposer_authors()})
            reqs, props = [], []
            for a in self.proposer_authors():
                if alloc.get(a, 0) > 0:
                    q, p = self.build_prompts(a, alloc[a], parent_show, tv, slot0=len(props))
                    reqs += q
                    props += p
            with open(pf + ".tmp", "w") as f:
                for q in reqs:
                    f.write(json.dumps(q, ensure_ascii=False) + "\n")
            os.replace(pf + ".tmp", pf)
            G["props_v"] = props
            G["props_p0"] = [p for p in props if p["author"] == "P0"]
            G["l1_alloc"] = self._alloc_info
            self.save()
        n = self.io.submit(f"g{self.g:03d}_chat", reqs)
        props = [dict(p) for p in props]
        _log(f"  proposers: {len(reqs)} chat calls ({n} new) for {len(props)} slots: "
             + ", ".join(f"{a} {sum(p['author'] == a for p in props)}" for a in self.proposer_authors())
             + (f"; waiting challengers {G['l1_alloc']['waiting']}" if G["l1_alloc"].get("waiting") else ""))
        rng = random.Random(f"B:{self.g}:{self.a.seed}")
        fresh = self.b_cells()
        rng.shuffle(fresh)
        nb = min(int(alloc.get("B", 0)) if getattr(self, "b_in_draw", True) else int(sl["B"]), len(fresh))
        cap = self.b_cells_cap() if hasattr(self, "b_cells_cap") else None
        if cap is not None and nb > cap:                            # RUN/cards.json b_cells (coordinator 11-11 01:40): arm B's setting cells capped
            _log(f"  arm B: {nb} setting cells drawn, capped to {cap} (cards.json b_cells)")
            nb = cap
        G["l1_alloc"]["B_capped"] = nb
        for k in range(nb):                                         # each distinct one-factor cell at most once per tip
            knob, val = fresh.pop()
            props.append(dict(author="B", slot=k, cand=dict(why="blind one-factor mutation", condition="always", procedure=None, settings={knob: val})))
        if hasattr(self, "add_w_props"):                            # B2: author G first (gen.w_budget_after_g), then S2-C weight cells on W
            self.add_w_props(props, rng, self.w_budget_after_g(props, sl) if hasattr(self, "w_budget_after_g") else sl)
        st = self.stack_entry()
        if st is not None:
            props.append(dict(author="system", slot=0, stack=st))
            _log(f"  stack candidate (author system): {len(st['parts'])} accepted, unmerged entries on {self.state['tip']['id']}: "
                 + ", ".join(x["cid"] for x in st["parts"]))
        G["props"] = props
        done, miss = self.io.wait([q["key"] for q in reqs], self.deadline, label="proposers")
        if miss:
            _log(f"  proposers: {len(miss)} calls unanswered at the deadline")
        k = self.pick_drafts(props, tv)
        nd = sum(1 for p in props if p.get("draft_keys"))
        if nd:
            _log(f"  drafts: {k} of {nd} multi-draft slots have a valid draft")

    # ------------------------------------------------------------------ memory inputs
    @staticmethod
    def _rel(author, lin):
        return "you" if author in lin else ("blind" if author in ("B", "G") else "another proposer")   # G: "blind" (coordinator 11-11, B2)

    def _tip_entries(self, tid):
        if tid == self.state["tip"]["id"]:
            return self.state["tip"]["entries"]
        for t in self.state["tips"]:
            if t["id"] == tid:
                return t["entries"]
        return None

    def _route_of(self, e, r):
        if e["cond"] == "always":
            return True
        x = self.io.get(self.route_key(e["cond"], r))
        if x is None:
            return None
        p = x.get("p")
        return bool(p >= 0.5) if isinstance(p, (int, float)) else bool(x.get("yes"))

    def trace_card(self, c, prop):
        """[{id, routed, tip_right}] of the candidate's card (SHOW rows): routed from the router result, tip_right from the parent tip's
        cached samples (None when unknown)."""
        if not prop or not prop.get("card"):
            return None
        pe = self._tip_entries(c["parent"])
        out = []
        for rid in prop["card"]:
            assert self.rows[rid]["v4_role"] == "show", f"card row {rid} is not a SHOW row"
            tr = None
            if pe is not None:
                self.load_routes(pe, [rid])
                if all(e["cond"] == "always" or (e["cond"], rid) in self.routes for e in self.flat(pe)):
                    vs = [s for s in self.samples(pe, rid) if not s.get("error") and s.get("ok") is not None]
                    tr = PR._is_right(vs) if vs else None
            out.append(dict(id=rid, routed=self._route_of(c["entry"], rid), tip_right=tr))
        return out

    def outcome_rows(self, reader, cards=False):
        """Outcome table rows of earlier generations for a reader version: every validated proposer / blind candidate (stack candidates
        left out), author relative to the reader (you = its lineage), D and lfsr per routed row (final numbers where present)."""
        lin, out = self.lineage(reader), []
        for gk, G in sorted(self.state["gens"].items(), key=lambda x: int(x[0])):
            if int(gk) >= self.g:
                continue
            props = {(p["author"], p["slot"]): p for p in G.get("props", [])}
            for c in G.get("cands", []):
                e = c.get("entry")
                if c["author"] == "system" or not e or not c.get("valid0"):
                    continue
                j = c.get("final") or c.get("judge") or {}
                rel = self._rel(c["author"], lin)
                n = j.get("n") if j.get("n") is not None else c.get("n_read")
                row = dict(gen=int(gk), author=rel, kind=c.get("kind"), condition=e.get("condition"), procedure=e.get("procedure"),
                           settings=e.get("settings") or {}, ops=e.get("ops") or [], routed_share=c.get("share"), rows_read=n,
                           D=j.get("D"), lfsr=j.get("lfsr"), accepted=bool(c.get("accepted")) if c.get("decided") else None)
                if cards and rel == "you":
                    row["card"] = self.trace_card(c, props.get((c["author"], c["slot"])))
                out.append(row)
        return out

    def _cls(self, x, r):
        try:
            return ECOMP.answer_class(ECOMP.pred(x, self.rows[r]), self.rows[r])
        except Exception:
            return ("raw", str(x.get("pred")))

    def trace_items(self, reader):
        """With / without examples for trace memory: routed SHOW rows of the reader lineage's candidates (last WINDOW generations) where
        the candidate's sample-0 answer differs from the parent tip's sample 0 (a change in ok first, then in the answer class, newest
        first). A read that is not there is skipped (never waited for)."""
        lin, items = self.lineage(reader), []
        show = self.role["show"]
        for gk, G in sorted(self.state["gens"].items(), key=lambda x: -int(x[0])):
            g = int(gk)
            if g >= self.g or g < self.g - WINDOW:
                continue
            for c in G.get("cands", []):
                if c["author"] not in lin or not c.get("entry") or not c.get("valid0") or c["author"] == "system":
                    continue
                pe = self._tip_entries(c["parent"])
                if pe is None:
                    continue
                ents = pe + [c["entry"]]
                self.load_routes(ents, show)
                j = c.get("final") or c.get("judge") or {}
                for r in show:
                    if not self.fires(c["entry"], r):
                        continue
                    if any(e["cond"] != "always" and (e["cond"], r) not in self.routes for e in self.flat(pe)):
                        continue
                    assert self.rows[r]["v4_role"] == "show", f"trace row {r} is not a SHOW row"
                    x = self.io.get(self.read_req(ents, r, 0, 5)[0]["key"])
                    y = self.io.get(self.read_key(self.view(pe, r)[2], r, 0))
                    if not x or not y or x.get("error") or y.get("error") or x.get("op_skipped") or x.get("ok") is None or y.get("ok") is None:
                        continue
                    d_ok = bool(x["ok"]) != bool(y["ok"])
                    if not d_ok and self._cls(x, r) == self._cls(y, r):
                        continue
                    items.append((not d_ok, -g, c["cid"], r, dict(
                        condition=c["entry"].get("condition"), procedure=c["entry"].get("procedure"), settings=c["entry"].get("settings") or {},
                        ops=c["entry"].get("ops") or [], routed_share=c.get("share"), D=j.get("D"), lfsr=j.get("lfsr"),
                        accepted=bool(c.get("accepted")) if c.get("decided") else None, row_prompt=self.rows[r]["prompt"],
                        with_=None, with_ok=bool(x["ok"]), without_ok=bool(y["ok"]), _with=x.get("content") or "", _without=y.get("content") or "")))
        items.sort(key=lambda t: t[:4])
        out = []
        for t in items[:SE.TRACE_MAX]:
            d = t[4]
            d["with"], d["without"] = d.pop("_with"), d.pop("_without")
            d.pop("with_")
            out.append(d)
        return out

    # ------------------------------------------------------------------ step l1
    def selfedit_answer(self, key):
        return self.io.get(key)

    def l1_rule(self, p, inc, ch):
        """AM15 decision from P(Y_ch > Y_inc): switch at >= 1 - alpha, drop at <= alpha, else stay."""
        return ("switch" if p >= 1 - ALPHA else "drop" if p <= ALPHA else "stay"), p

    def _l1_wait(self, keys, label):
        dl = max(getattr(self, "deadline", time.time()) + GRACE, time.time() + L1_WAIT)
        return self.io.wait(keys, dl, label=label)

    def _accepted_ids(self):
        return [e["id"] for e in self.transfer_edits(self.v_state()["inc"])]

    def _live(self):
        st = self.v_state()
        return ["P0", st["inc"]] + list(st["challengers"]) + list(st["challenger_queue"])

    def _dup_of(self, v, skip=()):
        cv = canon(v)
        for a in self._live():
            if a not in skip and canon(self.state["versions"][a]) == cv:
                return a
        return None

    def new_challenger(self, edit, origin):
        """A challenger from an edit on the incumbent: queued (FIFO), window from the next generation. (id, None) or (None, reason)."""
        st = self.v_state()
        V, inc = st["versions"], st["inc"]
        probs = SE.validate_selfedit(edit, V[inc], PR.PARAMS_MENU, self._accepted_ids(), self.rows)
        if probs:
            return None, "invalid: " + "; ".join(probs)
        nid = f"P{st['v_next']}"
        nv = SE.apply_selfedit(V[inc], edit, new_id=nid)
        dup = self._dup_of(nv)
        if dup:
            return None, f"duplicate of {dup}"
        st["v_next"] += 1
        nv.update(born_gen=self.g, origin=origin, base=inc, window_start=self.g + 1, status="queued")
        V[nid] = nv
        st["challenger_queue"].append(nid)
        return nid, None

    def rebase(self, ch):
        """Challenger ch with its edit re-applied on the current incumbent (same id, window restarts next generation). None or the reason
        it is dropped (the edit no longer valid or a no-op, or a duplicate)."""
        st = self.v_state()
        V, inc = st["versions"], st["inc"]
        old = V[ch]
        probs = SE.validate_selfedit(old["edit"], V[inc], PR.PARAMS_MENU, self._accepted_ids(), self.rows)
        if probs:
            return "re-base: " + "; ".join(probs)
        nv = SE.apply_selfedit(V[inc], old["edit"], new_id=ch)
        dup = self._dup_of(nv, skip=(ch,))
        if dup:
            return f"re-base: duplicate of {dup}"
        nv.update({k: old[k] for k in ("born_gen", "origin") if k in old})
        nv.update(base=inc, window_start=self.g + 1, status=old.get("status"), rebased=list(old.get("rebased") or []) + [self.g])
        V[ch] = nv
        return None

    def _drop(self, ch, why):
        st = self.v_state()
        for lst in (st["challengers"], st["challenger_queue"]):
            if ch in lst:
                lst.remove(ch)
        st["versions"][ch]["status"] = f"dropped g{self.g}: {why}"

    def l1_decide(self):
        st = self.v_state()
        inc, out, on = st["inc"], {}, self.l1_on()
        for ch in list(st["challengers"]):
            common = [g for g in self._gens(ch, self._ws(ch), self.g) if self._has_slots(ch, g) and self._has_slots(inc, g)]
            if not common:
                out[ch] = dict(decision="stay", p=None, reason="no common generation with the incumbent yet")
                continue
            mi, si, ai = self.slot_data(inc, common)
            mc, sc, ac = self.slot_data(ch, common)
            rng = _rng("am15", self.g, self.a.seed, inc, ch)
            yi, yc = boot_y(mi, si, BOOT, rng), boot_y(mc, sc, BOOT, rng)
            p_boot = p_greater(yc, yi)
            p_perm = perm_better(mc, mi, BOOT, _rng("am15perm", self.g, self.a.seed, inc, ch))
            p = p_perm if self.l1_rule_name() == "am15perm" else p_boot
            d, p = self.l1_rule(p, inc, ch)
            out[ch] = dict(decision=d, p=round(float(p), 4), p_boot=round(p_boot, 4), p_perm=round(p_perm, 4), gens=common, slots_inc=len(mi), slots_ch=len(mc), acc_inc=int(ai), acc_ch=int(ac),
                           Y_inc=float(np.mean(mi)) if len(mi) else 0.0, Y_ch=float(np.mean(mc)) if len(mc) else 0.0)
            if not on and d != "stay":
                out[ch].update(decision="stay", would=d, report_only=True, note="l1 rule off: reported only")
        for ch, o in out.items():
            if o["decision"] == "drop":
                self._drop(ch, f"P(Y_ch > Y_inc) = {o['p']:.3f} <= alpha")
        sw = [ch for ch, o in out.items() if o["decision"] == "switch"]
        if sw:
            best = max(sw, key=lambda ch: (out[ch]["p"], -st["challengers"].index(ch)))
            old = inc
            st["challengers"].remove(best)
            st["inc"] = best
            V = st["versions"]
            V[best]["status"] = f"incumbent since g{self.g}"
            if old != "P0":
                V[old]["status"] = f"retired g{self.g} (switched to {best})"
            for ch in sw:
                if ch != best:
                    out[ch]["note"] = f"also above 1 - alpha; {best} had the larger p"
            for ch in list(st["challengers"]) + list(st["challenger_queue"]):
                why = self.rebase(ch)
                out.setdefault(ch, {})["rebased"] = why is None
                if why:
                    self._drop(ch, why)
                    out[ch]["dropped"] = why
            out["_switch"] = dict(old=old, new=best)
        return out

    def l1_retire(self):
        """AM15: if a challenger waits in the queue while the active list is full, the active challenger with the lowest
        P(Y_ch > Y_inc) of this generation's decisions is retired when that P < 1/2 (ties: the older one). Not in a generation with a
        switch (the remaining challengers were just re-based, their P was against the old incumbent) or with the rule off."""
        st = self.v_state()
        dec = self.G().get("l1", {}).get("decide") or {}
        base = dict(queue=list(st["challenger_queue"]), active=list(st["challengers"]))
        if not self.l1_on():
            return dict(base, retired=None, reason="l1 rule off")
        if "_switch" in dec:
            return dict(base, retired=None, reason="switch this generation")
        if not st["challenger_queue"] or len(st["challengers"]) < MAX_CH:
            return dict(base, retired=None, reason="queue not blocked")
        ps = {ch: dec[ch]["p"] for ch in st["challengers"] if isinstance(dec.get(ch), dict) and dec[ch].get("p") is not None}
        if not ps:
            return dict(base, retired=None, reason="no active challenger has a P yet")
        ch = min(ps, key=lambda c: (ps[c], st["challengers"].index(c)))
        if ps[ch] >= P_RETIRE:
            return dict(base, retired=None, reason=f"lowest P(better) {ps[ch]:.3f} ({ch}) >= {P_RETIRE}")
        self._drop(ch, "retired")
        st["versions"][ch]["status"] = f"retired g{self.g}: queue blocked, P(better than {st['inc']}) = {ps[ch]:.3f} < {P_RETIRE}"
        _log(f"  L1 retire: {ch} (P(better than {st['inc']}) = {ps[ch]:.3f}) leaves for the queue {base['queue']}")
        return dict(base, retired=ch, p=ps[ch])

    def l1_selfedit(self):
        st = self.v_state()
        inc = st["inc"]
        V = st["versions"]
        f = os.path.join(self.rd, f"prompts_l1_g{self.g:03d}.jsonl")
        recs = {}
        if os.path.exists(f):
            for l in open(f):
                if l.strip():
                    q = json.loads(l)
                    recs[q["tag"]] = q

        def store(tag, q):
            recs[tag] = dict(q, tag=tag)
            with open(f + ".tmp", "w") as fh:
                for t in sorted(recs):
                    fh.write(json.dumps(recs[t], ensure_ascii=False) + "\n")
            os.replace(f + ".tmp", f)

        edits = self.transfer_edits(inc)
        orows = self.outcome_rows(inc)
        if "selfedit" not in recs:
            msgs = SE.selfedit_messages(V[inc], orows, PR.PARAMS_MENU, edits, self.rows)
            key = f"chat|{self.run_name}|g{self.g:03d}|selfedit|{inc}|{_sha16(json.dumps(msgs))[:8]}"
            store("selfedit", dict(key=key, mode="chat", engine="E", priority=0, messages=msgs, temperature=PROP_TEMP, max_tokens=PROP_MAX))
        q = {k: v for k, v in recs["selfedit"].items() if k != "tag"}
        self.io.submit(f"g{self.g:03d}_selfedit", [q])
        self._l1_wait([q["key"]], "self-edit")
        rec = dict(author=inc, key=q["key"], prompt_chars=_chars(q["messages"]), table_rows=len(orows), transferable=len(edits))
        x = self.selfedit_answer(q["key"])
        edit, err = self._parse_edit(x, V[inc])
        rec["first_error"] = err if err is None or isinstance(err, str) else "; ".join(err)
        if err is not None and x is not None and _format_only(err):
            if "fix1" not in recs:
                m = FIX_MSG.format(err=(err if isinstance(err, str) else "; ".join(err))[:300])
                store("fix1", dict(q, key=q["key"] + "|fix1", messages=q["messages"] + [dict(role="assistant", content=x.get("content") or ""), dict(role="user", content=m)]))
            q2 = {k: v for k, v in recs["fix1"].items() if k != "tag"}
            self.io.submit(f"g{self.g:03d}_selfeditfix", [q2])
            self._l1_wait([q2["key"]], "self-edit repair")
            edit, err = self._parse_edit(self.selfedit_answer(q2["key"]), V[inc])
            rec["repair_key"] = q2["key"]
            rec["repaired"] = err is None
        if x is None:
            err = "unanswered"
        if err is not None:
            rec.update(ok=False, error=err if isinstance(err, str) else "invalid: " + "; ".join(err), challenger=None)
            _log(f"  L1 self-edit of {inc}: no challenger ({rec['error'][:160]})")
            return rec
        nid, why = self.new_challenger(edit, "self")
        rec.update(ok=nid is not None, edit={k: v for k, v in edit.items() if k != "extra"}, challenger=nid, error=why)
        _log(f"  L1 self-edit of {inc}: " + (f"challenger {nid} = {json.dumps(rec['edit'], ensure_ascii=False)[:200]}" if nid else f"no challenger ({why})"))
        return rec

    def _parse_edit(self, x, v):
        if x is None:
            return None, "unanswered"
        edit, err = SE.parse_selfedit(x.get("content") or "")
        if err:
            return None, err
        probs = SE.validate_selfedit(edit, v, PR.PARAMS_MENU, self._accepted_ids(), self.rows)
        return (edit, None) if not probs else (edit, probs)

    def blind_options(self, v):
        """Every one-step edit of v (AM12): memory none / trajectory / trace, every other value of every card parameter, every
        transferable accepted edit not yet transferred. Lessons are left out (a blind lessons text has no content)."""
        out = [dict(type="memory", memory=m) for m in ("none", "trajectory", "trace") if m != v.get("memory", "none")]
        p = PR.params_of(v.get("params"))
        for name in PR.PARAMS_MENU:
            for val in PR._menu_values(PR.PARAMS_MENU, name):
                if not PR._in_menu(val, [p[name]]):
                    out.append(dict(type="param", param=name, value=val))
        have = {str(t) for t in v.get("transfers") or []}
        out += [dict(type="transfer", edit=e["id"]) for e in self.transfer_edits(v.get("id")) if e["id"] not in have]
        return [dict(o, why="blind self-mutation") for o in out]

    def l1_blind(self):
        st = self.v_state()
        inc = st["inc"]
        opts = self.blind_options(st["versions"][inc])
        rng = random.Random(f"blind:{self.g}:{self.a.seed}")
        tried = []
        while opts:
            e = opts.pop(rng.randrange(len(opts)))
            nid, why = self.new_challenger(e, "blind")
            tried.append(why)
            if nid:
                _log(f"  L1 blind self-mutation of {inc}: challenger {nid} = {json.dumps({k: v for k, v in e.items() if k != 'why'})}")
                return dict(author=inc, edit=e, challenger=nid, redraws=len(tried) - 1)
        return dict(author=inc, edit=None, challenger=None, error="no option left", redraws=len(tried))

    def l1_admit(self):
        st = self.v_state()
        adm = []
        while st["challenger_queue"] and len(st["challengers"]) < MAX_CH:
            ch = st["challenger_queue"].pop(0)
            if st["versions"][ch].get("base") != st["inc"]:
                why = self.rebase(ch)
                if why:
                    st["challenger_queue"].insert(0, ch)
                    self._drop(ch, why)
                    continue
            st["versions"][ch]["window_start"] = self.g + 1
            st["versions"][ch]["status"] = f"active since g{self.g + 1}"
            st["challengers"].append(ch)
            adm.append(ch)
        return adm

    def l1_report(self):
        """AM15 report per author with slots this generation (and B, P0): slots this generation, window slots and accepts, Y mean / sd
        (bootstrap) and the plug-in Y, P(Y > Y_B), P(Y > Y_P0) over the last WINDOW generations (each author from its window_start);
        self- vs blind-chosen challengers pooled over all their slots (each from its window_start)."""
        G = self.G()
        slots = {}
        for p in G.get("props", []):
            if p["author"] != "system":
                slots[p["author"]] = slots.get(p["author"], 0) + 1
        authors = sorted(set(slots) | {"B", "P0"}, key=lambda a: (a not in ("B", "P0"), a))
        lo = max(1, self.g - WINDOW + 1)
        out = dict(rule=self.l1_rule_name(), on=self.l1_on(), window=[lo, self.g], authors={})
        Y, Mw = {}, {}
        for a in authors:
            m, s_, na = self.slot_data(a, self._gens(a, lo, self.g))
            Mw[a] = m
            Y[a] = boot_y(m, s_, BOOT, _rng("am15rep", self.g, self.a.seed, a))
            out["authors"][a] = dict(slots=slots.get(a, 0), window_slots=len(m), accepted=int(na), Y_mean=float(Y[a].mean()), Y_sd=float(Y[a].std()),
                                     Y_point=float(np.mean(m)) if len(m) else 0.0)
        inc = self.v_state()["inc"]
        for a in authors:
            for ref in ("B", "P0"):
                if a != ref:
                    out["authors"][a][f"P_gt_{ref}"] = round(p_greater(Y[a], Y[ref]), 4)
            for lab_, ref in (("inc", inc), ("P0", "P0")):              # coordinator 11-10: permutation p (one-sided mid-p) of each author
                if a != ref and ref in Mw and len(Mw[a]) and len(Mw[ref]):   # against the incumbent and against P0, over the report window
                    pb = perm_better(Mw[a], Mw[ref], BOOT, _rng("am15permrep", self.g, self.a.seed, a, ref))
                    out["authors"][a][f"perm_p_vs_{lab_}"] = round(1.0 - float(pb), 4)
        out["inc"] = inc
        V = self.v_state()["versions"]
        og = {}
        for o in ("self", "blind"):
            M, S_, NA = [], [], 0
            for vid, v in V.items():
                if v.get("origin") == o:
                    m, s_, na = self.slot_data(vid, self._gens(vid, 1, self.g))
                    M += list(m)
                    S_ += list(s_)
                    NA += na
            if M:
                og[o] = (np.array(M), np.array(S_), NA)
        if og:
            yo = {o: boot_y(m, s_, BOOT, _rng("am15origin", self.g, self.a.seed, o)) for o, (m, s_, _) in og.items()}
            out["origin"] = {o: dict(slots=len(og[o][0]), accepted=int(og[o][2]), Y_mean=float(yo[o].mean()), Y_sd=float(yo[o].std())) for o in og}
            if len(og) == 2:
                out["origin"]["P_self_gt_blind"] = round(p_greater(yo["self"], yo["blind"]), 4)
        return out

    def v_step_l1(self):
        """Step "l1" (after "report"): report (Y of this generation's authors on the windows that held during it), decisions (AM15),
        self-edit, blind mutation, retire (blocked queue), admission from the queue. Sub-steps are marked in
        G["l1"]["done"], so a restart inside the step does not repeat a finished part."""
        G = self.G()
        st = self.v_state()
        L = G.setdefault("l1", dict(done=[]))
        for part, fn in (("report", self.l1_report), ("decide", self.l1_decide), ("selfedit", self.l1_selfedit), ("blind", self.l1_blind),
                         ("retire", self.l1_retire), ("admit", self.l1_admit)):
            if part in L["done"]:
                continue
            L[part] = fn()
            L["done"].append(part)
            self.save()
        L["inc"], L["challengers"], L["queue"] = st["inc"], list(st["challengers"]), list(st["challenger_queue"])
        st["authors"] = ["B"] + self.proposer_authors()
        d = {k: v.get("decision") for k, v in (L.get("decide") or {}).items() if not k.startswith("_")}
        _log(f"  L1: incumbent {st['inc']}, decisions {d}" + (f", SWITCH {L['decide']['_switch']}" if "_switch" in (L.get("decide") or {}) else "")
             + (f", retired {L['retire']['retired']}" if (L.get("retire") or {}).get("retired") else "")
             + f"; active {st['challengers']}, queue {st['challenger_queue']}; rule {self.l1_rule_name() if self.l1_on() else 'off (report only)'}; Y " + ", ".join(
                 f"{a} {o['Y_mean']:+.4f} (slots {o['window_slots']}, acc {o['accepted']})" for a, o in L["report"]["authors"].items()))
