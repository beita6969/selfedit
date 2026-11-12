"""selfedit v5 judge (design_1111_v5.md section 4 step 3). JUDGE rows only; the benchmark is the unit of evidence.

For a candidate c (one change to one object of the tip):
  system value  v_c(r) = expected ok of tip+c on JUDGE row r (mean over its read samples; a policy: exact expectation over cached samples;
                a routed entry: the tip's value on rows where it does not fire), tokens t_c(r) likewise (generated tokens).
  reference     the tip's OWN frontier at the candidate's tokens: m = mean_r t_c(r) / mean_r t_tip(r); ref(r) = Frontier.ref_at(m * base)
                (policy5: hull of the tip-only read policies from the tip's cached multi-sample reads; the mixture of the two adjacent
                vertices with exactly the candidate's tokens; strict = the next vertex above). m <= 1 -> the tip's single-read mean.
                A candidate measured against ANOTHER cached system (a removal: tip vs tip-without-entry, read with few samples) uses
                base= that system and the tip's frontier INCREMENT: d(r) = v_c(r) - v_base(r) - [F(m)(r) - F(1)(r)].
  d_b           mean over the JUDGE rows of benchmark b of v_c(r) - ref(r), cluster-robust se over picture clusters.
  S(c)          scored benchmarks = all benchmarks minus the candidate's source benchmark(s) (P1).
  accept iff    (0) |S| >= MIN_SCORED (3)
                (i)+(ii) p_gen = max_b p_macro(S - b) passes Holm at ALPHA over the family (the candidates judged in the generation);
                     p-values by sign-flip of picture clusters inside benchmarks (the t forms are printed beside them)
                (iii) no scored benchmark significantly negative (one-sided sign-flip p < ALPHA_NEG / |S|)
                (iv) routed entries: requests on rows where the entry does not fire are the tip's own (structural, flag unrouted_identical);
                     otherwise the unrouted rows are in d_b like any row
                (v) weight cells: the random-edit null audit. null arm vs the tip's cache at ITS tokens; when the audit fails (two-sided
                    p < 0.05: an inert edit is not a fresh tip sample on this engine) the cell is judged against the null arm instead.
  Entry audit   (removals, their own family): the entry as a candidate on top of the tip without it. GENERAL = accept on S (off source);
                ID_SPECIFIC = not general but the pooled effect over ALL benchmarks (source included) has p_macro <= ALPHA (Holm in the
                audit family) -> specialisation layer, OFF in the general system; UNPROVEN = neither (no effect shown even with the
                source benchmark) -> out of the general system, specialisation layer at most.
Nothing here reads a row whose role is not judge (rows are checked).
"""
import json
import math

import numpy as np
from scipy.stats import t as _t

import policy5 as P
import stats5 as ST

ALPHA = 0.10
ALPHA_NEG = 0.10
MIN_SCORED = 3
AUDIT_P = 0.05
NFLIP = 20000            # sign flips of the gating p-values (resolution 5e-5; a Holm threshold of 0.10 / 12 has a Monte-Carlo se of 6e-4)
SEED = 20261111


class JudgeError(ValueError):
    pass


def arm_values(arm, row_ids):
    """per-row mean ok, mean generated tokens, mean total tokens (generated + prompt), samples of a read arm on row_ids (rows without
    reads are left out)."""
    v, t, T, n = {}, {}, {}, {}
    for r in row_ids:
        xs = arm.get(r)
        if xs:
            v[r] = sum(x["ok"] for x in xs) / len(xs)
            t[r] = sum(x["tokens"] for x in xs) / len(xs)
            T[r] = sum(x["tokens"] + float(x.get("prompt_tokens") or 0.0) for x in xs) / len(xs)
            n[r] = len(xs)
    return v, t, T, n


class Judge:
    def __init__(self, rows, tip_arm, benches=None, alpha=ALPHA, alpha_neg=ALPHA_NEG, strict=False, hull_from=None, policies=None, rng=None,
                 frontier=None, roles=("judge",), hull_from_total=None):
        """roles: ("judge",) for the judge. The SAME computation on ("show", "monitor") is the signal-row pre-screen (never a decision);
        self.J is the row set in both cases and the two sets are never mixed.
        hull_from / hull_from_total: the vertex NAMES of the tip's frontier as chosen on the SIGNAL rows (generated / total tokens); the
        judge then only VALUES those policies on the JUDGE rows, so nothing about the reference is selected on JUDGE rows. None: the hull
        of this row set itself (slightly conservative for candidates: stage-1 A/A part B, mean d -0.03 to -0.06 pp)."""
        self.rows = rows
        self.roles = tuple(roles)
        if "judge" in self.roles and len(self.roles) > 1:
            raise JudgeError("JUDGE rows are never pooled with SHOW / MONITOR rows")
        self.J = sorted(r for r, x in rows.items() if x["role"] in self.roles)
        miss = [r for r in self.J if not tip_arm.get(r)]
        if miss:
            raise JudgeError(f"the tip has no cached reads on {len(miss)} of the {'/'.join(self.roles)} rows")
        self.benches = list(benches) if benches is not None else sorted({rows[r]["bench"] for r in self.J})
        self.alpha, self.alpha_neg, self.strict = alpha, alpha_neg, strict
        self.tip = {r: tip_arm[r] for r in self.J}
        self.F = frontier if frontier is not None else P.Frontier(self.tip, self.J, rows, policies=policies, rng=rng, hull_from=hull_from)
        # the same policies on the total-token axis (generated + prompt tokens of every read): printed beside the gating convention
        self.Ft = P.Frontier(self.tip, self.J, rows, policies=list(self.F.pols.values()), cost="total", table=self.F.table, hull_from=hull_from_total)
        self.tip_v = {r: self.F.table["vote1"][r]["ok"] for r in self.J}
        self.tip_t = {r: self.F.table["vote1"][r]["tokens"] for r in self.J}
        self.tip_T = {r: self.F.table["vote1"][r]["total"] for r in self.J}
        self.base_tokens = sum(self.tip_t.values()) / len(self.J)
        self.base_total = sum(self.tip_T.values()) / len(self.J)

    # ---------------------------------------------------------------------------------------- candidate forms
    def _check_rows(self, d, what):
        bad = [r for r in d if r not in self.tip]
        if bad:
            raise JudgeError(f"{what}: {len(bad)} rows are not {'/'.join(self.roles)} rows (e.g. {bad[0]})")

    def cand_from_arm(self, name, arm, kind, source=(), routed=None, **kw):
        """A candidate read as an arm (cell, operator, setting, entry). routed: rows where it applies (None = every JUDGE row); on other
        rows the system is the tip."""
        rows = self.J if routed is None else [r for r in routed if r in self.tip]
        v, t, T, n = arm_values(arm, rows)
        miss = [r for r in rows if r not in v]
        if miss:
            raise JudgeError(f"{name}: no reads on {len(miss)} of its {len(rows)} JUDGE rows")
        return dict(name=name, kind=kind, source=list(source), value=v, tokens=t, total=T, samples=n, routed=None if routed is None else sorted(rows), **kw)

    def system_of_arm(self, arm, routed=None):
        """value / tokens / total of a cached system read as an arm (for base= and null=)."""
        rows = self.J if routed is None else [r for r in routed if r in self.tip]
        v, t, T, _ = arm_values(arm, rows)
        return dict(value=v, tokens=t, total=T)

    def cand_from_policy(self, pol, extra_arm=None, kind="policy", source=(), **kw):
        tb = P.evaluate(pol, self.tip, self.J, extra_arm=extra_arm)
        return dict(name=kw.pop("name", P.name_of(pol)), kind=kind, source=list(source), value={r: x["ok"] for r, x in tb.items()},
                    tokens={r: x["tokens"] for r, x in tb.items()}, total={r: x["total"] for r, x in tb.items()},
                    esc=sum(x["esc"] for x in tb.values()) / len(tb), policy=pol, routed=None, **kw)

    def cand_from_views(self, name, pol, arms, kind="policy", source=(), **kw):
        """A views policy (policy5.views_policy: reads under the tip's view and under cached settings variants) as a candidate. arms =
        {view name: arm}; the view "tip" is the judge's own tip reads unless given."""
        A = {"tip": self.tip}
        A.update({v: {r: a[r] for r in self.J if r in a} for v, a in arms.items()})
        tb = P.evaluate_views(pol, A, self.J)
        return dict(name=name, kind=kind, source=list(source), value={r: x["ok"] for r, x in tb.items()}, tokens={r: x["tokens"] for r, x in tb.items()},
                    total={r: x["total"] for r, x in tb.items()}, esc=sum(x["esc"] for x in tb.values()) / len(tb), policy=pol, routed=None, **kw)

    # ---------------------------------------------------------------------------------------- scoring
    def system(self, c, cost="tokens"):
        """full-row value / cost of tip+c (the tip on rows the candidate does not touch). cost: "tokens" (generated) or "total"."""
        self._check_rows(c["value"], c["name"])
        tip_c = self.tip_t if cost == "tokens" else self.tip_T
        v = {r: c["value"].get(r, self.tip_v[r]) for r in self.J}
        cc = c.get(cost) or {}
        if cost == "total" and not cc:
            return v, None
        t = {r: cc.get(r, tip_c[r]) for r in self.J}
        return v, t

    def _ref(self, m, cost="tokens"):
        """per-row frontier value at multiple m of the tip's cost (m <= 1: the single read)."""
        F = self.F if cost == "tokens" else self.Ft
        if m <= 1.0 + 1e-9:
            return dict(self.tip_v), dict(lo="vote1", hi="vote1", w=0.0, multiple=1.0, strict=self.strict, tokens=F.base_tokens, acc=F.vertices[0][1])
        ref = F.ref_at(m * F.base_tokens, strict=self.strict)
        if ref is None:
            return None, None
        return ref[0], ref[2]

    def diffs(self, c, cost="tokens"):
        """-> (x {row: d}, info). x covers every JUDGE row. cost: the token convention of the equal-compute reference."""
        v, t = self.system(c, cost)
        if t is None:
            return None, dict(multiple=None, error="no total tokens for this candidate")
        base_cost = self.base_tokens if cost == "tokens" else self.base_total
        tip_c = self.tip_t if cost == "tokens" else self.tip_T
        Tc = sum(t.values()) / len(self.J)
        if c.get("base") is not None:                         # against another cached system: the tip's frontier increment
            bv = {r: c["base"]["value"].get(r, self.tip_v[r]) for r in self.J}
            bc = c["base"].get(cost) or {}
            bt = {r: bc.get(r, tip_c[r]) for r in self.J}
            Tb = sum(bt.values()) / len(self.J)
            m = Tc / Tb
            ref, desc = self._ref(m, cost)
            if ref is None:
                return None, dict(multiple=m, error="tokens above the tip's frontier: more tip samples needed")
            x = {r: v[r] - bv[r] - (ref[r] - self.tip_v[r]) for r in self.J}
            raw = {r: v[r] - bv[r] for r in self.J}
        else:
            m = Tc / base_cost
            ref, desc = self._ref(m, cost)
            if ref is None:
                return None, dict(multiple=m, error="tokens above the tip's frontier: more tip samples needed")
            x = {r: v[r] - ref[r] for r in self.J}
            raw = {r: v[r] - self.tip_v[r] for r in self.J}
        return x, dict(multiple=m, ref=desc, raw=raw, tokens=Tc)

    def _block(self, x, S):
        """effects and tests of one difference vector over the scored benchmarks S. Gating p-values = cluster sign-flip (stats5.Design
        .signflip, NFLIP flips, fixed seed); the t forms are kept beside them (p_macro_t, p_gen_t)."""
        if getattr(self, "_D", None) is None:
            self._D = ST.Design(self.J, self.rows, self.benches)
        E = ST.bench_effects(x, self.rows, floor_rows=0.0)
        Et = ST.bench_effects(x, self.rows)
        m = ST.macro(E, S)
        sf = self._D.signflip([x[r] for r in self.J], S, n=NFLIP, seed=SEED)
        pgt, _ = ST.p_gen(Et, S)
        neg = [(b, p) for b, p in zip(S, sf["p_neg"]) if p < self.alpha_neg / len(S)]
        return dict(d=m["d"], se=m["se"], p_macro=sf["p_macro"], p_gen=sf["p_gen"], carrier=(S[sf["carrier"]] if sf["carrier"] is not None and len(S) >= 3 else None),
                    p_macro_t=ST.p_macro(Et, S), p_gen_t=pgt, lobo_min=ST.lobo_point(E, S),
                    p_bench_signflip=ST.p_bench_signflip(E, S), p_bench_t=ST.p_bench_t(E, S), neg=neg,
                    npos=sum(E[b]["d"] > 0 for b in S), k=len(S), per_bench={b: dict(d=E[b]["d"], se=E[b]["se_raw"], n=E[b]["n"]) for b in self.benches})

    def score(self, c):
        S = [b for b in self.benches if b not in set(c.get("source") or ())]
        x, info = self.diffs(c)
        rec = dict(name=c["name"], kind=c["kind"], source=list(c.get("source") or ()), scored=S, multiple=info.get("multiple"),
                   routed_rows=None if c.get("routed") is None else len(c["routed"]), esc=c.get("esc"), policy=c.get("policy"), note=c.get("note"))
        if x is None:
            rec.update(error=info["error"], judged=False)
            return rec
        rec.update(judged=True, ref=info["ref"], tokens=info["tokens"], eq=self._block(x, S), raw=self._block(info["raw"], S),
                   pooled=self._block(x, self.benches))
        rec["acc"] = sum(self.system(c)[0].values()) / len(self.J)
        xt, tinfo = self.diffs(c, "total")                   # the other token convention (printed, never gating)
        if xt is not None:
            bt = self._block(xt, S)
            rec["total"] = dict(multiple=tinfo["multiple"], ref=tinfo["ref"], tokens=tinfo["tokens"], d=bt["d"], se=bt["se"], p_macro=bt["p_macro"],
                                p_gen=bt["p_gen"], per_bench=bt["per_bench"])
        else:
            rec["total"] = dict(multiple=tinfo.get("multiple"), error=tinfo.get("error"))
        if c.get("null") is not None:                         # (v) random-edit null audit for weight cells
            nv = dict(name=c["name"] + "|null", kind="null", source=[], value=c["null"]["value"], tokens=c["null"]["tokens"],
                      total=c["null"].get("total"), routed=None)
            xn, ninfo = self.diffs(nv)
            if xn is not None:
                nb = self._block(xn, self.benches)
                p2 = float(min(1.0, 2 * min(nb["p_macro"], 1 - nb["p_macro"] + 1.0 / (NFLIP + 1))))
                xd = {r: x[r] - xn[r] for r in self.J}
                rec["null"] = dict(d=nb["d"], se=nb["se"], p_two_sided=p2, multiple=ninfo["multiple"], audit_ok=bool(p2 >= AUDIT_P),
                                   vs_null=self._block(xd, S))
        return rec

    def source_effect(self, c, benches, rows=None):
        """The effect of a candidate on given benchmark(s) ONLY, for the admission of an in-distribution entry to the specialisation layer
        (coordinator 11-11): mean over the rows of those benchmarks (optionally only `rows`, e.g. the MONITOR rows) of candidate minus the
        reference at equal compute (rows where the entry does not fire count 0), cluster-robust se, one-sided sign-flip p of picture
        clusters. Never a general verdict: the scored set is the entry's own source benchmark."""
        x, info = self.diffs(c)
        if x is None:
            return dict(judged=False, error=info.get("error"))
        rs = [r for r in self.J if self.rows[r]["bench"] in set(benches) and (rows is None or r in rows)]
        if not rs:
            return dict(judged=False, error="no row of the source benchmark")
        xs = {r: x[r] for r in rs}
        E = ST.bench_effects(xs, self.rows, floor_rows=0.0)
        bs = sorted(E)
        m = ST.macro(E, bs)
        sf = ST.Design(rs, self.rows, bs).signflip([xs[r] for r in rs], n=NFLIP, seed=SEED)
        raw = sum(info["raw"][r] for r in rs) / len(rs)
        return dict(judged=True, d=m["d"], se=m["se"], p=sf["p_macro"], raw=raw, rows=len(rs), routed=sum(1 for r in rs if r in (c.get("routed") or rs)),
                    multiple=info["multiple"], benches=bs)

    # ---------------------------------------------------------------------------------------- decisions
    def decide(self, cands, family="generation"):
        """Score and decide a family of candidates. -> list of records with accept and reason."""
        recs = [self.score(c) for c in cands]
        live = [r for r in recs if r.get("judged") and r["eq"]["k"] >= MIN_SCORED]
        for r in live:                                         # the block that gates: the null arm when its audit fails
            r["gate"] = "vs_null" if (r.get("null") and not r["null"]["audit_ok"]) else "eq"
            r["p"] = (r["null"]["vs_null"] if r["gate"] == "vs_null" else r["eq"])["p_gen"]
        acc = ST.holm([r["p"] for r in live], self.alpha) if live else []
        order = np.argsort([r["p"] for r in live], kind="mergesort") if live else []
        K = len(live)
        thr = {}
        for j, i in enumerate(order):
            thr[live[i]["name"]] = self.alpha / (K - j)
        for r in recs:
            r["family"], r["K"] = family, K
            if not r.get("judged"):
                r.update(accept=False, reason=r.get("error", "not judged"))
                continue
            if r["eq"]["k"] < MIN_SCORED:
                r.update(accept=False, reason=f"only {r['eq']['k']} scored benchmarks (< {MIN_SCORED})")
                continue
        for r, a in zip(live, acc):
            g = r["null"]["vs_null"] if r["gate"] == "vs_null" else r["eq"]
            r["holm_threshold"] = thr[r["name"]]
            if not a:
                why = (f"p_gen {r['p']:.4f} > Holm {thr[r['name']]:.4f} (K {K})"
                       + (f"; macro {100 * g['d']:+.2f} pp at equal compute" if g["p_macro"] > self.alpha else f"; carried by {g['carrier']}"))
                r.update(accept=False, reason=why)
            elif g["neg"]:
                r.update(accept=False, reason="significantly negative on " + ", ".join(b for b, _ in g["neg"]))
            else:
                r.update(accept=True, reason=f"p_gen {r['p']:.4f} <= Holm {thr[r['name']]:.4f} (K {K}), no negative benchmark"
                         + ("; judged against the random-edit null (audit failed)" if r["gate"] == "vs_null" else ""))
        return recs

    def audit_entries(self, cands):
        """Entry audit (removals). cands: entries as candidates on top of the tip without them (base = the leave-one-out system).
        -> records with layer in GENERAL / ID_SPECIFIC / UNPROVEN."""
        recs = self.decide(cands, family="entry audit")
        live = [r for r in recs if r.get("judged")]
        pooled = ST.holm([r["pooled"]["p_macro"] for r in live], self.alpha) if live else []
        for r, a in zip(live, pooled):
            if r["accept"]:
                r["layer"] = "GENERAL"
            elif a:
                r["layer"] = "ID_SPECIFIC"
                r["reason"] += f" | pooled over all benchmarks {100 * r['pooled']['d']:+.2f} pp (p {r['pooled']['p_macro']:.4f}): in-distribution only -> specialisation layer"
            else:
                r["layer"] = "UNPROVEN"
                r["reason"] += f" | pooled over all benchmarks {100 * r['pooled']['d']:+.2f} pp (p {r['pooled']['p_macro']:.4f}, not significant in the audit family): removal costs nothing measurable"
        for r in recs:
            r.setdefault("layer", "NOT_JUDGED")
        return recs


# ------------------------------------------------------------------------------------------------ printing
def short(b):
    return b[:5]


def ref_text(d):
    if not d:
        return "-"
    if d["lo"] == d["hi"] or d["w"] in (0.0, 1.0):
        return (f"{d['hi'] if d['w'] == 1.0 else d['lo']} (x{d['multiple']:.2f}, acc {100 * d['acc']:.2f}"
                + ("; the frontier is flat above its last vertex" if d.get("flat") else "") + ")")
    return f"{1 - d['w']:.2f} {d['lo']} + {d['w']:.2f} {d['hi']} (x{d['multiple']:.2f}, acc {100 * d['acc']:.2f})"


def line(r, benches):
    """One verdict: head line (gating convention = generated tokens), per-benchmark effects with the scored set (* = not scored: the
    source benchmark), the reference used and its tokens, then the same under total tokens (generated + prompt)."""
    if not r.get("judged"):
        return f"{r['name']:34s} {r['kind']:8s} NOT JUDGED: {r.get('error')}"
    e, raw = r["eq"], r["raw"]
    pb = " ".join(f"{short(b)} {100 * e['per_bench'][b]['d']:+5.1f}+-{100 * e['per_bench'][b]['se']:3.1f}{'' if b in r['scored'] else '*'}" for b in benches)
    s = (f"{r['name']:34s} {r['kind']:8s} acc {100 * r['acc']:5.2f} | {'ACCEPT' if r.get('accept') else 'reject'}" + (f" [{r['layer']}]" if r.get("layer") else "")
         + f": {r.get('reason')}\n"
         f"    generated tokens x{r['multiple']:4.2f} | reference {ref_text(r['ref'])} | raw vs 1 tip read {100 * raw['d']:+5.2f} | at equal compute {100 * e['d']:+5.2f} +- {100 * e['se']:4.2f}"
         f" p_macro {e['p_macro']:.4f} p_gen {e['p_gen']:.4f} (worst without {short(e['carrier']) if e['carrier'] else '-'}; t forms {e['p_macro_t']:.4f} / {e['p_gen_t']:.4f}) | benchmark level: t p {e['p_bench_t']:.3f},"
         f" sign-flip p {e['p_bench_signflip']:.3f}, positive {e['npos']}/{e['k']}\n"
         f"    per benchmark (scored {len(r['scored'])}: {' '.join(short(b) for b in r['scored'])}{'; * = source, not scored' if len(r['scored']) < len(benches) else ''}): {pb}")
    t = r.get("total") or {}
    if t.get("d") is not None:
        s += (f"\n    total tokens with prompts x{t['multiple']:4.2f} | reference {ref_text(t['ref'])} | at equal compute {100 * t['d']:+5.2f} +- {100 * t['se']:4.2f}"
              f" p_macro {t['p_macro']:.4f} p_gen {t['p_gen']:.4f} (printed, not gating)")
    if r.get("null"):
        n = r["null"]
        s += (f"\n    random-edit null: null vs frontier {100 * n['d']:+.2f} +- {100 * n['se']:.2f} at x{n['multiple']:.2f} (audit {'ok' if n['audit_ok'] else 'FAILED'});"
              f" cell - null at equal compute {100 * n['vs_null']['d']:+.2f} +- {100 * n['vs_null']['se']:.2f} p_gen {n['vs_null']['p_gen']:.4f}")
    return s


def dump(recs, path):
    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
            return None
        return o
    json.dump(clean(recs), open(path, "w"), indent=1)
