"""selfedit v5, read policies evaluated OFFLINE from cached samples, and the tip's token-accuracy frontier (design sections 3, 4; P4).

A read policy is a small program over reads (label-free: it sees answer classes and declared-guess flags, never ok):
  dict(n0=1|2|3, trigger="none"|"disagree"|"guess"|"either", extra="tip"|<arm name>, n_extra=1, final="plur"|"extra")
    n0       initial reads of the tip
    trigger  when ONE escalation happens: the initial reads do not all agree ("disagree", n0 >= 2), any initial read ends in a declared
             guess ("guess"), or either
    extra    what the escalation reads: another tip read, or a read of another arm (a cell, an operator)
    n_extra  how many extra reads (default 1; plain votes of m are n0 = m, trigger none)
    final    "plur": plurality over all reads taken, ties to the earliest read; "extra": when escalated, the first extra read that has an
             answer is final (no answer -> plurality)
Its value on a row is the EXACT expectation over orderings of the row's cached samples (every ordered tuple of distinct tip samples x
every tuple of extra samples; sampled without replacement), so no draw order matters: (E ok, E generated tokens, P escalate).
Answer classes: cls(read) = read["cls"] when the export carries the grader's class, else the normalised pred (numbers by value, text
lower-cased); a read without an answer is the class "none", which is wrong.

Frontier (the equal-compute reference of the judge): the upper concave hull over (mean tokens, mean accuracy) of the tip-only policies
(TIP_POLICIES: votes of 1..M, early-stopping votes, guess / either triggers). ref_at(T) = the hull at T tokens = the mixture of the two
adjacent vertices whose expected tokens equal T (a randomised tip policy with tokens >= the candidate's, exactly equal); strict=True takes the
next vertex at or above T instead (more tokens than the candidate: conservative). T above the last vertex cannot be referenced (None).
"""
import itertools
import math

NONE = ("none",)


def cls_of(x):
    if x.get("cls") is not None:
        return x["cls"]
    p = x.get("pred")
    if p is None:
        return NONE
    s = str(p).strip()
    try:
        return ("num", round(float(s.replace(",", "")), 6))
    except ValueError:
        return ("v", " ".join(s.lower().split()))


def _plur(reads):
    """(class, ok) of the plurality over reads in order, ties to the earliest; the none class is wrong."""
    count, first, ok = {}, {}, {}
    for j, x in enumerate(reads):
        c = cls_of(x)
        count[c] = count.get(c, 0) + 1
        if c not in first:
            first[c], ok[c] = j, (False if c == NONE else bool(x["ok"]))
    best = min(count, key=lambda c: (-count[c], first[c]))
    return best, ok[best]


def _pt(x):
    """prompt tokens of a read (0 when the table does not carry them; an operator read may carry the prompt tokens of all its calls)."""
    return float(x.get("prompt_tokens") or 0.0)


def guess_of(x, gw=None):
    """the declared-guess flag of a read: a bool, or {window: bool} when several tail windows are carried (gw picks one)."""
    g = x.get("guess")
    if isinstance(g, dict):
        return g.get(gw if gw is not None else next(iter(g)))
    return g


def _triggered(init, trigger, gw=None):
    if trigger == "none":
        return False
    dis = len({cls_of(x) for x in init}) > 1
    gs = any(bool(guess_of(x, gw)) for x in init)
    return dis if trigger == "disagree" else gs if trigger == "guess" else (dis or gs)


def name_of(pol):
    if pol.get("name"):
        return pol["name"]
    if pol["trigger"] == "none":
        return f"vote{pol['n0']}"
    ne = pol.get("n_extra", 1)
    trig = pol["trigger"] + (f"@{pol['gw']}" if pol.get("gw") is not None and pol["trigger"] in ("guess", "either") else "")
    return f"n{pol['n0']}:{trig}->{pol['extra']}" + (f"x{ne}" if ne != 1 else "") + ("" if pol.get("final", "plur") == "plur" else ":" + pol["final"])


def needs_guess(pol):
    return pol["trigger"] in ("guess", "either")


def eval_row(pol, tip, extra=None, max_tuples=4000, rng=None):
    """Exact expectation on one row. tip: the row's cached tip reads; extra: the extra arm's reads on the row (None -> tip).
    -> dict(ok, tokens, esc, n) or None when the row has too few samples for the policy."""
    n0, trig, ne, final = pol["n0"], pol["trigger"], pol.get("n_extra", 1), pol.get("final", "plur")
    own = pol.get("extra", "tip") == "tip" or trig == "none"
    need = n0 + (ne if (own and trig != "none") else 0)
    if len(tip) < need or (not own and (extra is None or len(extra) < ne)):
        return None
    gw = pol.get("gw")
    if needs_guess(pol) and any(guess_of(x, gw) is None for x in tip):
        raise ValueError(f"policy {name_of(pol)} needs the declared-guess flag of every tip read")
    idx = range(len(tip))
    tuples = itertools.permutations(idx, n0)
    total = math.perm(len(tip), n0)
    if total > max_tuples and rng is not None:
        tuples = [tuple(rng.permutation(len(tip))[:n0]) for _ in range(max_tuples)]
    s_ok = s_tok = s_esc = s_tot = 0.0
    n = 0
    for tp in tuples:
        init = [tip[i] for i in tp]
        t0 = sum(x["tokens"] for x in init)
        p0 = sum(_pt(x) for x in init)
        if not _triggered(init, trig, gw):
            s_ok += _plur(init)[1]
            s_tok += t0
            s_tot += t0 + p0
            n += 1
            continue
        pool = [tip[i] for i in idx if i not in tp] if own else list(extra)
        combos = list(itertools.permutations(range(len(pool)), ne)) if ne <= 2 else [tuple(c) for c in itertools.combinations(range(len(pool)), ne)]
        o = t = pt = 0.0
        for cb in combos:
            ex = [pool[i] for i in cb]
            if final == "extra":
                ans = next((x for x in ex if cls_of(x) != NONE), None)
                o += bool(ans["ok"]) if ans is not None else _plur(init + ex)[1]
            else:
                o += _plur(init + ex)[1]
            t += t0 + sum(x["tokens"] for x in ex)
            pt += p0 + sum(_pt(x) for x in ex)
        s_ok += o / len(combos)
        s_tok += t / len(combos)
        s_tot += (t + pt) / len(combos)
        s_esc += 1.0
        n += 1
    return dict(ok=s_ok / n, tokens=s_tok / n, total=s_tot / n, esc=s_esc / n, n=n)


def evaluate(pol, tip_arm, row_ids, extra_arm=None, rng=None):
    """{row: dict(ok, tokens, esc)} over row_ids (a row with too few samples raises: the caller decides what to read)."""
    out = {}
    for r in row_ids:
        v = eval_row(pol, tip_arm[r], None if extra_arm is None else extra_arm.get(r), rng=rng)
        if v is None:
            raise ValueError(f"policy {name_of(pol)}: row {r} has too few samples")
        out[r] = v
    return out


def macro_mean(per_row, rows, key, benches=None):
    """macro mean over benchmarks (each benchmark weighs the same) of per_row[r][key]."""
    byb = {}
    for r, v in per_row.items():
        b = rows[r]["bench"]
        if benches is None or b in benches:
            byb.setdefault(b, []).append(v[key])
    return sum(sum(v) / len(v) for v in byb.values()) / len(byb)


# ------------------------------------------------------------------------------------------------ the grammar
def grammar(extras=("tip",), n0s=(1, 2), with_guess=True, windows=(None,)):
    """The small policy grammar of the design note: n0 in {1, 2}; trigger in {none, disagree, guess, either}; extra in {tip, cell...}; one
    escalation; final in {plur, extra}. Meaningless members are dropped (n0 = 1 cannot disagree; plurality of 1 + 1 reads is the first
    read, so n0 = 1 escalations use final = extra; n0 = 2 with trigger none is the first read at twice the tokens)."""
    out = [dict(n0=1, trigger="none", extra="tip")]
    for e in extras:
        for n0 in n0s:
            for trig in ("disagree", "guess", "either"):
                if n0 == 1 and trig != "guess":
                    continue
                if not with_guess and trig != "disagree":
                    continue
                for final in ("plur", "extra"):
                    if n0 == 1 and final == "plur":
                        continue
                    for gw in (windows if trig != "disagree" else (None,)):
                        out.append(dict(n0=n0, trigger=trig, extra=e, final=final, **({"gw": gw} if gw is not None else {})))
    return out


def tip_policies(max_vote=5, with_guess=True, windows=(None,)):
    """Tip-only policies for the frontier: votes of 1..max_vote, the grammar with extra = tip, and the early-stopping vote of 5 (read 3;
    2 more unless unanimous)."""
    out = [dict(n0=m, trigger="none", extra="tip") for m in range(1, max_vote + 1) if m != 2]
    out += [p for p in grammar(("tip",), with_guess=with_guess, windows=windows) if p["trigger"] != "none"]
    if max_vote >= 5:
        out.append(dict(n0=3, trigger="disagree", extra="tip", n_extra=2, final="plur"))
    return out


# ------------------------------------------------------------------------------------------------ frontier
def hull(points):
    """points: [(tokens, acc, name)] -> the upper concave hull, tokens increasing (the vertices a mixture cannot beat)."""
    pts = sorted(points, key=lambda p: (p[0], -p[1]))
    best = []
    for p in pts:                                   # drop dominated points (more tokens, no more accuracy)
        if not best or p[1] > best[-1][1] + 1e-12:
            best.append(p)
    h = []
    for p in best:
        while len(h) >= 2 and (h[-1][1] - h[-2][1]) * (p[0] - h[-2][0]) <= (p[1] - h[-2][1]) * (h[-1][0] - h[-2][0]) + 1e-15:
            h.pop()
        h.append(p)
    return h


class Frontier:
    """The tip's own token-accuracy frontier on one row set. table[name] = per-row dict(ok, tokens, esc); vertices = the hull of the macro
    means. ref_at(T) -> (per-row reference ok, per-row tokens, description) at T mean tokens, or None above the last vertex."""

    def __init__(self, tip_arm, row_ids, rows, policies=None, rng=None, hull_from=None, cost="tokens", table=None):
        """cost: "tokens" (generated tokens: the gating convention) or "total" (generated + prompt tokens of every read: printed beside it).
        table: a table computed before (another cost convention of the same reads) to avoid re-evaluating the policies."""
        self.rows, self.row_ids, self.cost = rows, list(row_ids), cost
        has_guess = all(guess_of(x) is not None for r in self.row_ids for x in tip_arm[r])
        nmin = min(len(tip_arm[r]) for r in self.row_ids)
        pols = policies if policies is not None else tip_policies(max_vote=min(5, nmin), with_guess=has_guess)
        self.table, self.pols = {}, {}
        for p in pols:
            need = p["n0"] + (p.get("n_extra", 1) if p["trigger"] != "none" else 0)
            if need > nmin:
                continue
            self.table[name_of(p)] = table[name_of(p)] if table is not None and name_of(p) in table else evaluate(p, tip_arm, self.row_ids, rng=rng)
            self.pols[name_of(p)] = p
        self.points = [(macro_mean(t, rows, cost), macro_mean(t, rows, "ok"), n) for n, t in self.table.items()]
        # hull_from: vertex NAMES chosen elsewhere (e.g. on signal rows), valued here; default: the hull of this row set
        if hull_from is not None:
            keep = [p for p in self.points if p[2] in set(hull_from)]
            if not any(p[2] == "vote1" for p in keep):
                raise ValueError("hull_from must hold vote1 (the single read)")
            self.vertices = sorted(keep, key=lambda p: p[0])
        else:
            self.vertices = hull(self.points)
        self.base_tokens = next(p[0] for p in self.points if p[2] == "vote1")

    def ref_at(self, T, strict=False):
        """The frontier at T tokens: f(T) = the best accuracy a mixture of tip-only policies reaches with T tokens. Between two vertices:
        their mixture with exactly T tokens (strict: the upper vertex). Above the last vertex the frontier is FLAT (more tip reads were
        evaluated and did not score higher) up to the largest evaluated policy; beyond that it is unknown (None: more tip samples needed)."""
        V = self.vertices
        flat = False
        if T <= V[0][0] + 1e-9:
            lo = hi = V[0]
            w = 0.0
        elif T > V[-1][0] + 1e-9:
            if T > max(p[0] for p in self.points) + 1e-9:
                return None
            lo = hi = V[-1]
            w, flat = 0.0, True
        else:
            j = next(i for i in range(1, len(V)) if V[i][0] >= T - 1e-9)
            lo, hi = V[j - 1], V[j]
            w = 1.0 if strict else (T - lo[0]) / (hi[0] - lo[0])
        a, b = self.table[lo[2]], self.table[hi[2]]
        ok = {r: (1 - w) * a[r]["ok"] + w * b[r]["ok"] for r in self.row_ids}
        tok = {r: (1 - w) * a[r][self.cost] + w * b[r][self.cost] for r in self.row_ids}
        desc = dict(lo=lo[2], hi=hi[2], w=round(w, 4), tokens=(1 - w) * lo[0] + w * hi[0], acc=(1 - w) * lo[1] + w * hi[1], strict=bool(strict),
                    multiple=((1 - w) * lo[0] + w * hi[0]) / self.base_tokens, flat=flat)
        return ok, tok, desc

    def lines(self):
        V = {p[2] for p in self.vertices}
        out = []
        for tk, acc, n in sorted(self.points):
            out.append(f"{n:28s} tokens {tk:7.0f} (x{tk / self.base_tokens:4.2f})  acc {100 * acc:6.2f}  {'VERTEX' if n in V else ''}")
        return out


# ------------------------------------------------------------------------------------------------ views (stage 3: settings, view-diverse votes)
def views_policy(init, extra=(), final="plur", trigger=None):
    """A read policy over VIEWS (the tip and each cached settings variant): init = the views of the initial reads in order, extra = the
    views of the extra reads taken when the initial reads do not all agree (trigger "disagree"; "none" when there is no extra), final =
    "plur" (plurality over all reads taken, ties to the earliest) or "extra" (the first extra read with an answer decides)."""
    return dict(init=list(init), extra_views=list(extra), final=final, trigger=trigger or ("disagree" if extra else "none"))


def views_name(pol):
    if pol.get("name"):
        return pol["name"]
    s = "[" + "+".join(pol["init"]) + "]"
    if pol["extra_views"]:
        s += "->[" + "+".join(pol["extra_views"]) + "]" + ("" if pol.get("final", "plur") == "plur" else ":" + pol["final"])
    return s


def eval_views_row(pol, reads, max_init=600, max_extra=60, rng=None):
    """Expectation of a views policy on one row. reads = {view: [the row's cached reads under that view]}. Reads of one view are drawn
    without replacement; reads of different views are independent. Exact (every assignment) while the initial assignments number at
    most max_init and the extra ones at most max_extra; beyond that a fixed-seed random subset (rng) of that size.
    -> dict(ok, tokens, total, esc) or None when a view has too few reads."""
    init, extra, final = pol["init"], pol.get("extra_views") or [], pol.get("final", "plur")
    need = {}
    for v in init + extra:
        need[v] = need.get(v, 0) + 1
    if any(len(reads.get(v) or ()) < k for v, k in need.items()):
        return None

    def assignments(views, used, cap):
        """index tuples for the positions `views`, distinct inside a view and outside `used` {view: set(indices)}."""
        pools = [[i for i in range(len(reads[v])) if i not in used.get(v, ())] for v in views]
        out = []
        for tp in itertools.product(*pools):
            seen, ok = {}, True
            for v, i in zip(views, tp):
                if i in seen.setdefault(v, set()):
                    ok = False
                    break
                seen[v].add(i)
            if ok:
                out.append(tp)
        if len(out) > cap:
            if rng is None:
                raise ValueError(f"{views_name(pol)}: {len(out)} assignments > {cap} and no rng to subsample")
            out = [out[i] for i in sorted(rng.choice(len(out), size=cap, replace=False))]
        return out
    s_ok = s_tok = s_tot = s_esc = 0.0
    A = assignments(init, {}, max_init)
    for tp in A:
        rs = [reads[v][i] for v, i in zip(init, tp)]
        t0, p0 = sum(x["tokens"] for x in rs), sum(_pt(x) for x in rs)
        if not extra or pol["trigger"] == "none" or len({cls_of(x) for x in rs}) <= 1:
            s_ok += _plur(rs)[1]
            s_tok += t0
            s_tot += t0 + p0
            continue
        used = {}
        for v, i in zip(init, tp):
            used.setdefault(v, set()).add(i)
        E = assignments(extra, used, max_extra)
        o = t = pt = 0.0
        for ep in E:
            ex = [reads[v][i] for v, i in zip(extra, ep)]
            if final == "extra":
                a = next((x for x in ex if cls_of(x) != NONE), None)
                o += bool(a["ok"]) if a is not None else _plur(rs + ex)[1]
            else:
                o += _plur(rs + ex)[1]
            t += t0 + sum(x["tokens"] for x in ex)
            pt += p0 + sum(_pt(x) for x in ex)
        s_ok += o / len(E)
        s_tok += t / len(E)
        s_tot += (t + pt) / len(E)
        s_esc += 1.0
    n = len(A)
    return dict(ok=s_ok / n, tokens=s_tok / n, total=s_tot / n, esc=s_esc / n, n=n)


def evaluate_views(pol, arms, row_ids, seed=7):
    """{row: dict(ok, tokens, total, esc)}; arms = {view: arm}. A row with too few reads under a view raises."""
    import numpy as _np
    out = {}
    for j, r in enumerate(row_ids):
        v = eval_views_row(pol, {k: a.get(r) for k, a in arms.items()}, rng=_np.random.default_rng([seed, j]))
        if v is None:
            raise ValueError(f"views policy {views_name(pol)}: row {r} has too few reads under one of its views")
        out[r] = v
    return out


def budget_policies(view):
    """the three report points of one view: one read, 2 reads + a third on disagreement (the default budget), 3 reads + 2 unless unanimous."""
    return [("x1.00", views_policy([view])), ("default", views_policy([view, view], [view])), ("x4.12", views_policy([view, view, view], [view, view]))]


def grammar_views(views):
    """View-diverse votes (coordinator 11-11, stage 3 B): for every ordered pair of different views (a, b):
    a pair of reads through the two views with a third read under a or b on disagreement (plurality, or the third read decides);
    two reads under a with the third under b; and the five-read form (a, a, b) + (a, b) unless unanimous."""
    out = []
    for a in views:
        for b in views:
            if a == b:
                continue
            for third in (a, b):
                for final in ("plur", "extra"):
                    out.append(views_policy([a, b], [third], final))
            for final in ("plur", "extra"):
                out.append(views_policy([a, a], [b], final))
            out.append(views_policy([a, a, b], [a, b], "plur"))
    return out
