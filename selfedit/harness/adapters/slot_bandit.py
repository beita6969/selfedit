"""1109c-final slot allocator (amendment 10; design /Users/admin/selfedit/fable_1109_slot_allocator.md §2, exact): Thompson sampling over operator
families. State R.state["slot_bandit"] = {"ledger": {proposal name: record}, "prior": {family: m0}, "deals": [...]}; the posterior is a pure function of
the ledger. One floor slot per available family, the rest by independent posterior draws (cap 6 TS slots per family), then a random permutation."""
import math
import numpy as np

FAMILIES = ("program", "cross", "patch", "gate", "tokens", "verbosity")
GAMMA, KAPPA0, A0, TAU0, CAP, FLOOR = 0.9, 1.0, 2.0, 4.0, 6, 1
PP_PER_Z, COST_RATE, COST_FREE = 1.0, 10.0, 0.1
W_STAGE1 = 210.0 / 1600.0
E_MAX = {1: 0.0, 2: 0.564, 3: 0.846, 4: 1.029, 5: 1.163, 6: 1.267, 8: 1.424, 10: 1.539, 12: 1.629, 16: 1.766, 29: 2.04}


def e_max(n):
    """E[max of n iid N(0,1)] (table; Blom approximation between entries)."""
    if n in E_MAX:
        return E_MAX[n]
    from scipy.stats import norm
    return float(norm.ppf((n - 0.375) / (n + 0.25)))


def grid_prior(grid_rows):
    """m0 per grid operator: clip(PP_PER_Z * (best-cell zx - E[max of n_cells N(0,1)]), 0, 3); families with no grid cell start at 0."""
    by = {}
    for x in grid_rows or []:
        by.setdefault(x["op"], []).append(float(x.get("zx") or 0.0))
    m0 = {k: 0.0 for k in FAMILIES}
    for o, z in by.items():
        if o in m0:
            m0[o] = float(np.clip(PP_PER_Z * (max(z) - e_max(len(z))), 0.0, 3.0))
    return m0


def family_of(op):
    if op in ("kernel",):
        return "patch"
    if op in ("gate", "gate_write"):
        return "gate"
    return op if op in FAMILIES else None


def y_eff(y, dC):
    return float(y) - COST_RATE * max(0.0, float(dC or 0.0) - COST_FREE)


def record(ledger, name, family, rnd, valid, y=None, w=None, dC=0.0, stage=None, why=None):
    """Write or revise (in place) one proposal's record."""
    ledger[name] = dict(family=family, round=int(rnd), valid=bool(valid), y=None if y is None else float(y), w=None if w is None else float(w),
                        dC=float(dC or 0.0), y_eff=None if y is None else y_eff(y, dC), stage=stage, invalid_reason=why,
                        revised_round=ledger.get(name, {}).get("revised_round") if name in ledger else None)


def posterior(ledger, m0, r):
    """Per family: Beta(1 + V, 1 + I) validity and NIG(m_n, kappa_n, a_n, b_n) effect, discounted by GAMMA^(r - r_j)."""
    out = {}
    for k in FAMILIES:
        recs = [x for x in ledger.values() if x["family"] == k]
        V = sum(GAMMA ** (r - x["round"]) for x in recs if x["valid"])
        I = sum(GAMMA ** (r - x["round"]) for x in recs if not x["valid"])
        obs = [(x["w"] * GAMMA ** (r - x["round"]), x["y_eff"]) for x in recs if x["valid"] and x["y_eff"] is not None and x["w"]]
        W = float(sum(o for o, _ in obs))
        mk = float(m0.get(k, 0.0))
        if W > 0:
            yb = sum(o * y for o, y in obs) / W
            ss = sum(o * (y - yb) ** 2 for o, y in obs)
        else:
            yb, ss = mk, 0.0
        kn = KAPPA0 + W
        mn = (KAPPA0 * mk + W * yb) / kn
        an = A0 + W / 2.0
        bn = A0 * TAU0 ** 2 + 0.5 * (ss + KAPPA0 * W * (yb - mk) ** 2 / kn)
        out[k] = dict(m=mn, kappa=kn, a=an, b=bn, V=V, I=I, W=W)
    return out


def draw(post, k, rng):
    p = post[k]
    pv = rng.beta(1.0 + p["V"], 1.0 + p["I"])
    tau2 = 1.0 / rng.gamma(p["a"], 1.0 / p["b"])
    mu = rng.normal(p["m"], math.sqrt(tau2 / p["kappa"]))
    return float(pv), float(mu), float(math.sqrt(tau2))


def deal(post, avail, n_slots, rng):
    """Floor slot per available family, then TS for the rest (cap CAP TS slots per family), then a random permutation. Returns (ops, rows)."""
    avail = [k for k in FAMILIES if k in avail]
    rows = [dict(family=k, mode="floor") for k in avail[:n_slots]]
    ts = {k: 0 for k in avail}
    while len(rows) < n_slots:
        best, bs, bd = None, -1e18, None
        for k in avail:
            if ts[k] >= CAP:
                continue
            pv, mu, tau = draw(post, k, rng)
            s = pv * mu
            if s > bs:
                best, bs, bd = k, s, dict(p=pv, mu=mu, tau=tau)
        if best is None:
            best = avail[len(rows) % len(avail)]
            bd, bs = None, None
        ts[best] = ts.get(best, 0) + 1
        rows.append(dict(family=best, mode="ts", draw=bd, score=bs))
    perm = rng.permutation(len(rows))
    rows = [rows[i] for i in perm]
    return [x["family"] for x in rows], rows


def p_best(post, avail, rng, n=1000):
    avail = [k for k in FAMILIES if k in avail]
    c = {k: 0 for k in avail}
    for _ in range(n):
        s = {k: (lambda d: d[0] * d[1])(draw(post, k, rng)) for k in avail}
        c[max(s, key=s.get)] += 1
    return {k: v / n for k, v in c.items()}
