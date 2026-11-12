"""selfedit v5, the statistics of the judge (design_1111_v5.md section 4 step 3; open point 6.1). numpy + scipy.stats.t only.

Units. A row difference x_i (candidate minus reference on one JUDGE row) -> picture clusters inside a benchmark -> the benchmark effect
d_b (mean of its rows) with a cluster-robust se_b -> the macro mean over the SCORED benchmarks S(c) (every benchmark weighs the same).

Tests (all one-sided, "the candidate is better"):
  p_macro(S)      row level, benchmarks fixed: z = mean_b d_b / sqrt(sum_b se_b^2) / k, Student t with nu = sum_b (G_b - 1).
  p_gen           THE v5 statistic = max over b in S of p_macro(S minus b): the macro mean must be significantly positive with ANY ONE scored
                  benchmark left out (an intersection-union test of "no effect, or an effect carried by one benchmark"). It merges conditions
                  (i) and (ii) of the design in one p-value with row-level resolution, so a family-wise correction is possible.
  p_bench         benchmark level (the benchmark as the unit): exact sign-flip permutation of d_b over S (2^k arrangements; the statistic is
                  the mean, which orders arrangements as the t statistic does); and the Student t over d_b (nu = k - 1) for reference.
  neg_bench       condition (iii): a scored benchmark is significantly negative when its one-sided p (d_b < 0, t with G_b - 1) is below
                  alpha_neg / k (Bonferroni over the scored benchmarks).
  Design.signflip Monte-Carlo sign-flip of picture clusters inside benchmarks (the row-level permutation null). THE JUDGE GATES ON THESE
                  p-values (p_macro, p_gen, the negative-benchmark test): they need no se floor and stay honest when a candidate differs
                  from its reference on few rows. The t forms above are what the A/A / power simulations use (same numbers when many rows
                  differ: aa_power.txt) and are printed beside them.
Family: holm() (family-wise error) and bh() (false discovery rate) over the p_gen of the candidates judged in one generation.
"""
import itertools
import math

import numpy as np
from scipy.stats import t as _t


# ------------------------------------------------------------------------------------------------ benchmark effects
FLOOR_ROWS = 0.5      # se_b floor = FLOOR_ROWS / n_b. Stage-1 pipeline A/A (aa_power.txt part B): escalation policies have sd(d) 0.48 pp; with the
                      # floor at 0 or 0.5 row the mean se is 0.48 (false accepts .085 at nominal .10); judge_bh's 1-row floor gives se 0.57 (.053)


def bench_effects(x, rows, benches=None, floor_rows=None):
    """x: {row: paired difference}; rows: {row: {bench, pic}}. -> {bench: dict(d, se, n, G)}; se cluster-robust over picture clusters
    (sqrt(G / (G - 1) * sum_c (sum_{i in c} (x_i - d))^2) / n), floored at floor_rows / n (the evidence of a benchmark cannot be sharper
    than if floor_rows more rows had flipped: the guard against a benchmark where almost no row differs)."""
    floor_rows = FLOOR_ROWS if floor_rows is None else floor_rows
    byb = {}
    for r, v in x.items():
        b = rows[r]["bench"]
        if benches is not None and b not in benches:
            continue
        byb.setdefault(b, {}).setdefault(rows[r]["pic"], []).append(float(v))
    out = {}
    for b, cl in byb.items():
        n = sum(len(v) for v in cl.values())
        G = len(cl)
        d = sum(sum(v) for v in cl.values()) / n
        ss = sum((sum(v) - d * len(v)) ** 2 for v in cl.values())
        se = math.sqrt(ss * G / max(G - 1, 1)) / n
        out[b] = dict(d=d, se=max(se, floor_rows / n), se_raw=se, n=n, G=G)
    return out


def macro(E, S):
    """macro mean of d_b over the benchmarks S, its row-level se and nu."""
    S = [b for b in S if b in E]
    k = len(S)
    if k == 0:
        return dict(d=float("nan"), se=float("nan"), nu=1.0, k=0)
    d = sum(E[b]["d"] for b in S) / k
    se = math.sqrt(sum(E[b]["se"] ** 2 for b in S)) / k
    return dict(d=d, se=se, nu=float(max(sum(E[b]["G"] - 1 for b in S), 1)), k=k)


def p_macro(E, S):
    m = macro(E, S)
    if m["k"] == 0:
        return 1.0
    return float(_t.sf(m["d"] / m["se"], m["nu"]))


def p_gen(E, S):
    """max over b in S of p_macro(S - {b}); the benchmark whose removal gives the max ("carrier"); needs k >= 3."""
    S = [b for b in S if b in E]
    if len(S) < 3:
        return 1.0, None
    worst, who = -1.0, None
    for b in S:
        p = p_macro(E, [c for c in S if c != b])
        if p > worst:
            worst, who = p, b
    return worst, who


def lobo_point(E, S):
    """condition (ii) as a point rule: min over b of the macro mean without b."""
    S = [b for b in S if b in E]
    if len(S) < 2:
        return float("-inf")
    return min(macro(E, [c for c in S if c != b])["d"] for b in S)


def p_bench_signflip(E, S):
    """exact one-sided sign-flip p over the scored benchmarks: #{s in {-1,+1}^k: sum s_b d_b >= sum d_b} / 2^k (ties count)."""
    d = np.array([E[b]["d"] for b in S if b in E], float)
    k = len(d)
    if k == 0:
        return 1.0
    obs = d.sum()
    cnt = 0
    for s in itertools.product((-1.0, 1.0), repeat=k):
        cnt += float(np.dot(s, d)) >= obs - 1e-12
    return cnt / 2 ** k


def p_bench_t(E, S):
    d = np.array([E[b]["d"] for b in S if b in E], float)
    k = len(d)
    if k < 2:
        return 1.0
    sd = d.std(ddof=1)
    if sd <= 0:
        return 0.0 if d.mean() > 0 else 1.0
    return float(_t.sf(d.mean() / (sd / math.sqrt(k)), k - 1))


def neg_bench(E, S, alpha_neg=0.10):
    """scored benchmarks that are significantly negative: one-sided p(d_b < 0) < alpha_neg / k."""
    S = [b for b in S if b in E]
    out = []
    for b in S:
        p = float(_t.cdf(E[b]["d"] / E[b]["se"], max(E[b]["G"] - 1, 1)))
        if p < alpha_neg / len(S):
            out.append((b, p))
    return out


# ------------------------------------------------------------------------------------------------ family
def holm(pvals, alpha=0.10):
    """Holm step-down accept mask (family-wise error <= alpha under any dependence)."""
    p = np.asarray(pvals, float)
    K = len(p)
    acc = np.zeros(K, bool)
    for j, i in enumerate(np.argsort(p, kind="mergesort")):
        if p[i] <= alpha / (K - j):
            acc[i] = True
        else:
            break
    return acc


def bh(pvals, alpha=0.10):
    p = np.asarray(pvals, float)
    K = len(p)
    if K == 0:
        return np.zeros(0, bool)
    order = np.argsort(p, kind="mergesort")
    ok = p[order] <= alpha * (np.arange(1, K + 1) / K)
    k = int(np.max(np.nonzero(ok)[0]) + 1) if ok.any() else 0
    acc = np.zeros(K, bool)
    acc[order[:k]] = True
    return acc


# ------------------------------------------------------------------------------------------------ fast array form (simulation, sign-flip)
class Design:
    """Fixed layout of one row set: row order, benchmark index, picture-cluster index. Statistics of many difference vectors at once."""

    def __init__(self, row_ids, rows, benches=None):
        self.rows = list(row_ids)
        self.benches = list(benches) if benches is not None else sorted({rows[r]["bench"] for r in self.rows})
        bi = {b: i for i, b in enumerate(self.benches)}
        self.b = np.array([bi[rows[r]["bench"]] for r in self.rows])
        cl = {}
        self.c = np.array([cl.setdefault((rows[r]["bench"], rows[r]["pic"]), len(cl)) for r in self.rows])
        self.nc = len(cl)
        self.cb = np.zeros(self.nc, int)
        self.cb[self.c] = self.b
        self.n_b = np.bincount(self.b, minlength=len(self.benches)).astype(float)
        self.G_b = np.bincount(self.cb, minlength=len(self.benches)).astype(float)
        self.csize = np.bincount(self.c, minlength=self.nc).astype(float)
        self.M_rc = np.zeros((len(self.rows), self.nc))
        self.M_rc[np.arange(len(self.rows)), self.c] = 1.0
        self.M_cb = np.zeros((self.nc, len(self.benches)))
        self.M_cb[np.arange(self.nc), self.cb] = 1.0

    def effects(self, X, floor_rows=None):
        """X: (R, n_rows) differences -> d (R, B), se (R, B) (cluster-robust, floored at floor_rows / n_b)."""
        floor_rows = FLOOR_ROWS if floor_rows is None else floor_rows
        X = np.atleast_2d(X)
        cs = X @ self.M_rc                                   # cluster sums
        d = (cs @ self.M_cb) / self.n_b
        res = cs - d[:, self.cb] * self.csize
        ss = (res ** 2) @ self.M_cb
        se = np.sqrt(ss * self.G_b / np.maximum(self.G_b - 1, 1)) / self.n_b
        return d, np.maximum(se, floor_rows / self.n_b)

    def stats(self, X, S=None, alpha_neg=0.10, floor_rows=None):
        """-> dict of arrays over replicates: macro d, p_macro, p_gen, lobo_min, p_bench (exact sign-flip), p_bench_t, neg (any scored
        benchmark significantly negative), npos (scored benchmarks with d_b > 0), d_b."""
        d, se = self.effects(X, floor_rows)
        idx = np.arange(len(self.benches)) if S is None else np.array([self.benches.index(b) for b in S])
        d, se, G = d[:, idx], se[:, idx], self.G_b[idx]
        k = len(idx)
        dm = d.mean(1)
        sem = np.sqrt((se ** 2).sum(1)) / k
        pm = _t.sf(dm / sem, max(G.sum() - k, 1))
        pg = np.zeros(len(dm))
        lob = np.full(len(dm), np.inf)
        for j in range(k):
            keep = [i for i in range(k) if i != j]
            dj = d[:, keep].mean(1)
            sj = np.sqrt((se[:, keep] ** 2).sum(1)) / (k - 1)
            pg = np.maximum(pg, _t.sf(dj / sj, max(G[keep].sum() - (k - 1), 1)))
            lob = np.minimum(lob, dj)
        signs = np.array(list(itertools.product((-1.0, 1.0), repeat=k)))
        pb = ((d @ signs.T) >= d.sum(1, keepdims=True) - 1e-12).mean(1)
        sd = d.std(1, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            pbt = np.where(sd > 0, _t.sf(dm / (sd / math.sqrt(k)), k - 1), np.where(dm > 0, 0.0, 1.0))
        neg = (_t.cdf(d / se, np.maximum(G - 1, 1)) < alpha_neg / k).any(1)
        return dict(d=dm, se=sem, p_macro=pm, p_gen=pg, lobo_min=lob, p_bench=pb, p_bench_t=pbt, neg=neg, npos=(d > 0).sum(1), d_b=d, se_b=se)

    def _flips(self, n, seed):
        key = (n, seed)
        if getattr(self, "_F", None) is None or self._F[0] != key:
            rng = np.random.default_rng(seed)
            self._F = (key, (rng.integers(0, 2, size=(n, self.nc), dtype=np.int8) * 2 - 1).astype(np.float32))
        return self._F[1]

    def signflip(self, x, S=None, n=20000, seed=0):
        """Sign-flip of picture clusters inside benchmarks (the row-level permutation null: a cluster's difference is as likely with
        either sign) for ONE difference vector x (in self.rows order). Exact in the sense of a Monte-Carlo permutation test with n
        flips (p = (1 + #{flipped >= observed}) / (n + 1)); it needs no se and no floor, so a candidate that differs from the reference
        on a handful of rows cannot look sharper than its rows allow (m active clusters cannot give p below 2^-m).
        -> dict(p_macro, p_gen, carrier index in S, p_lobo [per left-out benchmark], p_neg [per benchmark: lower tail])."""
        x = np.asarray(x, np.float32)
        idx = np.arange(len(self.benches)) if S is None else np.array([self.benches.index(b) for b in S])
        cs = x @ self.M_rc.astype(np.float32)
        F = self._flips(n, seed)
        Mcb = (self.M_cb / self.n_b).astype(np.float32)
        db = ((F * cs) @ Mcb)[:, idx]                           # flipped benchmark effects (n, k)
        ob = (cs @ Mcb)[idx]
        k = len(idx)
        tol = 1e-7
        pm = (1 + int((db.sum(1) >= ob.sum() - tol).sum())) / (n + 1)
        pl = []
        for j in range(k):
            keep = [i for i in range(k) if i != j]
            pl.append((1 + int((db[:, keep].sum(1) >= ob[keep].sum() - tol).sum())) / (n + 1) if keep else 1.0)
        pneg = [(1 + int((db[:, j] <= ob[j] + tol).sum())) / (n + 1) for j in range(k)]
        j = int(np.argmax(pl)) if pl else None
        return dict(p_macro=float(pm), p_gen=float(max(pl)) if k >= 3 else 1.0, carrier=j, p_lobo=[float(v) for v in pl], p_neg=[float(v) for v in pneg])
