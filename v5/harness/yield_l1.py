"""L1 statistic of v4 (design_1110_v4 §9.3, coordinator 11-10): the value of an author (a proposer version, the blind arm) is the
expected response to selection of the candidates it writes, not their mean.

Why not the mean: by the Price equation a generation's gain = selection term + transmission term. An author supplies the offspring
distribution; the lineage keeps only what the judge accepts. A timid author (every candidate ~ 0) has a higher mean than a bold one
(mean slightly negative, a few large real gains) and contributes nothing (Bassett, Potter & De Jong 2004: the mean effect of an
operator can be ~ 0 while its tail drives progress). The breeder's equation gives the quantity that matters.

Data: ALL reads of each candidate (final D and its cluster-robust covariance), not only rung 1. The continuation rule depends on the data
only, so it is ignorable for the posterior (likelihood principle); simulation at our sizes (scratchpad sim_l1_all.py, 11-10): equal
authors are switched in .02-.06 / .08-.10 / .14-.17 of runs by generation 2 / 4 / 8, the same as with rung-1 data, while a better
challenger is found about twice as fast (K 8 per author per generation: "good" .39 / .66 / .88, "bold" .34 / .63 / .84; rung 1 only:
.26 / .45 / .65 and .14 / .28 / .41). A worse challenger is adopted in <= 2 % of runs.

Model per author (same as judge_eb): gains D ~ N(theta, S), theta_j ~ N(mu, tau^2). For a candidate drawn from N(mu, tau^2), read
with standard error sigma and accepted iff its posterior sign probability under the family model is >= 1 - alpha:

    r = tau^2 / (tau^2 + sigma^2)          reliability (the heritability of the breeder's equation)
    accept  <=>  D >= t,   t = mu + (z_alpha * sqrt(r) * sigma - mu) / r
    Y(mu, tau) = E[theta * 1(accept)] = mu * Phi(a) + tau^2 / s * phi(a),   s = sqrt(tau^2 + sigma^2),  a = (mu - t) / s

Y is the expected accepted true gain per slot. The posterior of (mu, tau) given the author's candidates is computed on a grid; the prior
is the unit-information prior from the pooled ledger (all candidates of the window, tempered to the weight of one candidate), flat when
the ledger is empty. Decisions: P(Y_a > Y_b) from the two grid posteriors; Thompson draws of Y for slots.
No constants beyond alpha and the grids' ranges (an accuracy gain lies in [-1, 1])."""
import math

import numpy as np
from scipy.stats import norm

MUS = np.linspace(-0.25, 0.25, 201)            # family mean of the gain per ROUTED row; +-25 pp bounds any family mean here
_T = np.linspace(0.0, 1.0, 61)[1:]
TAUS, WT = 0.30 * _T ** 2, 2 * _T              # tau in (0, .30], flat prior in tau (grid in t^2 resolves small tau)


def y_of(mu, tau, sigma, alpha):
    """expected accepted true gain per candidate (formula in the module docstring); vectorised over mu, tau."""
    mu, tau = np.asarray(mu, float), np.asarray(tau, float)
    z = norm.ppf(1 - alpha)
    s2 = tau ** 2 + sigma ** 2
    r = np.maximum(tau ** 2 / s2, 1e-12)
    t = mu + (z * np.sqrt(r) * sigma - mu) / r
    a = (mu - t) / np.sqrt(s2)
    return mu * norm.cdf(a) + tau ** 2 / np.sqrt(s2) * norm.pdf(a)


def log_marginal(D, S):
    """log p(D | mu, tau) on the (tau, mu) grid for one author's candidates: D ~ N(mu 1, S + tau^2 I)."""
    D, S = np.asarray(D, float), np.asarray(S, float)
    K = len(D)
    out = np.zeros((len(TAUS), len(MUS)))
    if K == 0:
        return out
    one = np.ones(K)
    for a, tau in enumerate(TAUS):
        V = S + tau ** 2 * np.eye(K) + 1e-12 * np.eye(K)
        L = np.linalg.cholesky(V)
        Vi = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(K)))
        q1, q2, q3 = D @ Vi @ D, one @ Vi @ D, one @ Vi @ one
        out[a] = -0.5 * (q1 - 2 * MUS * q2 + MUS ** 2 * q3) - np.log(np.diag(L)).sum()
    return out


def unit_prior(D_pool, S_pool):
    """unit-information prior: the pooled log-likelihood of all candidates of the window divided by their number. Flat if empty."""
    n = 0 if D_pool is None else len(D_pool)
    if n == 0:
        return np.zeros((len(TAUS), len(MUS)))
    return log_marginal(D_pool, S_pool) / n


def yield_posterior(D, S, sigma, alpha, log_prior=None):
    """posterior of Y for one author: returns (y values (G,), weights (G,)) over the flattened (tau, mu) grid."""
    lw = log_marginal(D, S) + np.log(WT)[:, None] + (0.0 if log_prior is None else log_prior)
    w = np.exp(lw - lw.max())
    w /= w.sum()
    y = y_of(MUS[None, :], TAUS[:, None], sigma, alpha)
    return y.ravel(), w.ravel()


def yield_posterior_routed(d, S, shares, alpha, log_prior=None):
    """AM14 (11-10 17:10): the family model lives on the per-ROUTED-row scale (effects are exchangeable per routed row, not per whole set:
    on the whole-set scale a 6-row entry has a tiny standard error and drags the family). d (K,) = each candidate's mean paired
    difference on its routed rows, S (K, K) their covariance, shares (K,) = routed rows / judge rows ("always" = 1).
    Y = expected accepted WHOLE-SET gain per slot = mean_j share_j * E[theta 1(accept) | mu, tau, sigma_j], with the author's own
    reach and precision profile (share_j, sigma_j) taken as given. Returns (y values, weights) over the flattened (tau, mu) grid."""
    d, S, shares = np.asarray(d, float), np.asarray(S, float), np.asarray(shares, float)
    lw = log_marginal(d, S) + np.log(WT)[:, None] + (0.0 if log_prior is None else log_prior)
    w = np.exp(lw - lw.max())
    w /= w.sum()
    if len(d) == 0:
        return np.zeros(w.size), w.ravel()
    sig = np.sqrt(np.maximum(np.diag(S), 1e-12))
    y = np.zeros((len(TAUS), len(MUS)))
    for sh, sg in zip(shares, sig):
        y += sh * y_of(MUS[None, :], TAUS[:, None], sg, alpha)
    return (y / len(d)).ravel(), w.ravel()


def summary(y, w):
    m = float((w * y).sum())
    return dict(mean=m, sd=float(math.sqrt(max((w * (y - m) ** 2).sum(), 0.0))))


def p_greater(ya, wa, yb, wb):
    """P(Y_a > Y_b) for independent grid posteriors (ties count half)."""
    o = np.argsort(yb)
    yb, cb = yb[o], np.cumsum(wb[o])
    lo = np.searchsorted(yb, ya, side="left")
    hi = np.searchsorted(yb, ya, side="right")
    below = np.where(lo > 0, cb[np.maximum(lo - 1, 0)], 0.0)
    upto = np.where(hi > 0, cb[np.maximum(hi - 1, 0)], 0.0)
    return float((wa * (below + 0.5 * (upto - below))).sum())


def thompson(posts, n_slots, rng, floors=None):
    """slots by Thompson draws of Y: posts = {author: (y, w)}; floors = {author: minimum slots}."""
    names = list(posts)
    got = {a: int((floors or {}).get(a, 0)) for a in names}
    for _ in range(max(n_slots - sum(got.values()), 0)):
        draws = [rng.choice(posts[a][0], p=posts[a][1]) for a in names]
        got[names[int(np.argmax(draws))]] += 1
    return got


def decide(post_ch, post_inc, alpha):
    p = p_greater(*post_ch, *post_inc)
    return ("switch" if p >= 1 - alpha else "drop" if p <= alpha else "stay"), p


# ---------------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    rng = np.random.default_rng(11)
    alpha, s1, s_full = 0.10, 0.047, 0.024          # rung-1 se (a quarter of 444 judge rows), full-read se

    def draw(kind, K):
        if kind == "zero":   th = np.zeros(K)                                                 # proposes no-ops
        elif kind == "timid": th = rng.normal(0.002, 0.003, K)
        elif kind == "base":  th = rng.normal(-0.005, 0.015, K)
        elif kind == "bold":  th = np.where(rng.random(K) < 0.2, rng.normal(0.05, 0.015, K), rng.normal(-0.015, 0.01, K))   # mean -0.002
        elif kind == "good":  th = rng.normal(0.010, 0.020, K)
        elif kind == "bad":   th = rng.normal(-0.020, 0.010, K)
        return th, th + rng.normal(0, s1, K)

    def true_y(kind, n=400000):
        th, _ = draw(kind, n)
        D = th + rng.normal(0, s_full, n)
        return float((th * (D / s_full >= norm.ppf(1 - alpha))).mean())               # lone-candidate accept rule as the reference

    kinds = ["zero", "timid", "base", "bold", "good", "bad"]
    print("true accepted gain per slot (lone-candidate rule), pp:", {k: round(100 * true_y(k), 3) for k in kinds})
    print("true mean gain per slot, pp:", {k: round(100 * float(draw(k, 400000)[0].mean()), 3) for k in kinds})
    for K in (5, 8):
        for inc, ch in [("base", "base"), ("base", "good"), ("base", "bold"), ("timid", "bold"), ("zero", "bold"), ("base", "bad"), ("base", "zero"), ("bold", "timid")]:
            res = []
            for _ in range(R):
                Di, Dc, out = [], [], 0
                for g in range(1, 9):
                    Di += list(draw(inc, K)[1]); Dc += list(draw(ch, K)[1])
                    pool = np.array(Di + Dc)
                    lp = unit_prior(pool, s1 ** 2 * np.eye(len(pool)))
                    pi = yield_posterior(np.array(Di), s1 ** 2 * np.eye(len(Di)), s_full, alpha, lp)
                    pc = yield_posterior(np.array(Dc), s1 ** 2 * np.eye(len(Dc)), s_full, alpha, lp)
                    d, p = decide(pc, pi, alpha)
                    if d != "stay":
                        out = g if d == "switch" else -g
                        break
                res.append(out)
            res = np.array(res)
            print(f"K {K}  incumbent {inc:5s} challenger {ch:5s}:  P(switch by gen 2/4/8) " + " ".join(f"{((res > 0) & (res <= g)).mean():.2f}" for g in (2, 4, 8))
                  + "   P(drop by 2/4/8) " + " ".join(f"{((res < 0) & (-res <= g)).mean():.2f}" for g in (2, 4, 8)), flush=True)
