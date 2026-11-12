"""Family judge (design_1110_hl §3.1): one rule for every operator family, one parameter alpha.

Each operator family (same operator class, same parent) is treated as a population of edits. For candidate j, D_j is its paired gain over its
parent on the rows it answered (mean over picture clusters, rows as weights). The sampling covariance S of the D's is the cluster-robust one, so
candidates read on the same rows (and against the same parent samples) are not counted as independent evidence. Model:

    D | theta ~ N(theta, S),      theta_j ~ N(mu, tau^2)  (family mean and spread)

mu and tau get flat priors on their whole possible range (an accuracy difference lies in [-1, 1]) and are integrated out on a grid. The decision
quantity is the local false sign rate lfsr_j = P(theta_j <= 0 | all D of the family) (Stephens 2017). Accept iff lfsr_j <= alpha.

Limits: a lone candidate gets no shrinkage (tau is unidentified, the flat prior leaves D_j as it is) and lfsr is the one-sided p of its own
cluster test; a family of consistent small gains is accepted on the family mean (the polygenic case); a lone lucky draw in a family whose other
members sit at zero is pulled back toward zero. No fired / net / zx / cap / BH constants are left."""
import math

import numpy as np

try:
    from scipy.stats import t as _t
except Exception:                                    # scipy is in every venv we run; the normal tail is the fallback
    _t = None


def _tail(z, nu):
    """P(T_nu <= -z): one-sided tail with nu degrees of freedom (clusters - 1); normal when nu is large or scipy is missing."""
    z = np.asarray(z, float)
    if _t is None:
        return 0.5 * np.vectorize(math.erfc)(z / math.sqrt(2.0))
    return _t.cdf(-z, np.maximum(np.asarray(nu, float), 1.0))


def contributions(d, w):
    """d[c] = mean paired difference in picture cluster c (candidate right - parent right, over the paired seeds), w[c] = rows of the cluster.
    Returns D (weighted mean) and e[c] (linearised cluster residuals, sum of squares = cluster-robust variance of D before the small-sample factor)."""
    d, w = np.asarray(d, float), np.asarray(w, float)
    W = w.sum()
    D = float((w * d).sum() / W)
    return D, w * (d - D) / W


def family(cands):
    """cands: list of dict(name, d={cluster: mean diff}, w={cluster: rows}). Returns D (K,), S (K,K) cluster-robust covariance, nu (K,) = clusters - 1."""
    K = len(cands)
    clusters = sorted({c for x in cands for c in x["d"]})
    ix = {c: i for i, c in enumerate(clusters)}
    E = np.zeros((len(clusters), K))
    D, nu = np.zeros(K), np.zeros(K)
    for j, x in enumerate(cands):
        cs = sorted(x["d"])
        Dj, e = contributions([x["d"][c] for c in cs], [x["w"][c] for c in cs])
        D[j] = Dj
        n = len(cs)
        nu[j] = max(n - 1, 1)
        E[[ix[c] for c in cs], j] = e * math.sqrt(n / max(n - 1, 1))
    return D, E.T @ E, nu


def lfsr(D, S, nu=None, n_mu=161, n_tau=81):
    """Posterior sign error P(theta_j <= 0 | D) and posterior mean of theta under the family model, hyperparameters integrated on a grid.
    Flat prior on mu in [-1, 1] and on tau in [0, 1] (the range of an accuracy difference); the grid is refined where the likelihood lives."""
    D, S = np.asarray(D, float), np.asarray(S, float)
    K = len(D)
    nu = np.full(K, 1e9) if nu is None else np.asarray(nu, float)
    sd = np.sqrt(np.maximum(np.diag(S), 1e-12))
    lo, hi = max(-1.0, float(D.min() - 6 * sd.max())), min(1.0, float(D.max() + 6 * sd.max()))
    mus = np.linspace(lo, hi, n_mu)
    # tau grid: uniform prior on [0, 1]; points spaced as t^2 so that small tau is resolved, weight = d tau = 2 t dt
    t = np.linspace(0.0, 1.0, n_tau + 1)[1:]
    taus, wt = t ** 2, 2 * t
    one = np.ones(K)
    logw = np.empty((n_tau, n_mu))
    post = []
    for a, tau in enumerate(taus):
        V = S + tau ** 2 * np.eye(K)
        L = np.linalg.cholesky(V + 1e-12 * np.eye(K))
        Vi = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(K)))
        logdet = 2 * np.log(np.diag(L)).sum()
        A = tau ** 2 * Vi                               # theta | D, mu, tau ~ N(mu + A (D - mu), tau^2 (I - A))
        pv = np.maximum(tau ** 2 * (1 - np.diag(A)), 1e-14)
        q1, q2, q3 = D @ Vi @ D, one @ Vi @ D, one @ Vi @ one
        logw[a] = -0.5 * (q1 - 2 * mus * q2 + mus ** 2 * q3) - 0.5 * logdet + math.log(wt[a])
        post.append((A, pv))
    w = np.exp(logw - logw.max())
    w /= w.sum()
    ls, mean = np.zeros(K), np.zeros(K)
    for a, (A, pv) in enumerate(post):
        keep = w[a] > 1e-9
        if not keep.any():
            continue
        AD, A1 = A @ D, A @ one
        m = mus[keep, None] * (1 - A1)[None, :] + AD[None, :]              # (grid mu, K)
        ls += (w[a, keep, None] * _tail(m / np.sqrt(pv)[None, :], nu[None, :])).sum(0)
        mean += (w[a, keep, None] * m).sum(0)
    return ls, mean, dict(mu=float((w.sum(0) * mus).sum()), tau=float((w.sum(1) * taus).sum()))


def judge(cands, alpha):
    """Decision for one family. Returns one row per candidate: D, se, lfsr, post_mean, accept."""
    if not cands:
        return [], {}
    D, S, nu = family(cands)
    ls, mean, hyper = lfsr(D, S, nu)
    rows = [dict(name=x["name"], D=float(D[j]), se=float(math.sqrt(S[j, j])), clusters=int(nu[j] + 1), lfsr=float(ls[j]), post_mean=float(mean[j]),
                 accept=bool(ls[j] <= alpha)) for j, x in enumerate(cands)]
    return rows, hyper


MU_GRID = np.linspace(-0.5, 0.5, 401)         # a13e: a fixed mu grid (accuracy difference per row) so that a posterior can be the next family's prior


def mu_posterior(D, S, nu=None, prior=None, mus=MU_GRID, n_tau=81):
    """a13e (slot allocation, coordinator 11-10): the posterior of the family mean mu on a fixed grid, the same model as lfsr (theta_j ~ N(mu, tau^2),
    D | theta ~ N(theta, S), tau uniform on [0, 1] integrated on the same grid); prior: weights on mus (default flat). K = 0 returns the prior.
    Used for slots only: the decisions stay with lfsr / judge."""
    mus = np.asarray(mus, float)
    p0 = np.ones(len(mus)) if prior is None else np.maximum(np.asarray(prior, float), 0.0)
    if D is None or len(D) == 0:
        return p0 / p0.sum()
    D, S = np.asarray(D, float), np.asarray(S, float)
    K = len(D)
    t = np.linspace(0.0, 1.0, n_tau + 1)[1:]
    taus, wt = t ** 2, 2 * t
    one = np.ones(K)
    logw = np.empty((n_tau, len(mus)))
    for a, tau in enumerate(taus):
        V = S + tau ** 2 * np.eye(K)
        L = np.linalg.cholesky(V + 1e-12 * np.eye(K))
        Vi = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(K)))
        q1, q2, q3 = D @ Vi @ D, one @ Vi @ D, one @ Vi @ one
        logw[a] = -0.5 * (q1 - 2 * mus * q2 + mus ** 2 * q3) - np.log(np.diag(L)).sum() + math.log(wt[a])
    w = np.exp(logw - logw.max()).sum(0) * p0
    if w.sum() <= 0:
        return p0 / p0.sum()
    return w / w.sum()
