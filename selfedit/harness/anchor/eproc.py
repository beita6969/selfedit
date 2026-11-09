# selfedit 1109c: NEW (design_1109c §6.2). Judge ladder, picture-cluster bootstrap, sign-flip size, betting e-process. Pure numpy; no model calls.
from __future__ import annotations

import zlib

import numpy as np

# Pre-registered ladder (§6.2). q is the one-sided lower-bound quantile of the picture-cluster bootstrap of the mean per-row pool gain on R.
LADDER = {
    "J1": dict(z=1.96, q=0.05, level=0.95),
    "J2": dict(z=1.645, q=0.05, level=0.95),
    "J3": dict(z=1.28, q=0.10, level=0.90),
}
ORDER = ("J1", "J2", "J3")          # strictest first
CONFIRM_E = 20.0                     # Ville, alpha = .05
LAMBDA_MAX = 0.5
LAMBDA_FIRST = 0.1                   # lambda of read 1 (no earlier read to fit it on); pre-registered
N_CHUNKS = 4                         # replay: R split into picture-disjoint chunks, each a "read" of the e-process
BOOT_B = 2000


def cluster_index(clusters):
    """clusters: list of hashable cluster ids per row -> (int codes, number of clusters)."""
    u = {}
    codes = np.array([u.setdefault(c, len(u)) for c in clusters], int)
    return codes, len(u)


def cluster_boot_lower(d, clusters, q, B=BOOT_B, seed=0):
    """q-quantile of the picture-cluster bootstrap distribution of mean(d): pictures resampled with replacement, the mean taken over all rows drawn."""
    d = np.asarray(d, float)
    if len(d) == 0:
        return float("nan")
    codes, n = cluster_index(clusters)
    s = np.bincount(codes, weights=d, minlength=n)          # per-cluster sums
    c = np.bincount(codes, minlength=n).astype(float)        # per-cluster row counts
    rng = np.random.default_rng(seed)
    W = rng.multinomial(n, np.full(n, 1.0 / n), size=int(B)).astype(float)     # cluster draw counts
    means = (W @ s) / np.maximum(W @ c, 1e-12)
    return float(np.quantile(means, q))


def cluster_boot_p(d, clusters, B=BOOT_B, seed=0):
    """One-sided picture-cluster bootstrap p of mean(d) <= 0: (#{bootstrap means <= 0} + 1) / (B + 1). Same draws as cluster_boot_lower with the same seed."""
    d = np.asarray(d, float)
    if len(d) == 0:
        return 1.0
    codes, n = cluster_index(clusters)
    s = np.bincount(codes, weights=d, minlength=n)
    c = np.bincount(codes, minlength=n).astype(float)
    rng = np.random.default_rng(seed)
    W = rng.multinomial(n, np.full(n, 1.0 / n), size=int(B)).astype(float)
    means = (W @ s) / np.maximum(W @ c, 1e-12)
    return float((np.sum(means <= 0) + 1) / (B + 1))


def fired_z(d_cand, d_null_list, scale=None):
    """Paired fired-set z with sigma from the null redraws, not from the candidate: z = sum(d_cand) / sqrt(mean_j sum_i d_null_j,i^2).
    Under H0 a row's difference is symmetric about 0, so its variance is E[d^2], estimated from the parent's real re-seeded draws on the same rows."""
    d = np.asarray(list(d_cand), float)
    if len(d) == 0 or not d_null_list:
        return dict(n=len(d), net=float(d.sum()) if len(d) else 0.0, sigma=None, z=None)
    sc = list(scale) if scale is not None else [1.0] * len(d_null_list)
    v = float(np.mean([float(np.sum(np.asarray(x, float) ** 2)) * f for x, f in zip(d_null_list, sc)]))
    sig = np.sqrt(v) if v > 0 else None
    return dict(n=len(d), net=float(d.sum()), sigma=sig, z=None if sig is None else float(d.sum() / sig))


def harm_threshold(null_harms, k=1.65):
    """No-harm bound from the parent's real redraws: mean_j R_j + k sqrt(mean_j R_j) (a count-scale band; needs >= 1 redraw)."""
    x = [float(v) for v in null_harms if v is not None]
    if not x:
        return None
    m = float(np.mean(x))
    return m + k * np.sqrt(max(m, 1e-12))


def chunks_of(ids, clusters, n=N_CHUNKS, salt="1109c"):
    """Picture-disjoint chunk index per row (crc32 of the cluster)."""
    return np.array([zlib.crc32(f"{salt}|{c}".encode()) % n for c in clusters], int)


def eprocess(dbar, lam_first=LAMBDA_FIRST, lam_max=LAMBDA_MAX):
    """Betting e-process over reads j: E = prod_j (1 + lambda_j dbar_j), dbar_j in [-1, 1] (clipped), lambda_j predictable:
    fitted on reads before j as sum(dbar)/sum(dbar^2), clipped to [0, lam_max]; read 1 uses lam_first."""
    E, path, lams = 1.0, [], []
    past = []
    for x in dbar:
        x = float(np.clip(x, -1.0, 1.0))
        if past:
            num, den = float(np.sum(past)), float(np.sum(np.square(past)))
            lam = float(np.clip(num / den if den > 0 else 0.0, 0.0, lam_max))
        else:
            lam = float(lam_first)
        E *= 1.0 + lam * x
        lams.append(lam); path.append(E); past.append(x)
    return dict(E=E, path=path, lambdas=lams)


def eprocess_status(dbar, **kw):
    """confirmed (E >= 20), removed (E < 1 after >= 3 reads, or the running mean's 95% upper bound below 0 after >= 2 reads), else open."""
    e = eprocess(dbar, **kw)
    x = np.clip(np.asarray(dbar, float), -1, 1)
    st = "open"
    if e["E"] >= CONFIRM_E:
        st = "confirmed"
    elif len(x) >= 3 and e["E"] < 1:
        st = "removed"
    elif len(x) >= 2:
        ub = x.mean() + 1.96 * x.std(ddof=1) / np.sqrt(len(x))
        if ub < 0:
            st = "removed"
    return dict(e, status=st, reads=len(x))


def signflip_size(test, d_rows, clusters, n=2000, seed=0):
    """Size of a candidate test under H0 by picture-clustered sign flips. d_rows: dict name -> per-row array (aligned to clusters), all flipped by the
    same cluster signs (e.g. CE gain and accuracy change). test(flipped dict) -> bool. Returns the share of flips the test passes."""
    codes, k = cluster_index(clusters)
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(int(n)):
        s = rng.choice((-1.0, 1.0), size=k)[codes]
        if test({nm: np.asarray(v, float) * s for nm, v in d_rows.items()}):
            hits += 1
    return hits / float(n)


def mcnemar_rows(ref_right, cand_right):
    """VH stats.mcnemar_exact on per-row right/wrong of the pool without (ref) and with (cand) the candidate."""
    from .stats import mcnemar_exact
    return mcnemar_exact([bool(x) for x in ref_right], [bool(x) for x in cand_right])


# ---------------------------------------------------------------- amendment A3 (fable_1109d §1.2, patch 1): per-fired-cluster betting --------------
LAMBDA_FIRST_ROWS = 0.2
LAMBDA_MAX_ROWS = 0.6
MAX_READS = 5


def fired_cluster_d(ids, clusters, d_acc, fired):
    """One bet per fired picture-cluster of one read, in a fixed order (sorted cluster key): d = mean accuracy change on the cluster's fired rows, in [-1, 1]."""
    by = {}
    fs = set(fired)
    for i, c, d in zip(ids, clusters, d_acc):
        if i in fs:
            by.setdefault(str(c), []).append(float(d))
    return [float(np.clip(np.mean(by[c]), -1, 1)) for c in sorted(by)]


def eprocess_rows(reads, lam_first=LAMBDA_FIRST_ROWS, lam_max=LAMBDA_MAX_ROWS):
    """reads: [[d per fired cluster] per read], in read order. E = prod over every fired cluster of (1 + lambda d); lambda predictable: fitted on
    the clusters bet before it (sum d / sum d^2, clipped to [0, lam_max]); the first uses lam_first."""
    flat = [x for r in reads for x in r]
    e = eprocess(flat, lam_first=lam_first, lam_max=lam_max)
    return dict(e, bets=len(flat))


def eprocess_rows_status(reads, **kw):
    """confirmed at E >= 20 (anytime-valid at .05); removed at E < 1 after >= 3 reads, or after MAX_READS reads unconfirmed; else open."""
    e = eprocess_rows(reads, **kw)
    st = "open"
    if e["E"] >= CONFIRM_E:
        st = "confirmed"
    elif (len(reads) >= 3 and e["E"] < 1) or len(reads) >= MAX_READS:
        st = "removed"
    return dict(e, status=st, reads=len(reads))
