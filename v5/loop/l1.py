"""L1 glue for the v4 loop (AM7): author yield posteriors from the ledger, the challenger decision and Thompson slots.

Inputs are the judged candidates of the window: each has author, gen, judge {D, se} and dw {d: {cluster: d}, w: {cluster: rows}} (gen.py stores
dw for every judged candidate). D and the cluster-robust S over ALL candidates of the window come from judge_eb.family, so shared rows and
shared parent samples are not counted twice; each author's block of S goes to yield_l1. sigma = median full-read se of the window. Module
yield_l1 is used unchanged."""
import numpy as np

import judge_eb as EB
import yield_l1 as YL


def judged(cands):
    """Candidates with a judgement on the per-routed-row scale (AM14): dw.scale == "routed"."""
    return [c for c in cands if (c.get("judge") or {}).get("D") is not None and c.get("dw") and c["dw"].get("d") and c["dw"].get("scale") == "routed"]


def family_all(cands):
    """d (K,), S (K, K) over all judged candidates given (in order), per routed row."""
    xs = [dict(name=c["cid"], d=c["dw"]["d"], w=c["dw"]["w"]) for c in cands]
    D, S, _ = EB.family(xs)
    return D, S


def posteriors(cands, authors, alpha):
    """{author: (y, w)} over the window with yield_l1.yield_posterior_routed (AM14): d and S per routed row, shares = routed share,
    log prior = unit-information prior over all candidates of the window on the same scale. Authors without a judged candidate get
    Y = 0 everywhere (the routed yield needs the author's own reach profile). Also returns a summary per author (K, mean, sd of Y)."""
    C = judged(cands)
    out, summ = {}, {}
    D, S = family_all(C) if C else (np.zeros(0), np.zeros((0, 0)))
    lp = YL.unit_prior(D, S) if C else None
    for a in authors:
        ix = [i for i, c in enumerate(C) if c["author"] == a]
        sh = np.array([float(C[i]["share"]) for i in ix])
        y, w = YL.yield_posterior_routed(D[ix], S[np.ix_(ix, ix)], sh, alpha, lp)
        out[a] = (y, w)
        summ[a] = dict(K=len(ix), **YL.summary(y, w))
    return out, summ


def decide(posts, inc, ch, alpha):
    d, p = YL.decide(posts[ch], posts[inc], alpha)
    return d, p


def slots(posts, n, rng, floors):
    """Thompson draws of Y for n slots with per-author floors."""
    return YL.thompson(posts, n, rng, floors=floors)
