# selfedit 1109c: NEW adapter (§8.2 item 4, PoolScore). The pooled out-of-fold read of a candidate on the unseen evidence rows R (§6.2 item 3).
# S = refit DS pool accuracy. FIT rows of R (score rows outside the stage-1 draw) are read on the pool's picture-held-out F_oof (ds_agree's own folds),
# SELECT rows on F (never fitted), so no row of R is scored by a pool fitted on its own gold.
from __future__ import annotations

import numpy as np


def held_F(fu):
    """F with every FIT row replaced by its picture-held-out F_oof (as hboost.pool_cols)."""
    G = np.array(fu.F, float, copy=True)
    fit = np.asarray(fu.sp.fit, int)
    if getattr(fu, "F_oof", None) is not None and len(fit):
        G[fit] = fu.F_oof
    return G


def pool_answers(H, fu, members, votes, held=True):
    """THE pool answer column per Table row (1109c-final 11-10, one helper for every pool read): with ONE member the pool is that member's own
    plurality (H.maj_col of its samples) -- the DS-fused column relabels it (r1: pool FIT .660 vs stock .618); else the fused column, FIT rows held
    out by picture when held (as hboost.pool_cols)."""
    tab = fu.tab
    if len(members) == 1:
        v = votes[members[0]["name"]]
        a = np.full(len(tab.ids), -1, dtype=int)
        for i, k in tab.index.items():
            a[k] = H.maj_col(tab, k, v.get(i))
        return a
    return H.answer_of(tab, held_F(fu) if held else fu.F)


def rows_of(fu, ids):
    idx = np.array([fu.tab.index[i] for i in ids], int)
    return idx


def read_R(H, run, members, cand, votes, R_ids, base=None):
    """Pool without and with cand, refit on the same Table, read on R. Returns per-row arrays aligned to R_ids (rows not in the Table dropped) and totals.
    base: an earlier f0 on the same votes (reused across candidates)."""
    from vrsi27.boost.edge import row_ce
    f0 = base if base is not None else run.fz("fuse", run.fuser.fuse, members, votes, run.data, run.fusion)
    f1 = run.fz("fuse", run.fuser.fuse, list(members) + [cand], votes, run.data, run.fusion, sp=f0.sp)
    ids = [i for i in R_ids if i in f0.tab.index]
    idx = rows_of(f0, ids)
    G0, G1 = held_F(f0), held_F(f1)
    y, mk = f0.tab.y[idx], f0.tab.mask[idx]
    ce0, ce1 = row_ce(G0[idx], y, mk), row_ce(G1[idx], y, mk)
    pa = pool_answers(H, f0, members, votes)
    a0, a1 = pa[idx], pool_answers(H, f1, list(members) + [cand], votes)[idx]
    r0, r1 = (a0 >= 0) & (a0 == y), (a1 >= 0) & (a1 == y)
    return dict(ids=ids, clusters=[str(run.data.cluster(i)) for i in ids], d_ce=ce0 - ce1, d_acc=r1.astype(float) - r0.astype(float),
                right0=r0, right1=r1, S0=float(r0.mean()) if len(ids) else None, S1=float(r1.mean()) if len(ids) else None,
                up=int((r1 & ~r0).sum()), down=int((r0 & ~r1).sum()), f0=f0, f1=f1, pa=pa)


def pool_S(H, run, members, votes, R_ids):
    """Refit pool accuracy of a member set on R (held-out F_oof on FIT rows): an RRSI evaluation of the incumbent harness (one trial per row)."""
    fu = run.fz("fuse", run.fuser.fuse, members, votes, run.data, run.fusion)
    ids = [i for i in R_ids if i in fu.tab.index]
    idx = rows_of(fu, ids)
    a = pool_answers(H, fu, members, votes)[idx]      # 1109c audit 11-09 / 11-10: a one-member pool is that member's plurality
    y = fu.tab.y[idx]
    return ids, ((a >= 0) & (a == y)).astype(float)


def mean_tokens(v, ids):
    """C: mean generated tokens per answered sample of a vote dict on ids (None if no counts)."""
    t = [float(s.get("tok") or 0) for i in ids for s in (v.get(i) or {}).values() if s is not None and (s.get("tok") or 0) > 0]
    return float(np.mean(t)) if t else None


def read_lobo(H, run, members, cand, votes, R_ids, cache=None):
    """fable_1109d patch 15 (LOBO transfer read): for each bench b of R, the pool without and with cand is refit with every FIT row of b left out, and
    read on b's R rows with F (b never fitted, so the read is a transfer read for b). Returns {bench: dict(rows, inc, up, down)}.
    cache: dict for the members-only fits (shared across candidates of one round)."""
    import dataclasses
    f0 = run.fz("fuse", run.fuser.fuse, members, votes, run.data, run.fusion)
    tab, sp = f0.tab, f0.sp
    ids = [i for i in R_ids if i in tab.index]
    bench = {i: run.data.bench_of(i) for i in ids}
    out = {}
    for b in sorted(set(bench.values())):
        fit_b = np.array([k for k in sp.fit if run.data.bench_of(tab.ids[k]) != b], int)
        spb = dataclasses.replace(sp, fit=fit_b)
        key = (b, tuple(sorted(m["name"] for m in members)))
        if cache is not None and key in cache:
            g0 = cache[key]
        else:
            g0 = run.fz("fuse", run.fuser.fuse, members, votes, run.data, run.fusion, sp=spb)
            if cache is not None:
                cache[key] = g0
        g1 = run.fz("fuse", run.fuser.fuse, list(members) + [cand], votes, run.data, run.fusion, sp=spb)
        rows = [i for i in ids if bench[i] == b]
        idx = rows_of(g0, rows)
        y = g0.tab.y[idx]
        a0, a1 = pool_answers(H, g0, members, votes, held=False)[idx], pool_answers(H, g1, list(members) + [cand], votes, held=False)[idx]
        r0, r1 = (a0 >= 0) & (a0 == y), (a1 >= 0) & (a1 == y)
        out[b] = dict(rows=len(rows), inc=float((r1.astype(float) - r0).mean()) if rows else 0.0, up=int((r1 & ~r0).sum()), down=int((r0 & ~r1).sum()))
    return out
