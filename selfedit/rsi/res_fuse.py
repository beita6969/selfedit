"""Fusion step of the resident build. The pool's F comes from here, chosen by prereg "fusion".

CALL SIGNATURES (stable; the driver codes against these)

    fuse(members, votes, data, name="stacking", ids=None, sp=None, beside=(), **params) -> Fused
        members   the pool, list of member dicts. Refit from them on every call; nothing is cached between calls.
        votes     {name: load_member(m)}, the inheriting form, as today. A name in votes but not in members adds Table columns only, as Run.compute's extra. ds_agree cuts gate members to their scope itself (fusion_votes); the caller passes nothing extra.
        data      hboost.Data.
        name      "stacking" | "ds_agree"; the run takes it from mode_of(prereg).
        ids       Table rows, default data.learn. Rows outside sp.fit are scored, never fitted.
        sp        Splits, default hboost.splits(data, tab); accept_in_order's refit passes dataclasses.replace(sp, fit=sp.score). Any fit or score row whose role is not show or score raises ValueError.
        beside    other modes to compute on the same Table, e.g. ("stacking",): result.beside["stacking"] is a Fused.
    Fused: .F [rows, C] log posterior (exp sums to 1 over tab.mask on every row, -inf outside it); .F_oof [len(sp.fit), C] the same on FIT, held out by picture for ds_agree, F[sp.fit] for stacking, the shape of st.cur.F_oof;
           .tab .pool .sp .info .reads .beside; .acc(idx) = tab.acc(F, idx); .numbers() = dict(fit, select, n_fit, n_select); .answers() column answered per row, as Table.acc reads it.
    mode_of(prereg) -> "stacking" unless prereg["fusion"] says otherwise.
    increment(members, cand, votes, data, name="stacking", sp=None, base=None, seed=0) -> dict(ce_gain_select, ce_gain_select_lb, dacc_select)
        The pool increment of one candidate. "stacking": edge.learn_gain's own numbers. "ds_agree": fuse(members) against fuse(members + [cand]) on the same Table, same formula: boot_lower(row_ce(F0[sel]) - row_ce(F1[sel]), BOOT_Q, BOOT_B, seed). The threshold stays lb > 0. base: a Fused of members, to save its refit.
    pool_state(st, fused) -> RoundState with cur.F, cur.F_oof and g from this pool, for residual cards (lens.py:195,197) and err_corr_pool. Not for evidence_test: q, s and z_select keep the log-linear st.

"stacking" is the driver's path unchanged: Table -> Table.pool -> state_of -> round_state(evidence) -> st.cur.F.
"ds_agree" is the frozen form of dry/cons_id_freeze.py, supervised: a row where every valid sample of every answering member picks one column answers that column; other rows answer by Dawid-Skene with full letter confusion per member and fine class (bench|category|kind) on mc rows,
one-coin per member and fine class on open rows, MAP with kp=10 toward that member's overall table (dry/cons_algos.ibcc, level="fine", vb=False, scope dg), fit on the disagreement rows of sp.fit. Every EM row has gold, so EM stops after one pass.

Posterior added to that form, answers unchanged (dry/test_res_fuse.py checks the answers are bit-equal to the frozen code):
- Disagreement and no-vote rows: softmax of (ll + tie) / T. ll treats samples as independent, so one temperature T is fit on FIT disagreement rows held out by picture (crc32(cluster) % 5). T > 0 keeps the order.
- Agreed rows: P(agreed column) = a, the rest (1 - a) / (L - 1). a is the agreed column's accuracy per fine class on FIT's agreed rows, kp=10 toward the overall rate with base (2, 1), the frozen one-coin form. Floored at 1/L + 1e-6 so the agreed column stays on top; info["agree_floored"] counts rows.

Gate members in ds_agree: load_member gives a gate member its parent's answers outside its scope, which would count the parent twice. fusion_votes keeps only rows with ROWKEY == gate_scope(m); elsewhere the member abstains. Same rows and records as load_raw filtered that way, and no disk read.
Table, stacking, zx, member stats and cards keep the inheriting form. A member with gate_scope None (no gate, inherit_off) keeps all its votes.

Unseen class: a class with no FIT disagreement row has N = 0, so A = kp * overall and its table is that member's overall confusion (lines marked BACK-OFF); same for a.
Gold read: y on sp.fit only. Rows of SELECT, half 1 and OOD enter no fit: neither gold nor answers.
Gates that read log-linear columns stay on state_of.
"""
import dataclasses, sys, zlib
import numpy as np

REGISTRY = {}
FIT_ROLES = ("show", "score")
TABLE_READS = "Table, shared by every mode: on each of its rows the gold text and each vote's graded flag form the columns of an open row (right spellings merge into =gold, gold's column exists even if nobody else voted it). No fit reads those rows' y unless they are in sp.fit; Table.acc reads y to score."


def register(name, reads):
    """reads: plain statement of what the mode reads, carried on every result."""
    def deco(fn):
        REGISTRY[name] = dict(fn=fn, reads=reads)
        return fn
    return deco


VOTED_LETTERS = False       # 128 prereg ds_voted_letters (coordinator 11-08): an mc row the members disagree on answers only a letter some member voted. Off: byte-identical to ae113307
VOTED_EPS = 1e-3            # what the unvoted letters of such a row keep, shared: CE stays finite where gold was voted by nobody


def voted_only(F, tot, mc, agreecol, other, eps=VOTED_EPS):
    """F with each non-agreed mc row that has a vote cut to its voted letters: their posterior scaled to 1 - eps, eps shared by the unvoted letters of the row. Agreed rows, rows with no vote and open rows stay. Returns (F', rows touched)."""
    rows = mc & (tot.sum(1) > 0) & ~passes(agreecol, other)
    G = F.copy()
    for k in np.flatnonzero(rows):
        m = np.isfinite(F[k])
        v, u = m & (tot[k] > 0), m & ~(tot[k] > 0)
        if not u.any() or not v.any():
            continue
        q = np.exp(F[k, v] - F[k, v].max())
        G[k] = -np.inf
        G[k, v] = np.log(q / q.sum() * (1 - eps))
        G[k, u] = np.log(eps / u.sum())
    return G, rows


def answer_cols(F, mask, other, novote=None):
    """Column answered on each row, as Table.acc reads it: -1 where no voted column is left, or (novote) no member answered."""
    F = np.where(mask, F, -np.inf).copy()
    r = np.arange(len(F))
    ot = other >= 0
    F[r[ot], other[ot]] = -np.inf
    a = np.where(np.isfinite(F.max(1)), F.argmax(1), -1)
    return a if novote is None else np.where(novote, -1, a)


def mode_of(prereg):
    """The prereg field. Absent means stacking."""
    return (prereg or {}).get("fusion") or "stacking"


def _h():
    """The hboost in use: the driver runs as __main__, and a second import would hold its own globals."""
    m = sys.modules.get("__main__")
    if m is not None and hasattr(m, "Table") and hasattr(m, "state_of"):
        return m
    import hboost
    return hboost


@dataclasses.dataclass
class Inp:
    members: list
    votes: dict
    data: object
    tab: object
    pool: object
    sp: object
    params: dict


@dataclasses.dataclass
class Fused:
    name: str
    F: np.ndarray            # [rows, columns], rows = tab.ids
    tab: object
    pool: object
    sp: object
    info: dict
    reads: str
    F_oof: np.ndarray = None
    beside: dict = dataclasses.field(default_factory=dict)

    def acc(self, idx):
        return self.tab.acc(self.F, idx)

    def numbers(self):
        return dict(fit=self.acc(self.sp.fit), select=self.acc(self.sp.select), n_fit=len(self.sp.fit), n_select=len(self.sp.select))

    def answers(self):
        """Column answered on each row: over voted columns, as Table.acc reads it."""
        F = np.where(self.tab.mask, self.F, -np.inf).copy()
        rows = np.arange(len(self.tab.ids))
        ot = self.tab.other >= 0
        F[rows[ot], self.tab.other[ot]] = -np.inf
        return np.where(getattr(self.tab, "novote", False) | ~np.isfinite(F.max(1)), -1, F.argmax(1))


def check_fit_rows(data, tab, sp):
    """No gold from SELECT, half 1 or OOD: those rows have another role or none."""
    for what, idx in (("fit", sp.fit), ("score", sp.score)):
        bad = [tab.ids[k] for k in np.asarray(idx, int) if data.role.get(tab.ids[k]) not in FIT_ROLES]
        if bad:
            raise ValueError(f"sp.{what} holds {len(bad)} rows outside {FIT_ROLES}, first {bad[:3]}")


def fuse(members, votes, data, name="stacking", ids=None, sp=None, beside=(), **params):
    for n in (name, *beside):
        if n not in REGISTRY:
            raise KeyError(f"fusion mode {n!r}, known {sorted(REGISTRY)}")
    h = _h()
    tab = h.Table(data, list(data.learn) if ids is None else list(ids), votes)
    pool = tab.pool(data, members, votes)
    sp = h.splits(data, tab) if sp is None else sp
    check_fit_rows(data, tab, sp)
    def one(n, par):
        r = REGISTRY[n]
        F, info = r["fn"](Inp(list(members), votes, data, tab, pool, sp, par))
        oof = info.pop("F_oof", None)
        return Fused(n, F, tab, pool, sp, info, r["reads"] + " " + TABLE_READS, F[np.asarray(sp.fit, int)] if oof is None else oof)
    out = one(name, params)
    out.beside = {n: one(n, {}) for n in beside if n != name}
    return out


def increment(members, cand, votes, data, name="stacking", sp=None, base=None, seed=0):
    """Pool increment of cand: what edge.py's _learn_gain_evidence writes to ce_gain_select, ce_gain_select_lb, dacc_select, with both pools from this mode."""
    from vrsi27.boost.edge import BOOT_B, BOOT_Q, boot_lower, learn_gain, row_ce
    h = _h()
    if name == "stacking":
        tab = h.Table(data, list(data.learn), votes)
        P = tab.pool(data, list(members) + [cand], votes)
        sp = h.splits(data, tab) if sp is None else sp
        check_fit_rows(data, tab, sp)
        g = learn_gain(P, sp, h.state_of(P, sp, members), len(members), combiner="evidence", parent=h.REF(members))
        return {k: g.get(k) for k in ("ce_gain_select", "ce_gain_select_lb", "dacc_select")}
    f0 = fuse(members, votes, data, name, sp=sp) if base is None else base
    f1 = fuse(list(members) + [cand], votes, data, name, sp=f0.sp)
    sel, y, mk = f0.sp.select, f0.tab.y, f0.tab.mask
    d = row_ce(f0.F[sel], y[sel], mk[sel]) - row_ce(f1.F[sel], y[sel], mk[sel])
    return dict(ce_gain_select=float(d.mean()), ce_gain_select_lb=boot_lower(d, BOOT_Q, BOOT_B, seed), dacc_select=f1.acc(sel) - f0.acc(sel))


def pool_state(st, fused):
    """The members' RoundState with this pool's scores and FIT residual. Residual cards read st.cur.F_oof and st.g (lens.py:195,197)."""
    from vrsi27.boost.edge import residual_of
    fit = fused.sp.fit
    return dataclasses.replace(st, cur=dataclasses.replace(st.cur, F=fused.F, F_oof=fused.F_oof), g=residual_of(fused.F_oof, fused.pool.y[fit], fused.pool.opt_mask[fit]))


# ---------------------------------------------------------------- stacking ---------------------------------------------------------------------
@register("stacking", "Answers: Table.module's two-sample vote distributions (A=.5) of the pool's members on every Table row. Gold: y on sp.fit only (T and kappa, evidence weights; round_state's second fit on sp.score, which lies inside fit and does not change F). Also each member's origin and which one is stock. Rows outside sp.fit are read only as inputs to the fitted weights, at answer time.")
def _stacking(x):
    st = _h().state_of(x.pool, x.sp, x.members)
    return st.cur.F, dict(weights={m["name"]: float(w) for m, w in zip(x.members, st.cur.w)}, state=st, F_oof=st.cur.F_oof)


# ---------------------------------------------------------------- ds_agree ---------------------------------------------------------------------
def fusion_votes(members, votes):
    """votes with each gate member cut to its scope. Outside it the member has no record, so it abstains (sample_columns -3)."""
    h = _h()
    out, scoped = dict(votes), {}
    for m in members:
        c = h.gate_scope(m)
        if c is not None:
            out[m["name"]] = {i: s for i, s in votes[m["name"]].items() if h.ROWKEY.get(i) == c}
            scoped[m["name"]] = dict(scope=c, rows=len(out[m["name"]]))
    return out, scoped


def sample_columns(tab, members, votes):
    """[M, N, 2] Table column of each sample. -1 no parseable answer, -2 sample absent or abstain, -3 member has no record on the row. Same codes as dry/cons_extract.py. Only values >= 0 count anywhere below."""
    V = np.full((len(members), len(tab.ids), 2), -3, int)
    for j, m in enumerate(members):
        mv = votes[m["name"]]
        for k, i in enumerate(tab.ids):
            s = mv.get(i)
            if s is None:
                continue
            for t in (0, 1):
                v = s.get(t)
                c = None if v is None else tab.column(k, v)
                V[j, k, t] = -2 if v is None else -1 if c is None else c
    return V


def agreement(V, C):
    """agreecol: the one column every valid sample of every answering member picks, else -1. dis: more than one column voted. A row with no vote has neither."""
    N = V.shape[1]
    tot = np.zeros((N, C))
    for m in range(V.shape[0]):
        for s in range(2):
            ok = V[m, :, s] >= 0
            np.add.at(tot, (np.flatnonzero(ok), V[m, ok, s]), 1)
    ncol = (tot > 0).sum(1)
    return np.where(ncol == 1, tot.argmax(1), -1), ncol > 1, tot


def ds_letters(V, Y, mask, mc, cells, tr, use, iters=20, kp=10.):
    """dry/cons_algos.ibcc(level="fine", vb=False, unsup=False), same equations. Y: gold on tr, -99 elsewhere. tr: rows clamped to gold. use: rows whose answers enter EM. Returns ll [N, C] from the last E step and the fitted tables."""
    M, N, _ = V.shape
    K = mask.shape[1]
    L = mask.sum(1)
    nc = int(cells.max()) + 1
    rows = np.union1d(np.asarray(use, int), np.asarray(tr, int)).astype(int)
    tr = np.asarray(tr, int)
    tot = agreement(V, K)[2] + .1
    p = np.where(mask, tot, 0.)
    q = np.zeros((N, K))
    q[rows] = p[rows] / p[rows].sum(1, keepdims=True)
    q[tr] = 0
    q[tr, Y[tr]] = 1
    pri = np.ones((K, K)) + np.eye(K)
    base_r = np.array([2., 1.])
    votes = [[[int(v) for v in V[m, i] if v >= 0] for i in range(N)] for m in range(M)]       # ABSTAIN: a member with no valid sample on a row adds no count and no factor there
    for it in range(iters):
        N_, R_, pi = np.zeros((M, nc, K, K)), np.zeros((M, nc, 2)), np.ones(K)
        for i in rows:
            if mc[i]:
                pi += q[i]
                for m in range(M):
                    for v in votes[m][i]:
                        N_[m, cells[i], :, v] += q[i]
            else:
                for m in range(M):
                    for v in votes[m][i]:
                        for t in np.flatnonzero(q[i, :L[i]] > 0):
                            R_[m, cells[i], 0 if v == t else 1] += q[i, t]
        NG, RG = N_.sum(1), R_.sum(1)
        A = N_ + kp * ((NG + pri) / (NG + pri).sum(2, keepdims=True))[:, None]       # BACK-OFF: N_ = 0 in an unseen class leaves kp * this member's overall letter confusion
        LE = np.log(A / A.sum(3, keepdims=True))
        B = R_ + kp * ((RG + base_r) / (RG + base_r).sum(1, keepdims=True))[:, None]       # BACK-OFF: same for its overall right rate on open rows
        LR = np.log(B / B.sum(2, keepdims=True))
        lpi = np.log(pi / pi.sum())
        ll = np.zeros((N, K))
        for i in range(N):
            n = int(L[i])
            if mc[i]:
                ll[i, :n] = lpi[:n]
                for m in range(M):
                    for v in votes[m][i]:
                        ll[i, :n] += LE[m, cells[i], :n, v]
            else:         # OPEN: columns have no shared meaning between rows, so one-coin: the voted column gets the right rate, each other column the wrong rate over L - 1. OTHER is a column here, as in the frozen code
                for m in range(M):
                    for v in votes[m][i]:
                        a = np.full(n, LR[m, cells[i], 1] - np.log(max(n - 1, 1)))
                        a[v] = LR[m, cells[i], 0]
                        ll[i, :n] += a
        qn = np.zeros_like(q)
        lr = np.where(mask[rows], ll[rows], -np.inf)
        lr = lr - lr.max(1, keepdims=True)
        pr = np.exp(lr)
        qn[rows] = pr / pr.sum(1, keepdims=True)
        qn[tr] = 0
        qn[tr, Y[tr]] = 1
        d = np.abs(qn - q).max()
        q = qn
        if d < 1e-3:
            break
    return ll, dict(iters=it + 1, LE=LE, LR=LR, lpi=lpi, cells=nc)


def tie_break(ids, C):
    """crc32(row id|column), blind to gold, as the study's tie."""
    return np.array([[zlib.crc32(f"{i}|{c}".encode()) / 2 ** 32 for c in range(C)] for i in ids]) * 1e-7


def fine_cells(data, ids):
    """bench|category|kind of each row, by index."""
    h = _h()
    names = [f"{h.vh.key(data.rows[i])}|{data.rows[i].get('kind')}" for i in ids]
    order = {n: k for k, n in enumerate(sorted(set(names)))}
    return np.array([order[n] for n in names], int)


def right_spellings(tab, members, votes):
    """Rows where two right spellings merge into Table's =gold column: the one place a gold flag reaches a column before any fit. Both modes read this table. {row index: keys}"""
    out = {}
    for k, i in enumerate(tab.ids):
        if tab.other[k] < 0:
            continue
        keys = {v["key"] for m in members for v in votes[m["name"]].get(i, {}).values() if v["ok"] and v["key"] is not None}
        if len(keys) > 1:
            out[k] = sorted(keys)
    return out


def passes(agreecol, other):
    """Rows that pass unchanged. An agreed OTHER column is no answer in any voted column, so that row keeps the posterior (0 such rows on main2's r2 members)."""
    return (agreecol >= 0) & (agreecol != other)


def agree_accuracy(right, cells, nc, kp=10.):
    """Accuracy of the agreed column per fine class, one-coin form of ds_letters. BACK-OFF: a class with no agreed FIT row takes the overall rate."""
    R = np.zeros((nc, 2))
    np.add.at(R, (cells, 0), right.astype(float))
    np.add.at(R, (cells, 1), (~right).astype(float))
    G = R.sum(0) + np.array([2., 1.])
    B = R + kp * G / G.sum()
    return B[:, 0] / B.sum(1)


def ds_fit(V, y, mask, mc, cells, agreecol, dis, other, fit, iters=20, kp=10.):
    """The frozen fit on rows fit, plus the agreed rows' accuracy. y is read on fit rows only."""
    fit = np.asarray(fit, int)
    tr = fit[dis[fit]]
    Y = np.full(V.shape[1], -99)
    Y[tr] = y[tr]
    ll, tables = ds_letters(V, Y, mask, mc, cells, tr, tr, iters=iters, kp=kp)
    pr = fit[passes(agreecol, other)[fit]]
    return ll, agree_accuracy(agreecol[pr] == y[pr], cells[pr], int(cells.max()) + 1, kp), tables


def posterior(ll, tie, mask, agreecol, other, cells, acc, T):
    """Log posterior on every row. Non-agreed rows: softmax((ll + tie) / T). Agreed rows: a on the agreed column, the rest shared."""
    from vrsi27.boost.calibrate import log_softmax_masked
    F = log_softmax_masked((ll + tie) / T, mask)
    pr = np.flatnonzero(passes(agreecol, other))
    L = mask.sum(1)[pr]
    a = np.maximum(acc[cells[pr]], 1. / L + 1e-6)
    F[pr] = np.where(mask[pr], (np.log1p(-a) - np.log(np.maximum(L - 1, 1)))[:, None], -np.inf)
    F[pr, agreecol[pr]] = np.log(a)
    return F, int((acc[cells[pr]] < 1. / L + 1e-6).sum())


def fit_temperature(ll, y, mask):
    """T minimising CE of softmax(ll / T) on the given held-out rows. None of them: 1."""
    from scipy.optimize import minimize_scalar
    from vrsi27.boost.calibrate import log_softmax_masked
    if not len(y):
        return 1.
    r = np.arange(len(y))
    f = lambda lt: -log_softmax_masked(ll / np.exp(lt), mask)[r, y].mean()
    return float(np.exp(minimize_scalar(f, bounds=(np.log(.05), np.log(200.)), method="bounded", options=dict(xatol=1e-6)).x))


@register("ds_agree", "Answers: Table column of each sample of the pool's members, a gate member on its scope rows only. Row class: bench, category, kind of each row. Cluster of each FIT row, for the held-out folds. Gold: y on sp.fit only: disagreement rows for the tables and T, agreed rows for a. Counts: only FIT rows' answers. Rows outside sp.fit are read at answer time only: their votes, their class, and the agreement test on their votes. Member origin, stock and logp are not read.")
def _ds_agree(x):
    tab, sp = x.tab, x.sp
    iters, kp = x.params.get("iters", 20), x.params.get("kp", 10.)
    votes, scoped = fusion_votes(x.members, x.votes)
    V = sample_columns(tab, x.members, votes)
    agreecol, dis, tot = agreement(V, tab.C)
    mc = np.array([x.data.rows[i].get("kind") == "mc" for i in tab.ids])
    cells = fine_cells(x.data, tab.ids)
    fit, y, mask, other = np.asarray(sp.fit, int), tab.y, tab.mask, tab.other
    tie = tie_break(tab.ids, tab.C)
    fold = np.array([zlib.crc32(str(c).encode()) % 5 for c in x.pool.video_id])
    held, parts = [], []
    for f in range(5):
        te = fit[fold[fit] == f]
        if not len(te):
            continue
        ll_f, acc_f, _ = ds_fit(V, y, mask, mc, cells, agreecol, dis, other, fit[fold[fit] != f], iters, kp)
        parts.append((te, ll_f, acc_f))
        d_ = te[~passes(agreecol, other)[te]]
        held.append((ll_f[d_], y[d_], mask[d_]))
    T = fit_temperature(np.concatenate([h_[0] for h_ in held]), np.concatenate([h_[1] for h_ in held]), np.concatenate([h_[2] for h_ in held])) if held else 1.
    pos = {int(k): n for n, k in enumerate(fit)}
    F_oof = np.full((len(fit), tab.C), -np.inf)
    for te, ll_f, acc_f in parts:
        Ff = posterior(ll_f, tie, mask, agreecol, other, cells, acc_f, T)[0]
        if VOTED_LETTERS:
            Ff = voted_only(Ff, tot, mc, agreecol, other)[0]
        F_oof[[pos[int(k)] for k in te]] = Ff[te]
    ll, acc, fitted = ds_fit(V, y, mask, mc, cells, agreecol, dis, other, fit, iters, kp)
    F, floored = posterior(ll, tie, mask, agreecol, other, cells, acc, T)
    Fv, vrows = voted_only(F, tot, mc, agreecol, other)
    a0, a1 = answer_cols(F, mask, other), answer_cols(Fv, mask, other)
    ch = a0 != a1
    sel_ = np.asarray(sp.select, int)
    vl = dict(on=VOTED_LETTERS, eps=VOTED_EPS, rows_cut=int(vrows.sum()), changed=int(ch.sum()), changed_fit=int(ch[fit].sum()), changed_select=int(ch[sel_].sum()) if len(sel_) else 0,
              changed_fit_right=int((ch[fit] & (a1[fit] == y[fit])).sum()), changed_fit_wrong=int((ch[fit] & (a0[fit] == y[fit])).sum()), changed_ids=[tab.ids[k] for k in np.flatnonzero(ch)][:50])
    if VOTED_LETTERS:
        F = Fv
    share = lambda idx: float((agreecol[idx] >= 0).mean()) if len(idx) else float("nan")
    sel = np.asarray(sp.select, int)
    merged = right_spellings(tab, x.members, votes)
    fitset, selset = set(fit.tolist()), set(sel.tolist())
    info = dict(gate_scoped=scoped, rows_in_em=int(dis[fit].sum()), unlabelled_in_em=0, em_passes=fitted["iters"], T=T, agree_acc_overall=float((agreecol[fit[passes(agreecol, other)[fit]]] == y[fit[passes(agreecol, other)[fit]]]).mean()) if passes(agreecol, other)[fit].any() else None,
                cells=fitted["cells"], agree_share_fit=share(fit), agree_share_select=share(sel), passed=int(passes(agreecol, other).sum()), agree_floored=floored,
                agree_on_other=int(((agreecol >= 0) & (agreecol == other)).sum()), no_vote_rows=int((tot.sum(1) == 0).sum()), mc_rows_with_text_columns=int((mc & (other >= 0)).sum()),
                merged_right_spellings_fit=sum(1 for k in merged if k in fitset), merged_right_spellings_select=[tab.ids[k] for k in merged if k in selset], tables=fitted, agree_acc=acc, F_oof=F_oof, voted_letters=vl)
    return F, info
