"""v4 decision rule, AM15 (coordinator, 11-10 18:45 CST). Replaces the family-shrinkage decision (judge_eb.judge) for L0 accepts.

Why: in generations 1-2 the normal family prior gave decisions that (a) lifted z 1.6 candidates on a positive family mean, (b) shrank a
z 3.2 candidate to nothing when the family of 30 was mostly null, (c) blew up on a candidate with zero variance, and (d) depended on the
moment other members' reads arrived, so early accepts could contradict the final numbers. A decision rule has to be the same whenever
it is evaluated and must not depend on a prior shape we cannot yet estimate.

Rule (one parameter, alpha = false discovery rate among accepted edits):
 1. Per candidate: d = mean paired difference on its routed judge rows (candidate sample minus the parent's mean over its cached
    samples, or minus the equal-compute reference), se = its cluster-robust standard error, nu = clusters - 1.
    Degenerate guard: se is floored at 1 / n_rows (the evidence cannot be sharper than if one more row had flipped).
    p = one-sided P(T_nu >= d / se).
 2. Benjamini-Hochberg at level alpha inside each (author, generation): sort the author's p-values, accept the k smallest where k is the
    largest index with p_(k) <= k * alpha / K. This is the tail-area empirical-Bayes false discovery rate (Efron 2008): of the edits an
    author gets accepted, at most alpha are expected to be false. An author with more real effects earns a lower bar by itself.
 3. Generalisation (owner: accept only gains that hold out): a GLOBAL candidate (condition "always") must in addition keep a positive
    pooled difference when any single source benchmark is left out (leave-one-source-out, point estimate > 0 for every source).
 4. Interval decisions without bias: BH is monotone (lowering any p-value never removes an accept). With every pending read of a
    candidate counted wrong, its p-value is at its upper bound; counted right, at its lower bound. So:
      accepted for good  = BH accepts under ALL candidates' upper bounds (and LOBO holds under the pessimistic completion);
      rejected for good  = not accepted by BH under ALL candidates' lower bounds;
      otherwise wait. A final judgement on all reads can then never contradict an early decision.
 5. Estimates for ordering and reports (never for accepting): lower confidence bound lb = d - t_{nu, 1-alpha} * se, and
    contribution_lb = routed share * lb. The tip takes the accepted candidate with the largest contribution_lb.
"""
import math

import numpy as np
from scipy.stats import t as _t

ALPHA = 0.10


def pval(d, se, n_rows, nu):
    se = max(float(se), 1.0 / max(int(n_rows), 1))
    return float(_t.sf(float(d) / se, max(float(nu), 1.0))), se


def bh(pvals, alpha=ALPHA):
    """returns a boolean accept mask (Benjamini-Hochberg step-up)."""
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


def lobo_ok(rows_by_source):
    """rows_by_source: {source: (sum of paired differences, number of rows)} of a global candidate. True iff the pooled mean stays > 0
    with any one source left out (a candidate present in a single source fails)."""
    S = sum(v[0] for v in rows_by_source.values())
    N = sum(v[1] for v in rows_by_source.values())
    if len(rows_by_source) < 2:
        return False
    return all((S - v[0]) / max(N - v[1], 1) > 0 for v in rows_by_source.values())


def judge(cands, alpha=ALPHA):
    """cands: list of dict(name, d, se, n, nu, share, is_global[, lobo]) of ONE author in ONE generation (complete reads).
    Returns one dict per candidate: p, se_used, bh (BH accept), lobo, accept, lb, contribution_lb."""
    ps, ses = zip(*[pval(c["d"], c["se"], c["n"], c.get("nu", max(c["n"] - 1, 1))) for c in cands]) if cands else ((), ())
    acc = bh(ps, alpha)
    out = []
    for c, p, se, a in zip(cands, ps, ses, acc):
        nu = max(float(c.get("nu", max(c["n"] - 1, 1))), 1.0)
        lob = bool(c.get("lobo", True)) if c.get("is_global") else True
        lb = float(c["d"]) - float(_t.ppf(1 - alpha, nu)) * se
        out.append(dict(name=c["name"], d=float(c["d"]), se_used=se, p=p, bh=bool(a), lobo=lob, accept=bool(a and lob), lb=lb,
                        contribution_lb=float(c.get("share", 1.0)) * lb))
    return out


def interval(cands_pess, cands_opt, alpha=ALPHA):
    """cands_pess / cands_opt: the same author's candidates with (d, se, lobo) computed under the pessimistic / optimistic completion of
    every candidate's own pending reads. Returns {name: 'accept' | 'reject' | 'wait'}; decisions are final (monotonicity of BH)."""
    jp = {r["name"]: r for r in judge(cands_pess, alpha)}
    jo = {r["name"]: r for r in judge(cands_opt, alpha)}
    return {n: ("accept" if jp[n]["accept"] else "reject" if not jo[n]["accept"] else "wait") for n in jp}


if __name__ == "__main__":
    # replay of generations 1 and 2 of runs/main_1110 (d, se on the routed scale from the judge lines; LOBO from the per-source D_s)
    g1P = [("c00",80,-.0531,.0296),("c01",180,.0250,.0285),("c02",82,.0305,.0397),("c03",146,-.0120,.0324),("c04",80,-.0344,.0305),("c05",91,-.0110,.0310),("c06",45,.1056,.0602),("c08",6,-.0417,.1635),("c09",198,.0492,.0274),("c10",6,.1250,.0854),("c11",115,-.0478,.0287),("c12",57,.0570,.0357),("c19",444,.0028,.0174),("c21",15,.1667,.126),("c22",27,.0463,.0652),("c23",8,.1562,.1562),("c24",12,.1458,.1357),("c28",29,.0431,.0688),("c29",86,.1105,.0395)]
    g1B = [("c30 min1024",444,.0591,.0188),("c31 max8192",444,-.0130,.0162),("c32 min2048",444,.0614,.0187),("c33 temp.6",444,.0208,.0172),("c34 min4096",444,.0726,.0183),("c35 max4096",444,-.0175,.0162),("c36 temp.8",444,-.0107,.0162),("c37 temp1.2",444,-.0039,.0182),("c38 max2048",444,.0073,.0160)]
    g2P = [("c00",71,-.0176,.0386),("c01",95,.0579,.0352),("c02",444,.0023,.0164),("c03",86,-.0087,.0330),("c04",91,-.0522,.0379),("c05",89,-.1011,.0412),("c06",332,-.0203,.0174),("c07",86,-.0814,.0423),("c08",28,.1161,.0636),("c09",66,-.0455,.0476),("c10",10,.1000,.0667),("c11",4,-.3750,.2165),("c12",47,-.0053,.0093),("c13",59,-.0169,.0290),("c14",4,.1250,.3146),("c15",7,.0714,.2766),("c16",7,.2143,.2143),("c17",3,.1667,.4410),("c18",113,.0221,.0300),("c19",444,-.0068,.0147),("c20",14,-.0714,.1029),("c21",48,-.0208,.0581),("c22",47,-.1117,.0640),("c23",77,-.0357,.0428),("c24",14,-.0536,.1054),("c25",3,0.0,0.0),("c26",102,.1103,.0346),("c27",91,.0604,.0329),("c28",123,.0569,.0249),("c29",180,.0319,.0215)]
    g2B = [("temp.6",444,.0270,.0140),("max4096",444,-.0225,.0139),("min null",444,-.0743,.0197),("min2048",444,-.0113,.0152),("temp1.2",444,-.0351,.0156),("max8192",444,-.0120,.0162),("min1024",444,-.0135,.0162),("max2048",444,-.0426,.0145),("temp.8",444,-.0173,.0142)]
    for label, L, glob in [("g1 P0", g1P, False), ("g1 B", g1B, True), ("g2 P0", g2P, False), ("g2 B", g2B, True)]:
        cands = [dict(name=c, d=d, se=se, n=n, share=n / 444, is_global=glob or n == 444) for c, n, d, se in L]
        rows = judge(cands)
        acc = [r for r in rows if r["accept"]]
        print(f"\n{label}: K {len(rows)}, accepted {len(acc)}")
        for r in sorted(rows, key=lambda r: r["p"])[:6]:
            print(f"  {r['name']:12s} d {r['d']:+.3f} se {r['se_used']:.3f} p {r['p']:.4f}  BH {'yes' if r['bh'] else 'no '}  lb {r['lb']:+.3f}  contribution_lb {100 * r['contribution_lb']:+.2f} pp")
    # LOBO on the per-source differences of the three g1 B cells (74 rows per source)
    for name, Ds in {"c34 min4096": [.149,.142,.054,.081,-.017,.027], "c30 min1024": [.081,.034,-.014,.054,.118,.081], "c32 min2048": [.149,.047,-.068,.081,.051,.108], "wonder": [.20,0,0,0,0,0], "wonder-": [.20,-.02,-.02,-.02,-.02,-.02]}.items():
        print(f"LOBO {name}: {lobo_ok({i: (74 * x, 74) for i, x in enumerate(Ds)})}")
    # monotonicity check of the interval rule by simulation
    rng = np.random.default_rng(3); bad = 0; tot = 0
    for _ in range(4000):
        K = int(rng.integers(3, 30)); n = rng.integers(20, 200, K); frac = rng.uniform(0.3, 1.0, K)
        th = np.where(rng.random(K) < 0.15, rng.normal(0.08, 0.03, K), 0.0)
        fin, pes, opt = [], [], []
        for j in range(K):
            x = rng.normal(th[j], 0.35, n[j]).clip(-1, 1); m = int(frac[j] * n[j])
            def mk(v):
                return dict(name=str(j), d=float(v.mean()), se=float(v.std(ddof=1) / math.sqrt(len(v))), n=len(v), share=1.0, is_global=False)
            fin.append(mk(x)); pes.append(mk(np.concatenate([x[:m], np.full(n[j] - m, -1.0)]))); opt.append(mk(np.concatenate([x[:m], np.full(n[j] - m, 1.0)])))
        f = {r["name"]: r["accept"] for r in judge(fin)}
        for name, dec in interval(pes, opt).items():
            tot += 1
            bad += (dec == "accept" and not f[name]) or (dec == "reject" and f[name])
    print(f"\ninterval rule vs final judgement: {bad} contradictions in {tot} candidate decisions")
