"""selfedit v5: the declared-guess read state, derived IN DISTRIBUTION (coordinator 11-11: seed = the stem "guess" in the tail of the
reasoning; further tail patterns selected on SHOW + MONITOR reads by leave-one-benchmark-out within-row predictive value; markers2.py of
the OOD study is NOT used).

What a flag is for: the policy "read again when the read ends in a declared guess". Its value is label-free at run time but can be scored
offline on signal rows: replacing a flagged read i by a fresh read of the same row changes the expected score by
    c_i = (mean ok of the row's OTHER cached reads) - ok_i,
so the gain of a detector is G = sum over flagged reads of c_i / reads, and its cost is C = sum over flagged reads of the row's mean
tokens / sum of tokens (extra generated tokens as a multiple of one read). A flag that marks hard ROWS (all reads alike) has c = 0 and
only costs; a flag that marks the worse READS of a row gains. The net value at the tip's own token price is V = G - price * C, where
price = the slope of the tip's frontier from one read to its first vertex on SIGNAL rows (accuracy per x1 tokens): a pattern is added
only if its reads are a better place for tokens than the tip's vote.

Selection (greedy, forward, from the seed): candidates = word 1-3-grams of the tail window that occur in >= MIN_BENCH benchmarks (a
pattern tied to one task type cannot enter); add the pattern with the largest gain in V on the training benchmarks if V rises in at least
MIN_POS of them; stop otherwise or at MAX_ADD. Leave-one-benchmark-out: the selection runs on five benchmarks and the detector is scored on
the sixth; the window (chars of the tail) is chosen by the LOBO value, and added patterns are kept only when their held-out gain over
the seed is positive across benchmarks (exact sign-flip p <= KEEP_P). The final detector is the selection on all six signal benchmarks.

derive(reads, rows) -> detector dict(window, patterns, seed, price, lobo table); flag(detector, tail) -> bool.
reads: [{row, ok, tokens, tail}] of SHOW + MONITOR rows only (a judge row raises).
"""
import collections
import re

import numpy as np

SEED = "guess"
WINDOWS = (600, 1200, 2500)
MIN_BENCH = 5          # a pattern must occur in at least this many of the six benchmarks (with >= MIN_PER_BENCH reads each)
MIN_PER_BENCH = 3
MIN_READS = 15         # and in at least this many reads overall
MIN_POS = 4            # V must rise in at least this many of the training benchmarks (inside the greedy selection)
KEEP_P = 0.05          # additions are kept only if their LEAVE-ONE-BENCHMARK-OUT gain over the seed has a benchmark sign-flip p <= this
MAX_ADD = 8
_WORD = re.compile(r"[a-z']+")


def words(tail, window):
    return _WORD.findall((tail or "")[-window:].lower())


def grams(ws, nmax=3):
    out = set()
    for n in range(1, nmax + 1):
        for i in range(len(ws) - n + 1):
            out.add(" ".join(ws[i:i + n]))
    return out


def _has(pattern, text_words_joined):
    return (" " + pattern) in text_words_joined if pattern != SEED else (" " + SEED) in text_words_joined


def flag(det, tail):
    """True when the tail window holds the seed stem or one of the detector's patterns (word-boundary on the left for the seed stem:
    guess, guessing, guessed; whole word sequences for the patterns)."""
    t = " " + " ".join(words(tail, det["window"])) + " "
    if (" " + det["seed"]) in t:
        return True
    return any((" " + p + " ") in t for p in det["patterns"])


class _Data:
    def __init__(self, reads, rows, window):
        for x in reads:
            if rows[x["row"]]["role"] not in ("show", "monitor"):
                raise ValueError(f"guess5: row {x['row']} is not a SHOW / MONITOR row")
        self.benches = sorted({rows[x["row"]]["bench"] for x in reads})
        bi = {b: i for i, b in enumerate(self.benches)}
        byrow = collections.defaultdict(list)
        for i, x in enumerate(reads):
            byrow[x["row"]].append(i)
        n = len(reads)
        self.ok = np.array([float(x["ok"]) for x in reads])
        self.b = np.array([bi[rows[x["row"]]["bench"]] for x in reads])
        self.c = np.zeros(n)            # gain of replacing read i by a fresh read of its row
        self.t = np.zeros(n)            # tokens of that fresh read (row mean of the others)
        self.w = np.zeros(n)            # weight: every row counts 1 (reads of a row share it)
        tok = np.array([float(x["tokens"]) for x in reads])
        for r, ix in byrow.items():
            if len(ix) < 2:
                continue
            so, st = self.ok[ix].sum(), tok[ix].sum()
            for i in ix:
                self.c[i] = (so - self.ok[i]) / (len(ix) - 1) - self.ok[i]
                self.t[i] = (st - tok[i]) / (len(ix) - 1)
                self.w[i] = 1.0 / len(ix)
        self.rows_b = np.array([sum(1 for r in byrow if rows[r]["bench"] == b) for b in self.benches], float)
        self.base_tok_b = np.array([sum(tok[ix].mean() for r, ix in byrow.items() if rows[r]["bench"] == b) for b in self.benches]) / self.rows_b
        self.texts = [" " + " ".join(words(x["tail"], window)) + " " for x in reads]
        self.seed = np.array([(" " + SEED) in t for t in self.texts])
        # candidate patterns: word 1-3-grams by document frequency and benchmark spread
        df = collections.Counter()
        dfb = collections.defaultdict(collections.Counter)
        self.gr = []
        for i, x in enumerate(reads):
            g = grams(words(x["tail"], window))
            self.gr.append(g)
            df.update(g)
            for p in g:
                dfb[p][self.b[i]] += 1
        self.cands = sorted(p for p, k in df.items() if k >= MIN_READS and sum(v >= MIN_PER_BENCH for v in dfb[p].values()) >= MIN_BENCH
                            and SEED not in p)
        self.M = {p: np.array([p in g for g in self.gr]) for p in self.cands}

    def gain_cost(self, fl):
        """per benchmark: G_b (accuracy gain of re-reading flagged reads), C_b (extra tokens as a multiple of one read)."""
        B = len(self.benches)
        G = np.bincount(self.b, weights=fl * self.c * self.w, minlength=B) / self.rows_b
        C = np.bincount(self.b, weights=fl * self.t * self.w, minlength=B) / self.rows_b / self.base_tok_b
        return G, C

    def value(self, fl, price, benches=None):
        G, C = self.gain_cost(fl)
        v = G - price * C
        if benches is not None:
            return v[benches]
        return v


def select(D, price, train):
    """greedy forward selection on the benchmarks `train` (indices). -> patterns, flags."""
    fl = D.seed.copy()
    pats = []
    for _ in range(MAX_ADD):
        v0 = D.value(fl, price, train)
        best, bestgain, bestfl = None, 0.0, None
        for p in D.cands:
            if p in pats:
                continue
            f2 = fl | D.M[p]
            if f2.sum() == fl.sum():
                continue
            v = D.value(f2, price, train)
            if (v > v0 + 1e-12).sum() >= min(MIN_POS, len(train)) and v.mean() - v0.mean() > bestgain + 1e-12:
                best, bestgain, bestfl = p, v.mean() - v0.mean(), f2
        if best is None:
            break
        pats.append(best)
        fl = bestfl
    return pats, fl


def derive(reads, rows, price, windows=None):
    """price: accuracy per x1 tokens of the tip's frontier on signal rows (first segment). windows: tail windows to try (default
    WINDOWS; the loop keeps 1,500 characters of reasoning per read in memory, so it passes (600, 1500))."""
    out = dict(seed=SEED, price=price, windows={})
    best = None
    for W in (windows or WINDOWS):
        D = _Data(reads, rows, W)
        B = len(D.benches)
        lobo_seed, lobo_sel, picked = np.zeros(B), np.zeros(B), {}
        gs, cs, gl, cl = np.zeros(B), np.zeros(B), np.zeros(B), np.zeros(B)
        for h in range(B):
            train = [i for i in range(B) if i != h]
            pats, fl = select(D, price, train)
            picked[D.benches[h]] = pats
            lobo_seed[h] = D.value(D.seed, price)[h]
            lobo_sel[h] = D.value(fl, price)[h]
            G, C = D.gain_cost(D.seed)
            gs[h], cs[h] = G[h], C[h]
            G, C = D.gain_cost(fl)
            gl[h], cl[h] = G[h], C[h]
        pats, fl = select(D, price, list(range(B)))
        G, C = D.gain_cost(fl)
        rec = dict(window=W, patterns=pats, candidates=len(D.cands),
                   seed=dict(share=float(D.seed.mean()), acc_flagged=float(D.ok[D.seed].mean()) if D.seed.any() else None,
                             acc_unflagged=float(D.ok[~D.seed].mean()), G=float(gs.mean()), C=float(cs.mean()), V=float(lobo_seed.mean()),
                             V_b={b: float(v) for b, v in zip(D.benches, lobo_seed)}, G_b={b: float(v) for b, v in zip(D.benches, gs)}),
                   lobo=dict(G=float(gl.mean()), C=float(cl.mean()), V=float(lobo_sel.mean()), V_b={b: float(v) for b, v in zip(D.benches, lobo_sel)},
                             G_b={b: float(v) for b, v in zip(D.benches, gl)}, picked=picked, better_in=int((lobo_sel > lobo_seed + 1e-12).sum())),
                   final=dict(share=float(fl.mean()), acc_flagged=float(D.ok[fl].mean()), acc_unflagged=float(D.ok[~fl].mean()),
                              G=float(G.mean()), C=float(C.mean()), V=float((G - price * C).mean())))
        out["windows"][W] = rec
        # the window and whether additions are kept: by the LEAVE-ONE-BENCHMARK-OUT value (never by the in-sample one). Additions are kept
        # only when the held-out gain over the seed is positive at the benchmark level: exact sign-flip p over the benchmarks <= KEEP_P
        # (with six benchmarks: all six better, or five with the sixth the smallest). "Better in 4 of 6" passes pure noise one time in three
        # (the fake engine's random words did: t_gen5), this passes it at most one time in twenty.
        diff = lobo_sel - lobo_seed
        signs = np.array(list(__import__("itertools").product((-1.0, 1.0), repeat=B)))
        rec["lobo"]["p_signflip"] = float(((signs @ diff) >= diff.sum() - 1e-15).mean())
        for use_sel, v in ((False, float(lobo_seed.mean())), (True, float(lobo_sel.mean()))):
            if use_sel and (rec["lobo"]["p_signflip"] > KEEP_P or diff.sum() <= 0):
                continue
            if best is None or v > best[0] + 1e-12:
                best = (v, W, pats if use_sel else [], dict(picked) if use_sel else {})
    # picked: per benchmark, the patterns selected WITHOUT that benchmark (cross-fitting): flags of signal rows of benchmark b must be
    # computed with picked[b], so that the signal-row value of a guess-triggered policy is never in-sample for the detector
    out.update(window=best[1], patterns=best[2], lobo_value=best[0], picked=best[3])
    return out
