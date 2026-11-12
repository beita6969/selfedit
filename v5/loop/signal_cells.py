"""B4 (coordinator 11-11, Track B): the cell maker. MEASURED signals on signal rows -> RUN/signal/<tip>/cells.json for author G.

Input  (roles show + monitor only; a judge row or an unknown row anywhere in the input raises, as signal_export.trace_row does)
       tip    {row: {sample: read}}: the tip's cached reads (RUN/signal/<tip>/traces.json or a compact table of the same shape, or result
              lines named by the mapping, or the base_* columns of the per-read tables).
       cells  {cell id: spec}: any measured edit. spec = {"kind": "weight" | "setting" | "ops" | "reference" | "control", "name", "family",
              "edit": {W edit form} | "settings": {knob: value} | "ops": [...], "prior_family": int (cells of this library read before on
              sets holding JUDGE rows; REQUIRED for a weight cell, B4b), "library": str (what the count refers to; default the family), optional "menu", "propose": false (measurement only: reported, never proposed), "alpha", "band",
              "dose", "control": <cell id>, "continues": bool}. Nothing here assumes "direction x alpha x band": a cell is whatever its
              own edit file / settings / ops say.
       reads  {cell id: {row: {sample: read}}}; read = {ok, tokens[, pred][, skip][, cut]} (rec_of: error / cancelled / abort lines are
              no reads; cut = the read hit the token budget). Sources: the planner's reads.jsonl, engine result files + a mapping
              (ingest_results), per-read tables (ingest_tsv: batch, cell, row, sample, ok, tokens, finish, ..., base_ok, base_tokens).
       control (make's argument, e.g. W-null: the tip read on the cells' engine, same rows and sample indices): once it has reads, a
              weight cell's gain, tokens and classes are against it; its gain against the tip's cached reads stays in est["vs_tip"].
Output doc (cells.json; exactly author_g.g_read_cells's form, more fields beside): {"tip", "scale", "cells": [ITEM, ...] in rank order, ...}
       ITEM = {"kind": "weight", "name", "edit", "pred", "se", "lb", "pred_tokens_ratio", "family", "prior_family", "library", "p_pass", "K", ...}
            | {"kind": "setting", "name", "settings", ...} | {"kind": "routed", "base_kind", "route": {...}, ...} (skipped by author_g
              until the patch: a cell applied only on a class) ; no row ids and no per-row lists anywhere in the file or the table.
Numbers (unit = row; clusters = picture clusters; strata = benchmarks at the signal rows' shares, which are JUDGE's):
  gain     per row: the cell's mean right minus the reference's, under two pairings: "row" = the tip's mean over all its cached samples
           (the judge's convention) and "key" = the same sample indices only (seeds are common random numbers by (row, sample)). Both are
           reported with their cluster-robust se; decisions use the one with the smaller se (a tie keeps "row"); the judge-convention
           value is the one written to "pred" whenever the two differ by less than the larger of their se.
  tokens   multiple = 1 + added tokens per row / tip tokens per row on the same rows; gain per added token.
  pred     on the JUDGE's scale: a candidate whose tokens ratio rounds to m >= 2 is judged against the plurality of m tip samples
           (ecomp), so pred = gain - the vote gain of m tip reads, taken from the tip's own samples on the same rows.
  bounds   Benjamini-Hochberg at ALPHA over the K measured options, then every bound at the false-coverage-adjusted level
           ALPHA x |selected| / K (Benjamini-Yekutieli): "lb > 0" = "BH-selected"; K null cells give a positive bound with
           probability <= ALPHA in total.
  classes  the tip's router conditions (known before the read; from the prefixes or prefix hashes of the tip table and the tip file)
           and bins of the tip's own read length at quantiles of the data (known only after the base read: the base read is paid, the
           cell's read is added; the gate is the base read at the same sample index, the comparison follows the cell's pairing).
Frontier per partition into classes (multiple-choice knapsack, greedy = equal marginal value): from each class's current vertex the
           option with the largest gain per added token; taken while its paired bound against the previous vertex is > 0 and it fits
           the budget = the stated token multiple M (RUN/cards.json "g_token_multiple" with its source; no default). lambda = the ratio
           of the first increment that does not fit (the budget's shadow price; 0 when everything fits).
Judge    p_pass = probability of passing the judge's BH at K = cells sent + the sum over their distinct libraries of each library's
           prior_family (B4b; within a library its largest), for the planned JUDGE
           reads (AM17 quarter and LITE step-1 futility, LITE_M samples, LOBO for a global cell); a cell with p_pass < ALPHA is held.
Calibration (after the judge): every RUN/signal/*/realised.jsonl line pairs a prediction with the judge's pooled d. Per kind and
           pooled: slope b of realised on raw predicted through the origin, prior N(0, TAU0^2); new predictions are b x raw. The
           signal-quality number Q = 1 - sum w (d - pred)^2 / sum w d^2 on predictions as written BEFORE the judge goes to
           RUN/signal_calibration.json and RUN/signal_quality.txt, per generation and cumulative.
usage: python3 signal_cells.py make --run RUN --tip TIP --rows learn.jsonl (--plan DIR | --map MAP.json RESULT.jsonl ... | --tsv T.tsv ...
                                    [--batch-cells B.json] [--control W-null] [--proposable XS]) [--tip-file T.json] [--ref REF.json] [--dry]
       python3 signal_cells.py watch --dir DIR ... (make's inputs)           (rerun dry on every new per-read table, append the table)
       python3 signal_cells.py scan --rows learn.jsonl RESULT.jsonl ...      (batches, roles, samples, edit tags: for writing MAP.json)
       python3 signal_cells.py calibrate --run RUN
Python standard library + numpy."""
import argparse, collections, fnmatch, glob, hashlib, json, math, os, sys, time

import numpy as np

ALPHA = 0.10                         # the loop's alpha (gen.ALPHA, judge_bh.ALPHA): BH level and one-sided bounds
SIGNAL_ROLES = ("show", "monitor")   # signal_export.SIGNAL_ROLES
TAU0 = 1.0                           # prior sd of the calibration slope (realised = b x predicted; 1 = calibrated): declared choice
JUDGE_M, PARENT_M = 4, 8             # gen.LITE_M (samples per routed JUDGE row at the decision), gen.P_FULL (cached parent samples)
N_SIM = 4000                         # draws of the pass-probability simulation (numerical resolution only)
MAKER = "signal_cells v1 (11-11)"
SCALE = "accuracy per row, show+monitor rows"
INF = float("inf")


# ---------------------------------------------------------------------------------------------------------------- small statistics
def _betacf(a, b, x):
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 400):
        for aa in (m * (b - m) * x / ((qam + 2 * m) * (a + 2 * m)), -(a + m) * (qab + m) * x / ((a + 2 * m) * (qap + 2 * m))):
            d = 1.0 + aa * d
            d = 1.0 / (d if abs(d) > tiny else tiny)
            c = 1.0 + aa / c
            c = c if abs(c) > tiny else tiny
            h *= d * c
        if abs(d * c - 1.0) < 1e-15:
            break
    return h


def betainc(a, b, x):
    """Regularised incomplete beta I_x(a, b) (continued fraction)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lf = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lf) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lf) * _betacf(b, a, 1.0 - x) / b


def phi(x):
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def t_sf(t, nu):
    """P(T_nu >= t)."""
    if t == INF:
        return 0.0
    if t == -INF:
        return 1.0
    if nu > 1e6:
        return phi(-t)
    p = 0.5 * betainc(nu / 2.0, 0.5, nu / (nu + t * t))
    return p if t >= 0 else 1.0 - p


def t_ppf(q, nu):
    """Quantile of Student t (bisection on t_sf)."""
    if not 0.0 < q < 1.0:
        return INF if q >= 1.0 else -INF
    if q < 0.5:
        return -t_ppf(1.0 - q, nu)
    lo, hi = 0.0, 2.0
    while t_sf(hi, nu) > 1.0 - q and hi < 1e12:
        lo, hi = hi, hi * 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if t_sf(mid, nu) > 1.0 - q else (lo, mid)
        if hi - lo < 1e-12 * max(1.0, hi):
            break
    return 0.5 * (lo + hi)


def bh(ps, alpha=ALPHA):
    """Indices accepted by Benjamini-Hochberg (step-up) at alpha."""
    K = len(ps)
    order = sorted(range(K), key=lambda i: ps[i])
    k = 0
    for j, i in enumerate(order, 1):
        if ps[i] <= alpha * j / K:
            k = j
    return set(order[:k])


def spearman(x, y):
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        r, i = [0.0] * len(v), 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
                j += 1
            for k in range(i, j + 1):
                r[o[k]] = (i + j) / 2.0
            i = j + 1
        return np.array(r)
    if len(x) < 3:
        return None
    a, b = rank(list(x)), rank(list(y))
    a, b = a - a.mean(), b - b.mean()
    den = math.sqrt(float((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / den) if den > 0 else None


def _f(x, k=5):
    """A JSON number: rounded; None for a missing or infinite value."""
    if x is None or isinstance(x, bool):
        return x
    x = float(x)
    return None if (math.isnan(x) or math.isinf(x)) else round(x, k)


def _sha16(b):
    return hashlib.sha256(b).hexdigest()[:16]


def esha(edit):
    """engine_api section 7 / 8 edit_sha (gen.esha)."""
    return _sha16(json.dumps(edit, sort_keys=True, separators=(",", ":")).encode())


def cell_sha(spec):
    """Identity of a cell: its edit / settings / ops (what the engine applies), not its name."""
    return _sha16(json.dumps([spec.get("edit") or None, spec.get("settings") or {}, spec.get("ops") or [], spec.get("request") or None],
                             sort_keys=True).encode())


# ---------------------------------------------------------------------------------------------------------------- rows: the guard
class Rows:
    """Role map of the learn rows: the guard of every input, the strata (benchmark) and the clusters (picture) of the statistics."""

    def __init__(self, meta):
        self.meta = meta                                         # row id -> {role, bench, pic[, kind, precision, category]}
        self.sig = sorted(r for r, m in meta.items() if m.get("role") in SIGNAL_ROLES)
        n = collections.Counter(meta[r].get("bench") for r in self.sig)
        self.W = {b: c / len(self.sig) for b, c in n.items()} if self.sig else {}
        self.n_judge = dict(collections.Counter(m.get("bench") for m in meta.values() if m.get("role") == "judge"))   # counts only

    @classmethod
    def from_learn(cls, path):
        meta = {}
        with open(path) as fh:
            for l in fh:
                if l.strip():
                    r = json.loads(l)
                    meta[r["id"]] = dict(role=r.get("v4_role"), bench=r.get("bench"), pic=r.get("pic"), kind=r.get("kind"),
                                         precision=r.get("precision"), category=r.get("category"))
        return cls(meta)

    @classmethod
    def from_traces(cls, traces):
        return cls({r: dict(role=x.get("role"), bench=x.get("bench"), pic=x.get("pic"), kind=x.get("kind")) for r, x in traces.items()})

    def check(self, r, what="signal cells"):
        m = self.meta.get(r)
        if m is None:
            raise ValueError(f"{what}: row {r} is not in the role map; only rows known as {SIGNAL_ROLES} may be used")
        if m.get("role") not in SIGNAL_ROLES:
            raise ValueError(f"{what}: row {r} has role {m.get('role')!r}; only {SIGNAL_ROLES} may be used")
        return r

    def bench(self, r):
        return self.meta[r].get("bench")

    def cluster(self, r):
        return self.meta[r].get("pic") or r


def rec_of(x):
    """A read (ok, tokens, pred, skip) from a result line or a trace sample; None when it is not a sample (an error or cancelled line,
    a dropped read with finish "abort", no grade). op_skipped: the entry could not run on the row (skip: counts as the reference)."""
    if not isinstance(x, dict) or x.get("error") or x.get("cancelled") or x.get("finish") == "abort":
        return None
    if x.get("op_skipped"):
        return dict(ok=None, tokens=None, pred=None, skip=True)
    if x.get("ok") is None:
        return None
    q = dict(ok=float(bool(x["ok"])), tokens=float(x.get("tokens") or 0.0), pred=x.get("pred"), skip=False)
    if x.get("finish") == "length":
        q["cut"] = True                                          # the read hit the token budget (it did not stop by itself)
    return q


def tip_table(traces, R, what="tip table"):
    """{row: {sample: read}} from RUN/signal/<tip>/traces.json (signal_export's format); every row checked, its own role field too."""
    T = {}
    for r, x in traces.items():
        R.check(r, what)
        if x.get("role") is not None and x["role"] not in SIGNAL_ROLES:
            raise ValueError(f"{what}: row {r} has role {x['role']!r}; only {SIGNAL_ROLES} may be used")
        sm = {}
        for s, v in (x.get("samples") or {}).items():
            q = rec_of(v)
            if q and not q["skip"]:
                sm[int(s)] = q
        if sm:
            T[r] = sm
    return T


def add_read(table, R, row, sample, x, what="reads"):
    """One read into {row: {sample: read}}; the row is checked first, whatever the line holds."""
    R.check(row, what)
    q = rec_of(x)
    if q is not None:
        table.setdefault(row, {}).setdefault(int(sample or 0), q)
    return q is not None


def reads_table(lines, R, what="reads"):
    """{cell: {row: {sample: read}}} from lines {"cell", "row", "sample", "ok", "tokens"[, "pred", "op_skipped", "error"]}."""
    out = {}
    for x in lines:
        add_read(out.setdefault(x["cell"], {}), R, x.get("row"), x.get("sample"), x, what)
    return out


def ingest_results(files, mapping, R, what="result files"):
    """Engine result files (q/out/<batch>.jsonl lines: key, row, sample, ok, tokens, ..., edit / edit_sha / a tag) -> (cells, reads, tip).
    mapping = {"cells": [{spec..., "match": {"batch": glob on the file name without .jsonl, "edit_sha": s, "edit_file": glob on
    edit.file, "tag": value of the line's tag / edit_tag / cell / name field, "samples": [lo, hi]}}], "tip": {"match": {...}} (optional:
    the tip's reads when they come as result lines too)}. A line goes to the first cell all of whose match keys hold; mode route / chat
    lines and unmatched lines are counted, not used. Every matched line's row is checked against the role map (a judge row raises)."""
    specs = [dict(c) for c in mapping.get("cells") or []]
    for c in specs:
        c.setdefault("id", c.get("name") or cell_sha(c))
    tipm = (mapping.get("tip") or {}).get("match")
    cells = {c["id"]: {k: v for k, v in c.items() if k != "match"} for c in specs}
    reads, tip, info = {c["id"]: {} for c in specs}, {}, collections.Counter()

    def holds(m, batch, x):
        if not m:
            return False
        ed = x.get("edit") if isinstance(x.get("edit"), dict) else {}
        for k, v in m.items():
            if k == "batch":
                good = fnmatch.fnmatch(batch, v)
            elif k == "edit_sha":
                good = (x.get("edit_sha") or (esha(x["edit"]) if x.get("edit") else None)) == v
            elif k == "edit_file":
                good = fnmatch.fnmatch(str(ed.get("file") or ""), v)
            elif k == "tag":
                good = v in [x.get(f) for f in ("tag", "edit_tag", "cell", "name")]
            elif k == "samples":
                good = isinstance(x.get("sample"), int) and v[0] <= x["sample"] <= v[1]
            elif k == "no_edit":
                good = (not x.get("edit")) == bool(v)
            else:
                raise ValueError(f"{what}: match key {k!r} not known (batch, edit_sha, edit_file, tag, samples, no_edit)")
            if not good:
                return False
        return True
    for f in files:
        batch = os.path.basename(f)
        batch = batch[:-6] if batch.endswith(".jsonl") else batch
        with open(f) as fh:
            for l in fh:
                try:
                    x = json.loads(l)
                except ValueError:
                    info["unreadable"] += 1
                    continue
                if x.get("mode") not in (None, "read") or x.get("row") is None:
                    info["not a read"] += 1
                    continue
                c = next((c for c in specs if holds(c.get("match"), batch, x)), None)
                if c is None and not holds(tipm, batch, x):
                    info["unmatched"] += 1
                    continue
                got = add_read(tip if c is None else reads[c["id"]], R, x["row"], x.get("sample"), x, f"{what} ({batch})")
                info[("tip" if c is None else c["id"]) if got else "no sample (error / abort)"] += 1
    return cells, reads, tip, dict(info)


def cell_from_name(name, info=None, rules=None):
    """A cell spec for a per-read table's cell name (batch_cells.json may add edit / family / note; rules: {"control_cell": name,
    "proposable_families": [...], "prior_family": n}). The family is the name's first word; a cell of a family that is not proposable
    is measured and reported but never proposed ("propose": false); the control cell is kind "control". alpha and band are labels
    read from the name when it has them (..._all_+1, ..._noL63_+1, ..._L24-39_+1, ..._mid_+1, ...); nothing depends on them."""
    info, rules = dict(info or {}), rules or {}
    fam = info.get("family") or name.split("_")[0]
    pf = info["prior_family"] if info.get("prior_family") is not None else rules.get("prior_family")        # B4b: no default (make
    lib = info.get("library") or (rules.get("library") if info.get("prior_family") is None else None) or fam  # refuses a weight cell
    spec = dict(name=name, id=name, family=fam, menu=fam, prior_family=pf, library=lib)                      # without it)
    if name == rules.get("control_cell") or info.get("kind") == "control":
        spec.update(kind="control", prior_family=0)
        return spec
    spec["kind"] = info.get("kind") or "weight"
    if info.get("edit"):
        spec["edit"] = info["edit"] if isinstance(info["edit"], dict) else dict(op="surgeon", file=info["edit"])
    for k in ("settings", "ops", "note", "control", "continues"):
        if info.get(k) is not None:
            spec[k] = info[k]
    if rules.get("proposable_families") is not None and fam not in rules["proposable_families"]:
        spec["propose"] = False
    if info.get("propose") is not None:
        spec["propose"] = bool(info["propose"])
    parts = name.split("_")
    try:
        spec["alpha"] = float(parts[-1])
    except ValueError:
        pass
    bands = {"all": [0, 63], "early": [0, 20], "mid": [21, 42], "late": [43, 63], "noL63": [0, 62], "L63only": [63, 63]}
    for tok in parts[1:-1]:
        if tok in bands:
            spec["band"] = bands[tok]
        elif tok.startswith("L") and "-" in tok:
            a_, b_ = tok[1:].split("-", 1)
            if a_.isdigit() and b_.replace("noL63", "").isdigit():
                spec["band"] = [int(a_), int(b_.replace("noL63", "")) - (1 if "noL63" in b_ else 0)]
    return spec


def ingest_tsv(files, R, batch_cells=None, rules=None, what="per-read tables"):
    """Per-read tables (tab-separated, one header line: batch, cell, row, sample, ok, tokens, finish, loop, prompt_tokens, error,
    base_key, base_ok, base_tokens, base_finish; the engine agent writes one per finished batch) -> (cells, reads, base, info).
    A line with an error text or finish "abort" is no read. base = the tip's cached read of the same (row, sample) as the table
    gives it ({row: {sample: read}}): used to check the tip table, or as the tip table when there is none. batch_cells = the "cells"
    object of batch_cells.json ({batch: {"cell", "edit", "family", ...}}). Every row is checked against the role map first."""
    bc = (batch_cells or {}).get("cells") or {}
    rules = dict(rules or {})
    rules.setdefault("prior_family", (batch_cells or {}).get("prior_family"))
    rules.setdefault("library", (batch_cells or {}).get("library") or "batch_cells")   # B4b: one file-level count = ONE library
    byname = {}
    for b, x in bc.items():                                      # a later shard may name only the cell: its first full entry counts
        for k, v in x.items():
            byname.setdefault(x.get("cell"), {}).setdefault(k, v)
    cells, reads, base, info = {}, {}, {}, collections.Counter()
    for f in files:
        with open(f) as fh:
            head = fh.readline().rstrip("\n").split("\t")
            for l in fh:
                v = dict(zip(head, l.rstrip("\n").split("\t")))
                if not v.get("row") or not v.get("cell"):
                    info["unreadable"] += 1
                    continue
                R.check(v["row"], f"{what} ({os.path.basename(f)})")
                name = v["cell"]
                if name not in cells:
                    cells[name] = cell_from_name(name, byname.get(name), rules)
                    reads[name] = {}
                num = lambda k: float(v[k]) if v.get(k) not in (None, "") else None   # noqa: E731
                x = dict(ok=None if v.get("ok") in (None, "") else bool(int(float(v["ok"]))), tokens=num("tokens"), finish=v.get("finish") or None,
                         error=v.get("error") or None)
                got = add_read(reads[name], R, v["row"], int(v.get("sample") or 0), x, what)
                info[name if got else "no sample (error / abort)"] += 1
                if v.get("base_ok") not in (None, ""):
                    add_read(base, R, v["row"], int(v.get("sample") or 0), dict(ok=bool(int(float(v["base_ok"]))), tokens=num("base_tokens"), finish=v.get("base_finish") or None), what)
    return cells, reads, base, dict(info)


def base_check(base, tip):
    """How many base reads of the per-read tables (the tip's cached read at the same row and sample) differ from the tip table."""
    n = bad = 0
    for r, sm in base.items():
        for s_, q in sm.items():
            n += 1
            t = (tip.get(r) or {}).get(s_)
            bad += t is None or t["ok"] != q["ok"] or abs(t["tokens"] - q["tokens"]) > 0.5
    return f"{bad} of {n}"


def scan_results(files, R):
    """What engine result files hold, for writing the mapping: per file its lines by mode and by role of the row (roles are counted,
    nothing else is read from a judge line), sample indices, and the distinct edit tags (edit_sha, edit.file, tag / edit_tag / cell)."""
    out = []
    for f in files:
        c, tags, smp = collections.Counter(), collections.Counter(), set()
        with open(f) as fh:
            for l in fh:
                try:
                    x = json.loads(l)
                except ValueError:
                    c["unreadable"] += 1
                    continue
                if x.get("mode") not in (None, "read") or x.get("row") is None:
                    c[f"mode {x.get('mode')}"] += 1
                    continue
                role = (R.meta.get(x["row"]) or {}).get("role") or "unknown row"
                c[f"reads on {role} rows"] += 1
                if role not in SIGNAL_ROLES:
                    continue
                c["errors / aborts"] += rec_of(x) is None
                ed = x.get("edit") if isinstance(x.get("edit"), dict) else {}
                tags[json.dumps(dict(edit_sha=x.get("edit_sha") or (esha(x["edit"]) if x.get("edit") else None), edit_file=ed.get("file"),
                                     **{k: x[k] for k in ("tag", "edit_tag", "cell", "name") if x.get(k) is not None}), sort_keys=True)] += 1
                if isinstance(x.get("sample"), int):
                    smp.add(x["sample"])
        out.append(dict(file=f, batch=os.path.basename(f)[:-6] if f.endswith(".jsonl") else os.path.basename(f), lines=dict(c), samples=sorted(smp),
                        tags=[dict(json.loads(k), reads=v) for k, v in tags.most_common()]))
    return out


# ---------------------------------------------------------------------------------------------------------------- row-paired statistics
def strat_stat(vals, R, floor=False):
    """Mean of a per-row variable {row: value} with its cluster-robust se. Post-stratified by benchmark at the signal rows' shares when
    every benchmark has >= 2 rows here (else the plain mean); clusters = picture clusters; small-sample factor G / (G - strata);
    nu = G - strata. floor: se >= 1 / rows (judge_bh's guard for accuracy differences)."""
    rows = sorted(vals)
    n = len(rows)
    if n == 0:
        return dict(d=0.0, se=INF, nu=1.0, n=0, clusters=0, strat=False)
    x = np.array([vals[r] for r in rows], float)
    by = collections.defaultdict(list)
    for i, r in enumerate(rows):
        by[R.bench(r)].append(i)
    strat = set(by) == set(R.W) and all(len(v) >= 2 for v in by.values())
    e = np.zeros(n)
    if strat:
        d = 0.0
        for h, ix in by.items():
            m = float(x[ix].mean())
            d += R.W[h] * m
            e[ix] = R.W[h] * (x[ix] - m) / len(ix)
    else:
        d = float(x.mean())
        e = (x - d) / n
    H = len(by) if strat else 1
    cs = collections.defaultdict(float)
    for i, r in enumerate(rows):
        cs[R.cluster(r)] += float(e[i])
    G = len(cs)
    se, nu = (math.sqrt(G / (G - H) * sum(v * v for v in cs.values())), float(G - H)) if G - H >= 1 else (INF, 1.0)
    if floor:
        se = max(se, 1.0 / n)
    return dict(d=float(d), se=se, nu=nu, n=n, clusters=G, strat=strat)


def bound(st, level, upper=False):
    """One-sided bound of a strat_stat at the given level (lower unless upper)."""
    if st["se"] == INF:
        return INF if upper else -INF
    k = t_ppf(1.0 - level, st["nu"]) * st["se"]
    return st["d"] + k if upper else st["d"] - k


def pval(st):
    return 1.0 if st["se"] == INF else t_sf(st["d"] / st["se"], st["nu"]) if st["se"] > 0 else (0.0 if st["d"] > 0 else 1.0)


def views(x, ref):
    """Row-level view of a cell's reads x against reference reads ref (both {row: {sample: read}}) under the two pairings: "row" = the
    reference's mean over all its samples of the row, "key" = the same sample indices only. {pairing: {row: dict(a, t, a0, t0, m)}};
    a skipped sample of the cell counts as the reference."""
    out = {"row": {}, "key": {}}
    for r, sm in x.items():
        rf = ref.get(r)
        if not rf or not sm:
            continue
        a0 = sum(q["ok"] for q in rf.values()) / len(rf)
        t0 = sum(q["tokens"] for q in rf.values()) / len(rf)
        out["row"][r] = dict(a=sum(a0 if q["skip"] else q["ok"] for q in sm.values()) / len(sm),
                             t=sum(t0 if q["skip"] else q["tokens"] for q in sm.values()) / len(sm), a0=a0, t0=t0, m=len(sm), s=len(rf))
        ks = [s for s in sm if s in rf]
        if ks:
            out["key"][r] = dict(a=sum(rf[s]["ok"] if sm[s]["skip"] else sm[s]["ok"] for s in ks) / len(ks),
                                 t=sum(rf[s]["tokens"] if sm[s]["skip"] else sm[s]["tokens"] for s in ks) / len(ks),
                                 a0=sum(rf[s]["ok"] for s in ks) / len(ks), t0=sum(rf[s]["tokens"] for s in ks) / len(ks), m=len(ks), s=len(ks))
    return out


def vec_all(v, rows=None):
    """(D, A, T, w) of a view on all its rows, or (rows given: a class known before the read) only where the row is in the class."""
    D, A, T, w = {}, {}, {}, {}
    for r, q in v.items():
        m = 1.0 if rows is None or r in rows else 0.0
        D[r], A[r], T[r], w[r] = m * (q["a"] - q["a0"]), m * (q["t"] - q["t0"]), m * q["t0"], m
    return D, A, T, w


def vec_len(x, ref, lo, hi, continues=False, pairing="key"):
    """(D, A, T, w) of a cell on the class "the base read's length is in (lo, hi]" (per read). The gate is the reference's read at the
    same sample index when the two share one (the base read the row's seed gives), else every reference sample is a possible base
    read. The cell's read is compared with the reference as in its pairing: "key" = that base read itself, "row" = the reference's
    row mean (so the classes add up to the cell's gain on all rows under either pairing). The base read is always paid: the cell's
    whole read is added (continues: only its tokens beyond the base)."""
    D, A, T, w = {}, {}, {}, {}
    for r, sm in x.items():
        rf = ref.get(r)
        if not rf or not sm:
            continue
        a0 = sum(q["ok"] for q in rf.values()) / len(rf)
        ks = [s for s in sm if s in rf]
        if ks:
            g = [1.0 if lo < rf[s]["tokens"] <= hi and not sm[s]["skip"] else 0.0 for s in ks]
            D[r] = sum(gi * ((sm[s]["ok"] or 0.0) - (rf[s]["ok"] if pairing == "key" else a0)) for gi, s in zip(g, ks)) / len(ks)
            A[r] = sum(gi * ((sm[s]["tokens"] or 0.0) - (rf[s]["tokens"] if continues else 0.0)) for gi, s in zip(g, ks)) / len(ks)
            T[r] = sum(gi * rf[s]["tokens"] for gi, s in zip(g, ks)) / len(ks)
            w[r] = sum(g) / len(ks)
        else:
            xs = [q for q in sm.values() if not q["skip"]]
            if not xs:
                continue
            a, t = sum(q["ok"] for q in xs) / len(xs), sum(q["tokens"] for q in xs) / len(xs)
            g = {s: (1.0 if lo < q["tokens"] <= hi else 0.0) for s, q in rf.items()}
            D[r] = sum(g[s] * (a - (q["ok"] if pairing == "key" else a0)) for s, q in rf.items()) / len(rf)
            A[r] = sum(g.values()) / len(rf) * t
            T[r] = sum(g[s] * q["tokens"] for s, q in rf.items()) / len(rf)
            w[r] = sum(g.values()) / len(rf)
    return D, A, T, w


def cut_stats(x, ref):
    """Reads that hit the token budget (finish "length"): their share for the cell and for the reference on the same rows, and the
    token multiple among the reads that stopped by themselves (the plain multiple counts every read, as the judge does)."""
    xs = [q for r, sm in x.items() if r in ref for q in sm.values() if not q["skip"]]
    rs = [q for r, sm in ref.items() if r in x for q in sm.values()]
    if not xs or not rs:
        return None
    st = lambda qs: [q["tokens"] for q in qs if not q.get("cut")]   # noqa: E731
    return dict(cell=_f(sum(1 for q in xs if q.get("cut")) / len(xs), 4), ref=_f(sum(1 for q in rs if q.get("cut")) / len(rs), 4),
                tokens_multiple_stopped=_f((sum(st(xs)) / len(st(xs))) / (sum(st(rs)) / len(st(rs))), 3) if st(xs) and st(rs) else None)


def reread_reads(tip):
    """The free control of a length class: "re-read with the tip itself" = for sample s the tip's next cached sample (another seed)."""
    out = {}
    for r, sm in tip.items():
        ss = sorted(sm)
        if len(ss) >= 2:
            out[r] = {s: dict(sm[ss[(i + 1) % len(ss)]], skip=False) for i, s in enumerate(ss)}
    return out


# ---------------------------------------------------------------------------------------------------------------- judge scale (ecomp)
def ref_m(c, rule="round"):
    """Tip samples in the judge's plurality reference for a tokens ratio c: ecomp.ref_m_round (generations since AM16), ecomp.ref_m
    ("ceil"), or None ("off" / below the trigger: the tip's mean is the reference)."""
    if c is None or rule == "off":
        return None
    if rule == "ceil":
        return max(2, int(math.ceil(c - 1e-9))) if c >= 1.5 - 1e-9 else None
    m = int(math.floor(c + 0.5 + 1e-9))
    return m if m >= 2 else None


_EC = []


def _aclass(pred, meta):
    """The class a pred votes for: ecomp.answer_class when the module and the row's kind are at hand, else the pred as it is."""
    if pred is None:
        return "none"
    if not _EC:
        try:
            import ecomp
            _EC.append(ecomp)
        except Exception:                                        # noqa: BLE001 (no grader at hand: plain classes)
            _EC.append(None)
    if _EC[0] is not None and (meta or {}).get("kind"):
        try:
            return _EC[0].answer_class(pred, meta)
        except Exception:                                        # noqa: BLE001
            pass
    return ("v", str(pred))


def plurality_ok(recs, meta):
    """Right or wrong of the plurality answer of the reads in order (ecomp.plurality: ties to the earliest class, "none" is wrong)."""
    count, first, ok = {}, {}, {}
    for j, q in enumerate(recs):
        c = _aclass(q.get("pred"), meta)
        count[c] = count.get(c, 0) + 1
        if c not in first:
            first[c], ok[c] = j, (0.0 if c == "none" else q["ok"])
    return ok[min(count, key=lambda c: (-count[c], first[c]))]


def vote_rows(tip, R, m, by_ok=False):
    """{row: vote gain of m tip reads} = right of the plurality of m consecutive cached samples (every cyclic window) minus the row's
    mean right; rows with fewer than m samples are left out (the judge waits on them too). by_ok: the table has no pred, so the
    classes are "right" / "wrong" (all wrong answers as one class: a lower bound of the vote gain)."""
    out = {}
    for r, sm in tip.items():
        ss = sorted(sm)
        if len(ss) < m:
            continue
        meta = {} if by_ok else (R.meta.get(r) or {})
        qs = {s: (dict(q, pred="right" if q["ok"] else "wrong") if by_ok else q) for s, q in sm.items()}
        v = sum(plurality_ok([qs[ss[(i + j) % len(ss)]] for j in range(m)], meta) for i in range(len(ss))) / len(ss)
        out[r] = v - sum(q["ok"] for q in sm.values()) / len(sm)
    return out


# ---------------------------------------------------------------------------------------------------------------- classes
def length_classes(tip, n=2):
    """Bins of the tip's own read length (per read) at the i / n quantiles of its token counts on the signal rows (data, not constants)."""
    toks = np.array([q["tokens"] for sm in tip.values() for q in sm.values()], float)
    if n < 2 or len(toks) < n:
        return []
    cut = [float(np.quantile(toks, i / n)) for i in range(1, n)]
    b = [-INF] + cut + [INF]
    out = []
    for i in range(n):
        lo, hi = b[i], b[i + 1]
        if not lo < hi:
            continue
        name = f"len<={hi:.0f}" if lo == -INF else (f"len>{lo:.0f}" if hi == INF else f"{lo:.0f}<len<={hi:.0f}")
        out.append(dict(name=name, type="length", lo=lo, hi=hi, quantiles=[i / n, (i + 1) / n]))
    return out


def router_classes(traces, entries):
    """One class per conditional entry of the tip that carries a procedure: the rows on which the router's decision put that procedure
    into the tip's prefix. traces = RUN/signal/<tip>/traces.json (rows carry "prefix") or a compact table whose rows carry
    "prefix_sha" = sha256(prefix text)[:n] (then the set of procedures behind each hash is found by trying the subsets of the tip's
    procedures joined by a blank line in entry order; a hash no subset gives leaves its rows out of every class and is counted in
    the class's "undecoded"). entries = the tip file's entries."""
    es, seen = [], set()
    for e in [x for e in entries or [] for x in (e.get("parts") or [e])]:
        cond, proc = e.get("condition"), e.get("procedure")
        if cond and cond != "always" and proc and cond not in seen:
            seen.add(cond)
            es.append(e)
    fired, und = {}, 0                                           # row -> indices of the entries whose procedure is in its prefix
    if any("prefix" in x for x in traces.values()):
        fired = {r: {i for i, e in enumerate(es) if e["procedure"] in (x.get("prefix") or "")} for r, x in traces.items()}
    elif es and len(es) <= 18:
        want = {x.get("prefix_sha") for x in traces.values()} - {None}
        n = len(next(iter(want))) if want else 0
        dec = {}
        for mask in range(1 << len(es)):
            if len(dec) == len(want):
                break
            sub = [i for i in range(len(es)) if mask >> i & 1]
            h = hashlib.sha256("\n\n".join(es[i]["procedure"] for i in sub).encode()).hexdigest()[:n]
            if h in want:
                dec.setdefault(h, set(sub))
        fired = {r: dec[x["prefix_sha"]] for r, x in traces.items() if x.get("prefix_sha") in dec}
        und = sum(1 for x in traces.values() if x.get("prefix_sha") not in dec)
    return [dict(name="cond:" + _sha16(e["condition"].encode())[:8], type="cond", condition=e["condition"], rows={r for r, f in fired.items() if i in f},
                 from_cand=e.get("from_cand"), undecoded=und) for i, e in enumerate(es)] if fired else []


def routes_classes(routes):
    """Classes from a routes file {condition text: [rows] | {row: bool}}."""
    out = []
    for cond, v in (routes or {}).items():
        rows = {r for r, y in v.items() if y} if isinstance(v, dict) else set(v)
        out.append(dict(name="cond:" + _sha16(cond.encode())[:8], type="cond", condition=cond, rows=rows))
    return out


# ---------------------------------------------------------------------------------------------------------------- options and frontier
def option(cid, cls, D, A, T, w, R, votes, rule, t0pop, share=1.0):
    """One option = a cell applied on a class: its row vectors and numbers. P = D on the judge's scale; contributions are on the all-rows
    scale (zero outside the class), "routed" numbers on the class's own rows (divided by its share)."""
    sD, sA, sT = strat_stat(D, R, floor=True), strat_stat(A, R), strat_stat(T, R)
    c = 1.0 + sA["d"] / sT["d"] if sT["d"] > 0 else None                      # tokens ratio on the class rows (what the judge measures)
    m = ref_m(c, rule)
    P = D
    if m is not None and m in votes:
        P = {r: D[r] - w[r] * votes[m][r] for r in D if r in votes[m]}
    sP = strat_stat(P, R, floor=True) if P is not D else sD
    return dict(id=cid, cls=cls, D=D, A=A, P=P, w=w, sD=sD, sA=sA, sP=sP, c=c, m=m, share=share,
                mult=1.0 + sA["d"] / t0pop if t0pop > 0 else None, Aall=sA["d"], Pall=sP["d"])


def inc_stat(o_from, o_to, R):
    """Paired increment between two options of one class on their common rows (from None = the tip): (strat_stat of the judge-scale
    gain increment, added tokens per row of the increment)."""
    if o_from is None:
        return o_to["sP"], o_to["Aall"]
    P1, P0, A1, A0 = o_to["P"], o_from["P"], o_to["A"], o_from["A"]
    return (strat_stat({r: P1[r] - P0[r] for r in P1 if r in P0}, R, floor=True),
            strat_stat({r: A1[r] - A0[r] for r in A1 if r in A0}, R)["d"])


def menu_of(spec, cid):
    """The menu of a cell: cells of one menu are alternatives of each other (doses / bands of one direction, values of one knob);
    different menus are different coordinates and add up. spec["menu"], else its family, else the cell alone."""
    return str(spec.get("menu") or spec.get("family") or cid)


def frontier(menu, budget, level, R, inc=None):
    """Greedy multiple-choice knapsack over {(class, menu): [option]} (any hashable key). Returns dict(steps, choice, lam, used). From each class's current vertex
    (the tip at first) the candidate increments are PAIRED: inc(vertex | None, option) -> (stat of the gain increment, added tokens)
    on the rows (and, in make, the sample indices) the two share. The increment with the largest gain per added token over all classes
    is taken while its bound at `level` is > 0 and it fits the budget. lam = the ratio of the first increment that did not fit (None
    when everything fitted); after that cut only the cut class may still take vertices it can afford (flagged below_lambda). An
    increment whose bound is <= 0 is dropped and the class tries its next option from the same vertex (a larger step merges the two)."""
    inc = inc or (lambda a, b: inc_stat(a, b, R))
    cur = {c: None for c in menu}
    blocked = {c: set() for c in menu}
    used, lam, cut, steps, memo = 0.0, None, None, [], {}
    while True:
        best = None
        for c, opts in menu.items():
            if lam is not None and c != cut:
                continue
            for o in opts:
                if o["id"] in blocked[c] or o is cur[c]:
                    continue
                k_ = (c, cur[c]["id"] if cur[c] else None, o["id"])
                if k_ not in memo:
                    memo[k_] = inc(cur[c], o)
                st, dA = memo[k_]
                dP = st["d"]
                if dP <= 0 or st["se"] == INF:
                    continue
                k = (INF if dA <= 0 else dP / dA, dP)
                if best is None or k > best[0]:
                    best = (k, c, o, dA, dP, st)
        if best is None:
            break
        (rho, _), c, o, dA, dP, st = best
        if dA > budget - used + 1e-9:
            if lam is None:
                lam, cut = rho, c
            blocked[c].add(o["id"])
            o.setdefault("why_not", f"does not fit the budget (needs {dA:+.0f} tokens per row, {budget - used:.0f} left)")
            continue
        lb = bound(st, level)
        if not lb > 0:
            blocked[c].add(o["id"])
            o.setdefault("why_not", f"increment over {cur[c]['id'] if cur[c] else 'the tip'}: bound {lb:+.4f} <= 0")
            continue
        used += dA
        steps.append(dict(order=len(steps), key=c, cell=o["id"], frm=cur[c]["id"] if cur[c] else None, added_tokens=dA, gain=dP,
                          ratio=None if rho == INF else rho, inc_d=st["d"], inc_se=st["se"], inc_lb=lb,
                          below_lambda=bool(lam is not None and rho < lam)))
        o.pop("why_not", None)
        cur[c] = o
    return dict(steps=steps, choice={c: (o["id"] if o else None) for c, o in cur.items()}, lam=lam, used=used)


# ---------------------------------------------------------------------------------------------------------------- the judge's test
def variance_parts(v, D, R):
    """From a cell's row view: the within-benchmark variance of the row differences D split into the cell's sampling variance per
    sample (wc), the reference's per sample (wt) and the rest (tau2: real row-to-row spread of the effect). With one cell sample per
    row wc is not identified and the reference's is used."""
    by = collections.defaultdict(list)
    for r in D:
        by[R.bench(r)].append(D[r])
    ws = sum(R.W.get(h, 0.0) for h, x in by.items() if len(x) >= 2)
    tot = sum(R.W.get(h, 0.0) * float(np.var(x, ddof=1)) for h, x in by.items() if len(x) >= 2) / ws if ws > 0 else 0.0
    unb = lambda a, k: a * (1.0 - a) * (k / (k - 1.0) if k >= 2 else 1.0)   # noqa: E731
    wt = float(np.mean([unb(q["a0"], q["s"]) for q in v.values()])) if v else 0.0
    two = [q for q in v.values() if q["m"] >= 2]
    wc = float(np.mean([unb(q["a"], q["m"]) for q in two])) if two else wt
    wc, wt = max(wc, 1e-6), max(wt, 1e-6)
    im = float(np.mean([1.0 / q["m"] for q in v.values()])) if v else 1.0
    isr = float(np.mean([1.0 / q["s"] for q in v.values()])) if v else 1.0
    return dict(tau2=max(tot - wc * im - wt * isr, 0.0), wc=wc, wt=wt, total=tot)


def pass_prob(pred, se_pred, parts, n_bench, dev, K, is_w, is_global, alpha=ALPHA, m=JUDGE_M, pm=PARENT_M, n_sim=N_SIM, seed=0):
    """Probability that a cell with predicted gain N(pred, se_pred^2) passes the judge: (weight cell: the AM17 quarter at one sample
    must read d > 0;) LITE step 1 (one sample on every routed row) must read d > 0; the pooled m samples must give a one-sided
    p <= alpha / K (BH's bar for a cell that has to pass alone among K); a global cell must also hold LOBO. n_bench = routed JUDGE
    rows per benchmark, dev = the cell's (shrunk) per-benchmark deviation from its mean gain. Normal approximation of the row means."""
    hs = [h for h, n in sorted(n_bench.items()) if n >= 1]
    if not hs:
        return 0.0
    rng = np.random.default_rng(seed)
    n = np.array([float(n_bench[h]) for h in hs])
    q = np.ceil(n / 4.0) if is_w else n.copy()
    N = float(n.sum())
    vb, vc = parts["tau2"] + parts["wt"] / pm, parts["wc"]
    delta = pred + rng.standard_normal((n_sim, 1)) * se_pred + np.array([dev.get(h, 0.0) for h in hs])[None, :]
    z = lambda: rng.standard_normal((n_sim, len(hs)))            # noqa: E731
    bq, eq = z() * np.sqrt(vb / q), z() * np.sqrt(vc / q)
    rest = np.maximum(n - q, 1.0)
    br, er = z() * np.sqrt(vb / rest) * (n > q), z() * np.sqrt(vc / rest) * (n > q)
    e234 = z() * np.sqrt(vc / ((m - 1.0) * n)) if m > 1 else 0.0
    b_all, e1 = (q * bq + (n - q) * br) / n, (q * eq + (n - q) * er) / n
    dq = ((delta + bq + eq) * q).sum(1) / q.sum()
    d1 = ((delta + b_all + e1) * n).sum(1) / N
    dm_h = delta + b_all + (e1 + (m - 1.0) * e234) / m
    dm = (dm_h * n).sum(1) / N
    se = max(math.sqrt(float(((n / N) ** 2 * (vb + vc / m) / n).sum())), 1.0 / N)
    good = (d1 > 0) & (dm / se >= t_ppf(1.0 - alpha / max(K, 1), max(N - 1.0, 1.0)))
    if is_w:
        good &= dq > 0
    if is_global:
        tot = (dm_h * n).sum(1)
        good &= (len(hs) >= 2) & np.all(tot[:, None] - dm_h * n > 0, axis=1)
    return float(good.mean())


# ---------------------------------------------------------------------------------------------------------------- the cell maker
def read_budget(run=None, value=None, source=None, base=None):
    """The one stated budget: the allowed token multiple M with its source. From the arguments, else RUN/cards.json "g_token_multiple"
    (a number, or {"value", "source", "base"}) with "g_token_multiple_source" / "g_token_base". M is the measured token multiple of the
    stock model's own xhigh effort over medium on signal rows (coordinator 11-11); there is no default: absent -> ValueError.
    base = tokens per read the multiple refers to (the medium reference); absent = the tip's own tokens per read."""
    if value is None and run:
        p = os.path.join(run, "cards.json")
        if os.path.exists(p):
            c = json.load(open(p))
            v = c.get("g_token_multiple")
            if isinstance(v, dict):
                value, source, base = v.get("value"), v.get("source") or c.get("g_token_multiple_source"), v.get("base", c.get("g_token_base"))
            elif v is not None:
                value, source, base = v, c.get("g_token_multiple_source"), c.get("g_token_base")
            source = f"{p} g_token_multiple" + (f": {source}" if source else " (no source recorded)")
    if value is None:
        raise ValueError("no stated token budget: RUN/cards.json \"g_token_multiple\" (the measured xhigh / medium token multiple) or --budget is required")
    if not float(value) >= 1.0:
        raise ValueError(f"token multiple {value!r} must be >= 1")
    return dict(M=float(value), source=source or "argument (no source recorded)", base=float(base) if base else None)


def calib_paths(sig_dir):
    """(RUN/signal_calibration.json, RUN/signal_quality.txt): beside RUN/signal, not inside it (that directory holds one
    directory per tip and nothing else: t_signal and other readers list it)."""
    run = os.path.dirname(os.path.normpath(sig_dir))
    return os.path.join(run, "signal_calibration.json"), os.path.join(run, "signal_quality.txt")


def load_calibration(sig_dir):
    p = calib_paths(sig_dir)[0]
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return None


def shrink_of(calib, kind):
    """(slope b, its sd, where it comes from) for a cell kind: the kind's own posterior when it has realised pairs, else the pooled
    one, else 1 (no realised pair yet: predictions pass through, marked uncalibrated). Never below 0."""
    S = (calib or {}).get("slopes") or {}
    for k, label in ((kind, f"{kind}, {{n}} pairs"), ("pooled", "pooled, {n} pairs")):
        s = S.get(k)
        if s and s.get("n") and s.get("b") is not None:
            return max(float(s["b"]), 0.0), float(s["sd"]), label.format(n=s["n"])
    return 1.0, 0.0, "uncalibrated (no realised pair yet)"


def global_option(cid, x, ref, R, votes, rule, t0pop):
    """A cell on all rows under both pairings -> (views, the option in use, est). The option in use = the pairing with the smaller
    cluster-robust se of the judge-scale gain (a tie keeps "row"); o["written"] = the row (judge-convention) value when the two differ
    by less than the larger of their se (or row is in use), else the value in use. (views, None, None) with fewer than 2 paired rows."""
    V = views(x, ref)
    G = {pm: option(cid, "all", *vec_all(V[pm]), R, votes, rule, t0pop) for pm in ("row", "key") if len(V[pm]) >= 2}
    G = {pm: o for pm, o in G.items() if o["sP"]["se"] < INF}
    if not G:
        return V, None, None
    use = min(G, key=lambda pm: (G[pm]["sP"]["se"], pm != "row"))
    agree = len(G) == 2 and abs(G["row"]["sP"]["d"] - G["key"]["sP"]["d"]) < max(G["row"]["sP"]["se"], G["key"]["sP"]["se"])
    o = G[use]
    o["written"] = G["row"]["sP"]["d"] if (agree or use == "row") else o["sP"]["d"]
    est = {pm: dict(d=_f(G[pm]["sD"]["d"]), se=_f(G[pm]["sD"]["se"]), pred=_f(G[pm]["sP"]["d"]), pred_se=_f(G[pm]["sP"]["se"]), rows=G[pm]["sD"]["n"]) for pm in G}
    est.update(use=use, agree=bool(agree) if len(G) == 2 else None,
               rule="smaller cluster-robust se (a tie keeps row); pred = the row (judge-convention) value when |row - key| < the larger se")
    return V, o, est


def tip_numbers(tip, R, budget, rule="round"):
    """(tokens per read of the tip on signal rows, added tokens per row the budget leaves, vote gain rows by m, samples per row,
    whether the votes had to use right / wrong classes because the tip table carries no pred)."""
    t0pop = strat_stat({r: sum(q["tokens"] for q in sm.values()) / len(sm) for r, sm in tip.items()}, R)["d"]
    spr = sorted({len(sm) for sm in tip.values()})
    by_ok = not any(q.get("pred") is not None for sm in tip.values() for q in sm.values())
    votes = {m: vote_rows(tip, R, m, by_ok) for m in range(2, min(spr[-1] if spr else 1, PARENT_M) + 1)} if rule != "off" else {}
    return t0pop, budget["M"] * (budget.get("base") or t0pop) - t0pop, votes, spr, by_ok


def _name(spec, cid):
    return str(spec.get("name") or cid)


def prior_problems(cells):
    """B4b (coordinator 11-11): every weight cell states prior_family as an integer >= 0; a library, when given, is a non-empty string."""
    out = []
    for cid, sp in (cells or {}).items():
        pf, lib = sp.get("prior_family"), sp.get("library")
        if sp.get("kind") == "weight" and pf is None:
            out.append(f"{cid}: no prior_family")
        elif pf is not None and (isinstance(pf, bool) or not isinstance(pf, int) or pf < 0):
            out.append(f"{cid}: prior_family {pf!r} is not an integer >= 0")
        if lib is not None and (not isinstance(lib, str) or not lib.strip()):
            out.append(f"{cid}: library {lib!r} is not a non-empty string")
    return out


def family_prior(items):
    """(n, {library: prior}): prior cells summed over the DISTINCT libraries of the items, within one library its largest prior_family
    (the judge's g_prior in author_g.py does the same with the candidates' c["g"])."""
    lib = {}
    for it in items:
        if it.get("prior_family") is None:
            continue
        k = it.get("library") or it.get("family") or "?"
        lib[k] = max(lib.get(k, 0), int(it["prior_family"]))
    return sum(lib.values()), lib


def make(tip_id, tip, cells, reads, R, budget, classes=None, n_len=2, calib=None, rule="round", refs=None, caps=None, alpha=ALPHA, control=None):
    """The cell maker: -> the cells.json document (see the module text). tip / reads are checked row by row before anything is computed.
    control = the id of a concurrent control cell (the tip read on the same engine, rows and sample indices, e.g. W-null): once it has
    reads, every weight cell's primary gain is against it (a cell's own spec["control"] wins) and its gain against the tip's cached
    reads is kept beside it as secondary (est["vs_tip"]); the control itself is reported against the tip and never proposed."""
    for r in tip:
        R.check(r, "tip table")
    for cid, tb in reads.items():
        for r in tb:
            R.check(r, f"reads of {cid}")
    classes = list(classes or [])
    for c in classes:
        for r in c.get("rows") or ():
            R.check(r, f"class {c['name']}")
    bad = prior_problems(cells)
    if bad:
        raise ValueError("prior_family is required for every weight cell (an integer >= 0: cells of its library read before on sets "
                         "holding JUDGE rows; batch_cells.json / the mapping / the library entry states it): " + "; ".join(bad))
    acc0 = strat_stat({r: sum(q["ok"] for q in sm.values()) / len(sm) for r, sm in tip.items()}, R)
    t0pop, allow, votes, spr, votes_by_ok = tip_numbers(tip, R, budget, rule)
    lcl = length_classes(tip, n_len)
    pop_w = {}                                                   # population share of each class (all signal rows of the tip table)
    for c in classes:
        pop_w[c["name"]] = strat_stat({r: 1.0 if r in c["rows"] else 0.0 for r in tip}, R)["d"]
    for c in lcl:
        pop_w[c["name"]] = strat_stat({r: sum(1.0 for q in sm.values() if c["lo"] < q["tokens"] <= c["hi"]) / len(sm) for r, sm in tip.items()}, R)["d"]

    # -- every measured cell: both pairings, the one in use, its options per class
    cmap = {c["name"]: c for c in classes + lcl}
    tipset = set(tip)

    def ctl_of(cid):
        c = cells[cid].get("control") or (control if cid != control and cells[cid].get("kind") == "weight" else None)
        return c if c and reads.get(c) else None

    def refof(cid):
        return reads[ctl_of(cid)] if ctl_of(cid) else tip

    def build(cid, cname, x, use):
        """The option "cell cid on class cname" from the reads x of the cell, under the cell's pairing."""
        if cname == "all" or cname.startswith("rest of ") or cmap[cname]["type"] == "cond":
            rows = None if cname == "all" else (tipset - set(cmap[cname[8:]]["rows"])) if cname.startswith("rest of ") else cmap[cname]["rows"]
            sh = 1.0 if cname == "all" else (1.0 - pop_w[cname[8:]]) if cname.startswith("rest of ") else pop_w[cname]
            return option(cid, cname, *vec_all(views(x, refof(cid))[use], rows), R, votes, rule, t0pop, share=sh)
        c = cmap[cname]
        return option(cid, cname, *vec_len(x, refof(cid), c["lo"], c["hi"], bool(cells[cid].get("continues")), use), R, votes, rule, t0pop, share=pop_w[cname])
    M, skipped = {}, []
    for cid, spec in cells.items():
        x = reads.get(cid) or {}
        V, o, est = global_option(cid, x, refof(cid), R, votes, rule, t0pop)
        if o is None:
            skipped.append(dict(name=_name(spec, cid), why=f"too few paired rows ({len(V['row'])})"))
            continue
        est["against"] = f"control cell {_name(cells.get(ctl_of(cid)) or {}, ctl_of(cid))}" if ctl_of(cid) else "the tip's cached reads"
        if ctl_of(cid):                                          # secondary: the same cell against the tip's cached reads
            _, o2, e2 = global_option(cid, x, tip, R, votes, rule, t0pop)
            est["vs_tip"] = {k: e2[k] for k in ("row", "key", "use") if k in e2} if e2 else None
        est["not_stopped"] = cut_stats(x, refof(cid))
        M[cid] = dict(spec=spec, glob=o, est=est, use=est["use"], view=V[est["use"]], reads=sum(len(sm) for sm in x.values()),
                      per_row=sorted({len(sm) for sm in x.values()}), byc={c["name"]: build(cid, c["name"], x, est["use"]) for c in classes + lcl})
    rr = reread_reads(tip)
    ctl = {c["name"]: option("re-read (tip weights, next seed)", c["name"], *vec_len(rr, tip, c["lo"], c["hi"]), R, votes, rule, t0pop,
                             share=pop_w[c["name"]]) for c in lcl} if rr else {}

    def inc(o0, o1):
        """Increment between two cells on the keys they share: when their sample indices differ on the common rows, both are rebuilt
        on the common (row, sample) keys first (seeds are common by (row, sample))."""
        if o0 is None:
            return o1["sP"], o1["Aall"]
        x0, x1 = reads[o0["id"]], reads[o1["id"]]
        rows = [r for r in x0 if r in x1]
        if all(set(x0[r]) == set(x1[r]) for r in rows):
            return inc_stat(o0, o1, R)
        y0 = {r: {s: q for s, q in x0[r].items() if s in x1[r]} for r in rows}
        y0 = {r: v for r, v in y0.items() if v}
        if len(y0) < 2:
            return inc_stat(o0, o1, R)
        return inc_stat(build(o0["id"], o0["cls"], y0, M[o0["id"]]["use"]), build(o1["id"], o1["cls"], {r: {s: x1[r][s] for s in y0[r]} for r in y0}, M[o1["id"]]["use"]), R)

    # -- partitions: global; the read-length bins; one binary partition per router class (the class and the rest)
    nonprop = lambda sp: sp.get("kind") in ("reference", "control") or sp.get("propose") is False   # noqa: E731 (measured and reported, never proposed)
    cand = [cid for cid, x in M.items() if not nonprop(x["spec"])]
    mn = {cid: menu_of(M[cid]["spec"], cid) for cid in cand}      # cells of one menu are alternatives; menus (families) add up

    def menus(per_class):
        out = collections.defaultdict(list)
        for cname, opts in per_class.items():
            for o in opts:
                out[(cname, mn[o["id"]])].append(o)
        return dict(out)
    parts = [("global", menus({"all": [M[cid]["glob"] for cid in cand]}))]
    if lcl:
        parts.append(("length", menus({c["name"]: [M[cid]["byc"][c["name"]] for cid in cand] for c in lcl})))
    for c in classes:
        parts.append((c["name"], menus({c["name"]: [M[cid]["byc"][c["name"]] for cid in cand],
                                        "rest of " + c["name"]: [build(cid, "rest of " + c["name"], reads[cid], M[cid]["use"]) for cid in cand]})))
    plans, chosen, notes = {}, [], []
    for pname, menu in parts:
        menu = {k: [o for o in v if o["sP"]["n"] >= 2 and o["sP"]["se"] < INF] for k, v in menu.items()}
        opts = [o for v in menu.values() for o in v]
        ps = [pval(o["sP"]) for o in opts]
        sel = bh(ps, alpha)
        level = alpha * max(len(sel), 1) / max(len(opts), 1)
        for i, o in enumerate(opts):
            o.update(p=ps[i], bh=i in sel, lb=bound(o["sP"], level), level=level, part=pname)
        fr = frontier(menu, allow, level, R, inc)
        ch = [o for k, v in menu.items() for o in v if fr["choice"][k] == o["id"]]
        tot = None
        if ch:
            rows = set.intersection(*[set(o["P"]) for o in ch])
            tot = strat_stat({r: sum(o["P"][r] for o in ch) for r in rows}, R, floor=True)
        plans[pname] = dict(K=len(opts), selected=len(sel), level=level, lam=fr["lam"], steps=fr["steps"], choice=fr["choice"], total=tot, menu=menu,
                            used=sum(o["Aall"] for o in ch), menus=len({k[1] for k in menu}))
        same = set()                                             # a menu whose every class takes the same cell: that is the global cell
        for m_ in {k[1] for k in menu} if pname != "global" else ():
            got = {fr["choice"][k] for k in menu if k[1] == m_}
            if len(got) == 1 and None not in got and len([k for k in menu if k[1] == m_]) > 1:
                same.add(m_)
                notes.append(f"plan {pname}: {_name(cells[next(iter(got))], '')} in every class: that is the global cell, nothing routed")
        for st in fr["steps"]:
            if st["key"][1] not in same:
                chosen.append((pname, st, next(o for o in menu[st["key"]] if o["id"] == st["cell"])))

    def by_class(x):
        return {n_: dict(share=_f(c["share"], 4), gain=_f(c["sD"]["d"] / c["share"]) if c["share"] > 0 else None,
                         se=_f(c["sD"]["se"] / c["share"]) if c["share"] > 0 else None, pred_all_rows=_f(c["Pall"]),
                         tokens_ratio=_f(c["c"], 3), tokens_added_all_rows=_f(c["Aall"], 1), rows=sum(1 for v in c["w"].values() if v > 0),
                         pp_per_ktok=_f(1e5 * c["Pall"] / c["Aall"], 3) if c["Aall"] > 0 else None) for n_, c in x["byc"].items()}

    # -- items: the accepted vertices (a class that cannot be routed to is noted, not emitted)
    bench_n = R.n_judge or {b: 0 for b in R.W}
    items = []
    for pname, st, o in chosen:
        cid, cls = o["id"], o["cls"]
        x = M[cid]
        spec, kind = x["spec"], x["spec"].get("kind")
        if cls.startswith("rest of "):
            notes.append(f"plan {pname}: {_name(spec, cid)} on the rows where the condition does NOT hold "
                         f"({100 * o['Pall']:+.2f} pp on all rows): no router condition says that, not emitted")
            continue
        routed = cls != "all"
        sh = o["share"] if routed else 1.0
        b, bsd, bfrom = shrink_of(calib, "routed" if routed else kind)
        raw = (o.get("written", o["sP"]["d"])) / sh
        se_raw, lb_raw = o["sP"]["se"] / sh, o["lb"] / sh
        tq = t_ppf(1.0 - o["level"], o["sP"]["nu"])
        se = math.sqrt((b * se_raw) ** 2 + (bsd * raw) ** 2)
        lb = b * min(lb_raw, raw) - tq * bsd * abs(raw)           # the bound of the estimator in use, never above pred; wider by the slope's sd
        last = plans[pname]["choice"].get(st["key"]) == cid       # the menu's final vertex in its plan (earlier ones are cheaper alternatives)
        it = dict(kind="routed" if routed else kind, name=_name(spec, cid) + (f" on {cls}" if routed else ""), id=cid, family=spec.get("family"),
                  prior_family=int(spec.get("prior_family") or 0), library=spec.get("library") or spec.get("family"),
                  pred=_f(b * raw), se=_f(se), lb=_f(lb), pred_tokens_ratio=_f(o["c"], 3),
                  pred_raw=_f(raw), se_raw=_f(se_raw), lb_raw=_f(lb_raw), shrink=_f(b, 4), shrink_from=bfrom,
                  gain=_f(o["sD"]["d"] / sh), gain_se=_f(o["sD"]["se"] / sh), ref_m=o["m"], p=_f(o["p"], 6), level=_f(o["level"], 6),
                  share=_f(sh, 4), contribution=_f(b * raw * sh), contribution_lb=_f(lb * sh), plan_choice=last,
                  tokens_multiple=_f(o["mult"], 3), tokens_added=_f(o["Aall"], 1),
                  gain_per_ktok=_f(1e5 * o["Pall"] / o["Aall"], 3) if o["Aall"] > 0 else None,
                  n_rows=o["sP"]["n"], n_reads=x["reads"], samples_per_row=x["per_row"], nu=_f(o["sP"]["nu"], 1), plan=pname,
                  frontier=dict(order=st["order"], over=st["frm"] and _name(cells[st["frm"]], st["frm"]), added_tokens=_f(st["added_tokens"], 1),
                                gain=_f(st["gain"]), pp_per_ktok=_f(1e5 * st["ratio"], 3) if st["ratio"] is not None else None,
                                inc_lb=_f(st["inc_lb"]), below_lambda=st["below_lambda"]))
        for k in ("alpha", "band", "dose", "control", "continues"):
            if spec.get(k) is not None:
                it[k] = spec[k]
        if not routed:
            it["est"] = x["est"]
            it["by_class"] = by_class(x)
        else:
            c = cmap[cls]
            it["base_kind"] = kind
            it["route"] = (dict(type="cond", condition=c["condition"], share=_f(sh, 4), before_read=True) if c["type"] == "cond" else
                           dict(type="length", gt=_f(c["lo"], 1), le=_f(c["hi"], 1), quantiles=c["quantiles"], share=_f(sh, 4), before_read=False,
                                needs="an engine op: read with the tip first, re-read with the cell when the length is in the class"))
        for k in ("edit", "settings", "ops"):
            if spec.get(k):
                it[k] = spec[k]
        # the judge's test: variance parts from the cell's own rows, the class's JUDGE rows per benchmark, benchmark deviations shrunk
        by = collections.defaultdict(list)
        for r, v in o["P"].items():
            if o["w"][r] > 0:
                by[R.bench(r)].append(v / o["w"][r])
        gh = {h: float(np.mean(v)) for h, v in by.items() if len(v) >= 2}
        sh2 = {h: float(np.var(by[h], ddof=1) / len(by[h])) for h in gh}
        kap = max(0.0, 1.0 - float(np.mean(list(sh2.values()))) / float(np.var(list(gh.values()), ddof=1))) if len(gh) >= 2 and np.var(list(gh.values())) > 0 else 0.0
        gbar = float(np.mean(list(gh.values()))) if gh else 0.0
        clsrows = None if not routed or cmap[cls]["type"] != "cond" else cmap[cls]["rows"]
        wb = collections.Counter(R.bench(r) for r in (clsrows if clsrows is not None else R.sig) if r in R.meta)
        nb = collections.Counter(R.bench(r) for r in R.sig)
        frac = {h: (wb[h] / nb[h] if clsrows is not None else (sh if routed else 1.0)) for h in nb}
        it["_judge"] = dict(parts=variance_parts({r: v for r, v in x["view"].items() if o["w"].get(r, 0) > 0}, {r: o["D"][r] / o["w"][r] for r in o["D"] if o["w"][r] > 0}, R),
                            n_bench={h: int(round(frac.get(h, 0.0) * n)) for h, n in bench_n.items()}, dev={h: kap * (g - gbar) for h, g in gh.items()},
                            is_w=kind == "weight", is_global=not routed)
        items.append(it)

    # -- what is sent: p_pass at K = cells sent (within author G's caps) + the prior cells of the distinct libraries among them; below alpha: held
    caps = caps or {}
    items.sort(key=lambda it: (-(it["contribution_lb"] if it["contribution_lb"] is not None else -1.0), it["frontier"]["order"]))
    seen, uniq = set(), []
    for it in items:                                             # one line per (cell, class): the best plan's
        k = (it["id"], json.dumps(it.get("route"), sort_keys=True))
        if k not in seen:
            seen.add(k)
            uniq.append(it)
    sent, held = [], []
    for it in uniq:
        need = "edit" if (it.get("base_kind") or it["kind"]) == "weight" else "settings" if (it.get("base_kind") or it["kind"]) == "setting" else None
        if need and not it.get(need):
            held.append(dict(it, why_held=f"no {need} in the cell's spec (batch_cells.json / the mapping must name it): it cannot be proposed"))
        else:
            sent.append(it)
    while True:
        nw = sum(1 for it in sent if it["kind"] == "weight")
        ns = sum(1 for it in sent if it["kind"] == "setting")
        n_prop = min(nw, caps["g_cells"]) if caps.get("g_cells") is not None else nw
        n_prop += min(ns, caps["g_settings"]) if caps.get("g_settings") is not None else ns
        n_prop += sum(1 for it in sent if it["kind"] not in ("weight", "setting"))
        K = n_prop + family_prior(sent)[0]                      # B4b: the sum over the distinct libraries sent (as author_g.g_prior)
        drop = []
        for it in sent:
            j = it["_judge"]
            it["K"] = K
            it["p_pass"] = _f(pass_prob(it["pred"], it["se"], j["parts"], j["n_bench"], j["dev"], K, j["is_w"], j["is_global"], alpha), 4) if sum(j["n_bench"].values()) else None
            if it["p_pass"] is not None and it["p_pass"] < alpha:
                drop.append(it)
        if not drop:
            break
        worst = min(drop, key=lambda it: it["p_pass"])
        worst["why_held"] = f"p_pass {worst['p_pass']:.3f} < {alpha} at K {K} ({sum(worst['_judge']['n_bench'].values())} JUDGE rows x {JUDGE_M} samples): it cannot pass the judge"
        sent.remove(worst)
        held.append(worst)
    for it in uniq:
        j = it.pop("_judge")
        it["judge_plan"] = dict(rows=sum(j["n_bench"].values()), samples=JUDGE_M, parent_samples=PARENT_M, K=it.get("K"),
                                tau2=_f(j["parts"]["tau2"]), var_cell=_f(j["parts"]["wc"]), var_tip=_f(j["parts"]["wt"]),
                                futility="AM17 quarter + LITE step 1" if j["is_w"] else "LITE step 1", lobo=j["is_global"])
    emitted = {it["id"] for it in uniq if it["kind"] != "routed"}
    not_sel = []
    for cid, x in M.items():
        if cid in emitted:
            continue
        o, spec = x["glob"], x["spec"]
        ref = nonprop(spec)
        not_sel.append(dict(kind=spec.get("kind"), name=_name(spec, cid), id=cid, family=spec.get("family"), propose=not ref, by_class=by_class(x),
                            note=spec.get("note"), pred_raw=_f(o["sP"]["d"]),
                            se_raw=_f(o["sP"]["se"]), lb_raw=_f(o.get("lb")), gain=_f(o["sD"]["d"]), gain_se=_f(o["sD"]["se"]), ref_m=o["m"],
                            p=_f(o.get("p"), 6), tokens_multiple=_f(o["mult"], 3), tokens_added=_f(o["Aall"], 1),
                            gain_per_ktok=_f(1e5 * o["Pall"] / o["Aall"], 3) if o["Aall"] > 0 else None, n_rows=o["sP"]["n"], n_reads=x["reads"],
                            est=x["est"], why=("reference point (never proposed)" if spec.get("kind") == "reference" else
                                               "concurrent control (the tip on the cells' engine; never proposed)" if spec.get("kind") == "control" else
                                               "measurement only (never proposed)" + (f": {spec['note']}" if spec.get("note") else "")) if ref else o.get("why_not") or
                            ("bound <= 0 (not BH-selected)" if not o.get("bh") else "not on the frontier (another cell gains more per added token)")))
    g = plans["global"]
    doc = dict(tip=tip_id, scale=SCALE + "; pred on the judge's scale (equal-compute reference), x calibration slope", written=time.time(), maker=MAKER,
               cells=sent, held=held, not_selected=not_sel, skipped=skipped, notes=notes,
               budget=dict(M=budget["M"], source=budget["source"], base_tokens=_f(budget.get("base") or t0pop, 1),
                           base_is="stated medium reference" if budget.get("base") else "the tip's own tokens per read",
                           added_tokens_allowed=_f(allow, 1)),
               tip_stats=dict(rows=len(tip), samples_per_row=[spr[0], spr[-1]] if spr else None, accuracy=_f(acc0["d"]), tokens=_f(t0pop, 1)),
               alpha=alpha, lam=_f(g["lam"], 8), lam_pp_per_ktok=_f(1e5 * g["lam"], 3) if g["lam"] is not None else None, ecomp=rule,
               control=dict(cell=_name(cells.get(control) or {}, control), state="in use" if reads.get(control) else
                            ("no reads yet" if control in cells else "not in the input yet") + ": cells are against the tip's cached reads") if control else None,
               votes_from="right / wrong classes (the tip table has no pred): a lower bound of the vote gain" if votes_by_ok and votes else "the tip's preds",
               refs=list(refs or []), votes={str(m): dict(gain=_f(strat_stat(v, R)["d"]), tokens_multiple=m) for m, v in votes.items() if v},
               classes={c["name"]: dict(type=c["type"], share=_f(pop_w[c["name"]], 4), **({"condition": c["condition"]} if c["type"] == "cond" else
                                                                                           {"gt": _f(c["lo"], 1), "le": _f(c["hi"], 1), "quantiles": c["quantiles"]}))
                        for c in classes + lcl},
               reread={n_: dict(share=_f(c["share"], 4), pred_all_rows=_f(c["Pall"]), se=_f(c["sP"]["se"]), tokens_added_all_rows=_f(c["Aall"], 1)) for n_, c in ctl.items()},
               plans={pn: dict(K=p["K"], selected=p["selected"], level=_f(p["level"], 6), lam=_f(p["lam"], 8), added_tokens=_f(p["used"], 1),
                               tokens_multiple=_f(1.0 + p["used"] / t0pop, 3) if t0pop > 0 else None,
                               total=dict(pred=_f(p["total"]["d"]), se=_f(p["total"]["se"]), lb=_f(bound(p["total"], p["level"]))) if p["total"] else None,
                               choice={" | ".join(k): (v and _name(cells[v], v)) for k, v in p["choice"].items()}, menus=p["menus"],
                               steps=[dict(order=s["order"], cls=s["key"][0], menu=s["key"][1], cell=_name(cells[s["cell"]], s["cell"]),
                                           over=s["frm"] and _name(cells[s["frm"]], s["frm"]),
                                           added_tokens=_f(s["added_tokens"], 1), gain=_f(s["gain"]), inc_lb=_f(s["inc_lb"]),
                                           pp_per_ktok=_f(1e5 * s["ratio"], 3) if s["ratio"] is not None else None, below_lambda=s["below_lambda"]) for s in p["steps"]])
                      for pn, p in plans.items()},
               calibration=dict(state=(calib or {}).get("state") or "none", quality=(calib or {}).get("quality"),
                                slopes={k: dict(b=_f(v.get("b"), 4), sd=_f(v.get("sd"), 4), n=v.get("n")) for k, v in ((calib or {}).get("slopes") or {}).items()}))
    return doc


# ---------------------------------------------------------------------------------------------------------------- files and the table
def write_cells(sig_tip_dir, doc):
    """cells.json (atomic), its archived copy cells_<sha>.json (sha = author_g's cells_sha: realised.jsonl lines are joined to the
    predictions through it) and cells_table.txt. Returns (path, sha)."""
    os.makedirs(sig_tip_dir, exist_ok=True)
    raw = json.dumps(doc, ensure_ascii=False, indent=1).encode()
    sha = _sha16(raw)
    for name in ("cells.json", f"cells_{sha}.json"):
        p = os.path.join(sig_tip_dir, name)
        with open(p + ".tmp", "wb") as fh:
            fh.write(raw)
        os.replace(p + ".tmp", p)
    with open(os.path.join(sig_tip_dir, "cells_table.txt"), "w") as fh:
        fh.write(table(doc, sha))
    return os.path.join(sig_tip_dir, "cells.json"), sha


def _pp(x, w=6, k=2):
    return f"{100 * x:+{w}.{k}f}" if isinstance(x, (int, float)) else " " * (w - 1) + "-"


def table(doc, sha=None):
    """The human-readable table of a cells document (aggregates only)."""
    L = []
    ts, b = doc["tip_stats"], doc["budget"]
    L.append(f"signal cells for tip {doc['tip']} ({doc['maker']}; {cst(doc['written'])}"
             + (f"; cells_sha {sha}" if sha else "") + ")")
    L.append(f"tip on signal rows: {ts['rows']} rows, {ts['samples_per_row']} samples per row, accuracy {100 * ts['accuracy']:.1f} %, {ts['tokens']:.0f} tokens per read")
    L.append(f"budget: token multiple <= {b['M']:g} of {b['base_tokens']:.0f} tokens ({b['base_is']}) = {b['added_tokens_allowed']:+.0f} tokens per read; source: {b['source']}")
    L.append("lambda (shadow price of the budget, global plan): " + ("0: the budget does not bind" if doc["lam"] is None else f"{doc['lam_pp_per_ktok']:.3f} pp per 1k tokens"))
    for r in doc.get("refs") or []:
        r = dict(r, **{k: r[k] / 100.0 for k in ("gain", "se") if isinstance(r.get(k), (int, float)) and abs(r[k]) > 1})   # given in pp
        L.append(f"reference point {r.get('name')}: gain {_pp(r.get('gain'), 0)} pp" + (f" (se {100 * r['se']:.2f})" if isinstance(r.get("se"), (int, float)) else " (se not given)")
                 + (f", tokens x{r['tokens_ratio']:.2f}" if isinstance(r.get("tokens_ratio"), (int, float)) else "") + (f", {r['rows']} rows" if r.get("rows") else "")
                 + (f" [{r['source']}]" if r.get("source") else ""))
    if not any("xhigh" in str(r.get("name", "")).lower() for r in (doc.get("refs") or [])) and not any("xhigh" in str(x.get("name", "")).lower() for x in doc["not_selected"]):
        L.append("reference point xhigh: NOT SUPPLIED (give --ref FILE or a measured cell of kind \"reference\")")
    if doc.get("votes"):
        L.append("vote of m tip reads (free, from the tip's own samples): " + ", ".join(f"m {m}: {_pp(v['gain'], 0)} pp at x{m}" for m, v in sorted(doc["votes"].items()))
                 + ("" if doc.get("votes_from") in (None, "the tip's preds") else f"   [{doc['votes_from']}]"))
    if doc.get("control"):
        L.append(f"concurrent control: {doc['control']['cell']} ({doc['control']['state']}); a weight cell's gain, tokens and classes are against it when it is in use, "
                 "its gain against the tip's cached reads is the secondary number at the end of its line")
    g = doc["plans"]["global"]
    L.append(f"bounds: BH at {doc['alpha']} over K {g['K']} measured cells -> {g['selected']} selected; every bound at level {g['level']:.4f} (false-coverage-adjusted)")
    L.append("")
    hdr = f"{'':4s} {'cell':34s} {'rows':>4s} {'reads':>5s} {'gain':>7s} {'se':>5s} {'pred':>7s} {'se':>5s} {'lb':>7s} {'m':>2s} {'tok x':>6s} {'+tok':>6s} {'pp/1k':>6s} " \
          f"{'row d (se)':>13s} {'key d (se)':>13s} {'use':>4s} {'p_pass':>6s} {'K':>3s}"
    L.append(hdr + "   (pp; pred = judge scale x calibration)")

    def line(tag, it, sel):
        e = it.get("est") or {}
        ek = lambda k: (f"{100 * e[k]['d']:+6.2f} ({100 * e[k]['se']:4.2f})" if e.get(k) else f"{'-':>13s}")   # noqa: E731
        pr, se, lb = (it.get("pred"), it.get("se"), it.get("lb")) if sel else (it.get("pred_raw"), it.get("se_raw"), it.get("lb_raw"))
        return (f"{tag:4s} {str(it.get('name'))[:34]:34s} {it.get('n_rows') or 0:4d} {it.get('n_reads') or 0:5d} {_pp(it.get('gain'), 7)} {100 * (it.get('gain_se') or 0):5.2f} {_pp(pr, 7)} "
                f"{100 * (se or 0):5.2f} {_pp(lb, 7)} {str(it.get('ref_m') or '-'):>2s} {('x%.2f' % it['tokens_multiple']) if it.get('tokens_multiple') else '-':>6s} "
                f"{(it.get('tokens_added') or 0):+6.0f} {('%.2f' % it['gain_per_ktok']) if it.get('gain_per_ktok') is not None else '-':>6s} {ek('row')} {ek('key')} "
                f"{e.get('use') or '-':>4s} {('%.2f' % it['p_pass']) if it.get('p_pass') is not None else '-':>6s} {str(it.get('K') or '-'):>3s}")
    def sec(it):
        e = it.get("est") or {}
        v, ns = e.get("vs_tip"), e.get("not_stopped")
        out = ""
        if ns and (ns["cell"] or ns["ref"]):
            out += (f"   not stopped {100 * ns['cell']:.1f} % (reference {100 * ns['ref']:.1f} %)"
                    + (f", tokens x{ns['tokens_multiple_stopped']:.2f} among the reads that stopped" if ns.get("tokens_multiple_stopped") else ""))
        if v:
            out += "   vs the tip's cached reads: " + ", ".join(f"{k} {100 * v[k]['d']:+.2f} ({100 * v[k]['se']:.2f})" for k in ("row", "key") if v.get(k))
        return out

    def classes(it):
        for n_, c in (it.get("by_class") or {}).items():
            L.append(f"       {'on ' + n_:31s} share {c['share']:.2f}  rows {c['rows']:3d}  gain {_pp(c['gain'], 7)} (se {100 * (c['se'] or 0):.2f})  tokens x"
                     f"{c['tokens_ratio'] or 0:.2f}  all rows: {_pp(c['pred_all_rows'], 0)} pp, {c['tokens_added_all_rows']:+.0f} tok"
                     + (f", {c['pp_per_ktok']:.2f} pp/1k" if c.get("pp_per_ktok") is not None else ""))
    for i, it in enumerate(doc["cells"]):
        L.append(line(f"G{i}", it, True) + (f"   routed: {it['route']['type']} share {it['share']:.2f}" if it.get("kind") == "routed" and it.get("route") else "")
                 + (f"   prior_family {it['prior_family']}" if it.get("prior_family") else "") + sec(it))
        classes(it)
    for it in doc.get("held") or []:
        L.append(line("held", it, True) + f"   {it.get('why_held')}" + sec(it))
        classes(it)
    for it in doc["not_selected"]:
        tag = "ref" if it.get("kind") == "reference" else "ctl" if it.get("kind") == "control" else "meas" if it.get("propose") is False else "-"
        L.append(line(tag, it, False) + f"   {it.get('why')}" + sec(it))
        if tag != "-":
            classes(it)
    for x in doc.get("skipped") or []:
        L.append(f"skip {x['name']}: {x['why']}")
    L.append("")
    for n_, c in (doc.get("classes") or {}).items():
        L.append(f"class {n_}: share {c['share']:.2f}" + (f"  condition: {c['condition'][:110]}" if c.get("condition") else f"  tip read length quantiles {c.get('quantiles')}"))
    for n_, c in (doc.get("reread") or {}).items():
        L.append(f"control on {n_}: re-read with the tip itself (next seed): {_pp(c['pred_all_rows'], 0)} pp on all rows (se {100 * (c['se'] or 0):.2f}), {c['tokens_added_all_rows']:+.0f} tok")
    L.append("")
    for pn, p in doc["plans"].items():
        if not p["steps"]:
            L.append(f"plan {pn}: nothing kept (K {p['K']}, {p['selected']} BH-selected)")
            continue
        t = p["total"] or {}
        L.append(f"plan {pn}: total {_pp(t.get('pred'), 0)} pp (se {100 * (t.get('se') or 0):.2f}, lb {_pp(t.get('lb'), 0)}) at x{p['tokens_multiple']:.2f} tokens"
                 + (f" ({p['menus']} menus, taken as adding up)" if p.get("menus", 1) > 1 else "")
                 + (f"; lambda {1e5 * p['lam']:.2f} pp/1k" if p["lam"] is not None else "") + f"; K {p['K']}, level {p['level']:.4f}")
        for s in p["steps"]:
            L.append(f"   {s['order']}. [{s['cls']} | {s['menu']}] {s['over'] or 'tip'} -> {s['cell']}: {_pp(s['gain'], 0)} pp for {s['added_tokens']:+.0f} tok"
                     + (f" = {s['pp_per_ktok']:.2f} pp/1k" if s["pp_per_ktok"] is not None else " (no added tokens)") + f", increment lb {_pp(s['inc_lb'], 0)}"
                     + (" (below lambda: the cut class's affordable vertex)" if s["below_lambda"] else ""))
    for n_ in doc.get("notes") or []:
        L.append("note: " + n_)
    c = doc.get("calibration") or {}
    L.append("")
    L.append(f"calibration: {c.get('state')}; signal quality Q = {c.get('quality')}; slopes " +
             (", ".join(f"{k} {v['b']} (sd {v['sd']}, n {v['n']})" for k, v in (c.get("slopes") or {}).items()) or "none yet (predictions pass through)"))
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------------------------------------------- calibration
def load_pairs(sig_dir):
    """Predicted vs realised: every line of <sig_dir>/*/realised.jsonl (author_g.g_realised writes it in the parent tip's directory;
    <sig_dir>/realised.jsonl is read too) that has measured numbers; the raw prediction, kind and shrink come from the archived
    cells_<cells_sha>.json of that directory when it exists (else the line's own pred is taken as raw). A measurement shared by
    several lines (duplicates) is used once."""
    files = sorted(glob.glob(os.path.join(sig_dir, "*", "realised.jsonl")))
    top = os.path.join(sig_dir, "realised.jsonl")
    if os.path.exists(top):
        files.append(top)
    arch, pairs, seen = {}, [], set()
    num = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)   # noqa: E731
    for f in files:
        d = os.path.dirname(f)
        for l in open(f):
            try:
                x = json.loads(l)
            except ValueError:
                continue
            if not num(x.get("d")) or not num(x.get("pred")):
                continue
            se = x.get("se_used") if num(x.get("se_used")) else x.get("se")
            se = max(float(se) if num(se) else 0.0, 1.0 / x["n_rows"] if num(x.get("n_rows")) and x["n_rows"] else 0.0)
            key = (x.get("tip"), x.get("gen"), x.get("duplicate_of") or x.get("cid"))
            if se <= 0 or key in seen:
                continue
            seen.add(key)
            sha = x.get("cells_sha")
            if sha and (d, sha) not in arch:
                try:
                    doc = json.load(open(os.path.join(d, f"cells_{sha}.json")))
                    arch[(d, sha)] = {it.get("name"): it for it in (doc.get("cells") or []) if isinstance(it, dict)}
                except (OSError, ValueError):
                    arch[(d, sha)] = {}
            it = (arch.get((d, sha)) or {}).get(x.get("name")) or {}
            raw = x.get("pred_raw") if num(x.get("pred_raw")) else it.get("pred_raw") if num(it.get("pred_raw")) else x["pred"]
            pairs.append(dict(gen=x.get("gen"), tip=x.get("tip"), cid=x.get("cid"), name=x.get("name"), kind=x.get("kind") or it.get("kind") or "weight",
                              family=x.get("family"), pred=float(x["pred"]), pred_se=float(x["pred_se"]) if num(x.get("pred_se")) else 0.0,
                              pred_raw=float(raw), se_raw=float(it["se_raw"]) if num(it.get("se_raw")) else (float(x["pred_se"]) if num(x.get("pred_se")) else 0.0),
                              d=float(x["d"]), se=se, accepted=bool(x.get("accepted")), p_pass=x.get("p_pass", it.get("p_pass"))))
    return pairs


def fit_slope(P, tau0=TAU0):
    """Posterior of the slope b of realised on raw predicted through the origin, prior b ~ N(0, tau0^2); weights 1 / (se_realised^2 +
    b^2 se_pred^2) (iterated); the sd is widened by the misfit chi2 / (n - 1) when that is > 1. p_pos = P(b > 0) from a t with n - 1
    degrees of freedom (one pair: no statement)."""
    n = len(P)
    if n == 0:
        return dict(n=0, b=None, sd=tau0, p_pos=None)
    p, d = np.array([x["pred_raw"] for x in P]), np.array([x["d"] for x in P])
    s, sg = np.array([x["se"] for x in P]), np.array([x["se_raw"] for x in P])
    b = 1.0
    for _ in range(100):
        w = 1.0 / (s ** 2 + (b * sg) ** 2)
        prec = 1.0 / tau0 ** 2 + float((w * p * p).sum())
        nb = float((w * p * d).sum()) / prec
        if abs(nb - b) < 1e-12:
            b = nb
            break
        b = nb
    chi2 = float((w * (d - b * p) ** 2).sum())
    infl = max(1.0, chi2 / (n - 1)) if n >= 2 else 1.0
    sd = math.sqrt(infl / prec)
    den = float((w * p * p).sum())
    return dict(n=n, b=b, sd=sd, ls=float((w * p * d).sum()) / den if den > 0 else None, chi2=chi2, p_pos=(1.0 - t_sf(b / sd, n - 1.0)) if n >= 2 else None)


def quality(P):
    """Agreement of predictions AS WRITTEN (before the judge) with realised gains: Q = 1 - sum w (d - pred)^2 / sum w d^2, w = 1 /
    se_realised^2 (0 = no better than predicting no gain, 1 = exact, < 0 = overclaiming), Spearman rank correlation, the slope of
    realised on written pred, sign agreement and the coverage of the 90 % intervals pred +- 1.645 sqrt(pred_se^2 + se^2)."""
    if not P:
        return dict(n=0, Q=None)
    p, d = np.array([x["pred"] for x in P]), np.array([x["d"] for x in P])
    s, ps = np.array([x["se"] for x in P]), np.array([x["pred_se"] for x in P])
    w = 1.0 / s ** 2
    den = float((w * d * d).sum())
    z = t_ppf(0.95, 1e9)
    spp = float((w * p * p).sum())
    return dict(n=len(P), Q=_f(1.0 - float((w * (d - p) ** 2).sum()) / den, 4) if den > 0 else None, spearman=_f(spearman(p, d), 4),
                slope=_f(float((w * p * d).sum()) / spp, 4) if spp > 0 else None, sign=_f(float(np.mean((p > 0) == (d > 0))), 4),
                coverage=_f(float(np.mean(np.abs(d - p) <= z * np.sqrt(ps ** 2 + s ** 2))), 4), accepted=int(sum(1 for x in P if x["accepted"])),
                mean_pred=_f(float(p.mean())), mean_realised=_f(float(d.mean())))


def calibrate(sig_dir, write=True, tau0=TAU0):
    """Fit the shrinkage and the signal-quality number from every realised pair under sig_dir (RUN/signal). Writes
    RUN/signal_calibration.json ({"slopes": {kind | "pooled": {b, sd, n, ls, p_pos}}, "quality": Q over all pairs, "by_gen": [...],
    "state"}) and RUN/signal_quality.txt."""
    P = load_pairs(sig_dir)
    kinds = sorted({x["kind"] for x in P})
    slopes = {k: fit_slope([x for x in P if x["kind"] == k], tau0) for k in kinds}
    slopes["pooled"] = fit_slope(P, tau0)
    slopes = {k: {a: (_f(v, 5) if isinstance(v, float) else v) for a, v in s.items()} for k, s in slopes.items()}
    gens = sorted({x["gen"] for x in P if x["gen"] is not None})
    by = []
    for g in gens:
        cum = [x for x in P if x["gen"] is not None and x["gen"] <= g]
        qg, qc = quality([x for x in P if x["gen"] == g]), quality(cum)
        f = fit_slope(cum, tau0)
        by.append(dict(gen=g, n=qg["n"], Q_gen=qg["Q"], n_cum=qc["n"], Q=qc["Q"], spearman=qc["spearman"], slope_written=qc["slope"], sign=qc["sign"],
                       coverage=qc["coverage"], slope_raw=_f(f["b"], 4), slope_raw_sd=_f(f["sd"], 4), accepted_cum=qc["accepted"]))
    qa = quality(P)
    out = dict(state=f"{len(P)} realised pairs in {len(gens)} generations" if P else "no realised pair yet (predictions pass through)", pairs=len(P),
               slopes=slopes if P else {}, quality=qa.get("Q"), all=qa, by_gen=by, prior=f"slope ~ N(0, {tau0}^2)", t=time.time(),
               what="quality Q = 1 - sum w (d - pred)^2 / sum w d^2 over realised pairs, pred as written before the judge; cumulative per generation in by_gen")
    if write:
        p, pt = calib_paths(sig_dir)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p + ".tmp", "w") as fh:
            json.dump(out, fh, indent=1)
        os.replace(p + ".tmp", p)
        with open(pt, "w") as fh:
            fh.write(quality_text(out))
    return out


def quality_text(c):
    L = [f"signal quality (author G's predictions against the judge): {c['state']}",
         "Q = 1 - sum w (d - pred)^2 / sum w d^2, pred as written before the judge (0 = no better than predicting no gain, 1 = exact, < 0 = overclaiming)",
         f"{'gen':>4s} {'cells':>5s} {'Q gen':>7s} | {'cum n':>5s} {'Q':>7s} {'rank r':>7s} {'slope':>6s} {'sign':>5s} {'cover':>5s} {'raw slope (sd)':>15s} {'accepted':>8s}"]
    s = lambda v, w, k=2: (f"{v:{w}.{k}f}" if isinstance(v, (int, float)) else f"{'-':>{w}s}")   # noqa: E731
    for g in c.get("by_gen") or []:
        L.append(f"{g['gen']:4d} {g['n']:5d} {s(g['Q_gen'], 7)} | {g['n_cum']:5d} {s(g['Q'], 7)} {s(g['spearman'], 7)} {s(g['slope_written'], 6)} {s(g['sign'], 5)} "
                 f"{s(g['coverage'], 5)} {s(g['slope_raw'], 8)} ({s(g['slope_raw_sd'], 4)}) {g['accepted_cum']:8d}")
    for k, v in (c.get("slopes") or {}).items():
        L.append(f"slope {k}: b {s(v.get('b'), 0, 3)} (sd {s(v.get('sd'), 0, 3)}, n {v.get('n')}, least squares {s(v.get('ls'), 0, 3)}, P(b > 0) {s(v.get('p_pos'), 0, 3)})")
    return "\n".join(L) + "\n"


def quality_line(sig_dir):
    """One line for the generation report."""
    c = load_calibration(sig_dir)
    if not c or not c.get("pairs"):
        return "signal quality: no realised pair yet"
    g = (c.get("by_gen") or [{}])[-1]
    return (f"signal quality Q {c['quality']} over {c['pairs']} judged G cells (rank r {g.get('spearman')}, slope {g.get('slope_written')}, "
            f"coverage {g.get('coverage')}; raw slope {g.get('slope_raw')} sd {g.get('slope_raw_sd')})")


# ---------------------------------------------------------------------------------------------------------------- command line
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make", help="write RUN/signal/<tip>/cells.json from measured reads")
    m.add_argument("--out", help="directory (default RUN/signal/<tip>)")
    m.add_argument("--dry", action="store_true", help="print the table, write nothing")
    c = sub.add_parser("calibrate", help="fit the shrinkage and write the signal-quality number")
    c.add_argument("--run", required=True)
    w = sub.add_parser("watch", help="rerun on every new per-read table of a directory and append the table to a file (dry: writes no cells.json)")
    w.add_argument("--dir", required=True, help="directory the per-read tables (*.tsv) land in")
    w.add_argument("--table", help="file the tables are appended to (default DIR/cells_table.md)")
    w.add_argument("--every", type=float, default=60.0, help="seconds between polls")
    w.add_argument("--hours", type=float, default=36.0, help="lifetime of the watch")
    w.add_argument("--once", action="store_true", help="one run on what is there now, then stop")
    for x in (m, w):                                         # the same inputs for make and watch
        x.add_argument("--run", required=True)
        x.add_argument("--tip", required=True)
        x.add_argument("--rows", required=True, help="learn.jsonl (roles, benchmarks, picture clusters)")
        x.add_argument("--traces", help="the tip's traces.json (default RUN/signal/<tip>/traces.json)")
        x.add_argument("--tip-file", help="the tip's json (entries: router classes from the prefixes in traces.json)")
        x.add_argument("--routes", help="{condition: [rows] | {row: bool}} (router classes given directly)")
        x.add_argument("--plan", help="planner directory (state.json + reads.jsonl; default RUN/signal/<tip>/plan when it exists)")
        x.add_argument("--map", help="mapping json for engine result files (see ingest_results)")
        x.add_argument("results", nargs="*", help="engine result files (with --map)")
        x.add_argument("--tsv", nargs="*", default=[], help="per-read tables (see ingest_tsv)")
        x.add_argument("--batch-cells", help="batch_cells.json of the per-read tables (batch -> cell, edit, family; prior_family)")
        x.add_argument("--proposable", help="comma list of the families that may be proposed (others are measurement only)")
        x.add_argument("--control", help="name of the concurrent control cell (e.g. W-null)")
        x.add_argument("--budget", type=float, help="allowed token multiple M (default: RUN/cards.json g_token_multiple; no built-in value)")
        x.add_argument("--budget-source")
        x.add_argument("--budget-base", type=float, help="tokens per read the multiple refers to (default: cards.json g_token_base, else the tip's)")
        x.add_argument("--ref", help="json list of reference points [{name, gain, se, tokens_ratio, rows, source}] printed next to the frontier")
        x.add_argument("--len-classes", type=int, default=2, help="bins of the tip's read length (quantiles of the data)")
        x.add_argument("--ecomp", default="round", choices=("round", "ceil", "off"))
    sc = sub.add_parser("scan", help="what engine result files hold (batches, roles, samples, edit tags): for writing the mapping")
    sc.add_argument("--rows", required=True)
    sc.add_argument("results", nargs="+")
    a = ap.parse_args(argv)
    if a.cmd == "scan":
        for x in scan_results(a.results, Rows.from_learn(a.rows)):
            print(f"{x['batch']}: {x['lines']}; samples {x['samples']}")
            for t in x["tags"]:
                print("    " + json.dumps(t))
        return 0
    sig = os.path.join(a.run, "signal")
    if a.cmd == "calibrate":
        out = calibrate(sig)
        sys.stdout.write(quality_text(out))
        return 0
    if a.cmd == "watch":
        return watch(a)
    doc, notes = build(a)
    for n_ in notes:
        print(n_)
    if a.dry:
        sys.stdout.write(table(doc))
        return 0
    p, sha = write_cells(a.out or os.path.join(sig, a.tip), doc)
    sys.stdout.write(table(doc, sha))
    print(f"written: {p} (cells_sha {sha}; {len(doc['cells'])} cells sent, {len(doc['held'])} held, {len(doc['not_selected'])} not selected)")
    return 0


def build(a):
    """The document of one `make` call from its arguments (traces / plan / result files + mapping / per-read tables) -> (doc, notes)."""
    sig = os.path.join(a.run, "signal")
    R = Rows.from_learn(a.rows)
    tp = a.traces or os.path.join(sig, a.tip, "traces.json")
    traces = json.load(open(tp)) if os.path.exists(tp) else {}
    tip = tip_table(traces, R)
    cells, reads, notes = {}, {}, []
    plan = a.plan or (os.path.join(sig, a.tip, "plan") if os.path.isdir(os.path.join(sig, a.tip, "plan")) else None)
    if plan:
        st = json.load(open(os.path.join(plan, "state.json")))
        cells.update({cid: x["spec"] for cid, x in st["cells"].items()})
        with open(os.path.join(plan, "reads.jsonl")) as fh:
            reads.update(reads_table((json.loads(l) for l in fh if l.strip()), R, "plan reads"))
    if a.map:
        c2, r2, t2, info = ingest_results(a.results, json.load(open(a.map)), R)
        cells.update(c2)
        for k, v in r2.items():
            reads.setdefault(k, {}).update(v)
        for r, sm in t2.items():
            tip.setdefault(r, {}).update(sm)
        notes.append("ingest: " + json.dumps(info, sort_keys=True))
    if a.tsv:
        bc = json.load(open(a.batch_cells)) if a.batch_cells else {}
        rules = dict(control_cell=a.control, proposable_families=a.proposable.split(",") if a.proposable else None)
        c3, r3, base, info = ingest_tsv(a.tsv, R, bc, rules)
        cells.update(c3)
        for k, v in r3.items():
            reads.setdefault(k, {}).update(v)
        notes.append("ingest: " + json.dumps(info, sort_keys=True) + "; base reads that differ from the tip table: "
                     + (base_check(base, tip) if tip else "no tip table (the base columns are the tip)"))
        if not tip:
            tip = base
    if not tip:
        raise SystemExit(f"no tip reads: {tp} is missing and the mapping names no tip lines")
    classes = routes_classes(json.load(open(a.routes))) if a.routes else []
    if a.tip_file and traces:
        tf = json.load(open(os.path.expanduser(a.tip_file)))
        classes += router_classes(traces, tf.get("entries", tf) if isinstance(tf, dict) else tf)
    cards = {}
    try:
        cards = json.load(open(os.path.join(a.run, "cards.json")))
    except (OSError, ValueError):
        pass
    budget = read_budget(a.run, a.budget, a.budget_source, a.budget_base)
    calib = calibrate(sig, write=not a.dry)
    doc = make(a.tip, tip, cells, reads, R, budget, classes=classes, n_len=a.len_classes, calib=calib, rule=a.ecomp,
               refs=json.load(open(a.ref)) if a.ref else None, caps={k: cards.get(k) for k in ("g_cells", "g_settings")}, control=a.control)
    return doc, notes


def cst(t=None):
    return time.strftime("%m-%d %H:%M", time.gmtime((time.time() if t is None else t) + 8 * 3600)) + " CST"


def watch(a):
    """Rerun the cell maker (dry: nothing is written to the run) whenever the per-read tables of a.dir change, and append the table to
    a.table. A table set is taken once it is the same on two polls in a row (a file being copied is not read half-way). batch_cells.json
    is read again on every run. Stops after a.hours. An error is appended as text and the watch goes on. The last table set is
    kept in <table>.state."""
    seen = last = None
    end = time.time() + 3600.0 * a.hours
    out = a.table or os.path.join(a.dir, "cells_table.md")
    try:                                                         # a restarted watch does not write the same table again
        last = [tuple(x) for x in json.load(open(out + ".state"))]
    except (OSError, ValueError):
        pass
    while time.time() < end:
        fs = sorted(glob.glob(os.path.join(a.dir, "*.tsv")))
        extra = [f for f in (a.batch_cells, a.ref, a.traces) if f and os.path.exists(f)]
        state = [(f, os.path.getsize(f), int(os.path.getmtime(f))) for f in fs + extra]
        if fs and state == seen and state != last:
            new = sorted(set(os.path.basename(f) for f in fs) - set(os.path.basename(x[0]) for x in (last or [])))
            a.tsv, a.dry = fs, True
            try:
                doc, notes = build(a)
                text = "\n".join(notes) + "\n" + table(doc)
            except BaseException as ex:                          # noqa: BLE001 (a bad table must not stop the watch; a judge row is said so)
                if isinstance(ex, KeyboardInterrupt):
                    raise
                text = f"NOT RUN: {type(ex).__name__}: {ex}\n"
            with open(out, "a") as fh:
                fh.write(f"\n## {cst()}: {len(fs)} tables" + (f"; new: {', '.join(new)}" if new else "; inputs changed") + "\n\n```\n" + text + "```\n")
            print(f"{cst()}: {len(fs)} tables -> {out}", flush=True)
            last = state
            with open(out + ".state", "w") as fh:
                json.dump(state, fh)
        seen = state
        if a.once:
            if last == state:
                break
            seen = state
        time.sleep(0 if a.once else a.every)
    return 0


if __name__ == "__main__":
    sys.exit(main())
