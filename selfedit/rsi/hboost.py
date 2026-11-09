"""Paper B's HarnessBoost with one addition: the proposer may also edit the weights and structure of a copy of the executor.

Executor and proposer are both Qwen3.8-27B. The proposer and meta-proposer are the frozen stock checkpoint; only other copies are edited.
A member = a genome (vision settings + engine edits, vharness operators) + a harness program (sandboxed function: crops, views, prompt text; may abstain).
Thesis under test: harnesses on one base share errors, so a pool has little to fit; weight edits make members differ, the residual becomes fittable, fusion gains.
Main numbers, every round: error correlation among members and the pool's gain over its best single member.

Ported unchanged from Paper B (pb/vrsi27/boost): residual, lenses and sandbox, Thompson allocation, meta layer, evidence combiner and tests, leak guard, partial pooling.
New here: Table (answer votes -> option distributions), member programs, parents drawn from accepted members with crossover, SELECT lower bound in acceptance, signed z as the lens reward.

Distribution of a member on a row: votes over 2 samples, (count + .5) / (2 + .5 * options). MC rows: the letters. Yes/no: two options. Open rows: distinct answers of all members,
in an order that does not depend on the gold, plus a last column OTHER that every member holds at the floor; y = the gold's column if a member voted it, else OTHER, and predicting is over voted columns only.

  hboost.py plan WORK [--slots 16] [--stage2 3] [--rounds 20] [--limit N]
  hboost.py run WORK        CPU driver, LEARN rows only (IID half 0)
  hboost.py report WORK     separate process: scores members on ID (IID half 1) and OOD, writes report.txt; nothing it reads or writes reaches run
  hboost.py job-gen REQ OUT [loop.py args]   GPU job: the frozen stock checkpoint answers the round's prompts
"""
import collections, contextlib, dataclasses, fcntl, glob, hashlib, json, os, re, shlex, subprocess, sys, threading, time, zlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
sys.path[:0] = [HERE, os.path.join(CODE, "pb"), os.path.join(CODE, "mm")]
import vharness as vh

D, W, V = vh.D, vh.W, f"{vh.W}/vh3"
HB = f"{W}/hb"
MODELS = f"{D}/models/vh"
CACHE = f"{V}/cache"
OWN_STOCK = False
EVR = HB        # where a run's answers, programs and gate writes live; a plain run keeps its own, see set_run
IID, OOD = f"{V}/iid.jsonl", f"{CACHE}/rows/ood.jsonl"
C_MAX = 10
NARROW_MAX = 3               # narrow acceptances per round under prereg narrow_pool_rows
PLATEAU_SPLIT = dict(program=1, gate=3, cross=3, grid=5, from_round=1)      # slots of the round after a plateau, prereg plateau
PLATEAU_OPS = ("direction", "scale", "attn_temp", "skip", "vision_scale", "patch")       # the grid slots of such a round; tokens and verbosity alone are not weight edits
PLATEAU_MIN_RAN = 3          # of the 5 grid slots: fewer built and ran rows, and the round does not count toward the stop rule
PLATEAU_STOP = 3             # such rounds in a row with no accepted member stop the run
COMBO_MAX, COMBO_NET = 3, 2  # plateau combination test: members added at most, net SELECT rows at least
STAGE2_EXEMPT_MAX = 3        # coordinator 11-07: extra stage 2 slots of a plateau round, weight and vision ops the zx filter would stop
WRITE_RUN_MAX, WRITE_QUEUE_MAX = 1500, 3600       # a gate write: 1500 s from the moment a card starts it, not from its submit (paper1 r7: two valid writes waited 15 min in the queue, landed 3 min past a submit-time deadline, lost); a write no card has started gives up after an hour in the queue
EXEMPT_ANSWERED, EXEMPT_NOT_STOPPED = .8, .1      # the sanity floor of those slots: answered share of its stage 1 rows; too_long rate and not-stopped rate
DS_MODES = ("ds_agree", "ds_leaf")      # 128c: ds_leaf is ds_agree for the full members plus the specialists' two-parameter leaves; every ds_agree rule reads it the same way
GAIN_TOL = 1e-9              # |lb| and |ce| of a pool increment below this count as zero, on stacking and on ds_agree alike. Numerical, not a change of the gate
ASK_RETRY = 4 * 3600         # a row asked this long ago with no answer goes out again: its line failed or was lost
A = .5
LENSES = ("target_slice", "energy_slices", "residual_table", "conf_cards", "accuracy_style", "peer_gap_cases")
sha, canon = vh.sha, vh.canon
Bad = vh.Bad
# 128c (Fable 11-09 §4.1, §4.3, §4.4, §5 steps 1-2): every rule below is off unless prereg "c128" turns it on (Run.c128). Off, the code paths are A4's.
DRAW_OFF = 7919              # draw d of a member answers with --seed (tail seed) + d * DRAW_OFF: a fresh engine seed, same genome, program and rows
DRAW_RE = re.compile(r"/ev/d(\d+)_[^/]+$")
BAND_DRAWS = (1, 2, 3)       # the parent's noise band: three fresh-seed re-answers, each read paired against the parent's own two samples
NULL_DRAW, NULL_DRAW2 = 4, 5  # the null competitor: draw 4 (2 samples, stage 1) and draws 4+5 (4 samples, stage 2); independent of the band, which it could never beat (3 points: max z 1.15 < 1.65)
PAIR_DRAW = 3                # samples 2,3 of the 4-sample stage 2: the parent's draw 3 and the candidate's own draw 3 (same seeds, so paired)
DRAW_EPOCH = 10              # a null that passes voids the round and moves its parent's draws to the next block of DRAW_EPOCH
C128_FIELDS = dict(          # prereg "c128": every field and its default; absent block = A4. Steps 1 (judge) and 2 (scope, fusion) of Fable 11-09 §5
    from_round=1, judge=False, mode="replace",       # judge: 128c stage 1 + stage 2 decide; mode "and": the old verdict AND the new one (old vs new replay)
    null=True, band_k=1.65, band_center="mean",      # null competitor per parent per round (draws 4, 4+5); band = band_k x SD over draws 1-3; "mean": thr = max(0, band mean) + delta, "zero": thr = delta
    fired_min=8, stat="pool", s2_samples=4, z_min=1.96, stage2_max=None,      # stat "pool": per fired row 1[m right] - 1[pool right] (§4.3); "parent": paired right rate vs the parent's own samples
    joint=True, card_unseen=True, critic=False,      # joint read for |F| < fired_min; card gain > 0 with unseen gain <= 0 rejects; leakage critic before engine time
    refresh=None, rotate_seed=1109, read_cap=3, confirm=True,      # refresh "rotate" (option a) or "cap" (option b: read cap + two-round confirmation of wide members); None: SELECT fixed, no cap
    scope=False, leaf=False, leaf_lambda=1.0, diversity=False, div_share=.2, div_none=1)      # step 2: program mandatory for E, abstain outside scope; ds_leaf fusion; diversity floor
SCOPE_ON = [False]           # 128c scope rule: weight members (engine edits, gate writes) abstain outside their scope, no inherited answers


def jload(f, default=None):
    return json.load(open(f)) if os.path.exists(f) else default


def private_tmp(f, tail=""):
    """A temp name no other process or thread shares: the driver and the reporters write the same cache files (views, notes, rows)."""
    return f"{f}.{os.getpid()}.{threading.get_ident()}.new{tail}"


def settle(tmp, f):
    try:
        os.replace(tmp, f)
    except FileNotFoundError:
        if not os.path.exists(f):
            raise


def jdump(f, o):
    os.makedirs(os.path.dirname(f), exist_ok=True)
    t = private_tmp(f)
    with open(t, "w") as out:
        json.dump(o, out, ensure_ascii=False, default=lambda x: x.item() if hasattr(x, "item") else str(x))
    settle(t, f)


# ---------------------------------------------------------------- data ------------------------------------------------------------------------
class Data:
    def __init__(self, limit=None, iid=None, ood=None, exclude=None):
        self.iid = vh.read_jsonl([iid or IID])
        if exclude:                  # rows dropped by a rule on the image alone; IID only, OOD stays frozen
            drop = set(jload(exclude)["rows"])
            self.iid = {i: p for i, p in self.iid.items() if i not in drop}
        self.ood = vh.read_jsonl([ood or OOD])
        self.rows = {**self.iid, **self.ood}
        ROWKEY.update({i: vh.key(p) for i, p in self.rows.items()})
        POOL.update(self.rows); POOL_GEN[0] += 1          # coordinator 11-08: every stored record is graded from its pool row (gold, kind, options, category, precision), never from fields it kept at answer time
        base = lambda i: i[:-5] if i.startswith("3dsrbench/") and i.endswith("-flip") else i       # a flipped copy stays with its original
        tag = lambda i: self.iid[i].get("hb_role")        # 128 pool: the row states its role, so variants of one picture (different ids) stay on one side
        LR = ("show", "score", "select")
        odd = sorted({str(tag(i)) for i in self.iid if tag(i) and tag(i) not in LR + ("id",)})
        if odd:
            raise ValueError(f"hb_role must be one of show, score, select, id: {odd}")
        h0 = [i for i in self.iid if tag(i) in LR or (not tag(i) and vh.half(base(i)) == 0)]      # a row with no hb_role keeps the id half
        if limit:
            h0 = sorted(h0, key=lambda i: zlib.crc32(("hbl:" + i).encode()))[:limit]
        self.role = {}
        for i in h0:
            s = zlib.crc32(("hb:" + self.cluster(i)).encode()) % 8        # by image: rows sharing an image stay on one side
            self.role[i] = self.iid[i].get("hb_role") or ("show" if s < 3 else "score" if s < 6 else "select")      # v2.1 states each row's role: one picture, one role
        self.id_rows = [i for i in self.iid if tag(i) == "id" or (not tag(i) and vh.half(base(i)) == 1)]
        self.ood_rows = list(self.ood)
        self.learn = sorted(self.role)

    def cluster(self, i):
        p = self.rows[i]
        if p.get("pic"):             # v2.1: same image sha or same picture id stem
            return p["pic"]
        if i.startswith("3dsrbench/") and i.endswith("-flip") and i[:-5] in self.rows:       # same role and same variance cluster as its original
            p = self.rows[i[:-5]]
        return (p.get("img_sha256") or [i])[0]

    def of(self, role):
        return [i for i in self.learn if self.role[i] == role]

    def bench_of(self, i):
        return self.rows[i].get("bench") or i.split("/")[0]

    def rotate(self, r, seed, n_select=None):
        """128c split refresh (a), §4.4.4: per bench, the non-show LEARN rows (score + select, 88 on the 128 pool) are dealt again between score and select by a preregistered
        seed schedule, a different SELECT each round. Whole pictures move together; show and ID never move. n_select: SELECT rows per bench, default the bench's preregistered count.
        Returns {bench: (select rows, score rows)}."""
        if not hasattr(self, "role0"):
            self.role0 = dict(self.role)
        out = {}
        for b in sorted({self.bench_of(i) for i in self.learn}):
            rows = [i for i in self.learn if self.bench_of(i) == b and self.role0[i] in ("score", "select")]
            want = n_select if n_select is not None else sum(self.role0[i] == "select" for i in rows)
            pics = collections.defaultdict(list)
            for i in rows:
                pics[str(self.cluster(i))].append(i)
            order = sorted(pics, key=lambda c: (zlib.crc32(f"rot|{seed}|{r}|{b}|{c}".encode()), c))
            sel = []
            for c in order:                      # whole pictures, first fit; a picture that would overshoot waits for a smaller one
                if len(sel) + len(pics[c]) <= want:
                    sel += pics[c]
            sel_ = set(sel)
            for i in rows:
                self.role[i] = "select" if i in sel_ else "score"
            out[b] = (len(sel), len(rows) - len(sel))
        assert all(self.role[i] == self.role0[i] for i in self.learn if self.role0[i] == "show"), "rotation moved a show row"
        return out


# ---------------------------------------------------------------- votes ------------------------------------------------------------------------
_vcache = {}


_ORD = {w: str(k + 1) for k, w in enumerate("first second third fourth fifth sixth seventh eighth ninth tenth".split())}


def cell_of(s):
    """[row, column] of a grid position, read by its labels: 'Second row third column', 'Row 1, Column 3', 'Second column, first one' (column 2, row 1), 'R4C17'.
    One labelled axis takes the other from the one unlabelled value; no label at all reads the first two values as row, column ('(2,3)'). Words 'one' etc. are not values ('first one').
    Anything else gives fewer than two values, which never matches (11-08: the older rule read the axes in written order)."""
    t = str(s).lower()
    m = re.search(r"\br\s*(\d+)\s*,?\s*c\s*(\d+)\b", t)
    if m:
        return [m.group(1), m.group(2)]
    T = re.findall(r"\d+(?:st|nd|rd|th)\b|\d+|[a-z]+", t)
    val = lambda w: _ORD.get(w) or (re.match(r"\d+", w).group(0) if re.fullmatch(r"\d+(?:st|nd|rd|th)?", w) else None)
    ordinal = lambda w: w in _ORD or bool(re.fullmatch(r"\d+(?:st|nd|rd|th)", w))
    axis = lambda w: "row" if w in ("row", "rows") else "col" if w in ("column", "columns", "col", "cols") else None
    got, used = {}, set()
    for k, w in enumerate(T):
        a = axis(w)
        if a is None or a in got:
            continue
        if k + 1 < len(T) and val(T[k + 1]) and not ordinal(T[k + 1]) and k + 1 not in used:       # 'row 4'
            got[a] = val(T[k + 1]); used.add(k + 1)
        elif k > 0 and val(T[k - 1]) and ordinal(T[k - 1]) and k - 1 not in used:                 # 'second row', '2nd row'
            got[a] = val(T[k - 1]); used.add(k - 1)
        elif k + 1 < len(T) and val(T[k + 1]) and k + 1 not in used:                               # 'row second'
            got[a] = val(T[k + 1]); used.add(k + 1)
    free = [val(w) for k, w in enumerate(T) if k not in used and val(w)]
    if not got:
        return free[:2]
    if len(got) == 1 and len(free) == 1:
        got["col" if "row" in got else "row"] = free[0]
    return [got["row"], got["col"]] if len(got) == 2 else []


def cells_of(s):
    """The cells a gold or an answer names: several when every part between ';', 'and' or ', the' is a whole cell ('The first one in the first row, the third one in the first row', '1,1;1,3'), else the one cell_of reads; [] if none."""
    segs = [x for x in re.split(r";|\band\b|,\s*(?=the\b)", str(s), flags=re.I) if x.strip()]
    if len(segs) > 1:
        cs = [cell_of(x) for x in segs]
        if all(len(x) == 2 for x in cs):
            return cs
    c = cell_of(s)
    return [c] if len(c) == 2 else []


GATE_CATS = 80              # categories the proposer may name as a gate scope: every show category of the pool. Was 40, never binding on paper1 (23); the 128 pool has 69
GRID_GOLD_BENCHES = ("babyvision/",)      # the grid-position rule reads golds of these benches only: a CharXiv text gold may say "row" or "column" (11-08). On iid_paper + ood_learn the rule's golds are 2 babyvision rows, so nothing else moves


_grade_err = set()
NO_TEXT = {}         # bench -> {(row, sample)} graded with no text: read by the report as a count per bench


def grade(r):
    """grade_mm, except a gold written as a grid position in words is compared as a position. Rule on the gold text alone, fixed before any candidate: gold mentions row or column."""
    import grade_mm
    if r.get("finish") != "stop" or not r.get("content"):      # no text (too long, failed, cut, empty): wrong, no answer, before any parsing; counted once per row and sample by bench
        NO_TEXT.setdefault(str(r.get("id", "")).split("/")[0], set()).add((r.get("id"), r.get("sample")))
        return False, None
    try:                    # a grade never stops a read: a record grade_mm cannot read counts wrong (11-08)
        if r.get("kind") == "mc" or not str(r.get("id", "")).startswith(GRID_GOLD_BENCHES) or not re.search(r"\b(row|column)\b", str(r.get("answer")), re.I):
            return grade_mm.correct(r)
        if r.get("finish") != "stop" or not r.get("content"):
            return False, None
        a = grade_mm.final_answer(r["content"])
        if a is None:
            return False, None
        g, c = cells_of(r["answer"]), cells_of(a)
    except Exception as e:
        k = (r.get("id"), r.get("sample"), str(r.get("content"))[-200:])
        if k not in _grade_err:
            _grade_err.add(k)
            print(f"grade error on {r.get('id')} sample {r.get('sample')}: {type(e).__name__}: {str(e)[:160]}; answer {str(r.get('answer'))[:80]!r}; text {k[2][-160:]!r}; counted wrong", flush=True)
        return False, None
    return bool(g) and sorted(c) == sorted(g), ("; ".join(f"row {x[0]}, column {x[1]}" for x in sorted(c)) if c else a)     # one key per cell set, so 'column 2, row 1' and 'row 1, column 2' vote together


GEN_FIELDS = ("budget", "effort", "max_model_len", "kv_cache_dtype", "max_num_seqs", "max_num_batched_tokens", "crn", "seed", "mb", "tp", "prefix_cache")      # the answer regime: the generation settings of a job tail that change what a sample is
GEN_TAIL = {}        # prereg "gen" (128, 11-08): flag -> value set on every job tail, e.g. {"budget": 32768, "max-model-len": 46000, "effort": "medium"}; empty: vharness's tail as it is
GEN_KEY = {}         # the regime this run reads: GEN_FIELDS of its tail. Set only with a prereg "gen"; then an answer file is read only if its .gen stamp (written by the card before the file lands) equals it, and every answer file name carries it
GATE_TRACE_CAP = None    # prereg gate_trace_cap (128, 11-08): mm_gate --trace-cap; None: mm_gate's own 3000
REFUSED = {}         # answer file -> its stamp (None: no stamp), for files a read skipped because their regime is not the run's
_gcache = {}


def gen_tail(t):
    for k, v in GEN_TAIL.items():
        f = "--" + str(k).replace("_", "-")
        t = re.sub(rf"{re.escape(f)} \S+", f"{f} {v}", t) if f"{f} " in t else f"{t} {f} {v}"
    return t


def tail_gen(t):
    """GEN_FIELDS of a tail string, values as strings."""
    w, out = t.split(), {}
    for j, x in enumerate(w[:-1]):
        k = x[2:].replace("-", "_")
        if x.startswith("--") and k in GEN_FIELDS:
            out[k] = str(w[j + 1])
    return out


def stamp_of(a):
    """The regime a card answered in: GEN_FIELDS of its parsed arguments, effort as it was set for the item."""
    return {k: str(getattr(a, k)) for k in GEN_FIELDS if hasattr(a, k)}


def gsfx():
    """A file-name part for the run's regime: '' without a prereg gen (names as before)."""
    return ("." + sha(canon(GEN_KEY))[:8]) if GEN_KEY else ""


def gen_ok(f):
    """True if answer file f may be read: no regime set, or its stamp equals the run's. A file with no stamp (any answer made before 11-08, e.g. paper128's 8192 stock) is refused under a regime."""
    if not GEN_KEY:
        return True
    try:
        mt = os.stat(f + ".gen").st_mtime
    except OSError:
        mt = None
    want = draw_key(f)
    k = (f, mt, canon(want))
    if k not in _gcache:
        st = jload(f + ".gen") if mt is not None else None
        _gcache[k] = st is not None and all(str(st.get(x)) == v for x, v in want.items())
        if _gcache[k]:
            REFUSED.pop(f, None)
        else:
            REFUSED[f] = st
    return _gcache[k]


def draw_key(f):
    """The regime an answer file must carry: GEN_KEY, with a draw dir's own seed (128c)."""
    m = DRAW_RE.search(os.path.dirname(f))
    if not m or "seed" not in GEN_KEY:
        return GEN_KEY
    return dict(GEN_KEY, seed=str(int(GEN_KEY["seed"]) + int(m.group(1)) * DRAW_OFF))


def draw_of(m, d):
    """Member m re-answered with a fresh engine seed (128c): draw d >= 1, d 0 is m. Same genome, program and gate; its own answers in ev/d{d}_{key}, never mixed with m's.
    A draw of a gate member is cut to its scope like the member (load_member); under SCOPE_ON it never inherits."""
    if not d:
        return m
    key = m["mid"] if m.get("program") or m.get("gate") else vh.gid(m["genome"])
    return dict(m, name=f"{m['name']}~d{d}", mid=sha(f"{m['mid']}:draw:{d}")[:14], draw=dict(of=m["name"], d=int(d), key=key), origin="draw")


def answer_files(d):
    return [f for f in sorted(glob.glob(f"{d}/a.*.jsonl")) if gen_ok(f)]


BENCH_COST = {}      # bench -> expected generated tokens relative to the mean (prereg bench_cost, from stock's measured tokens); empty: plain sha order as before


def votes_of(path):
    """id -> sample -> dict(key, ok, tail). Samples 0 and 1 only: every file was made with the same per-problem seeds."""
    if not gen_ok(path):
        return {}                            # another answer regime (budget, effort, context length, tail): never read, its rows are asked again under the run's
    st = os.stat(path)
    k = (path, st.st_mtime, POOL_GEN[0])
    if k in _vcache:
        return _vcache[k]
    out = collections.defaultdict(dict)
    for l in open(path):
        r = json.loads(l)
        if r.get("sample", 0) > 1:
            continue
        row = POOL.get(r.get("id"))
        if row is None:
            if not POOL:
                raise RuntimeError(f"votes_of({path}) before any Data was built: no pool row to grade against, and stored golds are never used")
            continue                         # not a row of this pool (a shared answer dir also holds other pools' rows): never read
        r, ok, p = regrade(r, row)
        key = None
        if r.get("finish") == "stop" and p is not None:
            key = re.sub(r"\s+", " ", str(p)).strip().lower() if r["kind"] != "mc" else str(p)
        out[r["id"]][r.get("sample", 0)] = dict(key=key, ok=bool(ok), tail=(r.get("reasoning") or "")[-300:], said=(r.get("content") or "")[-100:], tok=r.get("tokens", 0), stop=r.get("finish") == "stop")
    _vcache[k] = dict(out)
    return _vcache[k]


def guard_groups():
    """row_groups() reads 'mds_<ds>:00012'. Our ids are bench/id, which give one group '' for every row. The driver's fit uses no groups; any path that does stops here instead of fitting on one collapsed group."""
    import vrsi27.boost.edge as E
    if getattr(E, "_guarded", False):
        return
    plain = E.row_groups
    def row_groups(qids):
        g = plain(qids)
        benches = {str(q).split("/")[0] for q in qids if "/" in str(q)}
        if len(benches) > 1 and len(set(g.tolist())) == 1:
            raise ValueError(f"row_groups collapsed {len(benches)} benches into one group: this path needs real groups")
        return g
    E.row_groups, E._guarded = row_groups, True


def hardcoded_values(text):
    """Flag only, for the round review: wording that states a fixed answer value (r1s10: 'standard chess=32, Audi=4 rings'). Nothing is rejected on it."""
    pats = (r"\b(?:standard|always|usually|typically|normally|by default)\b[^.\n]{0,60}\d+", r"\b[A-Za-z][A-Za-z ]{1,20}\s*=\s*\d+\b(?=[^\n]*[a-z])",
            r"\b\d+\s+(?:rings|legs|columns|squares|pieces|sides|wheels|players)\b")
    strs = []
    try:
        import ast
        strs = [n.value for n in ast.walk(ast.parse(text)) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    except (SyntaxError, ValueError):
        strs = [text]
    out = []
    for t in strs:
        for p in pats:
            out += [x.group(0)[:80] for x in re.finditer(p, t, re.I)]
    return sorted(set(out))


def set_run(work):
    """A plain run reads and writes only under its own directory: the one exception is stock's own answers, read from the shared stores (member_dirs)."""
    global EVR, OWN_STOCK, INHERIT, WRITE_RUN_MAX, WRITE_QUEUE_MAX, GATE_TRACE_CAP
    guard_groups()
    p = jload(f"{work}/prereg.json") or {}
    INHERIT = bool(p.get("inherit_gate"))
    if p.get("plain"):
        EVR = work
    OWN_STOCK = bool(p.get("own_stock"))     # stock too is answered here, on this machine, and read from nowhere else
    if "write_run_max" in p:          # 128 prereg: the write clock scaled to measured write times at the run's budget; absent, the constants stand (and a replay's scaled ones)
        WRITE_RUN_MAX = int(p["write_run_max"])
    if "write_queue_max" in p:
        WRITE_QUEUE_MAX = int(p["write_queue_max"])
    GATE_TRACE_CAP = int(p["gate_trace_cap"]) if p.get("gate_trace_cap") else None
    GEN_TAIL.clear()
    GEN_KEY.clear()
    if p.get("gen"):
        GEN_TAIL.update(p["gen"])
        GEN_KEY.update(tail_gen(gen_tail(vh.TAIL.format(lib=vh.DIRLIB))))
        for k in GEN_TAIL:
            assert str(k).replace("-", "_") in GEN_FIELDS or str(k).replace("-", "_") == "gpu_util", f"prereg gen: {k} is not a tail setting of the answer regime"
    BENCH_COST.clear()
    BENCH_COST.update(p.get("bench_cost") or {})     # 128 prereg: part order by bench, scheduling only


def is_stock(g):
    return not g.get("vision") and not g.get("engine") and not g.get("dirs")


def member_dirs(m):
    if m.get("draw"):
        return [f"{EVR}/ev/d{m['draw']['d']}_{m['draw']['key']}"]
    g = vh.gid(m["genome"])
    if m.get("program"):
        return [f"{EVR}/ev/{m['mid']}"]
    if EVR != HB and (OWN_STOCK or not is_stock(m["genome"])):
        return [f"{EVR}/ev/{g}"]
    return [f"{EVR}/ev/{g}"] + ([f"{HB}/ev/{g}"] if EVR != HB else []) + [f"{CACHE}/ev/{g}", f"{CACHE}/ood/{g}"]


INHERIT, ROWKEY, REG, LEAK_N, LEAK_SAME = False, {}, {}, 20, .70     # coordinator 11-07: a gate's rows outside its scope take the parent's answers
POOL = {}            # row id -> row of the loaded pool (Data): the only source of grading inputs at read time
POOL_GEN = [0]       # bumped by each Data: votes_of's cache never serves a read graded against an earlier pool
GRADE_FIELDS = ("answer", "kind", "options", "category", "precision", "bench")
TRACE_FIELDS = ("id", "sample", "content", "reasoning", "finish", "tokens", "loop")


def regrade(rec, row):
    """The record as graded: grading inputs from the pool row, the model's output from the record. Returns (graded record, ok, parsed answer). Never reads the record's stored gold or ok."""
    g = {k: rec[k] for k in TRACE_FIELDS if k in rec}
    g.update({k: row[k] for k in GRADE_FIELDS if k in row})
    g.setdefault("options", [])
    ok, p = grade(g)
    return g, bool(ok), p


def strip_gold(note, golds):
    """gate_v3 control: the note with its own category's gold answers removed: 'answer is/: X' clauses and every whole-word occurrence of a gold value
    of 2+ characters (single letters / digits only inside an 'answer' clause, so ordinary words survive)."""
    t = re.sub(r"(?i)\b(the\s+)?(correct\s+|right\s+)?answer\s*(is|:|=|would be|should be)\s*\(?[^\s.,;)]+\)?", "the answer", note)
    for g in sorted({str(x) for x in golds if x is not None and len(str(x).strip()) >= 2}, key=len, reverse=True):
        t = re.sub(r"(?i)(?<![\w])" + re.escape(g.strip()) + r"(?![\w])", "...", t)
    return re.sub(r"\s+", " ", t).strip()


def control_note(note, cat, pool, golds, fallback, tag):
    """gate_v3 control (coordinator 11-09, prereg 1109c-final): a real proposed gate note of a DIFFERENT category from this run's pool, the closest in
    length to the write's own note (ties by a fixed seed from the write's tag), its own category's gold answers stripped. The fixed fallback sentence
    only when no other category's note exists. Returns (note, source dict)."""
    C = [x for x in pool if x.get("cat") and x["cat"] != cat and x.get("note")]
    if not C:
        return fallback, dict(kind="fallback")
    C = [dict(x, stripped=strip_gold(x["note"], golds.get(x["cat"], ()))) for x in C]
    d0 = min(abs(len(x["stripped"]) - len(note)) for x in C)
    T = sorted([x for x in C if abs(len(x["stripped"]) - len(note)) == d0], key=lambda x: (x["cat"], x["note"]))
    x = T[int(np.random.default_rng(zlib.crc32(f"ctrl:{tag}".encode())).integers(len(T)))]
    return x["stripped"], dict(kind="off-category note", cat=x["cat"], name=x.get("name"), round=x.get("round"), len_own=len(note), len_ctrl=len(x["stripped"]), ties=len(T))


TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def gate_scope(m):
    """The category a gate member's note covers, when its other rows inherit the parent's answers; else None."""
    s = (m.get("gate") or {}).get("scope") or ""
    return s[4:] if (INHERIT or SCOPE_ON[0]) and s.startswith("cat:") and not m.get("inherit_off") and m.get("parents") and m["parents"][0] in REG else None


def has_engine(m):
    """Engine edits E of a member (weight edits, directions, a gate write); vision settings (patch, tokens) are genome, not E."""
    g = m.get("genome") or {}
    return bool(g.get("engine") or g.get("dirs") or m.get("gate"))


def specialist(m):
    """128c scope rule (§4.1): a member with engine edits is a scoped specialist; it votes only inside its scope. Off: nobody is."""
    return bool(SCOPE_ON[0]) and has_engine(m)


def own_rows(m, ids):
    """Rows a gate member itself must run: its scope, and a fixed sample of about 20 outside rows that checks the note does not leak."""
    c = gate_scope(m)
    if c is None:
        return list(ids)
    out = [i for i in ids if ROWKEY.get(i) != c]
    keep = set(sorted(out, key=lambda i: sha(f"leak:{m.get('mid')}:{i}"))[:LEAK_N])
    return [i for i in ids if ROWKEY.get(i) == c or i in keep]


def gate_leaks(m):
    """Same-correctness share against the parent on the outside rows it really ran; numeric drift alone gives about .86 (stock against stock). Below LEAK_SAME the member is scored fully."""
    c = gate_scope(m)
    if c is None:
        return None
    raw, pv = load_raw(m), load_member(REG[m["parents"][0]])
    ks = [(i, s) for i, v in raw.items() if ROWKEY.get(i) != c and i in pv for s in v if s in pv[i]]
    if not ks:
        return None
    return float(np.mean([bool(raw[i][s]["ok"]) == bool(pv[i][s]["ok"]) for i, s in ks])), len(ks)


def load_member(m):
    out = load_raw(m)
    c = gate_scope(m)
    if c is not None and SCOPE_ON[0]:        # 128c: inherit_gate off for weight members; outside its scope a gate member has no record, so it abstains everywhere
        return {i: s for i, s in out.items() if ROWKEY.get(i) == c}
    if c is not None:
        for i, s in load_member(REG[m["parents"][0]]).items():
            if ROWKEY.get(i) != c:
                out[i] = s
    return out


PRIOR_PHRASES = ("similar datasets", "datasets answer", "typically the answer", "the answer is typically", "usually the answer", "the answer is usually", "in datasets like", "such datasets", "datasets like this", "common answer", "most common answer", "benchmark", "this dataset", "these datasets")      # coordinator 11-08: reasoning that leans on a prior about the data, counted per member per round, report only


def per_sample(v, ids):
    """Mean over rows of the share of right samples: a 1:1 row counts half, where the tie-broken read (Table.module) counts it right or wrong by column order. A row with no record or an abstention counts 0, as there."""
    return float(np.mean([np.mean([bool(x["ok"]) for x in v[i].values()]) if v.get(i) else 0. for i in ids])) if len(ids) else float("nan")


def not_stopped(vs, ids):
    """Share of samples that did not stop (too long, cut), over the given members' votes on these rows."""
    xs = [not x.get("stop", True) for v in vs for i in ids for x in (v.get(i) or {}).values()]
    return float(np.mean(xs)) if xs else float("nan")


def oof_acc(fu):
    """Pool accuracy on FIT held out by picture: the fusion's F_oof (ds_agree: 5 folds by picture cluster), read as Table.acc reads F."""
    if fu is None or getattr(fu, "F_oof", None) is None:
        return None
    t, fit = fu.tab, np.asarray(fu.sp.fit, int)
    F = np.where(t.mask[fit], fu.F_oof, -np.inf).copy()
    ot = t.other[fit] >= 0
    F[np.flatnonzero(ot), t.other[fit][ot]] = -np.inf
    pred = np.where(getattr(t, "novote", np.zeros(len(t.ids), bool))[fit] | ~np.isfinite(F.max(1)), -1, F.argmax(1))
    return float((pred == t.y[fit]).mean()) if len(fit) else None


def se_pair(x, groups):
    """(mean, naive SE, SE clustered by group) of row values x. Clustered: sqrt(G/(G-1) * sum over groups of (sum of x - mean)^2) / n, the CR1 sandwich of a mean. Report only."""
    x = np.asarray(x, float)
    n = len(x)
    if n < 2:
        return (float(x.mean()) if n else None), None, None
    mu = float(x.mean())
    naive = float(x.std(ddof=1) / np.sqrt(n))
    by = collections.defaultdict(float)
    for v, g in zip(x, groups):
        by[g] += v - mu
    G = len(by)
    cl = float(np.sqrt(G / (G - 1) * sum(u * u for u in by.values())) / n) if G > 1 else None
    return mu, naive, cl


def right_rows(tab, F, idx):
    """Per row of idx: the fused answer is gold, as Table.acc reads F."""
    F = np.where(tab.mask, F, -np.inf).copy()
    r = np.arange(len(tab.ids))
    ot = tab.other >= 0
    F[r[ot], tab.other[ot]] = -np.inf
    pred = np.where(getattr(tab, "novote", np.zeros(len(tab.ids), bool)) | ~np.isfinite(F.max(1)), -1, F.argmax(1))
    return pred[idx] == tab.y[idx]


def se_table(data, tab, idx, cols):
    """{bench and 'all': {name: dict(acc, se, se_cluster, rows, groups)}} for row-value columns cols {name: bool array over tab rows}, groups by picture (Data.cluster)."""
    idx = np.asarray(idx, int)
    bench = np.array([data.rows[tab.ids[k]].get("bench") or tab.ids[k].split("/")[0] for k in idx])
    grp = np.array([str(data.cluster(tab.ids[k])) for k in idx])
    out = {}
    for b in ["all"] + sorted(set(bench.tolist())):
        sel = np.ones(len(idx), bool) if b == "all" else bench == b
        out[b] = {}
        for nm, col in cols.items():
            mu, se, sc = se_pair(np.asarray(col)[idx][sel], grp[sel])
            out[b][nm] = dict(acc=mu, se=se, se_cluster=sc, rows=int(sel.sum()), groups=int(len(set(grp[sel].tolist()))))
    return out


def sc_k(work, ids):
    """Stock's k-sample self-consistency from the side line sidecheck/stock4 once all its parts landed: plurality over the samples, a tie among n answers counts 1/n if gold is among them. None until then. Report only."""
    d = f"{work}/sidecheck/stock4"
    x = jload(f"{d}/item.json", None)
    if not x or not all(os.path.exists(o) for o in x.get("outs", [])) or not all(gen_ok(o) for o in x.get("outs", [])):
        return None
    c = f"{d}/sc.v3{gsfx()}.json"           # graded from the pool rows (11-08); an older sc.json is not read, nor one of another answer regime
    if not os.path.exists(c):
        per = {}
        for o in x["outs"]:
            for l in open(o):
                r = json.loads(l)
                if r["id"] not in POOL:
                    continue
                r, ok, a = regrade(r, POOL[r["id"]])
                key = None
                if r.get("finish") == "stop" and a is not None:
                    key = re.sub(r"\s+", " ", str(a)).strip().lower() if r["kind"] != "mc" else str(a)
                per.setdefault(r["id"], []).append(("=gold" if ok else key))
        sc = {}
        for i, ks in per.items():
            cnt = collections.Counter(k for k in ks if k is not None)
            if not cnt:
                sc[i] = [0., len(ks)]
                continue
            top = max(cnt.values())
            tied = [k for k, n in cnt.items() if n == top]
            sc[i] = [(1. / len(tied)) if "=gold" in tied else 0., len(ks)]
        jdump(c, sc)
    sc = jload(c)
    got = [i for i in ids if i in sc]
    return dict(acc=float(np.mean([sc[i][0] for i in got])) if got else None, rows=len(got), of=len(ids), samples=int(np.median([sc[i][1] for i in got])) if got else None)


def canonical_rate(v, rows, ids):
    """vlmsbiased: share of samples whose answer equals the row's expected_bias (the canonical answer of the unedited picture). None where no row carries the field."""
    use = [i for i in ids if rows[i].get("expected_bias") is not None and str(i).startswith("vlmsbiased/")]
    import grade_mm
    def same(key, r):               # the grader's own comparison of the parsed answer (key) with expected_bias, in the row's answer form
        try:
            if r.get("kind") == "num":
                return grade_mm.num_ok(grade_mm.to_num(str(key)), r["expected_bias"], r.get("precision"))
            return grade_mm.norm_txt(str(key)) == grade_mm.norm_txt(str(r["expected_bias"]))
        except Exception:
            return False
    xs = [x.get("key") is not None and same(x["key"], rows[i]) for i in use for x in (v.get(i) or {}).values()]
    return (float(np.mean(xs)) if xs else None), len(use)


def prior_hits(m, ids):
    """Reasoning samples of member m on rows ids that use a prior phrase: [(row, sample, phrase, snippet)]."""
    want, out = set(ids), []
    for d in member_dirs(m):
        for f in answer_files(d):
            for l in open(f):
                r = json.loads(l)
                if r["id"] not in want or r.get("sample", 0) > 1:
                    continue
                t = (r.get("reasoning") or "") + "\n" + (r.get("content") or "")
                lo = t.lower()
                for ph in PRIOR_PHRASES:
                    k = lo.find(ph)
                    if k >= 0:
                        out.append((r["id"], r.get("sample", 0), ph, t[max(0, k - 80): k + len(ph) + 80].replace("\n", " ")))
    seen, uniq = set(), []
    for x in out:                    # one record per row, sample and phrase: a row answered in two files counts once
        if x[:3] not in seen:
            seen.add(x[:3])
            uniq.append(x)
    return uniq


def load_raw(m):
    out, sources = {}, {}
    for d in member_dirs(m):
        for f in answer_files(d):
            for i, s in votes_of(f).items():
                out.setdefault(i, {})
                for k, v in s.items():
                    out[i].setdefault(k, v)
    for a in glob.glob(f"{EVR}/prog/{m['mid']}/abstain.*.json"):
        for i in json.load(open(a)):
            out.setdefault(i, {})                                     # evaluated: abstains
    return out


# ---------------------------------------------------------------- cross-sample score ----------------------------------------------------------
def zx_stats(tab, vp, vc, ids, cluster, pool=None):
    """Residual from one parent sample, change from the other parent sample and the child's sample with the same seed, both ways (GRADIENT_CHECK.md, gcheck/s2.py xstat + zc).
    A row the child abstains on counts as no change. dacc: paired accuracy gain over all the rows given. Strata: 3 levels from the parent's two samples (same samples, so regression to the mean is in them); 2 levels cross-fitted, unbiased."""
    use = [i for i in ids if {0, 1} <= set(vp.get(i, {})) and {0, 1} <= set(vc.get(i, {}))]
    if not use:
        return dict(zx=None, dacc=None, zx_rows=0)
    idx = np.array([tab.index[i] for i in use])
    mk, y, r = tab.mask[idx], tab.y[idx], np.arange(len(use))
    def L(mv, s):
        return tab.module({i: {s: mv[i][s]} for i in use})[0][idx]
    def G(lp):
        x = np.where(mk, lp, -np.inf)
        x = x - x.max(1, keepdims=True)
        p = np.where(mk, np.exp(x), 0.)
        p /= p.sum(1, keepdims=True)
        g = -p
        g[r, y] += 1
        return g * mk
    d = lambda a, b: np.where(mk, a - b, 0.)
    P0, P1, C0, C1 = L(vp, 0), L(vp, 1), L(vc, 0), L(vc, 1)
    u = .5 * ((d(C1, P1) * G(P0)).sum(1) + (d(C0, P0) * G(P1)).sum(1))
    def zc(u):
        uc = collections.defaultdict(float)
        for x, i in zip(u, use):
            uc[cluster(i)] += x
        a = np.array(list(uc.values()))
        v = len(a) / max(len(a) - 1, 1) * (a ** 2).sum()
        return float(u.sum() / np.sqrt(v)) if v > 0 else 0.
    zp = None
    if isinstance(pool, np.ndarray):     # the fused pool: its residual on these rows (held out by picture on FIT, on SELECT the pool fit on FIT), the change of the two samples. Logged beside zx, gates nothing
        gp = G(pool[idx])
        zp = zc(.5 * ((d(C1, P1) * gp).sum(1) + (d(C0, P0) * gp).sum(1)))
    elif pool:                   # the pool's residual from one sample of every member, weights as fitted, the change from the other sample. Raw vote logits, no calibration: logged beside zx, gates nothing
        def Fp(s):
            F = np.zeros((len(use), tab.C))
            for mv, w in pool:
                has = [i for i in use if s in mv.get(i, {})]
                if not has or not w:
                    continue
                L_ = tab.module({i: {s: mv[i][s]} for i in has})[0][idx]
                F += float(w) * np.where(np.isfinite(L_), L_, 0.)
            return F
        zp = zc(.5 * ((d(C1, P1) * G(Fp(0))).sum(1) + (d(C0, P0) * G(Fp(1))).sum(1)))
    ok = lambda mv, s: np.array([bool(mv[i][s]["ok"]) for i in use], float)
    p0, p1, c0, c1 = ok(vp, 0), ok(vp, 1), ok(vc, 0), ok(vc, 1)
    gain = .5 * ((c0 - p0) + (c1 - p1))
    lev = p0 + p1
    st = {n: dict(n=int((lev == k).sum()), gain=float(gain[lev == k].sum())) for n, k in (("stably_wrong", 0), ("mixed", 1), ("stably_right", 2))}
    xf = {n: dict(n=int(((p0 == k).sum() + (p1 == k).sum()) / 2), gain=float(.5 * ((c1 - p1)[p0 == k].sum() + (c0 - p0)[p1 == k].sum()))) for n, k in (("wrong_elsewhere", 0), ("right_elsewhere", 1))}
    return dict(zx=zc(u), zx_pool=zp, dacc=float(gain.sum() / max(1, len(ids))), zx_rows=len(use), strata=st, strata_cross=xf)


def err_corr_pool(tab, F, right_m, idx):
    """Correlation of a member's errors with the pool's on the rows idx; pool answers as in Table.acc."""
    F = np.where(tab.mask, F, -np.inf).copy()
    rows = np.arange(len(tab.ids))
    ot = tab.other >= 0
    F[rows[ot], tab.other[ot]] = -np.inf
    pred = np.where(getattr(tab, "novote", np.zeros(len(tab.ids), bool)) | ~np.isfinite(F.max(1)), -1, F.argmax(1))
    a, b = (pred[idx] != tab.y[idx]).astype(float), (~np.asarray(right_m[idx], bool)).astype(float)
    return float(np.corrcoef(a, b)[0, 1]) if len(idx) > 2 and a.std() > 0 and b.std() > 0 else None


def dacc_lb(tab, vp, vc, ids, cluster, B=1000, seed=0):
    """One-sided 95% cluster-bootstrap lower bound of the paired accuracy gain. Logged."""
    gain = collections.defaultdict(list)
    for i in ids:
        if {0, 1} <= set(vp.get(i, {})) and {0, 1} <= set(vc.get(i, {})):
            gain[cluster(i)].append(.5 * sum(float(vc[i][s]["ok"]) - float(vp[i][s]["ok"]) for s in (0, 1)))
    c = list(gain.values())
    if not c:
        return None
    rng = np.random.default_rng(seed)
    sm, n = np.array([sum(x) for x in c]), np.array([len(x) for x in c])
    k = rng.integers(len(c), size=(B, len(c)))
    return float(np.quantile(sm[k].sum(1) / n[k].sum(1), .05))


# ---------------------------------------------------------------- table and pool ----------------------------------------------------------------
class Table:
    """Option columns of every row, from all votes of the given members. The same table serves every member of one computation."""
    def __init__(self, data, ids, votes):
        self.ids, self.index = list(ids), {i: k for k, i in enumerate(ids)}
        self.letters, self.text, self.y, self.other, self.col = [], [], [], [], []
        import string
        U = string.ascii_uppercase
        for i in self.ids:
            p = data.rows[i]
            all_v = [v for mv in votes.values() for v in mv.get(i, {}).values()]
            kind, ans = p["kind"], str(p["answer"])
            if kind == "mc" and len(ans) == 1 and ans in U:
                n = max(len(p.get("options") or []), 4, U.index(ans) + 1, *[U.index(v["key"]) + 1 for v in all_v if v["key"] and len(v["key"]) == 1 and v["key"] in U], 0)
                n = min(n, C_MAX)
                L = list(U[:n])
                col = {x: k for k, x in enumerate(L)}
                self.letters.append(L); self.text.append(list(p.get("options") or []))
                self.y.append(col.get(ans, 0)); self.other.append(-1); self.col.append(col)
            elif kind == "yesno":
                col = {"yes": 0, "no": 1}
                self.letters.append(["A", "B"]); self.text.append(["yes", "no"])
                self.y.append(col.get(ans.lower(), 0)); self.other.append(-1); self.col.append(col)
            else:
                cnt = collections.Counter()
                for v in all_v:
                    if v["key"] is not None:
                        cnt["=gold" if v["ok"] else v["key"]] += 1
                keep = [k for k, _ in cnt.most_common(C_MAX - 1)]
                if "=gold" in cnt and "=gold" not in keep:
                    keep[-1] = "=gold"
                keep.sort(key=lambda k: zlib.crc32((i + "|" + (ans.lower() if k == "=gold" else k)).encode()))
                col = {k: j for j, k in enumerate(keep)}
                o = len(keep)
                self.letters.append(list(U[:o + 1]))
                self.text.append([(ans if k == "=gold" else k)[:80] for k in keep] + [f"none of these; the right answer is {ans[:60]}" if "=gold" not in col else "none of these"])
                self.y.append(col.get("=gold", o)); self.other.append(o); self.col.append(col)
        self.y, self.other = np.array(self.y), np.array(self.other)
        self.novote = np.array([not any(v.get("key") is not None for mv in votes.values() for v in mv.get(i, {}).values()) for i in self.ids], bool)     # no member answered: the pool has no answer there
        self.C = max(len(L) for L in self.letters)
        self.mask = np.zeros((len(self.ids), self.C), bool)
        for k, L in enumerate(self.letters):
            self.mask[k, :len(L)] = True

    def column(self, k, v):
        c = self.col[k]
        if v["key"] is None:
            return None
        if self.other[k] < 0:
            return c.get(v["key"])
        return c.get("=gold" if v["ok"] else v["key"], int(self.other[k]))

    def module(self, mv):
        """logp [N, C], pred, correct, source"""
        N = len(self.ids)
        lp, pred, cor, src = np.full((N, self.C), -np.inf), np.full(N, -1), np.zeros(N, bool), np.full(N, 3, dtype=np.int8)
        for k, i in enumerate(self.ids):
            if i not in mv:
                continue
            n = len(self.letters[k])
            cols = [self.column(k, v) for v in mv[i].values()]
            cols = [c for c in cols if c is not None]
            if not cols:
                lp[k, :n] = -np.log(n)
                src[k] = 2
                continue
            cnt = np.bincount(cols, minlength=n).astype(float)
            p = (cnt + A) / (len(cols) + A * n)
            lp[k, :n] = np.log(p)
            src[k] = 0
            use = cnt.copy()
            if self.other[k] >= 0:
                use[self.other[k]] = -1
            if use.max() > 0:
                pred[k] = int(np.argmax(use))
                cor[k] = pred[k] == self.y[k]
        return lp, pred, cor, src

    def pool(self, data, members, votes):
        from vrsi27.boost.io import Pool
        M, N = len(members), len(self.ids)
        lp, pred, cor, src = np.zeros((M, N, self.C)), np.zeros((M, N), int), np.zeros((M, N), bool), np.zeros((M, N), np.int8)
        for j, m in enumerate(members):
            lp[j], pred[j], cor[j], src[j] = self.module(votes[m["name"]])
        Z = lambda: np.zeros((M, N))
        return Pool(modules=[m["name"] for m in members], qids=self.ids, letters=self.letters, logp=lp, opt_mask=self.mask, y=self.y, pred=pred, correct=cor, source=src,
                    vis_tokens=Z(), calls=Z(), latency=Z(), ingest_tokens=Z(), split=np.array([data.role.get(i, "x") for i in self.ids]), video_id=np.array([data.cluster(i) for i in self.ids]))

    def acc(self, F, idx):
        """Over voted columns only; a row whose gold nobody voted counts as wrong."""
        F = np.where(self.mask, F, -np.inf).copy()
        rows = np.arange(len(self.ids))
        ot = self.other >= 0
        F[rows[ot], self.other[ot]] = -np.inf
        pred = np.where(self.novote | ~np.isfinite(F.max(1)), -1, F.argmax(1))      # 11-08: argmax of an all -inf row is column 0, which is the OTHER column (= gold) of an open row nobody answered: it counted right
        return float((pred[idx] == self.y[idx]).mean()) if len(idx) else float("nan")


def splits(data, tab):
    from vrsi27.boost.splits import Splits
    ix = lambda role: np.array([k for k, i in enumerate(tab.ids) if data.role.get(i) == role], int)
    sh, sc, se = ix("show"), ix("score"), ix("select")
    return Splits(fit=np.sort(np.concatenate([sh, sc])), select=se, held=np.array([], int), show=sh, score=sc)


def state_of(pool, sp, members, seed=0):
    from vrsi27.boost.edge import round_state
    guard_groups()
    idx = list(range(len(members)))
    kids = [k for k, m in enumerate(members) if m["origin"] == "proposal"]
    return round_state(pool, sp, idx, combiner="evidence", seed=seed, innovation="diff", children=kids, parent=REF(members))


def REF(members):
    """The reference column of the evidence basis: stock. A child enters as what it changes relative to stock; its genetic parent is a separate matter."""
    return next(k for k, m in enumerate(members) if m["name"] == "stock")


def err_stats(pool, members, rows):
    """Mean pairwise correlation of the wrong indicator among members with a vote; each pair on the rows both answered."""
    n = len(members)
    Cor = np.full((n, n), np.nan)
    for a in range(n):
        for b in range(a + 1, n):
            ok = (pool.source[a, rows] == 0) & (pool.source[b, rows] == 0)
            if ok.sum() < 20:
                continue
            x, y = ~pool.correct[a, rows][ok], ~pool.correct[b, rows][ok]
            if x.std() > 0 and y.std() > 0:
                Cor[a, b] = Cor[b, a] = float(np.corrcoef(x, y)[0, 1])
    v = Cor[np.triu_indices(n, 1)]
    return (float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")), Cor


# ---------------------------------------------------------------- member programs ---------------------------------------------------------------
API = """## Harness program (optional)
Python source with one top-level function `harness(row)`. Standard library only, no files, no classes. It sees row = {"rid", "question", "kind": "mc"|"num"|"yesno"|"short", "options": [...], "n_images": n, "sizes": [[w, h], ...]}.
Return None to abstain on this question (the member then has no vote there; use this to specialise), or a dict:
{"views": [{"image": 0, "crop": [x0, y0, x1, y1] (fractions 0..1, optional), "scale": 1..4 (optional), "autocontrast": true (optional), "grid": 2..16 (optional, draws a labelled grid)}, ...at most 4],
 "keep_original": true|false, "prefix": "text before the question, at most 600 characters", "suffix": "text after it, at most 600 characters", "image_at": "start" (default) | "before_options" | "end", where the images sit in the message relative to the text}.
The gate must be computable from the row alone. The answer format asked by the question must stay."""


def views_of(tab, pool, F, members, data, sets, beside=None):
    """Pool and each member on each row set: all rows, without vlmsbiased, macro over benches. A member's plain number counts a row it abstains on as not right, as pool.correct does; answered is the share it answers and on_answered its accuracy there."""
    bench = lambda k: data.rows[tab.ids[k]].get("bench") or tab.ids[k].split("/")[0]
    out = {}
    for name, idx in sets.items():
        idx = np.asarray(idx, int)
        no = np.array([k for k in idx if bench(k) != "vlmsbiased"], int)
        by = collections.defaultdict(list)
        for k in idx:
            by[bench(k)].append(k)
        def cell(f):
            return dict(all=f(idx) if len(idx) else None, no_vlmsbiased=f(no) if len(no) else None, macro_bench=float(np.mean([f(np.array(v, int)) for v in by.values()])) if by else None, n=len(idx), n_no_vlmsbiased=len(no), benches=len(by))
        v = dict(pool=cell(lambda x: tab.acc(F, x)), members={})
        if beside is not None:
            v["pool_stacking"] = cell(lambda x: tab.acc(beside, x))
        for k, m in enumerate(members):
            c = cell(lambda x, k=k: float(pool.correct[k, x].mean()))
            ans = pool.source[k, idx] == 0
            c.update(answered=float(ans.mean()) if len(idx) else None, on_answered=float(pool.correct[k, idx][ans].mean()) if ans.any() else None)
            v["members"][m["name"]] = c
        out[name] = v
    return out


def size_of(p):
    from PIL import Image
    with Image.open(p) as im:
        return list(im.size)


def prog_rows(data, ids):
    out = []
    for i in ids:
        p = data.rows[i]
        ims = p.get("image_paths") or []
        out.append(dict(rid=i.replace("/", "_"), question=p["prompt"], kind=p["kind"], options=p.get("options") or [], n_images=len(ims), sizes=[size_of(x) for x in ims]))
    return out


def run_program(src, rows, timeout=120):
    from vrsi27.boost.lens import run_lens
    wrap = src.rstrip() + "\n\n\ndef lens(state):\n    out = {}\n    for r in state[\"rows\"]:\n        out[r[\"rid\"]] = harness(r)\n    return out\n"
    r = run_lens(wrap, {"rows": rows}, timeout_s=timeout)
    if not r.get("ok"):
        raise Bad(f"program {r.get('stage')}: {str(r.get('error'))[:300]}")
    return r["output"]


def clean_spec(s, n_images):
    if s is None:
        return None
    if not isinstance(s, dict) or set(s) - {"views", "keep_original", "prefix", "suffix", "image_at"}:
        raise Bad("program: unknown keys")
    if s.get("image_at", "start") not in ("start", "before_options", "end"):
        raise Bad("program: image_at must be start, before_options or end")
    vs = s.get("views") or []
    if not isinstance(vs, list) or len(vs) > 4:
        raise Bad("program: at most 4 views")
    for v in vs:
        if not isinstance(v, dict) or set(v) - {"image", "crop", "scale", "autocontrast", "grid"}:
            raise Bad("program: unknown view keys")
        if not isinstance(v.get("image", 0), int) or not 0 <= v.get("image", 0) < max(1, n_images):
            raise Bad("program: image index")
        if "crop" in v:
            c = v["crop"]
            if not (isinstance(c, list) and len(c) == 4 and all(isinstance(x, (int, float)) for x in c) and 0 <= c[0] < c[2] <= 1 and 0 <= c[1] < c[3] <= 1):
                raise Bad("program: crop")
        if not 1 <= v.get("scale", 1) <= 4:
            raise Bad("program: scale")
        if "grid" in v and not (isinstance(v["grid"], int) and 2 <= v["grid"] <= 16):
            raise Bad("program: grid")
    for t in ("prefix", "suffix"):
        if not isinstance(s.get(t, ""), str) or len(s.get(t, "")) > 600:
            raise Bad(f"program: {t} must be text of at most 600 characters; on one row it was {len(s[t]) if isinstance(s[t], str) else type(s[t]).__name__}")
    return s


def place_images(prompt, n, at):
    """Image position in the message. The engine's image_content puts the images where the prompt has exactly one <image> per image, else first."""
    if at == "start" or not n or "<image>" in prompt:
        return prompt
    tags = "\n".join(["<image>"] * n)
    m = re.search(r"\n\s*(?:Options?:|Choices?:|\(?A[).:])|\s\(A\)\s|\sA[.:]\s", prompt) if at == "before_options" else None
    return prompt[:m.start()] + "\n" + tags + prompt[m.start():] if m else prompt + "\n" + tags


def render_view(path, v, out):
    """The render, plus a note beside it (source, view, sha of the pixels) from which the node renders the same image itself: the link is too slow to carry views."""
    r = _render_view(path, v, out)
    if not os.path.exists(out + ".json"):
        from PIL import Image
        jdump(out + ".json", dict(src=path, v=v, pixels=hashlib.sha256(Image.open(out).convert("RGB").tobytes()).hexdigest()))
    return r


def _render_view(path, v, out):
    if os.path.exists(out):
        return out
    im = render_image(path, v)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    t = private_tmp(out, ".png")
    im.save(t, format="PNG")
    settle(t, out)
    return out


def render_image(path, v):
    from PIL import Image, ImageDraw, ImageOps
    im = Image.open(path).convert("RGB")
    W_, H_ = im.size
    if "crop" in v:
        x0, y0, x1, y1 = v["crop"]
        im = im.crop((int(x0 * W_), int(y0 * H_), max(int(x1 * W_), int(x0 * W_) + 1), max(int(y1 * H_), int(y0 * H_) + 1)))
    s = v.get("scale", 1)
    if s != 1:
        tw, th = int(im.width * s), int(im.height * s)
        f = min(1., 4096 / max(tw, th))
        im = im.resize((max(1, int(tw * f)), max(1, int(th * f))), Image.BICUBIC)
    if v.get("autocontrast"):
        im = ImageOps.autocontrast(im)
    if v.get("grid"):
        g = v["grid"]
        d = ImageDraw.Draw(im)
        for k in range(1, g):
            d.line([(im.width * k / g, 0), (im.width * k / g, im.height)], fill=(255, 0, 0), width=1)
            d.line([(0, im.height * k / g), (im.width, im.height * k / g)], fill=(255, 0, 0), width=1)
        for k in range(g):
            d.text((im.width * k / g + 2, 2), str(k), fill=(255, 0, 0))
            d.text((2, im.height * k / g + 2), str(k), fill=(255, 0, 0))
    return im


def carried_gates(*parents):
    """The gate notes these parents' networks carry, each as its scope and the opening of its note."""
    out = {}
    for p in parents:
        for x in p.get("inherited_gates", []):
            out[x["scope"], x["head"]] = x
        if p.get("note"):
            x = dict(scope=p["gate"]["scope"], head=p["note"][:20], origin=p["name"])
            out[x["scope"], x["head"]] = x
    return [out[k] for k in sorted(out)]


def with_gates(m, parents):
    """A member inherits its parents' gates. A gate was written for the plain prompt of its category, so on those rows the program does not apply; a program member's id says so, since its records differ from the same program's without this rule."""
    g = carried_gates(*parents)
    if g:
        m["inherited_gates"] = g
        if m.get("program"):
            m["mid"] = sha(m["mid"] + canon(g))[:14]
    return m


def active_amendment(work, r):
    """The arm's amendment at round r. A "next" block takes over from its own from_round; the round before it finishes on the pool it began on."""
    am = jload(f"{work}/amendment.json", None) or {}
    n = am.get("next")
    return {**am, **n} if n and r >= n["from_round"] else am


def in_gate_scope(data, m, i):
    """Same rule as the write, recited and touched: the row's bench|category equals the scope. A row in two inherited scopes is one plain row with both gates in the network."""
    k = vh.key(data.rows[i])
    return any(x["scope"] == f"cat:{k}" for x in m.get("inherited_gates", [])) or (m.get("gate") or {}).get("scope") == f"cat:{k}"


def program_rows(data, m, ids, render=True):
    """Rows as this member's program shows them, and the ids it abstains on. Rows in an inherited gate's scope stay plain."""
    if not m.get("program"):
        return [data.rows[i] for i in ids], []
    specs = run_program(m["program"], prog_rows(data, [i for i in ids if not in_gate_scope(data, m, i)]))
    out, abst = [], []
    for i in ids:
        p = data.rows[i]
        if in_gate_scope(data, m, i):
            out.append(dict(p))
            continue
        ims = p.get("image_paths") or []
        s = clean_spec(specs.get(i.replace("/", "_")), len(ims))
        if s is None:
            abst.append(i)
            continue
        new = dict(p)
        if s.get("views"):
            paths = list(ims) if s.get("keep_original", True) else []
            for k, v in enumerate(s["views"]):
                out_ = f"{HB}/views/{m['mid']}/{sha(i + canon(v))[:16]}.png"
                paths.append(render_view(ims[v.get("image", 0)], v, out_) if render else out_)
            new["image_paths"] = paths
        new["prompt"] = (s.get("prefix", "") + "\n\n" if s.get("prefix") else "") + p["prompt"] + ("\n\n" + s["suffix"] if s.get("suffix") else "")
        new["prompt"] = place_images(new["prompt"], len(new.get("image_paths") or []), s.get("image_at", "start"))
        out.append(new)
    return out, abst


# ---------------------------------------------------------------- evaluation ---------------------------------------------------------------------
class Ev:
    def __init__(self, work, data, sleep=20, patience=6 * 3600):
        self.work, self.data, self.sleep, self.patience = work, data, sleep, patience
        self.q = vh.Queue()
        self.pending = []
        self.jobs = []
        self.use_worker = False
        self.tag = sha(work)[:4]
        self.sent, self.dead = {}, {}

    def submit(self, name, cmd, outs, flag):
        name = f"hb{self.tag}_{name}"
        self.q.submit(name, cmd)
        self.pending.append(dict(name=name, outs=outs, flag=flag))

    def wait(self, say=print):
        t0 = time.time()
        if self.pending:                      # a wait shows in the run log when it begins: a silent build looked like a stall on 11-07
            say(f"waiting on {len(self.pending)} jobs: {', '.join(sorted({j['name'] for j in self.pending})[:6])}")
        while self.poll(say, t0):
            time.sleep(self.sleep)
        self.again(say)

    def again(self, say):
        """A failed item goes out once more. A second failure marks its rows dead: a candidate that needs them stops (Run.stop_dead), it is never read on partial rows; a pool member's take failed records (write_failed)."""
        out_, self.sent = [(o, x) for o, x in self.sent.items() if not os.path.exists(o) and os.path.exists(o + ".failed")], {}
        if not out_:
            return
        say(f"{len(out_)} failed items go once more: {', '.join(os.path.relpath(o, EVR) for o, _ in out_[:6])}")
        self.evaluate_items([x for _, x in out_])
        t0 = time.time()
        while self.poll(say, t0):
            time.sleep(self.sleep)
        self.sent = {}
        for o, (m, ids) in out_:
            if not os.path.exists(o):
                self.dead.setdefault(self.dir_of(m), set()).update(ids)
                say(f"item failed twice: {m['name']} {os.path.relpath(o, EVR)}, {len(ids)} rows")

    def evaluate_items(self, reqs):
        for m, ids in reqs:
            d = self.dir_of(m)
            with self.locked(d):
                it = self.plan_item(m, ids)
                if it:
                    self.batch_job(it["model"], [it])
                    self.note(d, dict(rows=it["ids"], out=it["out"]))

    def dead_for(self, m, ids):
        return sorted(self.dead.get(self.dir_of(m), set()) & set(ids))

    def write_failed(self, m, ids, say):
        """Rows of a pool member whose item failed twice: answered wrong, as a too-long row, and said out loud. A file of its own, so the failed items stay as they are."""
        d, rows = self.dir_of(m), [dict(self.data.rows[i], id=i) for i in ids]
        f = f"{d}/a.failed_{sha(canon(sorted(ids)) + gsfx())[:10]}.jsonl"
        if not os.path.exists(f):
            if GEN_KEY:
                jdump(f + ".gen", dict(GEN_KEY, by="driver", finish="failed"))
            t = private_tmp(f)
            with open(t, "w") as o:
                for r in no_run_lines(rows, "failed", 2):
                    o.write(json.dumps(r, ensure_ascii=False) + "\n")
            settle(t, f)
            say(f"{m['name']}: {len(ids)} rows failed twice, written as failed, counted wrong")

    def copy_stranded(self, say):
        """A batch the node claimed may be stranded behind a dropped link: Vesta runs the same line as NAME_v, same rows and same out file. The first result to reach the out file stands; a later Vesta result goes to .later, a later node result to .node, and neither is ever read (node_copy_after in prereg, coordinator 11-07).
        Copy when the claim is older than node_copy_after seconds, or when the Mac's mark mac.hb is over 180 s old, the claim over 120 s, and a Vesta card is idle. Each copy is logged in copies.jsonl."""
        if not getattr(self, "node_copy_after", None) or time.time() - getattr(self, "copy_t", 0) < 60:
            return
        self.copy_t = time.time()
        extra, stale, idle = f"{vh.W}/evoq_extra.txt", None, None
        for j in self.pending:
            n = j["name"]
            cl = f"{HB}/claims/{n}"
            if j.get("worker") or j.get("copied") or n.endswith("_v") or all(os.path.exists(o) for o in j["outs"]):
                continue
            if not os.path.exists(cl) or os.path.getsize(cl):      # only the bridge leaves an empty claim; a card worker writes its card
                continue
            age = time.time() - os.path.getmtime(cl)
            if stale is None:
                stale = time.time() - os.path.getmtime(f"{HB}/mac.hb") if os.path.exists(f"{HB}/mac.hb") else 1e9
            why = "late" if age >= self.node_copy_after else None
            if not why and stale > 180 and age > 120:
                if idle is None:
                    o = subprocess.run("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits", shell=True, capture_output=True, text=True).stdout
                    idle = sum(1 for x in o.split() if x.isdigit() and int(x) < 8000)
                why = "link" if idle else None
            if not why:
                continue
            j["copied"] = True
            have = open(extra).read() if os.path.exists(extra) else ""
            if f"{n}_v\t" in have:
                continue
            cmd = next((l.rstrip("\n").split("\t", 1)[1] for l in open(self.q.qfile) if l.startswith(n + "\t")), None)
            if not cmd:
                continue
            with open(extra, "a") as f:
                f.write(f"{n}_v\t{cmd.replace(f' {n} ', f' {n}_v ')}\n")
            with open(f"{self.work}/copies.jsonl", "a") as f:
                f.write(json.dumps(dict(t=time.time(), name=n, why=why, claim_age=round(age), mac_hb_age=round(stale), outs=j["outs"])) + "\n")
            say(f"copy on Vesta: {n}, {why}, claim {age / 60:.0f} min, Mac mark {stale / 60:.1f} min")

    def poll(self, say, t0):
        if True:
            self.copy_stranded(say)
            left = []
            for j in self.pending:
                if all(os.path.exists(o) for o in j["outs"]) or os.path.exists(j["flag"]):
                    continue
                if j.get("worker"):
                    if not self.worker_alive() and glob.glob(f"{self.wdir()}/inbox/*.json"):
                        self.worker_ensure()              # the worker left with work still in its inbox: the next one takes it
                    age = self.worker_age()
                    L = jload(f"{self.wdir()}/launch.json", dict(t=0))
                    if self.worker_pending() and not self.worker_alive():
                        age = None                        # a heartbeat left by an earlier worker says nothing about one still queued or loading
                    if age is not None and age > 300 and time.time() - age >= L["t"] - 1 or (age is not None and age > 300 and time.time() - L["t"] > 900):      # a heartbeat older than the last launch is the old worker's: the new one is still starting
                        say(f"worker silent {age:.0f}s: {j['name']}")
                        open(j["flag"], "w").write("worker died\n")
                        continue
                if self.q.failed(j["name"]):
                    say(f"job failed: {j['name']}")
                    open(j["flag"], "w").write("job failed\n")
                    continue
                if time.time() - t0 > self.patience:
                    say(f"job past patience: {j['name']}")
                    open(j["flag"], "w").write("past patience\n")
                    continue
                left.append(j)
            self.pending = left
            if left and time.time() - getattr(self, "waiting_said", t0) >= 900:          # a long wait shows in the run log: how long, how many, which
                self.waiting_said = time.time()
                say(f"still waiting {(time.time() - t0) / 60:.0f} min on {len(left)} jobs: {', '.join(sorted({j['name'] for j in left})[:6])}")
            if getattr(self, "tick", None):
                self.tick()
            return left

    # the ledger of every client (driver, filler, reporter, queue-ahead), one file per answer dir: a row asked and still alive is not asked again by anyone, whatever chunk it is in. Append only, cancels are lines
    def job_tail(self, seed_off=0):
        t = gen_tail(vh.TAIL.format(lib=vh.DIRLIB))
        u = getattr(self, "gpu_util", None)
        t = t.replace("--gpu-util 0.85", f"--gpu-util {u}") if u else t
        if seed_off:                 # 1109c fix (11-09): a draw's seed goes on its own line's --seed; the card code (code-xf) never reads an item's seed_off
            t = re.sub(r"--seed (\d+)", lambda x: f"--seed {int(x.group(1)) + int(seed_off)}", t)
        return t

    def dir_of(self, m):
        if m.get("draw"):
            return member_dirs(m)[0]
        return f"{EVR}/ev/{m['mid'] if m.get('program') else vh.gid(m['genome'])}"

    @contextlib.contextmanager
    def locked(self, d):
        os.makedirs(d, exist_ok=True)
        with open(f"{d}/asked.lock", "a") as h:
            fcntl.flock(h, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(h, fcntl.LOCK_UN)

    def asked(self, d):
        out, f = {}, f"{d}/asked.jsonl"
        if os.path.exists(f):
            for l in open(f):
                x = json.loads(l)
                for i in x["rows"]:
                    if not x.get("cancel"):
                        out[i] = x
                    elif out.get(i, {}).get("out") == x["out"]:
                        del out[i]
        return out

    def note(self, d, x):
        with open(f"{d}/asked.jsonl", "a") as o:
            o.write(json.dumps(dict(x, t=time.time(), client=getattr(self, "client", "driver"))) + "\n")

    def line_of(self, x):
        return x.get("name") or (jload(x["out"] + ".job") or {}).get("name")

    def alive(self, x):
        """Its output can still come. Same retry as the reporter's old ledger; a claim that never got a line (a client died between the two) goes after 5 min."""
        if os.path.exists(x["out"]) or os.path.exists(x["out"] + ".failed") or time.time() - x["t"] > ASK_RETRY:
            return False
        n = self.line_of(x)
        return time.time() - x["t"] < 300 if n is None else not self.q.failed(n)

    def cancel(self, d, x):
        """A low line nobody has taken stops (a claim that says so, as fill_settle does). False if a card has it."""
        os.makedirs(f"{HB}/claims", exist_ok=True)
        try:
            fd = os.open(f"{HB}/claims/{x['name']}", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        os.write(fd, json.dumps(dict(cancelled=True, t=time.time())).encode())
        os.close(fd)
        self.note(d, dict(cancel=True, rows=x["rows"], out=x["out"], name=x["name"]))
        return True

    def take(self, d, need, wait=True):
        """Under the dir's lock: the rows of need this client must ask. A row of a low line waiting for a card comes to a client above it; a row someone is answering is waited on (wait) and not asked."""
        led, mine, on, done = self.asked(d), [], {}, {}
        for i in need:
            x = led.get(i)
            if x is not None and os.path.exists(x["out"]) and gen_ok(x["out"]):       # answered after this client read its rows; it reads them again once it has waited (a file of another regime is not an answer, 1109c fix 11-09)
                if x["out"] not in done:
                    with open(x["out"]) as h:
                        done[x["out"]] = {json.loads(l)["id"] for l in h}
                if i in done[x["out"]]:
                    continue
            if x is not None and not self.alive(x) and self.line_of(x) and self.q.failed(self.line_of(x)) and not os.path.exists(x["out"]) and not os.path.exists(x["out"] + ".failed"):
                open(x["out"] + ".failed", "w").write("job failed\n")     # nobody was waiting on it
            if x is None or not self.alive(x):
                mine.append(i)
            elif x.get("low") and not getattr(self, "low", False) and self.cancel(d, x):
                for j in x["rows"]:
                    led.pop(j, None)
                mine.append(i)
            else:
                on[x["out"]] = x
        if wait:
            for x in on.values():
                mark = jload(x["out"] + ".job") or {}
                self.pending.append(dict(name=self.line_of(x) or "asked", outs=[x["out"]], flag=x["out"] + ".failed", worker=mark.get("worker", False)))
        return mine

    def model(self, g):
        return f"{MODELS}/{sha(canon(g['vision']))[:12]}" if g.get("vision") else vh.STOCK

    def build(self, g):
        v = g.get("vision")
        if not v:
            return
        m = self.model(g)
        if os.path.exists(f"{m}/config.json"):
            return
        cmd = f"python -u rsi/vision_patch.py --model {vh.STOCK} --out {m} --patch {v['patch']} --kernel {v['kernel']} --max-tokens {v['max'] or 4096}"
        cmd += f" --min-tokens {v['min']}" if v.get("min") else ""
        cmd += f" --merge {v['merge']}" if v.get("merge") else ""
        self.submit(f"ck_{sha(canon(v))[:10]}", cmd, [f"{m}/config.json"], f"{m}.failed")

    def arm(self, g):
        f = f"{HB}/arms/{vh.gid(g)}.json"
        if not os.path.exists(f):
            jdump(f, [{"label": "a", "effort": "medium", "engine": g["engine"]}])
        return f

    def evaluate(self, m, ids):
        return self.evaluate_many([(m, ids)])

    def plan_item(self, m, ids):
        """Rows file and output of one member on rows it lacks; None if nothing to run."""
        if not ids:
            return None
        rows, abst = program_rows(self.data, m, ids)
        tag = sha(canon(sorted(ids)) + gsfx() + (f":seed+{m['draw']['d'] * DRAW_OFF}" if m.get("draw") else ""))[:10]          # with a prereg gen the regime is in the name: a file of another regime with the same rows never stands in for this one; a draw's own seed too (1109c fix 11-09: files a card made at the base seed are refused, and never block a re-ask)
        if abst:
            jdump(f"{EVR}/prog/{m['mid']}/abstain.{tag}.json", abst)
        if not rows:
            return None
        d = self.dir_of(m)
        os.makedirs(d, exist_ok=True)
        name = f"r{tag}"
        out, src = f"{d}/a.{name}.jsonl", f"{HB}/rows/{(m['mid'] if m.get('program') else 'p')}_{tag}.jsonl"
        if os.path.exists(out):
            return None
        tries = len(glob.glob(out + ".failed*"))
        if os.path.exists(out + ".failed"):
            if tries >= 2:                     # failed, went again, failed: the rows stay unanswered and the driver stops whoever needs them
                self.dead.setdefault(d, set()).update(x["id"] for x in rows)
                return None
            os.replace(out + ".failed", out + ".failed0")
            if os.path.exists(out + ".job"):
                os.replace(out + ".job", out + ".job0")
        os.makedirs(os.path.dirname(src), exist_ok=True)
        t = private_tmp(src)
        with open(t, "w") as o:
            for r in rows:
                o.write(json.dumps(r, ensure_ascii=False) + "\n")
        settle(t, src)
        g = m["genome"]
        return dict(kind="eval", model=self.model(g), engine=g["engine"], rows=src, out=out, n=len(rows), old=f"hb{self.tag}_{m['mid'][:8]}_{tag}", ids=[x["id"] for x in rows], **(dict(attempt=1) if tries else {}), **(dict(part=sorted(ids)) if abst else {}), **(dict(seed_off=m["draw"]["d"] * DRAW_OFF) if m.get("draw") else {}))

    def evaluate_many(self, reqs, cards=14, chunk=None):
        """One engine per checkpoint: members on one checkpoint share a job, as far as an even share of the cards allows; long lists are cut so idle cards can take them."""
        chunk = chunk or getattr(self, "chunk", 500)
        items = []
        lines = open(self.q.qfile).read() if os.path.exists(self.q.qfile) else ""
        noted = []
        try:
            for m, ids in reqs:
                have = load_raw(m) if gate_scope(m) is not None else load_member(m)
                need = [i for i in own_rows(m, ids) if i not in have]
                if not need:
                    continue
                d = self.dir_of(m)
                with self.locked(d):
                    need = sorted(self.take(d, need), key=lambda i: (-BENCH_COST.get(i.split("/")[0], 1.), sha(i)))      # one order for every client, so the same missing rows cut the same way; long benches first and in their own parts, so a long row does not hold a card at the end of a mixed part (11-08)
                    if not need:
                        continue
                    whole = self.plan_item(m, need) if len(need) <= chunk else None
                    if len(need) > chunk:
                        tag = sha(canon(sorted(need)) + gsfx())[:10]
                        old = f"hb{self.tag}_{m['mid'][:8]}_{tag}"
                        if old + "\t" in lines:                                  # already running as one job from before this change
                            self.pending.append(dict(name=old, outs=[f"{d}/a.r{tag}.jsonl"], flag=f"{d}/a.r{tag}.jsonl.failed"))
                            continue
                        for k in range(0, len(need), chunk):
                            it = self.plan_item(m, need[k:k + chunk])
                            if it:
                                items.append(it)
                                self.note(d, dict(rows=it["ids"], out=it["out"]))
                                noted.append((d, it))
                    elif whole:
                        items.append(whole)
                        self.note(d, dict(rows=whole["ids"], out=whole["out"]))
                        noted.append((d, whole))
        except Exception:                # a program failed on its rows: nothing of this call goes out, so its claims go too
            for d, it in noted:
                with self.locked(d):
                    self.note(d, dict(cancel=True, rows=it["ids"], out=it["out"]))
            raise
        todo = []
        for it in items:
            mark = jload(it["out"] + ".job")
            if it["old"] + "\t" in lines:
                self.pending.append(dict(name=it["old"], outs=[it["out"]], flag=it["out"] + ".failed"))
            elif mark:
                self.pending.append(dict(name=mark["name"], outs=[it["out"]], flag=it["out"] + ".failed", worker=mark.get("worker", False)))
            else:
                todo.append(it)
        by_out = {self.dir_of(m): m for m, _ in reqs}
        for it in todo:
            self.sent[it["out"]] = (by_out[os.path.dirname(it["out"])], it.get("part", it["ids"]))
        if not todo:
            return None
        per = max(min(sum(i["n"] for i in todo) / cards, getattr(self, "batch_cap", 1e9)), max(i["n"] for i in todo))
        groups = collections.defaultdict(list)
        for it in todo:
            groups[it["model"], int(it.get("seed_off") or 0)].append(it)
        for (model, soff), its in groups.items():
            cur, n, jobs = [], 0, []
            for it in its:
                if cur and n + it["n"] > per:
                    jobs.append(cur)
                    cur, n = [], 0
                cur.append(it)
                n += it["n"]
            jobs.append(cur)
            for b in jobs:
                if model == vh.STOCK and not soff and self.worker_idle():
                    self.to_worker(b)
                else:
                    self.batch_job(model, b)
        return None

    def fill_many(self, reqs, n=100):
        """Rows ahead of time on idle cards, lowest priority (name _s, in claims/LOW and the node's low_prefix): the stage 2 filler, the queue-ahead at decide and the reporter. One member per line, members in turn, not waited on. All in the ledger, so nobody asks them again, and a client above them takes a row back while no card has it."""
        per = []
        low, self.low = getattr(self, "low", False), True
        try:
            for m, ids in reqs:
                have = load_raw(m) if gate_scope(m) is not None else load_member(m)
                d = self.dir_of(m)
                with self.locked(d):
                    need = sorted(self.take(d, [i for i in ids if i not in have], wait=False), key=sha)
                per.append([(m, d, need[k:k + n]) for k in range(0, len(need), n)])
            self.fills = getattr(self, "fills", [])
            for k in range(max(map(len, per), default=0)):
                for L in per:
                    if k >= len(L):
                        continue
                    m, d, part = L[k]
                    with self.locked(d):
                        part = self.take(d, part, wait=False)
                        it = self.plan_item(m, part)
                        if not it:
                            continue
                        name = f"hb{self.tag}_s{sha(it['out'] + ('again' if it.get('attempt') else ''))[:12]}"
                        jdump(f"{HB}/jobs/{name}.json", [it])
                        self.q.submit(name, f"python -u rsi/hboost.py job-batch {HB}/jobs/{name}.json {name} --samples 2 --model {it['model']} --pool {IID} " + self.job_tail(int(it.get("seed_off") or 0)))
                        self.note(d, dict(rows=it["ids"], out=it["out"], name=name, low=True))
                    self.fills.append(dict(name=name, mid=m["mid"], out=it["out"], ids=it["ids"], d=d))
        finally:
            self.low = low

    def fill_settle(self, picked):
        """At stage 2: filler nobody has taken stops. A running line of a picked member is waited on, and its rows are not asked again. Returns mid -> rows running."""
        running = collections.defaultdict(set)
        for x in getattr(self, "fills", []):
            if os.path.exists(x["out"]):
                continue
            with self.locked(x["d"]):
                gone = self.cancel(x["d"], dict(name=x["name"], rows=x["ids"], out=x["out"]))
            if not gone and x["mid"] in picked:
                running[x["mid"]].update(x["ids"])
                self.pending.append(dict(name=x["name"], outs=[x["out"]], flag=x["out"] + ".failed"))
        self.fills = []
        return running

    def batch_job(self, model, its):
        name = f"hb{self.tag}_{'se' if getattr(self, 'early', False) else 'gb' if getattr(self, 'prio', False) else 'b'}{sha(canon([i['out'] for i in its] + ['again'] * any(i.get('attempt') for i in its)))[:12]}"      # gb: gate work, ahead of plain lines on Vesta
        f = f"{HB}/jobs/{name}.json"
        jdump(f, its)
        offs = {int(i.get("seed_off") or 0) for i in its}
        assert len(offs) == 1, f"one seed per line: {offs}"
        self.q.submit(name, f"python -u rsi/hboost.py job-batch {f} {name} --samples 2 --model {model} --pool {IID} " + self.job_tail(offs.pop()))
        self.jobs.append(name)
        for it in its:
            jdump(it["out"] + ".job", dict(name=name))
            self.pending.append(dict(name=name, outs=[it["out"]], flag=it["out"] + ".failed"))

    # the long-lived stock engine: the proposer between rounds, and stock-checkpoint evaluations meanwhile
    def wdir(self):
        return f"{self.work}/worker"

    def worker_age(self):
        f = f"{self.wdir()}/hb"
        return time.time() - os.path.getmtime(f) if os.path.exists(f) else None

    def worker_alive(self):
        a = self.worker_age()
        return self.use_worker and a is not None and a < 120

    def worker_idle(self):
        return self.worker_alive() and not glob.glob(f"{self.wdir()}/inbox/*.json") and not os.path.exists(f"{self.wdir()}/busy")

    def worker_pending(self):
        """The last launched worker has not ended: its queue line still waits, however long, or its job is running and its log still moves."""
        name = jload(f"{self.wdir()}/launch.json", {}).get("name")
        if not name or os.path.exists(f"{self.q.logs}/selfedit-{name}.rc"):
            return False
        log = f"{self.q.logs}/selfedit-{name}.log"
        return not os.path.exists(log) or time.time() - os.path.getmtime(log) < 900

    def worker_ensure(self):
        os.makedirs(f"{self.wdir()}/inbox", exist_ok=True)
        if self.worker_alive() or self.worker_pending():
            return
        L = jload(f"{self.wdir()}/launch.json", dict(n=0, t=0))
        n = L["n"] + 1
        name = f"hb{self.tag}_wk{n}"
        self.q.submit(name, f"python -u rsi/hboost.py job-worker {self.wdir()} {name} --samples 2 --model {vh.STOCK} --pool {IID} " + self.job_tail())
        self.jobs.append(name)
        jdump(f"{self.wdir()}/launch.json", dict(n=n, t=time.time(), name=name))

    def to_worker(self, its):
        name = jload(f"{self.wdir()}/launch.json", {}).get("name", "worker")
        for it in its:
            f = f"{self.wdir()}/inbox/{time.time():.6f}.json"
            jdump(f, it)
            if it["kind"] == "eval":
                jdump(it["out"] + ".job", dict(name=name, worker=True))
            self.pending.append(dict(name=name, outs=[it["out"]], flag=it["out"] + ".failed", worker=True))


NO_RUN = ("too_long", "failed")      # records of rows the engine never answered: wrong for the member, no vote in the pool, no input to a write


def no_conf_api(s):
    """The lens text names self.answer_distribution() and a confidence gate. A program here is harness(row): it sees the row only, so a proposal built on that text is invalid."""
    note = "A program cannot gate on this slice, since `harness(row)` sees the row only: find what these rows share that the row itself shows, and gate on that."
    s = s.replace("(not in the question text; observable at run time: `self.answer_distribution()` right after `super().answer_question(...)`)", "(not in the question text, and a harness program cannot read it: `harness(row)` sees the row only)")
    s = s.replace("Gate exactly on this slice with the confidence gate (harness API below); it cannot be read from the question text.", note)
    s = re.sub(r"Use the confidence gate with tau = [\d.]+ \(harness API below\)\.", note, s)
    s = re.sub(r"At runtime the parent's high confidence \(answer_distribution\(\) max >= ([\d.]+)\) confirms a prior is driving the answer; the gate should target the question TYPE, not the confidence itself\.", r"The parent's high confidence (max option probability >= \1) suggests a prior is driving the answer; the gate should target the question TYPE.", s)
    s = re.sub(r"Goal: where `([^`]+)` is unsure of its own answer, fetch better evidence and answer again; where it is confident, change nothing \(no extra VLM call\)\.", r"Goal: find what the rows where `\1` is unsure share that the row itself shows, and give those rows better evidence (views, crops, prompt text); leave the others as they are.", s)
    return re.sub(r"[^\n]*answer_distribution[^\n]*\n?", "", s)


def gain_positive(g):
    """The pool-increment gate lb > 0, with lb and ce below GAIN_TOL as zero."""
    lb, ce = g.get("ce_gain_select_lb"), g.get("ce_gain_select")
    return lb is not None and ce is not None and float(lb) >= GAIN_TOL and abs(float(ce)) >= GAIN_TOL


def no_run_lines(rows, finish, n, **more):
    """The answer lines of a row that never ran. votes_of gives key None, ok False, like a sample that did not stop: Table.module has no vote there and correct False, zx_stats and dacc count it wrong. Not an abstention, which has no samples."""
    out = []
    for p in rows:
        for k in range(n):
            r = {"id": p["id"], "kind": p["kind"], "answer": p["answer"], "sample": k, "reasoning": "", "content": "", "finish": finish, "tokens": 0, "start": 0, "ok": False, "loop": False, **more, **p.get("_too_long", {})}
            if p.get("image_paths"):
                r.update(options=p.get("options") or [], category=p.get("category"), images=list(p["image_paths"]))
            out.append(r)
    return out


def eval_rows(S, L, a, probs):
    """vLLM adds every request of a batch before it generates, and on an over-length prompt it aborts the ones added and raises (entrypoints/common/offline.py _render_and_add_requests), so nothing has run. The count of requests added names the row. It leaves the batch and the rest go again, seeds by row id so theirs do not move. No cost unless a row is over."""
    eng, long_, left, n = S.llm.llm_engine, [], list(probs), [0]
    add = eng.add_request
    def counted(*x, **y):
        o = add(*x, **y)
        n[0] += 1
        return o
    eng.add_request = counted
    try:
        while left:
            n[0] = 0
            try:
                return L.evaluate(S, left, a.samples), long_
            except Exception as e:
                m = re.search(r"prompt \(length (\d+)\) is longer than the maximum model length of (\d+)", str(e))
                if not m or n[0] >= len(left):
                    raise
                p = left.pop(n[0])
                long_.append(dict(p, _too_long=dict(prompt_len=int(m.group(1)), max_model_len=int(m.group(2)))))
                print(f"too long: {p['id']}: prompt {m.group(1)} > {m.group(2)}", flush=True)
        return [], long_
    finally:
        eng.add_request = add


def run_item(S, L, a, it):
    t0 = time.time()
    out = it["out"]
    h = None
    if it["kind"] == "eval" and os.path.exists(out):      # a copy of a batch another site finished first
        print(f"{out}: already there, skipped", flush=True)
        return
    try:
        if it["kind"] == "gen":
            reqs = json.load(open(it["req"]))
            msgs = [[{"role": "system", "content": r["system"]}, {"role": "user", "content": r["user"]}] for r in reqs]
            outs = S.generate(msgs, n=1, sampling=dict(L.SAMPLING, temperature=.7), max_tokens=12000, reasoning_effort="medium")
            res = []
            for r, o in zip(reqs, outs):
                s = o["samples"][0]
                res.append(dict(id=r["id"], content=s["content"], reasoning=s["reasoning"], finish=s["finish_reason"], tokens=s["num_tokens"]))
            print(f"generated {len(res)} in {time.time() - t0:.0f}s; tokens {[x['tokens'] for x in res]}", flush=True)
            json.dump(res, open(out + ".tmp", "w"), ensure_ascii=False)
            os.replace(out + ".tmp", out)
            jdump(out + ".time", dict(secs=time.time() - t0, n=len(res)))
            return
        probs = [json.loads(l) for l in open(it["rows"])]
        a.__dict__.setdefault("effort0", a.effort)        # the tail's effort (vharness: medium); before 11-08 set to medium here whatever the tail said
        a.effort = it.get("effort") or a.effort0
        a.__dict__.setdefault("seed0", a.seed)            # 128c: a draw answers with its own seed (crn_seeds reads args.seed per call); every other item with the tail's
        a.seed = a.seed0 + int(it.get("seed_off") or 0)
        if it.get("engine"):
            h = S.apply(it["engine"])
        L.mm_eval.PARTIAL = out + ".partial"
        R, long_ = eval_rows(S, L, a, probs)
        R = R + no_run_lines(long_, "too_long", a.samples)
        if h is not None:
            S.revert(h)
            h = None
        with open(out + ".tmp", "w") as f:
            for r in R:
                f.write(json.dumps({k: v for k, v in r.items() if k != "seq"}, ensure_ascii=False) + "\n")
        first = not os.path.exists(out)
        if first:
            jdump(out + ".gen", dict(stamp_of(a), model=it["model"]))      # the regime, on disk before the answers: a reader never sees the file without it
        os.replace(out + ".tmp", out if first else out + ".later")      # the first to finish stands; a later copy is kept beside it
        jdump(out + (".time" if first else ".later.time"), dict(secs=time.time() - t0, n=len(probs), model=it["model"], tokens=sum(r["tokens"] for r in R) / max(1, len(R))))
        print(f"item {os.path.basename(out)}: {len(probs)} rows in {time.time() - t0:.0f}s, {3600 * len(probs) / (time.time() - t0):.0f} rows/h" + (f", {len(long_)} too long" if long_ else ""), flush=True)
    except Exception:
        import traceback
        traceback.print_exc()
        if h is not None:
            S.revert(h)
        open(out + ".failed", "w").write("item failed\n")


def engine_up(argv, run_dir):
    t0 = time.time()
    os.makedirs(run_dir, exist_ok=True)
    argv, pc = list(argv), 0
    if "--prefix-cache" in argv:          # 128 tail flag (11-08), not one of loop.py's: vLLM prefix caching. Surgeon resets the cache after every weight edit and revert, and the encoder cache after vision edits
        k = argv.index("--prefix-cache")
        pc = int(argv[k + 1])
        del argv[k:k + 2]
    sys.argv = ["loop.py", "--run-dir", run_dir] + argv
    sys.path[:0] = [CODE, os.path.join(CODE, "engine")]
    import rsi.loop as L
    from surgery import Surgeon
    a = L.args
    a.prefix_cache = pc
    S = Surgeon(a.model, tp=a.tp, max_model_len=a.max_model_len, max_num_seqs=a.max_num_seqs, max_num_batched_tokens=a.max_num_batched_tokens, gpu_memory_utilization=a.gpu_util, kv_cache_dtype=a.kv_cache_dtype, multimodal=True, enable_prefix_caching=bool(pc))
    L._S = S
    return S, L, a, time.time() - t0


def cmd_job_batch(argv):
    spec, name = argv[0], argv[1]
    S, L, a, cold = engine_up(argv[2:], f"{HB}/jobs/run_{name}")
    its = json.load(open(spec))
    jdump(f"{HB}/timing/{name}.json", dict(cold=cold, items=len(its)))
    print(f"cold start {cold:.0f}s; {len(its)} items", flush=True)
    for it in its:
        run_item(S, L, a, it)
    if not any(os.path.exists(it["out"] + ".failed") for it in its):
        chain_batches(S, L, a, name, argv[2:])


STAY_IDLE, STAY_MAX = 45, 3600


def chain_batches(S, L, a, name, tail):
    """Keep the engine for the next queued batch with the same command tail: same checkpoint, same settings. Only under a card worker, whose claim file names this card. Any failed item ends the process, and the worker loads a fresh engine."""
    CL, logs = f"{HB}/claims", "/data/fd/mingdazhang/gpu-logs"
    card = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0]
    words = lambda f: open(f).read().split() if os.path.exists(f) else []
    try:
        if str(json.load(open(f"{CL}/{name}")).get("card")) != card:
            return
    except (OSError, ValueError):
        return
    want, t0, idle = " ".join(tail), time.time(), time.time()
    done = 0
    while time.time() - idle < STAY_IDLE and time.time() - t0 < STAY_MAX and not os.path.exists(f"{CL}/STOP"):
        allow, low = tuple(words(f"{CL}/ALLOW")), tuple(words(f"{CL}/LOW"))
        lines = []
        for f in (f"{vh.W}/evoq.txt", f"{vh.W}/evoq_extra.txt"):
            if os.path.exists(f):
                lines += [l.rstrip("\n").split("\t", 1) for l in open(f) if "\t" in l]
        took = None
        for n, cmd in sorted((x for x in lines if x[0].startswith(allow)), key=lambda x: (x[0].startswith(low), "_gb" not in x[0])):
            m = re.fullmatch(r"python -u rsi/hboost\.py job-batch (\S+) (\S+) (.*)", cmd)
            if not m or m.group(2) != n or m.group(3) != want or os.path.exists(f"{CL}/{n}"):
                continue
            try:
                fd = os.open(f"{CL}/{n}", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                continue
            os.write(fd, json.dumps(dict(card=card, t=time.time(), pid=os.getpid(), chained=name)).encode())
            os.close(fd)
            took = (n, m.group(1))
            break
        if not took:
            time.sleep(5)
            continue
        n, spec = took
        t1 = time.time()
        print(f"chained {n}", flush=True)
        ok = False
        try:
            its = json.load(open(spec))
            for it in its:
                run_item(S, L, a, it)
            ok = not any(os.path.exists(it["out"] + ".failed") for it in its)
        except Exception:
            import traceback
            traceback.print_exc()
        open(f"{logs}/selfedit-{n}.log", "a").write(f"ran inside {name}, engine kept; {time.time() - t1:.0f}s; ok {ok}\n")
        open(f"{logs}/selfedit-{n}.rc", "w").write("0\n" if ok else "1\n")
        if not ok:
            print(f"chained {n} failed: leaving, the worker loads a fresh engine", flush=True)
            return
        done += 1
        idle = time.time()
    print(f"chain done: {done} batches on one engine", flush=True)


NARROW = .20                 # a narrow member changes at most this share of the score rows. Coordinator 11-07 asked about .30 or what fits real coverage: r3 noweights ran .15 to .91, median .26, and .30 passes half the programs
WORKER_IDLE = 300            # cold start is about 106 s; an idle worker holds a whole card


def cmd_job_worker(argv):
    """Stays up: takes evaluations and proposer requests from the inbox. Nothing stays applied between items, so a proposer request meets the frozen stock weights."""
    import threading
    wd, name = argv[0], argv[1]
    def other():
        """Another worker on this directory: a fresh heartbeat, or a load begun in the last five minutes."""
        if os.path.exists(f"{wd}/hb") and time.time() - os.path.getmtime(f"{wd}/hb") < 120:
            return "heartbeat"
        x = jload(f"{wd}/loading.json", {})
        return f"loading {x['name']}" if x.get("name") not in (None, name) and time.time() - x.get("t", 0) < 300 else None
    if other():
        print(f"worker {name} leaves: another worker holds this directory ({other()})", flush=True)
        return
    tmp = f"{wd}/loading.{name}.tmp"
    jdump(tmp, dict(name=name, t=time.time()))
    os.replace(tmp, f"{wd}/loading.json")
    leaving = threading.Event()
    def beat():
        while True:
            if not leaving.is_set():
                open(f"{wd}/hb", "w").write(str(time.time()))
            time.sleep(30)
    S, L, a, cold = engine_up(argv[2:], f"{wd}/run")
    assert os.path.realpath(a.model) == os.path.realpath(vh.STOCK), "the proposer is the stock checkpoint"
    jdump(f"{HB}/timing/{name}.json", dict(cold=cold, items=0))
    print(f"worker up, cold start {cold:.0f}s", flush=True)
    threading.Thread(target=beat, daemon=True).start()
    idle = time.time()
    os.makedirs(f"{wd}/done", exist_ok=True)
    def rank(x):                   # a file can go between the glob and the read
        return (0 if jload(x, {}).get("kind") == "gen" else 1, x)
    while True:
        try:
            fs = sorted(glob.glob(f"{wd}/inbox/*.json"), key=rank)
        except (OSError, ValueError):
            time.sleep(1)
            continue
        if not fs:
            if time.time() - idle < WORKER_IDLE and not os.path.exists(f"{wd}/STOP"):
                time.sleep(5)
                continue
            # leaving: no heartbeat first, so a driver stops sending here at once instead of for the two minutes the last beat stays fresh; then one more look at the inbox
            leaving.set()
            time.sleep(.1)
            try:
                os.replace(f"{wd}/hb", f"{wd}/hb.left")      # a rename, nothing is deleted on the server: readers see no hb
            except OSError:
                pass
            if not glob.glob(f"{wd}/inbox/*.json") or os.path.exists(f"{wd}/STOP"):
                break
            leaving.clear()
            open(f"{wd}/hb", "w").write(str(time.time()))
            idle = time.time()
            continue
        taken = f"{wd}/done/{os.path.basename(fs[0])}"
        try:
            os.replace(fs[0], taken)                     # the claim: of two workers one gets the file, the other moves on
        except FileNotFoundError:
            continue
        open(f"{wd}/busy", "w").write(fs[0])
        run_item(S, L, a, jload(taken))
        try:
            os.replace(f"{wd}/busy", f"{wd}/busy.idle")  # a rename, nothing is deleted on the server: readers see no busy
        except FileNotFoundError:
            pass
        idle = time.time()
    print("worker done", flush=True)


def throughput(members, since, jobs):
    """Rows per hour on one card while generating, from each item's own clock, for items finished since the round began; cold start of this round's engines."""
    n, sec = 0, 0.
    for m in members:
        d = f"{EVR}/ev/{m['mid'] if m.get('program') else vh.gid(m['genome'])}"
        for f in glob.glob(f"{d}/*.time"):
            if os.path.getmtime(f) >= since and (not f.endswith(".jsonl.time") or gen_ok(f[:-5])):
                t = json.load(open(f))
                n += t["n"]
                sec += t["secs"]
    cold = [jload(f"{HB}/timing/{j}.json", {}).get("cold") for j in jobs]
    cold = [c for c in cold if c]
    return (3600 * n / sec if sec else float("nan")), n, (sum(cold) / len(cold) if cold else float("nan")), len(cold)


# ---------------------------------------------------------------- members ------------------------------------------------------------------------
def member(genome, program=None, name=None, origin="seed", **kw):
    g = dict(vision=genome.get("vision"), engine=genome.get("engine") or [], dirs=genome.get("dirs") or [])
    mid = sha(vh.gid(g) + (sha(program) if program else ""))[:14]
    return dict(name=name or mid[:8], mid=mid, genome=g, program=program, origin=origin, **kw)


def describe(m, code=True):
    g = m["genome"]
    ops = []
    for e in g["engine"]:
        ops.append(f"{e['op']} {e.get('param', '')}" + (f" alpha {e['alpha']}" if "alpha" in e else ""))
    s = f"`{m['name']}`: vision {canon(g['vision']) if g['vision'] else 'stock (patch 16)'}; {len(g['engine'])} engine edits" + (" (" + "; ".join(ops[:8]) + ")" if ops else "")
    if m.get("note"):
        s += f"; gated note: {m['note']}"
    if m.get("inherited_gates"):
        s += "; carries gate notes on: " + ", ".join(x["scope"][4:] for x in m["inherited_gates"])
    if m.get("ops"):
        s += f"; operators so far: {','.join(m['ops'])}"
    if code and m.get("program"):
        s += "\nits program:\n```python\n" + m["program"][:3000] + "\n```"
    elif code:
        s += "; no program"
    return s


def cross(a, b):
    ga, gb = a["genome"], b["genome"]
    if ga.get("vision") and gb.get("vision") and canon(ga["vision"]) != canon(gb["vision"]):
        raise Bad("crossover: the parents have different vision settings")
    ua = vh.units(ga["engine"]); ub = vh.units(gb["engine"])
    for k in ua:
        if ua[k] & ub.get(k, set()):
            raise Bad(f"crossover: units shared at {k}")
    en = list(ga["engine"]) + [e for e in gb["engine"] if canon(e) not in {canon(x) for x in ga["engine"]}]
    return dict(vision=ga["vision"] or gb["vision"], engine=en, dirs=list(ga.get("dirs") or []) + [x for x in (gb.get("dirs") or []) if x not in (ga.get("dirs") or [])])


def apply_edit(g, change):
    c = dict(vision=change.get("vision", g["vision"]), engine=list(change.get("replace", g["engine"])), dirs=list(g.get("dirs") or []))
    c["engine"] += change.get("add", [])
    x = change.get("add_dir")
    if x:
        for L in range(x["first"], x["last"] + 1):
            c["engine"].append(vh.direction_edit(x["name"], x["kind"], L, x["writer"], x["alpha"]))
        c["dirs"].append(x)
    return c


OP_SLOTS = ["gate"] * 4 + ["patch", "patch", "tokens", "tokens", "direction", "direction", "scale", "scale", "tune", "verbosity", "cross", "cross"]     # at most 4 notes of 16; the proposer picks values and targets


GRID_ROWS = 64
HIGH_D = 1000                # free parameters at or above this: a high-dimensional edit


def grid_edits(parent, rng):
    """Values to try for every operator that has them, on one parent. The proposer still chooses; these only measure."""
    g, E = parent["genome"], []
    for L in rng.choice([23, 31, 39, 47, 55, 59, 61, 63], 3, replace=False):
        for w in ("mlp", "attn"):
            E.append(dict(op="scale", layer=int(L), writer=w, alpha=float(rng.choice([.7, .85, 1.15, 1.3]))))
    for n in vh.DIRECTIONS:
        first = int(rng.choice([40, 48, 56]))
        for sign in (1, -1):
            E.append(dict(op="direction", name=n, first=first, last=first + 7, writer="mlp", alpha=sign * float(rng.choice([.5, 1, 2]))))     # alpha 4 collapsed answers in main's r2 grid (acc .047, .266); proposers keep their +-8
    for n in rng.choice(vh.VERBOSE, 2, replace=False):
        for sign in (1, -1):
            E.append(dict(op="verbosity", name=str(n), first=56, last=63, writer="mlp", alpha=sign * float(rng.choice([.5, 1, 2]))))
    E += [dict(op="patch", patch=p, kernel=k) for p, k in ((4, "bilinear"), (5, "bicubic"), (8, "box"), (11, "box"))]
    E += [dict(op="tokens", max=x, min=y) for x, y in ((1024, 0), (2048, 0), (4096, 3600), (8192, 2048))]
    E += [dict(op="tune", edit=i, factor=f) for i in range(min(2, len(g["engine"]))) for f in (.5, 1.5)]
    out, seen = [], set()
    for k, e in enumerate(E):
        try:
            change, d, _ = vh.validate(dict(e, scope="all"), dict(g), set())
            m = with_gates(member(apply_edit(g, change), parent.get("program"), name=f"g{k}_{e['op']}", origin="grid"), [parent])
        except (Bad, KeyError, ValueError, IndexError, TypeError):
            continue
        if m["mid"] not in seen and vh.gid(m["genome"]) != vh.gid(g):
            seen.add(m["mid"])
            out.append((e, int(d), m))
    return out


GRID_TOP = 8
ANCHOR_SHARE, GRID_MIN_SHARE = .8, .5      # a grid anchor answers at least this share of the grid's rows; a grid or cell under the second is an error, not a zero
NEW_CELLS = [dict(op="skip", layer=60, writer="mlp"), dict(op="attn_temp", first=47, last=63, alpha=1.3), dict(op="vision_scale", alpha=1.1)]        # one cell each of the operators added after r1


def answered_share(v, ids):
    """Share of these rows on which a member has a readable answer in a sample."""
    return sum(1 for i in ids if any(s.get("key") is not None for s in v.get(i, {}).values())) / max(1, len(ids))


def top_cells(rows, n):
    """The best cell of every operator, then the highest zx up to n."""
    rk = sorted(rows, key=lambda x: -float(x.get("zx") or 0))
    keep, ops = [], set()
    for x in rk:
        if x["op"] not in ops:
            ops.add(x["op"]); keep.append(x)
    for x in rk:
        if len(keep) >= n:
            break
        if x not in keep:
            keep.append(x)
    return [x["edit"] for x in keep]


def grid_cells(parent, edits):
    """These values on one parent, with the same checks as grid_edits. A tune cell needs edits on the parent."""
    g, out, seen = parent["genome"], [], set()
    for k, e in enumerate(list(edits) + [dict(op="tune", edit=0, factor=f) for f in (.5, 1.5) if g["engine"]]):
        try:
            change, d, _ = vh.validate(dict(e, scope="all"), dict(g), set())
            m = with_gates(member(apply_edit(g, change), parent.get("program"), name=f"g{k}_{e['op']}", origin="grid"), [parent])
        except (Bad, KeyError, ValueError, IndexError, TypeError):
            continue
        if m["mid"] not in seen and vh.gid(m["genome"]) != vh.gid(g):
            seen.add(m["mid"])
            out.append((e, int(d), m))
    return out


def weak_op(hist, ops, plan=()):
    """The operator with the fewest slots so far, this round included."""
    c = collections.Counter(o for p in hist if not p.get("injected") for o in (p.get("ops") or [])) + collections.Counter(plan)
    return min(sorted(ops), key=lambda o: c[o])


OP_CAP = 4       # from main's round 3: no grid operator holds more than this of the grid slots; a 64-row grid correlates about .45-.5 with held-out gain


def changes_network(m):
    """Weights, vision or a gate differ from the stock checkpoint."""
    g = m["genome"]
    return bool(g.get("vision") or g.get("engine") or m.get("note"))


def write_counts(write_rows, n_done, recites):
    """A1 labels of a round's gate writes. wr = (writes that passed finish_gate, writes, drops for not reaching unseen problems); rc = (recited >= .5, recites measured, their median, recited < .5, n/a: no sample of the key asked)."""
    wr = (n_done, len(write_rows), sum(1 for row in write_rows if "did not reach" in str(row.get("status"))))
    got = [x for x in recites if x is not None]
    rc = (sum(1 for x in got if x >= .5), len(got), float(np.median(got)) if got else None, sum(1 for x in got if x < .5), sum(1 for x in recites if x is None))
    return wr, rc


def split_ops(split, best, slots, rng, cap, can_cross, hist=None, gate_cap=None):
    """Fixed split of the slots by kind; grid slots in proportion to the operators' fit, under the cap. Until the pool has a weight member and a program member, or two weight members, nothing real can be crossed: those slots go half to gates, half to programs.
    gate_cap (amendment A1): at most this many gate slots; the slots it frees join the grid slots, which grid_slots assigns by fit under the cap as always."""
    gate, cro, prog, grid = split["gate"], split["cross"], split["program"], split["grid"]
    weak = 0
    if not can_cross:
        if hist is not None and best:   # coordinator 11-07: empty cross slots go to the weak operators
            weak, cro = cro, 0
        else:
            gate, prog, cro = gate + cro // 2, prog + cro - cro // 2, 0
    if gate_cap is not None and gate > gate_cap:
        if best:
            grid += gate - gate_cap
        else:
            prog += gate - gate_cap      # no grid this round: the freed slots stay with the proposer, as programs
        gate = gate_cap
    n = grid_slots(best, grid, cap) if best else {}
    plan = ["gate"] * gate + ["cross"] * cro + ["program"] * prog + [o for o in sorted(best) for _ in range(n.get(o, 0))]
    for _ in range(weak):
        plan.append(weak_op(hist, best, plan))
    plan += ["program"] * (slots - len(plan))
    return [plan[i] for i in rng.permutation(len(plan))][:slots]


def grid_slots(best, rest, cap=None):
    """Grid slots per operator in proportion to its best fit, each at least one, none over the cap; what the cap frees goes to the next-ranked operators."""
    ops = sorted(best)
    x = np.array([best[o] for o in ops], float)
    p = np.exp(x - x.max())
    p /= p.sum()
    n = {o: 1 for o in sorted(ops, key=lambda o: -p[ops.index(o)])[:rest]}      # by fit: alphabetical order left tokens and verbosity out whenever rest was short
    for _ in range(rest - sum(n.values())):
        free = [o for o in ops if cap is None or n.get(o, 0) < cap] or ops
        o = max(free, key=lambda o: p[ops.index(o)] * rest - n.get(o, 0))
        n[o] = n.get(o, 0) + 1
    return n


def allocate_ops(best, slots, rng, cap=None):
    """Slots by residual fit plus difference from the population: up to 4 notes, 2 crossovers, the rest in proportion, each operator at least one."""
    ops = sorted(best)
    rest = slots - 6
    if not ops:
        return ["gate"] * 4 + ["cross"] * (slots - 4)
    n = grid_slots(best, rest, cap)
    plan = ["gate"] * 4 + ["cross"] * 2 + [o for o in ops for _ in range(n.get(o, 0))]
    rng.shuffle(plan)
    return plan


def extra_seeds():
    """Direction and scale edits with the best earlier records, on stock. Neither was accepted then; they join for the differences they bring."""
    out, stock = [], dict(vision=None, engine=[], dirs=[])
    for d, op, name in (("stock.astra", "direction", "seed_direction"), ("stock.random", "scale", "seed_scale")):
        x = max((t for t in jload(f"{V}/{d}/state.json")["tried"] if t["op"] == op), key=lambda t: t.get("fit") or -1)
        p = normalize_edit(x["proposal"])
        p["scope"] = p.get("scope") or "all"
        change = vh.validate(p, dict(stock), {p["scope"][4:]})[0]
        out.append(member(apply_edit(stock, change), name=name, note_origin=f"{d}: earlier fit {x.get('fit'):+.3f}, not accepted"))
    return out


# ---------------------------------------------------------------- prompts -------------------------------------------------------------------------
SYSTEM = """You improve a pool of copies of a vision-language model answering questions about images. The pool's answer is a refit signed log-linear combination of its members.
The copy you are is frozen; you edit other copies. A new member is judged by what it adds to the pool: how well its change of the answer distribution lines up with the pool's residual on questions you do not see, then confirmed on a second held-out set.
A member that is merely good but makes the same mistakes as the others adds nothing. A member that is right where the pool is wrong adds a lot, even if it is worse alone or answers only some questions.
Think about what makes this model fail on the questions shown, then make the change that addresses that cause. Be specific and honest about what you can change."""


MENU_OPS = [None]            # 1109c (fable_1109d patch 4): the operators the proposer may use; None = every menu line (code-128c)


def menu():
    lines = [l for l in vh.MENU.splitlines() if l.startswith("- ")]
    if MENU_OPS[0] is not None:
        lines = [l for l in lines if l[2:].split(":", 1)[0].strip() in MENU_OPS[0]]
    return "\n".join(lines).replace("NAMES", ", ".join(vh.DIRECTIONS)).replace("VERB", ", ".join(vh.VERBOSE))


KEEP_RIGHT = ("\n\nRows marked currently right must stay right: your change has to keep them right while it addresses the rows marked currently wrong. "
              "This card is not a random sample of the questions: its rows were chosen to be shown, so how often an answer occurs on it is no evidence of how often that answer occurs.")


def tail(parents, cats, noweights=False):
    s = "\n\n## Parents you may build on\n" + "\n".join(describe(p) for p in parents)
    if noweights:                                                # the peer arm of the thesis: the copies' networks stay as they are
        s += "\n\nThe copies' networks are fixed: you change only the harness program around them.\n\n" + API
        return s + ('\n\n## Your answer\nReason first, then ONE ```json block: {"why": "the cause you found, and which errors of the pool this member fixes", "parents": ["name"], "program": "source of harness(row)"}. A new program is required. Do not repeat what the parents already have.')
    s += "\n\n## Edits you can make to a copy's network, without training\n" + menu()
    s += ("\nA gate edit stands alone in a proposal, with \"scope\": \"cat:<category>\", category from: " + "; ".join(cats[:GATE_CATS]) + ".\n") if cats else "\nNo category is open to a gate edit this round: make another edit.\n"
    s += "If a parent carries a gate note, rows of that note's category go to the engine as the plain prompt: your program runs on all other rows.\n\n" + API
    s += ('\n\n## Your answer\nReason first, then ONE ```json block: {"why": "the cause you found, and which errors of the pool this member fixes", "parents": ["name"] (two names to cross them: edits of both, then yours), '
          '"edits": [at most 3 edits, each with its "op" field, e.g. {"op": "patch", "patch": 8, "kernel": "box"}, or []], "program": "source of harness(row), or null to keep the parent\'s"}. At least one change is required. Do not repeat what the parents already have.')
    return s


def normalize_edit(e):
    """The proposer often names an edit by its fields, or nests them under the operator's name."""
    if not isinstance(e, dict):
        raise Bad("an edit must be an object")
    e = dict(e)
    for op in vh.OPS:
        if isinstance(e.get(op), dict):
            e = {**{k: v for k, v in e.items() if k != op}, **e[op], "op": op}
            break
    if e.get("op") not in vh.OPS:
        if "note" in e: e["op"] = "gate"
        elif "patch" in e: e["op"] = "patch"
        elif "max" in e or "min" in e: e["op"] = "tokens"
        elif "name" in e: e["op"] = "direction"
        elif "factor" in e: e["op"] = "tune"
        elif "layer" in e: e["op"] = "scale"
        elif "kernel" in e: e["op"] = "kernel"
        else: raise Bad(f"edit without an operator: {canon(e)[:120]}")
    return e


def parse_answer(text):
    ms = re.findall(r"```json\s*(.*?)```", text or "", re.S)
    for t in reversed(ms):
        try:
            return json.loads(t)
        except Exception:
            pass
    k = (text or "").rfind('{"why"')
    if k >= 0:
        try:
            return json.JSONDecoder().raw_decode(text[k:])[0]
        except Exception:
            pass
    return None


# ---------------------------------------------------------------- the loop ------------------------------------------------------------------------

# ---------------------------------------------------------------- 128c judge: pure functions (c128deploy/t_c128.py tests each) ------------------------------
def merge4(a, b):
    """Votes with samples 0,1 of a and samples 0,1 of b as samples 2,3: a 4-sample read from two 2-sample answer sets with different seeds. A row only b has keeps b's two."""
    out = {i: dict(s) for i, s in a.items()}
    for i, s in b.items():
        o = out.setdefault(i, {})
        for k, v in (s or {}).items():
            if int(k) <= 1:
                o[2 + int(k)] = v
    return out


def maj_col(tab, k, s, need=1):
    """Plurality column of samples s (sample -> vote) on Table row k, as Table.module's pred: OTHER and unparsed samples do not count, ties go to the lower column. -1 if no sample counts or the top count is below need."""
    cols = [tab.column(k, v) for v in (s or {}).values() if v is not None]
    cols = [c for c in cols if c is not None and not (tab.other[k] >= 0 and c == tab.other[k])]
    if not cols:
        return -1
    cnt = np.bincount(cols, minlength=len(tab.letters[k]))
    c = int(np.argmax(cnt))
    return c if cnt[c] >= need else -1


def answer_of(tab, F):
    """Column answered on every Table row, as Table.acc reads F; -1 for no answer."""
    F = np.where(tab.mask, F, -np.inf).copy()
    r = np.arange(len(tab.ids))
    ot = tab.other >= 0
    F[r[ot], tab.other[ot]] = -np.inf
    return np.where(getattr(tab, "novote", np.zeros(len(tab.ids), bool)) | ~np.isfinite(F.max(1)), -1, F.argmax(1))


def pool_cols(fu):
    """The pool's answer per Table row; on FIT rows the answer held out by picture (F_oof), so a FIT row's pool answer has not seen its own gold."""
    a = answer_of(fu.tab, fu.F)
    fit = np.asarray(fu.sp.fit, int)
    if getattr(fu, "F_oof", None) is not None and len(fit):
        G = np.array(fu.F, float, copy=True)
        G[fit] = fu.F_oof
        a[fit] = answer_of(fu.tab, G)[fit]
    return a


def right_rate(s):
    """Share of right samples among the samples present (a NO_RUN or unparsed sample is wrong, as everywhere); None if none."""
    xs = [bool(v["ok"]) for v in (s or {}).values() if v is not None]
    return float(np.mean(xs)) if xs else None


def paired(vc, vp, ids):
    """Per row of ids: right rate of vc's samples minus vp's. Rows where either has no sample (abstained, not run) are left out."""
    out = {}
    for i in ids:
        a, b = right_rate(vc.get(i)), right_rate(vp.get(i))
        if a is not None and b is not None:
            out[i] = a - b
    return out


def fired_rows(tab, mv, pa, ids, need=1):
    """F_m(R), §4.1: rows of ids in m's scope (it answered) where m's plurality answer differs from the pool's answer without m. pa: pool column per Table row."""
    out = []
    for i in ids:
        k = tab.index.get(i)
        if k is None or not mv.get(i):
            continue
        c = maj_col(tab, k, mv[i], need)
        if c >= 0 and c != pa[k]:
            out.append(i)
    return out


def vs_pool(tab, mv, pa, ids):
    """§4.3 net rows on a fired set, per row: 1[m's plurality answer is gold] - 1[the pool's answer is gold]. Summed over F it is |F| (q_m - q_a)."""
    out = {}
    for i in ids:
        k = tab.index[i]
        c = maj_col(tab, k, mv.get(i))
        out[i] = float(c >= 0 and c == tab.y[k]) - float(pa[k] >= 0 and pa[k] == tab.y[k])
    return out


def band_of(xs, k=1.65):
    """RRSI noise band of a statistic over fresh-seed re-answers of the parent: mean, SD (ddof 1) and delta = k SD. Needs two values or more."""
    x = np.asarray([v for v in xs if v is not None], float)
    if len(x) < 2:
        return dict(values=[float(v) for v in x], mean=None, sd=None, delta=None)
    sd = float(x.std(ddof=1))
    return dict(values=[float(v) for v in x], mean=float(x.mean()), sd=sd, delta=float(k * sd))


def hb_z(d):
    """HarnessBank significance on per-row paired differences: z = mean / (SD / sqrt(n)). SD 0: +-inf by the sign of the mean, 0 if the mean is 0."""
    x = np.asarray(list(d), float)
    n = len(x)
    if n == 0:
        return dict(n=0, mean=None, sd=None, z=None)
    mu = float(x.mean())
    sd = float(x.std(ddof=1)) if n > 1 else 0.
    z = mu / (sd / np.sqrt(n)) if sd > 0 else (float("inf") if mu > 0 else float("-inf") if mu < 0 else 0.)
    return dict(n=n, mean=mu, sd=sd, z=float(z))


def harm(d):
    """DarwinX no-harm statistic R = sum over rows of (-delta)+, in rows."""
    return float(sum(max(-v, 0.) for v in d.values()))


def diversity(tab, vc, vp, scope_ids, member_votes):
    """§4.3 diversity floor inputs: share of in-scope rows (both answered) where the candidate's plurality differs from its parent's; and, on in-scope rows where no member has a right sample
    ("none anywhere" among the members), how many the candidate has a right sample on."""
    both = [i for i in scope_ids if i in tab.index and maj_col(tab, tab.index[i], vc.get(i)) >= 0 and maj_col(tab, tab.index[i], vp.get(i)) >= 0]
    diff = sum(maj_col(tab, tab.index[i], vc.get(i)) != maj_col(tab, tab.index[i], vp.get(i)) for i in both)
    none = [i for i in scope_ids if not any(v is not None and v.get("ok") for mv in member_votes for v in (mv.get(i) or {}).values())]
    right = [i for i in none if any(v is not None and v.get("ok") for v in (vc.get(i) or {}).values())]
    return dict(rows=len(both), differ=int(diff), share=(diff / len(both)) if both else 0., none_anywhere=len(none), right_on_none=len(right), right_on_none_ids=right[:20])


CRITIC_SYSTEM = "You review proposed changes to the harness of a vision-language model before they are run. You judge one thing only: whether the change leaks answers."
CRITIC_ASK = ("Below is one proposed change: edits to the model, a note to be written into it, and/or a harness program that rewrites the question or the image. "
              "Does it encode a specific answer, count, option letter, value or row of particular questions (something that is right only for questions its author has seen), rather than a general procedure? "
              "Reply on the first line with exactly \"LEAK: yes\" or \"LEAK: no\", then one sentence naming the encoded item if there is one.")


def edit_summary(e):
    """One engine edit for the critic: every field except numeric vectors, which are shown as their length and norm (a 5120-float direction made the
    11-09 dry run's critic prompt 46001 tokens > 46000; numbers in a direction carry no answer text)."""
    out = {}
    for k, v in e.items():
        if isinstance(v, (list, tuple)) and len(v) > 8 and all(isinstance(x, (int, float)) for x in np.ravel(np.array(v, dtype=object))):
            a = np.array(v, dtype=float)
            out[k] = f"<vector {list(a.shape)} norm {float(np.linalg.norm(a)):.3g}>"
        else:
            out[k] = v
    return out


def critic_text(m):
    """What the leakage critic is shown: the change itself, not its parent's; engine edits as field summaries (edit_summary), capped at 4000 chars."""
    g = m["genome"]
    eng = canon([edit_summary(e) for e in g.get("engine")]) if g.get("engine") else "none"
    out = [f"Operators: {', '.join(m.get('own_ops') or m.get('ops') or []) or 'none'}", f"Vision setting: {canon(g.get('vision')) if g.get('vision') else 'stock'}",
           "Engine edits: " + (eng if len(eng) <= 4000 else eng[:4000] + f" ... ({len(eng)} chars)")]
    if (m.get("gate") or {}).get("note"):
        out.append(f"Note to be written (scope {m['gate'].get('scope')}): {m['gate']['note']}")
    if m.get("program"):
        out.append("Program:\n```python\n" + str(m["program"])[:6000] + "\n```")
    if m.get("why"):
        out.append(f"Author's reason: {str(m['why'])[:600]}")
    return "\n".join(out)


def critic_verdict(text):
    """'yes', 'no' or None from the critic's first LEAK: line."""
    x = re.search(r"LEAK:\s*(yes|no)\b", str(text or ""), re.I)
    return x.group(1).lower() if x else None


class Run:
    def __init__(self, work):
        self.work = work
        set_run(work)
        self.a = jload(f"{work}/prereg.json")
        self.data = Data(self.a.get("limit"))
        self.ev = Ev(work, self.data)
        self.ev.use_worker = True
        if self.a.get("batch_rows"):          # short parts, so idle cards share a member's rows: one 323-row batch on one card held r2 for 34 min
            self.ev.chunk = self.ev.batch_cap = self.a["batch_rows"]
        self.ev.gpu_util = self.a.get("gpu_util")
        self.ev.node_copy_after = self.a.get("node_copy_after")
        self.state = jload(f"{work}/state.json", dict(round=0, members=self.a["seeds"], rounds=[], hist=[]))
        self.F = 1 if self.a.get("plain") else 2      # a plain start has every rule on from round 1: there is no biased first round to set apart
        import res_fuse
        self.fuser = res_fuse
        self.fusion = res_fuse.mode_of(self.a)        # stacking: nothing below calls it, the driver's own path stands
        res_fuse.VOTED_LETTERS = bool(self.a.get("ds_voted_letters"))      # 128 prereg: ds_agree answers an mc row only with a voted letter
        self.narrow_rule = bool(self.a.get("narrow_pool_rows"))
        if self.narrow_rule and self.fusion != "ds_agree":
            raise SystemExit("prereg narrow_pool_rows is a test on the ds_agree pool: set fusion ds_agree")
        self.plateau_on = bool(self.a.get("plateau"))
        if self.plateau_on and self.fusion != "ds_agree":
            raise SystemExit("prereg plateau reads the ds_agree pool: set fusion ds_agree")
        self.fz_t = {}
        self.c128_init()
        self.c9 = None
        if (self.a or {}).get("c1109"):           # 1109c evolving harness (harness/adapters/hw_round.py); absent block = code-128c unchanged
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from harness.adapters import hw_round as HW
            self.HW = HW
            HW.init(self, sys.modules[__name__])

    def c128_init(self):
        """prereg c128 switches that hold for the whole run (scope and leaf change what a vote is, so they are not per round): SCOPE_ON, and fusion ds_leaf on top of ds_agree."""
        c = (self.a or {}).get("c128") or {}
        bad = sorted(set(c) - set(C128_FIELDS) - {"note"})
        if bad:
            raise SystemExit(f"prereg c128: unknown fields {bad}; known {sorted(C128_FIELDS)}")
        SCOPE_ON[0] = bool(c.get("scope"))
        if c.get("judge") and self.fusion not in DS_MODES:
            raise SystemExit("prereg c128 judge reads the fused pool: set fusion ds_agree")
        if c.get("leaf"):
            if not c.get("scope"):
                raise SystemExit("prereg c128 leaf needs scope: the specialists are the scoped weight members")
            if self.fusion != "ds_agree":
                raise SystemExit("prereg c128 leaf sits on ds_agree: set fusion ds_agree")
            import fus_leaf
            fus_leaf.LAMBDA = float(c.get("leaf_lambda", fus_leaf.LAMBDA))
            self.fusion = "ds_leaf"

    def stage(self, r, name):
        """Stage marks and, once a minute, cards free on Vesta: dry/idle_table.py joins them with the node's log for idle card-minutes per stage."""
        with open(f"{self.work}/stages.jsonl", "a") as f:
            f.write(json.dumps(dict(t=time.time(), round=r, stage=name)) + "\n")
        if getattr(self, "sampling", False):
            return
        self.sampling = True
        def go():
            while True:
                try:
                    o = subprocess.run("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits", shell=True, capture_output=True, text=True, timeout=60).stdout.split()
                    with open(f"{self.work}/idle_vesta.jsonl", "a") as f:
                        f.write(json.dumps(dict(t=time.time(), free=sum(1 for x in o if x.isdigit() and int(x) < 8000), cards=len(o))) + "\n")
                except Exception:
                    pass
                time.sleep(60)
        threading.Thread(target=go, daemon=True).start()

    def say(self, s):
        line = time.strftime("%H:%M:%S ") + s
        print(line, flush=True)
        open(f"{self.work}/run.txt", "a").write(line + "\n")

    def save(self):
        jdump(f"{self.work}/state.json", self.state)

    def votes(self, members):
        return {m["name"]: load_member(m) for m in members}

    def ensure(self, members, ids):
        for m in members:
            self.ev.build(m["genome"])
        self.ev.wait(self.say)
        self.ev.evaluate_many([(m, ids) for m in members])
        self.ev.wait(self.say)
        for m in members:
            lost = self.ev.dead_for(m, own_rows(m, ids))
            if lost:
                self.ev.write_failed(m, lost, self.say)

    def stop_dead(self, pairs, rows_of):
        """Who needs rows of an item that failed twice stops here with the reason. It is never scored on what landed."""
        for m, row in pairs:
            lost = self.ev.dead_for(m, rows_of(m, row))
            if lost and row.get("valid"):
                row.update(valid=False, status=f"item failed twice, {len(lost)} rows, e.g. {lost[0]}: stopped, not scored on partial rows")
                self.say(f"  {m['name']}: {row['status']}")

    def compute(self, members, extra=()):
        allm = list(members) + list(extra)
        votes = self.votes(allm)
        tab = Table(self.data, self.data.learn, votes)
        pool = tab.pool(self.data, allm, votes)
        sp = splits(self.data, tab)
        st = state_of(pool, sp, members)
        return votes, tab, pool, sp, st

    def accept_in_order(self, members, top, r):
        """Amendment 7. Passers of every gate go in by their pool-increment LB on SELECT, highest first. After each one the pool is refit and every remaining passer is read again against it: it stays only with SELECT dacc > 0 and LB > 0.
        The refit fits on the score rows, the only FIT rows an accepted member has answers on before its bring-up. 0 to all passers per round; each step is logged."""
        from vrsi27.boost.edge import learn_gain
        judge = self.c128_replaces(r)        # 128c: one judge for wide and narrow members; the narrow rule's own route is off
        left = sorted([(m, row) for m, row in top if row.get("would_accept") and (judge or not (self.narrow_rule and row.get("narrow")))], key=lambda x: -float(x[1]["ce_gain_select_lb"]))
        chosen, steps = [], []
        while left:
            m, row = left.pop(0)
            chosen.append(m)
            row["accepted"] = True
            row["accept_order"] = len(chosen)
            lb, da = row.get("lb_now", row.get("ce_gain_select_lb")), row.get("dacc_now", row.get("dacc_select"))
            steps.append(dict(step=len(chosen), accepted=m["name"], lb=lb, dacc=da, against=[x["name"] for x in members + chosen[:-1]]))
            self.say(f"r{r}: accept {len(chosen)}: {m['name']} pool LB {lb} dacc {da}")
            if not left:
                break
            now = members + chosen
            votes = self.votes(now + [x for x, _ in left])
            tab = Table(self.data, self.data.learn, votes)
            P = tab.pool(self.data, now + [x for x, _ in left], votes)
            sp = splits(self.data, tab)
            sp = dataclasses.replace(sp, fit=sp.score)
            st = state_of(P, sp, now)
            fu0 = self.fz("fuse", self.fuser.fuse, now, votes, self.data, self.fusion, sp=sp) if self.fusion != "stacking" else None
            stay = []
            for x, rw in left:
                g = learn_gain(P, sp, st, P.modules.index(x["name"]), combiner="evidence", parent=REF(now))
                beside = {}
                if fu0 is not None:
                    beside = dict(lb_stacking=g.get("ce_gain_select_lb"), dacc_stacking=g.get("dacc_select"))
                    g = self.fz("increment", self.fuser.increment, now, x, votes, self.data, self.fusion, sp=sp, base=fu0)
                l2, d2 = float(g.get("ce_gain_select_lb") or -1), float(g.get("dacc_select") or 0)
                keep = d2 > 0 and gain_positive(g)
                steps[-1].setdefault("reread", []).append(dict(name=x["name"], lb=l2, dacc=d2, stays=keep, **beside))
                rw.setdefault("reread_after", []).append(dict(after=m["name"], lb=l2, dacc=d2, stays=keep, **beside))
                rw.update(lb_now=l2, dacc_now=d2)
                self.say(f"r{r}:   after {m['name']}: {x['name']} pool LB {l2:.4f} dacc {d2:+.4f} -> {'stays' if keep else 'drops'}" + (f" | stacking lb {beside['lb_stacking']:.4f}" if beside else ""))
                if keep:
                    stay.append((x, rw))
                else:
                    rw["dropped_after"] = m["name"]
            left = sorted(stay, key=lambda y: -y[1]["lb_now"])
        if self.narrow_rule and not judge:
            self.accept_narrow(members, top, r, chosen, steps)
        return chosen, steps

    def pool_rows(self, now, x, votes, sp, base=None):
        """The fused pool with x against without it, every SELECT row: rows up, rows down, and what x itself says on the rows that went up. The two fits read the same votes, so the same Table."""
        f0 = base if base is not None else self.fz("fuse", self.fuser.fuse, now, votes, self.data, self.fusion, sp=sp)
        f1 = self.fz("fuse", self.fuser.fuse, list(now) + [x], votes, self.data, self.fusion, sp=f0.sp)
        a, b = [np.array([f.acc(np.array([k])) for k in f0.sp.select]) for f in (f0, f1)]
        return int((b > a).sum()), int((b < a).sum()), self.up_kinds(x, votes, f0.tab, f0.sp.select[b > a])

    def up_kinds(self, x, votes, tab, rows):
        """x's own samples on pool rows that went up, on the votes the fusion reads (a gate member: its scope only). up_direct every answered sample right, up_split some, up_wrong none, up_silent no answer. Only the first two are x answering the row."""
        mv = self.fuser.fusion_votes([x], votes)[0][x["name"]] if self.fusion != "stacking" else votes[x["name"]]
        n, detail = collections.Counter(), []
        for k in rows:
            i = tab.ids[int(k)]
            got = [v for v in (mv.get(i) or {}).values() if v is not None and v.get("key") is not None]
            rate = sum(bool(v.get("ok")) for v in got) / len(got) if got else None
            kind = "up_silent" if rate is None else "up_direct" if rate == 1 else "up_wrong" if rate == 0 else "up_split"
            n[kind] += 1
            detail.append(dict(id=i, keys=[v["key"] for v in got], ok=rate, kind=kind))
        return dict(up_direct=n["up_direct"], up_split=n["up_split"], up_wrong=n["up_wrong"], up_silent=n["up_silent"], up_detail=detail)

    def set_gain(self, f0, f1):
        """f1 against f0 on SELECT, the lines of res_fuse.increment, for a set of members."""
        from vrsi27.boost.edge import BOOT_B, BOOT_Q, boot_lower, row_ce
        sel, y, mk = f0.sp.select, f0.tab.y, f0.tab.mask
        d = row_ce(f0.F[sel], y[sel], mk[sel]) - row_ce(f1.F[sel], y[sel], mk[sel])
        a, b = [np.array([f.acc(np.array([k])) for k in sel]) for f in (f0, f1)]
        return dict(ce_gain_select=float(d.mean()), ce_gain_select_lb=boot_lower(d, BOOT_Q, BOOT_B, 0), dacc_select=f1.acc(sel) - f0.acc(sel), rows_up=int((b > a).sum()), rows_down=int((b < a).sum()), net_rows=int((b > a).sum() - (b < a).sum()))

    def fit_rows_read(self, allm):
        """Votes and rows of a read at the round's end. FIT is the score rows: what a member outside the pool has answered, as accept_in_order."""
        votes = self.votes(allm)
        sp = splits(self.data, Table(self.data, self.data.learn, votes))
        return votes, dataclasses.replace(sp, fit=sp.score)

    def set_increment(self, before, chosen):
        """What the round's accepted members together add on SELECT: the plateau test of the next round."""
        votes, sp = self.fit_rows_read(list(before) + list(chosen))
        f0 = self.fz("fuse", self.fuser.fuse, before, votes, self.data, self.fusion, sp=sp)
        f1 = self.fz("fuse", self.fuser.fuse, list(before) + list(chosen), votes, self.data, self.fusion, sp=f0.sp)
        return self.set_gain(f0, f1)

    def member_loo(self, members, votes, sp):
        """Leave-one-out gain of each member, held-out FIT CE (F_oof) of the pool without it minus with it. Positive helps. Report and parent score; prunes nothing."""
        from vrsi27.boost.edge import row_ce
        if len(members) < 2:
            return {m["name"]: 0. for m in members}
        f0 = self.fz("fuse", self.fuser.fuse, members, votes, self.data, self.fusion, sp=sp)
        fit = f0.sp.fit
        ce = lambda f: row_ce(f.F_oof, f0.tab.y[fit], f0.tab.mask[fit])
        c0, out = ce(f0), {}
        for m in members:
            out[m["name"]] = float((ce(self.fz("fuse", self.fuser.fuse, [x for x in members if x is not m], votes, self.data, self.fusion, sp=f0.sp)) - c0).mean())
        return out

    def fusion_side(self, r, rd, fu, members, votes):
        """Report only: ds_agree_scale and beta_m,c beside the pool, in a process of its own (rsi/fus_side.py), rNNN/fusion_side.json. The driver neither waits for it nor reads it for anything but the log line of the next round."""
        V = self.fuser.sample_columns(fu.tab, members, self.fuser.fusion_votes(members, votes)[0])
        np.savez_compressed(f"{rd}/fusion_side_in.npz", V=V, y=fu.tab.y, mask=fu.tab.mask, other=fu.tab.other, names=np.array([m["name"] for m in members]), ids=np.array(fu.tab.ids),
                            bench=np.array([self.data.rows[i].get("bench") or i.split("/")[0] for i in fu.tab.ids]), cluster=np.array([str(self.data.cluster(i)) for i in fu.tab.ids]), mc=np.array([self.data.rows[i].get("kind") == "mc" for i in fu.tab.ids]),
                            cells=self.fuser.fine_cells(self.data, fu.tab.ids), fit=np.asarray(fu.sp.fit, int), select=np.asarray(fu.sp.select, int))
        here = os.path.dirname(os.path.abspath(__file__))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4")
        subprocess.Popen(["nice", "-n", "15", sys.executable, "-u", f"{here}/fus_side.py", f"{rd}/fusion_side_in.npz", f"{rd}/fusion_side.json"], stdout=open(f"{rd}/fusion_side.log", "w"), stderr=subprocess.STDOUT, env=env, start_new_session=True)
        self.say(f"r{r}: fusion side lines started, report only: {rd}/fusion_side.json")

    def fusion_side_read(self, r):
        """The log line of an earlier round's side lines, when they are there."""
        x = jload(f"{self.work}/r{r:03d}/fusion_side.json", None) if r >= 1 else None
        if not x:
            return
        s, b = x["ds_agree_scale"], x["beta_mc"]
        self.say(f"r{r} side lines (report only): ds_agree T {x['ds_agree']['T']:.3f}; scale rho {s['rho']:.3f} T {s['T']:.3f} held-out FIT dCE {s['held_fit']['dce']:+.4f} lb {s['held_fit']['dce_lb20']:+.4f}, SELECT dCE {s['select']['dce']:+.4f} lb {s['select']['dce_lb20']:+.4f} up {s['select']['up']} down {s['select']['down']}; beta_m,c held-out FIT dCE {b['held_fit']['dce']:+.4f} lb {b['held_fit']['dce_lb20']:+.4f} dacc {100 * b['held_fit']['dacc']:+.2f} up {b['held_fit']['up']} down {b['held_fit']['down']}, SELECT dCE {b['select']['dce']:+.4f} lb {b['select']['dce_lb20']:+.4f} dacc {100 * b['select']['dacc']:+.2f} up {b['select']['up']} down {b['select']['down']}; beta_m " + ", ".join(f"{k} {v:.2f}" for k, v in b["beta"].items()))

    def plateau_inc(self, inc):
        """A SELECT increment that makes a plateau: LB <= 0, or net SELECT rows <= 0."""
        return not gain_positive(inc) or int(inc.get("net_rows", 1)) <= 0

    def plateau_stage2(self, r):
        """The extra stage 2 slots follow the same plateau state as the raised proposal slots."""
        return self.plateau_now(r)

    def held_rows(self, f, rows):
        """Per-row accuracy on FIT rows, held out by picture. Only rows of f.sp.fit: a SELECT row raises."""
        fit = set(int(k) for k in f.sp.fit)
        if not all(int(k) in fit for k in rows):
            raise RuntimeError("held_rows: a row outside FIT")
        F = f.F.copy()
        F[f.sp.fit] = f.F_oof
        return np.array([f.tab.acc(F, np.array([k])) for k in rows])

    def own_finish(self, m, rows):
        """The member's records on rows, samples 0 and 1: answered share, too_long rate and not-stopped rate per sample."""
        want, seen = set(rows), {}
        for d in member_dirs(m):
            for f in answer_files(d):
                for l in open(f):
                    x = json.loads(l)
                    if x["id"] in want and x.get("sample", 0) < 2:
                        seen.setdefault((x["id"], x.get("sample", 0)), x.get("finish"))
        abst = set()
        for a_ in glob.glob(f"{EVR}/prog/{m['mid']}/abstain.*.json"):
            abst |= set(json.load(open(a_))) & want
        run = sorted(want - abst)
        v_ = load_member(m)
        n = max(1, len(seen))
        return dict(rows=len(want), abstained=len(abst), samples=len(seen), answered_all=answered_share(v_, sorted(want)), answered=answered_share(v_, run) if run else 0., too_long=sum(v in NO_RUN for v in seen.values()) / n, not_stopped=sum(v != "stop" for v in seen.values()) / n)

    def stage2_exempt(self, have, top, members, r, rd, qm):
        """Extra stage 2 slots, plateau rounds only: weight and vision ops the zx filter stops. Ranked by FIT rows only: net pool rows up minus down, with against without the candidate, ds_agree, held out by picture, on its stage 1 rows; ties by answers the pool does not have. Reads no SELECT row and no member zx."""
        ours = set(PLATEAU_OPS) | {"kernel"}
        ids_top = {id(x[1]) for x in top}
        plateau = self.plateau_stage2(r)
        log = []
        for m, row in have:
            own = row.get("own_ops") or m.get("own_ops") or []
            row["stage2_filter"] = "kept" if id(row) in ids_top else "stopped"
            log.append(dict(name=m["name"], ops=own, zx_stage1=row.get("zx_score"), filter=row["stage2_filter"], by=row.get("stage2_by")))
        cands = [(m, row) for m, row in have if id(row) not in ids_top and set(row.get("own_ops") or m.get("own_ops") or []) & ours] if plateau and self.fusion in DS_MODES else []
        out = []
        if cands:
            allm = list(members) + [m for m, _ in cands]
            votes = self.votes(allm)
            tab = Table(self.data, self.data.learn, votes)
            sp = splits(self.data, tab)
            f0 = self.fz("fuse", self.fuser.fuse, members, votes, self.data, self.fusion, sp=sp)
            fitset = {int(k) for k in f0.sp.fit}
            cut_ = self.fuser.fusion_votes(list(members), votes)[0]        # it hands back every vote it was given: only the members are "the pool"
            base = {x["name"]: cut_[x["name"]] for x in members}
            for m, row in cands:
                q = list(qm(m))
                fin = self.own_finish(m, q)
                x = dict(name=m["name"], ops=row.get("own_ops") or m.get("own_ops"), zx_stage1=row.get("zx_score"), **fin)
                reason = None
                if fin["answered"] < EXEMPT_ANSWERED:
                    reason = f"answered {fin['answered']:.3f} of the {fin['rows'] - fin['abstained']} stage 1 rows its program does not abstain on"
                elif fin["too_long"] > EXEMPT_NOT_STOPPED:
                    reason = f"too_long {fin['too_long']:.3f}"
                elif fin["not_stopped"] > EXEMPT_NOT_STOPPED:
                    reason = f"not stopped {fin['not_stopped']:.3f}"
                rows = [int(tab.index[i]) for i in q if i in tab.index and int(tab.index[i]) in fitset]
                sel_ = {int(k) for k in f0.sp.select} & set(rows)
                if sel_:
                    raise RuntimeError(f"stage2_exempt: SELECT rows in the read: {sorted(sel_)[:5]}")
                x.update(rows_read=len(rows), select_rows_read=0, rows_asked=len(q))
                if reason is None:
                    f1 = self.fz("fuse", self.fuser.fuse, list(members) + [m], votes, self.data, self.fusion, sp=f0.sp)
                    a_, b_ = self.held_rows(f0, rows), self.held_rows(f1, rows)
                    uniq = uniq_right = 0
                    mine = votes[m["name"]]
                    for k in rows:
                        i = tab.ids[k]
                        others = {v.get("key") for b in base.values() for v in (b.get(i) or {}).values() if v is not None and v.get("key") is not None}
                        new = [v for v in (mine.get(i) or {}).values() if v is not None and v.get("key") is not None and v["key"] not in others]
                        uniq += bool(new)
                        uniq_right += any(v.get("ok") for v in new)
                    x.update(fit_up=int((b_ > a_).sum()), fit_down=int((b_ < a_).sum()), fit_net=int((b_ > a_).sum() - (b_ < a_).sum()), unique=uniq, unique_right=uniq_right)
                    if x["fit_net"] < 0 and uniq_right < 1:
                        reason = f"net FIT pool rows {x['fit_net']} and no unique right answer"
                x["floor"] = reason
                out.append((m, row, x))
            ok = sorted([y for y in out if y[2]["floor"] is None], key=lambda y: (-y[2]["fit_net"], -y[2]["unique"], y[2]["name"]))
            for m, row, x in ok[:STAGE2_EXEMPT_MAX]:
                x["slot"] = True
                row.update(stage2_exempt=dict(op=x["ops"], zx_stage1=x["zx_stage1"], fit_net=x["fit_net"], unique=x["unique"], unique_right=x["unique_right"]), stage2_by="exempt")
                m["stage2_exempt"] = r
            for m, row, x in out:
                self.say(f"r{r}: stage 2 exemption {x['name']} {x['ops']}: stage 1 zx {x['zx_stage1']} answered {x['answered']:.3f} of rows not abstained, {x['answered_all']:.3f} of all stage 1 rows ({x['abstained']} abstained) too_long {x['too_long']:.3f} not stopped {x['not_stopped']:.3f}; FIT rows read {x['rows_read']} of {x['rows_asked']}, SELECT rows read 0; net {x.get('fit_net')} (up {x.get('fit_up')} down {x.get('fit_down')}) unique {x.get('unique')} unique right {x.get('unique_right')} -> {'slot' if x.get('slot') else x['floor'] or 'over the cap'}")
        jdump(f"{rd}/stage2_filter.json", dict(plateau=plateau, candidates=log, exempt=[x for _, _, x in out]))
        self.say(f"r{r}: stage 2 zx filter: " + ", ".join(f"{x['name']} {'+'.join(x['ops'] or [])} zx {0. if x['zx_stage1'] is None else x['zx_stage1']:+.2f} {x['filter']}" for x in log) + f"; plateau {plateau}, exempted {[x['name'] for _, _, x in out if x.get('slot')]}")
        return [(m, row) for m, row, x in out if x.get("slot")]

    def grid_pool(self, r, rd, members, have, votes, tab, sp, sample, rows_for=None):
        """Report only, every weight cell of the grid: rows changed against the anchor, and pool rows up and down, ds_agree, held out by picture, on the grid's rows. FIT rows only."""
        f0 = self.fz("fuse", self.fuser.fuse, members, votes, self.data, self.fusion, sp=sp)
        fitset = {int(k) for k in f0.sp.fit}
        rows_of = lambda S: [int(tab.index[i]) for i in S if i in tab.index and int(tab.index[i]) in fitset]
        keys = lambda v, i: sorted({str(x.get("key")) for x in (v.get(i) or {}).values() if x is not None})
        out = []
        for e, d, m, an in have:
            if e["op"] not in PLATEAU_OPS:
                continue
            rows = rows_of(rows_for(e["op"]) if rows_for else sample)
            f1 = self.fz("fuse", self.fuser.fuse, list(members) + [m], votes, self.data, self.fusion, sp=f0.sp)
            a_, b_ = self.held_rows(f0, rows), self.held_rows(f1, rows)
            out.append(dict(cell=m["name"], op=e["op"], edit=e, on=an["name"], rows=len(rows), changed=sum(keys(votes[m["name"]], tab.ids[k]) != keys(votes[an["name"]], tab.ids[k]) for k in rows), pool_up=int((b_ > a_).sum()), pool_down=int((b_ < a_).sum())))
        jdump(f"{rd}/grid_pool.json", out)
        self.say(f"r{r}: grid cells, pool rows on the grid's FIT rows (report only): " + ", ".join(f"{x['cell']} changed {x['changed']} up {x['pool_up']} down {x['pool_down']}" for x in out))

    def plateau_now(self, r):
        """Round r follows a plateau: the round before accepted no member, or what it accepted adds no SELECT increment LB above zero."""
        R_ = self.state.get("rounds", [])
        if not self.plateau_on or not R_:
            return False
        last = R_[-1]
        return not last.get("accepted") or ("select_increment" in last and self.plateau_inc(last["select_increment"]))

    def plateau_at_end(self, chosen, inc):
        return not chosen or self.plateau_inc(inc)

    def plateau_end(self, members, top, chosen, steps, r, rd):
        """This round's end: its SELECT increment; on a plateau the combination test of the reserve. CPU only, answers already there."""
        t0 = time.time()
        reads = 0
        inc = self.set_increment(members, chosen) if chosen else None
        reads += bool(chosen)
        ended = self.plateau_at_end(chosen, inc)
        out = dict(select_increment=inc, plateau_end=ended, combo=None)
        self.say(f"r{r}: round's SELECT increment {inc if inc is None else {k: round(v, 5) for k, v in inc.items()}} -> plateau at this round's end: {ended}")
        if ended:
            out["combo"] = self.combo_test(members + chosen, top, r, rd)
            reads += bool(out["combo"].get("tested"))
            if out["combo"].get("accepted"):
                by = {m["name"]: (m, row) for m, row in top}
                for n_ in out["combo"]["set"]:
                    m, row = by[n_]
                    chosen.append(m)
                    m.update(new_rule_only=True, combo=r)
                    row.update(accepted=True, accept_order=len(chosen), combo=True, new_rule_only=True)
                steps.append(dict(step=len(chosen), accepted=out["combo"]["set"], rule="combination", lb=out["combo"]["ce_gain_select_lb"], dacc=out["combo"]["dacc_select"], rows_up=out["combo"]["rows_up"], rows_down=out["combo"]["rows_down"], against=out["combo"]["against"]))
                out["select_increment"] = self.set_increment(members, chosen)
                reads += 1
            out["plateau_after_combo"] = self.plateau_at_end(chosen, out["select_increment"])      # label only (A2 a): what the next round's plateau_now reads; plateau_end stays the trigger of the combination test
            self.say(f"r{r}: after the combination test: accepted {[m['name'] for m in chosen]}, SELECT increment LB {(out['select_increment'] or {}).get('ce_gain_select_lb')} net {(out['select_increment'] or {}).get('net_rows')} -> next round is a plateau round: {out['plateau_after_combo']}")
        out.update(reads=reads, secs=time.time() - t0)
        return out

    def combo_test(self, now, top, r, rd):
        """Greedy add from the reserve, up to COMBO_MAX, each step by held-out FIT (F_oof, held out by picture) on ds_agree; a step that raises nothing ends the walk. Then the one set against the pool on SELECT: LB > 0 and net rows >= COMBO_NET. One SELECT read a round. Every step and every rejected set is logged."""
        names = {m["name"] for m in now}
        res = [m for m, row in top if row.get("stage2_status") == "ok" and m["name"] not in names and not row.get("stage2_exempt") and not m.get("stage2_exempt")]      # an exempted candidate passes by the narrow or the wide gate only, never by a set
        rec = dict(round=r, reserve=[m["name"] for m in res], against=sorted(names), steps=[], set=[], tested=False, accepted=False)
        if res:
            votes, sp = self.fit_rows_read(list(now) + res)
            def held(f):
                F = f.F.copy()
                F[f.sp.fit] = f.F_oof
                return f.tab.acc(F, f.sp.fit)
            f0 = self.fz("fuse", self.fuser.fuse, now, votes, self.data, self.fusion, sp=sp)
            rec["held_fit_before"] = cur_h = held(f0)
            rec["fit_rows"] = len(f0.sp.fit)
            cur, left = list(now), list(res)
            while left and len(rec["set"]) < COMBO_MAX:
                got = sorted([(held(self.fz("fuse", self.fuser.fuse, cur + [x], votes, self.data, self.fusion, sp=f0.sp)), x["name"], x) for x in left], key=lambda y: (-y[0], y[1]))
                h, n_, x = got[0]
                raises = h > cur_h
                rec["steps"].append(dict(step=len(rec["steps"]) + 1, held_before=cur_h, tried={a_: b_ for b_, a_, _ in got}, best=n_, held_after=h, added=raises))
                self.say(f"r{r}: combination step {len(rec['steps'])}: held-out FIT {cur_h:.4f}, tried " + ", ".join(f"{a_} {b_:.4f}" for b_, a_, _ in got) + (f" -> add {n_}" if raises else " -> nothing raises it, walk ends"))
                if not raises:
                    break
                cur.append(x)
                left.remove(x)
                cur_h = h
                rec["set"].append(n_)
            rec["held_fit_after"] = cur_h
            if rec["set"]:
                g = self.set_gain(f0, self.fz("fuse", self.fuser.fuse, cur, votes, self.data, self.fusion, sp=f0.sp))
                rec.update(g, tested=True, accepted=bool(gain_positive(g) and g["net_rows"] >= COMBO_NET))
        self.say(f"r{r}: combination test: reserve {rec['reserve']}; set {rec['set']}; held-out FIT {rec.get('held_fit_before')} -> {rec.get('held_fit_after')}; SELECT LB {rec.get('ce_gain_select_lb')} ce {rec.get('ce_gain_select')} dacc {rec.get('dacc_select')} rows up {rec.get('rows_up')} down {rec.get('rows_down')} net {rec.get('net_rows')} -> {'ACCEPT' if rec['accepted'] else 'rejected' if rec['tested'] else 'no set to test'}")
        jdump(f"{rd}/combo.json", rec)
        return rec

    def weight_slots(self, r, reqs, props):
        """What the round's weight and vision slots did, and whether the round counts for the stop rule."""
        raised = self.plateau_now(r)
        ours = set(PLATEAU_OPS) | {"kernel"}
        w = [p for p in props if set(p.get("own_ops") or []) & ours]
        ran = [p["name"] for p in w if p.get("valid") and p.get("scored")]
        return dict(raised=raised, slots_asked=dict(collections.Counter(q.get("op") for q in reqs if q["id"] != "meta")), weight_proposed=len(w), weight_built=sum(1 for p in w if p.get("valid")), weight_ran=ran, counted=bool(raised and len(ran) >= PLATEAU_MIN_RAN))

    def accept_narrow(self, members, top, r, chosen, steps):
        """Passers of the narrow rule, after the wide members: by FIT net rows, highest first, at most NARROW_MAX. After any acceptance the pool is refit (on the score rows, as accept_in_order) and the next is tested again against it."""
        from vrsi27.boost.edge import learn_gain
        left = sorted([(m, row) for m, row in top if row.get("would_accept") and row.get("narrow")], key=lambda x: (-float(x[1].get("net_rows") or 0), x[0]["name"]))
        n = 0
        while left and n < NARROW_MAX:
            m, row = left.pop(0)
            now = members + chosen
            up, down = row["pool_rows_up"], row["pool_rows_down"]
            if chosen:
                votes = self.votes(now + [m] + [x for x, _ in left])
                tab = Table(self.data, self.data.learn, votes)
                sp = splits(self.data, tab)
                up, down, kinds = self.pool_rows(now, m, votes, dataclasses.replace(sp, fit=sp.score))
                row.setdefault("retest", []).append(dict(against=[x["name"] for x in chosen], up=up, down=down, **kinds))
                if not (up >= 1 and down == 0 and kinds["up_direct"] + kinds["up_split"] >= 1):
                    row["dropped_after"] = chosen[-1]["name"]
                    self.say(f"r{r}:   narrow {m['name']} after {chosen[-1]['name']}: pool rows up {up} (direct {kinds['up_direct']} split {kinds['up_split']} wrong {kinds['up_wrong']} silent {kinds['up_silent']}) down {down} -> drops")
                    continue
            chosen.append(m)
            n += 1
            row.update(accepted=True, accept_order=len(chosen), new_rule_only=not row["old_rule"])
            if row["new_rule_only"]:
                m["new_rule_only"] = True
            steps.append(dict(step=len(chosen), accepted=m["name"], rule="narrow", rows_up=up, rows_down=down, net_rows=row.get("net_rows"), old_rule=row["old_rule"], lb=row.get("ce_gain_select_lb"), dacc=row.get("dacc_select"), against=[x["name"] for x in now]))
            self.say(f"r{r}: accept {len(chosen)}: {m['name']} narrow rule, pool rows up {up} down {down}, net {row.get('net_rows')}" + (", by the new rule only" if row["new_rule_only"] else ""))
        for m, row in left:
            row["dropped_after"] = "narrow cap"
            self.say(f"r{r}:   narrow {m['name']}: past the cap of {NARROW_MAX}")

    def lens_ctx(self, pool, tab, parent):
        from vrsi27.boost.lens import StateContext
        meta = {}
        for k, i in enumerate(tab.ids):
            meta[i] = dict(question=self.data.rows[i]["prompt"], options=tab.text[k])
        ds = {i: self.data.rows[i].get("bench") or i.split("/")[0] for i in tab.ids}
        cat = {i: str(self.data.rows[i].get("category")) for i in tab.ids}
        return StateContext(parent=parent, meta=meta, dataset_of=ds, task_type_of=cat, option_aware=True)

    def guard(self, tab):
        from vrsi27.boost.lens import B2Guard
        roles = {i: self.data.role.get(i, "x") for i in tab.ids}
        for i in self.data.id_rows:
            roles[i] = "id"
        for i in self.data.ood_rows:
            roles[i] = "ood"
        meta = {i: dict(question=self.data.rows[i]["prompt"]) for i in roles}
        return B2Guard(roles, self.data.of("show"), meta, log_path=None)

    def overlap_table(self, pool, sp, members, st, parent):
        """Show rows only: which members make the parent's mistakes, which fix them. The proposer aims at what nobody fixes. A member that abstains on a row has no answer there: it is neither right nor wrong, and neither makes nor fixes an error."""
        show = sp.show
        par = pool.modules.index(parent)
        ans = pool.source[:, show] == 0
        wrong = ans[par] & ~pool.correct[par, show]
        out = ["\n## Where the members' errors overlap (shown questions only)", f"| member | answers, of {len(show)} shown | acc where it answers | of the {int(wrong.sum())} errors of `{parent}`, where it answers: also wrong | fixes |", "|---|---|---|---|---|"]
        for k, m in enumerate(members):
            sh, a_ = pool.correct[k, show], ans[k]
            both = wrong & a_
            ow = float((~sh[both]).mean()) if both.any() else float("nan")
            acc_ = f"{sh[a_].mean():.3f}" if a_.any() else "-"
            out.append(f"| {m['name']} | {int(a_.sum())} | {acc_} | {ow:.2f} | {(1 - ow if ow == ow else float('nan')):.2f} |")
        allwrong = (~pool.correct[[k for k in range(len(members))]][:, show]).all(0)
        out.append(f"{allwrong.mean():.3f} of the shown questions no member answers right (wrong or abstaining): no refit can reach those without a member that behaves differently there.")
        return "\n".join(out)

    def lineage_view(self, pool, tab, members, votes, pa, full, rid_row):
        """Where the parent abstains the pool sees no vote (edge.py:272, calibrate.py:133), but Table.module leaves a uniform distribution and correct=False. Those fields read as an unsure, wrong member. Here such a row carries the answer of the nearest ancestor that answered, and says so.
        In scope: rows the parent itself answers; for a gate member with a category scope, rows of that category. Returns the rows and a plain line with the true counts."""
        from vrsi27.boost.lens import CONF_THRESHOLDS, conf_slice_name
        kx = {m["name"]: k for k, m in enumerate(members)}
        by = {m["name"]: m for m in members}
        par = kx[pa["name"]]
        chain, x = [], pa
        while x.get("parents") and x["parents"][0] in by and x["parents"][0] not in chain:
            chain.append(x["parents"][0])
            x = by[x["parents"][0]]
        if "stock" in by and "stock" not in chain and pa["name"] != "stock":
            chain.append("stock")
        c = gate_scope(pa)
        pv, n_ans, n_right, n_all, n_abst = {}, 0, 0, 0, 0
        for x in full["rows"]:
            i = rid_row[x["rid"]]
            q = tab.ids[i]
            n_all += 1
            x["member_preds"] = {n: ("abstains" if q in votes[n] and not votes[n][q] else "no answer") if v == "?" else v for n, v in x["member_preds"].items()}
            if pool.source[par, i] == 0:
                n_ans += 1
                n_right += bool(pool.correct[par, i])
                pv[x["rid"]] = votes[pa["name"]].get(q, {})
                x.update(parent_abstains=False, in_scope=c is None or ROWKEY.get(q) == c)
                continue
            n_abst += q in votes[pa["name"]] and not votes[pa["name"]][q]
            anc = next((a for a in chain if pool.source[kx[a], i] == 0), None)
            x.update(parent_abstains=True, in_scope=False, answered_by=anc, parent_said_nothing=q in votes[pa["name"]] and bool(votes[pa["name"]][q]))
            pv[x["rid"]] = votes[anc].get(q, {}) if anc else {}
            if anc:
                k, L = kx[anc], x["letters"]
                n = len(L)
                pp = np.exp(np.where(pool.opt_mask[i], pool.logp[k, i], -np.inf))[:n]
                pp = pp / max(1e-12, pp.sum())
                cf = float(pp.max())
                pd = int(pool.pred[k, i])
                x.update(parent_probs={a: round(float(y), 4) for a, y in zip(L, pp)}, parent_pred=L[pd] if 0 <= pd < n else "?", parent_correct=bool(pool.correct[k, i]), parent_conf=round(cf, 4),
                         slices=[s for s in x["slices"] if s not in [conf_slice_name(t) for t in CONF_THRESHOLDS]] + [conf_slice_name(t) for t in CONF_THRESHOLDS if cf < t])
        full["ensemble"]["parent_show_acc"] = round(float(np.mean([x["parent_correct"] for x in full["rows"]])), 4) if full["rows"] else 0.
        out_ = n_all - n_ans
        line = None
        if out_:
            how = f"it abstains, no vote" if n_abst == out_ else (f"it gave no readable answer" if not n_abst else f"it abstains on {n_abst} and gave no readable answer on {out_ - n_abst}")
            line = f"`{pa['name']}` answers {n_ans} of the {n_all} shown questions and is right on {n_right} of those" + (f" ({n_right / n_ans:.0%})" if n_ans else "") + f". On the other {out_} {how}"
            line += f": there the card shows the answer of its nearest ancestor that answered, and any count or confidence in this card about `{pa['name']}` on those rows is that ancestor's." if chain else "."
        return pv, line

    def card_view(self, out, state, pa, line):
        """Text the lens wrote from the state: a sentence about the parent counts the ancestor's answers on rows it abstains on, so it says so. Rows of a narrow member's scope come first."""
        if not line:
            return out
        P = pa["name"]
        by = {x["rid"]: x for x in state["rows"]}
        cards = []
        for c in out["cards"]:
            s_ = re.sub(rf"`{re.escape(P)}` is right on", f"`{P}`, with its nearest answering ancestor where it abstains, is right on", c.get("summary") or "")
            cards.append(dict(c, rows=sorted(c["rows"], key=lambda x: not by[x].get("in_scope")), summary=line + ("\n" + s_ if s_ else "")))
        return dict(out, cards=cards)

    def card_text(self, state, out, pa):
        """lens.card_block with three differences: a row the parent abstains on says so and shows the nearest ancestor that answered, a narrow member's rows come under two headings, and nothing changes on any other row."""
        from vrsi27.boost.lens import _fmt
        rows = {r["rid"]: r for r in state["rows"]}
        f = set(out["fields"])
        parent = state["parent"]
        scope = (pa.get("gate") or {}).get("scope") or ""
        lines = []
        for k, c in enumerate(out["cards"], 1):
            head = "Target" if len(out["cards"]) == 1 else f"Card {k}"
            lines.append(f"\n## {head}: {c['title']}")
            if c["rule"]:
                lines.append(f"Runtime rule: {c['rule']}")
            if c["summary"]:
                lines.append(c["summary"])
            n_in = sum(1 for x in c["rows"] if rows[x].get("in_scope"))
            split_ = n_in < len(c["rows"]) and any("in_scope" in rows[x] for x in c["rows"])
            for j, rid in enumerate(c["rows"], 1):
                r = rows[rid]
                if split_ and j == 1 and n_in:
                    lines.append(f"\n### In the scope of `{parent}`: {n_in} rows" + (f" of {scope[4:]}" if scope.startswith("cat:") else " it answers"))
                if split_ and j == n_in + 1:
                    lines.append(f"\n### Outside the scope of `{parent}`: {len(c['rows']) - n_in} rows" + (f" outside {scope[4:]}; it takes its parent's answers here" if scope.startswith("cat:") else "; it abstains here"))
                ab = r.get("parent_abstains")
                title = f"example {j}"
                if "g_norm" in f:
                    title += f" (‖g‖ = {r['g_norm']:.3f})"
                lines.append(f"\n#### {title}")
                lines.append(f"- question: {r['question']}")
                for a, opt in zip(r["letters"], r["options"] or r["letters"]):
                    extra = ""
                    if "ens_probs" in f:
                        extra += f"   p_ensemble={_fmt(r['ens_probs'][a])}"
                    if "parent_probs" in f and not ab:
                        extra += f"   p_parent={_fmt(r['parent_probs'][a])}"
                    if "gold" in f and a == r["gold"]:
                        extra += " ← gold"
                    if "ens_pred" in f and a == r["ens_pred"] and not r["ens_correct"]:
                        extra += " ← ensemble favours (wrong)"
                    if "parent_pred" in f and a == r["parent_pred"] and (not ab or r.get("answered_by")):
                        extra += f" ← `{r.get('answered_by') if ab else parent}` answered" + ("" if r["parent_correct"] else " (wrong)")
                    if "prior_pred" in f and a == r.get("prior_pred"):
                        extra += " ← text-only prior"
                    lines.append(f"  - {opt}{extra}")
                if ab:
                    say_ = "gave no readable answer" if r.get("parent_said_nothing") else "abstains, no vote"
                    if r.get("answered_by"):
                        said_ = dict(zip(r["letters"], r["options"] or r["letters"])).get(r["parent_pred"], r["parent_pred"])
                        lines.append(f"- parent `{parent}`: {say_}. Nearest ancestor that answered, `{r['answered_by']}`: {said_}" + (" (right)" if r["parent_correct"] else " (wrong)") + (f", confidence {r['parent_conf']:.2f}" if "parent_conf" in f else ""))
                    else:
                        lines.append(f"- parent `{parent}`: {say_}. No ancestor answered this row.")
                elif "parent_conf" in f:
                    lines.append(f"- `{parent}` confidence (max option probability): {r['parent_conf']:.2f}")
                if "member_preds" in f and r["member_preds"]:
                    lines.append("- member answers: " + ", ".join(f"{m} {x}" for m, x in r["member_preds"].items()))
                if "slices" in f and r["slices"]:
                    lines.append("- slices: " + ", ".join(r["slices"]))
                if "evidence" in f and r.get("evidence"):
                    lines.append(f"- evidence: {r['evidence']}")
                if "note" in f and rid in out["notes"]:
                    lines.append(f"- note: {out['notes'][rid]}")
        return "\n".join(lines)

    def card_cases(self, state, cardout, pv, rid_row, tab):
        """pv: rid -> the samples the card reads, the nearest answering ancestor's where the parent abstains."""
        L = ["\n## How the parent answered these, in its own words"]
        by = {x["rid"]: x for x in state["rows"]}
        for c in cardout["cards"]:
            for rid in c["rows"][:12]:
                vs = list(pv.get(rid, {}).values())
                v = next((v for v in vs if not v["ok"]), None) or next(iter(vs), None)      # a wrong sample where there is one; a stably right row shows a right one
                ab = by[rid].get("parent_abstains")
                if v:
                    who = f" (the parent abstains; this is its ancestor `{by[rid].get('answered_by')}`)" if ab else ""
                    L.append(f"[{rid}]{who} end of thinking: {v['tail']}\n[{rid}] said: {v['said']}")
                elif ab:
                    L.append(f"[{rid}] the parent abstains and no ancestor answered.")
        return "\n".join(L)

    def prior_gains(self):
        """Gains per operator from our records, no rows and no test numbers."""
        T = collections.defaultdict(list)
        L = []
        if not self.a.get("plain"):       # a plain run starts with no history: nothing from earlier runs reaches its prompts
            for f in sorted(glob.glob(f"{V}/*/state.json")):
                for x in jload(f).get("tried", []):
                    if x.get("fit") is not None:
                        T[x["op"]].append((float(x["fit"]), bool(x.get("accepted"))))
            L = ["## What each operator has done so far", "Earlier runs, change of accuracy on the rows they could see:"]
            for o in ("gate", "patch", "tokens", "direction", "scale", "tune", "verbosity"):
                v = T.get(o)
                L.append(f"- {o}: " + (f"{len(v)} tried, median {np.median([a for a, _ in v]):+.3f}, best {max(a for a, _ in v):+.3f}, {sum(b for _, b in v)} accepted" if v else "no record"))
        H = collections.defaultdict(list)
        for p in self.state.get("hist", []):
            if p.get("scored") and p.get("round", 0) >= self.F and not p.get("injected"):
                for o in p.get("ops") or []:
                    H[o].append(float(p.get("reward") or 0))
        if H:
            L.append("## This run, fit of each operator's changes to the parent's own errors on unseen rows (zx):" if self.a.get("plain") else "This run, pull toward the pool's residual on unseen rows (zx):")
            L += [f"- {o}: {len(v)} scored, median {np.median(v):+.2f}, best {max(v):+.2f}" for o, v in sorted(H.items())]
        return "\n".join(L)

    def grid(self, r, rd, members, st):
        """Each operator's values on a small sample of FIT score rows: fit of the change to the residual (z, q, s) and overlap with the members' errors."""
        from vrsi27.boost.edge import proposal_stats, atom_scores
        f = f"{rd}/grid.json"
        if os.path.exists(f):
            return jload(f)
        t0 = time.time()
        w = dict(zip([m["name"] for m in members], st.cur.w)) if st.cur.w is not None else {}
        sample = sorted(self.data.of("score"), key=lambda i: sha(f"{r}:{i}"))[:GRID_ROWS]
        gr = self.a.get("grid_rows") or {}          # 128 prereg: op -> "fit" (every FIT row, show + score) or a row count; an op not named keeps the 64-row draw. Absent: as before
        if gr and self.a.get("grid_plateau_only") and not self.plateau_now(r):      # 128 prereg grid_plateau_only: grid_rows in plateau rounds only (raised weight slots, stop rule counts); 64 rows otherwise
            gr = {}
        fitrows = [i for i in self.data.learn if self.data.role[i] != "select"]
        rows_for = lambda op: fitrows if gr.get(op) == "fit" else (sorted(self.data.of("score"), key=lambda i: sha(f"{r}:{i}"))[:int(gr[op])] if isinstance(gr.get(op), int) else sample)
        anchor, anchor_share, anchor_fit = self.grid_anchor(members, sample, w)
        if anchor_share < GRID_MIN_SHARE:
            raise RuntimeError(f"r{r}: grid anchor {anchor['name']} answers {anchor_share:.2f} of the grid's rows: every cell would read about zero")
        rng = np.random.default_rng([self.a["seed"], r, 7])
        fast = r >= 2 and self.a.get("grid_fast")
        reused = []
        if fast:                     # coordinator 11-07: stock's cells do not change between rounds; a new parent runs only the top cells by evidence, at least one per operator
            reused = [dict(x, on=x.get("on") or "stock", reused=1) for x in jload(f"{self.work}/r001/grid.json")["rows_of"]]
            non = [m for m in members if m["name"] != "stock"]
            cands = []
            if non:
                cands = [(e, d, m, anchor) for e, d, m in grid_cells(anchor, top_cells(reused, GRID_TOP) + NEW_CELLS)]
        else:
            cands = [(e, d, m, anchor) for e, d, m in grid_edits(anchor, rng)]
        progs = [m for m in members if m.get("program")]
        if not fast and progs and not anchor.get("program"):        # also under the best program: a value can help only with it, or only without it
            bp = max(progs, key=lambda m: float(w.get(m["name"], 0)))
            more = grid_edits(bp, rng)
            for _, _, m in more:
                m["name"] += "_p"
            cands += [(e, d, m, bp) for e, d, m in more]
        for _, _, m, _ in cands:
            self.ev.build(m["genome"])
        self.ev.wait(self.say)
        self.ev.evaluate_many([(m, rows_for(e["op"])) for e, _, m, _ in cands])
        self.ev.wait(self.say)
        have, cell_errors = [], []
        for e, d, m, an in cands:
            share = answered_share(load_member(m), rows_for(e["op"]))
            if self.cell_ok(m, share):
                have.append((e, d, m, an))
            else:
                cell_errors.append(dict(cell=m["name"], op=e["op"], answered_share=share))
                self.say(f"r{r}: grid error: cell {m['name']} ({e['op']}) answers {share:.2f} of the grid's rows on {an['name']}: not read as a zero")
        votes, tab, P, sp, st2 = self.compute(members, [m for _, _, m, _ in have])
        ref = REF(members)
        kids = [k for k, m in enumerate(members) if m["origin"] == "proposal"]
        Ssc = atom_scores(P, sp.score, kids, ref)
        rows0 = sp.score[np.isin(np.array(tab.ids)[sp.score], sample)]
        rows_c = lambda op: rows0 if gr.get(op) != "fit" and not isinstance(gr.get(op), int) else (sp.fit[np.isin(np.array(tab.ids)[sp.fit], rows_for(op))] if gr.get(op) == "fit" else sp.score[np.isin(np.array(tab.ids)[sp.score], rows_for(op))])
        rows = rows0
        out = []
        for e, d, m, an in have:
            rows = rows_c(e["op"])
            all_wrong = ~P.correct[:len(members)][:, rows].any(0)
            k = P.modules.index(m["name"])
            x = proposal_stats(P, sp, st2, k, P.modules.index(an["name"]), combiner="evidence", S_sc=Ssc)
            bad = ~P.correct[k, rows].astype(bool)
            ov = {o["name"]: float((bad & ~P.correct[j, rows].astype(bool)).sum() / max(1, bad.sum())) for j, o in enumerate(members)}
            top = max(ov, key=ov.get)
            z = float(x.get("z_score") or 0)
            zx = float(zx_stats(tab, votes[an["name"]], votes[m["name"]], rows_for(e["op"]), self.data.cluster)["zx"] or 0)
            out.append(dict(edit=e, on=an["name"], op=e["op"], rows=int(len(rows)), d=d, dim="high" if d >= HIGH_D else "low", z=z, zx=zx, q=x.get("q_score"), s=x.get("s_score"), overlap=ov[top], overlap_with=top,
                            acc=float(P.correct[k, rows].mean()), fixes=int((all_wrong & ~bad).sum()), fit=float(np.clip(zx, -4, 4) + 2 * (1 - ov[top]))))
        if self.fusion in DS_MODES:
            try:
                self.grid_pool(r, rd, members, have, votes, tab, sp, sample, rows_for)
            except Exception as e:     # a report never stops a round
                self.say(f"r{r}: grid pool log failed: {type(e).__name__}: {str(e)[:200]}")
        rows = rows0
        res = dict(grid_rows=gr or None, anchor=anchor["name"], anchor_answered=anchor_share, anchor_fit=anchor_fit, rows=len(rows), rows_by_op={o: len(rows_c(o)) for o in sorted({e["op"] for e, _, _, _ in have})}, row_evals=sum(len(rows_for(e["op"])) for e, _, _, _ in cands), secs=time.time() - t0, tried=len(cands), cell_errors=cell_errors, rows_of=out + reused)
        jdump(f, res)
        self.say(f"r{r}: grid on {anchor['name']} (answers {anchor_share:.2f} of the rows, FIT {anchor_fit if anchor_fit is None else round(anchor_fit, 3)}): {len(out)} of {len(cands)} values, {len(rows)} rows" + (f", rows by op {res['rows_by_op']}, {res['row_evals']} cell rows in all" if gr else "") + f", {time.time() - t0:.0f}s")
        return res

    def cell_ok(self, m, share):
        return share >= GRID_MIN_SHARE

    def grid_anchor(self, members, sample, w):
        """The non-stock member with the highest FIT accuracy among those answering at least ANCHOR_SHARE of the grid's rows, else stock. An abstention counts wrong on FIT. Reads no pool weight."""
        fit = [i for i in self.data.learn if self.data.role[i] != "select"]
        best = None
        for m in members:
            v = load_member(m)
            share = answered_share(v, sample)
            acc = float(np.mean([np.mean([bool(s["ok"]) for s in v[i].values()] or [0.]) if i in v else 0. for i in fit])) if fit else 0.
            if m["name"] == "stock":
                stock = (m, share, acc)
            elif share >= ANCHOR_SHARE and (best is None or acc > best[2]):
                best = (m, share, acc)
        return best or stock

    def grid_text(self, G, op):
        L = [x for x in G["rows_of"] if x["op"] == op]
        if not L:
            return ""
        s = f"\n\n## Values of {op} measured on {G['anchor']}, {L[0].get('rows') or G['rows']} rows (zx: fit of the change to the anchor's own errors, read across its two samples; q, s: fit of the change to a log-linear pool's residual; overlap: share of its errors another member shares; fixes: rows every member misses that this gets; d: free parameters)\n"
        for x in sorted(L, key=lambda x: -x["fit"])[:8]:
            v = {k: u for k, u in x["edit"].items() if k != "op"}
            s += f"- {json.dumps(v)}: zx {x['zx']:+.2f} q {x['q'] if x['q'] is None else round(x['q'], 3)} s {x['s'] if x['s'] is None else round(x['s'], 3)} overlap {x['overlap']:.2f} with {x['overlap_with']} fixes {x['fixes']} acc {x['acc']:.3f} d {x['d']} ({x['dim']})\n"
        return s

    def review(self, r, rd, m, row, parent, ids):
        """One member against its parent on the score rows, for reading: classes, 5 rows fixed, 5 broken."""
        def raw(x):
            out = collections.defaultdict(dict)
            for d in member_dirs(x):
                for f in answer_files(d):
                    for l in open(f):
                        t = json.loads(l)
                        if t.get("id") in ids:
                            out[t["id"]].setdefault(t.get("sample", 0), t)
            return out
        A, B = raw(m), raw(parent)
        both = [i for i in ids if i in A and i in B]
        acc = lambda X, i: float(np.mean([regrade(t, self.data.rows[i])[1] for t in X[i].values()]))
        rows = lambda: None
        cls = collections.defaultdict(list)
        for i in both:
            cls[vh.class_of(self.data.rows[i]) or "other"].append(i)
        L = [f"# {m['name']}", "", f"parents {m.get('parents')}; ops {m.get('ops')}; d {m.get('d')} ({row.get('dim')})", f"why: {m.get('why')}", ""]
        if m.get("note"):
            L += [f"note: {m['note']}", f"scope: {m['gate']['scope']}", f"generation recites: {row.get('recite')} of its category's score samples", ""]
        if m.get("program"):
            L += ["program:", "```python", str(m["program"])[:1500], "```", ""]
        for x in m.get("inherited_gates", []):
            own = [i for i in A if vh.key(self.data.rows[i]) == x["scope"][4:]]
            T = [t for i in own for t in A[i].values()]
            L += [f"inherited gates recite: {x['origin']} {x['scope']}: {sum((t.get('reasoning') or '').lstrip().startswith(x['head']) for t in T)}/{len(T)} samples on {len(own)} rows of its category", ""]
        L += [f"status: {row.get('status')}; valid {row.get('valid')}", "zx {zx_score} dacc {dacc_paired_score} | z {z_score} z2 {z2_score} q {q_score} s {s_score} coverage {coverage_score} ce {ce_gain_score} | strata {strata_score} cross {strata_cross_score}".format_map(collections.defaultdict(lambda: None, {**row.get("diag", {}), **{k: v for k, v in row.items() if k.endswith("_score")}})), "",
              f"## Accuracy against {parent['name']} on {len(both)} score rows", "class | rows | member | parent"]
        for c, v in sorted(cls.items()):
            L.append(f"{c} | {len(v)} | {np.mean([acc(A, i) for i in v]):.3f} | {np.mean([acc(B, i) for i in v]):.3f}")
        L.append(f"all | {len(both)} | {np.mean([acc(A, i) for i in both]) if both else 0:.3f} | {np.mean([acc(B, i) for i in both]) if both else 0:.3f}")
        d = sorted(both, key=lambda i: acc(A, i) - acc(B, i))
        def show(title, v):
            L.extend(["", f"## {title}"])
            for i in v:
                x, y = A[i][min(A[i])], B[i][min(B[i])]
                L.extend([f"### {i} ({vh.key(self.data.rows[i])}), gold {x.get('answer')}", f"member {acc(A, i):.2f}: {json.dumps((x.get('reasoning') or '')[:300], ensure_ascii=False)}", f"member said: {json.dumps((x.get('content') or '')[-120:], ensure_ascii=False)}",
                          f"parent {acc(B, i):.2f}: {json.dumps((y.get('reasoning') or '')[:300], ensure_ascii=False)}", f"parent said: {json.dumps((y.get('content') or '')[-120:], ensure_ascii=False)}"])
        show("Fixed", [i for i in reversed(d) if acc(A, i) > acc(B, i)][:5])
        show("Broken", [i for i in d if acc(A, i) < acc(B, i)][:5])
        os.makedirs(f"{rd}/review", exist_ok=True)
        open(f"{rd}/review/{m['name']}.md", "w").write("\n".join(L) + "\n")

    def diag_stats(self, m, row):
        """z, z2, q, s of one member with records, whatever its fate: for reading only, the decision uses the scored rows."""
        from vrsi27.boost.edge import proposal_stats, atom_scores
        members = self.state["members"]
        votes, tab, P, sp, st = self.compute(members, [m])
        ref = REF(members)
        kid = [k for k, x in enumerate(members) if x["origin"] == "proposal"]
        read = getattr(self, "read_rows", None)         # filler rows are not read here either
        ran = np.array(sorted(tab.index[i] for i in self.data.of("score") if i in votes[m["name"]] and (read is None or i in read(m))), int)
        if len(ran) < len(sp.score):     # ranked on part of the score rows: read it on the rows it ran, as the members' residual there
            sp = dataclasses.replace(sp, score=ran)
            st = state_of(P, sp, members)
        x = proposal_stats(P, sp, st, P.modules.index(m["name"]), ref, combiner="evidence", S_sc=atom_scores(P, sp.score, kid, ref))
        row["diag"] = {k: x.get(k) for k in ("z_score", "z2_score", "q_score", "s_score", "coverage_score", "ce_gain_score")}
        pa = next(p for p in members if p["name"] == m["parents"][0])
        zz = zx_stats(tab, votes[pa["name"]], votes[m["name"]], self.data.of("score"), self.data.cluster)
        row["diag"].update(zx_score=zz["zx"], dacc_paired_score=zz["dacc"], strata_score=zz.get("strata"), strata_cross_score=zz.get("strata_cross"))

    def recited(self, m, q_score):
        """Share of its own category's samples, in real generation, that open with the note; None if it has none."""
        cat, head = m["gate"]["scope"][4:], m["note"][:20]
        ids = {i for i in q_score if vh.key(self.data.rows[i]) == cat}
        n = k = 0
        for d in member_dirs(m):
            for f in answer_files(d):
                for l in open(f):
                    r = json.loads(l)
                    if r.get("id") in ids:
                        n += 1
                        k += (r.get("reasoning") or "").lstrip().startswith(head)
        return k / n if n else None

    def prompts(self, r, pool, tab, sp, st, members, votes, pop, brd, rng, rd):
        from vrsi27.boost.lens import Budget, build_state, run_lens, validate_output, card_block, LensError, LeakError
        from vrsi27.boost.meta import allocate, pick_incumbent, pick_parent as lens_parent, select_examples, render_meta_prompt, meta_system_prompt
        f = f"{rd}/prompts.json"
        if os.path.exists(f):            # a restart inside the round: the offer the prompts were made with, so the parse checks the same scopes
            o_, gi_ = self.gate_open(r, sorted({vh.key(self.data.rows[i]) for i in self.data.of("show")}))
            self.gate_offer = set(o_) if gi_ else None
            return jload(f)
        a = self.a
        inc = pick_incumbent(brd, pop.incumbent, a["lens0"], min_n=4, posterior="normal")
        slots, info = allocate(brd, inc, n_slots=a["slots"], n_incumbent=2, rng=rng)
        pop.incumbent = inc
        pop.save()
        g = self.guard(tab)
        w = dict(zip([m["name"] for m in members], st.cur.w)) if st.cur.w is not None else {}
        cats = sorted({vh.key(self.data.rows[i]) for i in self.data.of("show")})
        cats, g_info = self.gate_open(r, cats)          # A1: only keys the writer can fit, read and score, not yet written
        self.gate_offer = set(cats) if g_info else None
        if g_info:
            jdump(f"{rd}/gate_open.json", g_info)
            self.say(f"r{r}: A1 gate keys: {len(g_info['eligible'])} eligible by fit>={g_info['rule']['fit']} read>={g_info['rule']['read']} score>={g_info['rule']['score']}, {len(g_info['written'])} written, {len(cats)} open: {cats}")
        alive = members
        plan = G = None
        S_hist = self.state.get("hist", [])
        if r >= self.F and not a.get("noweights"):
            G = self.grid(r, rd, members, st)
            best = {}
            for x in G["rows_of"]:
                best[x["op"]] = max(best.get(x["op"], -9), x["fit"])
            cap = OP_CAP if r >= 3 or a.get("plain") else None
            split = a.get("split") or active_amendment(self.work, r).get("split")
            wt_, pr_ = [m for m in members if changes_network(m)], [m for m in members if m.get("program")]
            can_cross = len(wt_) >= 2 or any(u["name"] != v["name"] for u in wt_ for v in pr_)
            raised = self.plateau_now(r)
            if raised:
                split, best = PLATEAU_SPLIT, {o: v for o, v in best.items() if o in PLATEAU_OPS}
                self.say(f"r{r}: plateau: slots {split}, grid from {sorted(best)}")
                grid_all = sorted(best)
            plan = split_ops(split, best, a["slots"], rng, cap, can_cross, S_hist if a.get("grid_fast") else None, gate_cap=len(cats) if g_info else None) if split and r >= split.get("from_round", 1) else allocate_ops(best, a["slots"], rng, cap)
            if raised:
                self.say(f"r{r}: grid_slots leaves out {sorted(set(grid_all) - set(plan))} of {grid_all}, best fit {({k: round(v, 2) for k, v in best.items()})}")
            uncapped = grid_slots(best, a["slots"] - 6) if best else {}
            jdump(f"{rd}/dose_cells.json", [dict(op=x["op"], name=x["edit"]["name"], alpha=x["edit"]["alpha"], first=x["edit"]["first"], last=x["edit"]["last"], acc=x["acc"], fit=x["fit"], zx=x["zx"], fixes=x["fixes"]) for x in G["rows_of"] if x["op"] in ("direction", "verbosity")])
            jdump(f"{rd}/slots.json", dict(cap=cap, slots=dict(collections.Counter(plan)), grid_uncapped=uncapped, best=best, raised=raised))
            self.say(f"r{r}: slots {dict(collections.Counter(plan))}; uncapped grid slots {uncapped}; cap {cap}; best fit {({k: round(v, 2) for k, v in best.items()})}")
        if self.c9:                      # 1109c: the slot plan comes from the P level (program / cross / W slots), W slots over the admitted families
            wt_, pr_ = [m for m in members if changes_network(m)], [m for m in members if m.get("program")]
            plan = self.HW.plan(self, r, rng, len(wt_) >= 2 or any(u["name"] != v["name"] for u in wt_ for v in pr_), have_grid=G is not None)
            best = best if G is not None else {}
            old_ = jload(f"{rd}/slots.json", None)       # 1109c-final: slots.json logs the deal actually used (the grid-split plan above is only logged beside it)
            jdump(f"{rd}/slots.json", dict(slots=dict(collections.Counter(plan)), order=list(plan), allocator=self.HW.cfg(self).get("allocator"),
                                           deal=((self.state.get("slot_bandit") or {}).get("deals") or [None])[-1] if self.HW.cfg(self).get("allocator") == "ts" else None,
                                           grid_split_plan_not_used=old_))
        score = {m["name"]: abs(w.get(m["name"], 0)) + float(pool.correct[k, sp.show].mean()) for k, m in enumerate(members)}
        if self.fusion in DS_MODES:     # no stacking weight: what the member adds to the held-out pool, plus show accuracy
            loo = self.member_loo(members, votes, sp)
            score = {m["name"]: loo[m["name"]] + float(pool.correct[k, sp.show].mean()) for k, m in enumerate(members)}
            jdump(f"{rd}/parent_scores.json", [dict(name=m["name"], loo_fit=loo[m["name"]], show_acc=float(pool.correct[k, sp.show].mean()), score=score[m["name"]], stacking_abs_w=abs(float(w.get(m["name"], 0)))) for k, m in enumerate(members)])
            self.say(f"r{r}: parent score = leave-one-out held-out FIT gain + show accuracy: " + ", ".join(f"{m['name']} {1e4 * loo[m['name']]:+.1f}e-4 + {float(pool.correct[k, sp.show].mean()):.3f}" for k, m in enumerate(members)))
        reqs, wrongs = [], []
        spread, used = self.spread_on(r), set()
        given = []                       # A4: operators given this round to cross slots without a fresh pair
        for k, lid in enumerate(slots):
            x, y = rng.choice(len(alive), 2, replace=False) if len(alive) > 1 else (0, 0)
            pa = alive[x] if score[alive[x]["name"]] >= score[alive[y]["name"]] else alive[y]
            parents = [pa]
            op = plan[k] if plan else None
            if op == "cross" and a.get("split"):    # a pair already crossed, in an earlier round or this one, gives its slot to a free program
                wt, pr = [m for m in alive if changes_network(m)], [m for m in alive if m.get("program")]
                ok_ = lambda u, v: not (u["genome"].get("vision") and v["genome"].get("vision") and canon(u["genome"]["vision"]) != canon(v["genome"]["vision"]))
                pairs = [(u, v) for u in wt for v in pr if u["name"] != v["name"] and ok_(u, v)] + [(u, v) for u in wt for v in wt if u["name"] < v["name"] and ok_(u, v)]
                seen_ = {frozenset(x) for x in self.state.get("cross_pairs", [])} | {frozenset(q["parents"]) for q in reqs if q.get("op") == "cross"}
                if all(frozenset((u["name"], v["name"])) in seen_ for u, v in pairs):
                    op = self.a4_cross_op(r, S_hist, best, plan, given) if a.get("grid_fast") and best else "program"
            if op == "program":              # a free program: any member of the pool, stock too
                pa = alive[int(rng.integers(len(alive)))]
                parents = [pa]
            if op == "tune" and not pa["genome"]["engine"]:
                op = "direction"
            if (op == "cross" or (not plan and k % 4 == 3)) and len(alive) > 1:
                wt, pr = [m for m in alive if changes_network(m)], [m for m in alive if m.get("program")]
                pairs = [(u, v) for u in wt for v in pr if u["name"] != v["name"]] if op == "cross" else []
                pairs = pairs or ([(u, v) for u in wt for v in wt if u["name"] < v["name"]] if op == "cross" else [])
                if op == "cross" and a.get("split"):
                    fresh = [(u, v) for u, v in pairs if ok_(u, v) and frozenset((u["name"], v["name"])) not in seen_]
                    pairs = fresh or [(u, v) for u in wt for v in wt if u["name"] < v["name"] and ok_(u, v) and frozenset((u["name"], v["name"])) not in seen_]
                if pairs:                    # a weight member and a program member: the child carries both edits
                    pa, o = pairs[int(rng.integers(len(pairs)))]
                    parents = [pa, o]
                else:
                    others = [m for m in alive if m["name"] != pa["name"]]
                    parents.append(others[int(rng.integers(len(others)))])
            ctx = self.lens_ctx(pool, tab, pa["name"])
            full, rid_row = build_state(pool, sp, st, ctx, round_idx=r, slot=k, n_slots=a["slots"], budget=Budget(), weights=None, accepted=[])      # no weight reaches a lens or a proposer; the parent choice above and the logs read them
            pv, view = self.lineage_view(pool, tab, members, votes, pa, full, rid_row)
            tried = [lid] + [x for x in (inc, a["lens0"]) if x != lid]
            done = None
            for t in tried:
                if t not in pop.entries or pop.entries[t].status != "active":
                    continue
                try:
                    state, out, cw = self.wrong_card(pop.source(t), full, r, pv, used if spread else None)
                    if spread:
                        used.update(cw.get("wrong_rows") or [])
                    out = self.card_view(out, state, pa, view)
                    wrongs.append(dict(slot=k, lens=t, parent_answers=view is None or view.split(" answers ")[1].split(" of ")[0], **cw))
                    card = self.card_text(state, out, pa)
                    if r >= self.F:
                        card = self.mark_rows(card, out, state)
                    head = f"# Round {r}: {out['cards'][0]['title']}\n\n" + re.sub(r"[^.\n]*refit weight[^.\n]*\.\s*", "", out["framing"] or "") + "\n"
                    must = (op is not None and op != "program") or (op is None and k % 4 < 3 and not a.get("noweights"))         # the thesis: members must differ inside the network; three slots of four change the weights or structure
                    user = head + self.overlap_table(pool, sp, members, st, pa["name"]) + "\n" + card + self.card_cases(state, out, pv, rid_row, tab) + tail(parents, cats, a.get("noweights"))
                    if r >= self.F:
                        user += KEEP_RIGHT
                    if must:
                        user += ' In this proposal "edits" must not be empty: change the network itself (a program may come with it). Pick the edit from what you think the network gets wrong on these questions.'
                    if op == "program":
                        user += " This proposal is a harness program: write it from the cause you find. Network edits are not asked for here."
                    elif op:
                        user += f' Operator of this proposal: "{op}"' + (" (two parents above; edits may be empty)" if op == "cross" else "") + ". You choose its values and targets. A direction may carry either sign of alpha.\n\n" + self.prior_gains() + self.grid_text(G, op)
                    if answered_share(load_member(pa), self.data.of("show")) <= NARROW or ((pa.get("gate") or {}).get("scope") or "").startswith("cat:"):
                        user = user.replace(f"Goal: raise the standalone accuracy of `{pa['name']}`:", f"Goal: `{pa['name']}` is a narrow member, so its accuracy over all questions is not the aim. Make it right on more of its own rows:")
                    sk_ = None
                    if self.c9:          # 1109c: proposer notes, search state and <= 2 code-matched skills, proposer prompts only
                        ex_, sk_ = self.HW.prompt_extra(self, sys.modules[__name__], r, k, op)
                        user += ex_
                    user, card = no_conf_api(user), no_conf_api(card)
                    g.audit(user)
                    done = dict(id=f"s{k:02d}", slot=k, op=op, lens_id=t, lens_hash=pop.entries[t].sha256[:12], parents=[p["name"] for p in parents], card_text=card, system=self.system_text(), user=user, must=must, card_rows=[tab.ids[rid_row[x]] for c in out["cards"] for x in c["rows"] if x in rid_row])
                    if self.c9:
                        done["c1109_skills"] = sk_
                        self.HW.check_cards(self, done)
                    break
                except (LensError, LeakError, Exception) as e:
                    self.say(f"lens {t} slot {k}: {type(e).__name__}: {str(e)[:200]}")
                    if len(alive) > 1:           # with one member a lens that reads differences between members has nothing to read: not its fault, it stays for later rounds
                        pop.fail(t, str(e))
            if done:
                reqs.append(done)
        hist = self.state["hist"]
        par = lens_parent(brd, rng, posterior="normal")
        exs = select_examples(hist, rng=rng)
        rew = "signed z"
        mp = render_meta_prompt(round_idx=r, parent_id=par, parent_source=pop.source(par), board_rows=brd, examples=exs, blind=False, code_of=lambda ex: open(ex["file"]).read() if ex.get("file") and os.path.exists(ex["file"]) else "",
                                rng=rng, reward_name=rew, evidence=True)
        mp = mp.replace("{name, weight (signed refit weight), show_acc}", "{name, show_acc}")
        try:
            g.audit(mp)
            sysm = meta_system_prompt(reward_name=rew, evidence=True).replace("a harness-synthesis loop for video multiple-choice question answering", "a loop that edits copies of a vision-language model and their harness programs").replace("video", "image")
            reqs.append(dict(id="meta", lens_parent=par, system=sysm, user=mp))
        except LeakError as e:
            self.say(f"meta prompt held back by the leak guard: {str(e)[:200]}")
        jdump(f, reqs)
        jdump(f"{rd}/card_wrong.json", wrongs)
        if spread:
            sp_ = dict(unique_wrong_rows=len(used), wrong_rows_on_cards=sum(x.get("n_wrong", 0) for x in wrongs), overlap_each_card=[x.get("overlap") for x in wrongs])
            jdump(f"{rd}/card_spread.json", sp_)
            self.say(f"r{r}: card rows: {sp_['unique_wrong_rows']} unique wrong rows of {sp_['wrong_rows_on_cards']} on cards; overlap with earlier cards {sp_['overlap_each_card']}")
        pop.save()
        return reqs

    def system_text(self):
        if self.fusion == "stacking":
            return SYSTEM
        return SYSTEM.replace("The pool's answer is a refit signed log-linear combination of its members.", "The pool's answer fuses its members' votes: where every member that answers agrees, that answer; elsewhere each vote counts by how that member's answers relate to the truth in that question's class.").replace("how well its change of the answer distribution lines up with the pool's residual on questions you do not see, then confirmed on a second held-out set", "how much its change fixes its parent's errors on questions you do not see and what it adds to the pool's answers on a second held-out set")

    def fz(self, what, fn, *a, **k):
        """A fusion call with its time; line_report writes them per round."""
        t = time.time()
        try:
            return fn(*a, **k)
        finally:
            self.fz_t.setdefault(what, []).append(time.time() - t)

    def spread_on(self, r):
        """Narrow slot in stage 2 and row spread across cards: from the round the arm's amendment names, so a restart inside an earlier round changes nothing."""
        return r >= (jload(f"{self.work}/amendment.json", {}) or {}).get("narrow_from", 10 ** 9)

    def touched(self, m, parent, ids):
        """Rows this member really changes against its parent. A network edit changes every row; a gate note with a category scope, the rows of that category; a program, the rows whose prompt or views differ."""
        scope = (m.get("gate") or {}).get("scope") or ""
        if scope.startswith("cat:"):
            return [i for i in ids if vh.key(self.data.rows[i]) == scope[4:]]
        if vh.gid(m["genome"]) != vh.gid(parent["genome"]):       # every row it answers; a row its program abstains on has no vote, so no change
            return [x["id"] for x in program_rows(self.data, m, ids, render=False)[0]] if m.get("program") else list(ids)
        if m.get("program") == parent.get("program"):
            return []
        mine = {x["id"]: x for x in program_rows(self.data, m, ids, render=False)[0]}
        base = {x["id"]: x for x in program_rows(self.data, parent, ids, render=False)[0]}
        key = lambda x: (x["prompt"], tuple(x.get("image_paths") or []))
        return [i for i in ids if i in mine and (i not in base or key(mine[i]) != key(base[i]))]

    def rank_cfg(self, r):
        """Amendment 6: stage 1 ranks on a part of the score rows that rotates each round; the gate's FIT part reads the rest."""
        c = active_amendment(self.work, r).get("rank")
        return c if c and r >= c["from_round"] else None

    def rank_draw(self, r, ids, n):
        """About n rows, stratified by category: whole picture clusters by hash(round, cluster), and a picture picked in one category is picked in all."""
        cats = collections.defaultdict(list)
        for i in ids:
            cats[vh.key(self.data.rows[i])].append(i)
        rank = lambda c: sha(f"hbr:{r}:{c}")           # not a crc: crcs of strings that differ only in the round are tied by a fixed xor, the draws would not rotate
        take = set()
        for c, rows in sorted(cats.items()):
            quota, got = max(1, round(n * len(rows) / len(ids))), 0
            by = collections.defaultdict(list)
            for i in rows:
                by[self.data.cluster(i)].append(i)
            for k in sorted(by, key=lambda k: (rank(k), k)):
                if got >= quota:
                    break
                take.add(k)
                got += len(by[k])
        return [i for i in ids if self.data.cluster(i) in take]

    def rank_rows(self, r, m, parent, draw, ids):
        """A wide member is ranked on the round's draw. A narrow one also gets touched rows from the rest, up to min(20, half of what it touches): the other half stays for its gate."""
        import math
        try:
            t = self.touched(m, parent, ids)
        except (Bad, KeyError, ValueError, OSError) as e:
            self.say(f"  rank rows {m.get('name')}: {type(e).__name__}: {str(e)[:120]}")
            return draw, 0
        scope = ((m.get("gate") or {}).get("scope") or "")
        if not (scope.startswith("cat:") or len(t) / max(1, len(ids)) <= NARROW):
            return draw, 0
        have_ = set(draw)
        need = min(20, math.ceil(len(t) / 2)) - sum(i in have_ for i in t)
        if need <= 0:
            return draw, 0
        rest = sorted((i for i in t if i not in have_), key=lambda i: (sha(f"hbx:{r}:{self.data.cluster(i)}"), i))
        add, cl = [], set()
        for i in rest:
            k = self.data.cluster(i)
            if len(add) >= need and k not in cl:
                break
            add.append(i)
            cl.add(k)
        self.say(f"  rank rows {m['name']}: narrow, {len(t)} touched, {len(t) - len(rest)} in the draw, {len(add)} added")
        return [i for i in ids if i in have_ or i in set(add)], len(add)

    def touch_stats(self, m, row, parent, ids, tab, votes, read=None):
        """Coverage by rows really changed, and gain on those rows beside the overall gain: a narrow member is read on what it touched."""
        try:
            t = self.touched(m, parent, ids)
            rows_ = [i for i in t if read is None or i in set(read)]         # filler rows of this candidate are not read here
            z = zx_stats(tab, votes[parent["name"]], votes[m["name"]], rows_, self.data.cluster) if rows_ else dict(zx=None, dacc=None, zx_rows=0)
            row.update(touched_rows=len(t), coverage_touched=len(t) / max(1, len(ids)), dacc_touched=z["dacc"], zx_touched=z["zx"], narrow=bool(((m.get("gate") or {}).get("scope") or "").startswith("cat:") or len(t) / max(1, len(ids)) <= NARROW))
            self.say(f"  coverage {m['name']}: {len(t)}/{len(ids)} rows touched ({row['coverage_touched']:.2f}), narrow {row['narrow']}; on touched rows dacc {100 * (z['dacc'] or 0):+.1f} zx {z['zx'] if z['zx'] is None else round(z['zx'], 2)}; overall dacc {100 * (row.get('dacc_paired_score') or 0):+.1f} zx {row.get('zx_score') or 0:+.2f}")
        except (Bad, KeyError, ValueError, OSError) as e:
            row.update(coverage_touched=None, narrow=False)
            self.say(f"  coverage {m.get('name')}: {type(e).__name__}: {str(e)[:150]}")

    def pick_stage2(self, have, r):
        """Top by cross-sample score. From the amendment's round one slot goes to the best narrow member with positive zx not already taken; if there is none it stays with zx."""
        n = self.a["stage2"]
        ranked = sorted([x for x in have if x[1].get("zx_score") is not None], key=lambda x: -float(x[1]["zx_score"]))
        if not self.spread_on(r):
            return ranked[:n]
        wide = self.a.get("stage2_wide")
        if wide:                     # coordinator 11-07: every candidate with zx above 1, up to the cap, beside the narrow slot; the same evidence rule accepts
            top = [x for x in ranked if float(x[1]["zx_score"]) > 1][:wide]
            for _, row in top:
                row["stage2_by"] = "zx>1"
            pick = next((x for x in ranked if x not in top and x[1].get("narrow") and float(x[1]["zx_score"]) > 0), None)
            if pick:
                pick[1]["stage2_by"] = "narrow"
            self.say(f"r{r}: stage 2 slots: " + ", ".join(f"{x[0]['name']} ({x[1]['stage2_by']})" for x in top + ([pick] if pick else [])))
            return top + ([pick] if pick else [])
        top = ranked[:n - 1]
        for _, row in top:
            row["stage2_by"] = "zx"
        pick = next((x for x in ranked[n - 1:] if x[1].get("narrow") and float(x[1]["zx_score"]) > 0), None)
        if pick:
            pick[1]["stage2_by"] = "narrow"
        else:
            pick = ranked[n - 1] if len(ranked) >= n else None
            if pick:
                pick[1]["stage2_by"] = "zx, no narrow member"
        self.say(f"r{r}: stage 2 slots: " + ", ".join(f"{x[0]['name']} ({x[1]['stage2_by']})" for x in top + ([pick] if pick else [])))
        return top + ([pick] if pick else [])

    def mark_rows(self, card, out, state):
        """Each example says plainly whether the pool now gets it wrong or right; the examples appear in the order of the card's rows."""
        by = {x["rid"]: x for x in state["rows"]}
        it = iter(x for c in out["cards"] for x in c["rows"])
        return re.sub(r"^#### example \d+[^\n]*$", lambda m: m.group(0) + (" — currently right" if by[next(it)]["ens_correct"] else " — currently wrong"), card, flags=re.M)

    def wrong_card(self, src, full, r, pv, used=None):
        """From round 2 a lens sees only the rows the pool gets wrong: smoothed 2-sample votes leave |g| near .11 on rows it gets right, and proposers spent proposals on confidence.
        The lens is also run on all rows, to log what it would have shown. A lens that cannot work on the wrong rows keeps the wrong rows of its own card."""
        from vrsi27.boost.lens import run_lens, validate_output, LensError
        def go(st):
            res = run_lens(src, st, timeout_s=20.)
            if not res.get("ok"):
                raise LensError(f"{res.get('stage')}: {str(res.get('error'))[:300]}")
            return validate_output(res["output"], st)
        ok = {x["rid"]: not x["ens_correct"] for x in full["rows"]}
        share = lambda o: float(np.mean([ok[x] for c in o["cards"] for x in c["rows"]]))
        base = go(full)
        before = share(base)
        if r < self.F or not any(ok.values()):
            return full, base, dict(before=before, after=before, n=sum(len(c["rows"]) for c in base["cards"]), how="all rows")
        # stably wrong: every sample of the parent wrong. Categories with most such rows first; a shared cause is what an edit can fix
        stable = {x["rid"]: bool(pv.get(x["rid"])) and not any(v["ok"] for v in pv[x["rid"]].values()) for x in full["rows"]}
        wrong = [x for x in full["rows"] if ok[x["rid"]]]
        cnt = collections.Counter(x["task_type"] for x in wrong if stable[x["rid"]])
        rank = lambda x: (-cnt[x["task_type"]], x["task_type"], not stable[x["rid"]], -x["g_norm"])
        some = [x for x in wrong if stable[x["rid"]]]
        seen = used or set()
        pool_ = some if len(some) >= 12 else wrong
        fresh = [x for x in pool_ if x["rid"] not in seen]
        if len(fresh) < 8:            # short of unused rows at this level: unused rows of any wrong kind, then the used ones as before
            fresh = [x for x in wrong if x["rid"] not in seen]
        st = dict(full, rows=sorted(fresh if len(fresh) >= 8 else pool_, key=rank))
        order = {x["rid"]: k for k, x in enumerate(sorted(wrong, key=rank))}
        try:
            out, how = go(st), "wrong rows"
        except LensError:
            keep = [x for x in base["cards"][0]["rows"] if ok[x]] or [x["rid"] for x in st["rows"][:12]]
            st, out, how = dict(full, rows=sorted(wrong, key=rank)), dict(base, cards=[dict(base["cards"][0], rows=keep)]), "own card, wrong rows"
        # owner 11-07 via coordinator: wrong-only cards taught base rates ("the expected answer fails"). About 8 stably wrong rows beside 4 stably right rows, up to 12, each marked
        byrid = {x["rid"]: x for x in full["rows"]}
        narrow_ = bool(full["rows"]) and sum(bool(x.get("in_scope")) for x in full["rows"]) < .5 * len(full["rows"])
        scoped = [x["rid"] for x in sorted(wrong, key=lambda x: (x["rid"] in seen, rank(x))) if x.get("in_scope")][:8] if narrow_ else []      # a narrow parent: its own wrong rows first, the rest from outside its scope
        left, cards = 8 - len(scoped), []
        for c in out["cards"]:
            take = sorted([x for x in c["rows"] if x not in scoped], key=lambda x: x in seen)[:left]       # unused rows first, each group in the lens's own order
            left -= len(take)
            if take:
                cards.append(dict(c, rows=sorted(take, key=order.get)))
        if scoped:
            cards = [dict(cards[0], rows=scoped + cards[0]["rows"])] + cards[1:] if cards else [dict(out["cards"][0], rows=scoped)]
            how += f", {len(scoped)} rows in scope"
        have = {x for c in cards for x in c["rows"]}
        if len(have) < 8:             # a card of a row or two is no evidence of a shared cause: fill it from the stably wrong rows, most common categories first
            more = [x["rid"] for x in sorted(wrong, key=lambda x: (x["rid"] in seen, rank(x))) if x["rid"] not in have][:8 - len(have)]
            if cards:
                cards[0] = dict(cards[0], rows=cards[0]["rows"] + more)
            else:
                cards = [dict(out["cards"][0], rows=more)]
            have |= set(more)
            how += ", filled"
        cats = [byrid[x]["task_type"] for c in cards for x in c["rows"]]
        same = list(dict.fromkeys(cats))
        right_ok = lambda x: not ok[x["rid"]] and bool(pv.get(x["rid"])) and all(v["ok"] for v in pv[x["rid"]].values())
        def lens_order(cand):           # the same lens order where it can rank them; by residual where it cannot
            try:
                seq = [r for c in go(dict(full, rows=cand))["cards"] for r in c["rows"]]
            except LensError:
                seq = []
            seq = list(dict.fromkeys(seq))
            return seq + [x["rid"] for x in sorted(cand, key=lambda x: -x["g_norm"]) if x["rid"] not in set(seq)]
        by = {t: sorted(lens_order(v), key=lambda q_: narrow_ and not byrid[q_].get("in_scope")) for t in same for v in [[x for x in full["rows"] if x["task_type"] == t and right_ok(x)]] if v}
        right = []
        while len(right) < 4 and any(by.values()):       # in turn across the card's categories
            for t in same:
                if by.get(t) and len(right) < 4:
                    right.append(by[t].pop(0))
        n_same = len(right)
        if n_same < 4:                  # too few in those categories: the same bench, and the card says so
            benches = {byrid[x]["dataset"] for x in have}
            cand = [x for x in full["rows"] if x["dataset"] in benches and x["task_type"] not in same and right_ok(x)]
            if cand:
                right += lens_order(cand)[:4 - n_same]
        if len(have) + len(right) < 12:   # fewer right rows than wanted: the wrong rows take the room
            more = [x["rid"] for x in sorted(wrong, key=lambda x: (x["rid"] in seen, rank(x))) if x["rid"] not in have][:12 - len(have) - len(right)]
            if len(right) < 4 and more:
                cards[0] = dict(cards[0], rows=cards[0]["rows"] + more)
                have |= set(more)
        w, rr = len(have), len(right)
        note = f"The lens's line above describes the rows it chose from the currently wrong ones. This card has {w} currently wrong rows and {rr} currently right rows, each marked."
        if rr > n_same:
            note += f" {rr - n_same} of the currently right rows come from other categories of the same benchmark: their own categories had too few."
        if rr == 0:
            note += " No currently right row was available in these categories or this benchmark."
        cards[0] = dict(cards[0], rows=cards[0]["rows"] + right, summary=((cards[0].get("summary") or "") + "\n" + note).strip())
        st = dict(full, rows=[byrid[x] for x in dict.fromkeys(list(have) + right + [y["rid"] for y in st["rows"]])])
        out = dict(out, cards=cards)
        rows_ = [x for c in cards for x in c["rows"]]
        wr = [x for x in rows_ if ok[x]]
        return st, out, dict(before=before, after=share(out), stable=float(np.mean([stable[x] for x in wr])) if wr else None, n=len(rows_), n_wrong=len(wr), n_right=len(rows_) - len(wr), wrong_rows=wr, overlap=sum(x in seen for x in wr), right_same_category=n_same, right_same_bench=rr - n_same,
                             stable_right=float(np.mean([right_ok(byrid[x]) for x in rows_ if not ok[x]])) if len(rows_) > len(wr) else None, how=how + ", right rows beside")

    def generate(self, reqs, rd):
        """Through the long-lived stock engine. If it dies, the next attempt starts it again.
        Request and answer are named by the sha of the exact prompts.json they belong to, so an answer to other prompts is never read and a late one lands on its own name.
        Requests for other prompts still in the inbox are set aside. Round files from before this naming count only if their request is byte-identical to prompts.json."""
        body = open(f"{rd}/prompts.json", "rb").read()
        tag = hashlib.sha256(body).hexdigest()[:16]
        out, q = f"{rd}/gen.{tag}.json", f"{rd}/gen.req.{tag}.json"
        if not os.path.exists(out) and os.path.exists(f"{rd}/gen.json") and os.path.exists(f"{rd}/gen.req.json") and open(f"{rd}/gen.req.json", "rb").read() == body:
            return jload(f"{rd}/gen.json")
        n = len(glob.glob(out + ".failed*"))
        if os.path.exists(out + ".failed") and n < 3:
            os.replace(out + ".failed", f"{out}.failed{n}")
        if not os.path.exists(out) and not os.path.exists(out + ".failed"):
            open(q + ".new", "wb").write(body)
            os.replace(q + ".new", q)
            self.ev.worker_ensure()
            wd = self.ev.wdir()
            waiting = False
            for f in glob.glob(f"{wd}/inbox/*.json"):
                try:
                    it = jload(f)
                    if it.get("kind") == "gen":
                        if it.get("out") == out:
                            waiting = True
                        else:
                            os.replace(f, f + ".stale")
                            self.say(f"r: set aside a proposer request for other prompts: {it.get('out')}")
                except (OSError, ValueError):
                    pass
            try:   # busy names the item as it was in the inbox; the worker moved it to done/
                busy = jload(f"{wd}/done/{os.path.basename(open(f'{wd}/busy').read().strip())}") if os.path.exists(f"{wd}/busy") and self.ev.worker_alive() else None
            except (OSError, ValueError):
                busy = None
            if busy and busy.get("out") == out:
                waiting = True
            if waiting:
                self.ev.pending.append(dict(name=jload(f"{wd}/launch.json", {}).get("name", "worker"), outs=[out], flag=out + ".failed", worker=True))
            else:
                self.ev.to_worker([dict(kind="gen", req=q, out=out, model=vh.STOCK)])
            self.ev.wait(self.say)
        if not os.path.exists(out):
            raise RuntimeError("the proposer job failed")
        self.say(f"  proposals answer prompts.json sha {tag}")
        return jload(out)

    def candidate(self, r, req, ans, members):
        by = {m["name"]: m for m in members}
        if not isinstance(ans, dict):
            raise Bad("no JSON answer")
        names = ans.get("parents") or req["parents"]
        names = [n for n in names if n in by][:2] or req["parents"]
        if req.get("op") == "cross" and len(req["parents"]) == 2:
            names = req["parents"]       # the slot's pair: the point of the slot is this combination
        if len(names) == 2:              # a child is measured against its first parent: the better one on FIT, by vote, on the same rows
            names = sorted(names, key=lambda n: -getattr(self, "parent_fit", {}).get(n, 0))
        ps = [by[n] for n in names]
        if self.a.get("noweights"):
            if not ans.get("program"):
                raise Bad("this arm changes programs only")
            ps, names, ans = ps[:1], names[:1], {**ans, "edits": []}
        g = cross(ps[0], ps[1]) if len(ps) == 2 else dict(ps[0]["genome"])
        prog = ps[0].get("program") or (ps[1].get("program") if len(ps) == 2 else None)
        if ans.get("program"):
            prog = str(ans["program"])
        elif len(ps) == 2 and ps[0].get("program") and ps[1].get("program") and ps[0]["program"] != ps[1]["program"]:
            raise Bad("the two parents carry different programs: give one program for the child")
        cats = {vh.key(self.data.rows[i]) for i in self.data.learn}
        ops, d, gate = (["cross"] if len(ps) == 2 else []), 0, None
        edits = ans.get("edits") or []
        if not isinstance(edits, list):
            raise Bad("edits must be a list")
        want = req.get("op")
        if req.get("must") and not edits and want != "cross":
            raise Bad("this slot asked for a network edit")
        edits = [normalize_edit(e) for e in edits]
        if want and want != "gate":
            edits = [e for e in edits if e.get("op") != "gate"]
        if any(isinstance(e, dict) and e.get("op") == "gate" for e in edits):
            edits = [next(e for e in edits if e.get("op") == "gate")]
        for e in edits[:3]:
            change, dd, s = vh.validate(e, dict(g, engine=g["engine"]), cats)
            d += dd
            ops.append(e["op"])
            if e["op"] == "gate":
                if getattr(self, "gate_offer", None) is not None and str(s)[4:] not in self.gate_offer:
                    raise Bad(f"gate scope {s} is not open this round (A1: too few rows to fit, read or score, or written already)")
                gate = dict(note=change["gate"]["note"], scope=s)
            else:
                g = apply_edit(g, change)
        if want and want not in ops and not (want == "patch" and "kernel" in ops) and not (want == "program" and ans.get("program")):
            raise Bad(f"this slot asked for operator {want}")
        if ans.get("program"):
            ops.append("program")
            specs = run_program(prog, prog_rows(self.data, self.data.of("show")[:60]))
            for i in self.data.of("show")[:60]:
                clean_spec(specs.get(i.replace("/", "_")), len(self.data.rows[i].get("image_paths") or []))
        if not ops or ops == ["cross"] and any(vh.gid(g) == vh.gid(p["genome"]) and prog == p.get("program") for p in ps):
            raise Bad("no change")
        if gate:                     # the write is fit on the plain rows: rows of its scope reach it plain (in_gate_scope), the parent's program runs on all others; a new program does not come with a note
            ops = [o for o in ops if o != "program"]
            prog = ps[0].get("program") or (ps[1].get("program") if len(ps) == 2 else None)
        mk = lambda pr: member(g, pr, name=None, origin="proposal", parents=names, ops=sorted(set(sum([p.get("ops", []) for p in ps], []) + ops)), d=d + sum(p.get("d", 0) for p in ps), round=r, slot=req["slot"], gate=gate, why=str(ans.get("why"))[:500])
        m = with_gates(mk(prog), ps)
        if gate:
            tag, m = sha(m["mid"] + canon(gate))[:14], with_gates(mk(prog), ps)        # the note is the change: without it every note on one parent has the parent's id and reads as a repeat
            m.update(gtag=tag, mid=tag)     # until the write lands the genome is the parent's, so the proposal keeps its own id; finish_gate gives the real one
        if MENU_OPS[0] is not None:          # 1109c: an operator outside the menu is not available (knobs stay in the grid side table only)
            bad_ = [o for o in ops if o not in set(MENU_OPS[0]) | {"program", "cross", "kernel"}]
            if bad_:
                raise Bad(f"operator not available in this run: {bad_}; available: {sorted(set(MENU_OPS[0]) | {'program'})}")
        m["own_ops"] = list(ops)     # m["ops"] carries the ancestors' too. Taken off again when the row is written
        return m

    def a2(self, r):
        """Amendment A2 (11-08): one gate write per category per round, and no program that abstains on every score row; from its from_round, None before it or without it."""
        x = (jload(f"{self.work}/amendment.json", {}) or {}).get("a2")
        return x if x and r >= x.get("from_round", 10 ** 9) else None

    def a2_check(self, r, m, kids):
        """A2 at parse, before a slot is kept, so the one re-ask can still use it: a gate on a category an earlier valid proposal of this round already took, and a program with fewer than min_score_rows score rows it does not abstain on (those reach stage 1 with no record and are dropped there)."""
        a2 = self.a2(r)
        if not a2:
            return
        sc = (m.get("gate") or {}).get("scope")
        if sc:
            other = next((x["name"] for x, _ in kids if (x.get("gate") or {}).get("scope") == sc), None)
            if other:
                raise Bad(f"gate scope {sc} is already taken this round by {other} (A2: one gate write per category per round); pick another open category")
        if m.get("program"):
            ids = self.data.of("score")
            _, abst = program_rows(self.data, m, ids, render=False)
            if len(ids) - len(abst) < int(a2.get("min_score_rows", 1)):
                raise Bad(f"the program returns None on {len(abst)} of {len(ids)} score rows, so it cannot be scored (A2: at least {int(a2.get('min_score_rows', 1))}); make its trigger match rows of the categories it targets")

    def a4(self, r):
        """Amendment A4 (11-08): cross slots with no fresh pair go to the weakest operator the grid kept, one at a time; the round line's pool on a members-only Table. From its from_round, None before it or without it."""
        x = (jload(f"{self.work}/amendment.json", {}) or {}).get("a4")
        return x if x and r >= x.get("from_round", 10 ** 9) else None

    def a4_cross_op(self, r, S_hist, best, plan, given):
        """A cross slot whose usable pairs were all crossed already (grid_fast, a grid this round). Before A4: weak_op over all of best with the plan as it was, so every such slot took the same operator, and on a plateau round the one grid_slots had just left out for low fit (r4: attn_temp x3).
        A4: weak_op over the operators of best that grid_slots kept in the plan, counting the plan and the operators already given to such slots this round; 'program' when the plan kept none. given is appended to."""
        if not self.a4(r):
            return weak_op(S_hist, best, plan)
        keep = [o for o in sorted(best) if o in plan]
        op = weak_op(S_hist, keep, list(plan) + given) if keep else "program"
        given.append(op)
        self.say(f"r{r}: A4: cross slot without a fresh pair -> {op} (grid kept {keep}; given to such slots this round {given})")
        return op

    def members_tables(self, members, votes):
        """A4: votes, Table, pool, splits and state of the members alone, as compute(members) gives them, from votes already loaded. The reporter builds its Table the same way."""
        vm = {m["name"]: votes[m["name"]] for m in members}
        tab = Table(self.data, self.data.learn, vm)
        pool = tab.pool(self.data, members, vm)
        sp = splits(self.data, tab)
        return vm, tab, pool, sp, state_of(pool, sp, members)

    # ------------------------------------------------------------ 128c (Fable 11-09 §4.1, §4.3, §4.4): judge, scope and fusion ------------------------------
    def c128(self, r=None, key=None, default=None):
        """prereg "c128" from its from_round: the whole block (key None) or one field, default when the block is off or lacks it. Absent: every 128c rule is off and A4 stands."""
        x = (self.a or {}).get("c128")
        on = bool(x) and (r is None or r >= int(x.get("from_round", 1)))
        if key is None:
            return x if on else None
        return x.get(key, default) if on else default

    def c128_replaces(self, r):
        """The 128c judge decides stage 1 and stage 2 (mode replace, the default); mode "and" keeps the old verdict and ANDs the new one into it."""
        return bool(self.c128(r, "judge")) and self.c128(r, "mode", "replace") == "replace"

    def no_id(self, ids, where):
        """ID guard: a decision reads LEARN rows only. ID (IID half 1) and OOD rows raise; so does any row with no LEARN role."""
        bad = [i for i in ids if i not in self.data.role]
        if bad:
            raise RuntimeError(f"128c ID guard: {where} asked for {len(bad)} rows outside LEARN (ID or OOD), e.g. {bad[:3]}")
        return ids

    def c128_rotate(self, r):
        """Split refresh (a): this round's SELECT, dealt by the preregistered seed schedule. Logged, and written to the round dir."""
        if self.c128(r, "refresh") != "rotate":
            return None
        out = self.data.rotate(r, int(self.c128(r, "rotate_seed", 1109)))
        self.say(f"r{r}: 128c SELECT rotation (seed {self.c128(r, 'rotate_seed', 1109)}): select/score rows per bench {out}")
        jdump(f"{self.work}/r{r:03d}/rotation.json", dict(round=r, benches=out, select=self.data.of("select")))
        return out

    def c128_scope_check(self, r, m):
        """§4.1: a program is mandatory for a member with engine edits; it defines the member's scope (where it does not return None). A gate member's scope is its category."""
        if self.c128(r, "scope") and has_engine(m) and not m.get("gate") and not m.get("program"):
            raise Bad("a member with engine edits must carry a program: the program defines where the edited member answers (return None elsewhere), and outside that scope it casts no vote")

    def c128_critic(self, r, rd, kids):
        """§4.4.6: before any engine time, the frozen model reads each valid proposal and says whether it encodes an answer. 'yes' drops the proposal. A failed critic call drops nothing (logged)."""
        if not self.c128(r, "critic"):
            return kids
        live = [(m, row) for m, row in kids if row.get("valid")]
        if not live:
            return kids
        d = f"{rd}/critic"
        os.makedirs(d, exist_ok=True)
        if not os.path.exists(f"{d}/prompts.json"):
            jdump(f"{d}/prompts.json", [dict(id=f"leak{row['slot']}", slot=row["slot"], system=CRITIC_SYSTEM, user=CRITIC_ASK + "\n\n" + critic_text(m)) for m, row in live])
        try:
            got = {x["id"]: x for x in self.generate(jload(f"{d}/prompts.json"), d)}
        except (RuntimeError, Bad, OSError) as e:
            self.say(f"r{r}: 128c leakage critic failed, nothing dropped: {str(e)[:200]}")
            return kids
        for m, row in live:
            v = critic_verdict((got.get(f"leak{row['slot']}") or {}).get("content"))
            row["leak_critic"] = v
            if v == "yes":
                row.update(valid=False, status="invalid: leakage critic: " + str((got.get(f"leak{row['slot']}") or {}).get("content"))[:300])
        self.say(f"r{r}: 128c leakage critic: " + ", ".join(f"{row.get('name')} {row.get('leak_critic')}" for _, row in live))
        return [(m, row) for m, row in kids if row.get("valid")]

    def c128_draw(self, p, d):
        """Draw d of parent p in its current epoch (a null that passed moved the epoch on)."""
        return draw_of(p, d + DRAW_EPOCH * int((self.state.get("c128_epoch") or {}).get(p["name"], 0)))

    def c128_request(self, r, kids, by, qm, q_full):
        """Asked beside stage 1, for every parent of this round's candidates: band draws 1-3 and the null's draw 4 on the rows its children are read on and on SELECT; draw 4 also on all score rows (the null's stage 2 rows, as a candidate's);
        draws 3 and 5 on SELECT for the 4-sample stage 2. On disk, cached while the parent is unchanged: evaluate_many asks only rows not there."""
        if not self.c128(r, "judge"):
            return
        sel = self.no_id(self.data.of("select"), "draw requests")
        reqs = []
        for pn in sorted({m["parents"][0] for m, _ in kids}):
            p = by[pn]
            rows = sorted(set().union(*[set(qm(m)) for m, _ in kids if m["parents"][0] == pn]) | set(sel))
            reqs += [(self.c128_draw(p, d), rows) for d in BAND_DRAWS]
            if self.c128(r, "null", True):
                reqs += [(self.c128_draw(p, NULL_DRAW), sorted(set(rows) | set(q_full))), (self.c128_draw(p, NULL_DRAW2), sel)]
        self.ev.evaluate_many(reqs)

    def c128_request_s2(self, r, top):
        """Each stage 2 candidate's own draw 3 on SELECT: its samples 2,3, paired by seed with the parent's draw 3."""
        if self.c128(r, "judge") and int(self.c128(r, "s2_samples", 4)) == 4:
            self.ev.evaluate_many([(draw_of(m, PAIR_DRAW), self.data.of("select")) for m, row in top if not row.get("null")])

    def c128_votes(self, members, more):
        """votes, Table, splits and the members' fused pool on one Table that holds every vote of members + more (draws included), so a draw's answer has its own column."""
        allm = list(members) + [m for m in more if m["name"] not in {x["name"] for x in members}]
        votes = self.votes(allm)
        tab = Table(self.data, self.data.learn, votes)
        sp = splits(self.data, tab)
        fu = self.fz("fuse", self.fuser.fuse, members, votes, self.data, self.fusion, sp=sp)
        return votes, tab, sp, fu

    def c128_stage1(self, r, rd, members, have, qm, by):
        """Stage 1 (§4.4.2): on its fired set in scope, |F| >= fired_min and net paired rows on F >= max(0, band mean) + band delta, the band being the parent's three draws read on the same F.
        zx stays the ranking. The null competitor of every parent (draw 4) goes through the same test. Returns stage 2's list: confirmations first (refresh cap), passers by zx, nulls that passed."""
        c = self.c128(r)
        fmin, k = int(c.get("fired_min", 8)), float(c.get("band_k", 1.65))
        nulls = []
        if c.get("null", True):
            for pn in sorted({m["parents"][0] for m, _ in have}):
                rows = sorted(set().union(*[set(qm(m)) for m, _ in have if m["parents"][0] == pn]))
                nm = dict(self.c128_draw(by[pn], NULL_DRAW), parents=[pn], null_of=pn, slot=f"null_{pn}")
                nulls.append((nm, dict(round=r, slot=nm["slot"], name=nm["name"], mid=nm["mid"], ops=["null"], own_ops=["null"], valid=True, scored=True, null=True, null_of=pn, c128_rows=rows)))
        cand = list(have) + nulls
        pm = {m["parents"][0]: by[m["parents"][0]] for m, _ in cand}
        votes, tab, sp, fu = self.c128_votes(members, [m for m, _ in cand] + [self.c128_draw(p, d) for p in pm.values() for d in BAND_DRAWS])
        pa = pool_cols(fu)
        for m, row in cand:
            p = pm[m["parents"][0]]
            ids = self.no_id(row.get("c128_rows") or list(qm(m)), "stage 1")
            vc, vp = votes[m["name"]], votes[p["name"]]
            F = fired_rows(tab, vc, pa, ids)
            stat = (lambda mv: vs_pool(tab, mv, pa, F)) if c.get("stat", "pool") == "pool" else (lambda mv: paired(mv, vp, F))      # pool: §4.3 |F| (q_m - q_a); parent: paired right rate against the parent's own samples
            net = float(sum(stat(vc).values()))
            b = band_of([float(sum(stat(votes[self.c128_draw(p, d)["name"]]).values())) for d in BAND_DRAWS], k)
            thr = None if b["delta"] is None else (b["delta"] if c.get("band_center", "mean") == "zero" else max(0., b["mean"]) + b["delta"])     # "zero": the spec's letter, net >= delta_F; "mean" (default): beat what the re-seeded parent does on F, max(0, its mean) + delta
            ok = len(F) >= fmin and thr is not None and net >= thr
            if row.get("null"):
                zz = zx_stats(tab, vp, vc, ids, self.data.cluster)
                row.update(zx_score=zz["zx"], dacc_paired_score=zz["dacc"], strata_score=zz.get("strata"))
            row["c128_s1"] = dict(rows=len(ids), fired=len(F), net=net, band=b, thr=thr, passes=bool(ok), why=None if ok else ("fired < %d" % fmin if len(F) < fmin else "no band" if thr is None else "net inside the null band"))
            self.say(f"  128c stage 1 {m['name']}: fired {len(F)} of {len(ids)} | net {net:+.2f} rows vs band {b['values']} -> thr {thr if thr is None else round(thr, 2)} -> {'pass' if ok else 'fail: ' + row['c128_s1']['why']}")
        n = int(c.get("stage2_max") or self.a.get("stage2_wide") or self.a["stage2"])
        passed = sorted([x for x in have if x[1]["c128_s1"]["passes"]], key=lambda x: -float(x[1].get("zx_score") or 0))
        conf = []
        for m0 in (self.state.get("c128_confirm") or []) if c.get("refresh") == "cap" and c.get("confirm", True) else []:
            m1 = dict(m0, slot=f"conf_{m0['mid']}")
            conf.append((m1, dict(round=r, slot=m1["slot"], name=m1["name"], mid=m1["mid"], ops=m1.get("ops"), own_ops=m1.get("own_ops"), valid=True, scored=False, card_rows=m1.get("card_rows"), confirm_of=m0.get("confirm_from"), stage2_by="confirm", narrow=m0.get("narrow"))))
        top = conf + passed
        cap = int(c.get("read_cap", 3)) if c.get("refresh") == "cap" else n + len(conf)
        for m, row in top[cap:]:
            row["stage2_by"] = "over the SELECT read cap"
        top = top[:cap]
        for m, row in top:
            row.setdefault("stage2_by", "128c stage 1")
        nl = [x for x in nulls if x[1]["c128_s1"]["passes"]]
        jdump(f"{rd}/c128_stage1.json", [dict(name=row["name"], null=bool(row.get("null")), zx=row.get("zx_score"), **row["c128_s1"]) for _, row in cand])
        self.say(f"r{r}: 128c stage 2 list: {[m['name'] for m, _ in top]}; nulls passing stage 1 (read in stage 2): {[m['name'] for m, _ in nl]}; nulls failing: {[m['name'] for m, row in nulls if not row['c128_s1']['passes']]}")
        return top + nl

    def c128_decide(self, r, members, top, by):
        """Stage 2 on SELECT (§4.4.3), after the round's usual stage 2 numbers: in scope, 4 samples (base + draw 3, the parent's base + draw 3), on the fired set:
        HarnessBank paired z >= z_min with mean > 0, the pool-increment CE LB > 0 (as now), DarwinX no-harm R(m) <= the band of R over the parent's draws on the same rows (2 samples each, against the parent's own),
        the card-vs-unseen rule, and (scope on) the diversity floor for weight members. |F| < fired_min: no single read; the other checks hold, the joint read decides. The null (draws 4+5) is read the same way and never accepted."""
        c = self.c128(r)
        fmin, k, zmin, s4 = int(c.get("fired_min", 8)), float(c.get("band_k", 1.65)), float(c.get("z_min", 1.96)), int(c.get("s2_samples", 4)) == 4
        sel = self.no_id(self.data.of("select"), "stage 2")
        more = []
        for m, row in top:
            p = by[m["parents"][0]]
            more += [m, p] + [self.c128_draw(p, d) for d in BAND_DRAWS]
            if s4:
                more.append(self.c128_draw(p, NULL_DRAW2) if row.get("null") else draw_of(m, PAIR_DRAW))
        seen, uniq = set(), []
        for x in more:
            if x["name"] not in seen:
                seen.add(x["name"]); uniq.append(x)
        votes, tab, sp, fu = self.c128_votes(members, uniq)
        pa = pool_cols(fu)
        mvotes = [votes[x["name"]] for x in members]
        out = dict(null_pass=[], small=[], read=[])
        for m, row in top:
            p = by[m["parents"][0]]
            vc, vp = votes[m["name"]], votes[p["name"]]
            if row.get("stage2_status") != "ok":
                row["c128"] = dict(passes=False, why=f"stage 2 status {row.get('stage2_status')}")
                row["would_accept"] = False
                continue
            if s4:
                c4 = merge4(vc, votes[(self.c128_draw(p, NULL_DRAW2) if row.get("null") else draw_of(m, PAIR_DRAW))["name"]])
                p4 = merge4(vp, votes[self.c128_draw(p, PAIR_DRAW)["name"]])
            else:
                c4, p4 = vc, vp
            scope = [i for i in sel if vc.get(i)]
            F = fired_rows(tab, c4, pa, scope)
            zz = hb_z((vs_pool(tab, c4, pa, F) if c.get("stat", "pool") == "pool" else paired(c4, p4, F)).values())
            R = harm(paired(vc, vp, scope))
            bR = band_of([harm(paired(votes[self.c128_draw(p, d)["name"]], vp, scope)) for d in BAND_DRAWS], k)
            harm_ok = bR["delta"] is not None and R <= bR["mean"] + bR["delta"]
            lb_ok = bool(gain_positive(row))
            card_ok = not (c.get("card_unseen", True) and (row.get("card_rows_n") or 0) > 0 and (row.get("dacc_card") or 0) > 0 and (row.get("dacc_fit") or 0) <= 0)
            div, div_ok = None, True
            if c.get("diversity") and specialist(m) and not row.get("null"):
                fitsc = [i for i in self.data.learn if self.data.role[i] in ("score", "select") and vc.get(i)]
                div = diversity(tab, vc, vp, fitsc, mvotes)
                div_ok = div["share"] >= float(c.get("div_share", .2)) and div["right_on_none"] >= int(c.get("div_none", 1))
            big = len(F) >= fmin
            z_ok = big and zz["mean"] is not None and zz["mean"] > 0 and zz["z"] >= zmin
            passes = bool(z_ok and lb_ok and harm_ok and card_ok and div_ok)
            small = bool(not big and harm_ok and card_ok and div_ok)
            why = [n for n, t in (("z", z_ok or not big), ("pool_lb", lb_ok), ("no_harm", harm_ok), ("card_vs_unseen", card_ok), ("diversity", div_ok)) if not t] + (["fired<%d: joint read only" % fmin] if not big else [])
            row["c128"] = dict(scope_rows=len(scope), fired=len(F), z=zz, harm=R, harm_band=bR, harm_thr=None if bR["delta"] is None else bR["mean"] + bR["delta"], pool_lb=row.get("ce_gain_select_lb"), card_ok=card_ok, diversity=div, passes=passes, small=small, why=why, samples=4 if s4 else 2)
            row["old_would_accept"] = bool(row.get("would_accept"))
            out["read"].append(m["name"])
            if row.get("null"):
                row["would_accept"] = False
                if passes or (small and lb_ok):
                    out["null_pass"].append(m["name"])
            else:
                new = passes and (row["old_would_accept"] if not self.c128_replaces(r) else True)
                if new and c.get("refresh") == "cap" and c.get("confirm", True) and not row.get("narrow") and not row.get("confirm_of"):
                    row["c128_confirm_next"] = True          # a wide member: accepted only if the next round's SELECT confirms it
                    new = False
                row["would_accept"] = bool(new)
                if small and not row.get("confirm_of"):
                    out["small"].append(m["name"])
                if self.c128_replaces(r):
                    row["rejected_by"] = why
            self.say(f"  128c stage 2 {m['name']}{' (null)' if row.get('null') else ''}: scope {len(scope)} fired {len(F)} z {zz['z']} mean {zz['mean']} | no-harm R {R:.2f} vs {row['c128']['harm_thr']} | pool LB {row.get('ce_gain_select_lb')} | card ok {card_ok} | diversity {div} -> {'PASS' if passes else 'joint read' if small else 'fail'} {why}")
        return out

    def c128_finish(self, r, rd, members, top, chosen, steps, dec):
        """After accept_in_order: one joint read of the small-fired members on the pool with the round's accepted (CE LB > 0, accuracy not down), then the null's verdict: a null that passed voids every acceptance of the round
        and moves its parent's draws to a new epoch (the band is measured again). Wide passers waiting for confirmation go to state (refresh cap)."""
        S, by_n = self.state, {m["name"]: (m, row) for m, row in top}
        joint = None
        small = [by_n[n] for n in dec["small"] if n in by_n]
        if small and self.c128(r, "joint", True):
            now = list(members) + list(chosen)
            votes, sp = self.fit_rows_read(now + [m for m, _ in small])
            f0 = self.fz("fuse", self.fuser.fuse, now, votes, self.data, self.fusion, sp=sp)
            f1 = self.fz("fuse", self.fuser.fuse, now + [m for m, _ in small], votes, self.data, self.fusion, sp=f0.sp)
            joint = dict(self.set_gain(f0, f1), members=[m["name"] for m, _ in small], against=[x["name"] for x in now])
            joint["passes"] = bool(float(joint["ce_gain_select_lb"] or 0) > GAIN_TOL and float(joint["dacc_select"]) >= -GAIN_TOL)
            self.say(f"r{r}: 128c joint read of {joint['members']}: LB {joint['ce_gain_select_lb']} dacc {joint['dacc_select']} rows up {joint['rows_up']} down {joint['rows_down']} -> {'ACCEPT' if joint['passes'] else 'no'}")
            if joint["passes"]:
                for m, row in small:
                    chosen.append(m)
                    row.update(accepted=True, accept_order=len(chosen), joint=True)
                steps.append(dict(step=len(chosen), accepted=joint["members"], rule="128c joint read", lb=joint["ce_gain_select_lb"], dacc=joint["dacc_select"], against=joint["against"]))
        void = None
        if dec["null_pass"]:
            void = [m["name"] for m in chosen]
            for m in chosen:
                by_n[m["name"]][1].update(accepted=False, void=True)
            del chosen[:]
            ep = S.setdefault("c128_epoch", {})
            for n in dec["null_pass"]:
                pn = by_n[n][0]["null_of"]
                ep[pn] = int(ep.get(pn, 0)) + 1
            steps.append(dict(rule="128c void", nulls=dec["null_pass"], voided=void))
            self.say(f"r{r}: 128c NULL PASSED ({dec['null_pass']}): the round's acceptances are void ({void}); band epochs now {ep}")
        if self.c128(r, "refresh") == "cap" and self.c128(r, "confirm", True):
            S["c128_confirm"] = [] if void is not None else [dict(m, confirm_from=r, narrow=row.get("narrow"), card_rows=row.get("card_rows")) for m, row in top if row.get("c128_confirm_next")]
            if S["c128_confirm"]:
                self.say(f"r{r}: 128c wide passers waiting for next round's SELECT confirmation: {[m['name'] for m in S['c128_confirm']]}")
        rec = dict(round=r, joint=joint, void=void, null_pass=dec["null_pass"], read=dec["read"], rows=[dict(name=row.get("name"), null=bool(row.get("null")), confirm_of=row.get("confirm_of"), s1=row.get("c128_s1"), s2=row.get("c128"), accepted=bool(row.get("accepted"))) for _, row in top])
        jdump(f"{rd}/c128.json", rec)
        return rec

    def a3(self, r):
        """Amendment A3 (11-08): distinct open keys for parallel re-asks of A2 gate-key refusals; from its from_round, None before it or without it."""
        x = (jload(f"{self.work}/amendment.json", {}) or {}).get("a3")
        return x if x and r >= x.get("from_round", 10 ** 9) else None

    def a3_assign(self, r, pending, kids):
        """A3, before the re-ask prompts: the slots A2 refused for a gate key already taken this round are re-asked in parallel, so two of them can pick the same open key (r4: s05 and s09 both took charxiv|number-in-general). With two or more such slots, each gets its own still-open key by slot order: open = this round's offer less the keys valid proposals took. Slots beyond the open keys keep the A2 reason (they stay invalid as before). Nothing without A3, without an offer, or with fewer than two such slots."""
        out = dict(colliding=[], open=[], assigned={}, extra=[])
        if not self.a3(r) or getattr(self, "gate_offer", None) is None:
            return out
        col = sorted(q["slot"] for q, _, row in pending if re.search(r"gate scope cat:.+ is already taken this round by \S+ \(A2:", str(row.get("status"))))
        if len(col) < 2:
            return out
        taken = {str((m.get("gate") or {}).get("scope") or "")[4:] for m, _ in kids if (m.get("gate") or {}).get("scope")}
        free = [k for k in sorted(self.gate_offer) if k not in taken]
        out.update(colliding=col, open=free, assigned=dict(zip(col, free)), extra=col[len(free):])
        self.say(f"r{r}: A3: {len(col)} re-asks for a taken gate key, slots {col}; open keys {free}; assigned " + (", ".join(f"s{s} cat:{k}" for s, k in out["assigned"].items()) or "none"))
        if out["extra"]:
            self.say(f"r{r}: A3: slots {out['extra']} get no key: {len(col)} colliding slots but {len(free)} open keys; they keep the A2 reason and stay invalid unless their answer finds a key no one took")
        return out

    def a3_why(self, a3_, slot, why_):
        """A3: the re-ask reason of a slot that was assigned a key names that key; every other reason is unchanged."""
        k = a3_["assigned"].get(slot)
        return why_ if k is None else why_.replace("; pick another open category", "") + f"; use cat:{k} (assigned to you; other open keys are claimed by other re-asks)"

    def a3_final(self, r, a3_, pending):
        """A3, after the re-ask: the parse (a2_check against the valid proposals so far, in slot order) is the final check; each colliding slot that is still invalid gets a log line saying why."""
        if not a3_["colliding"]:
            return
        after = {}
        for q, _, row in pending:
            s = q["slot"]
            if s not in a3_["colliding"]:
                continue
            after[str(s)] = "valid " + str(row.get("name")) if row["valid"] else str(row.get("status"))
            if row["valid"]:
                continue
            k = a3_["assigned"].get(s)
            why = (f"assigned cat:{k}, " if k else f"no open key left ({len(a3_['colliding'])} colliding slots, {len(a3_['open'])} open keys), ") + (f"re-ask answer refused: {str(row.get('status')).replace('invalid: ', '', 1)[:300]}" if row.get("status") != row.get("first_status") else "no new refusal reason (no re-ask answer, or the same refusal)")
            self.say(f"r{r}: A3: slot {s} stays invalid: {why}")
        a3_["after"] = after

    def a1(self, r):
        """Amendment A1 (11-08): gate eligibility, early drop and gate slot cap from its from_round; None before it or without it."""
        x = (jload(f"{self.work}/amendment.json", {}) or {}).get("a1")
        return x if x and r >= x.get("from_round", 10 ** 9) else None

    def gate_rows(self, cat):
        """FIT rows (learn outside SELECT) of a gate key, the rows the writer fits and reads."""
        return [i for i in self.data.learn if self.data.role[i] != "select" and vh.key(self.data.rows[i]) == cat]

    def gate_ok(self, ids, a1):
        """A1's size rule on a key's rows: writer split fit >= a1 fit, read >= a1 read, and score rows >= a1 score (stage 1 asks a gate member only score rows of its key). (ok, fit, read, score)."""
        nf_, nr_ = self.gate_split(ids)
        ns_ = sum(1 for i in ids if self.data.role[i] == "score")
        return nf_ >= a1["fit"] and nr_ >= a1["read"] and ns_ >= a1["score"], nf_, nr_, ns_

    def gate_written(self, r):
        """Keys a gate write of this run has already been made for in a round before r: a hist row with own op gate whose write file landed."""
        out = set()
        for p in self.state.get("hist", []):
            if int(p.get("round", 0)) < r and "gate" in (p.get("own_ops") or []) and p.get("write_file") and os.path.exists(p["write_file"]):
                d = os.path.dirname(p["write_file"])
                try:
                    k = open(f"{d}/note.tsv").read()      # the scope is not in the row; the writer's command named it, and its key rows are in traces.jsonl
                except OSError:
                    continue
                ids = {json.loads(l)["id"] for l in open(f"{d}/traces.jsonl")} if os.path.exists(f"{d}/traces.jsonl") else set()
                out |= {vh.key(self.data.rows[i]) for i in ids if i in self.data.rows}
        return out

    def gate_open(self, r, cats):
        """A1: the show keys a gate may name this round: size rule passed and not written yet in this run. All of cats without A1."""
        a1 = self.a1(r)
        if not a1:
            return list(cats), None
        done = self.gate_written(r)
        info = {c: self.gate_ok(self.gate_rows(c), a1) for c in cats}
        ok = [c for c in cats if info[c][0] and c not in done]
        return ok, dict(eligible=[c for c in cats if info[c][0]], written=sorted(done & {c for c in cats if info[c][0]}), open=ok, rule=dict(fit=a1["fit"], read=a1["read"], score=a1["score"]))

    def gate_split(self, ids):
        """mm_gate.py's split of the traced rows into fit and read: crc32 parity or, when all fall on side 0, a quarter by sha256. Counts only: the writer does its own split."""
        half = lambda i: zlib.crc32(i.encode()) % 2
        if all(half(i) == 0 for i in ids):
            half = lambda i: int(hashlib.sha256(i.encode()).digest()[0] % 4 == 0)
        return sum(half(i) == 0 for i in ids), sum(half(i) == 1 for i in ids)

    def write_gate(self, m, parent):
        """The last-layer write. Rows of the category outside SELECT and outside ID fit it; the parent's traces go in a file made for this write."""
        tag = m.get("gtag", m["mid"])
        d = f"{EVR}/gate/{tag}"
        os.makedirs(d, exist_ok=True)
        f = f"{d}/gt_{tag}.json"
        n = len(glob.glob(f + ".*"))
        if os.path.exists(f + ".failed") and n < 3:                  # a failed write gets one more try under a new job name
            os.replace(f + ".failed", f"{f}.failed{n}")
        if os.path.exists(f) or os.path.exists(f + ".failed"):
            return f
        s = m["gate"]["scope"]
        cat = s[4:]
        fit = [i for i in self.data.learn if self.data.role[i] != "select"]
        G = [i for i in fit if vh.key(self.data.rows[i]) == cat]
        bare = member(parent["genome"], None, name="bare")
        dump = {}
        for dd in member_dirs(bare):
            for x in answer_files(dd):
                for l in open(x):
                    r = json.loads(l)
                    if r["id"] in set(fit) and r.get("sample", 0) < 2 and r.get("finish") not in NO_RUN:
                        g_, ok_, _ = regrade(r, self.data.rows[r["id"]])          # the writer reads ok: from the pool row, as every other reader
                        r.update({k: g_[k] for k in GRADE_FIELDS if k in g_}, ok=ok_)
                        dump.setdefault((r["id"], r.get("sample", 0)), json.dumps(r, ensure_ascii=False) + "\n")
        wide = None
        kprompts = None
        c9_ = self.HW.cfg(self) if getattr(self, "HW", None) is not None and (self.a or {}).get("c1109") else {}
        if c9_.get("gate_key") == "prompts":          # gate_v3 (11-09, off by default): the key also fits on >= gate_key_min UNLABELED in-scope prompts from c1109.gate_prompts; ID/OOD excluded by id and image sha, pool rows by id
            bad_id = set(self.data.id_rows) | set(self.data.ood_rows) | set(self.data.rows)
            bad_sha = {x for i in list(self.data.id_rows) + list(self.data.ood_rows) for x in (self.data.rows[i].get("img_sha256") or [])}
            kprompts = [r for r in vh.read_jsonl([c9_["gate_prompts"]]).values() if vh.key(r) == cat and r["id"] not in bad_id and r.get("image_paths") and not set(r.get("img_sha256") or []) & bad_sha]
            if len(kprompts) < c9_.get("gate_key_min", 20):
                m["gate_drop"] = f"gate_v3 key {cat}: {len(kprompts)} unlabeled prompts after the ID/OOD exclusion < {c9_.get('gate_key_min', 20)}"
                return f
        if c9_:      # V1 (11-09, off by default): the trigger key is fit on every FIT row of the scope's bench, not on its 2-7 category rows
            if c9_.get("gate_key") == "bench":
                b_ = cat.split("|")[0]
                G = [i for i in fit if (self.data.rows[i].get("bench") or i.split("/")[0]) == b_]
                wide = c9_.get("gate_key_min", 40)
        gs = set(G)
        traced = sorted({i for (i, _), l in dump.items() if i in gs and json.loads(l).get("reasoning") and self.data.rows[i].get("image_paths")})
        if wide is not None and len(traced) < wide:
            m["gate_drop"] = f"V1 bench key {cat.split('|')[0]}: {len(traced)} traced FIT rows < {wide}"
            return f
        nf_, nr_ = self.gate_split(traced)
        if not nf_ or not nr_:         # the writer stops on an empty side: no engine time, no retries, one logged reason
            m["gate_drop"] = f"category {cat} has {len(G)} FIT rows, {len(traced)} traced: {nf_} to fit and {nr_} to read, the writer needs both"
            return f
        a1 = None if (wide is not None or kprompts is not None) else self.a1(getattr(self, "cur_round", self.state.get("round", 0) + 1))      # V1: the size rule is the bench minimum above
        if a1:                         # A1: the same size rule as the offer, on the traced rows; no engine time, no retries
            ok_, f1, r1, s1 = self.gate_ok(traced, a1)
            if not ok_:
                m["gate_drop"] = f"category {cat} has {len(G)} FIT rows, {len(traced)} traced: {f1} to fit, {r1} to read, {s1} score rows; A1 needs fit>={a1['fit']} read>={a1['read']} score>={a1['score']}"
                return f
        if not os.path.exists(f"{d}/others.jsonl"):
            with open(f"{d}/traces.jsonl", "w") as o:
                for (i, _), l in dump.items():
                    if i in gs:
                        o.write(l)
            with open(f"{d}/others.jsonl", "w") as o:
                for l in dump.values():
                    o.write(l)
        note_lines = f"{tag}\t{m['gate']['note']}\n"
        if kprompts is not None and c9_.get("gate_control_note"):     # gate_v3: a matched wrong-content control written with the same keys in the same job (arm gt_ctrl_<tag>.json); the judge reads it beside the gate
            golds = collections.defaultdict(set)
            for i_ in self.data.learn:
                golds[vh.key(self.data.rows[i_])].add(str(self.data.rows[i_].get("answer")))
            cn_, src_ = control_note(m["gate"]["note"], cat, self.state.get("c1109_gate_notes") or [], golds, c9_["gate_control_note"], tag)
            m["gate_control_src"] = src_
            note_lines += f"ctrl_{tag}\t{cn_}\n"
            m["gate_control_file"] = f"{d}/gt_ctrl_{tag}.json"
            m["gate_control_note"] = cn_
        open(f"{d}/note.tsv", "w").write(note_lines)
        keyarg = f"--cat {shlex.quote(cat.replace(' ', '~'))}"
        if kprompts is not None:
            with open(f"{d}/key_prompts.jsonl", "w") as o:
                for r in kprompts:
                    o.write(json.dumps({k: v for k, v in r.items() if k not in ("answer", "options_answer")}, ensure_ascii=False) + "\n")
            keyarg += f" --key-prompts {d}/key_prompts.jsonl"
        if wide is not None:           # V1: --plain = the bench's traced FIT rows (pool records); mm_gate fits the key on them and reads its unseen check on its own held-out quarter
            with open(f"{d}/plain.jsonl", "w") as o:
                for i in traced:
                    o.write(json.dumps(self.data.rows[i], ensure_ascii=False) + "\n")
            keyarg = f"--plain {d}/plain.jsonl"
        en = f" --engine {self.ev.arm(parent['genome'])}" if parent["genome"]["engine"] else ""
        if GEN_KEY.get("effort", "medium") != "medium":         # 128: the writer's chat template at the run's effort (mm_gate --effort); medium leaves the command as before
            en += f" --effort {GEN_KEY['effort']}"
        if GATE_TRACE_CAP:                                       # 128 prereg gate_trace_cap: the writer's trace cut and length filter (mm_gate's 3000 before)
            en += f" --trace-cap {GATE_TRACE_CAP}"
        self.__dict__.setdefault("write_jobs", {}).setdefault(f, []).append((f"hb{self.ev.tag}_w{n or ''}_{tag[:10]}", time.time()))
        self.ev.submit(f"w{n or ''}_{tag[:10]}", f"cd {TREE} && python -u rsi/mm_gate.py --model {self.ev.model(parent['genome'])} --traces {d}/traces.jsonl --others {d}/others.jsonl --pool {getattr(self, "pool_file", IID)} --notes {d}/note.tsv --out {d}/gt --arms {d} {keyarg} --strict-half0{en}", [f], f + ".failed")
        return f

    def write_clock(self, f):
        """(submitted, started) of the write's latest attempt: started is when a card claimed its line (or its _v copy), None while no card has."""
        js = getattr(self, "write_jobs", {}).get(f)
        if not js:
            return None, None
        n, t_ = js[-1]
        cl = [f"{HB}/claims/{x}" for x in (n, n + "_v") if os.path.exists(f"{HB}/claims/{x}")]
        return t_, (min(os.path.getmtime(x) for x in cl) if cl else None)

    def write_late(self, f, row):
        """The write's reason to stop waiting, or None: 1500 s after a card started it, or an hour in the queue with no card on it."""
        t_, st = self.write_clock(f)
        if t_ is None:
            return None
        now = time.time()
        row.update(write_submit=round(t_, 1), write_start=None if st is None else round(st, 1))
        if st is None:
            return f"write: never started, {(now - t_) / 60:.0f} min in the queue" if now - t_ > WRITE_QUEUE_MAX else None
        return f"write: past its deadline, {now - st:.0f}s after a card started it ({st - t_:.0f}s in the queue)" if now - st > WRITE_RUN_MAX else None

    @staticmethod
    def write_class(row):
        """ops_report: what became of a gate write. used / used late (landed while the round still waited on other jobs) / built late (landed after the round let it go) / not built / gate-check rejected / no write possible."""
        st, f = str(row.get("status") or ""), row.get("write_file")
        if row.get("scored") or row.get("valid"):
            return "used late" if row.get("write_used_late") else "used"
        if "the writer needs both" in st:
            return "no write possible"
        if "deadline" in st or "never started" in st:
            return "built late" if f and os.path.exists(f) else "not built"
        if "failed three times" in st or "write failed" in st:
            return "not built"
        if st.startswith("write:"):
            return "gate-check rejected"
        return None

    def finish_gate(self, m, parent, f):
        if not os.path.exists(f):
            raise Bad("write failed")
        for k_ in range(30):         # mm_gate writes this file in place (13 MB): a read during the write sees it cut, not bad
            try:
                arm = json.load(open(f))[0]["engine"]
                break
            except ValueError:
                if k_ == 29:
                    raise Bad("write file unreadable for 5 minutes")
                time.sleep(10)
        pe = parent["genome"]["engine"]
        if pe and canon(arm[:len(pe)]) != canon(pe):      # the writer emits only its own entries, after the parent's
            arm = list(pe) + arm
        for e in arm[len(pe):]:
            if not np.isfinite(np.array(e["value"], dtype=float)).all():
                raise Bad("not finite")
        new, old = vh.units(arm[len(pe):]), vh.units(pe)
        for k in new:
            if new[k] & old.get(k, set()):
                raise Bad("units reused")
        for t in glob.glob(f"{os.path.dirname(f)}/gt.*.txt"):
            t = open(t).read()
            x = re.search(r"unseen: all-text p median ([\d.]+)", t)
            y = re.search(r"p\(start the note\) mean ([\d.]+)", t)
            c9f = self.HW.cfg(self) if getattr(self, "HW", None) is not None and (self.a or {}).get("c1109") else {}
            rmin, mmax = (c9f.get("gate_recite_min", .95), c9f.get("gate_misfire_max", .01)) if c9f.get("gate_key") == "prompts" else (.5, .01)    # gate_v3: writer checks recital >= .95, misfire <= .01
            if x and float(x.group(1)) < rmin:
                raise Bad(f"write did not reach unseen problems, median {x.group(1)}")
            if y and float(y.group(1)) > mmax:
                raise Bad(f"note starts on other tasks, {y.group(1)}")
        g = dict(parent["genome"], engine=arm)
        n = member(g, m.get("program"), origin="proposal")
        m.update(genome=n["genome"], mid=n["mid"], note=m["gate"]["note"])
        if m.get("program") and m.get("inherited_gates"):
            m["mid"] = sha(m["mid"] + canon(m["inherited_gates"]))[:14]
        cf = m.get("gate_control_file")
        if cf:                       # gate_v3 (1109c-final): the matched wrong-content control, same keys, same program, same scope; the judge reads it beside the gate
            try:
                ca = json.load(open(cf))[0]["engine"]
                if pe and canon(ca[:len(pe)]) != canon(pe):
                    ca = list(pe) + ca
                if not all(np.isfinite(np.array(e["value"], dtype=float)).all() for e in ca[len(pe):]):
                    raise ValueError("control not finite")
                cn = member(dict(parent["genome"], engine=ca), m.get("program"), origin="gate_control")
                cmid = sha(cn["mid"] + canon(m["inherited_gates"]))[:14] if m.get("program") and m.get("inherited_gates") else cn["mid"]
                m["gate_control"] = dict(name=f"{m['name']}~ctrl", mid=cmid, genome=cn["genome"], note=m.get("gate_control_note") or (self.HW.cfg(self) or {}).get("gate_control_note", ""), src=m.get("gate_control_src"))
            except (OSError, ValueError, KeyError, IndexError) as e:
                m["gate_control"] = dict(missing=f"{type(e).__name__}: {e}"[:200])
        return m

    def round(self, r):
        self.cur_round = r                 # A1: write_gate reads the round its rule applies to
        from vrsi27.boost.edge import proposal_stats, learn_gain, atom_scores, select_stats
        from vrsi27.boost.meta import Population, board, parse_lens_code
        from vrsi27.boost.lens import lens_hash
        t0 = time.time()
        a, S = self.a, self.state
        rd = f"{self.work}/r{r:03d}"
        os.makedirs(rd, exist_ok=True)
        if self.c9:
            self.HW.round_start(self, sys.modules[__name__], r)      # 1109c: anchor guard, W admission from the probe files, SELECT rotation
        else:
            self.c128_rotate(r)              # 128c split refresh (a): this round's SELECT; off without prereg c128 refresh "rotate"
        self.stage(r, "prompts")
        members = S["members"]
        REG.update({m["name"]: m for m in members})
        learn = self.data.learn
        self.ensure(members, learn)
        if self.c9:                      # 1109c: each provisional member's e-process read on this round's (rotated) SELECT; a removed member leaves the pool here
            self.c9_eproc = self.HW.confirm_reads(self, sys.modules[__name__], r, members)
            members = S["members"]
        votes, tab, pool, sp, st = self.compute(members)
        ec, Cor = err_stats(pool, members, sp.fit)
        pop = Population(f"{self.work}/meta")
        hist = S["hist"]
        brd = board(pop, hist)
        rng = np.random.default_rng([a["seed"], r])
        st_cards = st
        if self.fusion != "stacking":                 # residual cards and pool answers from the fusion; parent choice, grid and every gate keep st
            st_cards = self.fuser.pool_state(st, self.fz("fuse", self.fuser.fuse, members, votes, self.data, self.fusion, sp=sp))
        reqs = self.prompts(r, pool, tab, sp, st_cards, members, votes, pop, brd, rng, rd)
        self.parent_fit = {m["name"]: float(pool.correct[k, sp.fit].mean()) for k, m in enumerate(members)}
        self.say(f"r{r}: {len(reqs)} prompts, {round(time.time() - t0)}s")
        self.stage(r, "proposals")
        t1 = time.time()
        gen = {g["id"]: g for g in self.generate(reqs, rd)}
        self.say(f"r{r}: proposals written in {round(time.time() - t1)}s")
        t2 = time.time()
        seen = {m["mid"] for m in members} | {h.get("mid") for h in hist}
        props, kids = [], []
        by = {m["name"]: m for m in members}
        names_of = {x.get("mid"): x.get("name") for x in hist}
        names_of.update({m["mid"]: m["name"] for m in members})
        def attempt(q, g, row):
            """One answer read against the slot's rules. Every program also runs on all score rows here, so its error reaches the re-ask and not stage 1."""
            try:
                m = self.candidate(r, q, parse_answer(g.get("content")), members)
                if m["mid"] in seen:
                    raise Bad(f"repeat of {names_of.get(m['mid']) or 'an earlier proposal of this round'}")
                if m.get("program"):
                    program_rows(self.data, m, self.data.of("score"), render=False)
                self.a2_check(r, m, kids)
                self.c128_scope_check(r, m)
                seen.add(m["mid"])
                m["name"] = f"r{r}s{q['slot']}_{m['mid'][:6]}"
                row.update(own_ops=m.pop("own_ops", None), name=m["name"], ops=m["ops"], mid=m["mid"], d=m["d"], dim="high" if m["d"] >= HIGH_D else "low", why=m.get("why"), valid=True, status=None)
                kids.append((m, row))
                return True
            except (Bad, KeyError, ValueError, IndexError, TypeError, OSError) as e:
                row.update(name=f"r{r}s{q['slot']}", status=f"invalid: {str(e)[:400]}")
                return False
        pending = []
        for q in reqs:
            if q["id"] == "meta":
                continue
            g = gen.get(q["id"], {})
            row = dict(round=r, slot=q["slot"], lens_id=q["lens_id"], lens_hash=q["lens_hash"], card_text=q["card_text"], valid=False, scored=False, accepted=False, reward=0., card_rows=q.get("card_rows"))
            fa = f"{rd}/answers/{q['id']}.json"
            jdump(fa, dict(content=g.get("content"), reasoning=(g.get("reasoning") or "")[-4000:], finish=g.get("finish"), tokens=g.get("tokens")))
            row["file"] = fa
            if not attempt(q, g, row):
                pending.append((q, g, row))
            props.append(row)
        # prereg extra_candidates: written by us, then an ordinary candidate through every gate. No lens, so the board and the bandit do not see it; it takes no proposer slot
        for i, x in enumerate(self.a.get("extra_candidates") or []):
            if x.get("round") != r:
                continue
            q = dict(id=f"x{i:02d}", slot=100 + i, op=None, parents=[x["parent"]])
            row = dict(round=r, slot=q["slot"], lens_id=None, lens_hash=None, card_text=None, valid=False, scored=False, accepted=False, reward=0., card_rows=None, injected=True)
            ok = attempt(q, dict(content="```json\n" + json.dumps(dict(why=x.get("why"), program=x["program"], edits=[])) + "\n```"), row)
            self.say(f"r{r}: injected candidate {row.get('name')}: {'valid' if ok else row.get('status')}")
            props.append(row)
        for m, row in kids:          # flag only, for the round review
            hard = hardcoded_values(m.get("program") or "") + hardcoded_values((m.get("gate") or {}).get("note") or "")
            if hard:
                row["hardcoded"] = hard
                self.say(f"  hard-coded values in {m['name']}: {hard[:4]}")
        # amendment 6: one re-ask of the frozen proposer for an invalid answer, with the plain reason; then the slot is lost and logged
        is_rep = lambda x: "repeat" in str(x.get("status"))
        ask = dict(invalid_first=len(pending), repeat_first=sum(is_rep(x[2]) for x in pending))
        t_re = time.time()
        if pending and r >= active_amendment(self.work, r).get("reask_from", 10 ** 9):
            a3_ = self.a3_assign(r, pending, kids)
            again = []
            for q, g, row in pending:
                why_ = str(row["status"]).replace("invalid: ", "")
                row["first_status"] = row["status"]
                why_ = self.a3_why(a3_, q["slot"], why_)
                extra = " Propose something different from it, on another cause or with another change." if is_rep(row) else ""
                again.append(dict(q, id=q["id"] + "_re", user=q["user"] + f"\n\n## Your previous answer was rejected\nReason: {why_}.{extra}\nYour previous answer ended:\n{str(g.get('content'))[-3000:]}\n\nAnswer again in the same form."))
            rd2 = f"{rd}/reask"
            os.makedirs(rd2, exist_ok=True)
            if not os.path.exists(f"{rd2}/prompts.json"):
                jdump(f"{rd2}/prompts.json", again)
            again = jload(f"{rd2}/prompts.json")
            try:
                gen2 = {x["id"]: x for x in self.generate(again, rd2)}
            except (RuntimeError, Bad, OSError) as e:
                self.say(f"r{r}: re-ask failed, the slots stay lost: {str(e)[:200]}")
                gen2 = {}
            for (q, g, row), q2 in zip(pending, again):
                g2 = gen2.get(q2["id"], {})
                jdump(f"{rd}/answers/{q2['id']}.json", dict(content=g2.get("content"), reasoning=(g2.get("reasoning") or "")[-4000:], finish=g2.get("finish"), tokens=g2.get("tokens")))
                row["reasked"] = True
                if g2:
                    q_ = dict(q, must=q.get("must"))
                    attempt(q_, g2, row)
            self.a3_final(r, a3_, pending)
            if a3_["colliding"]:
                ask["a3"] = dict(a3_, assigned={str(s): k for s, k in a3_["assigned"].items()})
            left = [x for x in pending if not x[2]["valid"]]
            ask.update(invalid_after=len(left), repeat_after=sum(is_rep(x[2]) for x in left), secs=round(time.time() - t_re))
            self.say(f"r{r}: re-ask: invalid {ask['invalid_first']} (repeat {ask['repeat_first']}) -> {ask['invalid_after']} (repeat {ask['repeat_after']}), {ask['secs']}s")
        jdump(f"{rd}/reask.json", dict(ask, slots={str(x[2]["slot"]): x[2].get("first_status") or x[2]["status"] for x in pending}))
        kids = self.c128_critic(r, rd, kids)        # 128c: the leakage critic before any engine time; off without prereg c128 critic
        if self.c9:
            kids = self.HW.critic(self, sys.modules[__name__], r, rd, kids)       # 1109c: denylist precheck + frozen-model LEAK critic
        # checkpoints; plain members go to the cards at once, gated writes run beside them
        for m, row in kids:
            self.ev.build(m["genome"])
        self.ev.wait(self.say)
        q_full = self.data.of("score")
        rc_ = self.rank_cfg(r)
        q_score = self.rank_draw(r, q_full, rc_["n"]) if rc_ else q_full
        q_m = {}
        if rc_:
            added = {}
            for m, row in kids:
                if row["valid"]:
                    q_m[m["slot"]], added[m["name"]] = self.rank_rows(r, m, by[m["parents"][0]], q_score, q_full)
            jdump(f"{rd}/rank.json", dict(draw=q_score, added={k: v for k, v in added.items() if v}, rows=len(q_score), of=len(q_full)))
            self.say(f"r{r}: ranking on {len(q_score)} of {len(q_full)} score rows; {sum(1 for v in added.values() if v)} narrow members topped up")
        self.read_rows = qm = lambda m: q_m.get(m["slot"], q_score)       # by slot: finish_gate gives a gate member its real id midway
        def check(pairs):
            for m, row in pairs:
                try:
                    program_rows(self.data, m, q_score[:5])
                except Bad as e:
                    row.update(valid=False, status=f"program on score rows: {str(e)[:200]}")
            return [x for x in pairs if x[1]["valid"]]
        def stage1(pairs):
            try:
                self.ev.evaluate_many([(m, qm(m)) for m, row in pairs])
            except Bad as e:
                self.say(f"a program failed on the score rows, evaluating one by one: {str(e)[:200]}")
                for m, row in pairs:
                    try:
                        self.ev.evaluate(m, qm(m))
                    except Bad as e2:
                        row.update(valid=False, status=f"program on score rows: {str(e2)[:200]}")
        gated = [x for x in kids if x[0].get("gate")]
        written, need = set(), {}
        def tick():                  # a review as soon as a member's score rows have landed
            for m, row in kids:
                if m["mid"] in written or (m.get("gate") and not m.get("note")) or not row["valid"]:
                    continue
                try:
                    if m["mid"] not in need:
                        need[m["mid"]] = {i.get("id") for i in program_rows(self.data, m, own_rows(m, qm(m)))[0]}
                    v = load_raw(m) if gate_scope(m) is not None else load_member(m)
                    if need[m["mid"]] and all(i in v for i in need[m["mid"]]):
                        written.add(m["mid"])
                        if m.get("note"):
                            row["recite"] = self.recited(m, qm(m))
                        self.diag_stats(m, row)
                        self.review(r, rd, m, row, by[m["parents"][0]], set(qm(m)))
                        if self.c9 and self.HW.cfg(self).get("early_a7"):      # amendment 11: its own stage-1 screen passes -> its A7 paired draw goes out now as a LOW (_se) line
                            self.HW.early_draw(self, sys.modules[__name__], r, members, m, row, qm)
                except Exception as e:
                    written.add(m["mid"])
                    self.say(f"  review {m.get('name')}: {type(e).__name__}: {str(e)[:150]}")
        self.ev.tick = tick
        self.stage(r, "stage 1")
        ranked = set(q_score).union(*[set(v) for v in q_m.values()]) if q_m else set(q_score)
        def fill_rows(m):            # never a row any candidate is ranked on, and for a gate only its scope: the leak check reads outside rows
            c, A = gate_scope(m), sorted(set(q_full) | set(self.data.of("select")))
            return [i for i in own_rows(m, A) if i not in ranked and (c is None or ROWKEY.get(i) == c)]
        def fill(pairs):
            try:
                if rc_ and a.get("fill_ahead"):
                    self.ev.fill_many([(m, fill_rows(m)) for m, row in pairs if row["valid"]])
            except Exception as e:
                self.say(f"r{r}: filler failed, the round goes on without it: {type(e).__name__}: {str(e)[:200]}")
        fit = [i for i in learn if self.data.role[i] != "select"]
        bares = {vh.gid(by[m["parents"][0]]["genome"]): member(by[m["parents"][0]]["genome"], None, name="bare") for m, _ in gated}
        self.ev.prio = True          # the rows each write needs go first, beside the program members and not after them
        self.ev.evaluate_many([(b, fit) for b in bares.values()])
        self.ev.prio = False
        first, self.ev.pending = self.ev.pending, []
        plain = check([x for x in kids if not x[0].get("gate")])
        stage1(plain)
        self.c128_request(r, kids, by, qm, q_full)      # 128c: the parents' band and null draws, beside stage 1
        if self.c9:
            self.HW.request_nulls(self, sys.modules[__name__], members)      # 1109c: every member's real re-seeded draws on score + SELECT
        fill(plain)
        keep, self.ev.pending = self.ev.pending, first
        self.ev.wait(self.say)
        if self.c9 and gated:            # 1109c (fable_1109d patch 6): the note as a text prefix on its scope's show rows first; no write without a pass
            gated = self.HW.prescreen(self, sys.modules[__name__], r, rd, gated, by)
        if self.c9:                      # gate_v3 control pool (coordinator 11-09): every proposed gate note of this run, kept in state, in order seen
            P_ = S.setdefault("c1109_gate_notes", [])
            seen_ = {(x["cat"], x["note"]) for x in P_}
            for m, _ in gated:
                k_ = ((m.get("gate") or {}).get("scope") or "")[4:]
                if k_ and (k_, m["gate"]["note"]) not in seen_:
                    P_.append(dict(cat=k_, note=m["gate"]["note"], name=m["name"], round=r)); seen_.add((k_, m["gate"]["note"]))
        writes = [(m, row, by[m["parents"][0]], self.write_gate(m, by[m["parents"][0]])) for m, row in gated]
        self.ev.pending += keep
        # each write scores as soon as it is ready; nothing waits for the slowest
        done, waiting, tw, late = [], list(writes), time.time(), []
        while waiting:
            self.ev.poll(self.say, tw)
            rest = []
            for w in waiting:
                m, row, pa, f = w
                row["write_file"] = f
                if m.get("gate_drop"):
                    row.update(valid=False, status=f"write: {m['gate_drop']}")
                    self.say(f"  {m.get('name') or row.get('name')}: {row['status']}")
                elif os.path.exists(f):
                    try:
                        self.finish_gate(m, pa, f)
                        row["mid"] = m["mid"]         # the row names the member the write made
                        done.append((m, row))
                        self.ev.prio = True
                        stage1(check([(m, row)]))
                        self.ev.prio = False
                        fill([(m, row)])
                    except (Bad, KeyError, ValueError, OSError) as e:
                        row.update(valid=False, status=f"write: {str(e)[:200]}")
                elif os.path.exists(f + ".failed") and len(glob.glob(f + ".*")) >= 3:
                    row.update(valid=False, status="write: failed three times")
                elif self.write_late(f, row):
                    row["write_late"] = self.write_late(f, row)
                    late.append(w)              # not dropped yet: the round still waits on its other jobs, and a write landing meanwhile is used
                else:
                    if os.path.exists(f + ".failed"):
                        self.write_gate(m, pa)
                    rest.append(w)
            waiting = rest
            if waiting:
                time.sleep(20)
        self.ev.wait(self.say)
        used_late = []
        for m, row, pa, f in late:
            st_ = self.write_clock(f)[1]
            row["write_start"] = None if st_ is None else round(st_, 1)        # a card may have started it after the deadline was read
            if os.path.exists(f):
                try:
                    self.finish_gate(m, pa, f)
                    row["mid"] = m["mid"]
                    row.update(write_used_late=True, write_landed=round(os.path.getmtime(f), 1))
                    done.append((m, row))
                    used_late.append((m, row))
                    self.say(f"  {m.get('name') or row.get('name')}: write landed after its deadline while the round waited ({row['write_late']}), used")
                except (Bad, KeyError, ValueError, OSError) as e:
                    row.update(valid=False, status=f"write: {str(e)[:200]}")
            else:
                row.update(valid=False, status=row["write_late"])
                self.say(f"  {m.get('name') or row.get('name')}: {row['status']}")
        if used_late:
            self.ev.prio = True
            stage1(check(used_late))
            self.ev.prio = False
            fill(used_late)
            self.ev.wait(self.say)
        raw = []
        for m, row in done:                  # nothing joins the pool unless the note really opens the thinking
            x = self.recited(m, qm(m))
            row["recite"] = x
            raw.append(x)
            if x is not None and x < .5:      # 1109c audit 11-09: a recite of None (no stage-1 rows in the key) is n/a, as the A1 note says, not a failure
                row.update(valid=False, status=f"write: real generation recites {x}")
            self.say(f"  {m['name']} recites in generation: {x}")
        wr, rc = write_counts([row for _, row, _, _ in writes], len(done), raw)
        t3 = time.time()
        allk = list(kids)
        self.stop_dead(kids, lambda m, row: qm(m))
        kids = [(m, row) for m, row in kids if row["valid"]]
        leaked = []
        for m, row in kids:
            lk = gate_leaks(m)
            if lk:
                row.update(leak_same=lk[0], leak_samples=lk[1])
                self.say(f"  leak check {m['name']}: same correctness outside scope {lk[0]:.3f} on {lk[1]} samples")
                if lk[0] < LEAK_SAME:
                    m["inherit_off"] = True
                    leaked.append((m, qm(m)))
                    self.say(f"  {m['name']} leaks outside its scope: scored fully")
        if leaked:
            self.ev.evaluate_many(leaked)
            self.ev.wait(self.say)
        have = []
        for m, row in kids:
            v = load_member(m)
            if any(i in v for i in qm(m)) and any(v.get(i) for i in qm(m)):
                have.append((m, row))
            else:
                row["status"] = "no records on the score rows"
        votes, tab, P, sp, st = self.compute(members, [m for m, _ in have])
        if have:
            ref = REF(members)
            child_idx = [k for k, m in enumerate(members) if m["origin"] == "proposal"]
            sp1 = dataclasses.replace(sp, score=np.array(sorted(tab.index[i] for i in q_score), int)) if rc_ else sp       # the draw: stage 1's statistics read the rows all proposals ran
            st1 = state_of(P, sp1, members) if rc_ else st       # the members' residual and refit on those same rows
            Ssc = atom_scores(P, sp1.score, child_idx, ref)
            for m, row in have:
                k = P.modules.index(m["name"])
                row.update(proposal_stats(P, sp1, st1, k, ref, combiner="evidence", S_sc=Ssc), scored=True)
                zz = zx_stats(tab, votes[m["parents"][0]], votes[m["name"]], qm(m), self.data.cluster)
                row.update(zx_score=zz["zx"], dacc_paired_score=zz["dacc"], strata_score=zz.get("strata"), strata_cross_score=zz.get("strata_cross"))
                row["reward"] = float(np.clip((zz["zx"] or 0.) if r >= self.F else (row.get("z_score") or 0.), -4, 4))     # continuous: the pull toward the residual on unseen rows
                if self.spread_on(r):
                    self.touch_stats(m, row, by[m["parents"][0]], q_full if rc_ else q_score, tab, votes, read=qm(m))
                self.say(f"  {m['name']} {'+'.join(m['ops'])} zx {row.get('zx_score') or 0:+.2f} dacc {100 * (row.get('dacc_paired_score') or 0):+.1f} z {row.get('z_score', 0):+.2f} z2 {row.get('z2_score', 0):.2f} q2>s {(row.get('q_score') or 0) ** 2 > (row.get('s_score') or 0)} ce {row.get('ce_gain_score', 0) * 1e3:+.2f}e-3")
        for m, row in allk:
            try:
                if not row.get("scored") and any(i in load_member(m) for i in qm(m)):
                    self.diag_stats(m, row)
                self.review(r, rd, m, row, by[m["parents"][0]], set(qm(m)))
            except Exception as e:
                self.say(f"  review {m.get('name')}: {type(e).__name__}: {str(e)[:150]}")
        self.ev.tick = None
        # stage 2: top by z2, the rest of LEARN
        if self.c9:                      # 1109c stage 1 (A1): fire gate + zx ranking, cap
            top = self.HW.stage1(self, sys.modules[__name__], r, rd, members, have, qm)
        elif r >= self.F and self.c128(r, "judge"):        # 128c stage 1: fired set against the null band; zx ranks; the parents' nulls go through it too
            top = self.c128_stage1(r, rd, members, have, qm, by)
        elif r >= self.F:              # ranked by the cross-sample score, no gate here (GRADIENT_CHECK.md: z2>1 passes a model against itself)
            top = self.pick_stage2(have, r)
            top = top + self.stage2_exempt(have, top, members, r, rd, qm)
            for m0 in (S.get("reserve") or []) if a.get("reserve") else []:
                if any(x["mid"] == m0["mid"] for x, _ in top):
                    continue
                m1 = dict(m0, slot=f"res_{m0['mid']}")
                if all(i in load_member(m1) for i in sorted(set(q_full) | set(self.data.of("select")))):
                    q_rows = dict(round=r, slot=m1["slot"], name=m1["name"], mid=m1["mid"], ops=m1.get("ops"), valid=True, scored=False, reserve=True, stage2=True, stage2_by="reserve")
                    have.append((m1, q_rows))
                    top.append((m1, q_rows))
                    allk.append((m1, q_rows))
        else:
            top = sorted([(m, row) for m, row in have if float(row.get("z2_score") or 0) > 1.], key=lambda x: -float(x[1]["z2_score"]))[: a["stage2"]]
        t4 = time.time()
        for m, row in top:
            row["stage2"] = True
        # amendment 6: all score rows, all SELECT and the member's own card rows; the other show rows only if it is accepted (the next round's ensure runs them)
        s2rows = lambda m, row: sorted(set(q_full) | set(self.data.of("select")) | ({i for i in (row.get("card_rows") or []) if i in self.data.role})) if rc_ else learn
        self.stage(r, "stage 2")
        self.read_rows = None
        self.c128_request_s2(r, top)
        running = self.ev.fill_settle({m["mid"] for m, _ in top})
        self.ev.evaluate_many([(m, [i for i in s2rows(m, row) if i not in running.get(m["mid"], ())]) for m, row in top])
        self.ev.wait(self.say)
        self.ev.evaluate_many([(m, s2rows(m, row)) for m, row in top])      # a filler line that failed leaves rows; the driver's own lines take them
        self.ev.wait(self.say)
        self.stage(r, "decide")
        accepted = fu0 = None
        if top:
            votes, tab, P, sp, st = self.compute(members, [m for m, _ in top])        # only the members being compared: a Table column on a row comes from all votes it is given
            ref = REF(members)
            fu0 = self.fz("fuse", self.fuser.fuse, members, votes, self.data, self.fusion, sp=sp) if self.fusion != "stacking" else None      # the same votes, so the same Table
            for m, row in top:
                k = P.modules.index(m["name"])
                v = votes[m["name"]]
                lost = self.ev.dead_for(m, s2rows(m, row))
                if lost:
                    row.update(stage2_status=f"item failed twice, {len(lost)} rows: stopped")
                    self.say(f"  {m['name']}: {row['stage2_status']}")
                    continue
                if any(i not in v for i in s2rows(m, row)):
                    row.update(stage2_status="missing rows")
                    continue
                row.update(learn_gain(P, sp, st, k, combiner="evidence", parent=ref), stage2_status="ok")
                row.update(select_stats(P, sp, st, k, ref))
                if fu0 is not None:   # the pool increment from the fusion; q, s and z_select stay on the log-linear state: they are not defined on these F
                    row.update({f"{x}_stacking": row.get(x) for x in ("ce_gain_select", "ce_gain_select_lb", "dacc_select")})
                    row.update(self.fz("increment", self.fuser.increment, members, m, votes, self.data, self.fusion, sp=sp, base=fu0))
        from vrsi27.boost.bayes import accepts
        for m, row in top:
            ok = accepts(row) and gain_positive(row)
            if r >= self.F:          # paired gain and cross-sample score on the stage 2 rows that are not SELECT, then SELECT: rows no proposer has seen
                fit_ids = [i for i in learn if self.data.role[i] != "select"]
                sel_ids = self.data.of("select")
                vp, vc = votes[m["parents"][0]], votes[m["name"]]
                pw = [(votes[x["name"]], w_) for x, w_ in zip(members, st.cur.w)] if st.cur.w is not None else None
                if fu0 is not None:
                    pw = fu0.F.copy()
                    pw[fu0.sp.fit] = fu0.F_oof
                gate_ids = [i for i in q_full if i not in set(qm(m))] if rc_ else fit_ids      # rows that played no part in picking this member
                zf, zs = zx_stats(tab, vp, vc, gate_ids, self.data.cluster, pw), zx_stats(tab, vp, vc, sel_ids, self.data.cluster, pw)
                row.update(zx_fit=zf["zx"], dacc_fit=zf["dacc"], strata_fit=zf.get("strata"), strata_cross_fit=zf.get("strata_cross"), zx_sel=zs["zx"], dacc_sel=zs["dacc"], strata_sel=zs.get("strata"), dacc_sel_lb=dacc_lb(tab, vp, vc, sel_ids, self.data.cluster))
                fit_old = (zf["dacc"] or 0) >= .02 and (zf["zx"] or 0) > 1
                fit_ok, dacc_ok, zx_ok = fit_old, (zf["dacc"] or 0) >= .02, (zf["zx"] or 0) > 1
                if rc_:
                    za, card = zx_stats(tab, vp, vc, [i for i in fit_ids if i in vc], self.data.cluster), [i for i in (row.get("card_rows") or []) if i in vc]
                    zc_ = zx_stats(tab, vp, vc, card, self.data.cluster) if card else dict(zx=None, dacc=None)
                    row.update(zx_allfit=za["zx"], dacc_allfit=za["dacc"], allfit_rows=za["zx_rows"], gate_rows=len(gate_ids), zx_card=zc_["zx"], dacc_card=zc_["dacc"], card_rows_n=len(card), rank_rows=len(qm(m)))
                    self.say(f"  card vs unseen {m['name']}: card rows {len(card)} dacc {zc_['dacc']} zx {zc_['zx']} | gate rows {len(gate_ids)} dacc {zf['dacc']} zx {zf['zx']} | all FIT run {za['zx_rows']} dacc {za['dacc']} zx {za['zx']}")
                    if row.get("reserve") and "narrow" not in row:
                        self.touch_stats(m, row, by[m["parents"][0]], q_full, tab, votes)
                    if row.get("narrow"):        # few touched gate rows: a count, since one row can be worth 10 points
                        tg = self.touched(m, by[m["parents"][0]], gate_ids)
                        zt = zx_stats(tab, vp, vc, tg, self.data.cluster, pw) if tg else dict(zx=None, dacc=None)
                        net = (zt["dacc"] or 0) * len(tg)
                        dacc_ok, zx_ok = net >= 2, (zt["zx"] or 0) > 1
                        fit_ok = dacc_ok and zx_ok
                        row.update(narrow_gate=dict(touched_gate_rows=len(tg), net_rows=net, zx_touched_gate=zt["zx"], passes=fit_ok, old_bar_passes=fit_old))
                        self.say(f"  narrow gate {m['name']}: touched gate rows {len(tg)} net {net:+.1f} zx {zt['zx']} -> {fit_ok}; old bar {fit_old}")
                ok = row.get("stage2_status") == "ok" and fit_ok and (zs["dacc"] or 0) > 0 and gain_positive(row)
                nr = None
                if fu0 is not None and row.get("stage2_status") == "ok":      # the same four counts for every candidate; only a narrow member is gated on them
                    up, down, kinds = self.pool_rows(members, m, votes, sp, fu0)
                    row.update(pool_rows_up=up, pool_rows_down=down, **kinds)
                    self.say(f"  up rows {m['name']}: pool rows up {up} down {down} | direct {kinds['up_direct']} split {kinds['up_split']} wrong {kinds['up_wrong']} silent {kinds['up_silent']} | " + "; ".join(f"{x['id']} keys {x['keys']} ok {x['ok']}" for x in kinds["up_detail"][:6]))
                if self.narrow_rule and row.get("narrow") and row.get("stage2_status") == "ok":
                    no_hit = bool(fit_ok and up >= 1 and down == 0)
                    nr = no_hit and kinds["up_direct"] + kinds["up_split"] >= 1
                    row.update(old_rule=bool(ok), new_rule=nr, rule_without_direct=no_hit, net_rows=(row.get("narrow_gate") or {}).get("net_rows"))
                    self.say(f"  narrow rule {m['name']}: FIT {fit_ok} net {row['net_rows']} | pool rows up {up} down {down} direct hits {kinds['up_direct'] + kinds['up_split']} | old rule {'pass' if ok else 'fail'}, new rule {'pass' if nr else 'fail'}" + (", direct hit missing" if no_hit and not nr else ""))
                    ok = nr
                k = P.modules.index(m["name"])
                F_ec = st.cur.F if fu0 is None else fu0.F
                row.update(**({"zx_fused_fit": zf.get("zx_pool"), "zx_fused_sel": zs.get("zx_pool")} if fu0 is not None else {"zx_stacking_fit": zf.get("zx_pool"), "zx_stacking_sel": zs.get("zx_pool")}), err_corr_pool_fit=err_corr_pool(tab, F_ec, P.correct[k], sp.fit), err_corr_pool_sel=err_corr_pool(tab, F_ec, P.correct[k], sp.select),
                           rejected_by=[n for n, t in (("dacc", dacc_ok), ("zx", zx_ok), *((("pool_rows", nr),) if nr is not None else (("dacc_select", (zs["dacc"] or 0) > 0), ("pool_lb", gain_positive(row))))) if not t])
                self.say(f"  zx gate {m['name']}: fit zx {zf['zx']} {'fused' if fu0 is not None else 'stacking'}-residual zx {zf.get('zx_pool')} dacc {zf['dacc']} strata {zf.get('strata')} | SELECT (unseen by proposers, reused across rounds) zx {zs['zx']} {'fused' if fu0 is not None else 'stacking'}-residual zx {zs.get('zx_pool')} dacc {zs['dacc']} lb {row['dacc_sel_lb']} | pool increment ce {row.get('ce_gain_select')} lb {row.get('ce_gain_select_lb')} | err corr with pool fit {row['err_corr_pool_fit']} select {row['err_corr_pool_sel']} | rejected by {row['rejected_by']}")
            if fu0 is not None and not row.get("narrow"):       # coordinator 11-07: a wide member must not lower the pool's SELECT accuracy
                pool_ok = float(row.get("dacc_select") or 0) >= -GAIN_TOL
                row.update(old_wide_verdict=bool(ok), pool_dacc_ok=pool_ok)
                self.say(f"  wide gate {m['name']}: pool dacc {row.get('dacc_select')} -> {pool_ok}; old verdict {'pass' if ok else 'fail'}, new {'pass' if ok and pool_ok else 'fail'}")
                if not pool_ok:
                    row.setdefault("rejected_by", []).append("pool_dacc")
                ok = bool(ok and pool_ok)
            row["would_accept"] = bool(ok)
            self.say(f"  stage 2 {m['name']}: q {row.get('q_fit')} s {row.get('s_fit')} z_select {row.get('z_select')} ce select {row.get('ce_gain_select')} lb {row.get('ce_gain_select_lb')} dacc {row.get('dacc_select')} -> {'ACCEPT' if ok else 'no'}" + (f" | stacking lb {row.get('ce_gain_select_lb_stacking')} ce {row.get('ce_gain_select_stacking')} dacc {row.get('dacc_select_stacking')}" if fu0 is not None else ""))
        c9_rec = None
        if self.c9:                      # 1109c stage 2 on R: MemberCandidate + RRSI judge, Holm across candidates, greedy re-read, null void
            dec = c128_rec = None
            rj_ = self.HW.rejudge(self, sys.modules[__name__], r, members, by)       # 1109c amendment 2: earlier stage-2 candidates judged with empty nulls
            for m_, row_ in rj_:
                by[m_["name"]] = m_
            top = list(top) + rj_
            self.HW.request_nulls(self, sys.modules[__name__], members, top)      # amendment 5: with the candidates' seed-paired reads
            self.ev.wait(self.say)
            chosen, steps, c9_rec = self.HW.decide(self, sys.modules[__name__], r, rd, members, top, by, q_score) if top else ([], [], None)
            if c9_rec:
                pv_ = c9_rec.pop("_prov", {})
                self.HW.admitted_reads(self, r, chosen, c9_rec, pv_.get("prov", ()), pv_.get("null_prov", ()))
            plateau_ = None
        else:
            dec = self.c128_decide(r, members, top, by) if self.c128(r, "judge") and top else None
            chosen, steps = self.accept_in_order(members, top, r)
            c128_rec = self.c128_finish(r, rd, members, top, chosen, steps, dec) if dec is not None else None
            plateau_ = self.plateau_end(members, top, chosen, steps, r, rd) if self.plateau_on and not self.c128_replaces(r) else None      # 128c: the combination test is an acceptance outside the judge
        accepted = chosen[0] if chosen else None
        # level 2: the meta child joins the lens population
        mg = gen.get("meta")
        if mg and mg.get("content"):
            code = parse_lens_code(mg["content"])
            if code:
                cid = f"m{r:03d}_{lens_hash(code)[:8]}"
                if cid not in pop.entries:
                    from vrsi27.boost.lens import run_lens, validate_output, Budget, build_state
                    reason = None
                    try:
                        ctx = self.lens_ctx(pool, tab, "stock")
                        for slot in (0, 1):
                            stt, _ = build_state(pool, sp, st, ctx, round_idx=r, slot=slot, n_slots=a["slots"], budget=Budget())
                            res = run_lens(code, stt, timeout_s=20.)
                            if not res.get("ok"):
                                raise ValueError(str(res.get("error"))[:300])
                            validate_output(res["output"], stt)
                    except Exception as e:
                        reason = str(e)[:300]
                    pq = next((q for q in reqs if q["id"] == "meta"), {})
                    pop.add(cid, code, parent=pq.get("lens_parent"), round_idx=r + 1, origin="meta", status="active" if reason is None else "invalid", reason=reason)
                    pop.save()
                    self.say(f"r{r}: meta lens {cid} {'valid' if not reason else 'invalid: ' + reason[:120]}")
        # the round's numbers
        if accepted:
            S["members"] = members + chosen
        S["hist"] = hist + props
        if a.get("reserve"):
            kept = {m["mid"] for m in chosen}
            S["reserve"] = [dict(m) for m, row in top if row.get("stage2_status") == "ok" and m["mid"] not in kept and not row.get("null") and not row.get("confirm_of")]
        S["cross_pairs"] = S.get("cross_pairs", []) + [q["parents"] for q in reqs if q.get("op") == "cross" and len(q.get("parents", [])) == 2]
        tab2, pool2, sp2, st2 = tab, P, sp, st
        votes2, a4_ = votes, self.a4(r)
        if a4_:              # A4: the round line on the members alone; votes/tab above also hold the stage 2 candidates, whose answers moved the pool's columns and tie-breaks (r4 SELECT .755 -> .745, no member change). The increments keep their shared Table.
            votes2, tab2, pool2, sp2, st2 = self.members_tables(members, votes)
        Fp = st2.cur.F
        fu_rec, Fs = {}, None
        if self.fusion != "stacking":
            fu = (None if a4_ else fu0) or self.fz("fuse", self.fuser.fuse, members, votes2, self.data, self.fusion, sp=sp2)
            Fs, Fp = Fp, fu.F
            fu_rec = dict(fusion=self.fusion, pool_fit_stacking=tab2.acc(Fs, sp2.fit), pool_select_stacking=tab2.acc(Fs, sp2.select), table_columns=int(tab2.C), fusion_info={k: v for k, v in fu.info.items() if k in ("rows_in_em", "em_passes", "agree_share_fit", "agree_share_select", "passed", "T", "agree_floored", "no_vote_rows")},
                          fusion_secs={k: dict(calls=len(v), total=sum(v), max=max(v)) for k, v in self.fz_t.items()})
            self.say(f"r{r}: fusion {self.fusion}: pool FIT {tab2.acc(Fp, sp2.fit):.3f} SELECT {tab2.acc(Fp, sp2.select):.3f} (stacking {fu_rec['pool_fit_stacking']:.3f} {fu_rec['pool_select_stacking']:.3f}) | T {fu.info.get('T'):.3f} rows at the floor {fu.info.get('agree_floored')} agreed FIT {fu.info.get('agree_share_fit'):.3f} SELECT {fu.info.get('agree_share_select'):.3f} | table columns {tab2.C} | seconds " + ", ".join(f"{k} {len(v)} calls {sum(v):.1f} total {max(v):.1f} max" for k, v in self.fz_t.items()))
            self.fz_t = {}
        n = len(members)
        acc = lambda k, idx: float(pool2.correct[k, idx].mean())
        fit_acc = tab2.acc(Fp, sp2.fit); sel_acc = tab2.acc(Fp, sp2.select)
        best = max(range(n), key=lambda k: acc(k, sp2.fit))
        sk = pool2.modules.index("stock")
        sel_gain = sel_acc - acc(best, sp2.select)
        try:                 # report only: nothing below reads it
            xr = self.report_extras(r, rd, members, votes, tab2, pool2, sp2, fu if self.fusion != "stacking" else None)
        except Exception as e:
            xr = dict(error=f"{type(e).__name__}: {str(e)[:200]}")
            self.say(f"r{r}: report extras failed: {xr['error']}")
        tps = throughput([m for m, _ in kids], t0, self.ev.jobs)
        self.ev.jobs = []
        cnt = collections.Counter(o for m, _ in kids for o in m["ops"])
        dropped = [dict(slot=p.get("slot"), name=p.get("name"), own_ops=p.get("own_ops"), reason=p.get("status") or "no reason recorded") for p in props if not p.get("valid")]
        self.say(f"r{r}: dropped {len(dropped)} of {len(props)}: " + ("; ".join(f"{x['name']} {x['own_ops'] or ''}: {x['reason'][:160]}" for x in dropped) or "none"))
        rec = dict(round=r, members=len(members), err_corr=ec, pool_fit=fit_acc, pool_select=sel_acc, best_name=pool2.modules[best], best_fit=acc(best, sp2.fit), best_select=acc(best, sp2.select), stock_fit=acc(sk, sp2.fit), stock_select=acc(sk, sp2.select),
                   pool_gain_select=sel_gain, weights={m["name"]: float(w) for m, w in zip(members, st2.cur.w)}, err_pairs={f"{members[a_]['name']}|{members[b]['name']}": (None if np.isnan(Cor[a_, b]) else float(Cor[a_, b])) for a_ in range(n) for b in range(a_ + 1, n)},
                   valid=sum(1 for p in props if p["valid"]), scored=sum(1 for p in props if p["scored"]), stage2=len(top), reask=ask, dropped=dropped, accepted=",".join(m["name"] for m in chosen) if chosen else None, accept_steps=steps, members_after=[m["name"] for m in S["members"]], secs=dict(prompts=t1 - t0, gen=t2 - t1, stage1=t3 - t2, stats=t4 - t3, stage2=time.time() - t4, total=time.time() - t0), rows_per_h_per_card=tps[0], cold_start_s=tps[2], engines=tps[3], writes_ok=wr[0], writes=wr[1], writes_not_reached=wr[2], writes_no_recitation=rc[3], recite_ok=rc[0], recite_scored=rc[1], recite_median=rc[2], recite_na=rc[4], ops=dict(cnt))
        rec["report_extras"] = xr
        if c128_rec is not None:
            rec["c128"] = {k: v for k, v in c128_rec.items() if k != "rows"}
        rec.update(fu_rec, views=views_of(tab2, pool2, Fp, members, self.data, dict(fit=sp2.fit, select=sp2.select), Fs))
        if self.fusion != "stacking":
            rec["weights_stacking"] = rec.pop("weights")
        reads_ = dict(stage2=sum(1 for _, x in top if x.get("stage2_status") == "ok"), reread=sum(len(x.get("reread", [])) for x in steps), narrow_retest=sum(len(x.get("retest", [])) for _, x in top), plateau=plateau_["reads"] if plateau_ else 0)
        rec["select_reads"] = dict(reads_, total=sum(reads_.values()))
        rec["stage2_exempt"] = [dict(name=m["name"], **row["stage2_exempt"], stage2_status=row.get("stage2_status"), would_accept=row.get("would_accept"), rejected_by=row.get("rejected_by"), accepted=bool(row.get("accepted"))) for m, row in top if row.get("stage2_exempt")]
        self.say(f"r{r}: exempted from the stage 2 zx filter, outcome: " + (", ".join(f"{x['name']} {x['stage2_status']} would_accept {x['would_accept']} rejected_by {x['rejected_by']} accepted {x['accepted']}" for x in rec["stage2_exempt"]) or "none"))
        self.say(f"r{r}: SELECT reads this round {rec['select_reads']}")
        if plateau_:
            rec.update(select_increment=plateau_["select_increment"], plateau_end=plateau_["plateau_end"], plateau_after_combo=plateau_.get("plateau_after_combo"), combo=plateau_["combo"], plateau_secs=plateau_["secs"], plateau=self.weight_slots(r, reqs, props))
        if self.c9:                      # 1109c: reflect (notes, <= 1 skill edit, policy challenger), skill lifecycle, P A/B
            H_ = sys.modules[__name__]
            rec["c1109"] = dict(decide={k: v for k, v in (c9_rec or {}).items() if k != "stage2"}, eproc=getattr(self, "c9_eproc", {}))
            try:
                rec["c1109"]["reflect"] = self.HW.reflect(self, H_, r, rd, props, kids)
            except (RuntimeError, OSError, Bad) as e:
                rec["c1109"]["reflect"] = dict(error=str(e)[:300])
                self.say(f"r{r}: 1109c reflect failed (nothing written): {str(e)[:200]}")
            rec["c1109"]["end"] = self.HW.round_end(self, H_, r, rd, reqs, props, chosen, dict(pool_S=sel_acc))
        S["rounds"].append(rec)
        S["round"] = r
        if plateau_:
            rec["stop_count"] = self.stop_count()
            self.say(f"r{r}: weight slots {rec['plateau']}; rounds counted toward the stop rule {rec['stop_count']} of {PLATEAU_STOP}")
        if self.narrow_rule or self.plateau_on:
            rec["accepted_new_rule_only"] = [m["name"] for m in S["members"] if m.get("new_rule_only") and m in chosen]
            rec["members_new_rule_only"] = [m["name"] for m in S["members"] if m.get("new_rule_only")]
            self.say(f"r{r}: accepted by the new narrow rule only: {rec['accepted_new_rule_only']}; all such members {rec['members_new_rule_only']}")
        rec["lines"] = len(self.line_report(r, rd, t0))
        if self.fusion in DS_MODES:
            rec["member_loo_fit"] = {x["name"]: x["loo_fit"] for x in (jload(f"{rd}/parent_scores.json", None) or [])}
            try:
                self.fusion_side(r, rd, fu, members, votes)
                self.fusion_side_read(r - 1)
            except Exception as e:     # a report never stops a round
                self.say(f"r{r}: fusion side lines failed: {type(e).__name__}: {str(e)[:200]}")
        jdump(f"{rd}/decision.json", rec)
        self.save()
        self.ahead(r)
        try:
            self.ops_report(r, rd, reqs, props, top, chosen, kids)
        except Exception as e:     # a report never stops a round
            self.say(f"r{r}: ops report failed: {type(e).__name__}: {str(e)[:200]}")
        cws = jload(f"{rd}/card_wrong.json", None) or []
        cw = (f"{np.mean([x['before'] for x in cws]):.2f}", f"{np.mean([x['after'] for x in cws]):.2f}") if cws else ("?", "?")
        mix = dict(collections.Counter(vh.class_of(self.data.rows[i]) or "other" for i in q_score))
        self.say(f"ROUND {r}: members {len(members)} error corr {ec:.3f} | pool FIT {fit_acc:.3f} SELECT {sel_acc:.3f} | best single {pool2.modules[best]} FIT {acc(best, sp2.fit):.3f} SELECT {acc(best, sp2.select):.3f} | pool gain SELECT {100 * sel_gain:+.1f} | stock FIT {acc(sk, sp2.fit):.3f} SELECT {acc(sk, sp2.select):.3f} {xr.get('stock_text', '')}| pool FIT held out by picture {xr.get('pool_fit_oof')} | not stopped by bench {xr.get('not_stopped_text')} | valid {rec['valid']}/{len(props)} ops {dict(cnt)} stage2 {len(top)} accepted {rec['accepted']} | {rec['secs']['total'] / 60:.0f} min, gen {rec['secs']['gen'] / 60:.0f} stage1 {rec['secs']['stage1'] / 60:.0f} stage2 {rec['secs']['stage2'] / 60:.0f}; {tps[0]:.0f} rows/h/card, cold start {tps[2]:.0f}s over {tps[3]} engines; writes ok {wr[0]}/{wr[1]}, did not reach unseen {wr[2]}, failed recitation {rc[3]}; generation recites {rc[0]}/{rc[1]} (n/a {rc[4]}), median {rc[2]}; dim {dict(collections.Counter(p.get("dim") for p in props if p.get("valid")))} | card wrong {cw[0]}->{cw[1]} | score rows {mix}")

    def ops_report(self, r, rd, reqs, props, top, chosen, kids):
        """Report only, read by nobody in the run: per operator, distinct candidates of this round proposed, built, changing answers, passing stage 1, FIT, SELECT, accepted. A candidate counts under its own operators, not its ancestors'. Reserves in stage 2 count apart. Then the gate members, and the side lines."""
        earlier = {p["name"]: p for p in self.state["hist"] if p.get("name") and p.get("own_ops") is not None}
        norm = lambda o: "patch" if o == "kernel" else o
        own = lambda row: {norm(o) for o in (row.get("own_ops") or earlier.get(row.get("name"), {}).get("own_ops") or row.get("ops") or [])}
        fit_ok = lambda row: row.get("rejected_by") is not None and not {"dacc", "zx"} & set(row["rejected_by"])
        sel_ok = lambda row: row.get("rejected_by") is not None and not {"dacc_select", "pool_lb", "pool_rows"} & set(row["rejected_by"])
        keys = ("proposed", "built", "changed", "stage1", "fit", "select", "accepted")
        T = collections.defaultdict(collections.Counter)
        asked = {q["slot"]: q.get("op") for q in reqs if q["id"] != "meta"}
        for row in props:
            ops = own(row) if row.get("valid") or row.get("name") else set()
            for o in ops | ({norm(asked[row["slot"]])} if asked.get(row.get("slot")) else set()):
                t = T[o]
                t["proposed"] += 1
                t["built"] += bool(row.get("valid")) and o in ops
                t["changed"] += bool(row.get("valid")) and o in ops and (row.get("touched_rows") or 0) > 0
                t["stage1"] += bool(row.get("stage2")) and o in ops
                t["fit"] += fit_ok(row) and o in ops
                t["select"] += sel_ok(row) and o in ops
                t["accepted"] += bool(row.get("accepted")) and o in ops
        for m, row in top:
            if row.get("reserve"):
                for o in own(row) or {norm(x) for x in m.get("ops") or []}:
                    t = T[o]
                    t["reserve_in_stage2"] += 1
                    t["reserve_fit"] += fit_ok(row)
                    t["reserve_select"] += sel_ok(row)
                    t["reserve_accepted"] += bool(row.get("accepted"))
        recs = [dict(kind="op", round=r, op=o, **{k: t[k] for k in keys}, **{k: v for k, v in t.items() if k not in keys}) for o, t in sorted(T.items())]
        gates = []
        for row in props:
            if "gate" not in (row.get("own_ops") or []):
                continue
            km = next((x for x, _ in kids if x["name"] == row.get("name")), None)
            gates.append(dict(kind="gate", round=r, name=row.get("name"), write=self.write_class(row), write_submit=row.get("write_submit"), write_start=row.get("write_start"), write_landed=row.get("write_landed") or (round(os.path.getmtime(row["write_file"]), 1) if row.get("write_file") and os.path.exists(row["write_file"]) else None), write_used_late=bool(row.get("write_used_late")), ran=bool(row.get("scored")), never_ran_because=None if row.get("scored") else row.get("status"), answer_dir=os.path.basename(member_dirs(km)[0]) if km and row.get("scored") else None, valid=row.get("valid"), status=row.get("status"), touched_rows=row.get("touched_rows"), recite=row.get("recite"), recital_fired=None if row.get("recite") is None else row["recite"] >= .5,
                              dacc_touched=row.get("dacc_touched"), zx_touched=row.get("zx_touched"), dacc_overall=row.get("dacc_paired_score"), zx_overall=row.get("zx_score"), accepted=row.get("accepted")))
        recs += gates
        side = self.side_lines(r, chosen, props, kids) if self.a.get("side_checks", True) else []
        recs += [dict(kind="side_queued", round=r, **x) for x in side]
        with open(f"{rd}/ops.jsonl", "a") as o:
            for x in recs:
                o.write(json.dumps(x, ensure_ascii=False) + "\n")
        self.say(f"r{r}: ops proposed/built/changed/stage1/FIT/SELECT/accepted: " + "; ".join(f"{x['op']} " + "/".join(str(x[k]) for k in keys) for x in recs if x["kind"] == "op") + f" | gates {len(gates)} proposed, {sum(1 for g in gates if g['ran'])} ran, recital fired {sum(1 for g in gates if g['recital_fired'])} | side lines queued {len(side)}")
        self.side_read(r, rd)

    def report_extras(self, r, rd, members, votes, tab, pool, sp, fu):
        """Report only (coordinator 11-08), read by nothing in the run: each member's tie-broken accuracy beside its per-sample mean; stock three ways (tie-broken, per sample, k-sample self-consistency from sidecheck/stock4 once landed); the pool's FIT held out by picture beside in-sample; not-stopped share per bench; vlmsbiased canonical-answer rate; prior-phrase reasoning, each hit with its row in prior_phrases.jsonl."""
        ids_ = lambda idx: [tab.ids[k] for k in idx]
        F_, S_ = ids_(sp.fit), ids_(sp.select)
        tie = lambda name, idx: float(pool.correct[pool.modules.index(name), idx].mean()) if len(idx) else float("nan")
        mem = {}
        for m in members:
            v = votes[m["name"]]
            cr, ncr = canonical_rate(v, self.data.rows, F_ + S_)
            mem[m["name"]] = dict(fit_tie=tie(m["name"], sp.fit), fit_per_sample=per_sample(v, F_), select_tie=tie(m["name"], sp.select), select_per_sample=per_sample(v, S_), canonical_rate=cr, canonical_rows=ncr)
        st = mem.get("stock", {})
        sc = sc_k(self.work, F_ + S_)
        scf, scs = (sc_k(self.work, F_), sc_k(self.work, S_)) if sc else (None, None)
        by = collections.defaultdict(list)
        for i in F_ + S_:
            by[self.data.rows[i].get("bench") or i.split("/")[0]].append(i)
        ns = {b: dict(stock=not_stopped([votes["stock"]], v) if "stock" in votes else None, members=not_stopped([votes[m["name"]] for m in members], v), rows=len(v)) for b, v in sorted(by.items())}
        hits = {}
        with open(f"{rd}/prior_phrases.jsonl", "w") as o:
            for m in members:
                h = prior_hits(m, F_ + S_)
                hits[m["name"]] = dict(samples=len({x[:2] for x in h}), rows=len({x[0] for x in h}), by_phrase=dict(collections.Counter(x[2] for x in h)))
                for x in h:
                    o.write(json.dumps(dict(round=r, member=m["name"], id=x[0], sample=x[1], phrase=x[2], text=x[3]), ensure_ascii=False) + "\n")
        f3 = lambda x: "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"
        cols = {m["name"]: pool.correct[pool.modules.index(m["name"])] for m in members}
        se_ = dict(fit=se_table(self.data, tab, sp.fit, cols), select=se_table(self.data, tab, sp.select, cols))
        if fu is not None:          # the fused pool on its own table
            pc = {"pool": right_rows(fu.tab, fu.F, np.arange(len(fu.tab.ids)))}
            for k_, idx_ in (("fit", fu.sp.fit), ("select", fu.sp.select)):
                for b, t in se_table(self.data, fu.tab, idx_, pc).items():
                    se_[k_].setdefault(b, {}).update(t)
        jdump(f"{rd}/se_cluster.json", se_)
        f4 = lambda x: "-" if x is None else f"{x:.3f}"
        self.say(f"r{r}: SE naive / clustered by picture, all rows: " + "; ".join(f"{k} " + ", ".join(f"{nm} {f4(v['se'])}/{f4(v['se_cluster'])}" for nm, v in t["all"].items() if nm in ("pool", "stock")) for k, t in se_.items()) + f"; per bench and member in {rd}/se_cluster.json")
        out = dict(members=mem, pool_fit_in_sample=float(tab.acc(fu.F, sp.fit)) if fu is not None else None, pool_fit_oof=None if fu is None or oof_acc(fu) is None else round(oof_acc(fu), 4),
                   stock=dict(fit_tie=st.get("fit_tie"), fit_per_sample=st.get("fit_per_sample"), fit_sc=scf, select_tie=st.get("select_tie"), select_per_sample=st.get("select_per_sample"), select_sc=scs),
                   not_stopped=ns, prior_phrases=hits, voted_letters=(fu.info.get("voted_letters") if fu is not None else None))
        out["stock_text"] = (f"(per sample FIT {f3(st.get('fit_per_sample'))} SELECT {f3(st.get('select_per_sample'))}; self-consistency "
                             + (f"{scf['samples']} samples FIT {f3(scf['acc'])} SELECT {f3(scs['acc'])}" if scf else "pending") + ") ")
        out["not_stopped_text"] = ", ".join(f"{b} {f3(x['stock'])}/{f3(x['members'])}" for b, x in ns.items()) + " (stock/members)"
        self.say(f"r{r}: members tie-broken / per sample, FIT | SELECT: " + "; ".join(f"{n} {x['fit_tie']:.3f}/{x['fit_per_sample']:.3f} | {x['select_tie']:.3f}/{x['select_per_sample']:.3f}" for n, x in mem.items()))
        if any(x["canonical_rows"] for x in mem.values()):
            self.say(f"r{r}: vlmsbiased canonical-answer rate: " + ", ".join(f"{n} {f3(x['canonical_rate'])}" for n, x in mem.items()) + f" on {max(x['canonical_rows'] for x in mem.values())} rows")
        self.say(f"r{r}: prior-phrase reasoning (samples, rows): " + ", ".join(f"{n} {x['samples']}/{x['rows']}" for n, x in hits.items()) + f"; hits in {rd}/prior_phrases.jsonl")
        vl = out["voted_letters"]
        if vl:
            self.say(f"r{r}: DS voted letters {'on' if vl['on'] else 'off (count only)'}: {vl['changed']} rows change answer (FIT {vl['changed_fit']}: {vl['changed_fit_right']} to right, {vl['changed_fit_wrong']} from right; SELECT {vl['changed_select']}), {vl['rows_cut']} rows cut")
        return out

    def side_item(self, tag, g, rows, meta, samples=2, part=None):
        """A low line whose answers land in sidecheck/, a dir no read of the run opens: never in ev/, never in the ledger, no client waits on it, a failure of it stops nothing.
        part: rows per line for a long side line (each part its own low line, so it never holds a card long); None keeps one line, as before."""
        d = f"{self.work}/sidecheck/{tag}"
        if (os.path.exists(f"{d}/item.json") and (jload(f"{d}/item.json") or {}).get("gen", {}) == GEN_KEY) or not rows:
            return None
        model = self.ev.model(g)
        if g.get("vision") and not os.path.exists(f"{model}/config.json"):
            return dict(tag=tag, skipped="no checkpoint")
        os.makedirs(d, exist_ok=True)
        parts = [rows] if not part or len(rows) <= part else [rows[k: k + part] for k in range(0, len(rows), part)]
        names, outs = [], []
        for k, P in enumerate(parts):
            sfx = ("" if len(parts) == 1 else f".{k:02d}") + gsfx()
            with open(f"{d}/rows{sfx}.jsonl", "w") as o:
                for x in P:
                    o.write(json.dumps(x, ensure_ascii=False) + "\n")
            name = f"hb{self.ev.tag}_s{sha('side' + tag + sfx)[:12]}"
            it = dict(kind="eval", model=model, engine=g["engine"], rows=f"{d}/rows{sfx}.jsonl", out=f"{d}/a{sfx}.jsonl", n=len(P), old=name, ids=[x["id"] for x in P])
            jdump(f"{HB}/jobs/{name}.json", [it])
            self.ev.q.submit(name, f"python -u rsi/hboost.py job-batch {HB}/jobs/{name}.json {name} --samples {samples} --model {model} --pool {IID} " + self.ev.job_tail())
            names.append(name)
            outs.append(it["out"])
        x = dict(meta, tag=tag, line=names[0], rows=len(rows), outs=outs, **(dict(gen=GEN_KEY) if GEN_KEY else {}))
        if len(parts) > 1 or samples != 2:
            x.update(lines=names, samples=samples)
        jdump(f"{d}/item.json", x)
        return {k: v for k, v in x.items() if k != "member"}

    def side_lines(self, r, chosen, props, kids):
        """Each accepted member with a weight edit plus other ops, same genome without it, on SELECT: network edits (engine, dirs) off, and patch off (stock checkpoint, the tokens live on the checkpoint and go with it). Each gate member: its note as a prompt prefix on its parent, on the rows it touched."""
        out, sel = [], self.data.of("select")
        for m in chosen:
            g = m["genome"]
            net, v = bool(g.get("engine") or g.get("dirs")), g.get("vision") or {}
            cuts = []
            if net and (m.get("program") or v):
                cuts.append(("network", dict(g, engine=[], dirs=[])))
            if v.get("patch") not in (None, 16) and (m.get("program") or net):
                cuts.append(("patch", dict(g, vision=None)))
            for what, g2 in cuts:
                rows, abst = program_rows(self.data, dict(m, genome=g2), sel)
                x = self.side_item(f"{m['name']}.no_{what}", g2, rows, dict(what=f"without {what}", member=m, name=m["name"], abstains=len(abst)))
                if x:
                    out.append(x)
        by = dict(REG)
        by.update({m["name"]: m for m in self.state["members"]})
        by.update({m["name"]: m for m, _ in kids})
        for row in props:
            m = by.get(row.get("name"))
            if not m or not row.get("valid") or "gate" not in (row.get("own_ops") or []) or not m.get("note"):
                continue
            parent = by.get(m["parents"][0])
            ids = [i for i in self.touched(m, parent, self.data.of("score")) if in_gate_scope(self.data, m, i)]
            rows = []
            for i in ids:
                p = dict(self.data.rows[i])
                p["prompt"] = place_images(m["note"] + "\n\n" + p["prompt"], len(p.get("image_paths") or []), "start")
                rows.append(p)
            x = self.side_item(f"{m['name']}.note_prompt", parent["genome"], rows, dict(what="note as prompt on the parent", member=m, name=m["name"], parent=parent["name"], parent_member=parent))
            if x:
                out.append(x)
        part = self.a.get("batch_rows") or 100
        for m in chosen:            # coordinator 11-08: a program member with a genome change also answers ID with its genome and no program. Report only, low, on idle cards
            g = m["genome"]
            if m.get("program") and not is_stock(g):
                rows = [dict(self.data.rows[i], id=i) for i in self.data.id_rows]
                x = self.side_item(f"{m['name']}.genome_id", g, rows, dict(what="genome without its program, ID rows", member=m, name=m["name"]), part=part)
                if x:
                    out.append(x)
        s4 = jload(f"{self.work}/sidecheck/stock4/item.json")
        if self.a.get("stock_sc", True) and (s4 is None or s4.get("gen", {}) != GEN_KEY):        # once a run: stock's 4 samples on every learn and ID row, for its self-consistency figure
            st = next((m for m in self.state["members"] if m["name"] == "stock"), None)
            if st:
                rows = [dict(self.data.rows[i], id=i) for i in list(self.data.learn) + list(self.data.id_rows)]
                x = self.side_item("stock4", st["genome"], rows, dict(what="stock, 4 samples", member=st, name="stock"), samples=4, part=part)
                if x:
                    out.append(x)
        return out

    def side_read(self, r, rd):
        """The side lines that have landed: accuracy per sample on their rows, the member's and the side line's. Into ops.jsonl and the log, nowhere else."""
        acc = lambda v, ids: float(np.mean([s["ok"] for i in ids for s in v.get(i, {}).values()] or [np.nan]))
        for f in sorted(glob.glob(f"{self.work}/sidecheck/*/item.json")):
            d = os.path.dirname(f)
            x = jload(f)
            outs = x.get("outs") or [f"{d}/a.jsonl"]
            if os.path.exists(f"{d}/read{gsfx()}.json") or not all(os.path.exists(o) and gen_ok(o) for o in outs) or x.get("samples", 2) != 2:     # stock4 is read by sc_k; an item of another regime is made again by side_item
                continue
            v = {}
            for o in outs:
                v.update(votes_of(o))
            mine = load_member(x["member"])
            ids = [i for i in v if i in mine]
            rec = dict(kind="side_read", round=r, name=x["name"], what=x["what"], rows=len(ids), acc_side=acc(v, ids), acc_member=acc(mine, ids), member_minus_side=acc(mine, ids) - acc(v, ids), not_run=sum(1 for s in v.values() for y in s.values() if y.get("key") is None))
            if x.get("parent_member"):
                par = load_member(x["parent_member"])
                rec.update(parent=x["parent"], acc_parent=acc(par, ids))
            jdump(f"{d}/read{gsfx()}.json", rec)
            with open(f"{rd}/ops.jsonl", "a") as o:
                o.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self.say(f"r{r}: side line {x['name']} {x['what']}: {len(ids)} rows, member {rec['acc_member']:.3f} side {rec['acc_side']:.3f} difference {rec['member_minus_side']:+.3f}")

    def ahead(self, r):
        """Fix 4, prereg ahead: at the decide of round r, low lines for what no decision waits on: each program member's bare answers (a gate write needs them) and the reporter's ID rows, so they run while the proposer writes.
        Bring-up starts anyway the moment this returns (round r + 1's ensure) and the grid follows it; both feed the prompts, so they stay where they are. Their rows are left out here. A row asked here and needed sooner comes back through the ledger."""
        if not self.a.get("ahead") or r >= self.a["rounds"]:
            return
        members = self.state["members"]
        fit = [i for i in self.data.learn if self.data.role[i] != "select"]
        test = self.data.id_rows[: self.a["limit"]] if self.a.get("limit") else self.data.id_rows
        bares = {vh.gid(m["genome"]): member(m["genome"], None, name="bare") for m in members if m.get("program")}
        c, self.ev.client = getattr(self.ev, "client", "driver"), "ahead"
        try:
            for m in members:
                self.ev.build(m["genome"])
            self.ev.pending = []         # ensure waits on the builds
            self.ev.fill_many([(b, fit) for b in bares.values()] + [(m, own_rows(m, test)) for m in members], getattr(self.ev, "chunk", 100))
            self.say(f"r{r}: queued ahead: {len(self.ev.fills)} low lines, bare of {len(bares)} genomes, ID rows of {len(members)} members")
        except Exception as e:
            self.say(f"r{r}: queue-ahead failed, the round goes on without it: {type(e).__name__}: {str(e)[:200]}")
        finally:
            self.ev.client, self.ev.fills = c, []       # not filler: fill_settle does not stop these

    def line_report(self, r, rd, t0):
        """One record per queue line that asked rows in this round, from the ledger, every client: rows wanted and answered, samples not stopped (finish other than stop), samples empty."""
        L = collections.defaultdict(lambda: dict(wanted=0, answered=0, samples=0, not_stopped=0, empty=0, pending=0, failed=0, too_long=0, client=None, member=None))
        names = {self.ev.dir_of(m): m["name"] for m in list(REG.values()) + self.state["members"]}
        for f in glob.glob(f"{EVR}/ev/*/asked.jsonl"):
            xs = [json.loads(l) for l in open(f)]
            gone = {x["out"] for x in xs if x.get("cancel")}       # 1109c audit 11-09: a cancelled ask is not pending (the cancel is its own later record)
            for x in xs:
                if x.get("cancel") or x.get("t", 0) < t0 or (x["out"] in gone and not os.path.exists(x["out"])):
                    continue
                c = L[self.ev.line_of(x) or x["out"]]
                c["client"] = x.get("client")
                c["member"] = names.get(os.path.dirname(f), os.path.basename(os.path.dirname(f)))
                c["wanted"] += len(x["rows"])
                if not os.path.exists(x["out"]):
                    c["failed" if os.path.exists(x["out"] + ".failed") else "pending"] += len(x["rows"])
                    continue
                seen, tl = set(), set()
                for a in open(x["out"]):
                    y = json.loads(a)
                    if y.get("finish") in NO_RUN:
                        tl.add(y["id"])
                    seen.add(y["id"])
                    c["samples"] += 1
                    c["not_stopped"] += y.get("finish") != "stop"
                    c["empty"] += not (y.get("content") or "").strip()
                c["answered"] += len(seen - tl)
                c["too_long"] += len(tl)
        flagged = {self.ev.dir_of(m) for m in list(REG.values()) + self.state["members"] if m.get("new_rule_only")}
        flagged = {names.get(x, os.path.basename(x)) for x in flagged}
        rows = [dict(line=k, **v, new_rule_only=v["member"] in flagged, not_stopped_pct=100. * v["not_stopped"] / max(1, v["samples"]), empty_pct=100. * v["empty"] / max(1, v["samples"])) for k, v in sorted(L.items())]
        G_ = jload(f"{rd}/grid.json", None) or {}
        with open(f"{rd}/lines.jsonl", "w") as o:
            for x in rows:
                o.write(json.dumps(x) + "\n")
            o.write(json.dumps(dict(kind="grid", line="grid", anchor=G_.get("anchor"), answered_share=G_.get("anchor_answered"), anchor_fit=G_.get("anchor_fit"), rows=G_.get("rows"), cell_errors=G_.get("cell_errors"))) + "\n")
        bad = collections.defaultdict(lambda: [0, 0])
        for x in rows:
            bad[x["member"]][0] += x["too_long"]
            bad[x["member"]][1] += x["failed"]
        if any(a_ or b_ for a_, b_ in bad.values()):
            self.say(f"r{r}: too long / failed rows by member: " + "; ".join(f"{k} {a_}/{b_}" for k, (a_, b_) in sorted(bad.items(), key=str) if a_ or b_))
        by = collections.defaultdict(lambda: [0, 0, 0, 0, 0, 0])
        for x in rows:
            b = by[x["client"]]
            for k_, f_ in enumerate(("wanted", "answered", "samples", "not_stopped", "empty")):
                b[k_] += x[f_]
            b[5] += 1
        self.say(f"r{r}: lines: " + "; ".join(f"{c}: {b[5]} lines, rows wanted {b[0]} answered {b[1]}, not stopped {100 * b[3] / max(1, b[2]):.1f}%, empty {100 * b[4] / max(1, b[2]):.1f}%" for c, b in sorted(by.items(), key=str)))
        return rows

    def stop_count(self):
        """Rounds since the last accepted member whose raised slots built and ran rows. A round with no accepted member that lost them to failures does not count, and does not end the streak."""
        n = 0
        for x in reversed(self.state.get("rounds", [])):
            if x.get("accepted"):
                break
            n += bool((x.get("plateau") or {}).get("counted"))
        return n

    def stop_reason(self):
        """Prereg stop rule (owner, 2026-11-07). The half 1 read is a stop condition only, never used to pick members."""
        if not self.a.get("stop_rule"):
            return None
        if os.path.exists(f"{self.work}/STOP"):
            return "owner"
        R_ = self.state.get("rounds", [])
        if self.plateau_on:
            if self.stop_count() >= PLATEAU_STOP:
                return f"{PLATEAU_STOP} rounds with no accepted member whose raised weight slots built and ran rows"
        elif len(R_) >= 3 and not any(x.get("accepted") for x in R_[-3:]):
            alive = [n for n, c in (self.state.get("c1109_carry") or {}).items() if c.get("status") == "carried"]
            if alive:                # 1109c audit 11-09: a round with an A8 carry still alive is active (its e-process is pending)
                self.say(f"stop rule: 3 rounds with no accepted member, but A8 carries alive {alive}: not stopped")
            else:
                return "3 rounds with no accepted member"
        reads = {}
        f = f"{self.work}/report_learn.jsonl"
        if os.path.exists(f):
            for l in open(f):
                x = json.loads(l)
                reads[x["round"]] = x["pool"]["id_pooled"]
        v = [reads[k] for k in sorted(reads)]
        if len(v) >= 4 and v[-1] - v[-4] < .005:
            return f"3 reporter reads with pool half 1 gain under 0.5 pp ({100 * (v[-1] - v[-4]):+.2f})"
        return None

    def run(self):
        from vrsi27.boost.meta import Population
        from vrsi27.boost.lens import initial_sources
        pop = Population(f"{self.work}/meta")
        if not pop.entries:
            src = initial_sources(LENSES)
            for k, s in src.items():
                pop.add(k, s)
            pop.incumbent = self.a["lens0"]
            pop.save()
        for r in range(self.state["round"] + 1, self.a["rounds"] + 1):
            if r >= 2 and not self.a.get("noweights") and not self.a.get("plain") and not self.state.get("extra_seeds"):
                self.state["members"] = self.state["members"] + extra_seeds()
                self.state["extra_seeds"] = True
                self.save()
            if r >= 2 and not self.a.get("plain") and not self.state.get("zx_reset"):       # round 1's rewards came from the biased z; they no longer steer the lenses or the operators
                for p in self.state["hist"]:
                    if p.get("round", 0) < 2 and "reward" in p:
                        p["reward_old"], p["reward"] = p["reward"], 0.
                self.state["zx_reset"] = True
                self.save()
            am = active_amendment(self.work, r) or None
            if am and r >= am["from_round"] and getattr(self, "switched", None) != am["iid"]:
                self.data = self.ev.data = Data(self.a.get("limit"), am["iid"], am["ood"], am.get("exclude"))
                self.switched = am["iid"]
                self.pool_file = am["iid"]                       # a write's fit rows come from the amended pool
                self.say(f"r{r}: amendment: pool {am['iid']}: learn {len(self.data.learn)} rows, ID {len(self.data.id_rows)}, OOD {len(self.data.ood_rows)}")
            if am:
                jdump(f"{self.work}/r{r:03d}.pool.json", dict(round=r, iid=am["iid"]))    # with the roles' hashes, this says which rows the round searched on
            why = self.stop_reason()
            if why:
                jdump(f"{self.work}/stop.json", dict(t=vh.NOW(), after_round=r - 1, why=why))
                self.say(f"STOP after round {r - 1}: {why}")
                return
            self.round(r)
        jdump(f"{self.work}/stop.json", dict(t=vh.NOW(), after_round=self.a["rounds"], why="rounds"))
        self.say("run done")


# ---------------------------------------------------------------- the report: ID and OOD, separate from selection ---------------------------------
def report_once(work, ev, data, with_ood=False):
    from vrsi27.boost.calibrate import calibrated_scores
    from vrsi27.boost.edge import atom_scores
    from vrsi27.boost.pooling import fit_fc_evidence
    a = jload(f"{work}/prereg.json")
    S = jload(f"{work}/state.json", dict(round=0, members=a["seeds"], rounds=[]))
    members = S["members"]
    REG.update({m["name"]: m for m in members})
    ood_rows = data.ood_rows if with_ood else []          # owner 11-06: ID first; OOD is scored once, together, at the end (report --ood)
    test = data.id_rows + ood_rows
    if a.get("limit"):
        test = data.id_rows[: a["limit"]] + ood_rows[: a["limit"]]
    # the rows of the report go in the one ledger: a landed chunk shifted what was missing, so the same rows went out again under new chunk names (r2: 7582 of 9991 reporter rows)
    ev.client = "reporter"
    for m in members:
        try:
            ev.build(m["genome"])
            ev.fill_many([(m, own_rows(m, test + data.learn))], getattr(ev, "chunk", 100))
            lost = ev.dead_for(m, own_rows(m, test + data.learn))
            if lost:
                ev.write_failed(m, lost, lambda x: print("report:", x, flush=True))
        except Bad as e:
            print("report:", m["name"], e, flush=True)
    ev.pending, ev.fills = [], []                                    # the report never waits on a job; it reads what has landed
    votes = {m["name"]: load_member(m) for m in members}
    ready = [m for m in members if all(i in votes[m["name"]] for i in test) and all(i in votes[m["name"]] for i in data.learn)]
    if not any(m["name"] == "stock" for m in ready) or len(ready) < 2:
        return None
    ids = data.learn + test
    tab = Table(data, ids, {m["name"]: votes[m["name"]] for m in ready})
    pool = tab.pool(data, ready, votes)
    ix = lambda f: np.array([k for k, i in enumerate(ids) if f(i)], int)
    fit, sel = ix(lambda i: data.role.get(i) in ("show", "score")), ix(lambda i: data.role.get(i) == "select")
    idr, oodr = set(data.id_rows), set(ood_rows)
    idt, oot = ix(lambda i: i in idr), ix(lambda i: i in oodr)
    ref = REF(ready)
    kids = [k for k, m in enumerate(ready) if m["origin"] == "proposal"]
    Sc = atom_scores(pool, fit, kids, ref)
    # the same fit as the driver's state_of: no groups. row_groups() gave one group '' on bench/id ids, which made the v columns copies of w and F0 lost part of each weight (11-07)
    res = fit_fc_evidence(Sc, pool.y, pool.opt_mask, fit, list(range(len(ready))), groups=None, own_precision=kids)
    Fsh, Fpp = res.F0, res.F
    import res_fuse
    mode, F_st = res_fuse.mode_of(a), None
    res_fuse.VOTED_LETTERS = bool(a.get("ds_voted_letters"))
    if mode != "stacking":        # the pool's answers from the fusion, on these same votes; stacking stays beside it
        from vrsi27.boost.splits import Splits
        vr = {m["name"]: votes[m["name"]] for m in ready}
        t_ = time.time()
        fu = res_fuse.fuse(ready, vr, data, mode, ids=ids, sp=Splits(fit=fit, select=sel, held=np.array([], int), show=ix(lambda i: data.role.get(i) == "show"), score=ix(lambda i: data.role.get(i) == "score")))
        t_ = time.time() - t_
        F_st, Fsh, Fpp = res.F0, fu.F, fu.F
    acc = lambda k, idx: float(pool.correct[k, idx].mean())
    best = max(range(len(ready)), key=lambda k: acc(k, fit))
    sk = REF(ready)
    ec_id = err_stats(pool, ready, idt)[0]
    ec_ood = err_stats(pool, ready, oot)[0] if len(oot) else None
    aood = (lambda f: f(oot)) if len(oot) else (lambda f: None)
    rec = dict(t=time.strftime("%H:%M:%S"), round=S["round"], members=[m["name"] for m in ready], total_members=len(members),
               pool=dict(fit=tab.acc(Fsh, fit), select=tab.acc(Fsh, sel), id_shared=tab.acc(Fsh, idt), id_pooled=tab.acc(Fpp, idt), ood=aood(lambda x: tab.acc(Fsh, x))),
               best=dict(name=ready[best]["name"], fit=acc(best, fit), select=acc(best, sel), id=acc(best, idt), ood=aood(lambda x: acc(best, x))),
               stock=dict(fit=acc(sk, fit), select=acc(sk, sel), id=acc(sk, idt), ood=aood(lambda x: acc(sk, x))), err_corr_id=ec_id, err_corr_ood=ec_ood,
               singles={m["name"]: dict(id=acc(k, idt), ood=aood(lambda x, k=k: acc(k, x))) for k, m in enumerate(ready)}, weights=dict(zip([m["name"] for m in ready], [float(x) for x in res.w])))
    ds = {}
    for name, idx in (("id", idt), ("ood", oot)):
        by = collections.defaultdict(list)
        for k in idx:
            by[data.rows[ids[k]].get("bench") or ids[k].split("/")[0]].append(k)
        F = Fpp if name == "id" else Fsh
        ds[name] = {b: dict(n=len(v), pool=tab.acc(F, np.array(v)), best=acc(best, np.array(v)), stock=acc(sk, np.array(v))) for b, v in by.items()}
    rec["by_dataset"] = ds
    try:                      # report only (coordinator 11-08): per-sample means beside the tie-broken figures, stock's self-consistency, the pool's FIT held out by picture
        I_ = lambda idx: [ids[k] for k in idx]
        ps = lambda k, idx: per_sample(votes[ready[k]["name"]], I_(idx))
        rec["per_sample"] = dict(best=dict(fit=ps(best, fit), select=ps(best, sel), id=ps(best, idt), ood=aood(lambda x: ps(best, x))), stock=dict(fit=ps(sk, fit), select=ps(sk, sel), id=ps(sk, idt), ood=aood(lambda x: ps(sk, x))),
                                 singles={m["name"]: dict(id=ps(k, idt), ood=aood(lambda x, k=k: ps(k, x))) for k, m in enumerate(ready)})
        rec["stock_sc"] = dict(fit=sc_k(work, I_(fit)), select=sc_k(work, I_(sel)), id=sc_k(work, I_(idt)))
        rec["pool_fit_oof"] = oof_acc(fu) if mode != "stacking" else None
        cols = {m["name"]: pool.correct[k] for k, m in enumerate(ready)}
        cols["pool"] = right_rows(fu.tab if mode != "stacking" else tab, Fpp, np.arange(len(ids)))
        rec["se_cluster"] = dict(id=se_table(data, tab, idt, cols), fit=se_table(data, tab, fit, cols), select=se_table(data, tab, sel, cols))
        rec["not_stopped_id"] = {b: dict(stock=not_stopped([votes[ready[sk]["name"]]], I_(v)), members=not_stopped([votes[m["name"]] for m in ready], I_(v)), rows=len(v)) for b, v in ((b, np.array(v)) for b, v in sorted(collections.defaultdict(list, {b: [k for k in idt if (data.rows[ids[k]].get("bench") or ids[k].split("/")[0]) == b] for b in {data.rows[ids[k]].get("bench") or ids[k].split("/")[0] for k in idt}}).items()))}
    except Exception as e:
        rec["report_extras_error"] = f"{type(e).__name__}: {str(e)[:200]}"
    byc = collections.defaultdict(list)
    for k in oot:
        byc[vh.class_of(data.rows[ids[k]], "ood") or "general"].append(k)
    rec["ood_by_class"] = {c: dict(n=len(v), pool=tab.acc(Fsh, np.array(v)), best=acc(best, np.array(v)), stock=acc(sk, np.array(v))) for c, v in sorted(byc.items())}
    rec["fusion"] = mode
    if F_st is not None:
        rec["pool_stacking"] = dict(fit=tab.acc(F_st, fit), select=tab.acc(F_st, sel), id_shared=tab.acc(F_st, idt), id_pooled=tab.acc(res.F, idt), ood=aood(lambda x: tab.acc(F_st, x)))
        rec["weights_stacking"] = rec.pop("weights")
        rec.update(fusion_secs=t_, table_columns=int(tab.C), table_rows=len(ids), table_over_cap=int(tab.C) > C_MAX, fusion_info={k: v for k, v in fu.info.items() if k in ("rows_in_em", "em_passes", "agree_share_fit", "agree_share_select", "passed", "T", "agree_floored", "no_vote_rows")})
    rec["views"] = views_of(tab, pool, Fpp, ready, data, dict(fit=fit, select=sel, id=idt, **(dict(ood=oot) if len(oot) else {})), F_st)
    if (a.get("narrow_pool_rows") or a.get("plateau")) and mode != "stacking":     # beside the main pool: without what came in by the new narrow rule only, or by a plateau combination, nor anything descended from it, which could not have existed without it
        out = {m["name"] for m in members if m.get("new_rule_only")}
        for m in members:
            if set(m.get("parents") or []) & out:
                out.add(m["name"])
        strict = [m for m in ready if m["name"] not in out]
        fs = res_fuse.fuse(strict, vr, data, mode, ids=ids, sp=fu.sp) if len(strict) < len(ready) else fu
        rec["pool_strict"] = dict(members=[m["name"] for m in strict], left_out=sorted(out), fit=tab.acc(fs.F, fit), select=tab.acc(fs.F, sel), id_shared=tab.acc(fs.F, idt), ood=aood(lambda x: tab.acc(fs.F, x)))
        print(f"r{S['round']} strict pool: {len(strict)} members, left out {sorted(out)}: FIT {rec['pool_strict']['fit']:.3f} SELECT {rec['pool_strict']['select']:.3f} ID {rec['pool_strict']['id_shared']:.3f}", flush=True)
    return rec


def cmd_report(work, learn=False, with_ood=False):
    am = jload(f"{work}/amendment.json", {}) if learn else {}
    data = Data(jload(f"{work}/prereg.json").get("limit"), am.get("iid"), am.get("ood"), jload(f"{work}/amendment.json", {}).get("exclude"))
    ev = Ev(work + ("/rep_learn" if learn else "/rep"), data)
    ev.chunk = ev.batch_cap = jload(f"{work}/prereg.json").get("batch_rows", 250)      # reports give way to round work: short batches, so their cards turn over often
    tail_ = "_learn" if learn else ""
    last = None
    used = am.get("iid")
    while True:
        if learn:                    # follows the arm: the report of a round reads the pool that round ran on
            r_ = jload(f"{work}/state.json", {}).get("round", 0)
            now = active_amendment(work, r_)
            if now.get("iid") != used:
                used = now["iid"]
                data = ev.data = Data(jload(f"{work}/prereg.json").get("limit"), now["iid"], now.get("ood"), now.get("exclude"))
        rec = report_once(work, ev, data, with_ood)
        if rec:
            rec["pool_file"] = used
            k = (rec["round"], tuple(rec["members"]))
            if k != last:
                last = k
                open(f"{work}/report{tail_}.jsonl", "a").write(json.dumps(rec, ensure_ascii=False) + "\n")
                p, b, s = rec["pool"], rec["best"], rec["stock"]
                o3 = lambda x: "-" if x is None else f"{x:.3f}"
                line = (f"{rec['t']} r{rec['round']} members {len(rec['members'])}/{rec['total_members']} | pool FIT {p['fit']:.3f} SELECT {p['select']:.3f} ID {p['id_pooled']:.3f} (shared {p['id_shared']:.3f}) OOD {o3(p['ood'])} "
                        f"| best {b['name']} FIT {b['fit']:.3f} SELECT {b['select']:.3f} ID {b['id']:.3f} OOD {o3(b['ood'])} | stock FIT {s['fit']:.3f} SELECT {s['select']:.3f} ID {s['id']:.3f} OOD {o3(s['ood'])} | err corr ID {rec['err_corr_id']:.3f} OOD {o3(rec['err_corr_ood'])}"
                        + (" | OOD by class (pool/best/stock, n) " + "; ".join(f"{c} {v['pool']:.3f}/{v['best']:.3f}/{v['stock']:.3f} {v['n']}" for c, v in rec["ood_by_class"].items()) if rec["ood_by_class"] else ""))
                vw = rec["views"]["id"]
                line += " | ID pool all {:.3f} without vlmsbiased {:.3f} macro by bench {:.3f}".format(vw["pool"]["all"], vw["pool"]["no_vlmsbiased"] if vw["pool"]["no_vlmsbiased"] is not None else float("nan"), vw["pool"]["macro_bench"])
                for n_, c_ in vw["members"].items():
                    f3 = lambda x: "-" if x is None else f"{x:.3f}"
                    line += f"\n    {n_}: ID all {f3(c_['all'])} without vlmsbiased {f3(c_['no_vlmsbiased'])} macro by bench {f3(c_['macro_bench'])} answers {f3(c_['answered'])} acc there {f3(c_['on_answered'])}"
                if rec.get("per_sample"):
                    f3_ = lambda x: "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"
                    ps_, sc_ = rec["per_sample"], rec.get("stock_sc") or {}
                    line += (f"\n    stock three ways: tie-broken ID {f3_(s['id'])} FIT {f3_(s['fit'])} SELECT {f3_(s['select'])} | per sample ID {f3_(ps_['stock']['id'])} FIT {f3_(ps_['stock']['fit'])} SELECT {f3_(ps_['stock']['select'])} | self-consistency "
                             + (" ".join(f"{k} {f3_(v['acc'])} ({v['samples']} samples, {v['rows']} rows)" for k, v in sc_.items() if v) or "pending")
                             + f" | best per sample ID {f3_(ps_['best']['id'])} | pool FIT held out by picture {f3_(rec.get('pool_fit_oof'))} in sample {f3_(p['fit'])}"
                             + " | not stopped on ID by bench (stock/members) " + ", ".join(f"{b} {f3_(x['stock'])}/{f3_(x['members'])}" for b, x in (rec.get("not_stopped_id") or {}).items()))
                if rec.get("table_over_cap"):
                    line += f"\n    TABLE WIDER THAN {C_MAX}: {rec['table_columns']} columns"
                print(line, flush=True)
                open(f"{work}/report{tail_}.txt", "a").write(line + "\n")
        time.sleep(120)


# ---------------------------------------------------------------- plan and the proposer's job ------------------------------------------------------
def cmd_stockfill(work):
    """Stock on every learn and ID row, at normal priority, once: a plain run's baseline exists before its first report."""
    am = jload(f"{work}/amendment.json", {})
    data = Data(jload(f"{work}/prereg.json").get("limit"), am.get("iid"), am.get("ood"), am.get("exclude"))
    ev = Ev(work + "/stockfill", data)
    ev.chunk = ev.batch_cap = 250
    stock = next(m for m in jload(f"{work}/prereg.json")["seeds"] if m["name"] == "stock")
    have = load_member(stock)
    print("stock has", sum(1 for i in data.id_rows if i in have), "of", len(data.id_rows), "ID rows,", sum(1 for i in data.learn if i in have), "of", len(data.learn), "learn rows", flush=True)
    ev.evaluate(stock, data.learn + data.id_rows)
    print("queued", len(ev.pending), "batches", flush=True)


def cmd_plan(work, argv):
    def opt(flag, default, f=int):
        return f(argv[argv.index(flag) + 1]) if flag in argv else default
    if os.path.exists(f"{work}/prereg.json"):
        raise SystemExit("prereg.json exists")
    st = lambda d: jload(f"{V}/{d}/state.json")
    stock = dict(vision=None, engine=[], dirs=[])
    p8 = jload(f"{V}/p8eq.model/prereg.json")["root"]
    rand = st("stock.random")["accepted"][0]
    cl = {x["gid"]: x for x in st("stock.claude")["accepted"]}
    seeds = [member(stock, name="stock"), member(p8, name="p8eq"), member(rand, name="p4bilinear", note_origin="stock.random r0, an accepted edit of the earlier run"),
             member(cl["3d9992e59001c0"], name="tok3600", note_origin="stock.claude r0, an accepted edit of the earlier run"), member(cl["bdf0c3b448faa2"], name="p8box", note_origin="stock.claude r0, an accepted edit of the earlier run")]
    if "--seeds" in argv:
        keep = argv[argv.index("--seeds") + 1].split(",")
        seeds = [m for m in seeds if m["name"] in keep]
    assert vh.gid(seeds[0]["genome"]) == "aefc601d4c18b9", vh.gid(seeds[0]["genome"])
    a = dict(written=vh.NOW(), slots=opt("--slots", 16), stage2=opt("--stage2", 3), rounds=opt("--rounds", 20), limit=opt("--limit", None), seed=opt("--seed", 1), lens0="residual_table", noweights="--noweights" in argv, arm="harness programs only on the stock base, no network differences" if "--noweights" in argv else "main", seeds=seeds, a=A, c_max=C_MAX, samples=2, proposer=vh.STOCK)
    jdump(f"{work}/prereg.json", a)
    print("planned", work, [m["name"] for m in seeds], {k: a[k] for k in ("slots", "stage2", "rounds", "limit")})


def cmd_job_gen(argv):
    """The frozen stock checkpoint, nothing applied to it. Thinking on."""
    q, out = argv[0], argv[1]
    try:
        rest = list(argv[2:])
        if "--prefix-cache" in rest:         # a 128 tail flag loop.py does not know
            k = rest.index("--prefix-cache")
            del rest[k:k + 2]
        sys.argv = ["loop.py", "--run-dir", os.path.dirname(out)] + rest
        sys.path[:0] = [CODE, os.path.join(CODE, "engine")]
        import rsi.loop as L
        from surgery import Surgeon
        a = L.args
        assert os.path.realpath(a.model) == os.path.realpath(vh.STOCK), "the proposer is the stock checkpoint"
        S = Surgeon(a.model, tp=a.tp, max_model_len=a.max_model_len, max_num_seqs=a.max_num_seqs, max_num_batched_tokens=a.max_num_batched_tokens, gpu_memory_utilization=a.gpu_util, kv_cache_dtype=a.kv_cache_dtype, multimodal=True)
        reqs = json.load(open(q))
        msgs = [[{"role": "system", "content": r["system"]}, {"role": "user", "content": r["user"]}] for r in reqs]
        t0 = time.time()
        outs = S.generate(msgs, n=1, sampling=dict(L.SAMPLING, temperature=.7), max_tokens=12000, reasoning_effort="medium")
        res = []
        for r, o in zip(reqs, outs):
            s = o["samples"][0]
            res.append(dict(id=r["id"], content=s["content"], reasoning=s["reasoning"], finish=s["finish_reason"], tokens=s["num_tokens"]))
        print(f"generated {len(res)} in {time.time() - t0:.0f}s; tokens {[x['tokens'] for x in res]}", flush=True)
        json.dump(res, open(out + ".tmp", "w"), ensure_ascii=False)
        os.replace(out + ".tmp", out)
    except BaseException:
        open(out + ".failed", "w").write("failed\n")
        raise


if __name__ == "__main__":
    c = sys.argv[1]
    if c in ("run", "report", "stockfill"):
        set_run(sys.argv[2])
    if c == "stockfill":
        cmd_stockfill(sys.argv[2]); sys.exit(0)
    if c == "plan":
        cmd_plan(sys.argv[2], sys.argv[3:])
    elif c == "run":
        Run(sys.argv[2]).run()
    elif c == "report":
        cmd_report(sys.argv[2], "--learn" in sys.argv, "--ood" in sys.argv)
    elif c == "job-gen":
        cmd_job_gen(sys.argv[2:])
    elif c == "job-batch":
        cmd_job_batch(sys.argv[2:])
    elif c == "job-worker":
        cmd_job_worker(sys.argv[2:])
    elif c == "side-read":
        set_run(sys.argv[2])
        R_ = Run(sys.argv[2])
        R_.side_read(R_.state["round"], f"{sys.argv[2]}/r{R_.state['round']:03d}")
    else:
        raise SystemExit(__doc__)
