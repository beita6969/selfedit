"""Shared helpers for the baselines (SFT, GRPO, PPO). Reuses the main system's grader and message format; nothing here re-implements them."""
import zlib, ast, json, os, re, sys, time, random, collections, hashlib, shutil

N = os.environ.get("BL_ROOT", "/home/bedicloud/sharestore2/zxc/selfedit")
CODE = f"{N}/code-hb"
STOCK = f"{N}/models/Qwen3.8-27B"
DATA = f"{N}/data/full"
SHM = "/dev/shm/selfedit-baseline"
OUT = f"{N}/baselines"
BD = os.environ.get("BL_DATA", f"{N}/baselines/data128_v3")   # code2 (grpo2, sft2, PPO): the 256-row pool v3 (prep128.py from xfer128_v3; v2 data128 kept); code/ keeps data/ for grpo1, sft1
PAPER_LEARN = f"{BD}/paper_learn.node.jsonl"     # main system's show + score rows (hb/iid_paper.jsonl), images remapped to the node, sha checked
PAPER_SELECT = f"{BD}/paper_select.node.jsonl"   # main system's select rows: reported, never trained on
TRAIN_FILES = [f"{BD}/train.jsonl"]   # prep_final.py: baseline train + CharXiv + show + score, minus everything overlapping select
VAL_FILES = [f"{BD}/val.jsonl"]
IID_EVAL = f"{BD}/iid_eval.node.jsonl"   # data128 only: the 128 IID evaluation rows per bench, read once at the end (--final-iid)
SHM_FLOOR_GB = 100          # owner: stop cleanly below this
DISK_FLOOR_GB = 8           # task: stop and report if the shared disk would fall below this
for p in (f"{CODE}/engine", f"{CODE}/mm"):
    if p not in sys.path:
        sys.path.insert(0, p)

# ---------- main system settings (rsi/vharness.py TAIL, rsi/loop.py SAMPLING)
EFFORT, BUDGET = "medium", 8192      # engine defaults; code2 runs pass max_tokens and effort explicitly in every generate op (--budget, --effort)
BUDGETS = (8192, 16384, 32768)       # coordinator 11-08: the owner picks one; main system, GRPO rollouts, evaluations, SFT traces and final IID use the same
EVAL_SAMPLING = dict(temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0, repetition_penalty=1.0)
ROLLOUT_SAMPLING = dict(temperature=1.0, top_p=1.0, top_k=-1, min_p=0.0, presence_penalty=0.0, repetition_penalty=1.0)
ENGINE = dict(tp=1, max_model_len=46000, max_num_batched_tokens=8192, max_num_seqs=64, kv_cache_dtype="fp8", multimodal=True)


HERE = os.path.dirname(os.path.abspath(__file__))
HB_DIR = f"{HERE}/hbgrade"            # the main driver's grader, unchanged: rsi/hboost.py, rsi/vharness.py, mm/grade_mm.py (frozen copies at freeze), pinned in pin.json
POOL_FILE = os.environ.get("BL_POOL", f"{N}/baselines/xfer128_v3/iid_paper128_v3.jsonl")   # pool v3 (coordinator 11-08): every record is graded from its pool row
OOD_EMPTY = f"{N}/baselines/xfer128_v3/ood_empty.jsonl"        # OOD later; Data needs a file
_sha = lambda f: hashlib.sha256(open(f, "rb").read()).hexdigest()


def _load_hb():
    """hboost.Data(None, pool v3) once, then hboost.regrade(record, pool_row) per answer (coordinator 11-08). The grading inputs (answer, kind,
    options, category, precision, bench) come from the pool row; stored golds, options or ok are never read. Every file is checked against pin.json."""
    pin = json.load(open(f"{HB_DIR}/pin.json"))
    files = dict(hboost=f"{HB_DIR}/rsi/hboost.py", vharness=f"{HB_DIR}/rsi/vharness.py", grade_mm=f"{HB_DIR}/mm/grade_mm.py", pool=POOL_FILE)
    got = {k: _sha(f) for k, f in files.items()}
    bad = {k: (got[k][:16], pin.get(k)) for k in files if not got[k].startswith(str(pin.get(k)))}
    if bad:
        raise RuntimeError(f"grader files differ from {HB_DIR}/pin.json: {bad}")
    sys.path.insert(0, f"{HB_DIR}/rsi")
    import hboost                                   # puts its own rsi, pb and mm first on sys.path, so grade_mm is the copy next to it
    hboost.Data(None, POOL_FILE, ood=OOD_EMPTY)
    assert len(hboost.POOL) == 1536, len(hboost.POOL)
    GRADER.update({k: got[k][:16] for k in files}, files=files, pool_rows=len(hboost.POOL))
    return hboost


GRADER = {}
GRADE_ERRORS = collections.Counter()
HB = _load_hb()


def rows_of(files):
    out = []
    for f in files:
        out += [json.loads(l) for l in open(os.path.join(DATA, f))]
    return out


def grade_rec(row, s):
    """s: a sample dict from Surgeon.generate. Returns (ok, pred): the driver's hboost.regrade of this output against the pool row of row["id"].
    Only the id is taken from row; its stored answer, options or kind are not used."""
    rec = dict(id=row["id"], sample=0, content=s.get("content") or "", reasoning=s.get("reasoning") or "", finish=s["finish_reason"], tokens=s.get("num_tokens", 0))
    if "loop" in s:
        rec["loop"] = s["loop"]
    pr = HB.POOL.get(row["id"])
    if pr is None:
        raise KeyError(f"{row['id']} is not a row of the pool {POOL_FILE}")
    try:
        g, ok, p = HB.regrade(rec, pr)
    except Exception as e:     # safety net only (hboost.grade already counts unreadable answers wrong): a grader exception must not end a 21.5 h run
        GRADE_ERRORS[type(e).__name__] += 1
        if sum(GRADE_ERRORS.values()) <= 20:
            print(f"GRADE_ERROR {row['id']} {type(e).__name__}: {str(e)[:120]}", file=sys.stderr, flush=True)
        return False, None
    gm = sys.modules.get("grade_mm")
    if gm is not None and "grade_mm_loaded" not in GRADER:
        GRADER["grade_mm_loaded"] = gm.__file__
        assert os.path.abspath(gm.__file__) == os.path.abspath(f"{HB_DIR}/mm/grade_mm.py"), f"grade_mm came from {gm.__file__}"
    return ok, p


def free_gb(path):
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 2**30


class Guard:
    """Space guard. stop() is True once /dev/shm or available RAM is below the owner's floor, or the shared disk below the task's floor. Other users share this RAM."""
    def __init__(self, shm_floor=SHM_FLOOR_GB, disk_floor=DISK_FLOOR_GB):
        self.shm_floor, self.disk_floor = shm_floor, disk_floor

    def state(self):
        avail = next(int(l.split()[1]) for l in open("/proc/meminfo") if l.startswith("MemAvailable")) / 2**20
        return dict(shm_free_gb=round(free_gb("/dev/shm"), 1), disk_free_gb=round(free_gb(N), 1), ram_avail_gb=round(avail, 1))

    def stop(self, need_shm_gb=0):
        s = self.state()
        if s["shm_free_gb"] - need_shm_gb < self.shm_floor:
            return f"shm free {s['shm_free_gb']} GB - need {need_shm_gb} < floor {self.shm_floor}"
        if s["ram_avail_gb"] - need_shm_gb < self.shm_floor:
            return f"RAM available {s['ram_avail_gb']} GB - need {need_shm_gb} < floor {self.shm_floor}"
        if s["disk_free_gb"] < self.disk_floor:
            return f"shared disk free {s['disk_free_gb']} GB < floor {self.disk_floor}"
        return None


def macro(oks_by_bench):
    return sum(sum(v) / len(v) for v in oks_by_bench.values()) / max(1, len(oks_by_bench))


def score_table(recs):
    """recs: list of dict(bench, ok). Per bench, macro over benches, all rows, all rows without vlmsbiased (the same four views as the main system's reports)."""
    by = collections.defaultdict(list)
    for r in recs:
        by[r["bench"]].append(bool(r["ok"]))
    no = [x for b, v in by.items() if b != "vlmsbiased" for x in v]
    al = [x for v in by.values() for x in v]
    return dict(per_bench={b: round(sum(v) / len(v), 4) for b, v in sorted(by.items())}, n={b: len(v) for b, v in sorted(by.items())},
                macro=round(macro(by), 4), macro_no_vlmsbiased=round(macro({b: v for b, v in by.items() if b != "vlmsbiased"}), 4) if len(by) > 1 else None,
                all=round(sum(al) / max(1, len(al)), 4), all_no_vlmsbiased=round(sum(no) / len(no), 4) if no else None)


def sqrt_sampler(rows, seed, q=0.5):
    """Bench sampling p_b proportional to n_b^q (Kang 2020 Eq. 1; q=1/2 is square-root). Within a bench, without replacement, reshuffled at each pass. Yields rows forever."""
    rng = random.Random(seed)
    by = collections.defaultdict(list)
    for r in rows:
        by[r["bench"]].append(r)
    benches = sorted(by)
    w = [len(by[b]) ** q for b in benches]
    pools = {b: [] for b in benches}
    while True:
        b = rng.choices(benches, w)[0]
        if not pools[b]:
            pools[b] = by[b][:]
            rng.shuffle(pools[b])
        yield pools[b].pop()


def focus_sampler(rows, focus_ids, seed, share=0.5, q=0.5):
    """Exactly `share` of draws come from the rows in focus_ids (select's categories) and the rest from all rows, both by sqrt_sampler. The whole-bench half uses `seed`, the focus half `seed + 1`, and the two alternate by a fixed rule, so the mix has no randomness of its own."""
    whole = sqrt_sampler(rows, seed, q); foc = sqrt_sampler([r for r in rows if r["id"] in focus_ids], seed + 1, q); acc = 0.0
    while True:
        acc += share
        if acc >= 1.0 - 1e-9:
            acc -= 1.0
            yield next(foc)
        else:
            yield next(whole)


def say(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)

CRN_SEED, CRN_GEN = 101, 0
# the main run's engine (Vesta 128 run tail: --max-num-seqs 64 --max-num-batched-tokens 8192, no --prefix-cache, fp8 KV, 46000, tp 1, util 0.85): every measurement runs on it
RUN_ENGINE = dict(max_num_seqs=64, max_num_batched_tokens=8192, enable_prefix_caching=False, kv_cache_dtype="fp8", max_model_len=46000, tp=1, util=0.85)
def crn_seeds(rows, seed=CRN_SEED, gen=CRN_GEN):
    """The main run's measurement seeds (Vesta res_driver_128/rsi/loop.py crn_seeds, --crn 1 --seed 101, CRN_GEN 0): one per row id; vLLM gives sample k seed + k."""
    return [zlib.crc32(f"{seed}|{gen}|{r['id']}".encode()) % (2 ** 31 - 1024) for r in rows]
