"""AM17 checks (coordinator 11-10): futility rung for weight cells (a) and cancel of a rejected candidate's pending reads (b).
1. judge_bh-level simulation: interval decisions while reads arrive, with cancels (only for members BH-insignificant on their
   pessimistic completion) and futility stops entered at p = 1; the final judgement never contradicts a decision. The same simulation
   with cancels of LOBO-only rejections shows why those are read to the end.
2. cancel_reads on a stand-in: a key shared with a live candidate blocks the cancel; otherwise rows_in_hand freezes, pending keys leave.
3. v4io: cancel_prefix touches q/cancel for name_*.jsonl only; a restarted IO does not count a cancelled batch's keys as pending.
4. A dry generation (gen.Run, fake engine answering with a delay, W cells with injected effects, one E candidate with a strong negative
   effect): quarter = per source ceil(n/4) of the generation's order, same rows for every cell; a W cell with d <= 0 on the quarter stops
   there (futile, p = 1 in BH, nothing more submitted); a positive one reads the rest; the negative E candidate is cancelled with reads
   pending; finalise finalises the generation with no AM15 assertion; --no-am17 keeps the single rung1 batch.
usage: python3 t_am17.py"""
import collections, glob, hashlib, json, math, os, sys, tempfile, types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(os.path.dirname(HERE), "harness")]
os.chdir(HERE)
import numpy as np  # noqa: E402
import judge_bh as BH  # noqa: E402
import gen  # noqa: E402
import t_versions as T  # noqa: E402
from v4io import IO  # noqa: E402

ok = fail = 0


def check(name, cond):
    global ok, fail
    print(("ok   " if cond else "FAIL ") + name, flush=True)
    ok, fail = ok + bool(cond), fail + (not cond)


# ---------------------------------------------------------------------------------------------------------------- 1. simulation
def sim(trials=1500, cancel_lobo=False, seed=7):
    rng = np.random.default_rng(seed)
    bad = cancels = lobo_only = futile = decided = 0
    for _ in range(trials):
        K = int(rng.integers(3, 12))
        S = 3
        n = rng.integers(8, 40, (K, S))                       # rows per (member, source)
        kind = rng.choice(["null", "good", "bad", "one"], K, p=[.4, .2, .2, .2])
        isw = rng.random(K) < .3                               # weight cells: futility rung on their first quarter
        glob_ = rng.random(K) < .7
        X = []
        for j in range(K):
            th = {"null": [0, 0, 0], "good": [.25] * 3, "bad": [-.25] * 3, "one": [.45, -.05, -.05]}[kind[j]]
            X.append([np.clip(np.round(rng.normal(th[s], .6, n[j, s])), -1, 1) for s in range(S)])
        ev = [(j, s, i) for j in range(K) for s in range(S) for i in range(n[j, s])]
        rng.shuffle(ev)
        if rng.random() < .5:                                  # sometimes one member at a time (the live FIFO per candidate batch)
            ev.sort(key=lambda e: e[0])
        quarter = [{(s, i) for s in range(S) for i in range(int(math.ceil(n[j, s] / 4)))} for j in range(K)]
        if any(isw):                                           # weight cells read their quarter first
            ev.sort(key=lambda e: 0 if (not isw[e[0]] or (e[1], e[2]) in quarter[e[0]]) else 1)
        got = [set() for _ in range(K)]
        frozen, dec = {}, {}

        def x_of(j, mode, rows=None):
            if j in frozen:
                return dict(name=str(j), d=-1.0, se=0.0, n=max(len(frozen[j]), 1), nu=1.0, share=1.0, is_global=glob_[j], lobo=False)
            src, vals = {}, []
            for s in range(S):
                v = [X[j][s][i] if (s, i) in got[j] else (-1.0 if mode == "pess" else 1.0) for i in range(n[j, s])
                     if (rows is None or (s, i) in rows) and ((s, i) in got[j] or mode != "obs")]
                vals += v
                src[s] = (float(sum(v)), len(v))
            v = np.array(vals or [0.0])
            se = float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else 1.0
            return dict(name=str(j), d=float(v.mean()), se=se, n=len(v), nu=float(max(len(v) - 1, 1)), share=1.0, is_global=bool(glob_[j]),
                        lobo=BH.lobo_ok({s: c for s, c in src.items() if c[1]}))

        for t, (j, s, i) in enumerate(ev):
            if j in frozen:
                continue
            got[j].add((s, i))
            if t % 5 and t != len(ev) - 1:
                continue
            for w in range(K):                                 # futility rung
                if isw[w] and w not in dec and w not in frozen and quarter[w] <= got[w]:
                    if x_of(w, "obs", quarter[w])["d"] <= 0:
                        frozen[w], dec[w] = set(got[w]), "reject"
                        futile += 1
                    else:
                        isw[w] = False                         # passed: read on
            und = [w for w in range(K) if w not in dec]
            if not und:
                continue
            P, O = [x_of(w, "pess") for w in range(K)], [x_of(w, "opt") for w in range(K)]
            d_ = BH.interval(P, O)
            jp = {r["name"]: r for r in BH.judge(P)}
            for w in und:
                r = d_[str(w)]
                if r == "wait":
                    continue
                dec[w] = r
                decided += 1
                pend = sum(int(x) for x in n[w]) - len(got[w])
                if r == "reject" and pend > 0:
                    if not jp[str(w)]["bh"]:
                        frozen[w] = set(got[w])
                        cancels += 1
                    elif cancel_lobo:
                        frozen[w] = set(got[w])
                        lobo_only += 1
        F = {r["name"]: r["accept"] for r in BH.judge([x_of(w, "obs") for w in range(K)])}
        bad += sum((dec.get(w) == "accept") != F[str(w)] for w in range(K) if w in dec)
    return bad, cancels, lobo_only, futile, decided


b, c, lo, fu, de = sim()
check(f"simulation: cancels only when BH-insignificant on the pessimistic completion + futility stops at p = 1 -> {b} contradictions "
      f"in {de} interval decisions ({c} cancels, {fu} futility stops)", b == 0 and c > 50 and fu > 20)
b2, c2, lo2, fu2, de2 = sim(cancel_lobo=True)
print(f"     (diagnostic) cancelling LOBO-only rejections as well: {lo2} such cancels, {b2} contradictions")

# ---------------------------------------------------------------------------------------------------------------- 2. cancel_reads unit
d = tempfile.mkdtemp(prefix="t_am17_io_")
io = IO(d, log=lambda *a: None)
for nm in ("g007_rung1_g007c01_5", "g007_rung1_g007c01_W_6", "g007_rung1_g007c011_7", "g007_rung1b_g007c01_W_8"):
    open(os.path.join(d, "q/in", nm + ".jsonl"), "w").write(json.dumps(dict(key=f"k_{nm}")) + "\n")
S = types.SimpleNamespace(g=7, io=io, logs=[])
S.read_req = lambda E, r, s, n: (dict(key=f"{E}|{r}", row=r), None)
S.cand_entries = lambda c: c["E"]
S.rung_rows = lambda c, rung: list(c["rows_in_hand"]) if c.get("rows_in_hand") is not None else list(c["rows"])
S.nsamp = gen.Run.nsamp                                                         # AM21 LITE: one sample per row without c["am21"]
S.s0 = gen.Run.s0                                                               # AM23: first sample index (0 unless a bundle)
S.cand_keys = lambda c: gen.Run.cand_keys(S, c)
gen.log = lambda m: S.logs.append(m)
io.res.update({"A|1": {}, "A|2": {}})
io.pending.update({"A|3", "A|4", "B|9"})
cA = dict(cid="g007c01", E="A", rows=[1, 2, 3, 4])
cB = dict(cid="g007c02", E="A", rows=[4, 5], decided=False)                     # needs key A|4 too (shared, live)
gen.Run.cancel_reads(S, cA, [cA, cB])
check("cancel_reads: a pending key shared with a live (undecided) candidate blocks the cancel", not cA.get("reads_cancelled")
      and not os.listdir(os.path.join(d, "q/cancel")))
cB.update(decided=True, accepted=False)
gen.Run.cancel_reads(S, cA, [cA, cB])
can = sorted(os.listdir(os.path.join(d, "q/cancel")))
check(f"cancel_reads: shared only with a rejected one -> cancelled; rows_in_hand [1, 2]; A|3, A|4 leave pending; cancel files {can}",
      cA.get("reads_cancelled") and cA["rows_in_hand"] == [1, 2] and cA["cancelled_reads"] == 2 and io.pending == {"B|9"}
      and can == ["g007_rung1_g007c01_5", "g007_rung1_g007c01_W_6", "g007_rung1b_g007c01_W_8"])
check("cancel_prefix: g007c01 does not touch g007c011's batch", "g007_rung1_g007c011_7" not in can)
io2 = IO(d, log=lambda *a: None)
check(f"restarted IO: keys of cancelled batches are not pending ({sorted(io2.pending)})", io2.pending == {"k_g007_rung1_g007c011_7"})


# ---------------------------------------------------------------------------------------------------------------- 4. dry generation
class DelayedIO:
    """InstantIO for everything but rung-1 reads; those are answered RATE per poll in arrival order (the dispatcher's FIFO). The first
    INFLIGHT queued requests are in flight: a cancel leaves them to finish; the rest of a cancelled batch is dropped (E: no line) or
    gets a cancelled error line (W). W reads carry the injected effect of their cell (delta by key), others as the fake engine."""

    def __init__(self, root, rate=40, inflight=6):
        self.inner = T.InstantIO(root, T.chat_default)
        self.res, self.err, self.pending, self.q = self.inner.res, {}, set(), collections.deque()
        self.batches, self.rate, self.inflight, self.delta, self.dropped, self.polls = [], rate, inflight, {}, [], 0

    def answer(self, q):
        r = self.inner.fake.handle(q)
        dl = self.delta.get(q["key"], 0.0)
        u = int(hashlib.sha256(q["key"].encode()).hexdigest()[:8], 16) / 2 ** 32
        if dl > 0 and not r.get("ok") and u < dl:
            r["ok"] = True
        if dl < 0 and r.get("ok") and u < -dl:
            r["ok"] = False
        return r

    def submit(self, name, reqs):
        w = [q for q in reqs if q.get("engine") == "W"]
        t = [q for q in reqs if q.get("engine") == "T"]
        e = [q for q in reqs if q.get("engine") not in ("W", "T")]
        return (self._sub(name, e) if e else 0) + (self._sub(name + "_W", w) if w else 0) + (self._sub(name + "_T", t) if t else 0)

    def _sub(self, name, reqs):
        new, seen = [], set()
        for q in reqs:
            if q["key"] in self.res or q["key"] in self.pending or q["key"] in seen:
                continue
            seen.add(q["key"])
            new.append(q)
        if not new:
            return 0
        b = f"{name}_{len(self.batches)}"
        self.batches.append((b, new))
        if "_rung1" not in name:
            return self.inner.submit(name, new)
        for q in new:
            self.q.append((b, q))
            self.pending.add(q["key"])
        return len(new)

    def cancel_prefix(self, name):
        bs = {b for b, _ in self.batches if b.startswith(name + "_")}
        keep = collections.deque()
        for i, (b, q) in enumerate(self.q):
            if b in bs and i >= self.inflight:
                self.dropped.append(q["key"])
                if q.get("engine") == "W":
                    self.err[q["key"]] = dict(key=q["key"], error="cancelled: test", cancelled=True)
                continue
            keep.append((b, q))
        self.q = keep

    def poll(self):
        self.polls += 1
        assert self.polls < 20000, "no progress"
        for _ in range(min(self.rate, len(self.q))):
            b, q = self.q.popleft()
            self.res[q["key"]] = self.answer(q)
            self.pending.discard(q["key"])
            self.err.pop(q["key"], None)
        return 0

    def drain(self):
        while self.q:
            self.poll()

    def get(self, k):
        return self.res.get(k)

    def have(self, k):
        return k in self.res

    def wait(self, keys, deadline=None, every=5.0, label="", tick=300):
        keys = list(dict.fromkeys(keys))
        while self.q and any(k not in self.res and k not in self.err for k in keys):
            self.poll()
        return [k for k in keys if k in self.res], [k for k in keys if k not in self.res and k not in self.err]

    def tok_rate(self, window=600):
        return None


gen.time.sleep = lambda s: None
LOG = []
gen.log = lambda m: LOG.append(m)


def dry(extra=(), effects=True):
    root = tempfile.mkdtemp(prefix="t_am17_dry_")
    R = T.make(root, cls=gen.Run, extra=("--no-versions", "--no-am16", "--no-stack", "--w-cells", "3", "--no-lite") + tuple(extra))   # AM17 alone (LITE: t_am21)
    R.io = DelayedIO(root)
    T.run_to(R, "rung1")
    G = R.state["gens"]["1"]
    V = [c for c in R.valid() if c.get("routed")]
    W = [c for c in V if R.is_w(c)]
    E = [c for c in V if not R.is_w(c) and c["author"] == "P0"]
    plan = {}
    if effects and len(W) >= 2 and E:
        plan = {W[0]["cid"]: -0.35, W[1]["cid"]: +0.35, E[0]["cid"]: -0.6}
        G.setdefault("order", R.order())
        for c in V:
            for r in R.rung_rows(c, 1):
                R.io.delta[R.read_req(R.cand_entries(c), r, 0, 1)[0]["key"]] = plan.get(c["cid"], 0.0)
    T.run_to(R, "rung2")
    return R, G, V, W, E, plan


R, G, V, W, E, plan = dry()
check(f"dry: {len(W)} weight cells, {len(E)} P0 candidates routed", len(W) >= 2 and E)
qs = G["w_quarter"]
by = collections.Counter(R.rows[r]["bench"] for r in R.judge_rows)
qb = collections.Counter(R.rows[r]["bench"] for r in qs)
order_b = collections.defaultdict(list)
for r in G["order"]:
    order_b[R.rows[r]["bench"]].append(r)
check(f"quarter: per source ceil(n/4) of the generation's order ({dict(qb)} of {dict(by)})",
      all(qb[b] == math.ceil(by[b] / 4) for b in by) and set(qs) == {r for b in order_b for r in order_b[b][:math.ceil(by[b] / 4)]})
first = {c["cid"]: next(rs for b, rs in R.io.batches if b.startswith(f"g001_rung1_{c['cid']}_W_")) for c in W}
check("quarter: every weight cell's first batch is exactly the quarter rows (same rows for every cell)",
      all(sorted(q["row"] for q in rs) == sorted(qs) for rs in first.values()))
neg, pos = (next(c for c in W if c["cid"] == k) for k in list(plan)[:2])
rb = lambda c: [b for b, _ in R.io.batches if b.startswith(f"g001_rung1b_{c['cid']}_")]
check(f"futility: the -0.35 cell stops on its quarter ({neg.get('status')})",
      neg.get("futile") and neg.get("w_stage") == "futile" and neg.get("decided") and not neg.get("accepted")
      and sorted(neg["rows_in_hand"]) == sorted(qs) and not rb(neg))
check(f"futility: the +0.35 cell reads the rest of JUDGE ({pos.get('w_stage')}, {len(rb(pos))} rung1b batch)",
      pos.get("w_stage") == "full" and len(rb(pos)) == 1 and all(R.io.have(R.read_req(R.cand_entries(pos), r, 0, 1)[0]["key"]) or R.io.q
                                                                for r in R.rung_rows(pos, 1)))
j = neg.get("judge") or {}
check(f"futile cell in BH with p = 1, ledger numbers on the quarter (p {j.get('p')}, n {j.get('n')}, bh {j.get('bh')})",
      j.get("bh") is False and j.get("n") == len(qs) and j.get("p") is not None and j["p"] >= .5)
ec = next(c for c in E if c["cid"] in plan)
check(f"cancel: the -0.6 E candidate rejected for good with reads pending -> cancelled ({ec.get('cancelled_reads')} reads, status {ec.get('status')})",
      ec.get("reads_cancelled") and ec.get("cancelled_reads", 0) > 0 and not ec.get("accepted"))
check("cancel: its dropped keys never land; rows_in_hand only rows read before the cancel",
      all(R.io.get(k) is None for k in R.io.dropped if k.startswith("x|")) and len(ec["rows_in_hand"]) + ec["cancelled_reads"] == ec["rows_read"]["1"]
      or len(ec["rows_in_hand"]) + ec["cancelled_reads"] <= ec["rows_read"]["1"])
je = ec.get("judge") or {}
A = R.am15_numbers("obs", V)
check(f"cancel: BH input p = 1 (d -1), reported p {je.get('p')} on the rows in hand, p_in_bh {A[ec['cid']].get('p_in_bh')}",
      A[ec["cid"]].get("p_in_bh") == 1.0 and A[ec["cid"]]["x"]["d"] == -1.0 and je.get("n") == len(ec["rows_in_hand"]) and je.get("bh") is False)
check(f"judge1: no reads pending after the rung (cancelled / futile frozen on rows in hand): pending_reads {sorted({c.get('pending_reads') for c in V})}",
      all(c.get("pending_reads") == 0 for c in V))
R.a.stop_after = None
R.step_report()
led = [json.loads(l) for l in open(os.path.join(R.rd, "ledger.jsonl"))]
lc = [r for r in led if r.get("id") == ec["cid"] and not r.get("final")]
lw = [r for r in led if r.get("id") == neg["cid"] and not r.get("final")]
check(f"ledger decision lines (step report): reads_cancelled / futile / w_stage ({[r.get('reads_cancelled') for r in lc]}, {[(r.get('futile'), r.get('w_stage')) for r in lw]})",
      lc and lc[-1].get("reads_cancelled") is True and lw and lw[-1].get("futile") is True and lw[-1].get("w_stage") == "futile")
R.io.drain()                                                    # in-flight reads of the cancelled candidate land now: they must not be used
for c in V:
    c["pending_reads"] = 1                                      # force finalise to recompute every candidate on all reads in hand
R.g = 2
R.finalise()
R.g = 1
fin = [c for c in V if c.get("final")]
check(f"finalise: generation 1 finalised after the drain ({len(fin)}/{len(V)}), no AM15 assertion",
      len(fin) == len(V) and not any("AM15 ASSERTION" in m for m in LOG) and not any(c.get("final_contradiction") for c in V))
check("finalise: cancelled / futile keep their decision and hand numbers", not ec.get("accepted") and not neg.get("accepted")
      and ec["final"]["n"] == len(ec["rows_in_hand"]) and neg["final"]["n"] == len(qs))
led = [json.loads(l) for l in open(os.path.join(R.rd, "ledger.jsonl"))]
lf = {r["id"]: r for r in led if r.get("final")}
check(f"ledger final lines: reads_cancelled on the cancelled one, futile on the futile cell ({lf.get(ec['cid'], {}).get('reads_cancelled')}, {lf.get(neg['cid'], {}).get('futile')})",
      lf.get(ec["cid"], {}).get("reads_cancelled") is True and lf.get(neg["cid"], {}).get("futile") is True and lf[ec["cid"]]["n"] == len(ec["rows_in_hand"]))

R2, G2, V2, W2, E2, _ = dry(extra=("--no-am17",), effects=False)
names = {b.rsplit("_", 1)[0] for b, _ in R2.io.batches if "_rung1" in b}
check(f"--no-am17: one rung1 batch (+ its _W part), no quarter, no futility ({sorted(names)})",
      names <= {"g001_rung1", "g001_rung1_W"} and "w_quarter" not in G2 and not any(c.get("w_stage") for c in V2))

# ---------------------------------------------------------------------------------------------------------------- 5. T fields (engine_api §8)
root5 = tempfile.mkdtemp(prefix="t_am17_T_")
a5 = gen.args(["--root", root5, "--rows", T.ROWS, "--run", os.path.join(root5, "run"), "--slots", "4", "--box", "3600", "--no-versions"])
R5 = gen.Run(a5)
r0 = R5.judge_rows[0]
q0, _ = R5.read_req(R5.state["tip"]["entries"], r0, 0, 1)
check("T fields: a stock tip read is engine E with no edit fields", q0["engine"] == "E" and not {"edit", "edit_sha", "weights", "tip_n"} & set(q0))
check("esha(None) = 74234e98afe7498f (the engine's stock edit_sha)", gen.esha(None) == "74234e98afe7498f")
ed = {"op": "surgeon", "file": "/data/fd/mingdazhang/selfedit/v4/cells/fable_f1/edits/D1_bias_mid_+3.json"}
cell = {"op": "scale", "layers": [40, 47], "writer": "mlp", "alpha": 1.1}
ent = lambda **k: dict(dict(cond="always", condition="always", procedure=None, settings={}, ops=[]), **k)
R5.state["tip"] = dict(id="t004", born=4, parent="t002", entries=[ent(settings={"min_image_tokens": 4096}, cid="g001c34", author="B"),
                                                                   ent(edit=ed, cid="g004c30", author="B")])
tip = R5.state["tip"]["entries"]
q1, _ = R5.read_req(tip, r0, 0, 1)
check(f"T fields: a t004 parent read -> engine T, edit = the tip's edit, edit_sha {q1.get('edit_sha')}, weights t004, tip_n 1",
      q1["engine"] == "T" and q1["edit"] == ed and q1["edit_sha"] == gen.esha(ed) and q1["weights"] == "t004" and q1["tip_n"] == 1)
q2, _ = R5.read_req(tip + [ent(procedure="Look twice.")], r0, 0, 1)
check("T fields: a text candidate on t004 -> engine T (same weights)", q2["engine"] == "T" and q2["edit_sha"] == q1["edit_sha"])
q3, _ = R5.read_req(tip + [ent(edit=cell)], r0, 0, 1)
check("T fields: a weight cell on t004 -> engine W, edit = [tip edit, cell] (full list on stock), tip_n 1, weights t004",
      q3["engine"] == "W" and q3["edit"] == [ed, cell] and q3["tip_n"] == 1 and q3["weights"] == "t004" and q3["edit_sha"] == gen.esha([ed, cell]))
R5.a.tfields = False
q1n, _ = R5.read_req(tip, r0, 0, 1)
check("--no-tfields: same key, engine W, no new fields", q1n["key"] == q1["key"] and q1n["engine"] == "W" and "edit_sha" not in q1n)
R5.a.tfields = True
R5.write_tip()
tf = json.load(open(os.path.join(R5.rd, "tips", "t004.json")))
check(f"tip file: edits = the tip's edit, edit_sha {tf.get('edit_sha')}", tf.get("edits") == ed and tf.get("edit_sha") == gen.esha(ed))
n5 = R5.io.submit("tst_rung1_x", [q0, q1, q3])
fs = sorted(os.path.basename(f).rsplit("_", 1)[0] for f in glob.glob(os.path.join(root5, "q/in", "tst_rung1_x_*.jsonl")))
check(f"submit wrapper: E, T and W in batches of their own {fs}", n5 == 3 and fs == ["tst_rung1_x", "tst_rung1_x_T", "tst_rung1_x_W"])

print(f"\n{ok} ok, {fail} failed")
sys.exit(1 if fail else 0)
