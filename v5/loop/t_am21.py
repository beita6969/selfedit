"""AM21 LITE checks (coordinator 11-10 22:45 CST): one extra read step, decision on the pooled 4 samples.
1. judge_bh-level simulation: step 1 = sample 0 on every routed row; d <= 0 after step 1 -> left (p = 1); d > 0 -> samples 1-3; interval
   decisions while samples arrive (unread samples of the 4 counted wrong / right, left = p 1); the final judgement on the pooled 4 samples
   (BH over the whole family, left = 1) never contradicts a decision.
2. A dry generation (gen.Run, fake engine answering with a delay, one strong positive and one strong negative P0 candidate): the negative
   one leaves after step 1 (futile, p = 1, no step-2 batch); the positive one gets samples 1-3 on all its routed rows in one batch and is
   accepted on the pooled samples; pooled d equals the hand computation; pess / opt / obs count an unread sample wrong / right / not at all;
   finalise after the drain gives no AM15 assertion; ledger lines carry am21 (step-1 line + pooled numbers); lite_factor is measured.
3. The box end: undecided candidates are decided under the pessimistic completion; BH-insignificant ones with reads pending are cancelled
   with a per-row sample mask, and their numbers do not change when the in-flight reads land.
4. The rule is fixed when the generation is created (G["am21"]); --no-lite generations keep one sample.
usage: python3 t_am21.py"""
import collections, hashlib, json, math, os, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(os.path.dirname(HERE), "harness")]
os.chdir(HERE)
import numpy as np  # noqa: E402
import judge_bh as BH  # noqa: E402
import gen  # noqa: E402
import t_versions as T  # noqa: E402

ok = fail = 0


def check(name, cond):
    global ok, fail
    print(("ok   " if cond else "FAIL ") + name, flush=True)
    ok, fail = ok + bool(cond), fail + (not cond)


# ---------------------------------------------------------------------------------------------------------------- 1. simulation
def sim(trials=300, seed=11, M=4):
    rng = np.random.default_rng(seed)
    bad = decided = left = accepted = 0
    for _ in range(trials):
        K = int(rng.integers(3, 14))
        S = 3
        X, ref, src_of = [], [], []
        glob_ = rng.random(K) < .5
        for j in range(K):
            n = int(rng.integers(20, 120))
            p = rng.beta(.35, .22, n)
            mu = rng.choice([0.0, 0.0, 0.0, .08, .15, -.1])
            q = np.clip(p + mu * (1 - p if mu > 0 else p), 0, 1)
            X.append((rng.random((n, M)) < q[:, None]).astype(float))
            ref.append((rng.random((n, 8)) < p[:, None]).mean(1))
            src_of.append(rng.integers(0, S, n))
        got = [np.zeros_like(x, bool) for x in X]
        nreq = [1] * K
        gone, dec = set(), {}

        def x_of(j, mode):
            n = len(ref[j])
            if j in gone:
                return dict(name=str(j), d=-1.0, se=0.0, n=n, nu=float(n - 1), share=1.0, is_global=bool(glob_[j]), lobo=False)
            Mj = nreq[j] if mode == "obs" else M
            g = got[j][:, :Mj]
            if mode == "obs":
                have = g.sum(1) > 0
                D = (np.where(g, X[j][:, :Mj], 0).sum(1)[have] / g.sum(1)[have]) - ref[j][have]
                ss = src_of[j][have]
            else:
                fill = 0.0 if mode == "pess" else 1.0
                D = np.where(g, X[j][:, :Mj], fill).mean(1) - ref[j]
                ss = src_of[j]
            if len(D) < 2:
                return dict(name=str(j), d=0.0, se=1.0, n=1, nu=1.0, share=1.0, is_global=bool(glob_[j]), lobo=False)
            src = {s: (float(D[ss == s].sum()), int((ss == s).sum())) for s in range(S) if (ss == s).any()}
            return dict(name=str(j), d=float(D.mean()), se=float(D.std(ddof=1) / math.sqrt(len(D))), n=len(D), nu=float(len(D) - 1), share=1.0,
                        is_global=bool(glob_[j]), lobo=BH.lobo_ok(src))

        ev = [(j, i, 0) for j in range(K) for i in range(len(ref[j]))]
        rng.shuffle(ev)
        t = 0
        while ev:
            j, i, s = ev.pop(0)
            if j in gone:
                continue
            got[j][i, s] = True
            t += 1
            if nreq[j] == 1 and got[j][:, 0].all() and j not in dec:          # step 1 complete: leave or step 2
                if x_of(j, "obs")["d"] <= 0:
                    gone.add(j)
                    dec[j] = "reject"
                    left += 1
                else:
                    nreq[j] = M
                    new = [(j, i2, s2) for i2 in range(len(ref[j])) for s2 in range(1, M)]
                    rng.shuffle(new)
                    ev += new
            if t % 25 and ev:
                continue
            und = [w for w in range(K) if w not in dec]
            if not und:
                continue
            P, O = [x_of(w, "pess") for w in range(K)], [x_of(w, "opt") for w in range(K)]
            d_ = BH.interval(P, O)
            for w in und:
                if d_[str(w)] != "wait":
                    dec[w] = d_[str(w)]
                    decided += 1
        F = {r["name"]: r["accept"] for r in BH.judge([x_of(w, "obs") for w in range(K)])}
        bad += sum((dec.get(w) == "accept") != F[str(w)] for w in range(K) if w in dec)
        accepted += sum(F.values())
    return bad, decided, left, accepted


b, de, le, ac = sim()
check(f"simulation: futility-only leaving after step 1 (p = 1) + interval decisions on the 4 samples -> {b} contradictions in {de} decisions "
      f"({le} left, {ac} accepted at the end)", b == 0 and de > 200 and le > 50 and ac > 10)


# ---------------------------------------------------------------------------------------------------------------- 2. dry generation
class DelayedIO:
    """As t_am17: InstantIO for everything but rung-1 reads (incl. the LITE step-2 batches, name *_rung1s2_*); those are answered RATE per
    poll in arrival order; the first INFLIGHT queued requests are in flight (a cancel leaves them to finish). Injected effect per key."""

    def __init__(self, root, rate=40, inflight=6, on_poll=None):
        self.inner = T.InstantIO(root, T.chat_default)
        self.res, self.err, self.pending, self.q = self.inner.res, {}, set(), collections.deque()
        self.batches, self.rate, self.inflight, self.delta, self.dropped, self.polls = [], rate, inflight, {}, [], 0
        self.on_poll = on_poll

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
        return self._sub(name, [q for q in reqs if q.get("engine") not in ("W", "T")])

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
                continue
            keep.append((b, q))
        self.q = keep

    def poll(self):
        self.polls += 1
        assert self.polls < 40000, "no progress"
        if self.on_poll:
            self.on_poll(self)
        for _ in range(min(self.rate, len(self.q))):
            b, q = self.q.popleft()
            self.res[q["key"]] = self.answer(q)
            self.pending.discard(q["key"])
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


def dry(extra=(), on_poll=None, rate=40):
    root = tempfile.mkdtemp(prefix="t_am21_dry_")
    R = T.make(root, cls=gen.Run, extra=("--no-versions", "--no-am16", "--no-stack", "--no-w", "--no-ecomp") + tuple(extra))
    R.io = DelayedIO(root, rate=rate, on_poll=on_poll)
    T.run_to(R, "rung1")
    G = R.state["gens"]["1"]
    V = [c for c in R.valid() if c.get("routed")]
    G.setdefault("order", R.order())
    E = sorted([c for c in V if c["author"] == "P0" and len(R.rung_rows(c, 1)) >= 20], key=lambda c: -len(R.rung_rows(c, 1)))
    plan = {}
    if len(E) >= 2:
        plan = {E[0]["cid"]: +0.7, E[1]["cid"]: -0.7}
        G.setdefault("order", R.order())
        for c in V:
            for r in R.rung_rows(c, 1):
                for s in range(gen.LITE_M):
                    R.io.delta[R.read_req(R.cand_entries(c), r, s, 1)[0]["key"]] = plan.get(c["cid"], 0.0)
    return R, G, V, E, plan


R, G, V, E, plan = dry()
check(f"dry: G['am21'] = 'lite' at creation, lite_on, {len(E)} P0 candidates with >= 20 routed rows", G.get("am21") == "lite" and R.lite_on() and len(E) >= 2)
T.run_to(R, "rung2")
pos, neg = (next(c for c in V if c["cid"] == k) for k in plan)
s2 = lambda c: [(b, rs) for b, rs in R.io.batches if b.startswith(f"g001_rung1s2_{c['cid']}_")]
check(f"negative: left after step 1 ({neg.get('status')})", neg["am21"].get("left") and neg["am21"]["M"] == 1 and neg.get("futile")
      and neg.get("decided") and not neg.get("accepted") and not s2(neg) and neg["am21"]["step1"]["d"] <= 0)
A = R.am15_numbers("obs", V)
check(f"negative: p = 1 in its family's BH (p_in_bh {A[neg['cid']].get('p_in_bh')}), reported on its step-1 rows",
      A[neg["cid"]].get("p_in_bh") == 1.0 and A[neg["cid"]]["x"]["d"] == -1.0 and A[neg["cid"]]["pend"] == 0)
rows = R.rung_rows(pos, 1)
b2 = s2(pos)
check(f"positive: one step-2 batch with samples 1-3 on all its {len(rows)} rows ({[len(rs) for _, rs in b2]} reads)",
      len(b2) == 1 and sorted((q["row"], q["sample"]) for q in b2[0][1]) == sorted((r, s) for r in rows for s in (1, 2, 3))
      and pos["am21"]["n"] == 4 and pos["am21"]["step1"]["d"] > 0)
check(f"positive: accepted on the pooled samples ({pos.get('status')})", pos.get("accepted") and pos.get("decided"))
tip = R.state["tip"]["entries"]
D = collections.defaultdict(list)
for r in rows:
    xs = [R.io.get(R.read_req(R.cand_entries(pos), r, s, 1)[0]["key"]) for s in range(4)]
    ps = R.samples(tip, r)
    D[R.rows[r].get("pic") or r].append(sum(bool(x["ok"]) for x in xs) / 4 - sum(bool(p["ok"]) for p in ps) / len(ps))
dhand = sum(sum(v) / len(v) for v in D.values()) / len(D)      # per-cluster means, cluster mean weighted by rows: recompute as gen does
dd, w, n, tc, tp, pend = R.diffs(pos, [1], "obs")
dgen = sum(dd[k] * w[k] for k in w) / sum(w.values())
dhand_rows = sum(sum(v) for v in D.values()) / sum(len(v) for v in D.values())
check(f"positive: pooled d on 4 samples per row = the hand computation ({dgen:+.4f} vs {dhand_rows:+.4f}), {n} rows, 0 pending",
      abs(dgen - dhand_rows) < 1e-9 and n == len(rows) and pend == 0)
single = [r for r in rows if sum(1 for x in rows if (R.rows[x].get("pic") or x) == (R.rows[r].get("pic") or r)) == 1][0]
cl = R.rows[single].get("pic") or single
k3 = R.read_req(R.cand_entries(pos), single, 3, 1)[0]["key"]
x3 = R.io.res.pop(k3)
oks = [bool(R.io.get(R.read_req(R.cand_entries(pos), single, s, 1)[0]["key"])["ok"]) for s in range(3)]
ref = sum(bool(p["ok"]) for p in R.samples(tip, single)) / len(R.samples(tip, single))
got = {m: R.diffs(pos, [1], m) for m in ("obs", "pess", "opt")}
check(f"unread sample: obs uses the 3 in hand, pess counts it wrong, opt right ({[round(got[m][0][cl], 4) for m in got]}), pending 1 in pess / opt",
      abs(got["obs"][0][cl] - (sum(oks) / 3 - ref)) < 1e-9 and abs(got["pess"][0][cl] - (sum(oks) / 4 - ref)) < 1e-9
      and abs(got["opt"][0][cl] - ((sum(oks) + 1) / 4 - ref)) < 1e-9 and got["pess"][5] == 1 and got["opt"][5] == 1 and got["obs"][5] == 1)
R.io.res[k3] = x3
check(f"every read candidate either left after step 1 or holds samples 1-3 ({collections.Counter('left' if c['am21'].get('left') else c['am21']['n'] for c in V)})",
      all(c["am21"].get("left") or (c["am21"]["n"] == 4 and s2(c)) for c in V))
R.a.stop_after = None
R.step_report()
led = [json.loads(l) for l in open(os.path.join(R.rd, "ledger.jsonl"))]
lp = [r for r in led if r.get("id") == pos["cid"] and not r.get("final")]
check(f"ledger: decision line carries am21 with the step-1 line and the pooled numbers (p {lp[-1].get('am15', {}).get('p') if lp else None})",
      lp and lp[-1].get("am21", {}).get("step1", {}).get("d") is not None and lp[-1]["am21"]["n"] == 4 and lp[-1].get("D") is not None)
R.io.drain()
for c in V:
    c["pending_reads"] = 1
R.g = 2
R.finalise()
R.g = 1
fin = [c for c in V if c.get("final")]
check(f"finalise: all finalised ({len(fin)}/{len(V)}), no AM15 assertion, final lines carry am21",
      len(fin) == len(V) and not any("AM15 ASSERTION" in m for m in LOG) and not any(c.get("final_contradiction") for c in V))
led = [json.loads(l) for l in open(os.path.join(R.rd, "ledger.jsonl"))]
check("ledger final lines: am21 present, the left one is futile", all(r.get("am21") for r in led if r.get("final"))
      and any(r.get("final") and r["id"] == neg["cid"] and r.get("futile") for r in led))
rr = sum(int(c["rows_read"]["1"]) * int(c["am21"]["n"]) - int(c.get("cancelled_reads") or 0) for c in V)
nr = sum(int(c["rows_read"]["1"]) for c in V)
fac, src = R.lite_factor(1)
check(f"lite_factor: measured in g1 = {fac:.3f} ({src}); first time {gen.LITE_FIRST}", abs(fac - rr / nr) < 1e-9 and R.lite_factor(None)[0] == gen.LITE_FIRST)

# ---------------------------------------------------------------------------------------------------------------- 3. box end
BOX = dict(R=None, at=None)


def close_box(io):
    if BOX["R"] is not None and io.polls >= BOX["at"]:
        BOX["R"].deadline = 0.0


R3, G3, V3, E3, plan3 = dry(rate=8, on_poll=close_box)
BOX.update(R=R3, at=R3.io.polls + 60)
T.run_to(R3, "rung2")
boxed = [c for c in V3 if "box end" in (c.get("decision") or {}).get("basis", "")]
canc = [c for c in boxed if c.get("reads_cancelled")]
check(f"box end: {len(boxed)} candidates decided under the pessimistic completion, {len(canc)} with reads cancelled (hand masks)",
      boxed and all(c["am21"].get("box_at") is not None for c in boxed) and canc and all(isinstance(c.get("hand"), dict) for c in canc))
check("box end: an accept there is BH under the pessimistic completion", all(c["decision"].get("bh") for c in boxed if c.get("accepted")))
before = {c["cid"]: R3.am15_numbers("obs", V3)[c["cid"]] for c in canc}
BOX["R"] = None
R3.io.drain()
after = R3.am15_numbers("obs", V3)
check("box end: frozen numbers do not change when in-flight reads land", all(abs(before[k]["d"] - after[k]["d"]) < 1e-12 and before[k]["n_read"] == after[k]["n_read"]
                                                                          for k in before))
check("box end: cancelled step-2 batches got cancel calls", all(c.get("cancelled_reads", 0) > 0 for c in canc))

# ---------------------------------------------------------------------------------------------------------------- 4. rule fixed at creation
root4 = tempfile.mkdtemp(prefix="t_am21_off_")
R4 = T.make(root4, cls=gen.Run, extra=("--no-versions", "--no-am16", "--no-stack", "--no-w", "--no-lite"))
R4.io = DelayedIO(root4)
T.run_to(R4, "rung1")
G4 = R4.state["gens"]["1"]
R4.a.lite = True
check("a generation created with --no-lite stays off when the code turns LITE on (resume)", G4.get("am21") is None and not R4.lite_on())
T.run_to(R4, "rung2")
V4 = [c for c in R4.valid() if c.get("routed")]
check("--no-lite generation: one sample per row, no step-2 batches, no am21 fields",
      not any(c.get("am21") for c in V4) and not any("_rung1s2_" in b for b, _ in R4.io.batches))
R.a.lite = False
check("a LITE generation stays LITE when the code turns it off", R.lite_on())

print(f"\n{ok} ok, {fail} failed")
sys.exit(1 if fail else 0)
