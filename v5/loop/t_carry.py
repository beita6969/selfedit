"""AM21 LITE carry-in checks (coordinator 11-10, automatic, one time at the first LITE generation): a generation-1 candidate with p <= alpha
(not accepted) and one with reads missing at its decision and d > 0 are carried (samples 0-3 on their routed rows, submitted at generation 2's
start), one with p > alpha and nothing missing is not; a family re-judged by hand is left out; the birth family is re-judged by BH over all
its read candidates with the ones not carried at p = 1; the accepted carry-in competes at generation 2's tip step and merges; the ledger
keeps the birth generation; generation 3 runs no carry-in again.
usage: python3 t_carry.py"""
import collections, hashlib, json, os, sys, tempfile, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(os.path.dirname(HERE), "harness")]
os.chdir(HERE)
import gen  # noqa: E402
import t_versions as T  # noqa: E402

ok = fail = 0


def check(name, cond):
    global ok, fail
    print(("ok   " if cond else "FAIL ") + name, flush=True)
    ok, fail = ok + bool(cond), fail + (not cond)


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
        if "_rung1" not in name and "_carry_" not in name:        # carry-in reads are answered like rung reads (delayed, with the effect)
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

root = tempfile.mkdtemp(prefix="t_carry_")
R = T.make(root, cls=gen.Run, slots=10, extra=("--no-versions", "--no-am16", "--no-stack", "--no-w", "--no-ecomp", "--no-lite"))
R.io = DelayedIO(root)
T.run_to(R, "tip")                                               # generation 1 under the old rule, stopped before its tip step
G1 = R.state["gens"]["1"]
check("generation 1 created without LITE", G1.get("am21") is None)
V1 = [c for c in G1["cands"] if c.get("rung") and c.get("entry")]
P0 = sorted([c for c in V1 if c["author"] == "P0" and len(R.rung_rows(c, 1)) >= 20 and not c.get("reads_cancelled") and not c.get("futile")],
            key=lambda c: c["cid"])
B = [c for c in V1 if c["author"] == "B"]
assert len(P0) >= 3 and B, (len(P0), len(B))
P, Q, Z = P0[:3]
for c in V1:                                                     # control the step-1 numbers: nothing accepted, P p .05, Q d > 0 with reads
    c["accepted"] = False                                        # missing at its decision, every other p .6 with nothing missing
    c.pop("final", None)
    c.setdefault("judge", {}).update(p=0.6, D=-0.01)
    c.setdefault("decision", {})["pend"] = 0
P["judge"].update(p=0.05, D=0.04)
Q["judge"].update(p=0.30, D=0.02)
Q["decision"]["pend"] = 7
B[0]["carry_in"] = dict(by="hand (test)", contribution_lb=-0.1)  # family B|g1 re-judged by hand
G1["done"] += ["tip", "report", "l1"]
G1["t_end"] = time.time()
R.save()
for r in R.rung_rows(P, 1):                                      # P has a real effect on its new samples
    for s in range(1, gen.LITE_M):
        R.io.delta[R.read_req(R.cand_entries(P), r, s, 1)[0]["key"]] = 0.8
R.a.lite = True
T.run_to(R, "propose", g=2)
run = R.state.get("carry_in_run") or {}
check(f"carry_select at generation 2: picked {run.get('picked')} = [P, Q], families {run.get('families')}, skipped {run.get('skipped')}",
      sorted(run.get("picked", [])) == sorted([P["cid"], Q["cid"]]) and run.get("families") == {"P0|g1": [P["cid"], Q["cid"]]} and "B|g1" in run.get("skipped", [])
      and Z["cid"] not in run.get("picked", []))
bt = {b.rsplit("_", 1)[0]: rs for b, rs in R.io.batches if "_carry_" in b}
nP = len(R.rung_rows(P, 1))
check(f"carry reads: samples 1-3 new on P's {nP} rows (sample 0 cached) ({len(bt.get('g002_carry_' + P['cid'], []))} reads)",
      sorted((q["row"], q["sample"]) for q in bt.get("g002_carry_" + P["cid"], [])) == sorted((r, s) for r in R.rung_rows(P, 1) for s in (1, 2, 3)))
check(f"why: P '{P['carry']['why']}', Q '{Q['carry']['why']}'", P["carry"]["why"] == "p <= alpha" and Q["carry"]["why"].startswith("d > 0"))
sl = R.slots_am16()
check(f"slots_am16 runs with the carry-in cost in its fixed costs ({sl['am16']['fixed_tok']})", "carry" in sl["am16"]["fixed_tok"])
T.run_to(R, "report", g=2)
K = len([c for c in V1 if c["author"] == "P0"])
check(f"P re-judged: BH over all {K} P0|g1 candidates, not carried p = 1 ({P['carry'].get('K')}, p {P['carry'].get('p')}, {P['carry'].get('mode')}) -> accepted",
      P["carry"].get("decided") and P["carry"]["K"] == K and P["carry"]["accepted"] and P.get("accepted") and P["carry"]["mode"] == "all reads")
check(f"Q re-judged on pooled samples ({Q['carry'].get('p')}, accepted {Q['carry'].get('accepted')})", Q["carry"].get("decided") and Q["carry"]["K"] == K)
G2 = R.state["gens"]["2"]
acc2 = [c for c in G2["cands"] if c.get("accepted")]
tip = R.state["tip"]
if not acc2:
    check(f"tip step: the accepted carry-in merges (tip {tip['id']} from {tip.get('from_cand')})", tip.get("from_cand") == P["cid"] and P.get("merged")
          and (G2.get("tip_change") or {}).get("cand") == P["cid"] and tip["entries"][-1].get("cid") == P["cid"])
else:
    best = max(acc2 + [P], key=lambda c: R.cl_of(c))
    check(f"tip step: the larger contribution_lb of generation 2's accepted and the carry-in merges ({tip.get('from_cand')} = {best['cid']})",
          tip.get("from_cand") == best["cid"])
led = [json.loads(l) for l in open(os.path.join(R.rd, "ledger.jsonl"))]
lc = [l for l in led if l.get("carry") and l["id"] == P["cid"]]
check("ledger: carry-in line keeps the birth generation and says 're-judged under AM21 LITE carry-in'",
      lc and lc[-1]["gen"] == 1 and "re-judged under AM21 LITE carry-in in g2 (birth g1)" in lc[-1]["status"] and lc[-1]["carry_numbers"]["K"] == K)
check("cl_of uses the pooled carry-in numbers", abs(R.cl_of(P) - P["carry"]["contribution_lb"]) < 1e-12)
T.run_to(R, "propose", g=3)
check("generation 3: no second carry-in", R.state["carry_in_run"]["g"] == 2 and not any("g003_carry_" in b for b, _ in R.io.batches))

root2 = tempfile.mkdtemp(prefix="t_carry_off_")
R2 = T.make(root2, cls=gen.Run, extra=("--no-versions", "--no-am16", "--no-stack", "--no-w", "--no-ecomp", "--no-carry"))
R2.io = DelayedIO(root2)
T.run_to(R2, "propose")
check("--no-carry: no carry-in run", not R2.state.get("carry_in_run"))

# ---------------------------------------------------------------------------------------------------------------- box_extend hook
root3 = tempfile.mkdtemp(prefix="t_box_")
R3 = T.make(root3, cls=gen.Run, extra=("--no-versions", "--no-am16", "--no-stack", "--no-w", "--no-ecomp"))
CLOSE = dict(on=False)
R3.io = DelayedIO(root3, rate=8, on_poll=lambda io: setattr(R3, "deadline", 0.0) if CLOSE["on"] else None)
T.run_to(R3, "rung1")
json.dump(dict(g=7, add_s=5), open(os.path.join(R3.rd, "box_extend.json"), "w"))
R3.deadline = 123.0
R3.box_extend()
check("box_extend: a file for another generation changes nothing", R3.deadline == 123.0)
json.dump(dict(g=1, add_s=1e6), open(os.path.join(R3.rd, "box_extend.json"), "w"))
CLOSE["on"] = True                                               # every poll closes the box; the hook reopens it on every wait cycle
n0 = len(LOG)
T.run_to(R3, "rung2")
V3 = [c for c in R3.valid() if c.get("routed")]
check(f"box_extend: the read step runs to its decisions although the box was closed on every poll (extended {R3.G().get('box_extended_s')} s)",
      R3.G().get("box_extended_s") == 1e6 and not any("box end" in (c.get("decision") or {}).get("basis", "") for c in V3)
      and any("box extended by" in m for m in LOG[n0:]) and all(c.get("decided") for c in V3))

print(f"\n{ok} ok, {fail} failed")
sys.exit(1 if fail else 0)
