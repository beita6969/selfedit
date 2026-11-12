"""B2 author G (coordinator 11-11, Track B): signal-driven cells from RUN/signal/<tip>/cells.json.
1. Units on a fresh run: rank order kept; a kind other than weight / setting, a bad edit, a setting outside the arm-B ranges, no numeric
   pred -> skipped with the reason; a cells.json for another tip -> no props; no file -> no props; W off -> weight cells wait, setting cells
   are still proposed; RUN/cards.json g_cells / g_settings caps; a cell already read on this tip (any author) -> skipped; --no-g -> nothing;
   w_budget_after_g: G takes its weight cells first, arm B gets the rest of BW.
2. Dry LITE generation (live class, versions path; W switched on by cards.json; BW = 3): G proposes 2 weight + 1 setting cell, arm B 1 weight
   cell; G candidates valid, carry c["g"], global, routed on every JUDGE row; weight cells on engine W (AM17 quarter first), the setting cell
   on E; family <tip>|G|g1 with LOBO; never bundle members; after the report RUN/signal/t000/realised.jsonl holds one line per decided G
   candidate, no row ids; the L1 report lists G; stage 1 (gen.Run, --no-versions) proposes G too.
usage: python3 t_g.py"""
import collections, hashlib, json, os, re, sys, tempfile

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


class WIO:
    """As t_cardsw.WIO: W / T reads answered too (server W0 / T0); rung-1 batches answered RATE per poll; effect per key (delta)."""

    def __init__(self, root, rate=40, inflight=6):
        self.inner = T.InstantIO(root, T.chat_default)
        self.res, self.err, self.pending, self.q = self.inner.res, {}, set(), collections.deque()
        self.batches, self.rate, self.inflight, self.delta, self.polls = [], rate, inflight, {}, 0
        self.cancelled = []

    def answer(self, q):
        r = self.inner.fake.handle(q)
        if q.get("engine") in ("W", "T"):
            r["server"] = q["engine"] + "0"
        dl = self.delta.get(q["key"], 0.0)
        u = int(hashlib.sha256(q["key"].encode()).hexdigest()[:8], 16) / 2 ** 32
        if dl > 0 and not r.get("ok") and u < dl:
            r["ok"] = True
        if dl < 0 and r.get("ok") and u < -dl:
            r["ok"] = False
        return r

    def submit(self, name, reqs):
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
        self.cancelled.append(name)
        bs = {b for b, _ in self.batches if b.startswith(name + "_")}
        self.q = collections.deque((b, q) for i, (b, q) in enumerate(self.q) if not (b in bs and i >= self.inflight))

    def poll(self):
        self.polls += 1
        assert self.polls < 40000, "no progress"
        for _ in range(min(self.rate, len(self.q))):
            b, q = self.q.popleft()
            self.pending.discard(q["key"])
            self.res[q["key"]] = self.answer(q)
        return 0

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


def put(R, obj, tid="t000"):
    d = os.path.join(R.rd, "signal", tid)
    os.makedirs(d, exist_ok=True)
    json.dump(obj, open(os.path.join(d, "cells.json"), "w"))


def cards(R, obj):
    os.makedirs(R.rd, exist_ok=True)
    json.dump(obj, open(os.path.join(R.rd, "cards.json"), "w"))


def new_gen(R, g):
    R.g = g
    R.stamp_cards(R.state["gens"].setdefault(str(g), dict(t0=0.0, done=[], times={})))
    return R.G()


def cells_doc(menu, tip="t000"):
    return {"tip": tip, "scale": "accuracy per row, show+monitor rows", "cells": [
        {"kind": "weight", "name": menu[0][1], "edit": menu[0][2], "pred": 0.021, "se": 0.009, "pred_tokens_ratio": 1.02, "family": "fam-a"},
        {"kind": "setting", "name": "temperature 0.73", "settings": {"temperature": 0.73}, "pred": 0.012, "se": 0.007},
        {"kind": "probe", "name": "a kind not handled", "pred": 0.3},
        {"kind": "weight", "name": "bad op", "edit": {"op": "melt"}, "pred": 0.05, "se": 0.01},
        {"kind": "weight", "name": menu[1][1], "edit": menu[1][2], "pred": 0.006, "se": 0.010},
        {"kind": "setting", "name": "temperature 3", "settings": {"temperature": 3.0}, "pred": 0.02, "se": 0.01},
        {"kind": "weight", "name": "no pred", "edit": menu[2][2], "pred": None, "se": 0.01},
    ]}


# ---------------------------------------------------------------------------------------------------------------- 1. units
root = tempfile.mkdtemp(prefix="t_g_")
R = T.make(root, cls=gen.Run, slots=4, extra=("--no-versions", "--no-ecomp", "--no-w", "--e-cards", "8"))
menu = R.w_menu()
check(f"the W menu has cells to use ({len(menu)})", len(menu) >= 3)
cards(R, {"w": True, "e_cards": 6})
new_gen(R, 1)
cs, info = R.g_read_cells()
check(f"no cells.json: nothing ({info['state']})", cs == [] and info["state"] == "no cells.json")
put(R, cells_doc(menu))
cs, info = R.g_read_cells()
why = {s["rank"]: s["why"] for s in info["skipped"]}
check(f"cells read in rank order: {[(c['rank'], c['kind']) for c in cs]}", [c["rank"] for c in cs] == [0, 1, 4] and [c["kind"] for c in cs] == ["weight", "setting", "weight"])
check(f"skipped with reasons: {why}", set(why) == {2, 3, 5, 6} and "not handled" in why[2] and "op must be" in why[3] and "outside" in why[5]
      and "pred" in why[6] and info["sha"] and info["scale"].startswith("accuracy"))
props = []
sl = R.w_budget_after_g(props, {"B": 2, "P0": 4, "BW": 3})
gp = [p for p in props if p["author"] == "G"]
check(f"W on, BW 3: G takes its 2 weight cells first, arm B gets 1 ({sl['BW']}); setting cell too; slots 0.. ({[p['slot'] for p in gp]})",
      sl["BW"] == 1 and len(gp) == 3 and [p["slot"] for p in gp] == [0, 1, 2] and sum(1 for p in gp if p["cand"].get("edit")) == 2
      and R.G()["g"]["w_cells"] == 2 and R.G()["g"]["proposed"] == 3)
c0 = gp[0]["cand"]
check(f"a G prop carries its cell: edit, condition always, g meta ({sorted(c0['g'])})", c0["edit"] == menu[0][2] and c0["condition"] == "always"
      and c0["g"]["pred"] == 0.021 and c0["g"]["rank"] == 0 and c0["g"]["tip"] == "t000" and c0["g"]["family"] == "fam-a"
      and c0["g"]["cells_sha"] == info["sha"] and gp[2]["cand"]["settings"] == {"temperature": 0.73} and "edit" not in gp[2]["cand"])
check("G props logged", any("G (signal cells, read): 3 props (2 weight on W of budget 3; 1 setting)" in m for m in LOG))
props = []
sl = R.w_budget_after_g(props, {"B": 2, "P0": 4, "BW": 1})
check(f"BW 1: G takes 1, arm B none ({sl['BW']}); 1 weight waits", sl["BW"] == 0 and sum(1 for p in props if p["cand"].get("edit")) == 1
      and R.G()["g"]["waiting_for_w"] == 1)
cards(R, {"w": False, "e_cards": 6})
new_gen(R, 2)
props = []
sl = R.w_budget_after_g(props, {"B": 2, "P0": 4, "BW": 3})
check(f"W off: weight cells wait, the setting cell is proposed ({R.G()['g']['waiting_for_w']} waiting)", len(props) == 1 and not props[0]["cand"].get("edit")
      and R.G()["g"]["waiting_for_w"] == 2 and sl["BW"] == 3 and any("2 weight cells wait for W cards" in m for m in LOG))
cards(R, {"w": True, "g_cells": 1, "g_settings": 0})
new_gen(R, 3)
props = []
sl = R.w_budget_after_g(props, {"B": 2, "P0": 4, "BW": 3})
check(f"cards.json g_cells 1, g_settings 0: one weight cell, no setting cell, arm B gets 2 ({sl['BW']})", len(props) == 1 and props[0]["cand"].get("edit") == menu[0][2]
      and sl["BW"] == 2)
put(R, cells_doc(menu, tip="t001"))
cs, info = R.g_read_cells()
props = []
sl = R.w_budget_after_g(props, {"B": 2, "P0": 4, "BW": 3})
check(f"cells.json for another tip: no props ({info['state']})", cs == [] and "for tip t001" in info["state"] and props == [] and sl["BW"] == 3)
put(R, cells_doc(menu))
R.state["gens"]["3"]["cands"] = [dict(cid="g003c00", author="B", parent="t000", entry=dict(cond="always", condition="always", edit=menu[0][2], settings={})),
                                 dict(cid="g003c01", author="P0", parent="t000", entry=dict(cond="always", condition="always", settings={"temperature": 0.73}))]
cards(R, {"w": True})
new_gen(R, 4)
cs, info = R.g_read_cells()
why = {s["rank"]: s["why"] for s in info["skipped"]}
check(f"cells already read on this tip (any author) skipped: {[c['rank'] for c in cs]}", [c["rank"] for c in cs] == [4]
      and why.get(0) == "read on this tip already" and why.get(1) == "read on this tip already")
R.a.no_g = True
props = []
check("--no-g: nothing", R.w_budget_after_g(props, {"B": 2, "P0": 4, "BW": 3})["BW"] == 3 and props == [])
R.a.no_g = False
put(R, {"tip": "t000", "cells": "nonsense"})
try:
    cs, info = R.g_read_cells()
    good = cs == []
except Exception as ex:                                       # noqa: BLE001
    good, info = False, dict(state=f"raised {type(ex).__name__}")
check(f"a malformed cells list: no props, no error ({info.get('state')})", good)
open(os.path.join(R.rd, "signal", "t000", "cells.json"), "w").write("{oops")
cs, info = R.g_read_cells()
check(f"unreadable cells.json: no props ({info['state'][:40]})", cs == [] and info["state"].startswith("unreadable"))


Ru = T.make(tempfile.mkdtemp(prefix="t_g_real_"), cls=gen.Run, slots=4, extra=("--no-versions", "--no-ecomp", "--no-w"))
new_gen(Ru, 1)
gm = dict(rank=1, kind="setting", name="t 0.8", pred=0.01, se=0.005, tip="t000")
Ru.G()["cands"] = [dict(cid="g001c03", author="P1", parent="t000", decided=True, accepted=False, judge=dict(D=-0.004, se=0.01, p=0.6, n_rows=80),
                        am21=dict(n=4), decision=dict(basis="pooled 4 samples"), tok_cand=900, tok_parent=1000),
                   dict(cid="g001c04", author="P2", parent="t000"),
                   dict(cid="g001c10", author="G", parent="t000", g=gm, duplicate_of="g001c03", status="invalid: duplicate"),
                   dict(cid="g001c11", author="G", parent="t000", g=dict(gm, rank=2), duplicate_of="tip", status="invalid: duplicate"),
                   dict(cid="g001c12", author="G", parent="t000", g=dict(gm, rank=3), duplicate_of="g001c04", status="invalid: duplicate")]
Lu = {x["cid"]: x for x in Ru.g_realised()}
check(f"realised for a duplicate of a proposer candidate: duplicate_of / author, that candidate's numbers ({Lu.get('g001c10')})",
      Lu.get("g001c10", {}).get("duplicate_of") == "g001c03" and Lu["g001c10"]["duplicate_author"] == "P1" and Lu["g001c10"]["d"] == -0.004
      and Lu["g001c10"]["n_rows"] == 80 and abs(Lu["g001c10"]["tokens_ratio"] - 0.9) < 1e-9 and Lu["g001c10"]["pred"] == 0.01)
check("realised for a duplicate of the tip: duplicate_of tip, no measured numbers; a duplicate of an undecided candidate waits",
      Lu.get("g001c11", {}).get("duplicate_of") == "tip" and Lu["g001c11"]["d"] is None and "g001c12" not in Lu)
Ru.G()["cands"][1].update(decided=True, judge=dict(D=0.002, se=0.01, p=0.4, n_rows=50))
check("once that candidate is decided, the waiting line is written (once)", [x["cid"] for x in Ru.g_realised()] == ["g001c12"] and Ru.g_realised() == [])


# ---------------------------------------------------------------------------------------------------------------- 2. dry generation
def dry(cls, extra, stop="rung1"):
    root = tempfile.mkdtemp(prefix="t_g_dry_")
    R = T.make(root, cls=cls, slots=6, extra=extra + ("--no-ecomp", "--no-w", "--e-cards", "8"))
    cards(R, {"w": True, "e_cards": 6})
    menu = R.w_menu()
    doc = cells_doc(menu)
    doc["cells"].append(dict(doc["cells"][1], name="temperature 0.73 again", pred=0.011))
    put(R, doc)
    orig = R.slots
    R.slots = lambda: dict(orig(), BW=3)
    R.io = WIO(root)
    T.run_to(R, stop)
    return R, menu


R, menu = dry(T.VRun, ())
G = R.state["gens"]["1"]
V = R.valid()
GA = sorted([c for c in G["cands"] if c["author"] == "G"], key=lambda c: c["cid"])
GD = [c for c in GA if c.get("duplicate_of")]
GC = [c for c in GA if not c.get("duplicate_of")]
check(f"a G cell equal to an earlier candidate: invalid duplicate with duplicate_of ({[(c['cid'], c.get('duplicate_of'), c['status'][:30]) for c in GD]})",
      len(GD) == 1 and not GD[0]["valid0"] and "duplicate" in GD[0]["status"] and GD[0]["duplicate_of"] == next(c["cid"] for c in GC if not R.is_w(c)))
BW_ = [c for c in G["cands"] if c["author"] == "B" and R.is_w(c)]
check(f"dry (versions, live path): G candidates {[c['cid'] for c in GC]}, arm-B weight cells {len(BW_)}", len(GC) == 3 and len(BW_) == 1)
pb = [q for nm, rs in R.io.batches if nm.startswith("g001_parent_") for q in rs]
pbg = [q for nm, rs in R.io.batches if nm.startswith("g001_parentbg_") for q in rs]
r0 = R.role["show"][0]
check(f"parent step: waited-for reads at priority {gen.PARENT_PRIO} ({len(pb)}), background at 5 ({len(pbg)}); the read key does not hold the priority "
      "(coordinator 11-11 02:45, live from gen 9)", gen.PARENT_PRIO == 0 and pb and {q["priority"] for q in pb} == {0} and {q["priority"] for q in pbg} == {5}
      and R.read_req([], r0, 0, 0)[0]["key"] == R.read_req([], r0, 0, 2)[0]["key"])
check("G candidates valid, carry c['g'], global, routed on every JUDGE row",
      all(c in V and c.get("g") and R.is_global(c) and c.get("routed") and len(R.rung_rows(c, 1)) == R.NJ for c in GC))
GW = [c for c in GC if R.is_w(c)]
GS = [c for c in GC if not R.is_w(c)]
check(f"G weight cells target E.W ({[c.get('target') for c in GW]}), the setting cell E.H with its settings ({GS[0]['entry'].get('settings') if GS else None})",
      len(GW) == 2 and all(c["target"] == "E.W" for c in GW) and len(GS) == 1 and GS[0]["entry"]["settings"] == {"temperature": 0.73})
T.run_to(R, "rung2")
qs = set(G.get("w_quarter") or [])
first = {c["cid"]: next((rs for nm, rs in R.io.batches if nm.startswith(f"g001_rung1_{c['cid']}_")), []) for c in GC}
check("G weight reads: engine W with the edit, first batch = the AM17 quarter at sample 0; the setting cell reads on E",
      all(first[c["cid"]] and all(q.get("engine") in ("W", "T") and q.get("edit") == c["entry"]["edit"] for q in first[c["cid"]])
          and {q["row"] for q in first[c["cid"]]} == qs for c in GW)
      and all(q.get("engine") not in ("W", "T") and not q.get("edit") for q in first[GS[0]["cid"]]))
A = R.am15_numbers("obs", [c for c in R.valid() if c.get("rung")])
check(f"G judged in its own family ({sorted({A[c['cid']]['group'] for c in GC})}) with LOBO, K = 3",
      all(A[c["cid"]]["group"].split("|")[-2:] == ["G", "g1"] and A[c["cid"]]["is_global"] and A[c["cid"]]["K"] == 3 and A[c["cid"]].get("lobo") is not None for c in GC))
check("no G candidate is AM23-eligible (never a bundle member)", not any(R.bundle_eligible(c) for c in GC))
check("every G candidate decided", all(c.get("decided") for c in GC))
T.run_to(R, "never")                                      # the rest of generation 1: report (-> realised), l1
rp = os.path.join(R.rd, "signal", "t000", "realised.jsonl")
L = [json.loads(l) for l in open(rp)] if os.path.exists(rp) else []
check(f"realised.jsonl: one line per decided G candidate and one for the duplicate ({len(L)})", sorted(x["cid"] for x in L) == [c["cid"] for c in GA]
      and G.get("g_realised") == [c["cid"] for c in GA])
xd = next((x for x in L if x.get("duplicate_of")), {})
xo = next((x for x in L if x["cid"] == xd.get("duplicate_of")), {})
check(f"the duplicate's line: its own prediction, the measured numbers of the candidate it duplicates ({xd.get('duplicate_of')}, {xd.get('duplicate_author')})",
      xd and xd["pred"] == 0.011 and xd["name"] == "temperature 0.73 again" and xd["duplicate_author"] == "G"
      and all(xd[k] == xo[k] for k in ("d", "se", "p", "n_rows", "samples_per_row", "accepted", "basis")))
x0 = next((x for x in L if x["rank"] == 0), {})
check(f"realised line: predicted vs pooled ({ {k: x0.get(k) for k in ('pred', 'pred_se', 'd', 'se', 'p', 'n_rows', 'samples_per_row', 'tokens_ratio', 'basis')} })",
      x0.get("pred") == 0.021 and x0.get("pred_se") == 0.009 and isinstance(x0.get("d"), (int, float)) and isinstance(x0.get("p"), (int, float))
      and x0.get("n_rows") and x0.get("kind") == "weight" and x0.get("tip") == "t000" and x0.get("cells_sha"))
rowids = set(R.rows)
txt = open(rp).read() if os.path.exists(rp) else ""


def has_ids(o):
    if isinstance(o, dict):
        return any(has_ids(v) for v in o.values())
    if isinstance(o, list):
        return any(has_ids(v) for v in o)
    return isinstance(o, str) and o in rowids


check("realised.jsonl has no row ids and no per-row lists", L and not any(has_ids(x) for x in L) and not any(isinstance(v, list) for x in L for v in x.values())
      and "row" not in {k for x in L for k in x})
R.g_realised()
check("g_realised again: no duplicate lines", len(open(rp).read().splitlines()) == len(L))
rep = (G.get("l1") or {}).get("report") or {}
check(f"the L1 report lists G ({sorted((rep.get('authors') or {}))})", "G" in (rep.get("authors") or {}) and rep["authors"]["G"]["slots"] == 4)
check("proposer memory labels G candidates 'blind' (coordinator 11-11; the ledger keeps author G)", T.VRun._rel("G", {"P1"}) == "blind"
      and T.VRun._rel("B", {"P1"}) == "blind" and T.VRun._rel("P2", {"P1"}) == "another proposer" and all(c["author"] == "G" for c in GA))
check("G is not a version: no L1 decision about G", "G" not in ((G.get("l1") or {}).get("decide") or {}))

R1, _ = dry(gen.Run, ("--no-versions",), stop="route")
G1 = R1.state["gens"]["1"]
check(f"stage 1 (gen.Run, --no-versions): G proposes too ({sum(c['author'] == 'G' and c['valid0'] for c in G1['cands'])} valid G, "
      f"{sum(c['author'] == 'B' and R1.is_w(c) for c in G1['cands'])} arm-B weight)",
      sum(c["author"] == "G" and c["valid0"] for c in G1["cands"]) == 3 and sum(c["author"] == "B" and R1.is_w(c) for c in G1["cands"]) == 1)

print(f"\n{ok} ok, {fail} failed")
sys.exit(1 if fail else 0)
