"""RUN/cards.json + LITE with W cards (coordinator 11-11 01:00 CST: W cards and T servers back at a boundary without a restart).
1. cards: without RUN/cards.json the generation takes --no-w / --e-cards; with it {"w", "e_cards"} (either key); read once when the generation
   is created and fixed in G["cards"] (a later edit of the file changes only the next generation); an unreadable file -> the arguments.
   No --no-w / --e-cards use outside the cards helpers (source check).
2. Dry LITE generation started with --no-w and switched on by cards.json (the live path), AM16 sizing as live, BW = 3 weight cells:
   W cells proposed (author B, edit), their reads go to engine W / T with the edit, AM17 quarter first; a cell with d <= 0 on the quarter
   stops there (futility, p = 1); a cell with d > 0 reads the rest of JUDGE at sample 0, then LITE step 2 (samples 1-3 on all rows) or
   leaves at step 1; W cells are global (family <tip>|B|g1, LOBO); never bundle members; the engine record of a W read vs an E parent says
   mixed (flag only, coordinator 11-10 g004c30). The next generation with cards.json w false proposes no W cell.
3. The box closes while W reads are pending: no error, undecided W cells decided under the pessimistic completion, pending counted.
usage: python3 t_cardsw.py"""
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
    """As t_am23.DelayedIO, but W / T reads are answered too (server W0 / T0, so eng_type says offline / server): rung-1 batches answered
    RATE per poll in arrival order; effect per key (delta)."""

    def __init__(self, root, rate=40, inflight=6, on_poll=None):
        self.inner = T.InstantIO(root, T.chat_default)
        self.res, self.err, self.pending, self.q = self.inner.res, {}, set(), collections.deque()
        self.batches, self.rate, self.inflight, self.delta, self.polls, self.on_poll = [], rate, inflight, {}, 0, on_poll
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
        if self.on_poll:
            self.on_poll(self)
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

# ---------------------------------------------------------------------------------------------------------------- 1. cards
src = open(os.path.join(HERE, "gen.py")).read()
body = {m.group(1): m.group(2) for m in re.finditer(r"\n    def (\w+)\(self[^\n]*\n((?:(?:        |\n)[^\n]*\n)*)", src)}
outside = [n for n, b in body.items() if re.search(r"self\.a\.(no_w|e_cards)\b", b) and n not in ("read_cards", "w_on", "e_cards_n")]
check(f"no --no-w / --e-cards use outside read_cards / w_on / e_cards_n ({outside})", not outside and "read_cards" in body)

root = tempfile.mkdtemp(prefix="t_cards_")
R = T.make(root, cls=gen.Run, slots=4, extra=("--no-versions", "--no-ecomp", "--no-w", "--e-cards", "8"))


def new_gen(g):
    """As gen.Run.generation at creation: the generation exists, its cards are stamped."""
    R.g = g
    R.stamp_cards(R.state["gens"].setdefault(str(g), dict(t0=0.0, done=[], times={})))
    return R.G()


G1 = new_gen(1)
check(f"no cards.json: the arguments ({G1['cards']})", G1["cards"] == dict(w=False, e_cards=8, b_cells=None, T=3600.0, src="arguments") and not R.w_on() and R.e_cards_n() == 8)
cj = os.path.join(R.rd, "cards.json")
json.dump({"w": True, "e_cards": 6}, open(cj, "w"))
check("cards.json written during the generation: this generation keeps its cards", not R.w_on() and R.e_cards_n() == 8 and R.cards_src() == "--e-cards")
G2 = new_gen(2)
check(f"next generation created: cards.json ({G2['cards']}), logged", G2["cards"] == dict(w=True, e_cards=6, b_cells=None, T=3600.0, src="cards.json") and R.w_on()
      and R.e_cards_n() == 6 and R.cards_src() == "cards.json" and any("cards (cards.json): W on, 6 E cards" in m for m in LOG))
json.dump({"w": False}, open(cj, "w"))
check("fixed in G: the file edited again -> generation 2 unchanged", R.w_on() and R.e_cards_n() == 6)
G3 = new_gen(3)
check(f"one key only (w false): e_cards from the arguments ({G3['cards']})", G3["cards"] == dict(w=False, e_cards=8, b_cells=None, T=3600.0, src="cards.json"))
open(cj, "w").write("{not json")
G4 = new_gen(4)
check(f"unreadable cards.json: the arguments, logged ({G4['cards']})", G4["cards"] == dict(w=False, e_cards=8, b_cells=None, T=3600.0, src="arguments")
      and any("cards.json: JSONDecodeError" in m for m in LOG))
json.dump({"w": False, "e_cards": 2, "b_cells": 0, "T": 9000}, open(cj, "w"))
G6 = new_gen(6)
check(f"the lane ruling (b_cells 0, T 9000): fixed in G, box_T 9000, cap 0, logged ({G6['cards']})",
      G6["cards"] == dict(w=False, e_cards=2, b_cells=0, T=9000.0, src="cards.json") and R.box_T() == 9000.0 and R.b_cells_cap() == 0
      and any("arm B setting cells <= 0, box T 9000 s" in m for m in LOG))
json.dump({"T": -5}, open(cj, "w"))
G7 = new_gen(7)
check(f"a bad T: the arguments ({G7['cards']})", G7["cards"]["src"] == "arguments" and R.box_T() == 3600.0 and R.b_cells_cap() is None)
R.g = 1
check("generation 1 still reads its own cards", not R.w_on() and R.e_cards_n() == 8 and R.box_T() == 3600.0 and R.b_cells_cap() is None)
R.state["gens"]["5"] = dict(t0=0.0, done=[], times={})
R.g = 5
check("a generation without cards (created by older code): the arguments, until stamped", not R.w_on() and R.e_cards_n() == 8 and "cards" not in R.G())
os.makedirs(R.rd, exist_ok=True)
R.save()
st = json.load(open(os.path.join(R.rd, "state.json")))
check("G['cards'] persists in state.json (a restart inside the generation keeps it)", st["gens"]["2"]["cards"]["w"] is True)


# ---------------------------------------------------------------------------------------------------------------- 2. LITE + W dry
def dry(close_at=None, rate=40, effects=(0.9, -1.0, None)):
    root = tempfile.mkdtemp(prefix="t_cardsw_dry_")
    R = T.make(root, cls=gen.Run, slots=6, extra=("--no-versions", "--no-ecomp", "--no-w", "--e-cards", "8"))
    os.makedirs(R.rd, exist_ok=True)
    json.dump({"w": True, "e_cards": 6}, open(os.path.join(R.rd, "cards.json"), "w"))
    orig = R.slots
    R.slots = lambda: dict(orig(), BW=len(effects))                 # AM16 sizing measures BW from W results; the dry engine has none

    def on_poll(io):
        if close_at is not None and io.polls >= close_at and R.g == 1:
            R.deadline = 0.0
    R.io = WIO(root, rate=rate, on_poll=on_poll)
    T.run_to(R, "rung1")
    G = R.state["gens"]["1"]
    G.setdefault("order", R.order())
    V = [c for c in R.valid() if c.get("routed")]
    W = sorted([c for c in V if R.is_w(c)], key=lambda c: c["cid"])
    plan = {c["cid"]: effects[i] for i, c in enumerate(W)}
    for c in V:
        e = plan.get(c["cid"], -0.9) if R.is_w(c) else -0.9
        if e is None:
            continue
        for r in R.rung_rows(c, 1):
            for s in range(gen.LITE_M):
                R.io.delta[R.read_req(R.cand_entries(c), r, s, 1)[0]["key"]] = e
    return R, G, V, W, plan


R, G, V, W, plan = dry()
check(f"dry: generation 1 created with cards.json W on over --no-w, LITE ({G['cards']}, am21 {G.get('am21')})",
      G["cards"]["w"] and G["cards"]["src"] == "cards.json" and G.get("am21") == "lite")
check(f"W cells proposed: {len(W)} B candidates with an edit, global, routed on every JUDGE row",
      len(W) == 3 and all(c["author"] == "B" and R.is_global(c) and len(R.rung_rows(c, 1)) == R.NJ for c in W))
T.run_to(R, "rung2")
qs = set(G["w_quarter"])
wb = {c["cid"]: [(nm, rs) for nm, rs in R.io.batches if nm.startswith(f"g001_rung1_{c['cid']}_")] for c in W}
check("W reads: engine W (stock weights, tip without edits) with the edit, the first batch = the stratified quarter at sample 0",
      all(wb[c["cid"]] and all(q.get("engine") in ("W", "T") and q.get("edit") == c["entry"]["edit"] for _, rs in wb[c["cid"]] for q in rs)
          and {q["row"] for q in wb[c["cid"]][0][1]} == qs and {q["sample"] for q in wb[c["cid"]][0][1]} == {0} for c in W))
pos, neg = W[0], W[1]
check(f"negative cell: futility on the quarter (w_stage {neg.get('w_stage')}), not accepted, p = 1, no rest / step-2 batch",
      neg.get("w_stage") == "futile" and not neg.get("accepted") and neg.get("decided")
      and not any(nm.startswith((f"g001_rung1b_{neg['cid']}_", f"g001_rung1s2_{neg['cid']}_")) for nm, _ in R.io.batches))
rb = [rs for nm, rs in R.io.batches if nm.startswith(f"g001_rung1b_{pos['cid']}_")]
s2 = [rs for nm, rs in R.io.batches if nm.startswith(f"g001_rung1s2_{pos['cid']}_")]
check(f"positive cell: quarter passed -> rest of JUDGE at sample 0 ({sum(len(x) for x in rb)} reads), then LITE step 2 on all {R.NJ} rows",
      pos.get("w_stage") == "full" and rb and {q["row"] for x in rb for q in x} == set(R.judge_rows) - qs
      and s2 and sorted((q["row"], q["sample"]) for x in s2 for q in x) == sorted((r, s) for r in R.judge_rows for s in range(1, gen.LITE_M))
      and all(q.get("engine") in ("W", "T") for x in s2 for q in x) and pos["am21"]["n"] == gen.LITE_M)
A = R.am15_numbers("obs", [c for c in R.valid() if c.get("rung")])
tip0 = R.state["tips"][0]["id"] if R.state["tips"] else R.state["tip"]["id"]
check(f"positive cell judged as global: family {A[pos['cid']]['group']}, LOBO {A[pos['cid']]['lobo']}, decided {pos.get('decided')} "
      f"accepted {pos.get('accepted')}", A[pos["cid"]]["is_global"] and A[pos["cid"]]["group"].split("|")[-2:] == ["B", "g1"] and pos.get("decided")
      and pos.get("accepted"))
mid = W[2]
check(f"null cell: decided through quarter / step 1 / pooled 4 samples without error ({mid.get('w_stage')}, left "
      f"{(mid.get('am21') or {}).get('left')}, {mid.get('status', '')[:70]})", mid.get("decided"))
check("no W cell is AM23-eligible (never a bundle member)", not any(R.bundle_eligible(c) for c in W)
      and not any(c["cid"] in (x.get("members") or []) for c in W for x in G["cands"] if x.get("kind") == "bundle"))
check(f"engine record: W read vs E parent -> mixed flagged ({pos.get('engine')})",
      (pos.get("engine") or {}).get("mixed") is True and "offline" in (pos["engine"].get("read") or {}))
T.run_to(R, "l1")
check(f"generation 1 ran to its end with W cards (tip {R.state['tip']['id']}, merged {[c['cid'] for c in G['cands'] if c.get('merged')]})",
      "report" in (G.get("done") or []))
json.dump({"w": False}, open(os.path.join(R.rd, "cards.json"), "w"))
R.io.q.clear()
T.run_to(R, "rung1", g=2)
G2 = R.state["gens"]["2"]
check(f"generation 2 with cards.json w false: no W cell proposed ({G2['cards']})", G2["cards"]["w"] is False
      and not any(R.is_w(c) for c in G2["cands"]))

# ---------------------------------------------------------------------------------------------------------------- 3. box end with W pending
Rb, Gb, Vb, Wb, _ = dry(close_at=25, rate=8, effects=(0.9, 0.9, 0.5))
T.run_to(Rb, "rung2")
boxed = [c for c in Wb if (c.get("am21") or {}).get("box_at") is not None]
check(f"box closed with W reads pending: no error; W cells decided ({[c.get('decided') for c in Wb]}); box-decided {[c['cid'] for c in boxed]}",
      all(c.get("decided") for c in Wb) and any("box + grace closed" in m for m in LOG))
check("box-decided W cells: decision under the pessimistic completion with the pending reads counted (or nothing pending)",
      boxed and all("box end" in ((c.get("decision") or {}).get("basis") or "") and (c.get("decision") or {}).get("pend") is not None for c in boxed))

# ---------------------------------------------------------------------------------------------------------------- 4. the lane ruling
for cls, extra, what in ((gen.Run, ("--no-versions",), "stage 1"), (T.VRun, (), "Stage 2 (versions, live)")):
    root4 = tempfile.mkdtemp(prefix="t_cardsw_lane_")
    R4 = T.make(root4, cls=cls, slots=6, extra=extra + ("--no-ecomp", "--no-w", "--e-cards", "8"))
    os.makedirs(R4.rd, exist_ok=True)
    json.dump({"w": False, "e_cards": 2, "b_cells": 0, "T": 9000}, open(os.path.join(R4.rd, "cards.json"), "w"))
    T.run_to(R4, "route")
    G4_ = R4.state["gens"]["1"]
    check(f"{what}: cards.json b_cells 0 -> no arm-B setting cell; T 9000 -> deadline t0 + 9000 ({R4.deadline - G4_['t0']:.0f} s)",
          not any(c["author"] == "B" and not (c.get("entry") or {}).get("edit") for c in G4_["cands"])
          and abs(R4.deadline - G4_["t0"] - 9000) < 1e-6 and any(c["author"] != "B" for c in G4_["cands"]))
    if "l1_alloc" in G4_:
        check(f"{what}: b_cells 0 -> B is not in the author draw; all {G4_['l1_alloc']['n_total']} drawn slots go to proposers ({G4_['l1_alloc']['slots']})",
              "B" not in G4_["l1_alloc"]["slots"] and sum(G4_["l1_alloc"]["slots"].values()) == G4_["l1_alloc"]["n_total"])
    check(f"{what}: the live class keeps B outside the draw anyway (gen.py sets b_in_draw = False; t_versions patches it True for its tests)",
          re.search(r"\n    b_in_draw = False ", src) is not None)
    root5 = tempfile.mkdtemp(prefix="t_cardsw_lane_")
    R5 = T.make(root5, cls=cls, slots=6, extra=extra + ("--no-ecomp", "--no-w"))
    os.makedirs(R5.rd, exist_ok=True)
    json.dump({"b_cells": 1}, open(os.path.join(R5.rd, "cards.json"), "w"))
    T.run_to(R5, "route")
    nb5 = sum(1 for c in R5.state["gens"]["1"]["cands"] if c["author"] == "B" and not (c.get("entry") or {}).get("edit"))
    check(f"{what}: b_cells 1 -> at most one arm-B setting cell ({nb5})", nb5 == 1)

print(f"\n{ok} ok, {fail} failed")
sys.exit(1 if fail else 0)
