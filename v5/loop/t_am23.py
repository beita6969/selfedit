"""AM23 checks (coordinator 11-10 23:30 CST, "portfolio acceptance") plus the 23:30 context-skip rule and the H1 merge path.
1. Dry LITE generation (gen.Run, fake engine with a delay): the bundle is planned at the read step, built once every eligible candidate's step 1
   is complete, members = the routed op-free proposer candidates with step-1 d > 0 (cid order), reads = samples 4-7 on the union of the
   members' routed JUDGE rows, none of them a member's key; not global (no LOBO); BH family system|g1; the system family is not decided while
   the bundle is planned; an accepted bundle with the largest contribution_lb merges all member entries (merged_through on the members).
2. Fewer than BUNDLE_MIN members -> no bundle; --no-bundle -> none.
3. Context errors: a final "maximum context length" error at a candidate's first sample makes the row unrouted (zero difference, not pending,
   not wrong), counted in skipped_ctx with the rows; > 1/4 of the routed rows -> mostly_unusable.
4. The "add" annotation keeps a synthetic multi-entry candidate in state["synthetic"]; carry_merges merges it at a boundary (all entries,
   members merged_through); a candidate whose entries are in the tip already is not merged again.
usage: python3 t_am23.py"""
import collections, hashlib, json, os, sys, tempfile

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
    """As t_am21: rung-1 batches (incl. step 2 and the bundle, *_rung1*) answered RATE per poll; effect per key (delta)."""

    def __init__(self, root, rate=40, inflight=6, on_poll=None):
        self.inner = T.InstantIO(root, T.chat_default)
        self.res, self.err, self.pending, self.q = self.inner.res, {}, set(), collections.deque()
        self.batches, self.rate, self.inflight, self.delta, self.polls, self.on_poll = [], rate, inflight, {}, 0, on_poll
        self.errkeys = {}

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
        new, seen = [], set()
        for q in reqs:
            if q.get("engine") in ("W", "T") or q["key"] in self.res or q["key"] in self.pending or q["key"] in seen:
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
        self.q = collections.deque((b, q) for i, (b, q) in enumerate(self.q) if not (b in bs and i >= self.inflight))

    def poll(self):
        self.polls += 1
        assert self.polls < 40000, "no progress"
        if self.on_poll:
            self.on_poll(self)
        for _ in range(min(self.rate, len(self.q))):
            b, q = self.q.popleft()
            self.pending.discard(q["key"])
            if q["key"] in self.errkeys:
                self.err[q["key"]] = dict(key=q["key"], error=self.errkeys[q["key"]])
                continue
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


def dry(extra=(), member_effect=0.9, bundle_effect=0.95, n_pos=3, ctx_rows=0, close_at=None, rate=40):
    root = tempfile.mkdtemp(prefix="t_am23_dry_")
    R = T.make(root, cls=gen.Run, slots=10, extra=("--no-versions", "--no-am16", "--no-w", "--no-ecomp") + tuple(extra))
    seen = dict(sys_decided_while_planned=False, bundle_keys=None)

    def on_poll(io):
        if close_at is not None and io.polls >= close_at and R.g == 1:
            R.deadline = 0.0
        G_ = R.state["gens"].get("1") or {}
        B = G_.get("bundle") or {}
        if B.get("state") == "planned" and any(c.get("decided") and c["author"] == "system" for c in G_.get("cands", [])):
            seen["sys_decided_while_planned"] = True
        if B.get("state") == "built" and seen["bundle_keys"] is None:
            b = next(c for c in G_["cands"] if c["cid"] == B["cid"])
            seen["bundle_keys"] = list(R.cand_keys(b))
            for k in seen["bundle_keys"]:
                io.delta[k] = bundle_effect
    R.io = DelayedIO(root, rate=rate, on_poll=on_poll)
    T.run_to(R, "rung1")
    G = R.state["gens"]["1"]
    G.setdefault("order", R.order())
    V = [c for c in R.valid() if c.get("routed")]
    el = sorted([c for c in V if R.bundle_eligible(c)], key=lambda c: -len(R.rung_rows(c, 1)))
    plan = {c["cid"]: (member_effect if i < n_pos else -1.0) for i, c in enumerate(el)}
    for c in V:
        for r in R.rung_rows(c, 1):
            for s in range(gen.LITE_M):
                R.io.delta[R.read_req(R.cand_entries(c), r, s, 1)[0]["key"]] = plan.get(c["cid"], -0.9)
    ctx = []
    if ctx_rows:
        c0 = el[0]
        for r in R.rung_rows(c0, 1)[:ctx_rows]:
            R.io.errkeys[R.read_req(R.cand_entries(c0), r, 0, 1)[0]["key"]] = ("RuntimeError: HTTP 400: Input length (58225) exceeds model's "
                                                                               "maximum context length (46000).")
            ctx.append(r)
    return R, G, V, el, plan, seen, ctx


# ---------------------------------------------------------------------------------------------------------------- 1. the bundle
R, G, V, el, plan, seen, _ = dry()
check(f"dry: LITE generation, {len(el)} eligible (routed, op-free, not global) proposer candidates", G.get("am21") == "lite" and len(el) >= 4)
T.run_to(R, "rung2")
B = G.get("bundle") or {}
b = next((c for c in G["cands"] if c.get("kind") == "bundle"), None)
check(f"bundle planned at the read step and built ({B.get('state')}, {b and b['cid']})", B.get("state") == "built" and b is not None and B.get("cid") == b["cid"])
mem = sorted(c["cid"] for c in el if (c.get("am21") or {}).get("step1", {}).get("d", 0) > 0 and not c["am21"].get("left"))
check(f"members = the eligible candidates with step-1 d > 0, in cid order ({b['members']} vs {mem})", b["members"] == mem and len(mem) >= gen.BUNDLE_MIN
      and [x["cid"] for x in b["entry"]["parts"]] == mem)
check("no member has step-1 d <= 0, no candidate with ops / global / B / system is a member",
      all(R.bundle_eligible(next(c for c in V if c["cid"] == i)) for i in b["members"]))
union = [r for r in R.judge_rows if any(R.fires(next(c for c in V if c["cid"] == i)["entry"], r) for i in b["members"])]
bb = [rs for nm, rs in R.io.batches if nm.startswith(f"g001_rung1_{b['cid']}_")]
got = sorted((q["row"], q["sample"]) for rs in bb for q in rs)
check(f"reads: one batch, samples 4-7 on the {len(union)} union rows ({len(got)} reads)",
      len(bb) == 1 and got == sorted((r, s) for r in union for s in range(4, 8)) and b["routed"] == len(union) and b["share"] == len(union) / R.NJ)
mk = {k for c in V if c["cid"] in b["members"] for k in R.cand_keys(c)}
check("every bundle key is new: none is a member's own key (step 1 or step 2)", not (set(R.cand_keys(b)) & mk) and len(R.cand_keys(b)) == 4 * len(union))
single = [r for r in union if sum(1 for i in b["members"] if R.fires(next(c for c in V if c["cid"] == i)["entry"], r)) == 1]
if single:
    r = single[0]
    m = next(c for c in V if c["cid"] in b["members"] and R.fires(c["entry"], r))
    check("on a row where one member holds alone the bundle's view is that member's (same signature), samples 4-7 are new",
          R.view(R.cand_entries(b), r)[2] == R.view(R.cand_entries(m), r)[2])
A = R.am15_numbers("obs", [c for c in R.valid() if c.get("rung")])
check(f"judged in system|g1 (group {A[b['cid']]['group']}, K {A[b['cid']]['K']}), not global, no LOBO", A[b["cid"]]["group"] == "system|g1"
      and not A[b["cid"]]["is_global"] and not R.is_global(b))
check("the system family was not decided while the bundle was planned", not seen["sys_decided_while_planned"])
check(f"bundle decided ({b.get('status', '')[:90]})", b.get("decided") and b.get("accepted"))
T.run_to(R, "l1")
t = R.state["tip"]
if t.get("from_cand") == b["cid"]:
    me = [x for x in t["entries"]]
    check(f"tip step: the bundle merged -> all {len(b['members'])} member entries are tip entries ({t['id']})",
          [x.get("cid") for x in me] == b["members"] and b.get("merged"))
    mm = [next(c for c in V if c["cid"] == i) for i in b["members"]]
    check("members: merged_through = the bundle, own merged flag False, status says 'merged through bundle'",
          all(c.get("merged_through") == b["cid"] and not c.get("merged") and "merged through bundle" in c["status"] for c in mm))
    tj = json.load(open(os.path.join(R.rd, "tip.json")))
    check("tip.json lists the member entries with their own cids", [e["from_cand"] for e in tj["entries"]] == b["members"])
    check("breeder: a tip made at this generation's tip step -> the realised gain is that of the tip it started from (t000)", R.breeder()["tip"] == "t000")
else:
    check(f"tip step: the bundle has the largest contribution_lb and merges (tip from {t.get('from_cand')})", False)
led = [json.loads(l) for l in open(os.path.join(R.rd, "ledger.jsonl"))]
lb = next((x for x in led if x["id"] == b["cid"] and not x.get("final")), {})
check("ledger: the bundle's line carries members, bundle info and its am21 (s0 = 4)", lb.get("members") == b["members"] and lb.get("bundle", {}).get("union") == len(union)
      and (lb.get("am21") or {}).get("s0") == 4)
R.io.q.clear()

# ---------------------------------------------------------------------------------------------------------------- 1b. member rule narrow
Rn, Gn, Vn, eln, _, _, _ = dry(n_pos=6)
json.dump({"members": "narrow"}, open(os.path.join(Rn.rd, "bundle_rule.json"), "w"))
T.run_to(Rn, "rung2")
bn = next((c for c in Gn["cands"] if c.get("kind") == "bundle"), None)
rs = sorted(c["routed"] for c in eln)
med = rs[len(rs) // 2] if len(rs) % 2 else (rs[len(rs) // 2 - 1] + rs[len(rs) // 2]) / 2
want = sorted(c["cid"] for c in eln if c["am21"].get("step1", {}).get("d", 0) > 0 and not c["am21"].get("left") and c["routed"] <= med)
check(f"rule narrow (bundle_rule.json): members = step-1 d > 0 with routed <= median {med} of the {len(eln)} eligible ({bn and bn['members']} vs {want})",
      (bn is not None and bn["members"] == want and Gn["bundle"].get("rule") == "narrow" and Gn["bundle"].get("rule_from") == "bundle_rule.json")
      or (bn is None and len(want) < gen.BUNDLE_MIN))
Rn.io.q.clear()

# ---------------------------------------------------------------------------------------------------------------- 2. too few members / off
R2, G2, V2, el2, _, _, _ = dry(n_pos=1)
T.run_to(R2, "rung2")
check(f"one member with step-1 d > 0 -> no bundle ({(G2.get('bundle') or {}).get('why')})", (G2.get("bundle") or {}).get("state") == "none"
      and not any(c.get("kind") == "bundle" for c in G2["cands"]))
R3, G3, _, _, _, _, _ = dry(extra=("--no-bundle",))
T.run_to(R3, "rung2")
check("--no-bundle -> none", (G3.get("bundle") or {}).get("state") == "none" and not any(c.get("kind") == "bundle" for c in G3["cands"]))

# ---------------------------------------------------------------------------------------------------------------- 3. context errors
R4, G4, V4, el4, _, _, ctx = dry(ctx_rows=12)
T.run_to(R4, "rung2")
c0 = el4[0]
dd, w, n, tc, tp, pend = R4.diffs(c0, [1], "obs")
ec = R4._ecomp[c0["cid"]]
check(f"ctx: the {len(ctx)} rows are skipped (counted, zero difference), not pending, listed ({len(ec['skipped_ctx_rows'])})",
      sorted(ec["skipped_ctx_rows"]) == sorted(ctx) and pend == 0 and n == len(R4.rung_rows(c0, 1)) - len(ctx))
check(f"ctx: skipped_ctx {c0.get('skipped_ctx')} on the candidate with the rows; mostly_unusable {c0.get('mostly_unusable')} "
      f"({len(ctx)} of {c0['routed']} routed rows)", c0.get("skipped_ctx") == len(ctx) and sorted(c0.get("skipped_ctx_rows")) == sorted(ctx)
      and c0.get("mostly_unusable") == (len(ctx) > 0.25 * c0["routed"]))
R4.io.q.clear()
R4.note_ctx(dict(c0, cid="x"))
c1 = dict(cid="y", routed=8)
R4._ecomp["y"] = dict(skipped_ctx_rows=["a", "b", "c"])
R4.note_ctx(c1)
check("ctx: 3 of 8 routed rows -> mostly unusable", c1["skipped_ctx"] == 3 and c1["mostly_unusable"])

# ---------------------------------------------------------------------------------------------------------------- 4. H1: add + carry_merges
R5 = T.make(tempfile.mkdtemp(prefix="t_am23_h1_"), cls=gen.Run, extra=("--no-versions", "--no-am16", "--no-w", "--no-ecomp"))
T.run_to(R5, "l1")
st = R5.state
tip0 = st["tip"]
G5 = st["gens"]["1"]
pc = [c for c in G5["cands"] if c.get("entry") and c["entry"].get("cond") not in ("always", None) and c["author"] != "system"
      and not c.get("merged") and c["entry"].get("cid")][:3]
H = dict(cid="H1", author="system", kind="bundle", parent=tip0["id"], members=[c["cid"] for c in pc],
         entry=dict(condition="bundle H1", cond="bundle", procedure=None, settings={}, ops=[], parts=[dict(c["entry"]) for c in pc], author="system"),
         accepted=True, carry_merge=True, carry_in=dict(contribution_lb=0.012, p=0.01, accept=True))
os.makedirs(os.path.join(R5.rd, "annotate", "add"), exist_ok=True)
json.dump(dict(gen=1, add=H, status_append="H1 accepted by hand (test)"), open(os.path.join(R5.rd, "annotate", "add", "H1_add.json"), "w"))
R5.save()
check("add annotation: H1 kept in state['synthetic'], not in any generation's cands, file applied",
      [x["cid"] for x in st.get("synthetic", [])] == ["H1"] and not any(c["cid"] == "H1" for G_ in st["gens"].values() for c in G_["cands"])
      and any(f.startswith("H1_add.json.applied") for f in os.listdir(os.path.join(R5.rd, "annotate", "add"))))
R5.g = 2
st["gens"].setdefault("2", dict(cands=[], t0=0))
R5.carry_merges()
t1 = st["tip"]
check(f"carry_merges: H1 merged at the boundary -> {t1['id']} = {tip0['id']} + its {len(pc)} entries",
      t1["id"] == "t002c" and [e.get("cid") for e in t1["entries"]] == [e.get("cid") for e in tip0["entries"]] + [c["cid"] for c in pc]
      and st["synthetic"][0].get("merged"))
check("carry_merges: H1's members are merged_through H1", all(c.get("merged_through") == "H1" for c in pc))
H2 = dict(H, cid="H2", parent=t1["id"], carry_in=dict(contribution_lb=0.02))
st["synthetic"].append(H2)
R5.g = 3
st["gens"].setdefault("3", dict(cands=[], t0=0))
R5.carry_merges()
check("a candidate whose entries are in the tip already is not merged again (tip unchanged)", st["tip"]["id"] == t1["id"] and not H2.get("merged")
      and "in the tip" in H2["status"])

# ---------------------------------------------------------------------------------------------------------------- 5. L1 credit split
cr = R.bundle_credit()
d1 = lambda m: ((m.get("am21") or {}).get("step1") or {}).get("d", 0.0)
mm = [next(c for c in V if c["cid"] == i) for i in b["members"]]
wt = {m["cid"]: m["share"] * max(d1(m), 0.0) for m in mm}
v_b = R._slot_value(b)[0]
want = {m["cid"]: v_b * wt[m["cid"]] / sum(wt.values()) for m in mm if not m.get("accepted") and wt[m["cid"]] > 0}
check(f"L1: bundle credit to members in proportion to share x step-1 d ({len(cr)} members credited, sum {sum(x[0] for x in cr.values()):+.4f} of "
      f"{v_b:+.4f})", set(cr) == set(want) and all(abs(cr[k][0] - want[k]) < 1e-12 for k in want))
au = mm[0]["author"]
mv, sv, na = R.slot_data(au, [1])
check(f"L1: slot_data counts the credited members of {au} as accepted slots ({na})", na >= sum(1 for m in mm if m["author"] == au and m["cid"] in cr))
cr5 = R5.bundle_credit()
check(f"L1: a hand bundle (synthetic H1) credits its members too ({sorted(cr5)})", set(cr5) <= set(H["members"]))

# ---------------------------------------------------------------------------------------------------------------- 6. LITE completion
def comp_case(change_tip):
    Rc, Gc, Vc, elc, _, _, _ = dry(extra=("--no-bundle",), member_effect=0.25, n_pos=7, close_at=30, rate=6)
    T.run_to(Rc, "l1")
    boxed = [c for c in Gc["cands"] if (c.get("am21") or {}).get("box_at") is not None and not c.get("accepted") and not c["am21"].get("left")
             and int((c.get("decision") or {}).get("pend") or 0) > 0 and ((c.get("judge") or {}).get("D") or 0) > 0 and not Rc.is_w(c)]
    tip_changed = Gc.get("tip_change") is not None
    if change_tip and not tip_changed:
        t = Rc.state["tip"]
        Rc.state["tips"].append(dict(t))
        Rc.state["tip"] = dict(t, id="t001x", entries=t["entries"] + [dict(condition="always", cond="always", procedure="x", settings={}, ops=[], cid="zz")])
    Rc.io.q.clear()
    T.run_to(Rc, "propose", g=2)
    return Rc, boxed, tip_changed


Rc, boxed, tch = comp_case(False)
run = (Rc.state.get("carry_runs") or {}).get("2") or {}
if not tch:
    check(f"LITE completion: picks the box-ended candidates of g1 with samples missing and d > 0 ({run.get('picked')} vs {sorted(c['cid'] for c in boxed)})",
          sorted(run.get("picked") or []) == sorted(c["cid"] for c in boxed) and boxed)
    pk = [c for c in Rc.state["gens"]["1"]["cands"] if c["cid"] in set(run.get("picked") or [])]
    bt = [rs for nm, rs in Rc.io.batches if "_lcomp_" in nm]
    allk = {k for c in pk for k in Rc.cand_keys(c)}
    sub = [q["key"] for x in bt for q in x]
    check(f"LITE completion: unfrozen (no hand mask / rows in hand), 4 samples, only missing reads submitted ({len(sub)} of {len(allk)} keys)",
          all("hand" not in c and "rows_in_hand" not in c and not c.get("reads_cancelled") and c["am21"]["n"] == 4 and c.get("completion") for c in pk)
          and bt and set(sub) <= allk and len(set(sub)) == len(sub) and len(sub) < len(allk))
    while Rc.io.q:
        Rc.io.poll()
    Rc.carry_decide(final=False)
    check("LITE completion: re-judged in the birth family on complete data (all reads), status says so",
          all((c.get("carry") or {}).get("decided") and c["carry"]["mode"] == "all reads" and "re-judged under LITE completion" in c["status"] for c in pk))
else:
    check("LITE completion: (the dry generation changed the tip) everything dropped", not run.get("picked"))
Rd, boxed_d, _ = comp_case(True)
run_d = (Rd.state.get("carry_runs") or {}).get("2") or {}
check(f"LITE completion: after a tip change the box-ended candidates are dropped ({run_d.get('dropped')})",
      not run_d.get("picked") and sorted(run_d.get("dropped") or []) == sorted(c["cid"] for c in boxed_d))

# ---------------------------------------------------------------------------------------------------------------- 7. build timing (11-11 01:50)
import time as _t  # noqa: E402


def timing_case():
    Rt, Gt, Vt, elt, _, _, _ = dry(extra=("--no-bundle",))
    for c in Vt:
        c.setdefault("am21", dict(M=gen.LITE_M, n=1))
        c["rung"] = 1
    Gt["bundle"] = dict(state="planned", eligible=[c["cid"] for c in elt], t=_t.time())
    return Rt, Gt, Vt, elt


def at(Rt, Gt, frac, span=1000.0):
    now = _t.time()
    Gt["rung1_t0"], Rt.deadline = now - frac * span, now + (1 - frac) * span


Rt, Gt, Vt, elt = timing_case()
elt[0]["am21"]["step1"] = dict(d=0.05, p=0.2)
n0 = len(LOG)
at(Rt, Gt, 0.10)
Rt.lite_bundle(Vt)
check("timing: at 10 % of the read window with step 1s pending -> not decided yet", Gt["bundle"]["state"] == "planned" and not Gt["bundle"].get("waiting"))
at(Rt, Gt, 0.65)
Rt.lite_bundle(Vt)
check(f"timing: at 65 % with 1 member (< 2) and step 1s pending -> waits until 80 % (logged)", Gt["bundle"]["state"] == "planned"
      and Gt["bundle"].get("waiting", {}).get("members") == 1 and any("waiting for the rest until 80%" in m for m in LOG[n0:]))
at(Rt, Gt, 0.85)
Rt.lite_bundle(Vt)
check(f"timing: at 85 % still 1 member -> no bundle, the decision logs eligible / complete ({Gt['bundle'].get('why')})",
      Gt["bundle"]["state"] == "none" and Gt["bundle"]["eligible_n"] == len(elt) and Gt["bundle"]["step1_complete"] == 1
      and any(f"build decision at 85% of the read window: {len(elt)} eligible, 1 with step 1 complete" in m for m in LOG[n0:]))
Rt.io.q.clear()
Rt, Gt, Vt, elt = timing_case()
elt[0]["am21"]["step1"] = dict(d=0.05, p=0.2)
elt[1]["am21"]["step1"] = dict(d=0.03, p=0.3)
at(Rt, Gt, 0.62)
Rt.lite_bundle(Vt)
b_ = next((c for c in Gt["cands"] if c.get("kind") == "bundle"), None)
check(f"timing: at 62 % with 2 members -> built then ({Gt['bundle']['state']}, {b_ and b_['members']})",
      Gt["bundle"]["state"] == "built" and b_ is not None and sorted(b_["members"]) == sorted([elt[0]["cid"], elt[1]["cid"]])
      and b_["bundle"]["read_frac"] >= 0.6)
Rt.io.q.clear()
Rt, Gt, Vt, elt = timing_case()
for c in elt:
    c["am21"]["step1"] = dict(d=-0.01, p=0.7)
elt[0]["am21"]["step1"] = dict(d=0.05, p=0.2)
at(Rt, Gt, 0.05)
Rt.lite_bundle(Vt)
check("timing: every eligible step 1 complete -> decided at once, even at 5 % (1 member -> none)", Gt["bundle"]["state"] == "none"
      and Gt["bundle"]["step1_complete"] == len(elt))
Rt.io.q.clear()

print(f"\n{ok} ok, {fail} failed")
sys.exit(1 if fail else 0)
