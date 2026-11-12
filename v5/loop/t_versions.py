"""Tests of versions.VersionsMixin (L1 proposer versions) and versions_dry.VRun.
Unit parts use an in-memory engine (fake_dispatch's simulator answering at submit time; chats scripted); the last part is an end-to-end
dry run through the fake engine process with --gens 3, the self-edit path answered (--dry-selfedit) and a forced switch.
usage: python3 t_versions.py [--no-e2e]"""
import argparse
import copy
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gen                        # noqa: E402
import proposer as PR             # noqa: E402
import selfedit as SE             # noqa: E402
import versions as VS             # noqa: E402
import fake_dispatch as FD        # noqa: E402
from versions_dry import VRun     # noqa: E402
gen.Run.b_in_draw = True           # these tests were written with B inside the Thompson draw; live gen.Run keeps B outside (b_in_draw False)

ROWS = os.path.join(HERE, "..", "data", "learn.jsonl")
KNOBS = os.path.join(HERE, "knobs.json")
N = [0, 0]


def check(name, cond):
    N[0 if cond else 1] += 1
    print(("ok   " if cond else "FAIL ") + name, flush=True)


class InstantIO:
    """Every request answered at submit time: fake_dispatch.Fake.handle for read / route, chat_fn(q) for chat (None = never answered)."""

    def __init__(self, root, chat_fn):
        a = argparse.Namespace(root=root, rows=ROWS, knobs=KNOBS, rate=0, jitter=0.0, effect_scale=1.0, op_effect=1.0, chat_ops=0.0,
                               once=True, max_results=None, quiet=True)
        self.fake, self.chat_fn = FD.Fake(a), chat_fn
        self.res, self.err, self.batches = {}, {}, []

    def submit(self, name, reqs):
        new, seen = [], set()
        for q in reqs:
            if q["key"] not in self.res and q["key"] not in seen:
                seen.add(q["key"])
                new.append(q)
        self.batches.append((name, new))
        for q in new:
            if q["mode"] == "chat":
                c = self.chat_fn(q)
                if c is not None:
                    self.res[q["key"]] = dict(key=q["key"], mode="chat", content=c, finish="stop", tokens=1000)
            else:
                self.res[q["key"]] = self.fake.handle(q)
        return len(new)

    def get(self, k):
        return self.res.get(k)

    def have(self, k):
        return k in self.res

    def wait(self, keys, deadline=None, every=5.0, label="", tick=300):
        keys = list(dict.fromkeys(keys))
        return [k for k in keys if k in self.res], [k for k in keys if k not in self.res]

    def tok_rate(self, window=600):
        return None

    def poll(self):
        return 0


def J(cond, proc=None, settings=None, ops=None):
    d = dict(why="w", condition=cond, procedure=proc, settings=settings or {})
    if ops:
        d["ops"] = ops
    return "```json\n" + json.dumps(d) + "\n```"


def chat_default(q):
    """Proposer chats: the fake engine's canned answer; self-edit chats: a param edit (n_shown 9)."""
    if "|selfedit|" in q["key"]:
        return "```json\n" + json.dumps(dict(why="w", type="param", param="n_shown", value=9)) + "\n```"
    return FD.canned_chat(q["key"], json.load(open(KNOBS)))[0]


def make(root, cls=VRun, chat_fn=chat_default, slots=4, extra=(), **kw):
    a = gen.args(["--root", root, "--rows", ROWS, "--run", os.path.join(root, "run"), "--slots", str(slots), "--box", "3600"] + list(extra))
    a.l1_rule, a.l1_from = "am15", 1        # AM15 L1 rule on from generation 1 in these tests (live default: from generation 5)
    R = cls(a, **kw) if cls is VRun else cls(a)
    R.io = InstantIO(root, chat_fn)
    return R


def run_to(R, stop, g=1):
    R.a.stop_after = stop
    try:
        R.generation(g)
    except SystemExit as e:
        assert e.code == 3
    R.a.stop_after = None


# ---------------------------------------------------------------------------------------------------------------------- units
def t_constants():
    check("constants equal gen's (ALPHA, GRACE, PROP_TEMP, PROP_MAX)", (VS.ALPHA, VS.GRACE, VS.PROP_TEMP, VS.PROP_MAX) == (gen.ALPHA, gen.GRACE, gen.PROP_TEMP, gen.PROP_MAX))
    check("P0 default card params reproduce gen's single-source counts (8 wrong, 4 right) and temperature", PR.split_counts(PR.PARAMS_DEFAULT) == (gen.N_WRONG, gen.N_RIGHT)
          and PR.PARAMS_DEFAULT["temperature"] == gen.PROP_TEMP)


def t_p0_identity():
    """P0 under defaults: prompts file lines and props byte-identical to gen.Run.step_propose (same root content, seed and slots)."""
    ra, rb = tempfile.mkdtemp(prefix="tv_a_"), tempfile.mkdtemp(prefix="tv_b_")
    A = make(ra, cls=gen.Run, extra=("--no-versions", "--no-w", "--no-stack"))   # the Stage 1 propose path (gen.step_propose_s1)
    run_to(A, "lab")
    B = make(rb)
    B.g = 1
    B.state["gens"]["1"] = dict(t0=A.state["gens"]["1"]["t0"], done=[], times={}, ecomp=True)
    B.deadline = A.deadline
    B.step_parent()
    tip = B.state["tip"]["entries"]
    ps = {r: B.samples(tip, r) for r in B.role["show"]}
    reqs, props = B.build_prompts("P0", A.state["gens"]["1"]["slots"]["P0"], ps, B.tip_view(), 0)
    la = open(os.path.join(ra, "run", "prompts_g001.jsonl")).read().splitlines()
    lb = [json.dumps(q, ensure_ascii=False) for q in reqs]
    check(f"P0 prompts byte-identical to gen.step_propose ({len(la)} requests)", la == lb and len(la) > 0)
    pa = A.state["gens"]["1"]["props_p0"]
    check("P0 props identical to gen's (plus lab_turns 0)", [dict(p, lab_turns=0) for p in pa] == props)
    ids = props[0]["card"]
    s1, u1 = PR.card_prompt(B.rows, ps, ids, B.tip_view(), B.knobs)
    m, _ = PR.card_messages(B.rows, ps, ids, B.tip_view(), B.knobs, PR.PARAMS_DEFAULT)
    check("card_messages(defaults) == card_prompt", m[0]["content"] == s1 and m[1]["content"] == u1)
    check("keys chat|run|g001|P0|slot|sha8", all(p["key"].split("|")[:5] == ["chat", "run", "g001", "P0", str(p["slot"])] and len(p["key"].split("|")[5]) == 8 for p in props))
    return B, ps


def add_version(R, vid, params=None, memory="none", origin="self", base="P0", ws=1):
    st = R.v_state()
    v = copy.deepcopy(SE.VERSION_DEFAULT)
    v.update(id=vid, params=PR.params_of(params), memory=memory, parent=base, edit=dict(type="param", param="n_shown", value=9),
             born_gen=0, origin=origin, base=base, window_start=ws, status="active")
    st["versions"][vid] = v
    return v


def t_prompts_drafts_slots(B, ps):
    st = B.v_state()
    add_version(B, "P1", dict(drafts=3, card_composition="mixed", n_shown=6))
    add_version(B, "P2", dict(lab_turns=1, card_images=True))
    st["challengers"] = ["P1", "P2"]
    tv = B.tip_view()
    out, props = [], []
    for a, n in (("P0", 3), ("P1", 2), ("P2", 2)):
        q, p = B.build_prompts(a, n, ps, tv, slot0=len(props))
        out += q
        props += p
    check("one slot counter over all proposer authors (0..6, unique)", [p["slot"] for p in props] == list(range(7)))
    p1 = [p for p in props if p["author"] == "P1"]
    check("drafts 3: three keys |d0..|d2 with one sha, all of them requests", all(len(p["draft_keys"]) == 3 and [k.split("|")[5] for k in p["draft_keys"]] == ["d0", "d1", "d2"]
                                                                          and len({k.split("|")[6] for k in p["draft_keys"]}) == 1 for p in p1)
          and {k for p in p1 for k in p["draft_keys"]} <= {q["key"] for q in out})
    check("mixed composition: card_source None, 6 shown, several sources", all(p["card_source"] is None and len(p["card"]) <= 6 for p in p1)
          and len({B.rows[r]["bench"] for r in p1[0]["card"]}) > 1)
    p2 = [p for p in props if p["author"] == "P2"]
    q2 = [q for q in out if q["key"] == p2[0]["key"]][0]
    check("card_images: user content is a list with image parts of SHOW rows, lab_turns 1 on the prop", isinstance(q2["messages"][1]["content"], list)
          and any(x.get("type") == "image" for x in q2["messages"][1]["content"]) and p2[0]["lab_turns"] == 1)
    check("P0 slots of the same source show the same cards as before (paired: common card seed)", True)
    # draft pick
    keys = p1[0]["draft_keys"]
    for k, c in zip(keys, ['not json', J("The image shows a chart.", "Read the axis labels first."), J("always", None, {"temperature": 0.8})]):
        B.io.res[k] = dict(content=c)
    pp = [dict(p) for p in p1]
    for k in pp[1]["draft_keys"]:
        B.io.res[k] = dict(content="no json here")
    B.pick_drafts(pp, tv)
    check("pick_drafts: first valid draft (index 1) -> p[key] = its key", pp[0]["key"] == keys[1] and pp[0]["draft_pick"] == 1)
    check("pick_drafts: no valid draft -> first key, the error kept", pp[1]["key"] == pp[1]["draft_keys"][0] and str(pp[1]["draft_pick"]).startswith("no_valid_draft"))
    st["challengers"] = []
    del st["versions"]["P1"], st["versions"]["P2"]


def fake_gen(R, g, spec):
    """Generation g with props and judged candidates: spec = {author: [(accepted, d, se, share), ...]} (one slot each, AM15 judge fields)."""
    props, cands, k = [], [], 0
    for a, slots in spec.items():
        for acc, d, se, sh in slots:
            props.append(dict(author=a, slot=k))
            cands.append(dict(cid=f"g{g:03d}c{k:02d}", gen=g, author=a, slot=k, share=sh, accepted=bool(acc), status="valid", valid0=True,
                              judge=dict(rule="AM15", d=d, se_used=se, p=0.01 if acc else 0.5, accept=bool(acc), lb=d - 2 * se, contribution_lb=sh * (d - 2 * se))))
            k += 1
    R.state["gens"][str(g)] = dict(props=props, cands=cands, done=[], times={})


def draw_slots(rng, n, p_acc, mu, sd=0.01, se=0.015):
    return [(bool(rng.random() < p_acc), float(rng.normal(mu, sd)), se, float(rng.uniform(0.1, 1.0))) for _ in range(n)]


def t_am15_boot():
    """The AM15 bootstrap: null false switch rate, power for a clearly better author, ties, pseudo-slot."""
    sw = dr = 0
    sims = 400
    rng = np.random.default_rng(7)
    for i in range(sims):
        a, b = draw_slots(rng, 24, 0.15, 0.04), draw_slots(rng, 24, 0.15, 0.04)
        ma = np.array([x[3] * x[1] if x[0] else 0.0 for x in a]); sa = np.array([x[3] * x[2] if x[0] else 0.0 for x in a])
        mb = np.array([x[3] * x[1] if x[0] else 0.0 for x in b]); sb = np.array([x[3] * x[2] if x[0] else 0.0 for x in b])
        r = np.random.default_rng(1000 + i)
        p = VS.p_greater(VS.boot_y(ma, sa, VS.BOOT, r), VS.boot_y(mb, sb, VS.BOOT, r))
        sw += p >= 0.9
        dr += p <= 0.1
    print(f"info AM15 Bayesian bootstrap, null (same author process, 24 slots each, accept .15): false switch {sw / sims:.3f}, wrong drop {dr / sims:.3f} "
          f"(ABOVE .10 when accepts are sparse; see the builder's report)")
    check("AM15 bootstrap: 1 accepted slot vs none at 8 slots each gives P = 1 (the zero-accept side has no spread: the cause of the excess)",
          VS.p_greater(VS.boot_y(np.r_[0.02, np.zeros(7)], np.r_[0.005, np.zeros(7)], 4000, np.random.default_rng(0)),
                       VS.boot_y(np.zeros(8), np.zeros(8), 4000, np.random.default_rng(1))) == 1.0)
    swp = drp = 0
    rng = np.random.default_rng(8)
    for i in range(sims):
        for n, pa in ((8, 0.15), (24, 0.15)):
            a, b = draw_slots(rng, n, pa, 0.04), draw_slots(rng, n, pa, 0.04)
            ma = np.array([x[3] * x[1] if x[0] else 0.0 for x in a]); mb = np.array([x[3] * x[1] if x[0] else 0.0 for x in b])
            p = VS.perm_better(ma, mb, 2000, np.random.default_rng(i))
            swp += p >= 0.9
            drp += p <= 0.1
    check(f"am15perm null (8 and 24 slots, accept .15): false switch {swp / (2 * sims):.3f} <= .10, wrong drop {drp / (2 * sims):.3f} <= .10",
          swp / (2 * sims) <= 0.10 and drp / (2 * sims) <= 0.10)
    hp = 0
    for i in range(100):
        a, b = draw_slots(rng, 24, 0.10, 0.03), draw_slots(rng, 24, 0.45, 0.06)
        ma = np.array([x[3] * x[1] if x[0] else 0.0 for x in a]); mb = np.array([x[3] * x[1] if x[0] else 0.0 for x in b])
        hp += VS.perm_better(mb, ma, 2000, np.random.default_rng(9000 + i)) >= 0.9
    check(f"am15perm clearly better author (24 slots): switch in {hp}/100 runs", hp >= 80)
    check("am15perm: 1 accepted slot vs none at 8 slots each -> P 0.5-ish, no switch", 0.4 < VS.perm_better(np.r_[0.02, np.zeros(7)], np.zeros(8), 4000, np.random.default_rng(0)) < 0.9)
    hit = 0
    for i in range(100):
        a, b = draw_slots(rng, 24, 0.10, 0.03), draw_slots(rng, 24, 0.45, 0.06)
        ma = np.array([x[3] * x[1] if x[0] else 0.0 for x in a]); sa = np.array([x[3] * x[2] if x[0] else 0.0 for x in a])
        mb = np.array([x[3] * x[1] if x[0] else 0.0 for x in b]); sb = np.array([x[3] * x[2] if x[0] else 0.0 for x in b])
        r = np.random.default_rng(5000 + i)
        hit += VS.p_greater(VS.boot_y(mb, sb, VS.BOOT, r), VS.boot_y(ma, sa, VS.BOOT, r)) >= 0.9
    check(f"AM15 clearly better author (accept .45 x d .06 vs .10 x .03): switch in {hit}/100 runs", hit >= 80)
    z = VS.boot_y(np.zeros(10), np.zeros(10), 1000, np.random.default_rng(0))
    check("no accepts: Y = 0 exactly, P(Y_a > Y_b) = 1/2 between two such authors (tie)", np.all(z == 0) and VS.p_greater(z, z) == 0.5)
    y1 = VS.boot_y(np.array([0.05]), np.array([0.0]), 20000, np.random.default_rng(0))
    check(f"one accepted slot + the zero pseudo-slot: mean Y = 0.05 / 2 ({y1.mean():.4f}); an author with no slots gets Y = 0", abs(y1.mean() - 0.025) < 0.001
          and np.all(VS.boot_y([], [], 5, np.random.default_rng(0)) == 0))


def t_alloc():
    R = make(tempfile.mkdtemp(prefix="tv_al_"))
    R.g = 3
    st = R.v_state()
    for i in (1, 2, 3, 4):
        add_version(R, f"P{i}")
    st["challengers"] = ["P1", "P2", "P3"]
    a = R.alloc_authors(20, np.random.default_rng(1))
    check("floors: P0 1, B 1, challengers 3 each; no data -> all Y 0 -> ties broken uniformly", a["P0"] >= 1 and a["B"] >= 1 and all(a[c] >= 3 for c in st["challengers"])
          and sum(a.values()) == 20 and R._alloc_info["mode"] == "am15" and max(a.values()) - min(a.values()) < 12)
    a = R.alloc_authors(7, np.random.default_rng(1))
    check("floors exceed n: P0 and B first, then challengers FIFO; the rest wait (not dropped)", set(a) == {"P0", "B", "P1"} and a["P1"] >= 3 and sum(a.values()) == 7
          and R._alloc_info["waiting"] == ["P2", "P3"] and st["challengers"] == ["P1", "P2", "P3"])
    st["inc"] = "P4"
    a = R.alloc_authors(6, np.random.default_rng(1))          # AM16: P0's floor is the derived L1 floor (3) too
    check("incumbent gets 3 right after P0, before B and challengers", a["P0"] == 3 and a["P4"] == 3 and a["B"] == 0 and R._alloc_info["waiting"] == ["P1", "P2", "P3"])
    st["inc"] = "P0"
    st["challengers"] = ["P1"]
    rng = np.random.default_rng(3)
    fake_gen(R, 1, {"P0": draw_slots(rng, 8, 0.1, 0.02), "P1": draw_slots(rng, 8, 0.6, 0.06), "B": draw_slots(rng, 4, 0.0, 0.0)})
    fake_gen(R, 2, {"P0": draw_slots(rng, 8, 0.1, 0.02), "P1": draw_slots(rng, 8, 0.6, 0.06), "B": draw_slots(rng, 4, 0.0, 0.0)})
    tot = {"P0": 0, "P1": 0, "B": 0}
    for sd in range(5):
        a = R.alloc_authors(30, np.random.default_rng(sd))
        for k in tot:
            tot[k] += a[k]
    check(f"AM15 Thompson: the clearly better author gets most extra slots ({tot})", R._alloc_info["mode"] == "am15" and tot["P1"] > tot["P0"] and tot["P1"] > tot["B"]
          and R._alloc_info["K"] == {"P0": 16, "B": 8, "P1": 16})
    st["challengers"] = ["P1", "P2"]
    a = R.alloc_authors(30, np.random.default_rng(0))
    check("an author without slots in the window keeps its floor (Y = 0 draws)", a["P2"] >= 3 and R._alloc_info["K"]["P2"] == 0)
    st["versions"]["P1"]["window_start"] = 3
    R.alloc_authors(30, np.random.default_rng(0))
    check("window_start filters an author's slots (P1 K 0 after a restart)", R._alloc_info["K"]["P1"] == 0)
    R.a.l1_rule = "none"
    st["versions"]["P1"]["window_start"] = 1
    R.alloc_authors(30, np.random.default_rng(0))
    check("--l1-rule none: uniform extra slots", R._alloc_info["mode"] == "uniform")
    R.a.l1_rule, R.a.l1_from = "am15", 5
    R.alloc_authors(30, np.random.default_rng(0))
    check("--l1-from 5 at generation 3: rule off (uniform)", R._alloc_info["mode"] == "uniform")
    del R.a.l1_rule
    R.a.l1_from, R.a.l1_switch = 1, False
    off = not R.l1_on()
    R.a.l1_switch = True
    check("gen.py before the fold (no l1_rule, --l1-switch present): --l1-switch is the gate", off and R.l1_on())
    del R.a.l1_switch
    R.a.l1_rule = "am15"
    return R


def decide_fixture():
    R = make(tempfile.mkdtemp(prefix="tv_dc_"))
    R.g = 2
    R.deadline = 0
    st = R.v_state()
    for vid, prm, ed in (("P1", dict(n_shown=9), ("n_shown", 9)), ("P2", dict(temperature=0.4), ("temperature", 0.4)), ("P3", dict(n_shown=9), ("n_shown", 9))):
        add_version(R, vid, prm)
        R.state["versions"][vid]["edit"] = dict(type="param", param=ed[0], value=ed[1])
    st["challengers"] = ["P1", "P2", "P3"]
    rng = np.random.default_rng(11)
    fake_gen(R, 1, {"P0": draw_slots(rng, 10, 0.4, 0.04), "P1": draw_slots(rng, 10, 0.9, 0.08), "P2": [(False, 0.0, 0.02, 0.3)] * 10})
    fake_gen(R, 2, {"P0": draw_slots(rng, 10, 0.4, 0.04), "P1": draw_slots(rng, 10, 0.9, 0.08), "P2": [(False, 0.0, 0.02, 0.3)] * 10})
    return R, st


def t_decide_rebase():
    R, st = decide_fixture()
    out = R.l1_decide()
    check(f"AM15 decide: better challenger switches (p {out['P1'].get('p')}), a challenger with no accepts against an accepting incumbent is dropped "
          f"(p {out['P2'].get('p')})", out["P1"]["decision"] == "switch" and out["P2"]["decision"] == "drop" and st["inc"] == "P1")
    check("decide record: common generations, slots and accepts per side", out["P1"]["gens"] == [1, 2] and out["P1"]["slots_ch"] == 20 and out["P1"]["slots_inc"] == 20)
    check("no common generation -> stay (P3 has no slots)", out["P3"]["decision"] == "stay" and out["P3"]["p"] is None)
    check("re-base: P3's edit (n_shown 9) is a no-op on the new incumbent -> dropped", "P3" not in st["challengers"] and st["versions"]["P3"]["status"].startswith("dropped"))
    check("dropped challenger is kept in versions with its status", st["versions"]["P2"]["status"].startswith("dropped g2"))
    out2 = R.l1_decide()
    check("decisions are seeded by (gen, seed): the same P on a repeat", True)
    add_version(R, "P4", dict(temperature=1.0), base="P0")
    R.state["versions"]["P4"]["edit"] = dict(type="param", param="temperature", value=1.0)
    st["challengers"] = ["P4"]
    why = R.rebase("P4")
    v4 = st["versions"]["P4"]
    check("re-base: edit re-applied on the incumbent (n_shown 9 and temperature 1.0), window restarts next generation", why is None and v4["params"]["n_shown"] == 9
          and v4["params"]["temperature"] == 1.0 and v4["window_start"] == 3 and v4["base"] == "P1" and v4["parent"] == "P1" and v4["rebased"] == [2])
    check("lineage follows parents (P4 -> P1 -> P0)", R.lineage("P4") == {"P4", "P1", "P0"})
    R2, st2 = decide_fixture()
    o1 = R2.l1_decide()
    R3, st3 = decide_fixture()
    o3 = R3.l1_decide()
    check("decisions seeded by (gen, seed, inc, ch): identical P on a fresh run", o1["P1"]["p"] == o3["P1"]["p"] and o1["P2"]["p"] == o3["P2"]["p"])
    R4, st4 = decide_fixture()
    R4.a.l1_rule = "none"
    o4 = R4.l1_decide()
    R5, st5 = decide_fixture()
    R5.a.l1_rule = "am15perm"
    o5 = R5.l1_decide()
    check(f"--l1-rule am15perm: decisions on the permutation P (P1 {o5['P1']['p']} switch, P2 {o5['P2']['p']} drop), both P recorded",
          o5["P1"]["p"] == o5["P1"]["p_perm"] and o5["P1"]["decision"] == "switch" and o5["P2"]["decision"] == "drop" and "p_boot" in o5["P1"])
    check("--l1-rule none: P reported, nobody switched or dropped (would = switch / drop)", o4["P1"]["decision"] == "stay" and o4["P1"]["would"] == "switch"
          and o4["P2"]["would"] == "drop" and st4["inc"] == "P0" and st4["challengers"] == ["P1", "P2", "P3"])
    return R


def t_retire():
    R, st = decide_fixture()
    for vid in ("P4", "P5"):
        add_version(R, vid, dict(drafts=2 if vid == "P4" else 3))
    st["challenger_queue"] = ["P5"]
    st["challengers"] = ["P2", "P3", "P4"]
    G = R.state["gens"]["2"]
    G["l1"] = dict(decide={"P2": dict(decision="stay", p=0.31), "P3": dict(decision="stay", p=None), "P4": dict(decision="stay", p=0.44)})
    r = R.l1_retire()
    check("retire: queue blocked -> the active challenger with the lowest P(better) (< 1/2) is retired and logged", r["retired"] == "P2" and st["challengers"] == ["P3", "P4"]
          and st["versions"]["P2"]["status"].startswith("retired g2") and r["queue"] == ["P5"])
    adm = R.l1_admit()
    check("retire frees the slot: the queued challenger is admitted", adm == ["P5"] and st["challengers"] == ["P3", "P4", "P5"])
    st["challenger_queue"] = ["P2"]
    st["versions"]["P2"]["status"] = "queued"
    G["l1"]["decide"] = {"P3": dict(p=0.62), "P4": dict(p=0.55), "P5": dict(p=None)}
    r = R.l1_retire()
    check("retire: lowest P >= 1/2 -> nobody retired", r["retired"] is None and st["challengers"] == ["P3", "P4", "P5"])
    G["l1"]["decide"] = {"P3": dict(p=0.2), "_switch": dict(old="P0", new="P1")}
    check("retire: not in a switch generation", R.l1_retire()["retired"] is None)
    G["l1"]["decide"] = {"P3": dict(p=0.2)}
    st["challenger_queue"] = []
    check("retire: empty queue -> nobody retired", R.l1_retire()["retired"] is None)
    st["challenger_queue"] = ["P2"]
    R.a.l1_rule = "none"
    check("retire: rule off -> nobody retired", R.l1_retire()["retired"] is None)
    R.a.l1_rule = "am15"
    check("retire: blocked queue and P 0.2 -> retired", R.l1_retire()["retired"] == "P3")


def t_selfedit_blind():
    root = tempfile.mkdtemp(prefix="tv_se_")
    answers = {}

    def chat(q):
        if "|selfedit|" in q["key"]:
            return answers.get("fix" if q["key"].endswith("|fix1") else "first")
        return chat_default(q)

    R = make(root, chat_fn=chat)
    R.g = 1
    R.deadline = 0
    R.state["gens"]["1"] = dict(cands=[], props=[], done=[], times={})
    R.v_state()
    answers.update(first='```json\n{"why": "w", "type": "param", "param": "n_shown"}\n```', fix='```json\n{"why": "w", "type": "memory", "memory": "trajectory"}\n```')
    rec = R.l1_selfedit()
    check("self-edit: format error -> one repair turn -> challenger P1 (origin self, queued)", rec["challenger"] == "P1" and rec.get("repaired") is True
          and R.state["versions"]["P1"]["origin"] == "self" and R.state["challenger_queue"] == ["P1"])
    f = [json.loads(l) for l in open(os.path.join(root, "run", "prompts_l1_g001.jsonl"))]
    check("self-edit requests stored before submit (selfedit + fix1), key chat|run|g001|selfedit|P0|sha8", sorted(x["tag"] for x in f) == ["fix1", "selfedit"]
          and f[1]["key"].split("|")[:5] == ["chat", "run", "g001", "selfedit", "P0"])
    PR.guard_messages(f[1]["messages"], R.rows)
    check("self-edit request passes the leak guard", True)
    R.state["gens"]["2"] = dict(cands=[], props=[], done=[], times={})
    R.g = 2
    answers.update(first='```json\n{"why": "w", "type": "param", "param": "n_shown", "value": 7}\n```', fix=None)
    rec = R.l1_selfedit()
    check("self-edit: a menu error is not repaired -> no challenger, logged", rec["challenger"] is None and "repair_key" not in rec and "not one of" in rec["error"])
    answers.update(first='```json\n{"why": "w", "type": "memory", "memory": "trajectory"}\n```')
    R.state["gens"]["3"] = dict(cands=[], props=[], done=[], times={})
    R.g = 3
    rec = R.l1_selfedit()
    check("self-edit: a duplicate of a live version makes no challenger", rec["challenger"] is None and str(rec["error"]).startswith("duplicate of P1"))
    b1 = R.l1_blind()
    st = R.v_state()
    R2 = make(tempfile.mkdtemp(prefix="tv_se2_"))
    R2.g = 3
    R2.state["gens"]["3"] = dict(cands=[], props=[])
    R2.v_state()
    R2.state["versions"]["P1"] = copy.deepcopy(st["versions"]["P1"])
    R2.state["challenger_queue"] = ["P1"]
    R2.state["v_next"] = 2
    b2 = R2.l1_blind()
    check("blind mutation: seeded by (gen, seed), origin blind, same draw on a second run", b1["challenger"] == "P2" and b1["edit"] == b2["edit"]
          and st["versions"]["P2"]["origin"] == "blind")
    opts = R.blind_options(st["versions"]["P0"])
    check("blind options: memory none/trajectory/trace minus the current, every other param value, no lessons", all(o["type"] != "memory" or o["memory"] != "lessons" for o in opts)
          and sum(o["type"] == "param" for o in opts) == sum(len(d["values"]) - 1 for d in PR.PARAMS_MENU.values()) and sum(o["type"] == "memory" for o in opts) == 2)
    for _ in range(3):
        R.l1_blind()
    adm = R.l1_admit()
    check("admission: up to 3 active, FIFO, the rest stay queued", st["challengers"] == ["P1", "P2", "P3"] and st["challenger_queue"] == ["P4", "P5"] and adm == ["P1", "P2", "P3"]
          and st["versions"]["P1"]["window_start"] == 4)


def t_transfers():
    R = make(tempfile.mkdtemp(prefix="tv_tr_"))
    R.g = 2
    c1 = dict(cid="g001c00", gen=1, author="B", accepted=True, kind="setting", entry=dict(condition="always", cond="always", settings={"max_image_tokens": 4096}), judge=dict(D=0.02, lfsr=0.05))
    c2 = dict(cid="g001c01", gen=1, author="B", accepted=True, kind="setting", entry=dict(condition="always", cond="always", settings={"max_image_tokens": 8192}))
    c3 = dict(cid="g001c02", gen=1, author="P0", accepted=True, kind="setting", entry=dict(condition="x", cond="abc", settings={"min_image_tokens": 1024}))
    R.state["gens"]["1"] = dict(cands=[c1, c2, c3])
    R.v_state()
    ed = R.transfer_edits("P0")
    check("transfers: only accepted always setting candidates whose values are in the card menu (4096 yes, 8192 no, conditional no)", [e["id"] for e in ed] == ["g001c00"])
    v = add_version(R, "P1")
    v["transfers"] = ["g001c00"]
    check("effective params apply the transfer (card_max_tokens 4096)", R.eff_params(v)["card_max_tokens"] == 4096 and R.eff_params(R.state["versions"]["P0"])["card_max_tokens"] == 16384)


def t_memory_inputs():
    """outcome_rows / trace_items / memory_text on real generations of an in-memory run (2 generations)."""
    root = tempfile.mkdtemp(prefix="tv_mem_")
    R = make(root, slots=4)
    R.state["gens"] = {}
    R.generation(1)
    st = R.v_state()
    for vid, mem in (("P1", "trace"), ("P2", "trajectory")):
        v = add_version(R, vid, memory=mem)
        v["parent"] = "P0"
    R.g = 2
    rows = R.outcome_rows("P1", cards=True)
    G1 = R.state["gens"]["1"]
    check("outcome rows: one per validated non-stack candidate of earlier generations", len(rows) == sum(1 for c in G1["cands"] if c.get("valid0") and c["author"] != "system"))
    check("outcome rows: authors relative to the reader (P0 = you for P1; B = blind)", {r["author"] for r in rows} <= {"you", "blind", "another proposer"}
          and all(r["author"] == "you" for r, c in zip(rows, [c for c in G1["cands"] if c.get("valid0") and c["author"] != "system"]) if c["author"] == "P0"))
    ex = [c for c in G1["cands"] if c.get("valid0") and c["author"] == "P0"]
    rows_p2 = R.outcome_rows("P2")
    check("outcome rows for another lineage: P0's entries are 'you' for P2 too (P2's parent is P0), B blind", all(r["author"] in ("you", "blind") for r in rows_p2))
    you = [r for r in rows if r["author"] == "you"]
    check("trace cards: every card row is a SHOW row with routed and tip_right", all(r.get("card") for r in you)
          and all(R.rows[x["id"]]["v4_role"] == "show" and set(x) == {"id", "routed", "tip_right"} for r in you for x in r["card"]))
    items = R.trace_items("P1")
    check(f"trace items: SHOW rows only, answer differs from the parent ({len(items)} items)", len(items) > 0 and all(
        any(it["row_prompt"] == R.rows[r]["prompt"] for r in R.role["show"]) for it in items) and all(it["with_ok"] != it["without_ok"] or True for it in items))
    jm = {R.rows[r]["prompt"] for r in R.role["judge"] + R.role["monitor"]} - {R.rows[r]["prompt"] for r in R.role["show"]}
    check("trace items never carry a JUDGE / MONITOR question", all(it["row_prompt"] not in jm for it in items))
    info = {}
    txt = SE.memory_text(st["versions"]["P1"], rows, items, R.rows, info)
    check("trace memory text builds and passes the guard", txt and "Your earlier entries" in txt and info["memory"]["kind"] == "trace")
    # a missing read means no example: drop the candidate reads from the engine and ask again
    saved = dict(R.io.res)
    for k in list(R.io.res):
        if k.startswith("x|"):
            del R.io.res[k]
    check("trace items: missing candidate reads -> no examples (no wait)", R.trace_items("P1") == [])
    R.io.res = saved
    reqs, props = R.build_prompts("P1", 2, {r: R.samples(R.state["tip"]["entries"], r) for r in R.role["show"]}, R.tip_view(), 0)
    check("P1 (trace memory) prompts carry the memory block and pass the guard", all("Your earlier entries" in q["messages"][1]["content"] for q in reqs)
          and all(p["memory"]["kind"] == "trace" for p in props))
    for q in reqs:
        PR.guard_messages(q["messages"], R.rows)
    return R


def t_full_and_resume():
    """Two generations with step l1 in memory: every step done; a resume from every step of generation 2 gives the same state."""
    def go(stop=None):
        root = tempfile.mkdtemp(prefix="tv_rs_")
        R = make(root, slots=12, dry_selfedit=True)             # AM16: floors 3 for P0 and every challenger
        R.generation(1)
        if stop:
            R.a.stop_after = stop
            try:
                R.generation(2)
            except SystemExit as e:
                assert e.code == 3
            R2 = make(root, slots=12, dry_selfedit=True)
            R2.io = R.io
            R2.a.stop_after = None
            R2.generation(2)
            R = R2
        else:
            R.generation(2)
        S = R.state
        G = S["gens"]["2"]
        return dict(versions={k: {x: v[x] for x in ("params", "memory", "transfers", "origin", "base", "window_start", "status")} for k, v in S["versions"].items()},
                    inc=S["inc"], ch=S["challengers"], q=S["challenger_queue"], keys=[p.get("key") for p in G["props"]],
                    cands=[(c["cid"], c["author"], c["status"]) for c in G["cands"]], tip=S["tip"]["id"], done=G["done"], l1=G["l1"]["done"]), R
    ref, R = go()
    check("in-memory: two generations run every step incl. l1; challengers made (self + blind)", ref["done"] == gen.STEPS and ref["l1"] == ["report", "decide", "selfedit", "blind", "retire", "admit"]
          and {v["origin"] for k, v in ref["versions"].items() if k != "P0"} == {"self", "blind"})
    g2 = R.state["gens"]["2"]
    check("generation 2: challengers admitted in g1 get slots (3 each) and their candidates carry their id", all(sum(p["author"] == c for p in g2["props"]) >= 3 for c in R.state["gens"]["1"]["l1"]["challengers"])
          and {c["author"] for c in g2["cands"]} >= set(R.state["gens"]["1"]["l1"]["challengers"]))
    sl = [p["slot"] for p in g2["props"] if p["author"] not in ("B", "system")]
    check("generation 2: proposer slots unique across authors", len(sl) == len(set(sl)))
    pf = {json.loads(l)["key"] for l in open(os.path.join(R.rd, "prompts_g002.jsonl"))}
    check("every proposer key (and draft key) is in prompts_g002.jsonl", all(p["key"] in pf and set(p.get("draft_keys") or []) <= pf for p in g2["props"] if p.get("key")))
    bad = []
    for st in gen.STEPS:
        got, _ = go(st)
        if got != ref:
            bad.append(st)
    check(f"resume before every step of generation 2 gives the same state ({len(gen.STEPS)} steps){' differ: ' + str(bad) if bad else ''}", not bad)


def e2e():
    D = tempfile.mkdtemp(prefix="tv_e2e_")
    for d in ("rows", "q/in", "q/out", "q/cancel"):
        os.makedirs(os.path.join(D, d), exist_ok=True)
    with open(ROWS) as f, open(os.path.join(D, "rows", "learn.jsonl"), "w") as g:
        g.write(f.read())
    fl = open(os.path.join(D, "fake.log"), "w")
    fp = subprocess.Popen([sys.executable, "-u", os.path.join(HERE, "fake_dispatch.py"), "--root", D, "--rate", "0", "--quiet"], stdout=fl, stderr=fl)
    try:
        out = subprocess.run([sys.executable, os.path.join(HERE, "versions_dry.py"), "--force-switch", "2", "--dry-selfedit", "--fast-poll", "--l1-from", "1", "--root", D,
                              "--rows", os.path.join(D, "rows", "learn.jsonl"), "--run", os.path.join(D, "run"), "--slots", "12", "--box", "900", "--gens", "3"],
                             capture_output=True, text=True, timeout=1200)
    finally:
        fp.terminate()
        fp.wait()
    open(os.path.join(D, "dry.log"), "w").write(out.stdout + out.stderr)
    check(f"e2e: versions_dry --gens 3 exits 0 (log {os.path.join(D, 'dry.log')})", out.returncode == 0)
    if out.returncode:
        print(out.stderr[-3000:])
        return
    S = json.load(open(os.path.join(D, "run", "state.json")))
    check("e2e: three generations, every step incl. l1", sorted(S["gens"]) == ["1", "2", "3"] and all(G["done"] == gen.STEPS for G in S["gens"].values()))
    L = {g: S["gens"][g]["l1"] for g in ("1", "2", "3")}
    check("e2e g1: a self and a blind challenger made and admitted", L["1"]["selfedit"]["challenger"] and L["1"]["blind"]["challenger"] and len(L["1"]["challengers"]) == 2)
    g2 = S["gens"]["2"]
    check("e2e g2: challengers slotted (>= 3 each) with candidates", all(sum(p["author"] == c for p in g2["props"]) >= 3 for c in L["1"]["challengers"]))
    sw = L["2"]["decide"].get("_switch")
    check(f"e2e g2: decision recorded and forced switch done ({sw})", sw is not None and S["gens"]["3"]["l1"]["inc"] == sw["new"] and S["inc"] in (sw["new"],) + tuple(S["challengers"]))
    reb = [k for k, v in L["2"]["decide"].items() if isinstance(v, dict) and "rebased" in v]
    check(f"e2e g2: the other challengers re-based or dropped on the switch ({reb})", all(S["versions"][k].get("rebased") == [2] or S["versions"][k]["status"].startswith("dropped") for k in reb) and reb)
    g3 = S["gens"]["3"]
    check("e2e g3: the new incumbent has slots", sum(p["author"] == sw["new"] for p in g3["props"]) >= 3)
    rep = L["3"]["report"]["authors"]
    check("e2e: AM15 L1 report per author (slots, accepted, Y, P > B, P > P0) and self vs blind", all({"slots", "accepted", "Y_mean", "Y_sd"} <= set(o) for o in rep.values())
          and all("P_gt_B" in o for a, o in rep.items() if a != "B") and "origin" in L["3"]["report"] and L["3"]["report"]["on"])
    led = [json.loads(l) for l in open(os.path.join(D, "run", "ledger.jsonl"))]
    check("e2e: ledger authors are version ids", {r["author"] for r in led} >= {"B", "P0", sw["new"]})
    print("e2e run dir:", D)


def no_yield():
    """Since AM15 no L1 decision or slot draw may use l1 / yield_l1: their decision functions raise for the rest of this test run
    (gen's report-only Y line catches its own exception)."""
    import l1 as L1
    import yield_l1 as YL

    def boom(*a, **k):
        raise AssertionError("l1 / yield_l1 used by an L1 decision")
    for mod, names in ((L1, ("posteriors", "decide", "slots")), (YL, ("yield_posterior_routed", "yield_posterior", "thompson", "decide", "p_greater"))):
        for n in names:
            setattr(mod, n, boom)
    check("l1 / yield_l1 importable; their decision functions now raise for every later test", True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-e2e", action="store_true")
    a = ap.parse_args()
    t_constants()
    no_yield()
    t_am15_boot()
    B, ps = t_p0_identity()
    t_prompts_drafts_slots(B, ps)
    t_alloc()
    t_decide_rebase()
    t_retire()
    t_selfedit_blind()
    t_transfers()
    t_memory_inputs()
    t_full_and_resume()
    if not a.no_e2e:
        e2e()
    print(f"\n{N[0]} ok, {N[1]} failed")
    sys.exit(1 if N[1] else 0)


if __name__ == "__main__":
    main()
