"""Amendment 7 run path: a7_apply on synthetic votes (member track admit / reject, pool path for an off-scope gate write, tripwire, C2 freeze), q_log + ab_side."""
import sys, os, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from harness.adapters import hw_round as HW, member_cand as MC
from rsi import hboost as HB

class Data:
    rows = {i: dict(key="x" if i < 100 else "y", answer="A", options=["A", "B"]) for i in range(400)}
    def cluster(self, i): return i // 2
class R_:
    def __init__(s): s.c9 = dict(a7=True, harness_ab=True); s.state = {}; s.data = Data(); s.c9_cfg = None; s.log = []
    def say(s, t): s.log.append(t)
H = types.SimpleNamespace(merge4=HB.merge4, paired=HB.paired, gate_scope=lambda m: m.get("scope"), vh=types.SimpleNamespace(key=lambda r: r["key"]),
                          right_rate=lambda s: float(np.mean([v == GOLD for v in s.values()])) if s else None)
GOLD = "A"
def votes(n_rows, p_right, seed):
    rng = np.random.default_rng(seed)
    return {i: {k: ("A" if rng.random() < p_right else "B") for k in (0, 1)} for i in range(n_rows)}

def rec(name, glob=True, dS=0.0):
    return dict(name=name, global_=glob, dS=dS, p_boot=0.5, p_holm=0.5, admissible=False, reason="pool", delta=0.01, C0=1.0, C1=1.0, novelty=0,
                checks=dict(no_harm=True, card_vs_unseen=True, critic=True, fired=True, z=True))

def setup(p_cand, p_null=0.5):
    N = 400
    V = dict(par=votes(N, .5, 1), par3=votes(N, .5, 2), good=votes(N, p_cand, 3), good3=votes(N, p_cand, 4), nul=votes(N, p_null, 5), nul3=votes(N, p_null, 6))
    four = {"par": dict(name="par3"), "good": dict(name="good3"), "nul": dict(name="nul3")}
    by = {"par": dict(name="par")}
    m = dict(name="good", parents=["par"])
    rc = rec("good")
    R = R_()
    return R, V, four, by, m, rc, list(range(N))

orig = HB.paired
def paired_gold(vc, vp, ids):
    out = {}
    for i in ids:
        if i in vc and i in vp:
            rc_ = np.mean([v == GOLD for v in vc[i].values()]); rp = np.mean([v == GOLD for v in vp[i].values()])
            out[i] = rc_ - rp
    return out
H.paired = paired_gold

def test_admit_and_reject():
    for pc, want in ((.70, True), (.50, False)):
        R, V, four, by, m, rc, Rb = setup(pc)
        trip, fr = HW.a7_apply(R, H, 3, [(m, {}, rc)], [(dict(name="nul"), by["par"])], by, V, four, {"good": Rb}, Rb, 0.05, [])
        assert rc["admissible"] is want, (pc, rc["a7"])
        assert rc["a7"]["path"] == "member" and "pool_line" in rc
        assert not trip and not fr
    print("ok   test_admit_and_reject")

def test_tripwire_and_freeze():
    R, V, four, by, m, rc, Rb = setup(.70, p_null=.70)
    trip, fr = HW.a7_apply(R, H, 3, [(m, {}, rc)], [(dict(name="nul"), by["par"])], by, V, four, {"good": Rb}, Rb, 0.05, [])
    assert trip == ["nul"], trip
    assert fr is False                   # 1 of 1 null reads at p <= .05: P(X >= 1 | 1, .05) = .05, not < .05
    trip, fr = HW.a7_apply(R, H, 4, [(m, {}, rec("good"))], [(dict(name="nul"), by["par"])], by, V, four, {"good": Rb}, Rb, 0.05, [])
    assert fr is True and R.state["c1109_a7_c2"]["n"] == 2   # 2 of 2: P = .0025 < .05 -> freeze
    print("ok   test_tripwire_and_freeze")

def test_gate_scope_and_v3():
    """1109c audit: a gate write is read on the member track on its own scope rows (cat:x = rows < 100); gate_v3 also needs it to beat its control."""
    for ctrl_p, v3, want in ((None, False, True), (.50, True, True), (.70, True, False), (None, True, False)):
        R, V, four, by, m, rc, Rb = setup(.75)
        m["gate"] = {"scope": "cat:x"}; rc["global_"] = False
        if v3:
            R.c9["gate_key"] = "prompts"
        if ctrl_p is not None:
            m["gate_control"] = dict(name="good~ctrl", mid="c0", genome={}, note="ctrl")
            V["good~ctrl"] = votes(400, ctrl_p, 7); V["ctrl3"] = votes(400, ctrl_p, 8); four["good~ctrl"] = dict(name="ctrl3")
        HW.a7_apply(R, H, 3, [(m, {}, rc)], [], by, V, four, {"good": Rb}, Rb, 0.05, [])
        assert rc["a7"]["path"] == "gate scope" and rc["a7"]["rows"] == 100, rc["a7"]
        assert rc["admissible"] is want, (ctrl_p, v3, rc["a7"])
    print("ok   test_gate_scope_and_v3")

def test_ab_and_q():
    R = R_()
    s3 = [HW.ab_side(R, 3, k)["notes"] for k in range(4)]
    assert s3 == [False, True, False, True], s3
    assert all(HW.ab_side(R, 2, k)["notes"] for k in range(4))
    for r in (3, 4):
        props = [dict(slot=k, name=f"r{r}s{k}", zx_score=(1.0 if k % 2 else 0.0) + 0.01 * r, c1109=None) for k in range(12)]
        if r == 3:
            for k in range(12): HW.ab_side(R, 3, k)
        else:
            for k in range(12): HW.ab_side(R, 4, k)
        out = HW.q_log(R, r, [dict(slot=k) for k in range(12)], props, set())
    assert out["ab_decision"]["on"] is True and out["ab_decision"]["window"] == "notes", out["ab_decision"]
    assert all(HW.ab_side(R, 11, k)["notes"] for k in range(2))
    R.state["c1109_ab_dec"]["skills"] = dict(on=False)
    assert not HW.ab_side(R, 12, 1)["skills"]
    print("ok   test_ab_and_q")

for f in (test_admit_and_reject, test_tripwire_and_freeze, test_gate_scope_and_v3, test_ab_and_q):
    f()
print("ALL GREEN")
