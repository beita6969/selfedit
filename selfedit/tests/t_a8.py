"""Amendment 8: carry_out picks D > 0, up > down, not admitted, cap; carry_judge accumulates fresh reads into the e-process, admits at E >= K/alpha
with the guards, drops on two non-positive reads; a7_apply keeps carried candidates out of the round's Holm family."""
import sys, os, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from harness.adapters import hw_round as HW
from rsi import hboost as HB

class Data:
    def cluster(self, i): return i
def votes(n, p, seed):
    rng = np.random.default_rng(seed)
    return {i: {k: ("A" if rng.random() < (p[i] if hasattr(p, "__len__") else p) else "B") for k in (0, 1)} for i in range(n)}
def paired_gold(vc, vp, ids):
    return {i: np.mean([v == "A" for v in vc[i].values()]) - np.mean([v == "A" for v in vp[i].values()]) for i in ids if i in vc and i in vp}
N = 400
class R_:
    def __init__(s, p_cand):
        s.c9 = dict(a7=True, a8=True, carry_cap=2); s.state = {}; s.data = Data(); s.c9_cfg = None; s.log = []; s.pc = p_cand; s.V = {}
    def say(s, t): s.log.append(t)
    def votes(s, ms):
        out = {}
        for m in ms:
            if m["name"] not in s.V:
                d = (m.get("draw") or {}).get("d", 0)
                base = m["name"].split("~")[0]
                s.V[m["name"]] = votes(N, s.pc if base == "cand" else .5, hash((base, d)) % 100000)
            out[m["name"]] = s.V[m["name"]]
        return out
def draw_of(m, d):
    return dict(m, name=f"{m['name']}~d{d}", draw=dict(of=m["name"], d=d))
H = types.SimpleNamespace(merge4=HB.merge4, paired=paired_gold, gate_scope=lambda m: None, draw_of=draw_of, harm=HB.harm, DRAW_EPOCH=10)

def rec(name):
    return dict(name=name, global_=True, dS=0.0, p_boot=0.5, p_holm=0.5, admissible=False, reason="pool", delta=0.01, C0=1.0, C1=1.0, novelty=0,
                checks=dict(no_harm=True, card_vs_unseen=True, critic=True, fired=True, z=True))

def run(p_cand, rounds=4):
    R = R_(p_cand)
    R.state["c1109_carry"] = {"cand": dict(since=1, reads=[], Ds=[], rows=0, harm=0.0, harm_thr=0.0, sumd=0.0, sumd2=0.0, status="carried")}
    par = dict(name="par")
    m = dict(name="cand", parents=["par"])
    res = []
    for r in range(2, 2 + rounds):
        row = dict(carried=True, carry_round=r, carry_parent=par)
        ok, info = HW.carry_judge(R, H, r, m, row, rec("cand"), [], list(range(N)), {}, {})
        res.append((ok, info))
        if ok or R.state["c1109_carry"]["cand"]["status"] != "carried":
            break
    return R, res

def test_admit_strong():
    R, res = run(.8)
    assert res[-1][0] is True, res[-1][1]
    assert R.state["c1109_carry"]["cand"]["status"].startswith("admitted")
    print("ok   test_admit_strong", [round(x[1]["E"], 1) for x in res])

def test_null_drops():
    R, res = run(.5)
    assert all(not ok for ok, _ in res)
    assert R.state["c1109_carry"]["cand"]["status"].startswith("dropped"), R.state["c1109_carry"]["cand"]["status"]
    print("ok   test_null_drops", R.state["c1109_carry"]["cand"]["status"], [round(x[1]["E"], 2) for x in res])

def test_carry_out_cap():
    R = R_(.5)
    recs = [(dict(name=f"c{k}", parents=["par"]), dict(name=f"c{k}", round=1, slot=k), dict(rec(f"c{k}"))) for k in range(4)]
    tracks = {"c0": dict(D=.04, up=10, down=5), "c1": dict(D=.02, up=8, down=4), "c2": dict(D=.03, up=9, down=3), "c3": dict(D=-.01, up=3, down=5)}
    out = HW.carry_out(R, 1, recs, tracks)
    assert out == ["c0", "c2"], out             # cap 2, by D; c3 has D <= 0
    assert R.state["c1109_carry"]["c0"]["member"]["name"] == "c0"
    print("ok   test_carry_out_cap")

for f in (test_admit_strong, test_null_drops, test_carry_out_cap):
    f()
print("ALL GREEN")
