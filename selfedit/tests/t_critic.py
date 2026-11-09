"""1109c-final: critic input stays small with vector edits (edit_summary) and the critic fails closed (retry once, then unchecked = invalid)."""
import sys, os, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from harness.adapters import hw_round as HW
from rsi import hboost as HB
m = dict(name="x", genome=dict(vision=None, engine=[dict(op="amplify", param="layers.56.mlp.down_proj", side="out", alpha=-1.0, u=list(np.random.default_rng(0).normal(size=5120)))], dirs=[]), ops=["verbosity"])
t = HB.critic_text(m)
assert len(t) < 1000 and "vector [5120]" in t, t
class L_:
    def __init__(s, plan): s.plan, s.k = plan, 0
    def generate_many(s, ph, items):
        p = s.plan[s.k]; s.k += 1
        if p == "fail": raise RuntimeError("proposer job failed")
        return {x["id"]: dict(content=p.get(x["id"], "")) for x in items}
def run(plan):
    R = types.SimpleNamespace(c9=dict(critic=True), say=lambda t: None)
    kids = [(dict(m, program=None), dict(valid=True, slot=k, name=f"s{k}")) for k in range(3)]
    HW.llm = lambda R_, rd: L_(plan)
    H = types.SimpleNamespace(critic_text=HB.critic_text, CRITIC_SYSTEM="", CRITIC_ASK="", critic_verdict=HB.critic_verdict, Bad=HB.Bad, hardcoded_values=lambda p: [])
    HW.precheck = lambda s: []
    return [r["name"] for _, r in HW.critic(R, H, 1, "/tmp", kids)], kids
ok, _ = run(["fail", "fail"]); assert ok == [], ok
ok, _ = run(["fail", {"leak0": "LEAK: no", "leak1": "LEAK: yes", "leak2": "LEAK: no"}]); assert ok == ["s0", "s2"], ok
ok, k = run([{"leak0": "LEAK: no", "leak1": "hmm"}, {"leak1": "LEAK: no", "leak2": "?"}]); assert ok == ["s0", "s1"], ok
assert k[2][1]["leak_critic"] == "unchecked"
print("ALL GREEN")
