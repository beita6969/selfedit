"""Amendment 9: plan() gives knob_min slots to the grid knobs (tokens, verbosity) out of program slots; the menu holds them."""
import sys, os, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from harness.adapters import hw_round as HW

class Meta:
    def plans(self, r): return {k: ("inc" if k % 2 else "ch") for k in range(12)}
    def policy(self, s): return dict(program=4, cross=2, w=6) if s == "inc" else dict(program=5, cross=2, w=5)
def mk(a9):
    R = types.SimpleNamespace(c9=dict(a9=a9, knob_ops=("tokens", "verbosity"), knob_min=2), c9_meta=Meta(), c9_wops=["gate", "patch"], a=dict(slots=12), state={}, log=[])
    R.say = R.log.append
    return R
for a9 in (False, True):
    R = mk(a9)
    out = HW.plan(R, 4, np.random.default_rng(1), False)
    c = {o: out.count(o) for o in set(out)}
    if a9:
        assert c.get("tokens") == 1 and c.get("verbosity") == 1, c
    else:
        assert "tokens" not in c and "verbosity" not in c, c
    print("ok   a9", a9, c)
print("ALL GREEN")
