"""Amendment 11: stage-1 screen (fired >= fmin, zx > 0, net > 0 when s1_net) and the early A7 draw: the early request is exactly request_nulls'
request for the candidate's pair (same member, same rows), it is named _se (LOW), and stage 1 does not wait on it (pending / sent restored)."""
import sys, os, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from harness.adapters import hw_round as HW
from rsi import hboost as HB

R = types.SimpleNamespace(c9=dict(s1_net=True, fired_min=3, a5=True, null_pairs=((4, 8), (6, 9))))
assert HW.screen_ok(R, 7, 3, 3.0, 2.0) and HW.screen_ok(R, 3, 3, 1.0, .1)
assert not HW.screen_ok(R, 2, 3, 3.0, 2.0) and not HW.screen_ok(R, 5, 3, 0.0, 1.0) and not HW.screen_ok(R, 5, 3, 2.0, -.1) and not HW.screen_ok(R, 5, 3, 2.0, None)
R.c9["s1_net"] = False
assert HW.screen_ok(R, 5, 3, 0.0, 1.0)
print("ok   screen")

class Data:
    def of(self, k): return {"score": ["b/1", "b/2", "b/3"], "select": ["b/4"]}[k]
    def cluster(self, i): return i
asked = []
class Ev:
    def __init__(s): s.pending, s.sent, s.early = [dict(name="hbx_b0")], {"o0": 1}, False
    def evaluate_many(s, reqs):
        assert s.early
        for m, rows in reqs:
            asked.append((m["name"], list(rows)))
            s.pending.append(dict(name="hbx_se1")); s.sent["o1"] = (m, rows)
R = types.SimpleNamespace(c9=dict(s1_net=True, fired_min=3, a5=True, null_pairs=((4, 8), (6, 9))), data=Data(), ev=Ev(), say=print,
                          votes=lambda ms: {m["name"]: {} for m in ms}, fz=lambda *a, **k: types.SimpleNamespace(tab=None), fuser=types.SimpleNamespace(fuse=None), fusion="ds",
                          no_id=lambda rows, why: rows)
H = types.SimpleNamespace(pool_cols=lambda fu: None, fired_rows=lambda tab, v, pa, Q: Q, vs_pool=lambda tab, v, pa, F: {i: 1 for i in F},
                          zx_stats=lambda *a: dict(zx=1.5), draw_of=HB.draw_of)
m = dict(name="r2s1_aa", mid="aa" * 7, genome=dict(vision=None, engine=[], dirs=[]), program=None, parents=["stock"])
row = {}
HW.early_draw(R, H, 2, [dict(name="stock")], m, row, lambda m_: ["b/1", "b/2", "b/3"])
want = HW.pair_of(R, H, m)
assert asked == [(want["name"], sorted(set(Data().of("score")) | set(Data().of("select"))))], asked
assert R.ev.pending == [dict(name="hbx_b0")] and R.ev.sent == {"o0": 1} and not R.ev.early, (R.ev.pending, R.ev.sent)
assert row["c1109_early"]["asked"] and m["_early"]
HW.early_draw(R, H, 2, [dict(name="stock")], m, row, lambda m_: ["b/1"])
assert len(asked) == 1                      # once per candidate
# the later request_nulls asks the same (member, rows): Ev.evaluate_many then finds the item's .job mark and waits on that line (no new line)
reqs = []
R.ev = types.SimpleNamespace(evaluate_many=lambda w: reqs.extend(w))
R.state = {}; R.c9["null_draws"] = (4, 6)
HW.request_nulls(R, types.SimpleNamespace(draw_of=HB.draw_of, DRAW_EPOCH=HB.DRAW_EPOCH), [], [(m, {})])
assert (want["name"], asked[0][1]) in [(x["name"], list(rows)) for x, rows in reqs], reqs
print("ok   early draw = later request, LOW, not waited")
print("ALL GREEN")
