"""amendment 2: HW.rejudge rebuilds p1109c r1's stage-2 candidates from their stored proposals with the same ids, asks score+SELECT only. CPU, read-only."""
import sys, os, json, copy
N = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [N + "/rsi", N + "/mm", N]
import hboost as H, res_fuse
from harness.adapters import hw_round as HW
W = "/data/fd/mingdazhang/selfedit/rsi/mm/hb/p1109c"
H.set_run(W); H.HB = W
pre0 = json.load(open(f"{W}/prereg.json")); am = json.load(open(f"{W}/amendment.json"))
data = H.Data(None, am["iid"], am.get("ood"), am.get("exclude"))
S = json.load(open(f"{W}/state.json"))
R = H.Run.__new__(H.Run)
R.work, R.data, R.a = W, data, dict(pre0)
R.state = copy.deepcopy(S); R.state["c1109_rejudge"] = ["r1s8_7638cc", "r1s7_5f3ea2", "r1s10_a0ac21"]
R.say = print
R.fuser, R.fz_t = res_fuse, {}
R.fusion = res_fuse.mode_of(R.a)
R.narrow_rule = bool(R.a.get("narrow_pool_rows")); R.plateau_on = bool(R.a.get("plateau")); R.F = 1
asked = []
class EvStub:
    def evaluate_many(self, reqs, *a, **k): asked.extend(reqs)
    def wait(self, *a, **k): return None
R.ev = EvStub()
R.c128_init()
H.REG.update({m["name"]: m for m in S["members"]})
by = {m["name"]: m for m in S["members"]}
out = HW.rejudge(R, H, 2, S["members"], by)
names = [m["name"] for m, _ in out]
ok = names == ["r1s8_7638cc", "r1s7_5f3ea2", "r1s10_a0ac21"]
roles = {data.role.get(i) for _, rows in asked for i in rows}
print(("PASS" if ok else "FAIL"), "rebuilt", names, [m.get("ops") for m, _ in out], [(m["genome"].get("vision") or {}).get("patch") for m, _ in out])
print(("PASS" if roles <= {"score", "select"} else "FAIL"), "asked roles", roles, "rows", len(asked[0][1]) if asked else 0)
print(("PASS" if "c1109_rejudge" not in R.state else "FAIL"), "once")
