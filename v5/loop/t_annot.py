"""apply_annotations: the failing file of 11-10 23:34 (g003c31_carry.json, ledger with "accepted") applies; a bad file changes nothing."""
import json, os, sys, tempfile, types
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen

ok_all = True


def check(name, ok):
    global ok_all
    ok_all &= bool(ok)
    print(("ok   " if ok else "FAIL ") + name)


def fake(rd):
    st = dict(gens={"3": dict(cands=[dict(cid="g003c31", status="not accepted", accepted=False)])})
    R = types.SimpleNamespace(rd=rd, state=st)
    R.apply_annotations = types.MethodType(gen.Run.apply_annotations, R)
    return R


ci = {"name": "g003c31", "d": 0.0152, "se_used": 0.0077, "p": 0.0245, "bh": False, "lobo": True, "accept": False, "contribution_lb": 0.0053, "K": 9,
      "samples": 4, "se": 0.0077, "by": "hand", "rule": "AM21 LITE carry-in", "note": "x"}
for led, want in ((dict(carry_in=ci, accepted=False), False), (dict(carry_in=ci), False), (dict(carry_in=ci, accepted=True), True)):
    with tempfile.TemporaryDirectory() as rd:
        os.makedirs(os.path.join(rd, "annotate"))
        f = os.path.join(rd, "annotate", "g003c31_carry.json")
        json.dump(dict(gen=3, cid="g003c31", set=dict(carry_in=ci), status_append="re-judged -> not accepted", ledger=led), open(f, "w"))
        R = fake(rd)
        out = R.apply_annotations()
        c = R.state["gens"]["3"]["cands"][0]
        check(f"ledger keys {sorted(led)}: applied, one line, accepted {want}, carry_in set, status appended once",
              len(out) == 1 and out[0][1]["accepted"] is want and out[0][1]["carry_in"] == ci and c["carry_in"] == ci
              and c["status"].count("re-judged") == 1 and not os.path.exists(f + ".failed"))
        R.apply_annotations()
        check("  applying the same file again leaves the status as it is", c["status"].count("re-judged") == 1)
with tempfile.TemporaryDirectory() as rd:
    os.makedirs(os.path.join(rd, "annotate"))
    f = os.path.join(rd, "annotate", "bad.json")
    json.dump(dict(gen=3, cid="g003c31", set=dict(carry_in=ci), status_append="s", ledger=["not", "a", "dict"]), open(f, "w"))
    R = fake(rd)
    out = R.apply_annotations()
    c = R.state["gens"]["3"]["cands"][0]
    check("a malformed ledger: nothing applied, candidate unchanged, file renamed .failed", not out and "carry_in" not in c and os.path.exists(f + ".failed"))
print("ALL OK" if ok_all else "SOME FAILED")
sys.exit(0 if ok_all else 1)
