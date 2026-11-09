# selfedit 1109c: CPU unit tests for skill_store, meta_level, admit (no data, no model).
import os, sys, json, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import harness
from harness.adapters.skill_store import SkillStore, precheck, parent_check
from harness.adapters.meta_level import MetaLevel, Anchor, valid_policy
from harness.ops import admit as ADMIT

BODY = "## When it applies\nA.\n## What the output must contain\nB.\n## Known failure modes\nC.\n## Read first\nD."
NEW = dict(id="reread", new=True, pattern="commit_early", scope="figure", detector="commits before describing panels", procedure=["describe", "compare", "abstain if unreadable"], body=BODY)


def test_precheck():
    assert precheck("the answer is 3") and precheck("the answer is B") and precheck("on blink questions") and precheck("row lego/1234")
    assert not precheck("Describe the orbit of each object, left to right, before comparing options.")


def test_skill_lifecycle():
    d = tempfile.mkdtemp()
    s = SkillStore(d)
    st, info = s.write(NEW, 1)
    assert st == "probation", (st, info)
    assert s.write(dict(NEW, id="bad", body=BODY + "\nthe answer is 4"), 1)[0] == "quarantined"
    assert [x["front"]["id"] for x in s.match()] == ["reread"]          # quarantined never matched
    st2, i2 = s.write(dict(id="reread", edits=[dict(op="append", content="E. extra line")]), 2)
    assert st2 == "probation" and i2["version"] == 2, (st2, i2)
    tr = s.lifecycle(2, {"reread": 1}, {"reread": 1})
    assert tr and tr[0]["to"] == "active", tr
    assert "skill reread" in SkillStore.block(s.match())


def test_parent_check_rewrite():
    ok, _ = parent_check(BODY, "## When it applies\nX.\n")
    assert not ok


def test_meta_ab():
    st = {}
    M = MetaLevel(st, 12, lambda: 0.01)
    assert valid_policy(dict(program=4, cross=2, w=6), 12)[0] and not valid_policy(dict(program=4, cross=2, w=5), 12)[0]
    assert M.propose(1, dict(program=8, cross=2, w=2))[0]
    side = M.plans(2)
    assert sum(v == "ch" for v in side.values()) == 6
    M.record(2, side, {1: 0.05}); M.record(3, M.plans(3), {3: 0.03})
    assert M.S["incumbent"] == dict(program=8, cross=2, w=2) and not M.S["challenger"], M.S


def test_anchor_guard():
    d = tempfile.mkdtemp()
    os.makedirs(f"{d}/anchor"); os.makedirs(f"{d}/frozen")
    open(f"{d}/anchor/a.py", "w").write("x=1")
    st = {}
    A = Anchor(d, st); A.check("start")
    open(f"{d}/anchor/a.py", "w").write("x=2")
    try:
        A.check("later"); raise AssertionError("anchor change not caught")
    except SystemExit:
        pass


def test_admit_from_files():
    d = tempfile.mkdtemp()
    reg = os.path.join(d, "registry.json")
    open(reg, "w").write(open(os.path.join(os.path.dirname(ADMIT.__file__), "registry.json")).read())
    os.makedirs(f"{d}/s0/P1"); os.makedirs(f"{d}/s0/Q3")
    # amendment 11-09: gate_write on P1 alone (Q3 no longer in its rule); a missing or failed P1 keeps it out
    r = ADMIT.admit(reg, f"{d}/s0")
    assert ADMIT.admitted_ops(r) == [], r["families"]["gate_write"]["why"]
    json.dump({"pass": False}, open(f"{d}/s0/P1/verdict.json", "w"))
    json.dump({"pass": True}, open(f"{d}/s0/Q3/verdict.json", "w"))
    assert ADMIT.admitted_ops(ADMIT.admit(reg, f"{d}/s0")) == []
    json.dump({"pass": True}, open(f"{d}/s0/P1/verdict.json", "w"))
    assert ADMIT.admitted_ops(ADMIT.admit(reg, f"{d}/s0")) == ["gate"]   # re-read at each call: a verdict landing mid-run joins at the next round


if __name__ == "__main__":
    fails = 0
    for k, f in list(globals().items()):
        if k.startswith("test_") and callable(f):
            try:
                f(); print("ok  ", k)
            except Exception as e:
                fails += 1; print("FAIL", k, repr(e))
    print("ALL GREEN" if not fails else f"{fails} FAILED")
    sys.exit(1 if fails else 0)
