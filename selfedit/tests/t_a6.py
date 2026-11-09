# selfedit 1109c amendment 6 unit tests (CPU, no data): null rule = candidate rule (a6_null), dominance void (beats_null, a6_void), and the size of the
# dominance void under a bias shared by candidate and null (the case the void rule exists for).
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import harness
from harness.adapters import member_cand as MC
from harness.anchor.config import RRSIConfig
from harness.anchor.selection import Candidate

CFG = RRSIConfig(beta0=0.1, beta1=10.0, w_s=0.0, w_c=15.0, w_n=0.5)


def fake_rec(name, d_acc, d_ce, clusters, p_boot, tok1, tok0=2500.0, delta=0.054, guards=(), alpha=0.05, q=0.05):
    """A stage-2 record as MC.stage2 builds it (the fields decide_holm / a6_null / beats_null read). S1 - S0 = mean(d_acc)."""
    ids = [f"r{k}" for k in range(len(d_acc))]
    right0 = np.full(len(d_acc), 0.6)
    right1 = right0 + np.asarray(d_acc, float)
    ev1, ev0 = MC.ev_of(name, right1, ids, tok1), MC.ev_of("incumbent", right0, ids, tok0)
    cand = Candidate(name, [], ev=ev1)
    rd = dict(ids=ids, d_ce=np.asarray(d_ce, float), d_acc=np.asarray(d_acc, float), clusters=list(clusters))
    from harness.anchor.selection import judge
    g = list(guards)
    dec = judge(cand, ev0, ev0.S, delta, CFG, {}, g)
    return dict(name=name, dS=ev1.S - ev0.S, delta=delta, weight=False, within_band=(ev1.S - ev0.S) <= delta, alpha=alpha, q=q, p_boot=p_boot, lb=0.0,
                admissible=bool(dec.admissible), reason=dec.reason, _rd=rd, _judge=(cand, ev0, ev0.S, delta, CFG, {}, g))


def shift_rows(n, dS, rng):
    """n rows of accuracy change in {-1, 0, 1} with mean exactly dS (k up rows more than down)."""
    k = int(round(dS * n))
    d = np.zeros(n); d[:20 + k] = 1.0; d[20 + k:40 + k] = -1.0
    rng.shuffle(d)
    return d


def test_a6_null_pilot_and_6pn_cases():
    """Pilot r2 stock~d4 (p .0455, dS +.0165 within delta .054, 2.7 % cheaper): admissible before A6 on the shaped branch, and under A6 on the gain branch.
    cal 6pn r1 stock~d4 (p .011, 1.1 % dearer): NOT admissible before A6 (cost rule), admissible under A6. A cheaper null at p .20: admissible before A6
    only if its pool line passed; under A6 not admissible (pool_holm)."""
    rng = np.random.default_rng(0)
    n = 423; cl = np.repeat(np.arange(212), 2)[:n]
    d = shift_rows(n, 0.0165, rng)
    pil = fake_rec("stock~d4", d, d, cl, 0.0455, tok1=2500 * (1 - 0.0272))
    assert pil["admissible"] and "within delta" in pil["reason"], pil["reason"]
    MC.a6_null(pil)
    assert pil["admissible_pre_a6"] and pil["admissible"] and pil["branch"] == "gain", pil["reason"]
    d2 = shift_rows(n, 0.0071, rng)
    six = fake_rec("stock~d4", d2, d2, cl, 0.011, tok1=2500 * 1.0106)
    assert not six["admissible"] and "shaped" in six["reason"], six["reason"]
    MC.a6_null(six)
    assert not six["admissible_pre_a6"] and six["admissible"], six["reason"]
    weak = fake_rec("stock~d6", d, d, cl, 0.20, tok1=2400)
    MC.a6_null(weak)
    assert not weak["admissible"] and "pool_holm" in weak["reason"], weak["reason"]


def test_beats_null_and_void_order():
    rng = np.random.default_rng(1)
    n = 400; cl = np.repeat(np.arange(200), 2)
    noise = lambda: rng.normal(0, 0.05, n)
    null = fake_rec("stock~d4", shift_rows(n, 0.015, rng), 0.004 + noise(), cl, 0.04, tok1=2400)
    strong = fake_rec("cand_strong", shift_rows(n, 0.06, rng), 0.03 + noise(), cl, 0.001, tok1=2500)
    weak = fake_rec("cand_weak", shift_rows(n, 0.015, rng), 0.004 + noise(), cl, 0.03, tok1=2500)
    t = MC.beats_null(strong, null)
    assert t["ok"] and t["rows"] == n and t["dce"] > 0, t
    assert not MC.beats_null(weak, null)["ok"]
    null["admissible"] = True
    v, k, _ = MC.a6_void([strong, weak], [null])
    assert k == ["cand_strong"] and v == ["cand_weak"], (k, v)
    v, k, _ = MC.a6_void([weak, strong], [null])      # strong was read against weak (accepted first), which is void: strong is void too
    assert k == [] and v == ["cand_weak", "cand_strong"], (k, v)
    null["admissible"] = False
    v, k, _ = MC.a6_void([weak], [null])              # no admissible null: nothing void
    assert v == [] and k == ["cand_weak"]


def test_dominance_size_under_shared_bias():
    """H0 for the candidate: candidate and null are exchangeable draws with the same bias b (the fresh-draw ensembling gain). P(candidate beats the
    null) must be <= alpha (+ MC error), whatever b: the bias cancels in the paired difference."""
    rng = np.random.default_rng(2)
    n = 400; pics = 200; cl = np.repeat(np.arange(pics), 2)
    for b in (0.0, 0.02):
        hits = 0; S = 300
        for s in range(S):
            pic = rng.normal(0, 0.03, pics)[cl]          # a picture effect shared by both reads (same rows, same reference pool content)
            dc = b + pic + rng.normal(0, 0.06, n); dn = b + pic + rng.normal(0, 0.06, n)
            ac = np.sign(dc) * (rng.random(n) < 0.15); an = np.sign(dn) * (rng.random(n) < 0.15)
            c = fake_rec("c", ac, dc, cl, 0.01, tok1=2500); x = fake_rec("x", an, dn, cl, 0.01, tok1=2500)
            hits += MC.beats_null(c, x, seed=s)["ok"]
        rate = hits / S
        print(f"dominance size at shared bias {b:+.2f}: {rate:.4f} ({S} sims, alpha .05)")
        assert rate <= 0.05 + 2 * np.sqrt(0.05 * 0.95 / S), (b, rate)


def test_a6p_null_path():
    """A6': pilot r2 stock~d4 (dS +.0165 within delta .054, admitted on the shaped branch) is a cost-path null; a null with dS above delta is gain-path."""
    rng = np.random.default_rng(0)
    n = 423; cl = np.repeat(np.arange(212), 2)[:n]
    d = shift_rows(n, 0.0165, rng)
    pil = fake_rec("stock~d4", d, d, cl, 0.0455, tok1=2500 * (1 - 0.0272))
    assert pil["admissible"] and MC.null_path(pil) == "cost"
    g = fake_rec("stock~d6", shift_rows(n, 0.07, rng), d, cl, 0.001, tok1=2500)
    assert MC.null_path(g) == "gain"


def test_a6_off_is_old_rule():
    assert MC.A6[0] is False and MC.A6P[0] is False    # default off: hw_round / cal behave as amendments 1-5


if __name__ == "__main__":
    fails = 0
    for k, f in list(globals().items()):
        if k.startswith("test_") and callable(f):
            try:
                f(); print("ok  ", k)
            except Exception as e:
                import traceback; traceback.print_exc()
                fails += 1; print("FAIL", k, repr(e))
    print(f"{'ALL GREEN' if not fails else str(fails) + ' FAILED'}")
    sys.exit(1 if fails else 0)
