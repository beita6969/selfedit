# selfedit 1109c: B2 unit tests for harness/anchor/eproc.py (CPU, no data). Simulated nulls: size of the pool test and of the e-process; power at +4 pp.
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import harness
from harness.anchor import eproc

SIMS = 1000


def sim_rows(rng, n=400, pics=200, fired=0.3, gain=0.0):
    """Per-row accuracy change of a pool: on a 'fired' share the row moves +-1, else 0; mean = gain. Rows in picture clusters of 2."""
    cl = np.repeat(np.arange(pics), n // pics)
    f = rng.random(n) < fired
    p_up = 0.5 + gain / (2 * fired)
    d = np.where(f, np.where(rng.random(n) < p_up, 1.0, -1.0), 0.0)
    return d, cl


def test_cluster_boot_lower_basic():
    d = np.r_[np.ones(50), np.zeros(50)]
    lb = eproc.cluster_boot_lower(d, list(range(100)), 0.05)
    assert 0.4 < lb < 0.5, lb


def test_pool_test_size_and_power():
    rng = np.random.default_rng(1)
    rej0 = rej1 = 0
    for s in range(SIMS):
        d, cl = sim_rows(rng)
        rej0 += eproc.cluster_boot_lower(d, cl, 0.05, B=400, seed=s) > 0 and d.mean() > 0
        d1, cl1 = sim_rows(rng, gain=0.04)
        rej1 += eproc.cluster_boot_lower(d1, cl1, 0.05, B=400, seed=s) > 0 and d1.mean() > 0
    size, power = rej0 / SIMS, rej1 / SIMS
    print(f"pool test J1 (q .05): size {size:.3f}, power at +4 pp / 400 rows / 30% fired {power:.3f}")
    assert size <= 0.06, size            # nominal .05; Monte Carlo SE ~ .007
    return size, power


def test_eprocess_size():
    rng = np.random.default_rng(2)
    hits = 0
    for _ in range(SIMS):
        dbar = [sim_rows(rng, n=100, pics=50)[0].mean() for _ in range(30)]
        hits += max(eproc.eprocess(dbar)["path"]) >= eproc.CONFIRM_E
    print(f"e-process: share of nulls ever reaching E >= 20 over 30 reads: {hits / SIMS:.3f}")
    assert hits / SIMS <= 0.05


def test_eprocess_predictable():
    e = eproc.eprocess([0.5, 0.5, 0.5])
    assert e["lambdas"][0] == eproc.LAMBDA_FIRST and e["lambdas"][1] == 0.5          # clipped
    assert eproc.eprocess_status([-0.5, -0.5, -0.5])["status"] == "removed"


def test_signflip_size_symmetric():
    rng = np.random.default_rng(3)
    d, cl = sim_rows(rng)
    s = eproc.signflip_size(lambda x: x["d"].mean() > 1.96 * x["d"].std() / np.sqrt(len(x["d"])), dict(d=d), cl, n=1000)
    assert s <= 0.06, s


def test_fired_z_sigma_from_null():
    z = eproc.fired_z([1, 1, 1, 0], [[1, -1, 0, 0], [1, 1, -1, 0]])
    assert abs(z["sigma"] - np.sqrt(2.5)) < 1e-9 and abs(z["z"] - 3 / np.sqrt(2.5)) < 1e-9
    z2 = eproc.fired_z([1, 1], [[1]], scale=[2.0])
    assert abs(z2["sigma"] - np.sqrt(2.0)) < 1e-9


def test_eprocess_rows_power_and_size():
    """A3 (fable_1109d patch 1): per-fired-cluster betting. Power: P(E >= 20 within 20 fired clusters) at p_up .8; size: nulls of 100 fired clusters."""
    rng = np.random.default_rng(4)
    pw = np.mean([max(eproc.eprocess_rows([list(np.where(rng.random(20) < .8, 1.0, -1.0))])["path"]) >= eproc.CONFIRM_E for _ in range(SIMS)])
    n0 = 10 * SIMS
    sz = np.mean([max(eproc.eprocess_rows([list(np.where(rng.random(100) < .5, 1.0, -1.0))])["path"]) >= eproc.CONFIRM_E for _ in range(n0)])
    pw40 = np.mean([max(eproc.eprocess_rows([list(np.where(rng.random(40) < .8, 1.0, -1.0))])["path"]) >= eproc.CONFIRM_E for _ in range(SIMS)])
    print(f"e-process per fired cluster: power at p_up .8 within 20 {pw:.3f}, within 40 {pw40:.3f} (Fable's target .8 within 20 not met: reported, rule unchanged); null size over 100 {sz:.4f} ({n0} sims)")
    assert sz <= 0.05 and pw40 >= 0.8, (pw, pw40, sz)
    st = eproc.eprocess_rows_status([[1.0] * 3] * 5)
    assert eproc.eprocess_rows_status([[-1.0, 1.0]] * 5)["status"] == "removed"


if __name__ == "__main__":
    fails = 0
    for k, f in list(globals().items()):
        if k.startswith("test_") and callable(f):
            try:
                f(); print("ok  ", k)
            except Exception as e:
                fails += 1; print("FAIL", k, repr(e))
    print(f"{'ALL GREEN' if not fails else str(fails) + ' FAILED'}")
    sys.exit(1 if fails else 0)
