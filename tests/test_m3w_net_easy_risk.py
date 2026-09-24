import itertools
from types import SimpleNamespace

import numpy as np
import pytest

from src.world_model import m3w_net_easy_risk as risk


def test_targets_unknown_and_cancellation():
    q = risk.targets(np.array([[2., 0.], [0., 1.], [np.nan, np.nan]]),
                     np.array([3., 1., np.nan]), np.array([4., 2., 1.]),
                     np.array([True, True, False]), 4.)
    np.testing.assert_array_equal(q, [[0., .5, .75, 1.], [.5, 0., .25, 1.], [0., 0., 0., 0.]])
    positive, signed, denominator = risk.moments(q, np.array([4., 2., 1.]), 4.)
    np.testing.assert_array_equal(signed, [-2., 1., 0.])
    assert positive.sum() == 1 and denominator.sum() == 4


def test_negative_risk_can_offset_larger_positive_risk():
    bits, receipt = risk.allocate([1., 9., 2.], [-5., 6., 3.], np.ones(3, bool), 1.)
    np.testing.assert_array_equal(bits, [True, True, False])
    assert receipt['optimal'] and receipt['constraint_pass']


@pytest.mark.parametrize('scale', [1., 1e-9, 1e9])
def test_exhaustive_signed_and_exact_count(scale):
    rng = np.random.default_rng(718)
    for _ in range(18):
        g = rng.uniform(.01, 9, 6) * scale
        q = rng.uniform(-4, 7, 6) * scale
        budget = 1.3 * scale
        for count in [None, 0, 2, 4]:
            feasible = []
            for v in itertools.product([False, True], repeat=6):
                b = np.array(v)
                if (count is None or b.sum() == count) and q[b].sum() <= budget:
                    feasible.append(g[b].sum())
            bits, receipt = risk.allocate(g, q, np.ones(6, bool), budget, count=count)
            if feasible:
                assert receipt['optimal'] and receipt['exact_count_pass'] and receipt['constraint_pass']
                assert g[bits].sum() == pytest.approx(max(feasible))
            else:
                assert not receipt['optimal'] and not bits.any()


def test_solver_original_units_checked(monkeypatch):
    monkeypatch.setattr(risk, 'milp', lambda *a, **k: SimpleNamespace(
        status=0, x=np.ones(2), fun=-2., mip_dual_bound=-2.))
    bits, receipt = risk.allocate([1., 1.], [2e-12, 2e-12], np.ones(2, bool), 1e-12)
    assert not bits.any() and not receipt['optimal']


def test_coherence_and_invalid_inputs():
    with pytest.raises(ValueError):
        risk.moments(np.array([[.7, .5, .2, 1.]]), np.array([1.]), 2.)
    with pytest.raises(ValueError):
        risk.allocate([1.], [np.nan], np.ones(1, bool), 0.)
    with pytest.raises(ValueError):
        risk.allocate([1.], [1.], np.ones(1, bool), 0., count=True)


def test_positive_choice_is_signed_feasible_at_same_count():
    g = np.array([1., 4., 3., 2.]); positive = np.array([2., 3., 1., 4.])
    signed = positive - np.array([3., 4., 0., 1.])
    b, r = risk.allocate(g, positive, np.ones(4, bool), 4.)
    c, t = risk.allocate(g, signed, np.ones(4, bool), 4., count=int(b.sum()))
    assert r['optimal'] and t['optimal'] and b.sum() == c.sum()
    assert g[c].sum() >= g[b].sum()
