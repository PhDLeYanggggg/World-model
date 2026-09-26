import numpy as np
import pytest

from src.evaluation.m3w_fixed_cap_diagnostic import fixed_cap_accounting, diagnose_bank


def test_exact_nonnegative_error_partition():
    p, y, c = np.array([.2, .4]), np.array([.5, 2.]), np.ones(2)
    r = fixed_cap_accounting(p, y, c, np.ones(2))
    assert r['mse'] == pytest.approx(1.325)
    assert r['projection_floor'] == pytest.approx(.5)
    assert r['distance_to_projection'] == pytest.approx(.225)
    assert r['boundary_cross_term'] == pytest.approx(.6)
    assert r['fraction_above_cap'] == .5


def test_positive_projection_floor_does_not_establish_conditional_bias():
    # This constant predictor is the exact conditional mean for a fair binary future.
    r = fixed_cap_accounting(np.ones(2), np.array([0., 2.]), np.ones(2), np.ones(2))
    assert r['mean_target_minus_cap'] == 0
    assert r['mse'] == 1
    assert r['floor_share_percent'] == 50
    assert not r['conditional_bias_identified']
    assert not r['deployable_oracle']


def test_unknown_labels_require_zero_weight():
    r = fixed_cap_accounting(np.array([.2, .8]), np.array([.5, np.nan]), np.ones(2), np.array([1., 0.]))
    assert r['rows'] == 1
    with pytest.raises(ValueError):
        fixed_cap_accounting(np.array([.2, .8]), np.array([.5, np.nan]), np.ones(2), np.ones(2))


def test_weight_rescaling_and_coordinate_rescaling():
    p, y, c, w = map(np.array, ([.2, .4], [.5, 2.], [1., 1.], [1., 3.]))
    a = fixed_cap_accounting(p, y, c, w)
    b = fixed_cap_accounting(p, y, c, w * 17)
    assert a == b
    d = fixed_cap_accounting(p * 10, y * 10, c * 10, w)
    assert d['mse'] == pytest.approx(100 * a['mse'])
    assert d['floor_share_percent'] == pytest.approx(a['floor_share_percent'])


@pytest.mark.parametrize('p,y,c,w', [
    ([2.], [1.], [1.], [1.]), ([-1.], [1.], [1.], [1.]),
    ([.2], [-1.], [1.], [1.]), ([.2], [1.], [-1.], [1.]),
    ([.2], [1.], [1.], [0.]), ([.2], [1.], [1.], [-1.]),
    ([.2], [np.inf], [1.], [0.]), ([.2, .3], [1.], [1.], [1.]),
])
def test_invalid_inputs_fail_closed(p, y, c, w):
    with pytest.raises(ValueError):
        fixed_cap_accounting(*map(np.array, (p, y, c, w)))


def test_zero_error_has_no_defined_fraction():
    r = fixed_cap_accounting(np.array([0., 1.]), np.array([0., 1.]), np.ones(2), np.ones(2))
    assert r['mse'] == 0
    assert r['floor_share_percent'] is None


def test_banks_exclude_unknown_and_zero_envelope():
    p = np.array([[1., 1., .5, .3], [0., 0., 0., 0.], [1., 1., .5, .3]])
    y = np.array([[1., 2., 1., 2.], [0., 0., 0., 0.], [np.nan] * 4])
    r = diagnose_bank(p, y, np.array([2., 0., 2.]), np.array([1., 0., 0.]))
    assert r['uniform_positive_envelope']['frozen_harm_cap']['rows'] == 1
    assert r['uniform_positive_envelope']['causal_envelope_cap']['projection_floor'] == 0
    assert r['uniform_positive_envelope']['frozen_harm_cap']['projection_floor'] == 1
    assert sum(b['rows'] for b in r['uniform_positive_envelope']['risk_bins']) == 1


def test_partial_unknown_nested_costs_are_rejected():
    p = np.array([[1., 1., .5, .3]])
    with pytest.raises(ValueError):
        diagnose_bank(p, np.array([[1., np.nan, 1., 1.]]), np.ones(1), np.zeros(1))


def test_target_outside_causal_envelope_is_rejected():
    p = np.array([[1., 1., .5, .3]])
    with pytest.raises(ValueError):
        diagnose_bank(p, np.array([[1., 2., 1., 2.]]), np.ones(1), np.ones(1))
