import numpy as np
import pytest

from src.evaluation.m3w_forecast_cost_bounds import (
    disagreement, partial_gain_bounds, bounded_fractions, energy_concentration,
)


def test_missing_endpoint_bounds_are_sharp_without_imputation():
    baseline = np.zeros((1, 3, 2))
    candidate = np.array([[[3., 4.], [0., 2.], [3., 0.]]])
    target = np.array([[[3., 4.], [np.nan, np.nan], [np.nan, np.nan]]])
    valid = np.array([[True, False, False]])
    a = partial_gain_bounds(candidate, baseline, target, valid, np.ones(1))
    np.testing.assert_allclose(a['lower'], 0)
    np.testing.assert_allclose(a['upper'], 10/3)
    for fill, name in ((baseline, 'lower'), (candidate, 'upper')):
        y = np.where(valid[..., None], target, fill)
        gain = (np.linalg.norm(baseline-y, axis=-1)-np.linalg.norm(candidate-y, axis=-1)).mean(1)
        np.testing.assert_allclose(gain, a[name])
    assert not a['exact_CV_still_possible'][0]


def test_unknown_is_not_zero_harm_and_candidate_agreement_is_structural():
    b = np.zeros((2, 2, 2))
    p = b.copy()
    p[0, :, 0] = 2
    a = partial_gain_bounds(p, b, np.full_like(p, np.nan), np.zeros((2, 2), bool), np.ones(2))
    np.testing.assert_array_equal(a['lower'], [-2., 0.])
    np.testing.assert_array_equal(a['upper'], [2., 0.])
    np.testing.assert_array_equal(a['exact_CV_harm_upper'], [2., 0.])


def test_exact_zero_rule_has_no_epsilon():
    b = np.zeros((2, 2, 2))
    p = np.ones_like(b)
    y = b.copy()
    y[1, 0, 0] = 1e-15
    a = partial_gain_bounds(p, b, y, np.ones((2, 2), bool), np.ones(2))
    np.testing.assert_array_equal(a['exact_CV_still_possible'], [True, False])
    np.testing.assert_array_equal(a['lower'], a['upper'])


def test_random_complete_costs_obey_full_grid_bound():
    rng = np.random.default_rng(319)
    p, b, y = rng.normal(size=(3, 400, 12, 2))
    s = rng.uniform(.01, 100, 400)
    a = partial_gain_bounds(p, b, y, np.ones((400, 12), bool), s)
    costs = np.column_stack((np.maximum(a['lower'], 0), np.maximum(-a['lower'], 0)))
    z = bounded_fractions(costs, disagreement(p, b, s).mean(1), np.ones(400, bool))
    assert np.all(z >= 0) and np.all(z.sum(1) <= 1+1e-12)


def test_partial_costs_are_not_full_grid_targets():
    z = bounded_fractions(np.array([[1., 0.], [100., 0.]]), np.ones(2), np.array([True, False]))
    np.testing.assert_array_equal(z[0], [1., 0.])
    assert np.isnan(z[1]).all()
    with pytest.raises(ValueError):
        bounded_fractions(np.array([[2., 0.]]), np.ones(1), np.ones(1, bool))


def test_energy_zero_and_tail():
    assert energy_concentration(np.zeros(100))['top_fraction_energy_share'] is None
    assert energy_concentration(np.r_[np.zeros(99), 2])['top_fraction_energy_share'] == 1
