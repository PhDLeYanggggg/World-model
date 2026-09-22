import numpy as np
import pytest

from src.evaluation.m3w_temporal_fit_support import population_summary, moving_zero_support, standardized_support


def test_selected_underprediction_can_coexist_with_global_overprediction():
    r = population_summary(np.array([[0., 8.], [0., 0.], [np.nan, np.nan]]),
                           np.array([[8., 1.], [0., 10.], [2., 1.]]),
                           np.array([10., 12., 4.]), np.array([True, False, True]), np.ones(3))
    assert not r['all']['harm_underpredicted'] and r['selected']['harm_underpredicted']
    assert r['selected']['complete'] == 1 and r['selected']['not_complete'] == 1
    assert r['selected']['harm_ratio'] == 8


def test_weighted_and_unweighted_costs_kept_distinct():
    r = population_summary(np.array([[0., 8.], [0., 0.]]), np.ones((2, 2)),
                           np.ones(2)*10, np.ones(2, bool), np.array([1., 4.]))
    assert r['selected']['realized_harm'] == 4
    assert r['selected']['fit_weighted_harm'] == 1.6


def test_moving_zero_support_counts_draws_without_future_inputs():
    p = np.zeros((3, 8, 2)); p[1:, -1, 0] = 1
    r = moving_zero_support(p, np.array([0., 0., np.nan]), np.array([True, True, False]),
                            np.array([False, True, True]), np.array([2, 0, 0]))
    assert r['complete_exact_zero_moving'] == dict(rows=1, selected=1, sampled_draws=0, unsampled_rows=1)
    assert r['complete_exact_zero_stopped']['rows'] == 1


def test_support_uses_supplied_fit_statistics_not_held_statistics():
    r = standardized_support(np.array([[0., 0.], [100., 0.]]), np.zeros(2), np.ones(2), np.array([False, True]))
    assert r['selected']['max_abs_z_quantiles']['max'] == 100
    assert r['all']['counts_above']['50'] == 1


def test_reject_invalid_weights():
    with pytest.raises(ValueError):
        population_summary(np.zeros((1, 2)), np.ones((1, 2)), np.ones(1), np.ones(1, bool), np.array([-1.]))


def test_reject_complete_unknown_reference():
    with pytest.raises(ValueError):
        moving_zero_support(np.zeros((1, 8, 2)), np.array([np.nan]), np.ones(1, bool), np.ones(1, bool))


def test_reject_invalid_standardization():
    with pytest.raises(ValueError):
        standardized_support(np.ones((2, 1)), np.ones(1), np.zeros(1), np.ones(2, bool))
