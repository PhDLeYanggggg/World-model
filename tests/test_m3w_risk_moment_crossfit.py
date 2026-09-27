import numpy as np
import pytest
from src.evaluation.m3w_risk_moment_crossfit import (labels, screen, split_controller,
    score_edges, summarize, flatten_pair)


def test_disjoint_inner_roles():
    fit, held = split_controller(np.arange(8), np.repeat(['a','b','c','d'], 2), 'a', ['p'], ['r'])
    np.testing.assert_array_equal(held, [0, 1])
    np.testing.assert_array_equal(fit, np.arange(2, 8))
    with pytest.raises(ValueError):
        split_controller(np.arange(4), np.array(['a','b','c','d']), 'a', ['b'], ['r'])


def test_unknown_and_zero_cost_labels():
    y = labels([0, 2, np.nan], [1, 1, np.nan])
    np.testing.assert_array_equal(y, [[0, 1], [2, 0], [np.nan, np.nan]])
    with pytest.raises(ValueError): labels([1], [np.nan])


def test_float64_threshold_matches_deployment_convention():
    p = np.array([[7, .14]], np.float32)
    assert not screen(p)[0]
    with pytest.raises(ValueError): screen(p, .05)


def test_denominator_inflation_not_only_harm_underprediction():
    p = np.array([[100, 1], [100, 1]])
    y = np.array([[10, .5], [10, .5]])
    r = summarize(p, y, [10, .5], 10, dict(reference=[100], harm=[1]))
    assert r['screen_rate'] == 1
    assert r['screen_predicted_harm_ratio'] == .01
    assert r['screen_actual_harm_ratio'] == .05
    assert r['screen_harm_actual_over_predicted'] == .5
    assert r['screen_reference_actual_over_predicted'] == .1


def test_zero_reference_not_epsilon_regularized():
    r = summarize([[10, 0]], [[0, 1]], [1, 1], 1, dict(reference=[], harm=[]))
    assert r['screen_actual_harm_ratio'] is None
    assert r['screen_zero_reference_harmed'] == 1
    assert r['screen_positive_harm_mean'] == 1


def test_unknown_not_counted_or_imputed():
    r = summarize([[10, 0], [10, 0]], [[2, 0], [np.nan, np.nan]], [1, 1], 1,
                  dict(reference=[], harm=[]))
    assert r['known_rows'] == 1 and r['unknown_rows'] == 1
    assert r['reference_actual_mean'] == 2


def test_bins_only_depend_on_fitting_scores_weights():
    p = np.array([[1, 0], [2, 1], [3, 2], [4, 3.]])
    edges = score_edges(p, np.ones(4)/4, [.5, .9])
    assert edges == dict(reference=[2., 4.], harm=[1., 3.])
    r = summarize(p, p, p.mean(0), 1, edges)
    assert r['reference_MSE_skill_percent'] == 100
    assert sum(v['rows'] for v in r['reference_bins']) == 4


def test_paired_fit_equal_locality_not_row_weighted():
    r = summarize([[1, .1], [1, .1]], [[1, 0], [2, 1]], [1, 1], 1,
                  dict(reference=[], harm=[]))
    a, b, c = [dict(r) for _ in range(3)]
    a['reference_actual_over_predicted'] = 1
    b['reference_actual_over_predicted'] = 2
    c['reference_actual_over_predicted'] = 6
    result = flatten_pair(r, [a, b, c])
    assert result['fit_reference_actual_over_predicted'] == 3
    assert result['gap_reference_actual_over_predicted'] == -1.5


def test_features_are_same_label_free_parent_schema():
    from src.world_model.m3w_dimensionless_intervention import inference_features
    import inspect
    assert list(inspect.signature(inference_features).parameters) == ['geometry', 'cv', 'candidate']
