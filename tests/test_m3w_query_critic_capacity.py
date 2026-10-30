import numpy as np
import pytest
from src.evaluation import m3w_query_critic_capacity as api


def test_weighted_triangle_projection_matches_exhaustive_grid():
    target = np.array([[-5., 3.], [2., .2], [.2, .8], [-1., -2.], [.8, .1]])
    weight = np.broadcast_to([2., 5.], target.shape)
    p = api.project_triangle(target, np.ones(len(target)), weight)
    assert (p >= 0).all() and (p[:, 1] <= p[:, 0]).all() and (p <= 1).all()
    grid = np.array([(a, b) for a in np.linspace(0, 1, 201) for b in np.linspace(0, a, 201)])
    for t, w, point in zip(target, weight, p):
        analytic = ((point-t)**2*w).sum()
        assert analytic <= np.min(((grid-t)**2*w).sum(1))+1e-10
    np.testing.assert_array_equal(p[-1], target[-1])


def fixture():
    p = np.broadcast_to([.8, .1, 1., .5, .01], (4, 7, 5)).copy()
    y = np.broadcast_to([0., .5, 1., .5, .3], (4, 7, 5)).copy()
    y[1] = np.nan
    return p, np.ones((4, 7)), y, np.ones(4), np.array([0, 0, -1, 0]),


def test_unknown_selected_and_empty_queries_are_not_zero_risk_targets():
    p, env, y, scales, c = fixture()
    result = api.diagnose(p, env, y, scales, np.ones((4, 7), bool), c,
        [np.array([0, 1]), np.array([2]), np.array([3])], np.ones(5))
    assert result['queries'] == 3 and result['complete_selected_queries'] == 1
    assert result['unknown_selected_queries'] == 1 and result['empty_selected_queries'] == 1
    assert result['known_legal_actions'] == 21
    assert result['point_harm_above_protected_cap_actions'] == 21
    assert result['query_target_above_all_harm_cap'] == 1
    assert result['query_projection_lower_MSE_sum'] > 0
    assert result['point_projection_lower_MSE_sum'] > 0
    assert result['label_aware_diagnostic_only'] and not result['new_policy_scored']
    assert not result['conditional_mean_misspecification_proved']


def test_geometry_oracle_does_not_claim_causal_prediction_and_rejects_partial_moment_labels():
    p, env, y, scales, c = fixture(); y[1, 0, 0] = 0
    with pytest.raises(ValueError):
        api.diagnose(p, env, y, scales, np.ones((4, 7), bool), c,
            [np.array([0, 1]), np.array([2, 3])], np.ones(5))
