import numpy as np
import pytest

from src.world_model.m3w_european_protected_motion import (
    motion_candidate_inputs, target_labels, protected_query,
)
from src.world_model.m3w_european_conditional_risk import query_controls


def test_motion_inputs_are_past_only_and_cv_anchored():
    h = np.stack((np.arange(8), np.zeros(8)), 1)[None].astype(float)
    g = np.zeros((1, 476), np.float32)
    g[:, :16] = (h-h[:, -1:]).reshape(1, 16)
    g[:, 16:24] = np.arange(-7, 1)/12
    out = motion_candidate_inputs(g, h, h[:, -1])
    np.testing.assert_allclose(out['b'][0, :, 0], np.arange(1, 13))
    np.testing.assert_allclose(out['p'][0, :, 0], np.cumsum(.97**np.arange(12)))
    assert out['x'].shape == (1, 355) and np.isfinite(out['x']).all()
    g[:, :16] = 0
    stationary = motion_candidate_inputs(g, np.zeros_like(h), np.zeros((1, 2)))
    assert stationary['same'].all()
    assert not stationary['p'].any()
    with pytest.raises(ValueError):
        motion_candidate_inputs(g, h, h[:, -1])


def test_future_labels_are_separate_and_unknown_is_not_zero():
    cv = np.array([0., 1., 4., np.nan])
    candidate = np.array([2., 0., 6., np.nan])
    expected = {
        'utility': [[0, 2], [1, 0], [0, 2], [np.nan, np.nan]],
        'all': [[0, 2], [1, 0], [4, 2], [np.nan, np.nan]],
        'easy': [[0, 0], [1, 0], [0, 0], [np.nan, np.nan]],
    }
    for task, y in expected.items():
        np.testing.assert_equal(target_labels(cv, candidate, 2., task), y)
    with pytest.raises(ValueError):
        target_labels(cv, candidate, 2., 'oracle_inference')
    with pytest.raises(ValueError):
        target_labels(cv, np.nan_to_num(candidate), 2., 'utility')


def query_args(moments):
    n = len(moments)
    baseline = np.zeros((n, 12, 2))
    return dict(utility=np.ones(n), moments=np.asarray(moments, float), moving=np.ones(n, bool),
        current_xy=np.column_stack((np.arange(n)*20., np.zeros(n))), widths=np.ones(n),
        baseline=baseline, candidate=baseline+1, support_available=True,
        budget=.02, pair_weight=.1, radius_widths=3., threshold_widths=.5, seconds=2.)


def test_fixed_budget_control_matches_ordinary_original_case():
    args = query_args([[2., .01], [1., .1], [2., .02]])
    new, old = protected_query(**args), query_controls(**args)
    for arm in ('independent', 'scene_uniform', 'joint', 'unary_exact', 'joint_exact'):
        np.testing.assert_array_equal(new[arm]['switch'], old[arm]['switch'])
        assert new[arm]['solver_optimal']
    assert new['matched'] and new['matched_nonzero']


def test_tiny_denominator_is_optimal_fallback_not_relaxed_risk():
    out = protected_query(**query_args([[1e-21, 1.]]))
    assert out['pruned_infeasible'] == 1
    for arm in ('independent', 'scene_uniform', 'joint', 'unary_exact', 'joint_exact'):
        assert out[arm]['solver_optimal'] and not out[arm]['switch'].any()


def test_absent_source_support_abstains_without_label_filter():
    args = query_args([[1., 0.], [2., 0.]])
    args['support_available'] = False
    out = protected_query(**args)
    assert out['agents'] == 2 and not out['joint']['switch'].any()
    assert out['joint']['predicted_event_ratio'] == 0
