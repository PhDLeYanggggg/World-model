from copy import deepcopy
import json

import numpy as np
import pytest

from src.evaluation.m3w_stationary_start_probe import past_features, stationary_label, stationary_run, score_probabilities, FEATURE_NAMES


def inputs():
    t = np.arange(-7, 1, dtype=float)
    n = np.zeros((8, 8, 2))
    n[0, :, 0] = 3 - np.arange(8) * .1
    return {'history_xy': np.zeros((8, 2)), 'history_frame_offsets': t, 'history_velocity': np.zeros((7, 2)),
            'history_mask': np.ones(8, bool), 'neighbor_xy': n, 'neighbor_mask': np.tile([True]+[False]*7, (8, 1)).T,
            'neighbor_frame_offsets': np.tile(t, (8, 1)), 'baseline_rollouts': np.zeros((7, 12, 2)),
            'prediction_frame_offsets': np.arange(1, 13), 'causal_features': np.r_[np.zeros(6), 1., np.zeros(7)]}


def test_geometry_ratios_invariant_to_coordinate_scale_and_rotation():
    x = inputs()
    original = past_features(x)
    x['neighbor_xy'] = x['neighbor_xy'] @ np.array([[0., -1.], [1., 0.]]) * 100
    np.testing.assert_allclose(original, past_features(x), rtol=1e-12, atol=1e-12)
    assert len(original) == len(FEATURE_NAMES)


def test_future_label_only_and_endpoint_not_allowed():
    x = inputs()
    before = past_features(x)
    assert not stationary_label(np.zeros((8, 2)), np.zeros((12, 2)))['changed']
    moving = np.zeros((12, 2)); moving[4:, 0] = 1
    label = stationary_label(np.zeros((8, 2)), moving)
    assert label['changed'] and label['first_change_step'] == 5
    np.testing.assert_array_equal(before, past_features(x))
    x['future_endpoint'] = [1, 1]
    with pytest.raises(ValueError, match='schema'):
        past_features(x)


def test_future_context_and_incomplete_neighbor_excluded():
    x = inputs(); x['neighbor_frame_offsets'][0, -1] = 1
    with pytest.raises(ValueError, match='past'):
        past_features(x)
    x = inputs(); x['neighbor_mask'][0, 0] = False
    assert past_features(x)[1] == 0
    x['neighbor_xy'][0] *= 100
    assert past_features(x)[1] == 0


def test_run_is_episode_not_overlapping_window_count():
    points = np.array([[i, 1, 0 if i < 10 else 1, 0] for i in range(15)], dtype=float)
    a, b = stationary_run(points, 7), stationary_run(points, 8)
    assert a['first_row'] == b['first_row'] == 0
    assert a['whole_run_rows_label_only'] == b['whole_run_rows_label_only'] == 10
    assert a['starts_at_track_entry'] and not a['ends_at_track_exit']
    assert json.loads(json.dumps(a)) == a


def test_prior_scores_and_single_class_honest():
    s = score_probabilities([0, 1, 1], [2/3]*3, prior=2/3)
    assert s['brier_lift_over_prior'] == pytest.approx(0)
    assert s['auroc'] == .5
    s = score_probabilities([0, 0], [0, 0], prior=.1)
    assert s['auroc'] is None and s['average_precision'] is None
    assert s['rank_metric_status'].startswith('single_class')


def test_pooled_context_is_permutation_invariant_and_ignores_absent_slots():
    from src.evaluation.m3w_stationary_pooled_context import pool_context
    x = past_features(inputs())[None]
    before = pool_context(x)
    permuted = x.copy()
    permuted[:, 3:] = x[:, 3:].reshape(-1, 8, 10)[:, ::-1].reshape(-1, 80)
    np.testing.assert_array_equal(pool_context(permuted), before)
    assert before.shape == (1, 13)
    with pytest.raises(ValueError):
        pool_context(x[:, :-1])


def test_group_weighting_does_not_count_repeated_windows_as_more_groups():
    from scripts.summarize_m3w_stationary_probe import group_balanced_brier
    a = group_balanced_brier([1, 0], [.8, .1], .5, [('a',), ('b',)])
    b = group_balanced_brier([1]*10+[0], [.8]*10+[.1], .5, [('a',)]*10+[('b',)])
    assert a['groups'] == b['groups'] == 2
    assert a['brier'] == pytest.approx(b['brier'])
    assert not a['independence_established']
