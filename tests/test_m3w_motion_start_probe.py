import numpy as np
import pytest

from src.evaluation.m3w_motion_start_probe import (
    probe_features, join_rows, group_brier, paired_agent_interval,
)


def test_feature_ablation_is_exact_and_quality_is_a_matched_control():
    n = np.zeros((3, 83)); q = np.ones((3, 7, 5)); m = np.ones((3, 7, 10))
    assert [probe_features(n, m, q, a).shape[1] for a in
            ('neighbors', 'quality', 'magnitude', 'directed')] == [13, 48, 76, 118]
    np.testing.assert_array_equal(probe_features(n, m, q, 'quality'),
                                  probe_features(n, m*999, q, 'quality'))
    changed = m.copy(); changed[:, :, :6] = 999
    np.testing.assert_array_equal(probe_features(n, m, q, 'magnitude'),
                                  probe_features(n, changed, q, 'magnitude'))
    with pytest.raises(ValueError):
        probe_features(n, m, q*2, 'directed')


def rows():
    all_rows = [dict(recording='eth', agent=i, frame=8, fold=0, scene='eth') for i in range(2)]
    selected = [dict(recording_id='eth', agent_id=i, frame_id=8, fit_fold=0,
                     physical_scene='eth', data_role='fit') for i in range(2)]
    return selected, all_rows, np.zeros((2, 476)), np.zeros((2, 12, 2))


def test_join_rejects_duplicates_role_changes_missing_rows_and_target_mismatch():
    selected, full, g, y = rows()
    np.testing.assert_array_equal(join_rows(selected, full, g, y, [0, 0]), [0, 1])
    for bad in (selected[:1], selected[:1]*2):
        with pytest.raises(ValueError):
            join_rows(bad, full, g, y, [0]*len(bad))
    with pytest.raises(ValueError):
        join_rows(selected, full, g, y, [1, 0])
    selected[0]['data_role'] = 'confirmation'
    with pytest.raises(ValueError):
        join_rows(selected, full, g, y, [0, 0])


def test_future_mutation_changes_supervision_not_features():
    selected, full, g, y = rows()
    n = np.zeros((2, 83)); m = np.ones((2, 7, 10)); q = np.ones((2, 7, 5))
    before = probe_features(n, m, q, 'directed')
    y[0, -1] = [2, -3]
    join_rows(selected, full, g, y, [1, 0])
    np.testing.assert_array_equal(before, probe_features(n, m, q, 'directed'))


def test_group_metrics_do_not_count_repeated_windows_as_new_agents():
    y = np.array([0, 1]); p = np.array([.1, .3]); ref = np.array([.5, .5]); keys = np.array(['a', 'b'])
    score = group_brier(y, p, ref, keys)
    repeated = np.array([0, 0, 0, 1])
    assert group_brier(y[repeated], p[repeated], ref[repeated], keys[repeated]) == score
    ci = paired_agent_interval(y, p, ref, keys)
    assert ci == paired_agent_interval(y[repeated], p[repeated], ref[repeated], keys[repeated])
    assert ci['agents'] == 2 and not ci['independent_confirmation']


def test_seed_interval_averages_loss_not_ensemble_probability():
    ci = paired_agent_interval([0, 0], [[0, 1], [1, 0]], [[.5, .5], [.5, .5]], ['a', 'b'])
    assert ci['agent_balanced_lift'] == -.25
