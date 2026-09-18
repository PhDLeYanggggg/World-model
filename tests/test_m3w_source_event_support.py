import numpy as np
import pytest

from src.evaluation.m3w_source_event_support import (
    past_episode_context, future_change_labels, nonoverlapping_window_count,
    neighbor_support, concentration,
)


def track():
    x = np.zeros((501, 9))
    x[:, 0] = 1
    x[:, 3:5] = 10
    x[:, 5] = np.arange(len(x))
    x[:, 8] = 1
    x[::30, 8] = 0
    x[200:300, [1, 3]] += 2
    return x


def test_plateau_identity_and_later_poison_or_truncation():
    a = track()
    expected = past_episode_context(a, [120, 180, 250, 384])
    np.testing.assert_array_equal(expected['plateau_start_frame'], [0, 0, 200, 300])
    np.testing.assert_array_equal(expected['full_raw_history_static'], [True, True, False, True])
    for q in (120, 180, 250, 384):
        before = past_episode_context(a, [q])
        changed = a.copy()
        changed[changed[:, 5] > q, 1:5] = np.nan
        for alternative in (changed, a[a[:, 5] <= q]):
            after = past_episode_context(alternative, [q])
            for k in before:
                np.testing.assert_array_equal(before[k], after[k])


def test_gaps_lost_rows_and_sample_aliasing_break_continuous_history():
    a = track()
    a[70, [1, 3]] += 1
    assert not past_episode_context(a, [120])['full_raw_history_static'][0]
    a = track()
    a[70, 6] = 1
    assert past_episode_context(a, [120])['plateau_start_frame'][0] == 71
    assert past_episode_context(np.delete(track(), 70, axis=0), [120])['plateau_start_frame'][0] == 71
    with pytest.raises(ValueError):
        past_episode_context(a, [70])


def test_future_labels_are_horizon_bounded_and_separate():
    a = track()
    labels = future_change_labels(a, [50, 120, 180, 250, 480])
    np.testing.assert_array_equal(labels['first_changed_raw_frame'], [-1, 200, 200, 300, -1])
    np.testing.assert_array_equal(labels['future_raw_complete'], [True, True, True, True, False])
    a[200, 6] = 1
    assert not future_change_labels(a, [120])['future_raw_complete'][0]
    assert future_change_labels(a, [120])['first_changed_raw_frame'][0] == 201


def test_disjoint_spans_are_track_scoped_and_inclusive():
    assert nonoverlapping_window_count(['a']*4, [84, 85, 312, 313]) == 2
    assert nonoverlapping_window_count(['a', 'b'], [84, 84]) == 2
    assert nonoverlapping_window_count([], []) == 0


def test_neighbor_availability_uses_past_masks_and_current_time():
    g = np.zeros((2, 476))
    p = g[:, 38:166].reshape(-1, 8, 8, 2)
    t = g[:, 166:230].reshape(-1, 8, 8)
    m = g[:, 230:294].reshape(-1, 8, 8)
    m[0, 0] = 1; t[0, 0] = np.arange(-7, 1); p[0, 0, -1] = 1
    m[0, 1, -1] = 1
    v = neighbor_support(g)
    assert v['current_neighbor_count'].tolist() == [2, 0]
    assert v['moving_neighbor_count'].tolist() == [1, 0]
    assert v['full_eight_neighbor_count'].tolist() == [1, 0]
    t[0, 0, 0] = 1
    with pytest.raises(ValueError): neighbor_support(g)


def test_concentration_is_not_row_count_or_independent_n():
    a = concentration(np.ones(4), ['a', 'a', 'b', 'b'])
    assert a['groups'] == a['inverse_herfindahl'] == 2
    assert a['largest_share'] == .5
    assert concentration([], [])['inverse_herfindahl'] is None
