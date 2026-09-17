import numpy as np
import pytest

from src.world_model.m3w_sdd_state_support import (
    control_provenance, future_event_labels, nonoverlap_count,
    past_features, past_window_indices, track_support, validate_track,
)


def track(xy, frames=None):
    xy = np.asarray(xy, float)
    n = len(xy)
    a = np.zeros((n, 9))
    a[:, 0] = 5
    a[:, 1:3], a[:, 3:5] = xy-1, xy+1
    a[:, 5] = np.arange(n) if frames is None else frames
    return a


def at_query(a, q=7, stride=1):
    queries, histories = past_window_indices(a, stride)
    select = queries == q
    assert select.sum() == 1
    return future_event_labels(a, queries[select], histories[select], stride)


def test_start_stop_turn_and_no_change_known_labels():
    start = track(np.column_stack((np.r_[np.zeros(8), np.arange(1, 13)], np.zeros(20))))
    assert at_query(start)['events']['exact_static_to_movement'][0]
    stop = track(np.column_stack((np.r_[np.arange(8), np.full(12, 7)], np.zeros(20))))
    assert at_query(stop)['events']['moving_to_stop'][0]
    turn = track(np.r_[np.column_stack((np.arange(8), np.zeros(8))),
                      np.column_stack((np.full(12, 7), np.arange(1, 13)))])
    assert at_query(turn)['events']['turn_45_to_135'][0]
    straight = track(np.column_stack((np.arange(20), np.zeros(20))))
    assert not any(v[0] for v in at_query(straight)['events'].values())


def test_reverse_is_separate_from_turn():
    a = track(np.column_stack((np.r_[np.arange(8), 7-np.arange(1, 13)], np.zeros(20))))
    e = at_query(a)['events']
    assert e['reverse_at_least_135'][0] and not e['turn_45_to_135'][0]


def test_eligibility_and_past_features_do_not_use_future_labels():
    a = track(np.column_stack((np.arange(20), np.zeros(20))))
    b = a[:8].copy()
    qa, ia = past_window_indices(a, 1)
    qb, ib = past_window_indices(b, 1)
    assert qb.tolist() == [7]
    for name, value in past_features(a, ia[qa == 7]).items():
        np.testing.assert_array_equal(value, past_features(b, ib)[name])
    labels = at_query(b)
    assert not labels['future_complete_nonlost'][0]
    assert not any(v[0] for v in labels['events'].values())


def test_missing_or_lost_future_changes_labels_not_past_eligibility():
    a = track(np.column_stack((np.r_[np.zeros(8), np.arange(1, 13)], np.zeros(20))))
    a[14, 6] = 1
    assert 7 in past_window_indices(a, 1)[0]
    assert not at_query(a)['future_complete_nonlost'][0]
    assert not at_query(np.delete(a, 14, axis=0))['future_complete_nonlost'][0]


def test_frame_grid_global_and_exact_no_gap_interpolation():
    a = track(np.column_stack((np.arange(60), np.zeros(60))))
    q, _ = past_window_indices(a, 3)
    assert q[0] == 21 and np.all(q % 3 == 0)
    b = np.delete(a, 18, axis=0)
    assert 21 not in past_window_indices(b, 3)[0]


def test_label_scale_invariance_without_claiming_metric():
    a = track(np.column_stack((np.r_[np.zeros(8), np.arange(1, 13)], np.zeros(20))))
    b = a.copy()
    b[:, 1:5] = b[:, 1:5]*25+987
    ea, eb = at_query(a)['events'], at_query(b)['events']
    for key in ea:
        np.testing.assert_array_equal(ea[key], eb[key])


def test_interpolation_provenance_bracketing_and_future_control():
    a = track(np.column_stack((np.arange(20), np.zeros(20))))
    a[1:19, 8] = 1
    p = control_provenance(a)
    assert np.all(p['max_box_interpolation_error'] == 0)
    assert p['next_control_row'][7] == 19 and p['previous_control_row'][7] == 0
    s = track_support(a, 1, p)
    assert s['histories_with_post_query_controls'] == 12
    assert s['histories_with_at_least_two_controls'] == 0
    a[-1, 8] = 1
    p = control_provenance(a)
    assert not p['bracketed'][7]
    assert track_support(a, 1, p)['histories_with_unbracketed_generated_rows'] > 0


def test_occlusion_is_reported_separately_not_hiding_events():
    a = track(np.column_stack((np.r_[np.zeros(8), np.arange(1, 13)], np.zeros(20))))
    a[12, 7] = 1
    s = track_support(a, 1)
    assert s['exact_static_to_movement_windows'] == 1
    assert s['exact_static_to_movement_unoccluded_windows'] == 0


def test_nonoverlap_does_not_count_overlapping_queries_as_independent():
    assert nonoverlap_count([7, 8, 9, 26, 27, 47], 1) == 3
    assert nonoverlap_count([], 12) == 0


def test_invalid_source_or_zero_scale_rejected_or_explicit():
    a = track(np.zeros((20, 2)))
    with pytest.raises(ValueError):
        validate_track(a[::-1])
    with pytest.raises(ValueError):
        past_window_indices(a, 0)
    a[:, 1:5] = 0
    s = track_support(a, 1)
    assert s['past_scale_invalid'] == 13
    assert s['exact_static_to_movement_windows'] == 0


def test_control_span_includes_unsampled_past_controls_but_not_later_controls():
    from scripts.analyze_m3w_sdd_state_support import control_span_counts
    controls = np.arange(0, 101, 10)
    np.testing.assert_array_equal(control_span_counts(controls, [84], 12), [9])
    # Only frames 0 and 60 are controls in the sampled 0,12,...,84 history.
    assert np.isin(np.arange(0, 85, 12), controls).sum() == 2
    np.testing.assert_array_equal(control_span_counts(controls, [19], 1), [0])
