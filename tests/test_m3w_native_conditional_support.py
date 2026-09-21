import numpy as np
import pytest
from src.evaluation.m3w_native_conditional_support import rollout_distance, verify_cost_geometry, summarize_group


def test_triangle_bound_and_zero_reference_identity():
    b = np.zeros((3, 12, 2)); c = b.copy(); c[:, :, 0] = [1., 2., 3.][0]
    d = rollout_distance(c, b, np.array([1., 2., 3.]))
    np.testing.assert_array_equal(d, [1., 2., 3.])
    r = verify_cost_geometry(d, np.array([0., 1., np.nan]), np.array([1., 1., np.nan]),
        np.array([0., 0., np.nan]), np.array([True, True, False]))
    assert r['zero_reference_rows'] == 1 and r['all_checks_passed']
    with pytest.raises(ValueError):
        verify_cost_geometry(d, np.zeros(3), d+1, np.zeros(3), np.ones(3, bool))


def test_group_counts_unknown_separately_and_tracks_not_windows():
    mask, full = np.ones(4, bool), np.array([True, True, True, False])
    cv, h = np.array([0., 0., 1., np.nan]), np.array([1., 2., 0., np.nan])
    r = summarize_group(mask, full, cv, np.zeros(4), h, np.array([.001, .005, .5, .3]),
        np.zeros(4), np.ones(4), np.array(['a', 'a', 'b', 'c']), np.array(['s', 's', 't', 't']))
    assert r['positive_event_rows'] == 2 and r['unique_positive_tracks'] == 1
    assert r['incomplete_rows'] == 1 and r['false_negative_score_le_0p01_tracks'] == 1


def test_distance_has_no_future_mask_argument_and_preserves_agreement_zero():
    b = np.zeros((2, 12, 2))
    np.testing.assert_array_equal(rollout_distance(b, b, np.ones(2)), [0., 0.])
    with pytest.raises(ValueError):
        rollout_distance(b[:, :8], b[:, :8], np.ones(2))
