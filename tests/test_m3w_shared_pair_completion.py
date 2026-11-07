import numpy as np
import pytest

from src.evaluation.m3w_shared_pair_completion import audit_targets, coupling_bound, shared_bound


def test_swapped_forecasts_cancel_only_for_the_same_target():
    a = np.array([[0., 0.], [4., 0.]])
    b = a[::-1]
    w = np.array([.5, .5])
    keys = np.array([[10, 20, 30], [10, 20, 30]])
    assert shared_bound(keys, a, b, w) == (4., 0., 2, 1)
    keys[1, -1] += 1
    assert shared_bound(keys, a, b, w) == (4., 4., 0, 0)


def test_weighted_coupling_bounds_shared_target_for_unequal_query_weights():
    a = np.array([[0., 0.], [4., 0.], [5., 3.]])
    b = a[[1, 0, 2]]
    w = np.array([.1, .3, .4])
    bound = coupling_bound(a, b, w)
    assert bound == pytest.approx(.8)
    y = np.random.default_rng(4).normal(0., 10., (1000, 2))
    gain = (np.linalg.norm(a-y[:, None], axis=2)-np.linalg.norm(b-y[:, None], axis=2))@w
    assert np.max(np.abs(gain)) <= bound+1e-12


def test_repeated_labels_must_match_recording_agent_and_absolute_frame():
    r = np.array([1, 1, 2])
    a = np.array([8, 8, 8])
    f = np.array([5, 6, 6])
    y = np.zeros((3, 2, 2))
    v = np.ones((3, 2), dtype=bool)
    y[2] = 100.
    assert audit_targets(r, a, f, y, v) == 1
    y[1, 0, 1] = 1.
    with pytest.raises(ValueError, match='position'):
        audit_targets(r, a, f, y, v)
    v[1, 0] = False
    with pytest.raises(ValueError, match='mask'):
        audit_targets(r, a, f, y, v)
