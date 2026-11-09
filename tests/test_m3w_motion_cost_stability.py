import numpy as np
import pytest

from src.world_model import m3w_motion_cost_stability as model
from scripts.audit_m3w_sparse_anchor_risk import risk_floor


def test_contests_are_local_causal_and_radius_comparable():
    q = np.array([1, 2, 2, 1, 2, 3, 3])
    r = np.array([2., 1., 4., 3., 4., 1., 3.])
    p = model.contests(q, r, np.ones(len(r), bool), np.arange(len(r)))
    np.testing.assert_array_equal(p, [[0, 3]])
    p = model.contests(q, r, np.array([1, 0, 1, 1, 1, 1, 1], bool), np.arange(len(r)))
    np.testing.assert_array_equal(p, [[0, 3], [2, 4]])


def test_pessimism_can_change_choice_at_identical_count_and_peak():
    p = np.array([[0, 1], [2, 3]])
    s = model.decide(np.array([3., 1., 1., 3.]), np.array([[5., 1., 1., 5.], [-1., 1., 1., -1.]]), p)
    np.testing.assert_array_equal(s['omit_mean'], [0, 3])
    np.testing.assert_array_equal(s['omit_min'], [1, 2])
    ols = np.zeros((4, 12, 2))
    r = np.array([2., 3., 5., 5.])
    b = np.broadcast_to((r / np.sqrt(2))[:, None, None], ols.shape)
    for selected in s.values():
        z = model.edited_path(ols, b, r, p, selected)
        np.testing.assert_allclose(np.linalg.norm(z[selected], axis=-1).max(1), [2., 5.])
    with pytest.raises(ValueError, match='fixed contest'):
        model.edited_path(ols, b, r, p, np.array([2, 1]))


def test_risk_floor_removes_most_harmful_track_per_query_and_keeps_denominator():
    cv = np.array([1., 2., 3., 4., np.nan])
    ols = np.array([5., 4., 8., 6., np.nan])
    q = np.array([0, 0, 1, 1, 1])
    known = np.isfinite(cv)
    result = risk_floor(ols, cv, known, q, np.ones(len(q), bool))
    assert result == dict(optimistic_harm=4., maximum_reference=10., lower_ratio=.4)
    assert risk_floor(ols, cv, known, q, np.zeros(len(q), bool))['lower_ratio'] is None
