import numpy as np
import pytest
from src.world_model.m3w_dual_event_bridge import features, targets, choices, scalar_replay, POLICIES


def test_shared_reference_event_labels_and_unknowns():
    y = targets(np.array([1., 5., 0., np.nan]), np.array([2., 3., 0., np.nan]),
                np.array([4., 1., 0., np.nan]), 2.)
    np.testing.assert_array_equal(y['utility'][:3], [[0, 2], [2, 0], [0, 0]])
    np.testing.assert_array_equal(y['all_risk'][:3], [[2, 2], [3, 0], [0, 0]])
    np.testing.assert_array_equal(y['easy_risk'][:3], [[2, 2], [0, 0], [0, 0]])
    assert all(np.isnan(v[-1]).all() for v in y.values())


def test_two_event_constraint_is_not_all_event_safety():
    u = np.tile([3., 1.], (4, 1)); a = np.tile([100., 1.], (4, 1))
    e = np.array([[1., 1.], [100., 1.], [100., 1.], [100., 1.]])
    moving, env = np.array([True, True, False, True]), np.array([1., 1., 1., 0.])
    np.testing.assert_array_equal(choices(u, a, e, moving, env, arm='all_risk_only'), [1, 1, 0, 0])
    np.testing.assert_array_equal(choices(u, a, e, moving, env, arm='dual_risk'), [0, 1, 0, 0])
    for arm in POLICIES:
        np.testing.assert_array_equal(choices(u, a, e, moving, env, arm=arm),
                                      scalar_replay(u, a, e, moving, env, arm))


def test_features_are_causal_and_zero_envelope_is_exact():
    g = np.zeros((3, 476), np.float32)
    g[:, 16:24] = np.arange(-7, 1)
    cv = np.ones((3, 12, 2)); b = np.zeros((3, 12, 2)); p = b.copy(); p[1] = 2
    x, env = features(g, cv, b, p, np.zeros((3, 4), bool))
    assert x.shape == (3, 383)
    np.testing.assert_array_equal(env[[0, 2]], [0, 0])
    assert env[1] == np.sqrt(8)
    with pytest.raises(ValueError): features(g, cv, b, p, np.zeros((3, 4)))
    with pytest.raises(TypeError): features(g, cv, b, p, np.zeros((3, 4), bool), future=np.ones(3))


def test_unknown_support_cannot_disappear():
    with pytest.raises(ValueError): targets(np.array([np.nan]), np.array([1.]), np.array([2.]), 2.)


def test_invalid_scores_fail_closed():
    for bad in (-1., np.nan, np.inf):
        with pytest.raises(ValueError):
            choices(np.array([[bad, 0.]]), np.ones((1, 2)), np.ones((1, 2)),
                    np.ones(1, bool), np.ones(1), arm='dual_risk')
