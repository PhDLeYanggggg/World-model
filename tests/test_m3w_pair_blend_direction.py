import numpy as np

from src.evaluation.m3w_pair_blend_direction import evaluate, minimum_slope, slope_terms


def test_joint_direction_can_help_even_when_either_endpoint_alone_hurts():
    r = np.array([[1., 0.], [0., 0.]])
    d = np.array([[[-.4, -.4], [0., 0.]], [[1., -1.], [0., 0.]]])
    c, m = slope_terms(r, d, np.array([True, False]))
    lo, hi, u = minimum_slope(c, m)
    np.testing.assert_allclose([lo, hi], [-.4, -.4], atol=1e-12)
    np.testing.assert_allclose(u, [.5, .5], atol=1e-12)


def test_shared_direction_can_fail_and_separate_direction_can_improve():
    h = np.array([[[0., 0.]], [[2., 0.]]])
    ref = np.zeros((2, 2, 2))
    prop = ref.copy()
    prop[0, :, 0] = 1.
    prop[1, :, 0] = 2.
    target = ref.copy()
    target[0, :, 0] = 1.
    cv = ref-1.
    graph = lambda _: np.array([[0], [1]])
    s, j = evaluate(h, ref, prop, cv, target, np.ones((2, 2), bool), np.ones(2, bool), graph)
    assert s[0] == 'no_gain_to_tolerance'
    assert j[0] == 'verified_gain'
    np.testing.assert_allclose(j[1], 1., atol=1e-7)


def test_missing_future_is_not_treated_as_zero_target():
    h = np.zeros((2, 1, 2))
    ref = np.zeros((2, 2, 2))
    prop = ref.copy()
    prop[0, :, 0] = 1.
    graph = lambda _: np.array([[0], [1]])
    s, j = evaluate(h, ref, prop, ref-1., np.full_like(ref, np.nan), np.zeros((2, 2), bool),
                    np.array([True, False]), graph)
    assert s[0] == j[0] == 'no_gain_to_tolerance'
