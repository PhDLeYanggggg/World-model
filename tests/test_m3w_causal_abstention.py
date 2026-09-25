import numpy as np
import pytest

from src.world_model.m3w_causal_abstention import (
    causal_features, fit_support, matched_count, query_ids, support_count, variants, verify_variants)


def test_past_stop_not_future_label():
    h = np.zeros((3, 8, 2)); h[0, :, 0] = np.arange(8); h[1, :, 0] = [0, 1, 2, 3, 4, 5, 6, 6]
    n = np.zeros((3, 12, 2)); d = n.copy()
    x, state = causal_features(h, n, d)
    assert state.tolist() == [2, 1, 0]
    assert x[0, 0] == 1 and x[1, 0] == 0
    assert np.isfinite(x).all()


def test_coordinate_invariance():
    rng = np.random.default_rng(31); h = rng.normal(size=(30, 8, 2)); n = rng.normal(size=(30, 12, 2)); d = n*.8
    rotation = np.array([[0, -1], [1, 0]])
    x, s = causal_features(h, n, d)
    y, t = causal_features(23*h@rotation+6, 23*n@rotation+6, 23*d@rotation+6)
    np.testing.assert_allclose(x, y, atol=1e-12); np.testing.assert_array_equal(s, t)


def test_source_support_and_unknown_exclusion():
    x = np.ones((100, 3)); x[-1] = 1e6
    states = np.full(100, 2); sources = np.array(['a']*50+['b']*50); known = np.arange(100) != 99
    fit = fit_support(x, states, sources, known)
    assert fit['boxes'][1]['count'] == 49
    assert support_count(np.ones((1, 3)), np.array([2]), fit).tolist() == [2]
    assert support_count(np.ones((1, 3)), np.array([1]), fit).tolist() == [0]
    assert support_count(np.full((1, 3), 1e6), np.array([2]), fit).tolist() == [0]


def test_matching_is_frame_local_and_stable():
    q = query_ids(['a', 'a', 'a', 'b', 'b'], [1, 1, 2, 1, 1])
    original = np.ones(5, bool); guard = np.array([1, 0, 0, 0, 1], bool)
    got = matched_count(original, guard, q, np.array([3, 2, 1, 0, 4]), np.arange(5))
    assert got.tolist() == [False, True, False, True, False]
    np.testing.assert_array_equal(np.bincount(q[got], minlength=3), np.bincount(q[guard], minlength=3))
    with pytest.raises(ValueError): matched_count(~original, guard, q, np.arange(5), np.arange(5))


def test_all_variants_scalar_replay():
    rng = np.random.default_rng(31); x = rng.normal(size=(150, 3)); states = rng.integers(0, 3, 150)
    fit = fit_support(x, states, np.array(['a']*75+['b']*75), np.ones(150, bool), minimum_rows=10)
    ids = np.arange(150); original = rng.random(150) > .3; q = np.repeat(np.arange(30), 5)
    risk = rng.random((150, 3))
    out = variants(original, x, states, fit, q, risk, ids, 17)
    assert verify_variants(out, original, x, states, fit, q, risk, ids, 17) == 10
    for g in ('stop', 'support', 'combined'):
        for c in ('risk', 'random'):
            np.testing.assert_array_equal(np.bincount(q[out[g]], minlength=30), np.bincount(q[out[g+'_'+c]], minlength=30))


def test_invalid_history_fails_closed():
    with pytest.raises(ValueError): causal_features(np.zeros((2, 7, 2)), np.zeros((2, 12, 2)), np.zeros((2, 12, 2)))
    h = np.zeros((2, 8, 2)); h[0, 0, 0] = np.nan
    with pytest.raises(ValueError): causal_features(h, np.zeros((2, 12, 2)), np.zeros((2, 12, 2)))
