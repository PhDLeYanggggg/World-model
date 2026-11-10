import numpy as np
import pytest

from src.evaluation.m3w_relative_label_stress import hide_plan, masked, pair_step_terms


def test_equal_support_distinguishes_first_and_last_missing():
    gain = np.broadcast_to(np.arange(12, dtype=float), (3, 12))
    head, tail, random = hide_plan(3, 3, 17)
    for m in (head, tail, random):
        np.testing.assert_array_equal(m.sum(-1), 3)
    assert masked(gain, gain, head)['observed'] == pytest.approx(7.)
    assert masked(gain, gain, tail)['observed'] == pytest.approx(4.)
    for m in (head, tail, random):
        r = masked(gain, gain, m)
        assert np.all(r['lower'] <= gain.mean())
        assert np.all(r['upper'] >= gain.mean())


def test_pair_terms_require_all_agents_and_steps_in_query():
    y = np.zeros((6, 12, 2))
    a = y.copy()
    a[0, :, 0] = 1.
    a[2:, :, 0] = 100.
    valid = np.ones(y.shape[:2], bool)
    valid[3, 11] = False
    y[~valid] = np.nan
    gains, distances, support = pair_step_terms(
        dict(a=a, b=np.zeros_like(a)), y, valid, np.array([2, 3, 1]), {'b_minus_a': ('b', 'a')})
    assert support == {'queries': 1, 'pair_steps': 12}
    np.testing.assert_array_equal(gains['b_minus_a'], np.ones((1, 12)))
    np.testing.assert_array_equal(distances['b_minus_a'], np.ones((1, 12)))
