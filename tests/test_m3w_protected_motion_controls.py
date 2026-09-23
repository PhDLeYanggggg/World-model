import numpy as np
import pytest

from src.data_unification.m3w_causal_recordings import BASELINES
from src.evaluation.m3w_protected_motion_controls import (
    ACTIONS, causal_candidate, protected_decisions, match_strict_count, supported_costs)


def test_all_registered_causal_actions_and_exact_layout():
    g = np.arange(2*476, dtype=np.float32).reshape(2, 476)
    assert len(ACTIONS) == 7 and ACTIONS[-1] == 'transformer'
    for name in ACTIONS[:-1]:
        k = BASELINES.index(name)
        p = causal_candidate(g, name)
        np.testing.assert_array_equal(p.reshape(2, 24), g[:, 308+24*k:332+24*k])
        p[:] = 0
        assert g[:, 308+24*k:332+24*k].any()
    with pytest.raises(ValueError):
        causal_candidate(g, 'constant_velocity_causal_fd')


def test_strict_guard_rejects_stationary_zero_gain_and_excess_harm():
    past = np.zeros((6, 8, 2)); past[:, -1, 0] = 1; past[0] = 0
    scores = np.array([[.9, .01], [0, 0], [.1, .1], [.6, .2], [.8, .08], [.8, .081]])
    out = protected_decisions(scores, past, np.array([1, 0, 1, 1, 1, 1]))
    np.testing.assert_array_equal(out['strict_stop'], [False, False, False, False, True, False])
    np.testing.assert_array_equal(out['net_stop'], [False, False, False, True, True, True])
    with pytest.raises(ValueError):
        protected_decisions(scores*10, past, np.ones(6))


def test_matched_counts_use_only_strict_pool_and_stable_tie_ids():
    ids = np.array([9, 3, 5, 1])
    score = np.array([[2., 0], [2, 0], [9, 0], [1, 0]])
    a, b = match_strict_count(score, score, np.array([1, 1, 0, 1], bool),
                             np.array([0, 0, 1, 0], bool), ids)
    np.testing.assert_array_equal(a, [False, True, False, False])
    np.testing.assert_array_equal(b, [False, False, True, False])
    assert a.sum() == b.sum() == 1
    c, d = match_strict_count(score, score, np.zeros(4, bool), b, ids)
    assert not c.any() and not d.any()


def test_partial_outcomes_never_become_training_costs():
    y, cv = supported_costs(np.array([1., 6, 999, np.nan]), np.array([3., 2, 3, np.nan]),
                            np.array([1, 1, 0, 0], bool))
    np.testing.assert_array_equal(y[:2], [[2, 0], [0, 4]])
    assert np.isnan(y[2:]).all() and np.isnan(cv[2:]).all()


def test_policy_and_candidates_are_invariant_to_label_mutation():
    rng = np.random.default_rng(12)
    g = rng.normal(size=(5, 476)).astype(np.float32)
    p = causal_candidate(g, ACTIONS[0])
    score = np.tile([.8, .02], (5, 1))
    before = protected_decisions(score, g[:, :16].reshape(-1, 8, 2), np.ones(5))
    labels = rng.normal(size=(5, 12, 2)); labels[:] = np.nan
    np.testing.assert_array_equal(p, causal_candidate(g, ACTIONS[0]))
    for k, v in before.items():
        np.testing.assert_array_equal(v, protected_decisions(score, g[:, :16].reshape(-1, 8, 2), np.ones(5))[k])
