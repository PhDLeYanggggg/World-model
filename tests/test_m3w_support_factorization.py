import numpy as np
import pytest

from src.world_model.m3w_causal_abstention import fit_support
from src.world_model.m3w_support_factorization import memberships, guards_and_reasons, decisions, independent_verify, partition_ledger


def test_same_source_joint_is_not_separate_marginal_support():
    h = np.array([[1, 1, 0, 0], [1, 1, 0, 0], [1, 0, 0, 0]], bool)
    d = np.array([[0, 0, 1, 1], [1, 1, 0, 0], [0, 1, 1, 0]], bool)
    guards, reasons = guards_and_reasons(np.ones(3, bool), h, d)
    assert reasons.tolist() == [4, 0, 1]
    assert guards['separate'].tolist() == [True, True, False]
    assert guards['joint'].tolist() == [False, True, False]


def test_disagreement_cannot_change_history_projection():
    x = np.ones((64, 3)); states = np.full(64, 2)
    fitted = fit_support(x, states, np.array(['a']*32+['b']*32), np.ones(64, bool))
    h, d = memberships(x, states, fitted); x[:, 2] = 1e6
    hh, dd = memberships(x, states, fitted)
    np.testing.assert_array_equal(h, hh); assert d.all() and not dd.any()
    fitted['boxes'].append(fitted['boxes'][0])
    with pytest.raises(ValueError): memberships(x, states, fitted)


def test_complete_reason_partition_and_no_reactivation():
    h = np.array([[1, 1], [0, 0], [1, 1], [0, 0], [1, 1]], bool)
    d = np.array([[1, 1], [1, 1], [0, 0], [0, 0], [1, 1]], bool)
    guards, reasons = guards_and_reasons(np.array([1, 1, 1, 1, 0], bool), h, d)
    assert reasons.tolist() == [0, 1, 2, 3, 5]
    assert all(not v[-1] for v in guards.values())


def test_scalar_replay_and_current_frame_control_counts():
    rng = np.random.default_rng(17); x = rng.normal(size=(200, 3)); states = np.full(200, 2)
    fitted = fit_support(x, states, np.repeat(['a', 'b', 'c', 'd'], 50), np.ones(200, bool))
    stop = rng.random(200) > .3; query = np.repeat(np.arange(40), 5); ids = np.arange(200); risk = rng.random((200, 2))
    choice, reason = decisions(stop, x, states, fitted, query, risk, ids, 17)
    assert independent_verify(choice, reason, stop, x, states, fitted, query, risk, ids, 17) == 13
    for g in ('history', 'disagreement', 'separate', 'joint'):
        for c in ('risk', 'random'):
            np.testing.assert_array_equal(np.bincount(query[choice[g]], minlength=40), np.bincount(query[choice[g+'_'+c]], minlength=40))


def test_cost_partition_is_additive_and_unknown_not_zero():
    reason = np.array([1, 2, 3, 4, 0, 5, 2])
    n = np.array([4, 1, 5, 1, 2, 4, np.nan], float); d = np.array([2, 3, 2, 2, 2, 2, np.nan], float)
    r = partition_ledger(reason, n, d, np.array(['a']*7), np.ones(7, bool))['a']
    rows = list(r['categories'].values())
    assert sum(v['indexed_rows'] for v in rows) == 7 and sum(v['known_rows'] for v in rows) == 6
    removed = rows[1:5]
    assert sum(v['harm']-v['benefit'] for v in removed) == 2
    assert r['floor_error_sum'] == 13
