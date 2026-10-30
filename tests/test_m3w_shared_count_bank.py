import itertools
from types import SimpleNamespace

import numpy as np
import pytest
from src.world_model import m3w_shared_count_bank as api
from tests.test_m3w_frame_action_allocation import problem, objective


def data():
    p, ok = problem(); ok[0, 6] = True
    p[0, 6] = [5., .005, 1., 1., .005]
    return p, ok, np.array(['a', 'a']), np.array([10, 10]), np.array([4, 3])


def test_both_banks_are_reoptimized_at_same_physical_anchor_count():
    p, ok, rec, frame, ids = data()
    choices, ranks, info = api.allocate(p, ok, rec, frame, ids, seconds=2.)
    np.testing.assert_array_equal(choices['physical_independent'], [0, 0])
    np.testing.assert_array_equal(choices['physical_joint'], [1, 0])
    np.testing.assert_array_equal(choices['neural_joint'], [6, 0])
    assert all((c >= 0).sum() == 2 for c in choices.values())
    assert not (choices['physical_joint'] == 6).any()
    assert objective(p, choices['neural_joint']) > objective(p, choices['physical_joint'])
    assert info['same_count_every_query'] and info['predicted_bank_dominance_every_query']
    assert not info['labels_used'] and not info['observed_risk_guarantee']
    with pytest.raises(TypeError):
        api.allocate(p, ok, rec, frame, ids, seconds=2., future_endpoint=np.zeros((2, 2)))


def test_neural_only_admissible_query_is_not_given_an_invented_physical_action():
    p, ok, rec, frame, ids = data(); ok[:, :6] = False
    c, _, info = api.allocate(p, ok, rec, frame, ids, seconds=2.)
    assert all((v == -1).all() for v in c.values())
    assert info['common_anchor_selected'] == 0 and info['zero_count_queries'] == 1
    assert info['legacy_neural_selected'] == 1
    assert info['empty_is_not_a_safety_pass']


def test_neural_fallback_retains_best_feasible_physical_solution(monkeypatch):
    p, ok, rec, frame, ids = data(); p[0, 6] = [8., .1, 1., 1., .1]
    # Physical unconstrained optimum is feasible; neural expansion needs MILP.
    monkeypatch.setattr(api.base, 'milp', lambda **_: SimpleNamespace(status=1, x=None))
    c, _, info = api.allocate(p, ok, rec, frame, ids, seconds=2.)
    np.testing.assert_array_equal(c['physical_joint'], [1, 0])
    np.testing.assert_array_equal(c['neural_joint'], c['physical_joint'])
    assert info['nonoptimal_queries']['neural_joint'] == 1
    assert info['predicted_bank_dominance_every_query']


def test_exact_banks_match_exhaustive_search_at_common_count():
    rng = np.random.default_rng(330)
    for _ in range(6):
        p = np.zeros((3, 7, 5)); p[..., 2:4] = 1.
        p[..., 0] = rng.uniform(.3, 2., (3, 7))
        p[..., 1] = rng.uniform(.001, .045, (3, 7)); p[..., 4] = .9*p[..., 1]
        p[:, 0, 1] = p[:, 0, 4] = .001
        ok = np.zeros((3, 7), bool); ok[:, [0, 1, 6]] = True
        c, _, info = api.allocate(p, ok, np.array(['a']*3), np.zeros(3, int), np.arange(3), seconds=2.)
        k = (c['physical_independent'] >= 0).sum()
        for name, actions in (('physical_joint', (0, 1)), ('neural_joint', (0, 1, 6))):
            legal = ok.copy()
            if name == 'physical_joint':
                legal[:, 6] = False
            values = [objective(p, np.array(v)) for v in itertools.product((-1, *actions), repeat=3)
                if sum(a >= 0 for a in v) == k and api.base.feasible(p, legal, np.array(v))]
            assert objective(p, c[name]) == pytest.approx(max(values), abs=1e-10)
            assert info['nonoptimal_queries'].get(name, 0) == 0


def test_canonical_trimming_and_allocation_ignore_input_row_order():
    p, ok, rec, frame, ids = data(); p = np.tile(p, (2, 1, 1)); ok = np.tile(ok, (2, 1))
    rec = np.array(['a']*4); frame = np.array([1, 1, 2, 2]); ids = np.array([9, 8, 7, 6])
    direct = api.allocate(p, ok, rec, frame, ids, seconds=2.)
    perm = np.array([3, 0, 2, 1]); inverse = np.argsort(perm)
    other = api.allocate(p[perm], ok[perm], rec[perm], frame[perm], ids[perm], seconds=2.)
    for name in api.ARMS:
        np.testing.assert_array_equal(other[0][name][inverse], direct[0][name])
        np.testing.assert_array_equal(other[1][name][inverse], direct[1][name])


def test_real_interrupted_recovery_is_exact_and_rejects_changed_predictions():
    p, ok, _, _, _ = data(); p = np.tile(p, (3, 1, 1)); ok = np.tile(ok, (3, 1))
    rec, frame, ids = np.array(['a']*6), np.repeat(np.arange(3), 2), np.arange(6)
    saved = []
    def stop(state):
        saved.append(state); raise RuntimeError('planned interruption')
    with pytest.raises(RuntimeError, match='planned'):
        api.allocate(p, ok, rec, frame, ids, seconds=2., checkpoint=stop)
    resumed = api.allocate(p, ok, rec, frame, ids, seconds=2., resume_state=saved[-1])
    direct = api.allocate(p, ok, rec, frame, ids, seconds=2.)
    for name in api.ARMS:
        np.testing.assert_array_equal(resumed[0][name], direct[0][name])
        np.testing.assert_array_equal(resumed[1][name], direct[1][name])
    assert resumed[2] == direct[2]
    bad = p.copy(); bad[0, 0, 0] += .1
    with pytest.raises(ValueError, match='Same frozen'):
        api.allocate(bad, ok, rec, frame, ids, seconds=2., resume_state=saved[-1])
