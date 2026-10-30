import inspect
import itertools
from types import SimpleNamespace

import numpy as np
import pytest
from src.world_model import m3w_frame_action_allocation as api
from src.world_model.m3w_multiaction_cost import select_actions


def problem():
    p = np.zeros((2, 7, 5))
    p[..., 2:4] = 1.
    ok = np.zeros((2, 7), bool); ok[:, 0] = True; ok[0, 1] = True
    p[:, 0, 0] = 1.; p[:, 0, 1] = .001; p[:, 0, 4] = .001
    p[0, 1] = [4., .03, 1., 1., .03]
    return p, ok


def objective(p, choice):
    at = np.flatnonzero(choice >= 0)
    return float(np.sum(p[at, choice[at], 0]-p[at, choice[at], 1]))


def test_joint_budget_changes_action_at_identical_count_without_labels():
    p, ok = problem()
    parent, _ = select_actions(p, ok, arm='physical_neural')
    choice, info = api.solve_frame(p, ok, parent, exact_count=2, seconds=2.)
    np.testing.assert_array_equal(parent, [0, 0])
    np.testing.assert_array_equal(choice, [1, 0])
    assert info['optimal'] and info['accepted'] and info['selected'] == 2
    assert objective(p, choice) > objective(p, parent)
    assert 'target' not in inspect.signature(api.solve_frame).parameters
    with pytest.raises(TypeError):
        api.solve_frame(p, ok, parent, exact_count=2, seconds=2., future_endpoint=np.zeros((2, 2)))


def test_solver_matches_exhaustive_multi_choice_budget_and_cardinality():
    rng = np.random.default_rng(291)
    for _ in range(8):
        p = np.zeros((3, 7, 5)); p[..., 2:4] = 1.
        p[..., 0] = rng.uniform(.2, 2., (3, 7))
        p[..., 1] = rng.uniform(.001, .05, (3, 7)); p[..., 4] = p[..., 1]*.9
        p[:, 0, 1] = .001; p[:, 0, 4] = .001
        ok = np.zeros((3, 7), bool); ok[:, :3] = True
        parent, _ = select_actions(p, ok, arm='physical_neural')
        for count in (None, int((parent >= 0).sum())):
            candidates = []
            for values in itertools.product(range(-1, 3), repeat=3):
                c = np.array(values)
                if (count is None or (c >= 0).sum() == count) and api.feasible(p, ok, c):
                    candidates.append(objective(p, c))
            choice, info = api.solve_frame(p, ok, parent, exact_count=count, seconds=2.)
            assert info['optimal'] and api.feasible(p, ok, choice)
            assert objective(p, choice) == pytest.approx(max(candidates), abs=1e-10)


def test_timeout_and_numerical_rejection_retain_original_count_and_policy(monkeypatch):
    p, ok = problem(); p[1, 0, 1] = .019; p[1, 0, 4] = .019
    parent, _ = select_actions(p, ok, arm='physical_neural')
    monkeypatch.setattr(api, 'milp', lambda **_: SimpleNamespace(status=1, x=np.ones(3)))
    c, info = api.solve_frame(p, ok, parent, exact_count=2, seconds=2.)
    np.testing.assert_array_equal(c, parent)
    assert not info['optimal'] and not info['accepted'] and info['reason'] == 'solver_status_1_parent_fallback'
    # A numerically accepted but actually over-budget solution must not be deployed.
    monkeypatch.setattr(api, 'milp', lambda **_: SimpleNamespace(status=0, x=np.array([0., 1., 1.])))
    c, info = api.solve_frame(p, ok, parent, exact_count=2, seconds=2.)
    np.testing.assert_array_equal(c, parent)
    assert info['reason'] == 'postcheck_failed_parent_fallback'


def test_common_unit_rescaling_preserves_decisions_and_tiny_easy_budget():
    p, ok = problem(); parent, _ = select_actions(p, ok, arm='physical_neural')
    for scale in (1e-12, 1., 1e12):
        c, info = api.solve_frame(p*scale, ok, parent, exact_count=2, seconds=2.)
        np.testing.assert_array_equal(c, [1, 0]); assert info['optimal']
    p[..., 3:5] *= 1e-40
    c, _ = api.solve_frame(p, ok, parent, exact_count=2, seconds=2.)
    np.testing.assert_array_equal(c, [1, 0])


def test_query_groups_keep_record_frame_identity_and_canonical_id_order():
    records = np.array(['b', 'a', 'a', 'b']); frames = np.array([1, 2, 2, 2]); ids = np.array([9, 7, 6, 8])
    groups = api.query_groups(records, frames, ids)
    assert [v.tolist() for v in groups] == [[2, 1], [0], [3]]
    with pytest.raises(ValueError):
        api.query_groups(records, frames, np.array([9, 7, 6, 9]))
    with pytest.raises(ValueError):
        api.query_groups(records, np.array([1, 2, np.nan, 2]), ids)


def test_full_controls_preserve_original_independent_and_never_pool_frames():
    p, ok = problem(); p = np.tile(p, (2, 1, 1)); ok = np.tile(ok, (2, 1))
    recs = np.array(['a']*4); frames = np.array([1, 1, 2, 3]); ids = np.array([8, 7, 6, 5])
    choices, ranks, info = api.allocate(p, ok, recs, frames, ids, seconds=2.)
    expected, scores = select_actions(p, ok, arm='physical_neural')
    np.testing.assert_array_equal(choices['independent'], expected)
    np.testing.assert_array_equal(ranks['independent'], scores)
    assert info['queries'] == 3 and info['multiagent_queries'] == 1
    assert (choices['joint_matched'] >= 0).sum() == (expected >= 0).sum()
    assert choices['joint_matched'][2] == expected[2]
    assert choices['joint_matched'][3] == expected[3]
    assert not info['labels_used'] and not info['pair_interactions_optimized']


def test_invalid_or_fake_actions_are_rejected_and_empty_support_is_not_optimal_gain():
    p, ok = problem(); parent, _ = select_actions(p, ok, arm='physical_neural')
    c, info = api.solve_frame(p, np.zeros_like(ok), np.full(2, -1), exact_count=0, seconds=2.)
    np.testing.assert_array_equal(c, [-1, -1]); assert info['reason'] == 'zero_count_exact'
    with pytest.raises(ValueError):
        api.solve_frame(p, ok.astype(float), parent, exact_count=2, seconds=2.)
    with pytest.raises(ValueError):
        api.solve_frame(p*np.nan, ok, parent, exact_count=2, seconds=2.)
    with pytest.raises(ValueError):
        api.solve_frame(p, ok, np.array([6, 6]), exact_count=2, seconds=2.)


def test_interrupted_real_solver_recovery_is_exact_and_rejects_changed_predictions():
    p, ok = problem(); p = np.tile(p, (3, 1, 1)); ok = np.tile(ok, (3, 1))
    rec = np.array(['a']*6); frames = np.repeat(np.arange(3), 2); ids = np.arange(6)
    saved = []
    def stop(state):
        saved.append(state)
        raise RuntimeError('interrupt after atomic inference checkpoint')
    with pytest.raises(RuntimeError, match='interrupt'):
        api.allocate(p, ok, rec, frames, ids, seconds=2., checkpoint=stop)
    resumed = api.allocate(p, ok, rec, frames, ids, seconds=2., resume_state=saved[-1])
    direct = api.allocate(p, ok, rec, frames, ids, seconds=2.)
    for key in api.ARMS:
        np.testing.assert_array_equal(resumed[0][key], direct[0][key])
        np.testing.assert_array_equal(resumed[1][key], direct[1][key])
    assert resumed[2] == direct[2]
    changed = p.copy(); changed[0, 0, 0] += .1
    with pytest.raises(ValueError, match='Same frozen'):
        api.allocate(changed, ok, rec, frames, ids, seconds=2., resume_state=saved[-1])
