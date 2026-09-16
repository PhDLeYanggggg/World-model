from copy import deepcopy
from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest

from src.world_model.m3w_joint_intervention import (
    InterventionProblem, compare_at_independent_coverage, select_interventions,
)


def problem():
    costs = np.zeros((1, 2, 2))
    costs[0, 1, 0] = 1.
    return InterventionProblem(np.array([.8, -.1, .5]), np.zeros(3), np.ones(3, dtype=bool),
                               np.array([[0, 1]]), costs, .3, .1, 2)


def objective(p, bits):
    pairs = p.pair_cost[np.arange(len(p.edges)), bits[p.edges[:, 0]], bits[p.edges[:, 1]]]
    return -np.mean(bits*p.expected_gain) + p.pair_weight*(pairs.mean() if len(pairs) else 0.)


def test_matched_pair_can_change_identity_without_changing_intervention_count():
    p = problem()
    result = compare_at_independent_coverage(p)
    np.testing.assert_array_equal(result['reference']['switch'], [True, False, True])
    np.testing.assert_array_equal(result['joint_exact']['switch'], [True, True, False])
    assert result['reference_count'] == result['joint_count'] == 2
    assert result['matched'] and result['nonzero_matched']
    assert result['joint_exact']['mean_pair_proxy'] < result['reference']['mean_pair_proxy']
    assert not result['deployment_policy'] and not result['realized_risk_certified']
    assert not result['equal_realized_harm_claim']


def test_forced_count_may_have_worse_predicted_gain_than_floor_and_is_not_deployed():
    p = problem()
    p.expected_gain[:] = [.8, -.1, -2.]
    p.pair_weight = 2.
    ordinary = select_interventions(p, mode='joint')
    result = compare_at_independent_coverage(p)
    assert ordinary['switch'].sum() == 2
    assert result['reference_count'] == result['joint_count'] == 1
    assert result['joint_exact']['mean_predicted_gain'] < 0
    assert result['joint_exact']['objective'] > 0
    assert result['joint_exact']['diagnostic_count_control']
    assert not result['deployment_policy']


@pytest.mark.parametrize('mode', ['independent', 'joint'])
def test_fixed_count_milp_matches_enumeration(mode):
    rng = np.random.default_rng(432)
    for _ in range(20):
        n = 5
        edges = np.array([(i, j) for i in range(n) for j in range(i+1, n)])
        costs = rng.uniform(0, 1, (len(edges), 2, 2))
        costs[:, 0, 0] = 0
        p = InterventionProblem(rng.normal(size=n), rng.uniform(0, .2, n), rng.random(n) > .2,
                                edges, costs, .4, .2, 3)
        for k in range(4):
            values = []
            for bits in product((0, 1), repeat=n):
                b = np.array(bits)
                if (b.sum() == k and np.all(b <= p.supported)
                        and np.mean(b*np.maximum(p.expected_harm, -p.expected_gain)) <= .2):
                    values.append(objective(p, b) if mode == 'joint' else -np.mean(b*p.expected_gain))
            result = select_interventions(p, mode=mode, exact_interventions=k)
            if values:
                assert result['solver_optimal'] and result['exact_count_satisfied']
                value = result['objective'] if mode == 'joint' else -result['mean_predicted_gain']
                assert value == pytest.approx(min(values), abs=1e-8)
            else:
                assert not result['solver_optimal'] and not result['switch'].any()
                assert result['exact_count_satisfied'] is False


@pytest.mark.parametrize('value', [-1, 4, .5, True, np.bool_(False), float('nan')])
def test_invalid_exact_counts_refused(value):
    with pytest.raises(ValueError, match='Exact intervention'):
        select_interventions(problem(), mode='joint', exact_interventions=value)


@pytest.mark.parametrize('mode', ['floor', 'uncontrolled', 'scene_uniform'])
def test_count_constraint_not_silently_ignored_by_other_controls(mode):
    with pytest.raises(ValueError, match='Exact intervention'):
        select_interventions(problem(), mode=mode, exact_interventions=1)


def test_infeasible_fixed_count_returns_floor_but_does_not_claim_matched():
    p = problem()
    p.supported[:] = False
    result = select_interventions(p, mode='joint', exact_interventions=1)
    assert not result['switch'].any() and not result['exact_count_satisfied']
    assert result['predicted_constraints_satisfied']
    assert not result['solver_optimal']


def test_zero_reference_coverage_is_not_claimed_as_a_coupling_benefit():
    p = problem()
    p.expected_gain[:] = -1
    result = compare_at_independent_coverage(p)
    assert result['matched'] and not result['nonzero_matched']
    assert result['status'] == 'matched_zero_not_evidence_of_coupling'


def test_joint_solver_timeout_is_explicit_not_silently_removed(monkeypatch):
    import src.world_model.m3w_joint_intervention as mod
    original = mod.milp
    calls = []
    def solver(**kwargs):
        calls.append(1)
        return original(**kwargs) if len(calls) == 1 else SimpleNamespace(success=False, status=1)
    monkeypatch.setattr(mod, 'milp', solver)
    result = compare_at_independent_coverage(problem())
    assert result['reference_count'] == 2 and result['joint_count'] == 0
    assert not result['matched']
    assert result['status'] == 'not_matched_solver_or_feasibility_failure'


def test_failed_reference_does_not_create_a_fake_zero_zero_match(monkeypatch):
    import src.world_model.m3w_joint_intervention as mod
    monkeypatch.setattr(mod, 'milp', lambda **_: SimpleNamespace(success=False, status=1))
    result = compare_at_independent_coverage(problem())
    assert result['reference_count'] == result['joint_count'] == 0
    assert not result['matched'] and not result['nonzero_matched']


def test_fractional_success_cannot_certify_a_zero_zero_match(monkeypatch):
    import src.world_model.m3w_joint_intervention as mod
    monkeypatch.setattr(mod, 'milp', lambda **kwargs: SimpleNamespace(success=True, x=np.full(len(kwargs['c']), .1)))
    result = compare_at_independent_coverage(problem())
    assert result['reference_count'] == result['joint_count'] == 0
    assert not result['matched']
    assert result['reference']['reason'] == 'solver_solution_invalid_floor'


def test_comparison_does_not_mutate_inputs_or_ordinary_policy():
    p = problem()
    before = deepcopy(p)
    ordinary = select_interventions(p, mode='joint')
    compare_at_independent_coverage(p)
    after = select_interventions(p, mode='joint')
    for key, value in vars(before).items():
        np.testing.assert_array_equal(getattr(p, key), value)
    np.testing.assert_array_equal(ordinary['switch'], after['switch'])
    assert not after['diagnostic_count_control'] and after['exact_count_satisfied'] is None


def test_scene_matched_branch_is_opt_in_and_has_no_future_dependency(tmp_path):
    import torch
    from test_m3w_supervised_intervention import datasets
    from test_m3w_development_evaluation import BaselineCopy, zero_head, POLICY, GEOMETRY
    from src.evaluation.m3w_development_evaluation import decide_scene, score_scene
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        _, train, _ = datasets(tmp_path)
        reader = train.readers[0]
        scene = reader.get_scene_inputs(290, 120)
        kwargs = dict(baseline=train.baseline_name, policy=POLICY, geometry=GEOMETRY,
                      device='cpu', solver_seconds=2.)
        head, model = zero_head(scene, train.baseline_name), BaselineCopy()
        ordinary = decide_scene(scene, model, head, **kwargs)
        assert 'coverage_match' not in ordinary
        before = decide_scene(scene, model, head, include_matched_coverage=True, **kwargs)
        comparison = before['coverage_match']
        assert comparison['matched'] and comparison['reference_count'] == 2
        decisions = {**before, 'arms': {name: comparison[name] for name in ('reference', 'joint_exact')}}
        rows = score_scene(scene, decisions, reader.get_scene_labels(scene), label_policy='available_steps')
        assert len(rows) == 2 and all(row['baseline_fde'] is None for row in rows)
        assert all(set(row['arms']) == {'reference', 'joint_exact'} for row in rows)
        changed = np.asarray(reader.points).copy()
        changed[changed[:, 0] > 290, 2:] = np.nan
        reader.points = changed
        after = decide_scene(reader.get_scene_inputs(290, 120), model, head, include_matched_coverage=True, **kwargs)
        for name in ('reference', 'joint_exact'):
            np.testing.assert_array_equal(comparison[name]['switch'], after['coverage_match'][name]['switch'])
            np.testing.assert_array_equal(comparison[name]['prediction'], after['coverage_match'][name]['prediction'])
        assert after['coverage_match']['reference_count'] == 2
        for mode in ordinary['arms']:
            np.testing.assert_array_equal(ordinary['arms'][mode]['prediction'], before['arms'][mode]['prediction'])
    finally:
        torch.set_num_threads(previous)
