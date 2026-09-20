from dataclasses import replace
from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest

from src.world_model.m3w_joint_intervention import InterventionProblem, proximity_cost_table
from src.world_model.m3w_interaction_controls import (
    compare_interaction_controls, decompose_pair_objective, geometry_aware_independent, solve_control,
)


def additive_fixture():
    return InterventionProblem(np.array([.8, .7]), np.zeros(2), np.ones(2, bool),
        np.array([[0, 1]]), np.array([[[0., .125], [.875, 1.]]]), 1., .1, 1)


def test_joint_can_beat_naive_independent_with_no_nonadditive_interaction():
    result = compare_interaction_controls(additive_fixture())
    arms = result['controls']
    np.testing.assert_array_equal(arms['independent']['switch'], [True, False])
    np.testing.assert_array_equal(arms['unary_geometry']['switch'], [False, True])
    np.testing.assert_array_equal(arms['joint']['switch'], [False, True])
    assert arms['joint']['full_objective'] < arms['independent']['full_objective']
    assert result['predicted_full_objective_advantage'] == 0
    assert result['absolute_product_sum_bound'] == 0
    assert result['matched'] and not result['deployment_policy']


def test_real_proximity_fixture_requires_product_not_just_unary_penalties():
    baseline = np.array([[[0., 0.]], [[3., 0.]], [[9., 0.]]])
    candidate = np.array([[[1.5, 0.]], [[1.5, 0.]], [[9., 1.]]])
    edges = np.array([[0, 1]])
    table = proximity_cost_table(baseline, candidate, edges, distance_threshold=.75)
    p = InterventionProblem(np.array([1., .9, .8]), np.zeros(3), np.ones(3, bool), edges, table, 1., .1, 2)
    r = compare_interaction_controls(p)
    np.testing.assert_array_equal(r['controls']['unary_geometry']['switch'], [True, True, False])
    np.testing.assert_array_equal(r['controls']['joint']['switch'], [True, False, True])
    assert r['predicted_full_objective_advantage'] > 0 and r['potential_nonadditive_edges_at_count'] == 1


def test_geometry_objective_never_redefines_original_harm_budget():
    p = additive_fixture()
    # Geometry makes both adjusted net scores negative; that is not predicted ADE harm.
    p.pair_cost[:] = [[[0., 2.], [3., 5.]]]
    p.max_mean_predicted_harm = 0.
    result = geometry_aware_independent(p, exact_interventions=1)
    assert result['solver_optimal'] and result['switch'].sum() == 1
    assert result['mean_predicted_harm'] == 0 and result['unary_objective'] > 0
    assert not result['deployment_policy']


def test_negative_gain_harm_floor_is_preserved():
    p = additive_fixture()
    p.expected_gain[:] = [-2., -1.]
    r = geometry_aware_independent(p, exact_interventions=1)
    assert not r['solver_optimal'] and not r['exact_count_satisfied']
    assert not r['switch'].any()


def test_pair_decomposition_and_all_optima_match_exhaustive_enumeration():
    rng = np.random.default_rng(7231)
    for _ in range(30):
        n = 6
        edges = np.array(list((i, j) for i in range(n) for j in range(i+1, n)))
        table = rng.random((len(edges), 2, 2)); table[:, 0, 0] = 0
        p = InterventionProblem(rng.normal(.2, .3, n), rng.random(n)*.15, rng.random(n) > .15,
                               edges, table, .6, .08, 3)
        d = decompose_pair_objective(p)
        r = compare_interaction_controls(p)
        assert r['matched']
        objectives = []
        for bits in product((0, 1), repeat=n):
            x = np.asarray(bits)
            direct = (-np.mean(x*p.expected_gain)+p.pair_weight*
                      np.mean(table[np.arange(len(edges)), x[edges[:, 0]], x[edges[:, 1]]]))
            unary = d['unary_objective']@x
            expanded = unary+d['product_objective']@(x[edges[:, 0]]*x[edges[:, 1]])
            assert expanded == pytest.approx(direct, abs=1e-12)
            if (x.sum() == r['reference_count'] and not np.any(x > p.supported)
                    and np.mean(x*np.maximum(p.expected_harm, -p.expected_gain)) <= p.max_mean_predicted_harm+1e-12):
                objectives.append((unary, direct))
        assert r['controls']['unary_geometry']['unary_objective'] == pytest.approx(min(o[0] for o in objectives), abs=1e-8)
        assert r['controls']['joint']['full_objective'] == pytest.approx(min(o[1] for o in objectives), abs=1e-8)


@pytest.mark.parametrize('change', ['zero_weight','no_edges','zero_count','one_count'])
def test_zero_product_or_small_counts_do_not_prove_coupling(change):
    p = additive_fixture()
    p.pair_cost[0, 1, 1] = 3
    if change == 'zero_weight':
        p.pair_weight = 0.
    elif change == 'no_edges':
        p.edges = np.empty((0, 2), int); p.pair_cost = np.empty((0, 2, 2))
    elif change == 'zero_count':
        p.max_interventions = 0
    r = compare_interaction_controls(p)
    assert r['potential_nonadditive_edges_at_count'] == 0
    assert r['predicted_full_objective_advantage'] == pytest.approx(0)


@pytest.mark.parametrize('malformed', [np.array([.5,.5]), np.array([1.,1.]), np.array([np.nan,0]), None])
def test_invalid_solver_output_falls_back_and_is_not_matched(monkeypatch, malformed):
    import src.world_model.m3w_interaction_controls as m
    monkeypatch.setattr(m, 'milp', lambda **kw:SimpleNamespace(success=True, x=malformed))
    r = geometry_aware_independent(additive_fixture(), exact_interventions=1)
    assert not r['solver_optimal'] and not r['switch'].any()
    assert not r['exact_count_satisfied']


def test_timeout_is_not_a_valid_zero_control(monkeypatch):
    import src.world_model.m3w_interaction_controls as m
    monkeypatch.setattr(m, 'milp', lambda **kw:SimpleNamespace(success=False))
    r = compare_interaction_controls(additive_fixture())
    assert not r['matched'] and r['predicted_full_objective_advantage'] is None
    assert r['controls']['unary_geometry']['reason'] == 'solver_not_optimal_floor'


def test_micro_cost_regression_matches_exhaustive_optimum():
    rng = np.random.default_rng(0)
    n = 6
    edges = np.array([(i,j) for i in range(n) for j in range(i+1,n)])
    table = rng.random((len(edges),2,2))*1e-5; table[:,0,0] = 0
    p = InterventionProblem(np.full(n,.05),np.full(n,.01),np.ones(n,bool),edges,table,1.,.1,3)
    bits = np.array(list(product((0,1),repeat=n)))
    bits = bits[bits.sum(1) == 3]
    cost = -(bits*p.expected_gain).mean(1)+table[np.arange(len(edges)),bits[:,edges[:,0]],bits[:,edges[:,1]]].mean(1)
    r = solve_control(p, objective_kind='joint', exact_interventions=3)
    assert r['solver_optimal'] and r['numerical']['numerical_certificate_pass']
    assert r['numerical']['objective_scale'] > 1
    assert r['full_objective'] == pytest.approx(cost.min(), abs=1e-11)


@pytest.mark.parametrize('certificate', ['gap', 'missing', 'false_primal', 'dual_above_primal'])
def test_success_flag_without_valid_original_unit_certificate_is_rejected(monkeypatch, certificate):
    import src.world_model.m3w_interaction_controls as m
    def fabricated(**kw):
        bits = np.array([0.,1.])
        f = float(kw['c']@bits)
        # Original units differ by 1e-4, regardless of numerical scaling.
        delta = abs(kw['c'][1]/(-.225))*1e-4
        out = dict(success=True, x=bits, fun=f, mip_dual_bound=f)
        if certificate == 'gap': out['mip_dual_bound'] -= delta
        elif certificate == 'false_primal': out['fun'] += delta
        elif certificate == 'dual_above_primal': out['mip_dual_bound'] += delta
        else: out.pop('mip_dual_bound')
        return SimpleNamespace(**out)
    monkeypatch.setattr(m, 'milp', fabricated)
    r = geometry_aware_independent(additive_fixture(), exact_interventions=1)
    assert not r['solver_optimal'] and not r['switch'].any()
    assert r['reason'] == 'solver_certificate_failed_floor'


def test_continuous_product_inconsistency_is_rejected(monkeypatch):
    import src.world_model.m3w_interaction_controls as m
    monkeypatch.setattr(m, 'milp', lambda **kw:SimpleNamespace(success=True,x=np.array([0.,1.,.5])))
    r = solve_control(additive_fixture(), objective_kind='joint', exact_interventions=1)
    assert not r['solver_optimal'] and r['reason'] == 'solver_solution_invalid_floor'


@pytest.mark.parametrize('count', [True, -1, 2, 1.5])
def test_invalid_exact_count_rejected(count):
    with pytest.raises(ValueError, match='Exact count'):
        geometry_aware_independent(additive_fixture(), exact_interventions=count)


def test_inputs_are_not_mutated():
    p = additive_fixture()
    before = {k:v.copy() for k,v in vars(p).items() if isinstance(v, np.ndarray)}
    compare_interaction_controls(p)
    for k,v in before.items():
        np.testing.assert_array_equal(v, getattr(p, k))


def test_agent_relabeling_preserves_objective_without_claiming_tie_identity():
    p = additive_fixture()
    swapped = replace(p, expected_gain=p.expected_gain[::-1], expected_harm=p.expected_harm[::-1],
                      pair_cost=p.pair_cost.transpose(0,2,1))
    a, b = compare_interaction_controls(p), compare_interaction_controls(swapped)
    for arm in a['controls']:
        assert a['controls'][arm]['full_objective'] == pytest.approx(b['controls'][arm]['full_objective'])
    assert not a['tie_break_identity_invariance_claim']


def test_opt_in_scene_adapter_keeps_labels_out_and_forecasts_fixed():
    from copy import deepcopy
    from src.evaluation.m3w_interaction_control_evaluation import attach_interaction_controls
    from src.evaluation.m3w_development_evaluation import score_scene
    ids = [1,2,3]
    scene = dict(recording_id='synthetic', physical_scene='constructed', frame_id=0, horizon_raw=1,
        agents=[dict(agent_id=i, inputs={'prediction_frame_offsets':np.array([1.])},
            coordinate_transform={'origin_xy':np.zeros(2),'rotation':np.eye(2),'scale':1.}) for i in ids])
    b = np.array([[[0.,0.]],[[3.,0.]],[[9.,0.]]])
    c = np.array([[[1.5,0.]],[[1.5,0.]],[[9.,1.]]])
    decision = dict(agent_ids=ids, baseline=b, candidate=c, predicted_gain=np.array([1.,.9,.8]),
        predicted_harm=np.zeros(3), supported=np.ones(3,bool), arms={})
    policy = dict(pair_weight=1.,max_mean_predicted_harm=.1,max_intervention_fraction=2/3,
                  min_predicted_gain=0.,max_agent_predicted_harm=1.)
    kwargs = dict(policy=policy, geometry={'graph_radius':20.,'proximity_threshold':.75})
    original = deepcopy(decision)
    result = attach_interaction_controls(scene, decision, **kwargs)
    assert decision['arms'] == {} and 'interaction_controls' not in decision
    np.testing.assert_array_equal(original['candidate'], decision['candidate'])
    assert result['interaction_controls']['matched']
    labels = [dict(agent_id=i, future_frame_ids=np.array([1.]),future_xy_dataset_local=np.zeros((1,2)),
                   future_label_mask=np.array([i != 2])) for i in ids]
    rows = score_scene(scene, result, labels, label_policy='complete_requested_path')
    assert len(rows) == 3 and rows[1]['baseline_ade'] is None
    assert result['interaction_controls']['reference_count'] == 2
    with pytest.raises(ValueError, match='overwrite'):
        attach_interaction_controls(scene, result, **kwargs)
    mismatched = deepcopy(scene)
    mismatched['agents'][-1]['inputs']['prediction_frame_offsets'][0] = 2
    with pytest.raises(ValueError, match='grid'):
        attach_interaction_controls(mismatched, decision, **kwargs)
