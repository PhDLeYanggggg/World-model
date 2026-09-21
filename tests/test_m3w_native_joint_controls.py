import itertools
import numpy as np
import pytest
from src.world_model.m3w_joint_intervention import InterventionProblem
from src.world_model.m3w_interaction_controls import decompose_pair_objective, _describe
from src.world_model.m3w_native_joint_controls import make_problem, rank_reference, compare


def problem(seed=7):
    rng = np.random.default_rng(seed); n = 6
    edges = np.array(list(itertools.combinations(range(n), 2)))
    costs = rng.uniform(0, .2, (len(edges), 2, 2)); costs[:, 0, 0] = 0
    return InterventionProblem(rng.uniform(.01, .5, n), rng.uniform(0, .01, n),
        np.ones(n, bool), edges, costs, 1., 1., n)


@pytest.mark.parametrize('seed', range(20))
def test_joint_matches_enumeration(seed):
    p = problem(seed); ids = np.arange(6)
    choices, report = compare(p, ids, np.ones(6, bool), .5)
    assert report['matched'] and report['count'] == 3
    p.max_interventions = 3; p.max_mean_predicted_harm = report['predicted_harm_budget']
    parts = decompose_pair_objective(p)
    feasible = []
    for pick in itertools.combinations(range(6), 3):
        bits = np.zeros(6, bool); bits[list(pick)] = True
        d = _describe(p, bits, parts)
        if d['predicted_constraints_satisfied']: feasible.append(d)
    for arm, key in [('joint','full_objective'),('unary','unary_objective')]:
        value = _describe(p, choices[arm], parts)[key]
        assert abs(value-min(r[key] for r in feasible)) < 1e-8


def test_full_count_is_exact_frozen_policy():
    p = problem(); p.supported[[0, 3]] = False
    choices, report = compare(p, np.arange(6), np.ones(6, bool), 1.)
    assert report['matched'] and report['joint_unary_changed_agents'] == 0
    for name in ('independent', 'unary', 'joint'):
        np.testing.assert_array_equal(choices[name], p.supported)
    assert not choices['scene_uniform'].any()


def test_one_switch_has_no_pair_product_or_tie_effect():
    p = problem(); p.supported[[0, 1, 2, 3]] = False
    c, r = compare(p, np.arange(6), np.ones(6, bool), .5)
    assert r['count'] == 1
    np.testing.assert_array_equal(c['unary'], c['joint'])


def test_future_targets_not_part_of_interface_and_unknown_forecasts_not_priced():
    b = np.zeros((3, 12, 2)); b[1, :, 0] = 1; b[2, :, 0] = .5
    c = b.copy(); c[0, :, 0] = .5
    args = dict(baseline=b, candidate=c, current=b[:, 0], forecast_valid=np.array([True, True, False]),
        target_mask=np.array([True, True, False]), eligible=np.array([True, True, False]),
        benefit=np.array([1., 2., 0.]), harm=np.zeros(3), scale=2., past_target_scales=np.ones(2)*3)
    p, d = make_problem(**args)
    assert len(p.edges) == 1 and d['unknown_forecast_edges'] == 2
    args['candidate'][2] += 100000
    q, _ = make_problem(**args)
    np.testing.assert_array_equal(p.pair_cost, q.pair_cost)
    with pytest.raises(TypeError): make_problem(**args, future_target=np.zeros((3, 12, 2)))


def test_zero_and_tie_counts():
    p = problem(); p.expected_gain[:] = 1
    np.testing.assert_array_equal(rank_reference(p, np.arange(6)[::-1], .5), [False]*3+[True]*3)
    p.supported[:] = False
    c, r = compare(p, np.arange(6), np.ones(6, bool), .5)
    assert r['matched'] and r['count'] == 0 and not any(v.any() for v in c.values())
