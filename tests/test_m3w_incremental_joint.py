import inspect
import itertools
from dataclasses import replace
import numpy as np
import pytest
from src.world_model.m3w_incremental_joint import problem, controls
from src.world_model.m3w_joint_intervention import InterventionProblem
from src.world_model.m3w_interaction_controls import decompose_pair_objective, _describe


@pytest.mark.parametrize('seed', range(16))
def test_matched_solvers_against_full_enumeration(seed):
    rng = np.random.default_rng(seed); n = 7
    edges = np.array(list(itertools.combinations(range(n), 2)))
    pair = rng.random((len(edges), 2, 2)); pair[:, 0, 0] = 0
    pool = np.array([True]*6+[False])
    p = InterventionProblem(rng.random(n)+.1, rng.random(n)/10, pool, edges, pair, .1, 1., 6)
    bits, r = controls(p, np.arange(n)); assert r['matched']
    q = replace(p, max_interventions=3, max_mean_predicted_harm=r['predicted_harm_budget'])
    parts = decompose_pair_objective(q); feasible = []
    for chosen in itertools.combinations(range(6), 3):
        x = np.zeros(n, bool); x[list(chosen)] = True
        d = _describe(q, x, parts)
        if d['predicted_constraints_satisfied']: feasible.append(d)
    for name, key in [('half_joint', 'full_objective'), ('half_unary', 'unary_objective')]:
        assert abs(r['arms'][name][key]-min(v[key] for v in feasible)) < 1e-8
    assert all(bits[p].sum() == 3 for p in ('half_joint', 'half_unary', 'half_hash', 'half_independent'))
    assert not bits['scene_uniform'].any()


def arguments():
    return dict(floor=np.zeros((4, 12, 2)), neural=np.ones((4, 12, 2)),
        old=np.array([True, False, False, False]), pool=np.array([False, True, True, True]),
        current=np.zeros((4, 2)), widths=np.ones(4), utility=np.tile([1., .1], (4, 1)),
        risk=np.tile([10., .1], (4, 1)), cost_scale=2.)


def test_incumbent_neural_is_immutable_context():
    args = arguments(); p = problem(**args); bits, report = controls(p, np.arange(4))
    for v in bits.values():
        assert not np.any(v & args['old'])
        assert (args['old'] | v)[0]
    assert report['count'] == 1
    np.testing.assert_array_equal(bits['half_joint'], bits['half_unary'])
    assert report['nonadditive_supported_edges_at_count'] == 0


def test_no_future_interface_and_pool_validation():
    assert not any('future' in p or 'target' in p for p in inspect.signature(problem).parameters)
    a = arguments(); a['pool'][0] = True
    with pytest.raises(ValueError): problem(**a)
    a = arguments(); a['utility'][1] = [0., 1.]
    with pytest.raises(ValueError): problem(**a)


def test_no_additions_and_hash_reproducible():
    a = arguments(); a['pool'][:] = False
    c, r = controls(problem(**a), np.arange(4))
    assert r['matched'] and r['count'] == 0 and not any(v.any() for v in c.values())
    a = arguments(); a['old'][:] = False; a['pool'][:] = True
    one, _ = controls(problem(**a), np.arange(4)); two, _ = controls(problem(**a), np.arange(4))
    for key in one: np.testing.assert_array_equal(one[key], two[key])


def test_translation_invariance():
    a = arguments(); p = problem(**a)
    for key in ('floor', 'neural', 'current'): a[key] += 1234.
    q = problem(**a)
    np.testing.assert_allclose(p.pair_cost, q.pair_cost)
    np.testing.assert_array_equal(p.edges, q.edges)
