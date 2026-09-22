import itertools

import numpy as np
import pytest

from src.evaluation.m3w_protected_joint_support import audit_problem
from src.world_model.m3w_joint_intervention import InterventionProblem


def problem():
    table = np.zeros((1, 2, 2))
    table[0, 1, 1] = 2.
    return InterventionProblem(np.array([4., 3., 2., 1.]), np.zeros(4),
        np.ones(4, bool), np.array([[0, 1]]), table, 1., 1., 4)


def test_product_changes_joint_optimum_without_outcomes():
    r = audit_problem(problem(), np.arange(4))
    assert r['status'] == 'exhaustively_checked' and r['combinations'] == 6
    assert r['feasible'] == 6 and r['changed_identities'] == 2
    assert r['objective_advantage'] == 1.75 and r['meaningful_advantage']
    assert not r['future_labels_used']


def test_weak_coupling_cannot_overturn_unary_gap():
    p = problem()
    p.pair_cost *= .01
    r = audit_problem(p, np.arange(4))
    assert r['product_range_below_unary_gap'] and r['changed_identities'] == 0
    assert not r['meaningful_advantage']


def test_harm_budget_is_reference_fixed_not_selected_after_readout():
    p = problem()
    p.expected_harm[:] = [0., 0., 10., 10.]
    r = audit_problem(p, np.arange(4))
    assert r['predicted_harm_budget'] == 0 and r['feasible'] == 1
    assert r['unary_runner_up_gap'] is None and not r['meaningful_advantage']


def test_one_switch_and_unsupported_edge_are_structural_nulls():
    p = problem()
    p.supported[-1] = False
    assert audit_problem(p, np.arange(4))['status'] == 'structural_null'
    p = problem()
    p.supported[:2] = False
    r = audit_problem(p, np.arange(4))
    assert r['status'] == 'structural_null' and r['nonadditive_edges'] == 0


def test_cap_records_blocker_without_sampling():
    r = audit_problem(problem(), np.arange(4), enumeration_cap=5)
    assert r['status'] == 'enumeration_cap_blocker'
    assert r['objective_advantage'] is None and not r['exact_enumeration']


def test_invalid_or_future_interface_rejected():
    with pytest.raises(ValueError):
        audit_problem(problem(), np.zeros(4))
    with pytest.raises(TypeError):
        audit_problem(problem(), np.arange(4), future_target=np.zeros((4, 12, 2)))


def test_array_loader_rejects_future_members_before_opening(tmp_path):
    from scripts.audit_m3w_protected_joint_support import causal_arrays
    missing = tmp_path/'not_opened.npz'
    for key in ('target', 'valid', 'future_endpoint', 'oracle_best'):
        with pytest.raises(ValueError, match='causal arrays'):
            causal_arrays(missing, (key,))
    path = tmp_path/'scores.npz'
    np.savez(path, ids=np.arange(2), target=np.full((2, 12, 2), np.nan))
    np.testing.assert_array_equal(causal_arrays(path, ('ids',))['ids'], np.arange(2))


@pytest.mark.parametrize('seed', range(6))
def test_exact_result_matches_independent_scalar_pair_sum(seed):
    rng = np.random.default_rng(seed)
    n = 6
    edges = np.array(list(itertools.combinations(range(n), 2)))
    table = rng.uniform(0, 1, (len(edges), 2, 2))
    table[:, 0, 0] = 0
    p = InterventionProblem(rng.uniform(.1, 3, n), np.zeros(n), np.ones(n, bool),
        edges, table, 1., 1., n)
    scores = []
    for c in itertools.combinations(range(n), 3):
        take = set(c)
        forecast = -sum(p.expected_gain[i] for i in c)/n
        unary = sum(table[e, 1, 0]*(i in take)+table[e, 0, 1]*(j in take)
                    for e, (i, j) in enumerate(edges))/len(edges)
        joint = sum(table[e, int(i in take), int(j in take)]
                    for e, (i, j) in enumerate(edges))/len(edges)
        scores.append((forecast+unary, forecast+joint))
    s = np.array(scores)
    expected = s[np.argmin(s[:, 0]), 1]-s[:, 1].min()
    np.testing.assert_allclose(audit_problem(p, np.arange(n))['objective_advantage'], expected, atol=1e-12)
