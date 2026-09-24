from itertools import product

import numpy as np
import pytest

from src.world_model.m3w_joint_intervention import InterventionProblem
from src.world_model.m3w_interaction_controls import decompose_pair_objective, solve_control
from src.world_model.m3w_scaled_risk_controls import solve_scaled_risk_control


def test_micro_budget_overrun_regression():
    p = InterventionProblem(np.array([.7,.6]), np.array([.00024,.00023686983573009292]),
        np.ones(2,bool),np.empty((0,2),int),np.empty((0,2,2)),1.,.00023826085103710763,2)
    old = solve_control(p, objective_kind='independent')
    assert not old['solver_optimal'] and old['reason'] == 'solver_solution_invalid_floor'
    new = solve_scaled_risk_control(p, objective_kind='independent')
    assert new['solver_optimal'] and new['predicted_constraints_satisfied']
    np.testing.assert_array_equal(new['switch'],[True,False])
    assert new['numerical']['risk_unit_factor'] > 1


@pytest.mark.parametrize('kind',['independent','unary_geometry','joint'])
@pytest.mark.parametrize('seed',range(8))
def test_original_unit_optimum_matches_exhaustive_search(kind,seed):
    rng = np.random.default_rng(seed); n = 6
    edges = np.array([(i,j) for i in range(n) for j in range(i+1,n)])
    table = rng.random((len(edges),2,2))*1e-3;table[:,0,0]=0
    p = InterventionProblem(rng.uniform(.001,.1,n),rng.uniform(1e-6,1e-4,n),
        np.ones(n,bool),edges,table,1.,2e-5,4)
    count = None if kind == 'independent' else 2
    parts = decompose_pair_objective(p)
    candidates = []
    for bits in product((False,True),repeat=n):
        x = np.array(bits)
        if x.sum()>4 or count is not None and x.sum()!=count:continue
        if np.mean(x*p.expected_harm)>p.max_mean_predicted_harm:continue
        cost = (-np.mean(x*p.expected_gain) if kind=='independent' else parts['unary_objective']@x)
        if kind=='joint':cost+=parts['product_objective']@(x[edges[:,0]] & x[edges[:,1]])
        candidates.append(cost)
    result = solve_scaled_risk_control(p,objective_kind=kind,exact_interventions=count)
    if not candidates:
        assert not result['solver_optimal']
    else:
        assert result['solver_optimal'] and result['predicted_constraints_satisfied']
        assert result['numerical']['recomputed_primal'] == pytest.approx(min(candidates),abs=1e-10)


def test_no_feasible_negative_gain_assignment_stays_floor():
    p = InterventionProblem(np.array([-2.,-1.]),np.zeros(2),np.ones(2,bool),
        np.empty((0,2),int),np.empty((0,2,2)),1.,1e-4,1)
    r = solve_scaled_risk_control(p,objective_kind='independent',exact_interventions=1)
    assert not r['solver_optimal'] and not r['switch'].any()
