"""Fixed query-level expected easy-risk budgets; not calibrated safety."""
from dataclasses import replace

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from src.world_model.m3w_interaction_controls import solve_control, decompose_pair_objective, _describe

ARMS = ('floor', 'net_stop', 'strict_stop', 'pointwise', 'aggregate_selected',
        'aggregate_population', 'scene_uniform', 'aggregate_unary', 'aggregate_joint')


def selected_budget(gain, q, r, support, rho, seconds):
    """Signed knapsack: the easy denominator includes selected targets only."""
    n = len(gain)
    coeff = q-rho*r
    bits = np.zeros(n, bool)
    if not support.any() or np.sum(np.minimum(coeff[support], 0)) == 0 and np.all(coeff[support] > 0):
        return bits, True, 'analytic_empty'
    if coeff[support].sum() <= 0:
        return support.copy(), True, 'all_positive_gain_candidates_feasible'
    scale = 1e4/max(float(np.max(gain)), 1e-12)
    risk_scale = max(float(np.max(np.abs(coeff))), 1e-12)
    res = milp(c=-gain*scale, integrality=np.ones(n),
        bounds=Bounds(np.zeros(n), support.astype(float)),
        constraints=LinearConstraint(coeff[None, :]/risk_scale, -np.inf, 0.),
        options={'time_limit':seconds, 'mip_rel_gap':0.})
    if not res.success:
        return bits, False, 'selected_solver_not_optimal_floor'
    x = np.asarray(res.x); proposed = x > .5
    direct = float(-gain@proposed)
    primal, dual = float(res.fun)/scale, float(getattr(res, 'mip_dual_bound', np.nan))/scale
    tol = 1e-9*(1+abs(direct))
    if (not np.allclose(x, proposed, atol=1e-7, rtol=0) or np.any(proposed & ~support)
            or float(coeff@proposed) > 1e-10 or not np.isfinite([direct,primal,dual]).all()
            or abs(primal-direct)>tol or not -tol<=direct-dual<=tol):
        return bits, False, 'selected_numerical_certificate_failed_floor'
    return proposed, True, 'selected_checked_optimum'


def allocate(problem, q, r, target, *, cost_scale, pointwise, strict, rho=.02, seconds=5.):
    """No observed errors, labels, future masks or test statistics in this API."""
    problem.validate()
    q, r = np.asarray(q, float), np.asarray(r, float)
    target, pointwise, strict = map(np.asarray, (target, pointwise, strict))
    n = len(q)
    if (r.shape != (n,) or problem.supported.shape != (n,)
            or not np.isfinite(q).all() or not np.isfinite(r).all() or (q<0).any() or (r<0).any()
            or any(v.shape!=(n,) or v.dtype!=bool for v in (target,pointwise,strict))
            or not target.any() or np.any(problem.supported & ~target)
            or np.any(pointwise & ~problem.supported) or np.any(strict & ~problem.supported)
            or np.any(q[~target]!=0) or np.any(r[~target]!=0)
            or not np.isfinite(cost_scale) or cost_scale<=0 or rho!=.02
            or not np.isfinite(seconds) or seconds<=0):
        raise ValueError('Aligned causal query moments and fixed tolerance required')
    budget = rho*float(r.sum())
    gain = np.maximum(problem.expected_gain,0)
    p = replace(problem, expected_gain=gain, expected_harm=q/cost_scale,
        max_mean_predicted_harm=budget/(n*cost_scale), max_interventions=int(problem.supported.sum()))
    p.validate()
    parts = decompose_pair_objective(p)
    chosen = dict(floor=np.zeros(n,bool), net_stop=p.supported.copy(),
                  strict_stop=strict.copy(), pointwise=pointwise.copy())
    chosen['aggregate_selected'], ok, reason = selected_budget(
        gain, q/cost_scale, r/cost_scale, p.supported, rho, seconds)
    info = dict(selected_optimal=ok, selected_reason=reason, population_budget=budget,
                denominator_target_rows=int(target.sum()))
    # Positive-risk candidates individually exceeding the entire budget are infeasible.
    feasible_support = p.supported & (q<=budget)
    reduced = replace(p, supported=feasible_support, max_interventions=int(feasible_support.sum()))
    if q[feasible_support].sum()<=budget:
        ref = dict(switch=feasible_support.copy(), solver_optimal=True,
                   reason='all_individually_feasible_candidates_fit', **_describe(p, feasible_support, parts))
    else:
        ref = solve_control(reduced, objective_kind='independent', time_limit_seconds=seconds)
    chosen['aggregate_population'] = ref['switch']
    k = int(ref['switch'].sum())
    feasible = bool(np.all(~target | p.supported) and q[target].sum()<=budget)
    chosen['scene_uniform'] = target.copy() if feasible else np.zeros(n,bool)
    unique = k in (0, int(feasible_support.sum()))
    if unique:
        unary = dict(ref, switch=ref['switch'].copy(), reason='unique_assignment_at_reference_count')
    else:
        unary = solve_control(reduced, objective_kind='unary_geometry', exact_interventions=k,
                              time_limit_seconds=seconds)
    active = feasible_support[p.edges].all(1) & (parts['product_objective']!=0)
    if unique or k<=1 or not active.any():
        joint = dict(unary, switch=unary['switch'].copy(), reason='joint_same_feasible_objective')
    else:
        joint = solve_control(reduced, objective_kind='joint', exact_interventions=k,
                              time_limit_seconds=seconds)
    chosen['aggregate_unary'], chosen['aggregate_joint'] = unary['switch'], joint['switch']
    matched = all(v['solver_optimal'] and int(v['switch'].sum())==k
                  and float(q@v['switch'])<=budget+1e-8 for v in (ref,unary,joint))
    info.update(population_optimal=ref['solver_optimal'], unary_optimal=unary['solver_optimal'],
        joint_optimal=joint['solver_optimal'], matched=bool(matched), reference_count=k,
        nonadditive_edges=int(active.sum()) if k>=2 else 0,
        joint_unary_changed_agents=int(np.count_nonzero(joint['switch']!=unary['switch'])))
    info['policies']={name:dict(selected=int(bits.sum()),
        predicted_easy_harm=float(q@bits), selected_easy_denominator=float(r@bits),
        population_budget_satisfied=bool(q@bits<=budget+1e-8),
        expected_gain=float(np.sum(gain*bits)),
        pair_proxy=float(p.pair_cost[np.arange(len(p.edges)),bits[p.edges[:,0]].astype(int),bits[p.edges[:,1]].astype(int)].mean()) if len(p.edges) else 0.)
        for name,bits in chosen.items()}
    return chosen, info
