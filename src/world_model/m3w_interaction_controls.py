"""Diagnostic removal of non-additive pair costs; no deployment-policy change.

The original gain/harm vectors, support, harm cap and exact reference count stay
fixed. Geometry changes the objective only, never the predicted-risk budget.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csc_matrix

from src.world_model.m3w_joint_intervention import InterventionProblem

OBJECTIVE_TOLERANCE = 1e-9


def decompose_pair_objective(problem: InterventionProblem) -> dict:
    problem.validate()
    n, m = len(problem.expected_gain), len(problem.edges)
    unary = np.zeros(n, dtype=float)
    residual = np.zeros(m, dtype=float)
    if m:
        p = problem.pair_cost
        np.add.at(unary, problem.edges[:, 0], p[:, 1, 0])
        np.add.at(unary, problem.edges[:, 1], p[:, 0, 1])
        unary /= m
        residual = (p[:, 1, 1]-p[:, 1, 0]-p[:, 0, 1])/m
    return dict(unary_proxy=unary, product_proxy=residual,
        unary_objective=-problem.expected_gain/n+problem.pair_weight*unary,
        product_objective=problem.pair_weight*residual,
        original_coherent_harm=np.maximum(problem.expected_harm, -problem.expected_gain))


def _describe(problem, bits, parts):
    p, x = problem, np.asarray(bits)
    if x.dtype != bool or x.shape != p.supported.shape:
        raise ValueError('Aligned Boolean decisions required')
    pair = p.pair_cost[np.arange(len(p.edges)), x[p.edges[:, 0]].astype(int), x[p.edges[:, 1]].astype(int)]
    gain = float(np.mean(x*p.expected_gain))
    harm = float(np.mean(x*parts['original_coherent_harm']))
    proxy = float(pair.mean()) if len(pair) else 0.
    return dict(mean_predicted_gain=gain, mean_predicted_harm=harm,
        mean_pair_proxy=proxy, full_objective=-gain+p.pair_weight*proxy,
        unary_objective=float(parts['unary_objective']@x),
        product_objective=float(parts['product_objective']@(x[p.edges[:, 0]] & x[p.edges[:, 1]])),
        active_product_edges=int(np.count_nonzero(x[p.edges[:, 0]] & x[p.edges[:, 1]] & (parts['product_objective'] != 0))),
        predicted_constraints_satisfied=bool(not np.any(x & ~p.supported)
            and x.sum() <= p.max_interventions and harm <= p.max_mean_predicted_harm+1e-10))


def solve_control(problem: InterventionProblem, *, objective_kind: str,
                  exact_interventions: int | None = None, time_limit_seconds: float = 30.) -> dict:
    """Versioned numerical control solver; the legacy policy remains unchanged.

    Positive objective scaling preserves minimizers and original risk constraints.
    Solver success alone is insufficient: check the original-unit primal/dual gap
    and recompute the objective from rounded binary decisions and exact products.
    """
    p = problem
    parts = decompose_pair_objective(p)
    n = len(p.expected_gain)
    k = exact_interventions
    if objective_kind not in {'independent', 'unary_geometry', 'joint'}:
        raise ValueError('Unknown objective control')
    if k is not None and (isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer))
            or not 0 <= k <= p.max_interventions):
        raise ValueError('Exact count must be an integer within the original cap')
    if not np.isfinite(time_limit_seconds) or time_limit_seconds <= 0:
        raise ValueError('Positive finite time limit required')
    edges = p.edges if objective_kind == 'joint' else np.empty((0, 2), dtype=int)
    m = len(edges)
    unary = -p.expected_gain/n if objective_kind == 'independent' else parts['unary_objective']
    objective = np.r_[unary, parts['product_objective'] if m else np.empty(0)]
    magnitude = float(np.abs(objective).max())
    scale = min(1e12, max(1., 1e4/max(magnitude, 1e-12))) if magnitude else 1.
    rows, cols, values = [], [], []
    upper = [p.max_mean_predicted_harm, p.max_interventions]
    for i in range(n):
        rows.extend([0, 1]); cols.extend([i, i]); values.extend([parts['original_coherent_harm'][i]/n, 1.])
    for e, (i, j) in enumerate(edges):
        for coefficients, cap in [({n+e:1., i:-1.}, 0.), ({n+e:1., j:-1.}, 0.),
                                  ({i:1., j:1., n+e:-1.}, 1.)]:
            row = len(upper)
            for column, value in coefficients.items():
                rows.append(row); cols.append(column); values.append(value)
            upper.append(cap)
    lower = np.full(len(upper), -np.inf)
    if k is not None:
        lower[1] = upper[1] = int(k)
    matrix = csc_matrix((values, (rows, cols)), shape=(len(upper), n+m))
    result = milp(c=objective*scale, integrality=np.r_[np.ones(n), np.zeros(m)],
        bounds=Bounds(np.zeros(n+m), np.r_[p.supported.astype(float), np.ones(m)]),
        constraints=LinearConstraint(matrix, lower, np.array(upper)),
        options={'time_limit':time_limit_seconds, 'mip_rel_gap':0.})
    switch = np.zeros(n, dtype=bool)
    optimal, reason = False, 'solver_not_optimal_floor'
    numerical = dict(objective_scale=scale, objective_tolerance=OBJECTIVE_TOLERANCE,
        recomputed_primal=None, reported_primal=None, dual_bound=None, absolute_gap=None,
        numerical_certificate_pass=False)
    if result.success:
        proposed = np.asarray(getattr(result, 'x', None))
        if proposed.shape == (n+m,) and np.isfinite(proposed).all():
            bits = proposed[:n] > .5
            products = bits[edges[:, 0]] & bits[edges[:, 1]]
            valid = (np.allclose(proposed[:n], bits, atol=1e-7, rtol=0)
                     and np.allclose(proposed[n:], products, atol=1e-7, rtol=0)
                     and (k is None or bits.sum() == k)
                     and _describe(p, bits, parts)['predicted_constraints_satisfied'])
            if valid:
                direct = float(objective@np.r_[bits, products])
                primal = float(getattr(result, 'fun', np.nan))/scale
                dual = float(getattr(result, 'mip_dual_bound', np.nan))/scale
                tol = OBJECTIVE_TOLERANCE*(1+abs(direct))
                certificate = (np.isfinite([primal, dual, direct]).all()
                    and abs(primal-direct) <= tol and -tol <= direct-dual <= tol)
                numerical.update(recomputed_primal=direct,
                    reported_primal=primal if np.isfinite(primal) else None,
                    dual_bound=dual if np.isfinite(dual) else None,
                    absolute_gap=direct-dual if np.isfinite(dual) else None,
                    numerical_certificate_pass=bool(certificate))
                if certificate:
                    optimal = True
                    if k is not None or direct < 0:
                        switch, reason = bits, 'numerically_checked_'+objective_kind+'_control'
                    else:
                        reason = 'no_positive_gain_floor'
                else:
                    reason = 'solver_certificate_failed_floor'
            else:
                reason = 'solver_solution_invalid_floor'
        else:
            reason = 'solver_solution_invalid_floor'
    return dict(switch=switch, solver_optimal=optimal, reason=reason,
        exact_count_satisfied=None if k is None else bool(switch.sum() == k),
        exact_interventions_requested=None if k is None else int(k),
        solver_version='scaled_primal_dual_checked_controls_v1', numerical=numerical,
        **_describe(p, switch, parts), risk_budget_uses_original_scores=True,
        deployment_policy=False, realized_risk_certified=False, physical_safety_certified=False)


def geometry_aware_independent(problem: InterventionProblem, *, exact_interventions: int,
                               time_limit_seconds: float = 30.) -> dict:
    """Remove only x_i*x_j terms while preserving the original feasible set."""
    return solve_control(problem, objective_kind='unary_geometry', exact_interventions=exact_interventions,
                         time_limit_seconds=time_limit_seconds)


def compare_interaction_controls(problem: InterventionProblem, *, time_limit_seconds: float = 30.) -> dict:
    """Compare risk-only, unary geometry and full pair geometry at one causal count."""
    parts = decompose_pair_objective(problem)
    reference = solve_control(problem, objective_kind='independent', time_limit_seconds=time_limit_seconds)
    k = int(reference['switch'].sum())
    unary = geometry_aware_independent(problem, exact_interventions=k, time_limit_seconds=time_limit_seconds)
    joint = solve_control(problem, objective_kind='joint', exact_interventions=k, time_limit_seconds=time_limit_seconds)
    controls = dict(independent=reference, unary_geometry=unary, joint=joint)
    for arm in controls.values():
        arm.update(_describe(problem, arm['switch'], parts))
    matched = all(arm['solver_optimal'] and arm['predicted_constraints_satisfied']
                  and int(arm['switch'].sum()) == k for arm in controls.values())
    loss_bound = float(np.abs(parts['product_objective']).sum())
    if matched:
        advantage = unary['full_objective']-joint['full_objective']
        # Exact minimizers imply this range; tolerance is numerical, not risk assurance.
        tol = 4*OBJECTIVE_TOLERANCE*(1+abs(unary['full_objective'])+abs(joint['full_objective']))
        if not -tol <= advantage <= loss_bound+tol:
            raise ArithmeticError('Matched objective ordering or decomposition bound failed')
    else:
        advantage = None
    supported_edges = (problem.supported[problem.edges[:, 0]] & problem.supported[problem.edges[:, 1]])
    possible = int(np.count_nonzero(supported_edges & (parts['product_objective'] != 0))) if k >= 2 else 0
    return dict(controls=controls, reference_count=k, matched=bool(matched),
        nonzero_matched=bool(matched and k > 0),
        joint_minus_unary_switch_identities=int(np.count_nonzero(unary['switch'] != joint['switch'])),
        potential_nonadditive_edges_at_count=possible,
        predicted_full_objective_advantage=advantage,
        absolute_product_sum_bound=loss_bound,
        cardinality_source='independent_past_only_decision_before_labels',
        same_original_harm_budget=True, equal_realized_harm_claim=False,
        tie_break_identity_invariance_claim=False,
        matched_on_past_support_not_future_label_availability=True,
        deployment_policy=False, formal_protocol_changed=False,
        realized_accuracy_lift='not_run_requires_frozen_predictor_and_labels',
        physical_safety_certified=False)
