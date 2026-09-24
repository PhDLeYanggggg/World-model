"""Numerical unit change for tiny risk budgets, preserving the original problem."""
from dataclasses import replace

import numpy as np

from src.world_model.m3w_interaction_controls import (
    OBJECTIVE_TOLERANCE, _describe, decompose_pair_objective, solve_control,
)


def solve_scaled_risk_control(problem, **kwargs):
    """Scale all cost units, never the feasible set or gain/geometry tradeoff.

    The legacy solver scales its objective but not its risk row. An equivalent
    change of cost units brings that row above its absolute feasibility tolerance.
    Both risk feasibility and primal/dual agreement are rechecked in original units.
    """
    problem.validate()
    p = problem
    parts = decompose_pair_objective(p)
    supported = parts['original_coherent_harm'][p.supported]/len(p.supported)
    magnitude = max(p.max_mean_predicted_harm, float(np.abs(supported).max()) if len(supported) else 0.)
    factor = max(1., min(1e12, 1./max(magnitude, 1e-12)))
    scaled = replace(p, expected_gain=p.expected_gain*factor,
        expected_harm=p.expected_harm*factor, pair_weight=p.pair_weight*factor,
        max_mean_predicted_harm=p.max_mean_predicted_harm*factor)
    result = solve_control(scaled, **kwargs)
    bits = result['switch'].copy()
    numeric = dict(result['numerical'])
    for key in ('recomputed_primal','reported_primal','dual_bound','absolute_gap'):
        if numeric[key] is not None:
            numeric[key] /= factor
    numeric['objective_scale'] *= factor
    numeric['risk_unit_factor'] = factor
    desc = _describe(p, bits, parts)
    kind = kwargs['objective_kind']
    direct = (-float(np.mean(p.expected_gain*bits)) if kind == 'independent'
              else desc['unary_objective'] if kind == 'unary_geometry' else desc['full_objective'])
    tol = OBJECTIVE_TOLERANCE*(1+abs(direct))
    count = kwargs.get('exact_interventions')
    if result['solver_optimal']:
        numeric_ok = (all(numeric[k] is not None for k in ('reported_primal','dual_bound'))
            and abs(numeric['reported_primal']-direct) <= tol
            and -tol <= direct-numeric['dual_bound'] <= tol)
        if not (numeric_ok and desc['predicted_constraints_satisfied']
                and (count is None or bits.sum() == count)):
            bits[:] = False
            result.update(solver_optimal=False, reason='scaled_risk_original_unit_check_failed_floor')
            numeric['numerical_certificate_pass'] = False
            desc = _describe(p, bits, parts)
    return dict(result, switch=bits, numerical=numeric, **desc,
        solver_version='risk_unit_scaled_original_certificate_v1',
        exact_count_satisfied=None if count is None else bool(bits.sum() == count))
