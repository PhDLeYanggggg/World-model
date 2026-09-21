"""Fixed source-only coupling comparison; no fitted policy or safety guarantee."""
import numpy as np

from src.world_model.m3w_joint_intervention import InterventionProblem, past_proximity_edges, proximity_cost_table
from src.world_model.m3w_interaction_controls import decompose_pair_objective, solve_control, _describe


def make_problem(*, baseline, candidate, current, forecast_valid, target_mask,
                 eligible, benefit, harm, scale, past_target_scales):
    b, c, xy = map(np.asarray, (baseline, candidate, current))
    valid, target, support = map(np.asarray, (forecast_valid, target_mask, eligible))
    n = len(b)
    if (b.shape != (n, 12, 2) or c.shape != b.shape or xy.shape != (n, 2)
            or any(v.shape != (n,) or v.dtype != bool for v in (valid, target, support))
            or np.any(support & (~valid | ~target)) or not target.any()
            or not np.isfinite(scale) or scale <= 0):
        raise ValueError('Explicit common-frame inputs/support and train-only cost scale required')
    s = np.asarray(past_target_scales, float)
    if s.shape != (int(target.sum()),) or not np.isfinite(s).all() or np.any(s <= 0):
        raise ValueError('Past-only scale for each forecast target required')
    radius = float(np.median(s))
    edges = past_proximity_edges(xy, radius=radius)
    known = valid[edges].all(1)
    unknown_edges = int((~known).sum())
    edges = edges[known]
    cost = proximity_cost_table(b, c, edges, distance_threshold=.1*radius)
    gain = (np.asarray(benefit, float)-np.asarray(harm, float))/scale
    risk = np.asarray(harm, float)/scale
    p = InterventionProblem(gain, risk, support, edges, cost, 1.,
        float(np.mean(support*np.maximum(risk, -gain))), int(support.sum()))
    p.validate()
    return p, dict(radius=radius, proximity_threshold=.1*radius, known_edges=len(edges),
        unknown_forecast_edges=unknown_edges, unsupported_context_rows=int((~valid).sum()),
        physical_safety_certified=False)


def rank_reference(p, ids, fraction):
    p.validate(); ids = np.asarray(ids)
    if ids.shape != p.supported.shape or len(np.unique(ids)) != len(ids) or fraction not in (1., .5):
        raise ValueError('Unique deterministic identities and fixed budget fraction required')
    count = int(np.floor(p.supported.sum()*fraction))
    use = np.flatnonzero(p.supported)
    selected = use[np.lexsort((ids[use], -p.expected_gain[use]))[:count]]
    bits = np.zeros(len(ids), bool); bits[selected] = True
    return bits


def compare(p, ids, target_mask, fraction, solver_seconds=5.):
    reference = rank_reference(p, ids, fraction); k = int(reference.sum())
    risk = np.maximum(p.expected_harm, -p.expected_gain)
    budget = float(np.mean(reference*risk))
    matched = InterventionProblem(p.expected_gain, p.expected_harm, p.supported,
        p.edges, p.pair_cost, p.pair_weight, budget, k)
    parts = decompose_pair_objective(matched)
    def analytic(bits, reason):
        return dict(switch=bits.copy(), solver_optimal=True, reason=reason,
            exact_count_satisfied=int(bits.sum()) == k,
            **_describe(matched, bits, parts))
    # Reference top-k is unconstrained optimal and feasible in its own budget.
    ref = analytic(reference, 'top_gain_reference_exact_count_and_harm_anchor')
    if k in (0, int(p.supported.sum())):
        unary = analytic(reference, 'unique_feasible_assignment_no_solver_needed')
        joint = analytic(reference, 'unique_feasible_assignment_no_solver_needed')
    else:
        unary = solve_control(matched, objective_kind='unary_geometry', exact_interventions=k,
                              time_limit_seconds=solver_seconds)
        active = p.supported[p.edges].all(1) & (parts['product_objective'] != 0)
        if k <= 1 or not active.any():
            joint = {**unary, 'switch':unary['switch'].copy(), 'reason':'algebraically_identical_to_unary_at_fixed_count'}
        else:
            joint = solve_control(matched, objective_kind='joint', exact_interventions=k,
                                  time_limit_seconds=solver_seconds)
    matched_ok = all(r['solver_optimal'] and r['predicted_constraints_satisfied']
        and int(r['switch'].sum()) == k for r in (ref, unary, joint))
    if matched_ok:
        advantage = unary['full_objective']-joint['full_objective']
        bound = float(np.abs(parts['product_objective']).sum())
        tol = 4e-9*(1+abs(unary['full_objective'])+abs(joint['full_objective']))
        if not -tol <= advantage <= bound+tol:
            raise ArithmeticError('Joint decomposition ordering failed')
    else:
        advantage = None
    target = np.asarray(target_mask)
    if target.shape != p.supported.shape or target.dtype != bool:
        raise ValueError('Explicit target-only whole-scene proposal required')
    whole = np.zeros(len(target), bool)
    if (not np.any(target & ~p.supported) and int(target.sum()) <= k
            and np.mean(target*risk) <= budget+1e-10 and _describe(matched, target, parts)['full_objective'] < 0):
        whole = target.copy()
    summary = dict(count=k, pool=int(p.supported.sum()), predicted_harm_budget=budget,
        matched=bool(matched_ok), joint_unary_changed_agents=int(np.count_nonzero(unary['switch'] != joint['switch'])),
        unary_reference_changed_agents=int(np.count_nonzero(unary['switch'] != reference)),
        nonadditive_supported_edges=int(np.count_nonzero(p.supported[p.edges].all(1) & (parts['product_objective'] != 0))) if k >= 2 else 0,
        objective_advantage=advantage, product_absolute_bound=float(np.abs(parts['product_objective']).sum()),
        controls={name:{key:value for key,value in r.items() if key != 'switch'}
                  for name,r in dict(independent=ref, unary=unary, joint=joint).items()},
        equal_realized_risk_claim=False, physical_safety_certified=False)
    return dict(independent=reference, unary=unary['switch'], joint=joint['switch'], scene_uniform=whole), summary
