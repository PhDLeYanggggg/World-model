"""Matched motion/forecast controls with unchanged conditional-risk budgets."""
import numpy as np

from src.world_model.m3w_european_source_forecast import baseline_numpy
from src.world_model.m3w_european_source_intervention import causal_cost_features, paired_cost_labels
from src.world_model.m3w_european_conditional_risk import event_labels, pointwise_rule, CONTROL_ARMS
from src.world_model.m3w_event_risk_feasibility import bounded_risk_coefficients
from src.world_model.m3w_joint_intervention import (
    InterventionProblem, past_proximity_edges, proximity_cost_table, select_interventions,
)
from src.world_model.m3w_scaled_risk_controls import solve_scaled_risk_control


def motion_candidate_inputs(geometry, history, origin):
    h, o = np.asarray(history), np.asarray(origin)
    if o.shape != (len(h), 2) or not np.array_equal(h[:, -1], o):
        raise ValueError('Observed last position must anchor both causal rollouts')
    g = np.asarray(geometry)
    if g.shape != (len(h), 476) or not np.array_equal(
            g[:, :16], (h-o[:, None]).reshape(len(h), 16).astype(np.float32)):
        raise ValueError('Geometry history and rollout history disagree')
    b = baseline_numpy(h, 1)-o[:, None]
    p = baseline_numpy(h, 3)-o[:, None]
    x, scale = causal_cost_features(geometry, b, p)
    return dict(b=b, p=p, x=x, scale=scale, same=np.all(b == p, axis=(1, 2)))


def target_labels(reference, candidate_error, easy_cut, task):
    costs = paired_cost_labels(reference, candidate_error)
    if task == 'utility':
        return costs
    if task not in ('all', 'easy'):
        raise ValueError('Registered utility/all/easy task required')
    return event_labels(reference, costs[:, 1], easy_cut=easy_cut, event=task)


def protected_query(utility, moments, moving, current_xy, widths, baseline, candidate,
                    *, support_available, budget, pair_weight, radius_widths,
                    threshold_widths, seconds):
    u, m = np.asarray(utility, float), np.asarray(moments, float)
    pointwise_rule(u, m, moving, budget=budget, support_available=support_available)
    widths = np.asarray(widths, float)
    if widths.shape != u.shape or not np.isfinite(widths).all() or np.any(widths <= 0):
        raise ValueError('Positive observed widths required')
    bounded = bounded_risk_coefficients(u, m, moving, budget=budget, support_available=support_available)
    edges = past_proximity_edges(current_xy, radius=radius_widths*float(np.median(widths)))
    pair = proximity_cost_table(baseline, candidate, edges,
        distance_threshold=threshold_widths*float(np.median(widths)))
    problem = InterventionProblem(bounded['expected_gain'], bounded['expected_harm'],
        bounded['supported'], edges, pair, pair_weight, budget, len(u))
    independent = solve_scaled_risk_control(problem, objective_kind='independent', time_limit_seconds=seconds)
    count = int(independent['switch'].sum())
    result = dict(independent=independent,
        scene_uniform=select_interventions(problem, mode='scene_uniform'),
        joint=solve_scaled_risk_control(problem, objective_kind='joint', time_limit_seconds=seconds),
        unary_exact=solve_scaled_risk_control(problem, objective_kind='unary_geometry',
            exact_interventions=count, time_limit_seconds=seconds),
        joint_exact=solve_scaled_risk_control(problem, objective_kind='joint',
            exact_interventions=count, time_limit_seconds=seconds))
    mass = bounded['predicted_mass']
    for row in result.values():
        numerator = float(np.dot(row['switch'], m[:, 1]))
        row['switch_rate'] = float(row['switch'].mean())
        row['predicted_event_ratio'] = numerator/mass if mass > 0 else None
        if row['switch'].any() and (not support_available or mass <= 0 or numerator > budget*mass+1e-8):
            raise ValueError('Original conditional-risk constraint violated')
    matched = all(result[k]['solver_optimal'] and result[k]['predicted_constraints_satisfied']
        and result[k]['switch'].sum() == count for k in ('independent', 'unary_exact', 'joint_exact'))
    return dict(**result, matched=bool(matched), matched_nonzero=bool(matched and count > 0),
        reference_count=count, agents=len(u), edges=len(edges), predicted_denominator_mean=mass/len(u),
        pruned_infeasible=int((np.asarray(moving) & (u > 0) & support_available
            & (mass > 0) & (m[:, 1] > budget*mass)).sum()))


def candidate_decisions(reg, data, a, held, qmask, utility, moments, available):
    held = np.asarray(held)
    if held.ndim != 1 or len(np.unique(held)) != len(held) or np.asarray(qmask).dtype != bool:
        raise ValueError('Unique target ids and causal query mask required')
    local = qmask[held]
    ids = held[local]
    moving = np.linalg.norm(np.diff(data['history'][held], axis=1), axis=2).sum(1) > 0
    choices = dict(pointwise_ids=held, ids=ids,
        pointwise=pointwise_rule(utility, moments, moving, budget=reg['predicted_risk_budget'],
            support_available=available), matched=np.zeros(len(ids), bool),
        matched_nonzero=np.zeros(len(ids), bool), **{k:np.zeros(len(ids), bool) for k in CONTROL_ARMS})
    keys, inverse = np.unique(np.column_stack((data['recordings'][ids], data['frames'][ids])),
        axis=0, return_inverse=True)
    queries = []
    for i, key in enumerate(keys):
        loc = np.flatnonzero(inverse == i)
        ix = ids[loc]
        out = protected_query(utility[local][loc], moments[local][loc], moving[local][loc],
            data['origin'][ix], data['width'][ix], a['b'][ix]+data['origin'][ix, None],
            a['p'][ix].astype(float)+data['origin'][ix, None], support_available=available,
            budget=reg['predicted_risk_budget'], pair_weight=reg['pair_weight'],
            radius_widths=reg['edge_radius_bbox_widths'], threshold_widths=reg['proximity_threshold_bbox_widths'],
            seconds=reg['solver_seconds'])
        for k in (*CONTROL_ARMS, 'matched', 'matched_nonzero'):
            choices[k][loc] = out[k]['switch'] if k in CONTROL_ARMS else out[k]
        queries.append(dict(recording=int(key[0]), frame=int(key[1]), site=str(data['sites'][ix[0]]),
            agents=len(ix), **{k:out[k] for k in ('edges', 'matched', 'matched_nonzero', 'reference_count',
                'predicted_denominator_mean', 'pruned_infeasible')},
            arms={k:{f:out[k][f] for f in ('reason', 'solver_optimal', 'predicted_constraints_satisfied',
                'mean_pair_proxy', 'mean_predicted_gain', 'mean_predicted_harm', 'switch_rate',
                'predicted_event_ratio')} for k in CONTROL_ARMS}))
    return choices, queries
