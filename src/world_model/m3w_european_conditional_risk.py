"""Source-only event-moment risk experiment, not calibrated risk control."""
import numpy as np

from src.world_model.m3w_european_cv_reference import CONTROL_ARMS
from src.world_model.m3w_joint_intervention import (
    InterventionProblem, past_proximity_edges, proximity_cost_table, select_interventions,
)
from src.world_model.m3w_scaled_risk_controls import solve_scaled_risk_control


def event_labels(reference, positive_harm, *, easy_cut, event):
    r, h = np.asarray(reference, float), np.asarray(positive_harm, float)
    if (r.ndim != 1 or r.shape != h.shape or event not in ('all', 'easy')
            or not np.isfinite(easy_cut) or easy_cut <= 0
            or np.isinf(r).any() or np.isinf(h).any()
            or not np.array_equal(np.isnan(r), np.isnan(h))
            or np.any(r[np.isfinite(r)] < 0) or np.any(h[np.isfinite(h)] < 0)):
        raise ValueError('Paired fitting-only supported nonnegative labels required')
    known = np.isfinite(r)
    mask = known if event == 'all' else known & (r > 0) & (r <= easy_cut)
    y = np.zeros((len(r), 2), float)
    y[mask, 0], y[mask, 1] = r[mask], h[mask]
    y[~known] = np.nan
    return y


def fitting_support(reference, sites):
    r, s = np.asarray(reference), np.asarray(sites)
    if (r.ndim != 1 or r.shape != s.shape or np.isinf(r).any()
            or np.any(r[np.isfinite(r)] < 0)):
        raise ValueError('Aligned fitting rows required')
    z = np.isfinite(r) & (r == 0)
    return dict(zero_rows=int(z.sum()), zero_localities=sorted(set(s[z])),
                known_rows=int(np.isfinite(r).sum()), unknown_rows=int(np.isnan(r).sum()),
                gate_available=bool(z.any()), interpretation='presence_is_not_a_safety_certificate')


def nonnegative_moments(predicted, same_rollout):
    p = np.asarray(predicted, float).copy()
    same = np.asarray(same_rollout)
    if (p.ndim != 2 or p.shape[1] != 2 or not np.isfinite(p).all()
            or same.shape != (len(p),) or same.dtype != bool):
        raise ValueError('Finite denominator/harm estimates and causal equality required')
    p = np.maximum(p, 0)
    # Equal forecasts imply zero added harm, not zero baseline-error mass.
    p[same, 1] = 0
    return p


def pointwise_rule(utility, moments, moving, *, budget, support_available):
    u, m, moving = np.asarray(utility), np.asarray(moments), np.asarray(moving)
    if (u.ndim != 1 or m.shape != (len(u), 2) or moving.shape != u.shape
            or moving.dtype != bool or not np.isfinite(u).all()
            or not np.isfinite(m).all() or np.any(m < 0)
            or not isinstance(support_available, (bool, np.bool_))
            or not np.isfinite(budget) or budget < 0):
        raise ValueError('Explicit finite predicted scores and fitting support required')
    return moving & (u > 0) & (m[:, 0] > 0) & (m[:, 1] <= budget*m[:, 0]) & support_available


def query_controls(utility, moments, moving, current_xy, widths, baseline, candidate,
                   *, support_available, budget, pair_weight, radius_widths,
                   threshold_widths, seconds):
    u, m, moving = np.asarray(utility, float), np.asarray(moments, float), np.asarray(moving)
    pointwise_rule(u, m, moving, budget=budget, support_available=support_available)
    widths = np.asarray(widths, float)
    if widths.shape != u.shape or not np.isfinite(widths).all() or np.any(widths <= 0):
        raise ValueError('Positive observed widths required')
    width = float(np.median(widths))
    edges = past_proximity_edges(current_xy, radius=radius_widths*width)
    pair = proximity_cost_table(baseline, candidate, edges, distance_threshold=threshold_widths*width)
    denominator = float(m[:, 0].mean())
    supported = moving & (u > 0) & support_available & (denominator > 0)
    risk = m[:, 1]/denominator if denominator > 0 else np.zeros(len(u))
    problem = InterventionProblem(np.maximum(u, 0), risk, supported, edges, pair,
                                  pair_weight, budget, len(u))
    independent = solve_scaled_risk_control(problem, objective_kind='independent', time_limit_seconds=seconds)
    count = int(independent['switch'].sum())
    unary = solve_scaled_risk_control(problem, objective_kind='unary_geometry',
                                     exact_interventions=count, time_limit_seconds=seconds)
    exact = solve_scaled_risk_control(problem, objective_kind='joint',
                                     exact_interventions=count, time_limit_seconds=seconds)
    joint = solve_scaled_risk_control(problem, objective_kind='joint', time_limit_seconds=seconds)
    uniform = select_interventions(problem, mode='scene_uniform')
    results = dict(independent=independent, unary_exact=unary, joint_exact=exact,
                   joint=joint, scene_uniform=uniform)
    for row in results.values():
        row['switch_rate'] = float(row['switch'].mean())
        numerator = float(np.dot(row['switch'], m[:, 1]))
        row['predicted_event_ratio'] = numerator/float(m[:, 0].sum()) if denominator > 0 else None
        if row['switch'].any() and (not support_available or denominator <= 0
                                   or numerator > budget*m[:, 0].sum()+1e-8):
            raise ValueError('Original conditional-moment constraint violated')
    matched = all(results[k]['solver_optimal'] and results[k]['predicted_constraints_satisfied']
                  and results[k]['switch'].sum() == count for k in ('independent', 'unary_exact', 'joint_exact'))
    return dict(**results, matched=bool(matched), matched_nonzero=bool(matched and count > 0),
                reference_count=count, agents=len(u), edges=len(edges),
                predicted_denominator_mean=denominator)


def decisions(reg, *, history, origin, widths, recordings, frames, sites, baseline,
              candidate, utility, moments, held_ids, query_mask, support_available):
    held = np.asarray(held_ids)
    local_query = np.asarray(query_mask)[held]
    if held.ndim != 1 or len(np.unique(held)) != len(held) or local_query.dtype != bool:
        raise ValueError('Unique indexed targets and Boolean causal query mask required')
    ids = held[local_query]
    moving = np.linalg.norm(np.diff(history[held], axis=1), axis=2).sum(1) > 0
    point = pointwise_rule(utility, moments, moving, budget=reg['predicted_risk_budget'],
                           support_available=support_available)
    bits = {k: np.zeros(len(ids), bool) for k in CONTROL_ARMS}
    matched, nonzero = np.zeros(len(ids), bool), np.zeros(len(ids), bool)
    keys = np.column_stack((recordings[ids], frames[ids]))
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    summaries = []
    for i, key in enumerate(unique):
        loc = np.flatnonzero(inverse == i)
        ix = ids[loc]
        result = query_controls(utility[local_query][loc], moments[local_query][loc],
            moving[local_query][loc], origin[ix], widths[ix], baseline[ix]+origin[ix, None],
            candidate[ix].astype(float)+origin[ix, None], support_available=support_available,
            budget=reg['predicted_risk_budget'], pair_weight=reg['pair_weight'],
            radius_widths=reg['edge_radius_bbox_widths'], threshold_widths=reg['proximity_threshold_bbox_widths'],
            seconds=reg['solver_seconds'])
        matched[loc], nonzero[loc] = result['matched'], result['matched_nonzero']
        for arm in CONTROL_ARMS:
            bits[arm][loc] = result[arm]['switch']
        summaries.append(dict(recording=int(key[0]), frame=int(key[1]), site=str(sites[ix[0]]),
            agents=len(ix), edges=result['edges'], matched=result['matched'],
            matched_nonzero=result['matched_nonzero'], reference_count=result['reference_count'],
            predicted_denominator_mean=result['predicted_denominator_mean'],
            arms={k: {f: result[k][f] for f in ('reason', 'solver_optimal', 'predicted_constraints_satisfied',
                'mean_pair_proxy', 'mean_predicted_gain', 'mean_predicted_harm', 'switch_rate',
                'predicted_event_ratio')} for k in CONTROL_ARMS}))
    return dict(ids=ids, pointwise_ids=held, pointwise=point, matched=matched,
                matched_nonzero=nonzero, **bits), summaries
