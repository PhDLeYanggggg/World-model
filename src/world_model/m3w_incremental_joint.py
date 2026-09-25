"""Matched-count incremental interventions with immutable incumbent actions.

All arguments are past observations, predictions or fitted scores. No future
labels or validity masks are accepted. Pair proximity is a coordinate proxy,
not measured collision risk. Predicted harm caps are not calibrated guarantees.
"""
from dataclasses import replace
import hashlib

import numpy as np

from src.world_model.m3w_joint_intervention import (
    InterventionProblem, past_proximity_edges, proximity_cost_table)
from src.world_model.m3w_interaction_controls import (
    decompose_pair_objective, solve_control, _describe)
from src.world_model.m3w_native_joint_controls import compare


def problem(*, floor, neural, old, pool, current, widths, utility, risk, cost_scale,
            pair_weight=.1, radius_widths=3., threshold_widths=.5):
    floor, neural = np.asarray(floor, float), np.asarray(neural, float)
    n = len(floor)
    old, pool, current, widths = map(np.asarray, (old, pool, current, widths))
    utility, risk = np.asarray(utility, float), np.asarray(risk, float)
    if (not n or floor.shape != (n, 12, 2) or neural.shape != floor.shape
            or old.shape != (n,) or old.dtype != bool or pool.shape != (n,)
            or pool.dtype != bool or np.any(old & pool) or current.shape != (n, 2)
            or widths.shape != (n,) or np.any(widths <= 0)
            or utility.shape != (n, 2) or risk.shape != (n, 2)
            or np.any(utility < 0) or np.any(risk < 0)
            or not all(np.isfinite(a).all() for a in (floor, neural, current, widths, utility, risk))
            or not np.isfinite(cost_scale) or cost_scale <= 0):
        raise ValueError('Aligned finite past-only arrays and disjoint Boolean actions required')
    gain = (utility[:, 0]-utility[:, 1])/cost_scale
    if np.any(pool & (gain <= 0)):
        raise ValueError('Additional candidates must satisfy the frozen positive-gain gate')
    incumbent = np.where(old[:, None, None], neural, floor)
    edges = past_proximity_edges(current, radius=radius_widths*float(np.median(widths)))
    pair = proximity_cost_table(incumbent, neural, edges,
        distance_threshold=threshold_widths*float(np.median(widths)))
    harm = risk[:, 1]/cost_scale
    return InterventionProblem(gain, harm, pool, edges, pair, pair_weight,
        float(np.mean(pool*np.maximum(harm, -gain))), int(pool.sum()))


def controls(p, ids, *, seconds=2., salt='m3w-incremental-joint-control-v1'):
    """Half-count controls; the full pool is a unique-action replay, not joint evidence."""
    ids = np.asarray(ids)
    if ids.shape != p.supported.shape or len(np.unique(ids)) != len(ids):
        raise ValueError('Unique stable row IDs required')
    bits, report = compare(p, ids, np.ones(len(ids), bool), .5, solver_seconds=seconds)
    k = report['count']; parts = decompose_pair_objective(p)
    cap = report['predicted_harm_budget']
    matched = replace(p, max_interventions=k, max_mean_predicted_harm=cap)
    independent = bits['independent']
    # Positive hash priorities leave the original coherent harm constraint intact
    # on the supported pool. The hash order never sees predicted or actual gain.
    priority = np.array([int.from_bytes(hashlib.sha256(f'{salt}|{i}'.encode()).digest()[:8], 'big')/2**64
        for i in ids]) + 1e-12
    if k == 0 or k == int(p.supported.sum()):
        hashed = independent.copy(); hash_ok = True; hash_reason = 'unique_assignment'
    else:
        hp = replace(matched, expected_gain=priority,
                     expected_harm=parts['original_coherent_harm'])
        rr = solve_control(hp, objective_kind='independent', exact_interventions=k,
                           time_limit_seconds=seconds)
        hash_ok = rr['solver_optimal'] and rr['exact_count_satisfied']
        hashed = rr['switch'] if hash_ok else independent.copy(); hash_reason = rr['reason']
    hash_ok = bool(hash_ok and _describe(matched, hashed, parts)['predicted_constraints_satisfied'])
    # A solver failure is not a different-coverage scientific comparison.
    # Revert the whole matched set to the reference and retain the failure flag.
    valid = report['matched'] and hash_ok
    if not valid:
        bits['unary'] = independent.copy(); bits['joint'] = independent.copy(); hashed = independent.copy()
    out = dict(half_independent=independent, half_hash=hashed,
               half_unary=bits['unary'], half_joint=bits['joint'], full_add=p.supported.copy())
    uniform = np.zeros(len(ids), bool)
    full_desc = _describe(matched, p.supported, parts)
    if full_desc['predicted_constraints_satisfied'] and full_desc['full_objective'] < 0:
        uniform = p.supported.copy()
    out['scene_uniform'] = uniform
    for name in ('half_independent', 'half_hash', 'half_unary', 'half_joint'):
        assert int(out[name].sum()) == k and not np.any(out[name] & ~p.supported)
        assert _describe(matched, out[name], parts)['predicted_constraints_satisfied']
    descriptors = {name: _describe(matched if name != 'full_add' else p, v, parts) for name, v in out.items()}
    products = parts['product_objective']
    supported_edges = p.supported[p.edges].all(1)
    nonadditive = int(np.count_nonzero(supported_edges & (products != 0))) if k >= 2 else 0
    return out, dict(count=k, pool=int(p.supported.sum()), agents=len(ids), edges=len(p.edges),
        matched=bool(valid), hash_solver_ok=hash_ok, hash_reason=hash_reason,
        base_comparison=report, predicted_harm_budget=cap,
        nonadditive_supported_edges_at_count=nonadditive,
        joint_unary_changed_agents=int(np.count_nonzero(out['half_joint'] != out['half_unary'])),
        joint_independent_changed_agents=int(np.count_nonzero(out['half_joint'] != independent)),
        arms=descriptors, physical_safety_certified=False, realized_risk_certified=False)
