"""Frozen-predictor cost heads and matched-count source-development controls."""
from dataclasses import replace
import numpy as np
from src.world_model.m3w_european_source_intervention import causal_cost_features
from src.world_model.m3w_geometric_cost_head import rollout_envelope
from src.world_model.m3w_joint_intervention import (
    InterventionProblem, past_proximity_edges, proximity_cost_table, select_interventions)
from src.world_model.m3w_scaled_risk_controls import solve_scaled_risk_control
from src.world_model.m3w_interaction_controls import _describe, decompose_pair_objective


def inference_features(geometry, cv, candidate):
    x, scale = causal_cost_features(geometry, cv, candidate)
    return x, rollout_envelope(cv, candidate), scale


def allowed(utility, all_risk, easy_risk, moving, budget):
    arrays = [np.asarray(a, float) for a in (utility, all_risk, easy_risk)]
    moving = np.asarray(moving)
    if (moving.dtype != bool or any(a.shape != (len(moving), 2) for a in arrays)
            or any(not np.isfinite(a).all() or (a < 0).any() for a in arrays)
            or not np.isfinite(budget) or budget < 0):
        raise ValueError('Finite nonnegative causal costs and explicit budget required')
    u, a, e = arrays
    return moving & (u[:, 0] > u[:, 1]) & (a[:, 1] <= budget*a[:, 0]) & (e[:, 1] <= budget*e[:, 0])


def joint_controls(utility, all_risk, easy_risk, moving, current, widths, cv, candidate, scale, cfg):
    support = allowed(utility, all_risk, easy_risk, moving, cfg['risk_budget'])
    widths = np.asarray(widths, float)
    if (not np.isfinite(scale) or scale <= 0 or widths.shape != (len(support),)
            or not np.isfinite(widths).all() or (widths <= 0).any()):
        raise ValueError('Training-only positive cost scale and current widths required')
    width = float(np.median(widths))
    edges = past_proximity_edges(current, radius=cfg['edge_radius_bbox_widths']*width)
    pair = proximity_cost_table(cv, candidate, edges, distance_threshold=cfg['proximity_threshold_bbox_widths']*width)
    # Support requires both pointwise risk screens. Utility units and pair tradeoff
    # retain the training-only native cost scale; predictions are not certificates.
    gain = (utility[:, 0]-utility[:, 1])/scale
    harm = np.maximum(utility[:, 1], all_risk[:, 1])/scale
    cap = cfg['risk_budget']*float(all_risk[:, 0].mean())/scale
    p = InterventionProblem(gain, harm, support, edges, pair, cfg['pair_weight'], cap, len(support))
    scene = select_interventions(p, mode='scene_uniform')
    p = replace(p, max_interventions=int(np.ceil(cfg['fraction']*support.sum())))
    ref = solve_scaled_risk_control(p, objective_kind='independent', time_limit_seconds=cfg['solver_seconds'])
    count = int(ref['switch'].sum())
    unary = solve_scaled_risk_control(p, objective_kind='unary_geometry', exact_interventions=count,
                                     time_limit_seconds=cfg['solver_seconds'])
    joint = solve_scaled_risk_control(p, objective_kind='joint', exact_interventions=count,
                                     time_limit_seconds=cfg['solver_seconds'])
    matched = all(r['solver_optimal'] and r['predicted_constraints_satisfied'] and int(r['switch'].sum()) == count
                  for r in (ref, unary, joint))
    result = dict(point=support, scene_uniform=scene['switch'], half_independent=ref['switch'],
                  half_unary=unary['switch'], half_joint=joint['switch'])
    parts = decompose_pair_objective(p)
    arms = {}
    for name, bits in result.items():
        desc = _describe(replace(p, max_interventions=len(support)), bits, parts)
        arms[name] = dict(**desc, switches=int(bits.sum()))
    product = parts['product_objective']
    active = int(np.count_nonzero((product != 0) & support[edges[:, 0]] & support[edges[:, 1]]))
    return result, dict(agents=len(support), supported=int(support.sum()), edges=len(edges), matched=bool(matched),
        matched_nonzero=bool(matched and count > 0), reference_count=count, nonadditive_supported_edges=active,
        joint_changes=int(np.count_nonzero(joint['switch'] != ref['switch'])), arms=arms,
        solver_reasons=[r['reason'] for r in (ref, unary, joint)], realized_risk_certified=False)
