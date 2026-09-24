"""Nested source-cost learning primitives; no reserved labels or safety claim."""
import hashlib

import numpy as np

from src.world_model.m3w_native_gain_harm import cost_features
from src.world_model.m3w_joint_intervention import (
    InterventionProblem, past_proximity_edges, proximity_cost_table, select_interventions)
from src.world_model.m3w_scaled_risk_controls import solve_scaled_risk_control


def nested_producers(outer):
    if outer not in range(3):
        raise ValueError('Exactly three registered source groups required')
    remaining = sorted(set(range(3))-{outer})
    return {remaining[0]:remaining[1],remaining[1]:remaining[0]}


def causal_cost_features(geometry, baseline, candidate):
    g = np.array(geometry,dtype=np.float32,copy=True)
    b,p = np.asarray(baseline),np.asarray(candidate)
    if g.ndim!=2 or g.shape[1]!=476 or b.shape!=(len(g),12,2) or p.shape!=b.shape:
        raise ValueError('Aligned fixed causal schema and rollouts required')
    if not all(np.isfinite(v).all() for v in (g,b,p)):
        raise ValueError('Nonfinite past or prediction')
    h = g[:,:16].reshape(-1,8,2)
    neighbor = g[:,38:166].reshape(-1,8,8,2)
    valid = g[:,230:294].reshape(-1,8,8)>0
    scale = np.maximum(1,np.maximum(np.linalg.norm(h,axis=-1).max(1),
        np.where(valid,np.linalg.norm(neighbor,axis=-1),0).max((1,2))))
    g[:,:16] /= scale[:,None]
    g[:,38:166] /= scale[:,None]
    g[:,332:356] = (b/scale[:,None,None]).reshape(-1,24)
    x,_ = cost_features(g,p/scale[:,None,None],scale)
    return x,scale


def paired_cost_labels(reference, candidate):
    b,c = np.asarray(reference,float),np.asarray(candidate,float)
    if (b.shape!=c.shape or b.ndim!=1 or not np.array_equal(np.isnan(b),np.isnan(c))
            or np.isinf(b).any() or np.isinf(c).any()
            or np.any(b[np.isfinite(b)]<0) or np.any(c[np.isfinite(c)]<0)):
        raise ValueError('Paired nonnegative costs with identical unknown support required')
    return np.column_stack((np.maximum(b-c,0),np.maximum(c-b,0)))


def query_subset(sites, recordings, frames, count, salt):
    sites,rec,frame = np.asarray(sites),np.asarray(recordings),np.asarray(frames)
    if sites.shape!=rec.shape or sites.shape!=frame.shape or sites.ndim!=1 or count<=0:
        raise ValueError('Aligned past query keys and positive fixed cap required')
    selected = np.zeros(len(sites),bool)
    for site in sorted(set(sites)):
        ids = np.flatnonzero(sites==site)
        keys = sorted(set(zip(rec[ids].tolist(),frame[ids].tolist())))
        chosen = set(sorted(keys,key=lambda k:hashlib.sha256(
            f'{salt}|{site}|{k[0]}|{k[1]}'.encode()).hexdigest())[:count])
        selected[ids] = [k in chosen for k in zip(rec[ids].tolist(),frame[ids].tolist())]
    return selected


def controls(costs, moving, current_xy, widths, baseline, candidate, *, budget,
             pair_weight, radius_widths, threshold_widths, seconds):
    cost = np.asarray(costs,float)
    if cost.shape!=(len(moving),2) or not np.isfinite(cost).all() or np.any(cost<0):
        raise ValueError('Finite predicted positive benefit and harm required')
    widths = np.asarray(widths,float)
    if widths.shape!=(len(moving),) or not np.isfinite(widths).all() or np.any(widths<=0):
        raise ValueError('Positive current detection widths required')
    width = float(np.median(widths))
    edges = past_proximity_edges(current_xy,radius=radius_widths*width)
    pair = proximity_cost_table(baseline,candidate,edges,distance_threshold=threshold_widths*width)
    gain = cost[:,0]-cost[:,1]
    supported = np.asarray(moving,bool)&(gain>0)
    problem = InterventionProblem(gain,cost[:,1],supported,edges,pair,pair_weight,budget,len(cost))
    independent = solve_scaled_risk_control(problem,objective_kind='independent',time_limit_seconds=seconds)
    count = int(independent['switch'].sum())
    unary = solve_scaled_risk_control(problem,objective_kind='unary_geometry',exact_interventions=count,
                                      time_limit_seconds=seconds)
    joint_exact = solve_scaled_risk_control(problem,objective_kind='joint',exact_interventions=count,
                                            time_limit_seconds=seconds)
    joint = solve_scaled_risk_control(problem,objective_kind='joint',time_limit_seconds=seconds)
    uniform = select_interventions(problem,mode='scene_uniform')
    for result in (independent,unary,joint_exact,joint,uniform):
        result['switch_rate'] = float(result['switch'].mean())
    matched = all(r['solver_optimal'] and r['predicted_constraints_satisfied']
                  and int(r['switch'].sum())==count for r in (independent,unary,joint_exact))
    return dict(independent=independent,scene_uniform=uniform,joint=joint,unary_exact=unary,
                joint_exact=joint_exact,matched=bool(matched),matched_nonzero=bool(matched and count>0),
                edges=len(edges),agents=len(cost),reference_count=count)
