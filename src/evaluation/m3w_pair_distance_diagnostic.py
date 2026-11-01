"""Label-only evaluation of frozen paths; never a trajectory selection routine."""
import numpy as np
from src.world_model.m3w_frame_action_allocation import query_groups
from src.evaluation.m3w_multiaction_cost_readout import chosen_values


def empty():
    return dict(requested_pair_steps=0,known_pair_steps=0,unknown_pair_steps=0,
        known_error_mass=dict(floor=0.,unary_joint=0.,pair_joint=0.),unknown_disagreement_mass=0.)


def finish(value):
    out=dict(value); n=value['requested_pair_steps']; known=value['known_pair_steps']
    delta=value['known_error_mass']['unary_joint']-value['known_error_mass']['pair_joint']
    bound=value['unknown_disagreement_mass']
    out.update(known_only_mean_error={k:v/known if known else None for k,v in value['known_error_mass'].items()},
        full_requested_mean_gain=dict(known_contribution=delta/n if n else None,
            completion_lower=(delta-bound)/n if n else None,completion_upper=(delta+bound)/n if n else None),
        zero_pair_status='not_estimable' if not n else 'defined_requested_pairs',
        physical_collision_or_metric_claim=False)
    return out


def evaluate(packet,origins,choices,*,target_labels,valid_labels):
    """All rows stay;only observed pair-step error is known and others get bounds."""
    n=len(packet['ids']); target=np.asarray(target_labels,float); valid=np.asarray(valid_labels)
    origin=np.asarray(origins,float); scales=np.asarray(packet['scales'],float)
    if (set(choices)!={'unary_joint','pair_joint'} or target.shape!=(n,12,2)
            or valid.shape!=(n,12) or valid.dtype!=bool or origin.shape!=(n,2)
            or scales.shape!=(n,) or not np.isfinite(origin).all() or not np.isfinite(scales).all()
            or (scales<=0).any() or not np.isfinite(target[valid]).all()):
        raise ValueError('Aligned frozen paths and separately supplied valid evaluation labels required')
    for choice in choices.values():
        c=np.asarray(choice)
        if c.shape!=(n,) or c.dtype.kind not in 'iu' or (c < -1).any() or (c>=7).any():
            raise ValueError('Frozen action IDs required')
    paths={a:chosen_values(packet['rollouts'],packet['floor_rollout'],c).astype(float)+origin[:,None]
        for a,c in choices.items()}; paths['floor']=packet['floor_rollout'].astype(float)+origin[:,None]
    if any(not np.isfinite(p).all() for p in paths.values()): raise ValueError('Finite frozen predictions required')
    records={str(r):empty() for r in sorted(set(packet['recordings']))}
    for group in query_groups(packet['recordings'],packet['frames'],packet['ids']):
        i,j=np.triu_indices(len(group),1); i,j=group[i],group[j]
        if not len(i): continue
        rec=records[str(packet['recordings'][group[0]])]; known=valid[i]&valid[j]
        denominator=np.sqrt(scales[i]*scales[j])[:,None]
        truth=np.linalg.norm(target[i]-target[j],axis=-1)
        distances={a:np.linalg.norm(p[i]-p[j],axis=-1) for a,p in paths.items()}
        rec['requested_pair_steps']+=known.size; rec['known_pair_steps']+=int(known.sum())
        rec['unknown_pair_steps']+=int((~known).sum())
        for a,d in distances.items():
            rec['known_error_mass'][a]+=float(np.where(known,np.abs(d-truth)/denominator,0.).sum())
        rec['unknown_disagreement_mass']+=float(np.where(~known,
            np.abs(distances['unary_joint']-distances['pair_joint'])/denominator,0.).sum())
    total=empty()
    for value in records.values():
        for k in ('requested_pair_steps','known_pair_steps','unknown_pair_steps','unknown_disagreement_mass'):
            total[k]+=value[k]
        for a in total['known_error_mass']: total['known_error_mass'][a]+=value['known_error_mass'][a]
    return dict(total=finish(total),recordings={k:finish(v) for k,v in records.items()},
        label_role='evaluation_only_after_frozen_actions',all_query_targets_retained=True,
        inference_or_policy_change=False,source_fitting_posthoc_diagnostic=True,
        bounds='reverse_triangle_inequality_for_absolute_pair_distance_error',
        original_ADE_or_risk_gate_replacement=False)
