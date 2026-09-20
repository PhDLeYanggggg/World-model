"""Opt-in exact-count controls over already-issued, past-only scene forecasts."""
from __future__ import annotations

import numpy as np

from src.data_unification.m3w_causal_recordings import restore_scene_rollouts
from src.evaluation.m3w_development_evaluation import _validate_policy
from src.world_model.m3w_joint_intervention import (
    InterventionProblem, past_proximity_edges, proximity_cost_table,
)
from src.world_model.m3w_interaction_controls import compare_interaction_controls


def make_control_problem(scene, decision, *, policy, geometry):
    """Restore one shared coordinate frame without loading labels or forecasting."""
    _validate_policy(policy)
    ids = [a['agent_id'] for a in scene['agents']]
    if not ids or len(set(ids)) != len(ids) or decision['agent_ids'] != ids:
        raise ValueError('Identical unique scene agent identities required')
    offsets = scene['agents'][0]['inputs']['prediction_frame_offsets']
    if any(not np.array_equal(a['inputs']['prediction_frame_offsets'], offsets) for a in scene['agents']):
        raise ValueError('Shared future request grid required, not future target values')
    b, c = map(np.asarray, (decision['baseline'], decision['candidate']))
    if b.shape != (len(ids), len(offsets), 2) or c.shape != b.shape:
        raise ValueError('Frozen scene forecast alignment failed')
    common_b = restore_scene_rollouts(scene, dict(zip(ids, b)))['xy_dataset_local']
    common_c = restore_scene_rollouts(scene, dict(zip(ids, c)))['xy_dataset_local']
    positions = np.stack([a['coordinate_transform']['origin_xy'] for a in scene['agents']])
    edges = past_proximity_edges(positions, radius=geometry['graph_radius'])
    pairs = proximity_cost_table(common_b, common_c, edges, distance_threshold=geometry['proximity_threshold'])
    return InterventionProblem(np.asarray(decision['predicted_gain']), np.asarray(decision['predicted_harm']),
        np.asarray(decision['supported']), edges, pairs, policy['pair_weight'], policy['max_mean_predicted_harm'],
        int(np.floor(len(ids)*policy['max_intervention_fraction'])))


def attach_interaction_controls(scene, decision, *, policy, geometry, time_limit_seconds=30.):
    """No forecaster or label reader: identical fixed forecasts enter every arm."""
    problem = make_control_problem(scene, decision, policy=policy, geometry=geometry)
    b, c = map(np.asarray, (decision['baseline'], decision['candidate']))
    comparison = compare_interaction_controls(problem, time_limit_seconds=time_limit_seconds)
    arms = dict(decision['arms'])
    names = {'independent':'independent_coupling_reference', 'unary_geometry':'unary_geometry_exact_count',
             'joint':'joint_coupling_exact_count'}
    if set(names.values()) & set(arms):
        raise ValueError('Never overwrite a previously attached control')
    for source, name in names.items():
        arm = comparison['controls'][source]
        changed = np.any(c != b, axis=(1, 2))
        arms[name] = {**arm, 'prediction':np.where(arm['switch'][:, None, None], c, b),
            'selected_nonidentical_forecasts':int(np.count_nonzero(arm['switch'] & changed))}
    comparison['candidate_nonidentical_forecasts'] = int(changed.sum())
    comparison['selected_nonidentical_forecasts'] = {s:arms[name]['selected_nonidentical_forecasts']
                                                   for s,name in names.items()}
    comparison['matching_scope'] = 'selected_agent_count_not_changed_forecast_count_or_realized_risk'
    return {**decision, 'arms':arms, 'interaction_controls':comparison,
            'interaction_control_prediction_source':'identical_preissued_candidates_no_model_rerun'}
