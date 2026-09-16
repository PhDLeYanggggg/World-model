"""Materialize the delegated, explicitly development-only 2026-09-16 decision."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.data_unification.m3w_causal_recordings import RecordingWindows, PROTOCOL_STEPS
from src.evaluation.m3w_experiment_contract import ExperimentContract, file_digest, protocol_digest


def main():
    destination = ROOT / 'configs/m3w_8to12_development_v1.json'
    if destination.exists():
        raise SystemExit('Protocol exists; verify it or create a new version, never overwrite approval')
    p = json.loads((ROOT / 'configs/m3w_independent_experiment.draft.json').read_text())
    decision = 'outputs/publication_readiness_2026_09/research_route_decision.md'
    p.update(status='approved', scope='exploratory', study_design='development_only')
    p.pop('unresolved', None)
    p['assignments'] = {r: 'excluded' for r in p['records']}
    fit = ['eth_eth', 'eth_hotel', 'ucy_zara01', 'ucy_zara02', 'ucy_zara03']
    development = ['ucy_students01', 'ucy_students03']
    p['assignments'].update({r: 'fit' for r in fit})
    p['assignments'].update({r: 'development' for r in development})
    p['fit_folds'] = {r: (0 if r == 'eth_eth' else 1 if r == 'eth_hotel' else 2) for r in fit}
    p['task'].update(history_steps=8, prediction_unit='observation_steps', horizon=12,
                     primary_metric='ade', aggregation='equal_physical_scene')
    p.update(seeds=[17, 29, 43], bootstrap_unit='physical_scene', bootstrap_resamples=2000)
    control = 'data/stage_cvpr2027_experiments/8to12_v1/control'
    for stage in ('calibration', 'confirmation'):
        p[f'{stage}_receipt'] = f'{control}/{stage}_forbidden.json'
    errors, displacements, counts = [], [], {}
    for name in fit:
        reader = RecordingWindows(ROOT / p['records'][name]['cache_path'])
        ids = np.flatnonzero((reader.index['protocol'] == PROTOCOL_STEPS) &
                             (reader.index['future_steps'] == 12))
        counts[name] = len(ids)
        for i in ids:
            inputs = reader.get_inputs(int(i))
            row = reader.index[i]
            xy = reader.points[int(row['history_start']):int(row['current_row']) + 1, 2:4]
            from src.data_unification.m3w_causal_recordings import causal_coordinate_transform
            frames = reader.points[int(row['history_start']):int(row['current_row']) + 1, 0]
            transform = causal_coordinate_transform(xy, frames, int(row['horizon_raw']))
            target = reader.get_labels(int(i))['future_xy_dataset_local']
            target = (target - transform['origin_xy']) @ transform['rotation'] / transform['scale']
            errors.append(float(np.linalg.norm(inputs['baseline_rollouts'][1] - target, axis=1).mean()))
            if name.startswith('ucy_zara'):
                displacements.extend(np.linalg.norm(np.diff(xy, axis=0), axis=1).tolist())
    easy, hard = map(float, np.quantile(errors, [.25, .75]))
    step_scale = float(np.median([d for d in displacements if d > 0]))
    p['risk'].update(delta=None, risks=[], easy_definition={
        'kind': 'baseline_error_at_most', 'metric': 'ade',
        'error_unit': 'past_normalized', 'threshold': easy})
    policies = {
        'conservative': {'pair_weight': .1, 'max_mean_predicted_harm': .01,
                         'max_intervention_fraction': .25, 'min_predicted_gain': .02,
                         'max_agent_predicted_harm': .05},
        'moderate': {'pair_weight': .1, 'max_mean_predicted_harm': .03,
                     'max_intervention_fraction': .5, 'min_predicted_gain': .01,
                     'max_agent_predicted_harm': .1},
    }
    p['development_evaluation'] = {
        'error_unit': 'past_normalized', 'label_policy': 'complete_requested_path', 'query_stride': 1,
        'geometry_by_recording': {r: {'graph_radius': 6 * step_scale,
                                     'proximity_threshold': .5 * step_scale} for r in development},
        'easy_threshold': easy, 'hard_threshold': hard,
        'eligible_arms': ['independent', 'scene_uniform', 'joint'], 'solver_seconds': .25,
        'bootstrap_seed': 917, 'policies': policies,
    }
    p['gain_harm_training'] = {'width': 64, 'loss': 'squared_benefit_harm',
        'fit_settings': {'steps': 1000, 'batch_size': 128, 'learning_rate': .0003,
                         'checkpoint_every': 100, 'heartbeat_every': 20}}
    p['derived_fit_only_statistics'] = {'baseline': 'constant_velocity_causal_fd',
        'complete_fit_rows': counts, 'easy_hard_quantiles': [.25, .75],
        'geometry_step_scale': step_scale, 'geometry_source': 'ucy_zara_fit_past_displacements',
        'development_labels_used': False}
    bindings = set(p['bindings']) | {decision, 'scripts/prepare_m3w_8to12_development.py',
        'configs/m3w_intervention_backend.json', 'src/world_model/m3w_supervised_intervention.py',
        'src/world_model/m3w_neural_gain_harm.py', 'src/evaluation/m3w_development_evaluation.py'}
    p['bindings'] = {name: file_digest(ROOT / name) for name in sorted(bindings)}
    p['approval'] = {'approved_by': 'research_agent_under_explicit_user_delegation_2026-09-16',
                     'decision_reference': decision, 'protocol_sha256': protocol_digest(p)}
    contract = ExperimentContract(p, ROOT)
    with destination.open('x') as stream:
        json.dump(p, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'protocol': str(destination), 'sha256': contract.digest,
        'fit_rows': sum(counts.values()), 'easy_threshold': easy, 'hard_threshold': hard,
        'scope': p['scope'], 'confirmation_permitted': False}))


if __name__ == '__main__':
    main()
