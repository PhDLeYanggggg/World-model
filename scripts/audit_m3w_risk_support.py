"""Read-only scene-support audit plus explicitly hypothetical bound sensitivity."""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from src.evaluation.m3w_experiment_contract import file_digest
from src.world_model.m3w_joint_intervention import screen_cluster_risks


def support_audit(protocol):
    reviewed = []
    for name, record in protocol['records'].items():
        directory = (ROOT / record['cache_path']).resolve()
        if not directory.is_relative_to(ROOT):
            raise ValueError('Cache path outside workspace')
        metadata_path = directory / 'metadata.json'
        if file_digest(metadata_path) != record['metadata_sha256']:
            raise ValueError('Changed recording metadata')
        metadata = json.loads(metadata_path.read_text())
        if metadata['id'] != name or metadata['physical_scene'] != record['physical_scene']:
            raise ValueError('Recording/scene identity changed')
        for relative, expected in metadata['artifacts'].items():
            path = (directory / relative).resolve()
            if not path.is_relative_to(directory) or file_digest(path) != expected['sha256']:
                raise ValueError('Changed raw-position cache identity')
        reviewed.append({'recording_id': name, 'physical_scene': record['physical_scene'],
                         'historical_use': record['historical_use'], 'role': protocol['assignments'][name]})
    scenes = sorted({r['physical_scene'] for r in reviewed})
    return {'source_data': 'cached_verified_metadata_and_array_hashes_no_labels',
            'recordings': reviewed, 'recording_count': len(reviewed), 'physical_scene_count': len(scenes),
            'historically_unexposed_reviewed_scene_count': len({r['physical_scene'] for r in reviewed if r['historical_use'] == 'unexposed_reviewed'}),
            'assigned_calibration_scene_count': len({r['physical_scene'] for r in reviewed if r['role'] == 'calibration'}),
            'protocol_approved': protocol['status'] == 'approved', 'independence_proven': False}


def sensitivity(scene_count):
    rows = []
    for m, k in ((1, 1), (5, 2), (10, 2), (20, 2)):
        for tolerance in (.02, .1, .2):
            delta = .05
            result = screen_cluster_risks(losses=np.zeros((scene_count, m, k)),
                cluster_ids=[f'hypothetical_{i}' for i in range(scene_count)], fitted_cluster_ids=[],
                policy_ids=[f'p{i}' for i in range(m)], lower=np.zeros(k), upper=np.ones(k),
                tolerance=np.full(k, tolerance), delta=delta)
            rows.append({'hypothetical_independent_scenes': scene_count, 'policies': m, 'risks': k,
                         'illustrative_tolerance': tolerance, 'illustrative_delta': delta,
                         'assumed_observed_loss': 0., 'upper_bound': float(result['upper_risk_bound'][0, 0]),
                         'any_accepted': bool(result['accepted'].any()),
                         'optimistic_zero_loss_scenes_required_for_this_bound': math.ceil(math.log(m*k/delta)/(2*tolerance**2))})
    return rows


def correlated_window_diagnostic():
    """Illustrate pseudo-replication, not a claim about actual trajectory losses."""
    seed, draws, clusters, copies = 927, 20000, 6, 1000
    true_risk, tolerance, delta = .2, .1, .05
    losses = np.random.default_rng(seed).binomial(1, true_risk, size=(draws, clusters))
    mean = losses.mean(1)
    correct = np.minimum(1., mean + math.sqrt(math.log(1 / delta) / (2 * clusters)))
    naive = np.minimum(1., mean + math.sqrt(math.log(1 / delta) / (2 * clusters * copies)))
    return {'scope': 'synthetic_perfect_within_scene_dependence_not_real_calibration',
            'seed': seed, 'trials': draws, 'independent_clusters': clusters,
            'identical_window_copies_per_cluster': copies, 'true_risk': true_risk,
            'illustrative_tolerance': tolerance, 'illustrative_delta': delta,
            'correct_cluster_false_acceptance_fraction': float(np.mean(correct <= tolerance)),
            'naive_window_false_acceptance_fraction': float(np.mean(naive <= tolerance)),
            'analytical_all_zero_cluster_probability': (1 - true_risk) ** clusters}


def main():
    path = ROOT / 'configs/m3w_independent_experiment.draft.json'
    audit = support_audit(json.loads(path.read_text()))
    table = sensitivity(audit['physical_scene_count'])
    dependence = correlated_window_diagnostic()
    report = {'result_source': 'fresh_run_support_count_and_analytical_sensitivity', 'support': audit,
              'sensitivity': table, 'sensitivity_is_not_actual_calibration': True,
              'correlated_window_diagnostic': dependence,
              'actual_calibration': 'not_run_protocol_unapproved_no_assigned_independent_scenes',
              'illustrative_numbers_are_not_approved_risk_budgets': True,
              'all_scenes_hypothetically_allocated_to_calibration_is_optimistic_and_not_a_split_proposal': True,
              'larger_window_count_does_not_increase_independent_n': True,
              'easy_2pct_relative_degradation_is_not_0point02_bounded_harm': True,
              'stage5c_executed': False, 'smc_enabled': False,
              'code_sha256': {p: file_digest(ROOT / p) for p in ('scripts/audit_m3w_risk_support.py',
                   'src/world_model/m3w_joint_intervention.py', 'src/evaluation/m3w_experiment_contract.py')},
              'draft_sha256': file_digest(path)}
    directory = ROOT / 'outputs/publication_readiness_2026_09/risk_calibration'
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'support_audit.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    lines = ['# Calibration Support and Bound Sensitivity', '',
             'Fresh metadata/array identity checks and analytical calculation; no future labels, training or calibration result.', '',
             f'Current reader: {audit["recording_count"]} recordings / {audit["physical_scene_count"]} physical-scene groups; '
             f'{audit["historically_unexposed_reviewed_scene_count"]} unexposed-reviewed scenes; '
             f'{audit["assigned_calibration_scene_count"]} assigned calibration scenes. Protocol is unapproved.', '',
             'The table hypothetically treats all current scene groups as independent calibration units with zero observed loss. '
             'This is deliberately optimistic, not a legal data split or a claim of independence. Tolerances and delta are illustrative, not chosen for the experiment.', '',
             f'| Frozen policies | Risks | Tolerance | Upper bound with {audit["physical_scene_count"]} zero-loss scenes | Zero-loss scene count needed by this bound |',
             '| ---: | ---: | ---: | ---: | ---: |']
    lines.extend(f'| {r["policies"]} | {r["risks"]} | {r["illustrative_tolerance"]:.2f} | {r["upper_bound"]:.4f} | '
                 f'{r["optimistic_zero_loss_scenes_required_for_this_bound"]} |' for r in table)
    lines += ['', 'For a [0,1] scene loss, this implementation uses mean + sqrt(log(M*K/delta)/(2*n)), capped at 1. '
              'The sample requirement is the rearranged inequality at empirical loss zero. It is a limitation of this conservative bound, '
              'not an information-theoretic lower bound or evidence that every alternative calibration method requires that many scenes.', '',
              'Window replication, bootstrap draws and more agents cannot be substituted for independent scenes. '
              'A bounded clipped-harm tolerance of 0.02 is not a 2% relative ADE/FDE guarantee; clipping and denominators differ. '
              'Changing risk definitions, units, population or query schedules requires protocol approval.', '',
              '## Synthetic Dependence Diagnostic', '',
              f'{dependence["trials"]:,} independent simulations draw 6 Bernoulli scene losses with true risk 0.20, then copy each loss into 1,000 identical windows. '
              f'At illustrative tolerance 0.10 and delta 0.05, the scene-level screen falsely accepts {dependence["correct_cluster_false_acceptance_fraction"]:.2%}; '
              f'the invalid window-as-independent screen falsely accepts {dependence["naive_window_false_acceptance_fraction"]:.2%}. '
              'The analytical all-zero-scene probability is 0.8^6 = 26.2144%. This constructed example exposes pseudo-replication; '
              'it does not estimate real M3W failure rates or prove general calibration power.', '',
              'No actual statistical risk certificate, untouched confirmation, deployment promotion or metric/seconds claim is produced.']
    (directory / 'support_audit.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'recordings': audit['recording_count'], 'scenes': audit['physical_scene_count'],
                      'assigned_calibration_scenes': audit['assigned_calibration_scene_count'], 'hypothetical_scenarios': len(table)}))


if __name__ == '__main__':
    main()
