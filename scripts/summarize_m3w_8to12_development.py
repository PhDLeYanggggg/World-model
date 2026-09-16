"""Export light, explicitly exploratory results from the fixed development study."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.data_unification.m3w_causal_recordings import BASELINES, causal_coordinate_transform
from src.evaluation.m3w_experiment_contract import ExperimentContract, file_digest


def main():
    protocol = ROOT / 'configs/m3w_8to12_development_v1.json'
    contract = ExperimentContract(json.loads(protocol.read_text()), ROOT)
    data = ROOT / 'data/stage_cvpr2027_experiments/8to12_v1'
    output = ROOT / 'outputs/publication_readiness_2026_09/8to12_development_v1'
    output.mkdir(parents=True, exist_ok=True)
    fits, rows, selections, manifests, local_metrics, oracle = [], [], {}, {}, {}, {}
    for seed in contract.protocol['seeds']:
        devpath = data / f'seed{seed}_development/development_report.json'
        dev = json.loads(devpath.read_text())
        manifests[str(devpath.relative_to(ROOT))] = file_digest(devpath)
        selections[str(seed)] = dev['selection']
        local_metrics[str(seed)] = {name: s['dataset_local_metrics'] for name, s in dev['summaries'].items()}
        oracle_rows = []
        reference = f'seed{seed}_neural_cost_conservative'
        for receipt in devpath.parent.glob('*.receipt.json'):
            saved = json.loads(receipt.read_text())
            if saved['key'][0] != reference:
                continue
            cache = receipt.with_name(receipt.name.replace('.receipt.json', '.rows.json'))
            if file_digest(cache) != saved['cache_sha256']:
                raise ValueError('Changed development row cache')
            oracle_rows.extend(r for r in json.loads(cache.read_text()) if r['baseline_ade'] is not None)
        values = np.asarray([[r['baseline_ade'], r['arms']['uncontrolled']['ade']] for r in oracle_rows])
        scene_costs = []
        for scene in sorted({r['physical_scene'] for r in oracle_rows}):
            v = values[[r['physical_scene'] == scene for r in oracle_rows]]
            scene_costs.append([v[:, 0].mean(), np.minimum(v[:, 0], v[:, 1]).mean(),
                                np.maximum(v[:, 0] - v[:, 1], 0).mean(),
                                np.maximum(v[:, 1] - v[:, 0], 0).mean()])
        base, best, benefit, harm = np.mean(scene_costs, axis=0)
        oracle[str(seed)] = {'count': len(values), 'scope': 'future_label_oracle_diagnostic_not_inference',
                            'candidate_win_fraction': float((values[:, 1] < values[:, 0]).mean()),
                            'oracle_gain_pct': float(100 * (base - best) / base) if base > 0 else None,
                            'mean_positive_gain': float(benefit), 'mean_positive_harm': float(harm)}
        local_oracle = {}
        for recording in sorted({r['recording_id'] for r in oracle_rows}):
            part = [r for r in oracle_rows if r['recording_id'] == recording]
            b = np.array([r['baseline_ade'] * r['scale'] for r in part])
            n = np.array([r['arms']['uncontrolled']['ade'] * r['scale'] for r in part])
            local_oracle[recording] = {'coordinate_claim': 'dataset_local_unverified',
                'baseline_ade': float(b.mean()), 'candidate_ade': float(n.mean()),
                'candidate_gain_pct': float(100 * (b.mean() - n.mean()) / b.mean()) if b.mean() > 0 else None,
                'oracle_gain_pct': float(100 * (b.mean() - np.minimum(b, n).mean()) / b.mean()) if b.mean() > 0 else None}
        oracle[str(seed)]['per_recording_native_units'] = local_oracle
        for candidate, summary in dev['summaries'].items():
            for arm, metrics in summary['arms'].items():
                rows.append({'seed': seed, 'candidate': candidate, 'arm': arm,
                    'ade': metrics['all']['selected_error'], 'fde': metrics['secondary_fde']['selected_error'],
                    'easy_baseline_ade': metrics['easy'].get('baseline_error'),
                    'easy_selected_ade': metrics['easy'].get('selected_error'),
                    'easy_excess_ade': metrics['easy'].get('mean_excess_over_floor'),
                    'all_improvement_pct': metrics['all']['improvement_pct'],
                    'easy_degradation_pct': -metrics['easy']['improvement_pct'] if metrics['easy']['count'] else None,
                    'hard_improvement_pct': metrics['hard']['improvement_pct'],
                    'switch_rate_all_past_supported': metrics['switch_rate_all_past_supported'],
                    'switch_rate_complete_labels': metrics['all']['switch_rate_valid_labels'],
                    'positive_harm': metrics['all']['mean_positive_harm'],
                    'complete_queries': metrics['all']['count'],
                    'fde_endpoint_label_queries': metrics['secondary_fde']['count'],
                    'past_supported_queries': summary['agent_query_count'],
                    'scene_count': summary['physical_scene_count'],
                    'scene_ci_status': metrics['all']['bootstrap']['status']})
        for name in ['full', 'hold0', 'hold1', 'hold2', 'neural_cost']:
            path = data / f'seed{seed}_{name}/fit_report.json'
            report = json.loads(path.read_text())
            checkpoint = Path(report['checkpoint'])
            losses = report['losses']
            fits.append({'seed': seed, 'model': name, 'training_complete': report['training_complete'],
                'steps': report['steps_completed'], 'checkpoint_sha256': file_digest(checkpoint),
                'elapsed_seconds': report.get('elapsed_seconds', json.loads(
                    (path.parent / 'heartbeat.jsonl').read_text().splitlines()[-1])['elapsed_seconds']),
                'runtime': report['runtime'],
                'first_20_batch_mean_loss': float(np.mean(losses[:20])),
                'last_20_batch_mean_loss': float(np.mean(losses[-20:])),
                'losses_are_training_not_validation': True})
            manifests[str(path.relative_to(ROOT))] = file_digest(path)
    baseline = {}
    for recording, role in contract.protocol['assignments'].items():
        if role not in {'fit', 'development'}:
            continue
        reader, ids = contract.open_recording(recording, purpose=role)
        errors, raw_errors = [], []
        for index in ids:
            inputs = reader.get_inputs(int(index))
            row = reader.index[index]
            history = reader.points[int(row['history_start']):int(row['current_row']) + 1]
            t = causal_coordinate_transform(history[:, 2:4], history[:, 0], int(row['horizon_raw']))
            labels = reader.get_labels(int(index))
            target = (labels['future_xy_dataset_local'] - t['origin_xy']) @ t['rotation'] / t['scale']
            error = np.linalg.norm(inputs['baseline_rollouts'] - target[None], axis=-1)
            value = np.stack((error.mean(-1), error[:, -1]), axis=-1)
            errors.append(value)
            raw_errors.append(value * t['scale'])
        means = np.mean(errors, axis=0)
        raw = np.mean(raw_errors, axis=0)
        baseline[recording] = {'role': role, 'physical_scene': reader.metadata['physical_scene'],
            'rows': len(ids), 'error_unit': 'past_normalized', 'result_source': 'fresh_run',
            'native_raw_horizons_for_12_steps': reader.metadata['observation_step_actual_raw_horizons'],
            'raw_t50_available_complete_windows_not_evaluated': reader.metadata['raw_exact_windows']['50'],
            'baselines': {name: {'ade': float(means[i, 0]), 'fde': float(means[i, 1])}
                          for i, name in enumerate(BASELINES)},
            'dataset_local_unverified_baselines': {name: {'ade': float(raw[i, 0]), 'fde': float(raw[i, 1])}
                                                   for i, name in enumerate(BASELINES)}}
    result = {'result_source': 'fresh_run_real_training_and_development_evaluation',
              'scope': 'historically_exposed_development_only_not_confirmatory',
              'protocol_sha256': contract.digest, 'task': contract.protocol['task'],
              'seeds': contract.protocol['seeds'], 'fits': fits, 'comparisons': rows,
              'selection_by_seed': selections, 'baselines': baseline, 'source_manifests': manifests,
              'dataset_local_metrics': local_metrics,
              'baseline_candidate_oracle_diagnostic': oracle,
              'independent_confirmation': False, 'stage5c_executed': False, 'smc_enabled': False}
    (output / 'metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    lines = ['# Real 8-to-12 Development Results', '',
             'Result source: `fresh_run`. Historical recordings remain development-exposed.',
             'All results below are exploratory, not a corrected Stage37 confirmation or deployment certificate.', '',
             '| Seed | Head/policy | Arm | ADE | FDE | All gain % | Hard gain % | Easy degradation % | Switch % |',
             '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in rows:
        easy = 'NA' if r['easy_degradation_pct'] is None else f"{r['easy_degradation_pct']:.3f}"
        lines.append(f"| {r['seed']} | {r['candidate']} | {r['arm']} | {r['ade']:.5f} | {r['fde']:.5f} | "
                     f"{r['all_improvement_pct']:.3f} | {r['hard_improvement_pct']:.3f} | {easy} | "
                     f"{100*r['switch_rate_all_past_supported']:.3f} |")
    lines.extend(['', 'ADE/FDE are past-normalized, not meters. Gains use the fixed causal constant-velocity floor.',
                  'Switch rate in this table includes all past-supported queries; the JSON also reports complete-label coverage.',
                  'Secondary FDE uses valid endpoint labels; ADE requires the complete path. Their denominators are explicitly separate.',
                  'The JSON also reports absolute easy-slice error and excess; a near-zero baseline makes percentages highly sensitive.',
                  'One physical development scene supports no meaningful scene-bootstrap interval. Three seeds measure training variability, not new sites.',
                  'The raw-frame t+50 supplement, matched realized-count control, public forecaster comparison and independent confirmation remain separate pending work.', '',
                  '## Candidate Headroom', '',
                  '| Seed | Candidate wins % | Label-oracle gain % | Mean benefit | Mean harm |',
                  '| --- | ---: | ---: | ---: | ---: |'])
    for seed, r in oracle.items():
        lines.append(f"| {seed} | {100*r['candidate_win_fraction']:.3f} | {r['oracle_gain_pct']:.3f} | "
                     f"{r['mean_positive_gain']:.5f} | {r['mean_positive_harm']:.5f} |")
    lines.extend(['', 'The oracle reads future labels for diagnosis only. It is not a policy input or a deployable result.', '',
                  '## Causal Baseline Audit', '',
                  '| Recording | Role | Rows | Lowest ADE baseline | ADE |', '| --- | --- | ---: | --- | ---: |'])
    for recording, b in baseline.items():
        best = min(b['baselines'], key=lambda name: b['baselines'][name]['ade'])
        lines.append(f"| {recording} | {b['role']} | {b['rows']} | {best} | {b['baselines'][best]['ade']:.5f} |")
    lines.extend(['', 'This post-run baseline audit does not change the already fitted floor or tune a test-set threshold.',
                  'The lowest development baseline is a descriptive oracle over global baseline choices, not an independently selected deployment choice.', ''])
    (output / 'results.md').write_text('\n'.join(lines))
    print(json.dumps({'output': str(output), 'fits': len(fits), 'comparisons': len(rows),
                      'selections': {seed: selection['selected'] for seed, selection in selections.items()}}))


if __name__ == '__main__':
    main()
