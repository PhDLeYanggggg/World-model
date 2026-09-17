"""Replay both scene-probe versions; summarize all settings without selection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from src.evaluation.m3w_experiment_contract import file_digest
from src.evaluation.m3w_stationary_scene_context import FEATURE_SETS, forecast_metrics
from src.evaluation.m3w_stationary_start_probe import score_probabilities


def summarize_trials(trials):
    required = {(f, s, feature, model) for f in (0, 1) for s in (17, 29, 43)
                for feature in FEATURE_SETS for model in ('linear', 'extra_trees')}
    observed = [(t['fold'], t['seed'], t['feature_name'], t['family']) for t in trials]
    if len(observed) != len(required) or set(observed) != required:
        raise ValueError('Every frozen fold/seed/feature/model must appear exactly once')
    result = []
    for fold in (0, 1):
        for feature in FEATURE_SETS:
            for model in ('linear', 'extra_trees'):
                part = [t for t in trials if (t['fold'], t['feature_name'], t['family']) == (fold, feature, model)]
                def mean(field, key):
                    vals = [t[field][key] for t in part]
                    return float(np.mean(vals)) if all(v is not None for v in vals) else None
                result.append({'held': 'ETH' if fold==0 else 'Hotel', 'features': feature, 'model': model,
                    'brier_lift': mean('classification', 'brier_lift_over_prior'),
                    'auc': mean('classification', 'auroc'),
                    'positive_brier_seeds': sum(t['classification']['brier_lift_over_prior']>0 for t in part),
                    'run_balanced_brier_lift': mean('run_balanced_brier', 'brier_lift_over_prior'),
                    'agent_balanced_brier_lift': mean('agent_balanced_brier', 'brier_lift_over_prior'),
                    'native_ade': mean('trajectory_unrestricted', 'native_ade'),
                    'unrestricted_gain_pct': mean('trajectory_unrestricted', 'gain_vs_cv_pct'),
                    'endpoint_angle': mean('trajectory_unrestricted', 'angular_error_degrees'),
                    'guarded_gain_pct_by_seed': [t['trajectory_fixed_gate']['gain_vs_cv_pct'] for t in part],
                    'guarded_easy_absolute_harm_by_seed': [t['trajectory_fixed_gate']['easy_absolute_harm'] for t in part],
                    'guarded_switch_rate_by_seed': [t['fixed_gate_switch_rate'] for t in part]})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--repaired', type=Path, required=True)
    parser.add_argument('--report-dir', type=Path, required=True)
    args = parser.parse_args()
    import joblib
    report_root = ROOT/'outputs/publication_readiness_2026_09'
    study = {}
    for version, path, report_path in (
        ('initial', args.original, report_root/'stationary_scene_probe/metrics.json'),
        ('corner_repaired', args.repaired, args.report_dir/'metrics.json')):
        report = json.loads(report_path.read_text())
        receipt = json.loads((path/'scene_rows.receipt.json').read_text())
        if file_digest(path/'scene_rows.npz') != receipt['cache_sha256'] or report['cache_sha256'] != receipt['cache_sha256']:
            raise ValueError('Source cache changed')
        with np.load(path/'scene_rows.npz', allow_pickle=False) as arrays:
            x, y, target, basis, scale, parent_scale = [arrays[k].copy() for k in
                ('features', 'start', 'native', 'basis', 'scale', 'parent_scale')]
            rows = json.loads(str(arrays['rows_json']))
        folds = np.array([r['fit_fold'] for r in rows])
        replay_max = 0.
        for t in report['trials']:
            name = f"fold{t['fold']}_seed{t['seed']}_{t['feature_name']}_{t['family']}"
            checkpoint = path/(name+'.joblib')
            if file_digest(checkpoint) != t['checkpoint_sha256']:
                raise ValueError('Model checkpoint changed')
            models = joblib.load(checkpoint)
            held = np.flatnonzero(folds==t['fold'])
            values = x[held][:, FEATURE_SETS[t['feature_name']]]
            p = models['classifier'].predict_proba(values)[:, 1]
            forecast = models['regressor'].predict(values).reshape(-1, 12, 2)
            forecast = np.einsum('nki,nji->nkj', forecast, basis[held])*scale[held, None, None]
            # A zero basis implements the recorded undefined-frame fallback.
            guarded = forecast*(p>=.9)[:, None, None]
            with np.load(path/(name+'.predictions.npz'), allow_pickle=False) as saved:
                np.testing.assert_array_equal(held, saved['held'])
                for actual, key in ((p, 'probability'), (forecast, 'prediction'), (guarded, 'guarded')):
                    difference = float(np.max(np.abs(actual-saved[key])))
                    if difference > 1e-12:
                        raise ValueError('Saved prediction replay mismatch')
                    replay_max = max(replay_max, difference)
            classification = score_probabilities(y[held], p, prior=t['classification']['train_only_prior'])
            np.testing.assert_allclose(classification['brier'], t['classification']['brier'], atol=1e-12, rtol=0)
            for prediction, key in ((forecast, 'trajectory_unrestricted'), (guarded, 'trajectory_fixed_gate')):
                current = forecast_metrics(prediction, target[held], parent_scale[held], .02349376610737637)
                for metric in ('native_ade', 'native_fde', 'parent_normalized_ade', 'gain_vs_cv_pct', 'easy_absolute_harm'):
                    np.testing.assert_allclose(current[metric], t[key][metric], atol=1e-10, rtol=1e-12)
        population = {}
        for fold in (0, 1):
            part = folds==fold
            norm = np.linalg.norm(target[part], axis=-1)
            population['ETH' if fold==0 else 'Hotel'] = {
                'rows': int(part.sum()), 'changed_future_rows': int(y[part].sum()),
                'nonzero_endpoint_rows': int((norm[:, -1]>0).sum()),
                'changed_but_zero_endpoint_rows': int(((y[part]==1)&(norm[:, -1]==0)).sum()),
                'native_cv_ade_quantiles': np.quantile(norm.mean(1), [0, .25, .5, .75, 1]).tolist(),
                'static_scale_values': np.unique(scale[part]).tolist(),
                'parent_scale_values': np.unique(parent_scale[part]).tolist()}
        study[version] = {'source_metrics_sha256': file_digest(report_path), 'model_replays': 2*len(report['trials']),
                          'max_prediction_replay_difference': replay_max, 'aggregates': summarize_trials(report['trials']),
                          'population': population, 'undefined_static_frames': report['undefined_static_frames'],
                          'unrestricted_positive_count': sum(t['trajectory_unrestricted']['gain_vs_cv_pct']>0 for t in report['trials']),
                          'guarded_positive_count': sum(t['trajectory_fixed_gate']['gain_vs_cv_pct']>0 for t in report['trials']),
                          'total_fit_seconds': sum(t['fit_seconds'] for t in report['trials'])}
    summary = {'result_source': 'fresh_run_saved_model_replay_and_aggregate_analysis',
               'analysis_code_sha256': file_digest(Path(__file__)), 'versions': study,
               'selection_performed': False, 'parent_primary_changed': False, 'new_deployment': False}
    (args.report_dir/'comparison.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    corrected = study['corner_repaired']
    lines = ['# Static-Scene Start and Trajectory Probes', '',
        'Fit-only, eight observed/twelve predicted native steps; no new deployment or confirmation.',
        '72 models per version: 36 start classifiers plus 36 multi-output trajectory regressors.',
        'Initial version and corrected shared-corner version are both retained. No model/threshold selection.', '',
        '## Corrected Run: Every Fixed Setting', '',
        'Three-seed means below. Gains are stationary-subset ADE versus CV, not the full forecasting primary result.',
        'Brier differences are absolute scores, not trajectory percentages. Linear seeds are identical.', '',
        '| Held | Features | Family | AUC | Brier lift | Positive Brier seeds | Unrestricted ADE gain % | Endpoint angle, degrees | Fixed-gate gain % by seed |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |']
    for r in corrected['aggregates']:
        gains=', '.join(f'{v:+.5f}' for v in r['guarded_gain_pct_by_seed'])
        lines.append(f"| {r['held']} | {r['features']} | {r['model']} | {r['auc']:.4f} | {r['brier_lift']:+.5f} | {r['positive_brier_seeds']}/3 | {r['unrestricted_gain_pct']:+.3f} | {r['endpoint_angle']:.2f} | {gains} |")
    lines += ['', 'All corrected unrestricted regressors are worse than CV. None of the corrected fixed-gate',
              'regressors has positive ADE gain. Zero gain means fallback, not successful prediction.', '',
              '## Easy Cases and Intervention', '',
              'All easy labels in this stationary subset have exactly zero CV error. Percentage easy degradation',
              'is undefined; absolute excess error is reported instead. Positive absolute error violates exact',
              'preservation here. A null ratio must never be turned into a pass. Units below use the unchanged',
              'parent normalizer, not verified meters. Endpoint angles exclude undefined/zero vectors;',
              'all-row ADE still includes them and every zero target.', '',
              '| Held | Features | Family | Fixed-gate switch rates | Easy absolute harm by seed |',
              '| --- | --- | --- | --- | --- |']
    for r in corrected['aggregates']:
        switch=', '.join(f'{v:.4f}' for v in r['guarded_switch_rate_by_seed'])
        harm=', '.join(f'{v:.6f}' for v in r['guarded_easy_absolute_harm_by_seed'])
        lines.append(f"| {r['held']} | {r['features']} | {r['model']} | {switch} | {harm} |")
    lines += ['', '## Source and Repair Boundaries', '',
              'The shared-corner correction removes false directional ambiguity; it does not change any native',
              'future coordinate or held row. The two apparent tiny positive gated results in the initial',
              'Hotel scene-neighbor trees disappear after correction. Both versions are archived.',
              'See ../stationary_scene_probe/corner_repair.md for exact provenance.', '',
              'ETH has 59 nonzero endpoints among 59 changed futures. Hotel has 92 nonzero endpoints among',
              '129 changed futures: 37 windows change and return to their initial recorded endpoint.',
              'Annotation precision, motion and interpolation are not distinguished by that fact alone.', '',
              'Reference images contain people with unknown capture time and are NOT encoded. Supplied',
              'destinations/groups and video frames are excluded. The XML is an unverified static proxy.',
              'All ETH points and 82.78% of Hotel points project inside the reference dimensions under',
              'the supplied H convention. This numerical check is not an alignment or calibration certificate.',
              'Dataset-local coordinates and native steps remain unverified metric/time quantities.', '',
              'All 144 fitted models across both versions replay their saved predictions within 1e-12.',
              'Only 31 agents, 45 stationary runs and two physical fit sites support these experiments.',
              'No independent-scene significance or CI is claimed. The unchanged forecasting primary and',
              'raw50 supplement are not replaced by these diagnostic slices. No Stage5C or SMC.', '']
    (args.report_dir/'results.md').write_text('\n'.join(lines))
    print(json.dumps({'versions': {key: {k: v[k] for k in ('model_replays', 'max_prediction_replay_difference',
        'undefined_static_frames', 'unrestricted_positive_count', 'guarded_positive_count', 'total_fit_seconds')}
        for key, v in study.items()}, 'corrected_aggregates': corrected['aggregates']}, indent=2))


if __name__ == '__main__':
    main()
