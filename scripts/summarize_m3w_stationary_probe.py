"""Replay saved fit probes and report dependent-window, run and agent summaries."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from src.evaluation.m3w_experiment_contract import file_digest
from src.evaluation.m3w_stationary_start_probe import GEOMETRY_COLUMNS
from src.evaluation.m3w_stationary_pooled_context import pool_context, GEOMETRY_COLUMNS as POOLED_GEOMETRY


def group_balanced_brier(y, p, prior, keys):
    """Each observed group gets equal weight. Not an independence assumption."""
    y, p = np.asarray(y), np.asarray(p)
    if y.shape != p.shape or len(keys) != len(y) or not len(y):
        raise ValueError('Aligned nonempty labels, probabilities and groups required')
    groups = {}
    for i, key in enumerate(keys):
        groups.setdefault(tuple(key), []).append(i)
    model = float(np.mean([np.mean((p[indices]-y[indices])**2) for indices in groups.values()]))
    baseline = float(np.mean([np.mean((prior-y[indices])**2) for indices in groups.values()]))
    return {'groups': len(groups), 'brier': model, 'prior_brier': baseline, 'brier_lift_over_prior': baseline-model,
            'independence_established': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--pooled', type=Path, required=True)
    parser.add_argument('--report-dir', type=Path, required=True)
    args = parser.parse_args()
    import joblib
    directory = args.report_dir
    base = json.loads((directory/'metrics.json').read_text())
    pooled = json.loads((directory/'pooled_metrics.json').read_text())
    receipt = json.loads((args.source/'fit_rows.receipt.json').read_text())
    if file_digest(args.source/'fit_rows.npz') != receipt['cache_sha256']:
        raise ValueError('Cached data changed')
    with np.load(args.source/'fit_rows.npz', allow_pickle=False) as data:
        x, y, rows = data['features'], data['targets'], json.loads(str(data['rows_json']))
    results = []
    for version, report, model_dir, values in [('ordered', base, args.source, x), ('pooled', pooled, args.pooled, pool_context(x))]:
        fits = [t for t in report['trials'] if t.get('model') in ('logistic', 'extra_trees')]
        if len(fits) != 24 or any(t['status'] != 'fresh_run' for t in fits):
            raise ValueError('Expected all 24 original fresh fits per version')
        for t in fits:
            held = np.array([i for i, r in enumerate(rows) if r['fit_fold'] == t['fold']])
            name = f"fold{t['fold']}_seed{t['seed']}_{t['features']}_{t['model']}"
            model_path = model_dir/(name+'.joblib')
            if file_digest(model_path) != t['model_sha256']:
                raise ValueError('Probe checkpoint changed')
            if t['features'] == 'geometry':
                columns = GEOMETRY_COLUMNS if version == 'ordered' else POOLED_GEOMETRY
            else:
                columns = list(range(values.shape[1]))
            probability = joblib.load(model_path).predict_proba(values[held][:, columns])[:, 1]
            brier = float(np.mean((probability-y[held])**2))
            if abs(brier-t['brier']) > 1e-12:
                raise ValueError('Saved model score replay differs')
            result = {k: t[k] for k in ('fold', 'seed', 'features', 'model', 'brier', 'prior_brier', 'brier_lift_over_prior', 'auroc', 'average_precision')}
            result.update(version=version, run_balanced=group_balanced_brier(y[held], probability, t['train_only_prior'],
                [(rows[i]['recording_id'], rows[i]['agent_id'], rows[i]['first_row']) for i in held]),
                agent_balanced=group_balanced_brier(y[held], probability, t['train_only_prior'],
                [(rows[i]['recording_id'], rows[i]['agent_id']) for i in held]))
            results.append(result)
    aggregates = []
    for version in ('ordered', 'pooled'):
        for fold in (0, 1):
            for feature in ('geometry', 'geometry_motion'):
                for model in ('logistic', 'extra_trees'):
                    part = [t for t in results if (t['version'], t['fold'], t['features'], t['model']) == (version, fold, feature, model)]
                    if len(part) != 3:
                        raise ValueError('Missing seed in summary')
                    aggregates.append({'version': version, 'held': 'ETH' if fold == 0 else 'Hotel', 'features': feature, 'model': model,
                        'auroc_mean': float(np.mean([t['auroc'] for t in part])),
                        'brier_lift_mean': float(np.mean([t['brier_lift_over_prior'] for t in part])),
                        'agent_balanced_brier_lift_mean': float(np.mean([t['agent_balanced']['brier_lift_over_prior'] for t in part])),
                        'run_balanced_brier_lift_mean': float(np.mean([t['run_balanced']['brier_lift_over_prior'] for t in part])),
                        'positive_brier_seeds': sum(t['brier_lift_over_prior'] > 0 for t in part)})
    summary = {'result_source': 'fresh_run_saved_model_prediction_replay_and_descriptive_group_reweighting',
               'source_hashes': {p.name: file_digest(p) for p in (directory/'metrics.json', directory/'pooled_metrics.json')},
               'analysis_code_sha256': file_digest(Path(__file__)), 'aggregates': aggregates, 'trials': results,
               'saved_model_replays': len(results), 'max_allowed_brier_replay_difference': 1e-12,
               'selection_performed': False, 'independent_confirmation': False}
    (directory/'comparison.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    lines = ['# Stationary-Start Source Audit and Fit-Only Probes', '',
        'Fresh source replay and 48 fitted classifiers; no development, calibration or confirmation labels opened.',
        'The parent eight-observed/twelve-predicted-step task and primary forecasting metric are unchanged.', '',
        '## Source Support', '',
        '| Fit recording | All fit windows | Stationary windows | Changed future | Agents | Stationary runs | Runs with change |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name, a in base['audit'].items():
        lines.append(f"| {name} | {a['fit_windows']} | {a['stationary_windows']} | {a['positive_windows']} | {a['agents']} | {a['stationary_runs']} | {a['runs_with_requested_future_change']} |")
    lines += ['', 'Canonical source positions exactly reproduce every cached row in the five fit recordings.',
        'No malformed source row or history gap occurs here. This proves cache lineage, not true physical stillness',
        'or causal annotation construction. Any future coordinate change is a diagnostic label, not an intent label.',
        'The 365 windows are 31 agents and 45 stopped runs. Repeated seeds and overlapping windows are not independent sites.',
        'Zara has no exact-stationary windows; its stationary held-fold comparison is not_run, not a success.', '',
        '## Fixed Models and Single-Factor Repair', '',
        'Initial ordered context uses 19 geometry or 83 geometry/motion features. The adaptive repair uses',
        'six/13 permutation-invariant summaries. Same rows, folds, labels, seeds, logistic C=1 and ExtraTrees',
        '256/min-leaf10; no threshold search. Feature scaling is observed-query/fit-only. Every fitted setting is shown.',
        'Positive Brier lift means lower probability error than the opposite-scene training-only prior.',
        'The prior rates differ: Hotel->ETH 45.45% versus held 72.84%; ETH->Hotel 72.29% versus held 45.42%.', '',
        '| Features | Held | Model | Context | Mean AUC | Brier lift | Run-balanced lift | Agent-balanced lift | Positive seeds |',
        '| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for r in aggregates:
        lines.append(f"| {r['version']} | {r['held']} | {r['model']} | {r['features']} | {r['auroc_mean']:.4f} | {r['brier_lift_mean']:+.5f} | {r['run_balanced_brier_lift_mean']:+.5f} | {r['agent_balanced_brier_lift_mean']:+.5f} | {r['positive_brier_seeds']}/3 |")
    lines += ['', '## Evidence Boundary', '',
        'Saved model predictions reproduce all 48 reported Brier scores within 1e-12. Run/agent reweighting is descriptive;',
        'it does not establish independence or replace the registered forecasting metric. No confidence interval or',
        'statistical significance claim is made for the two exposed fit sites. Deterministic logistic seeds are identical,',
        'not three independent replications. The pooling follow-up was chosen after seeing the first diagnostic and',
        'is explicitly adaptive. It is not a final-test improvement or a new deployment.', '',
        'Scene image cues, future direction prediction and a start-aware neural trajectory head were not run.',
        'No metric/seconds, true-3D, foundation, Stage5C or SMC claim. The initial serialization failure and repair',
        'are preserved in extraction_failure.md. Full original/follow-up results are in metrics.json and pooled_metrics.json.', '']
    (directory/'results.md').write_text('\n'.join(lines))
    print(json.dumps(aggregates, indent=2))


if __name__ == '__main__':
    main()
