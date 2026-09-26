"""Registered matched cost contrasts, never selects a model or threshold."""
import csv
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_european_cap_auxiliary_cost as run
from scripts.report_m3w_european_support_fractional import paired_contrasts
from scripts.report_m3w_european_frozen_harm_readout import summarize_contrasts
import numpy as np

COMPARISONS = {'aux_vs_control': ('cap_aux', 'cost_only'), 'aux_vs_original': ('cap_aux', 'original'),
    'aux_vs_shuffled': ('cap_aux', 'shuffled_aux'), 'control_vs_original': ('cost_only', 'original'),
    'shuffled_vs_control': ('shuffled_aux', 'cost_only')}
PRIMARY = 'envelope_positive__harm_MSE_gain_percent'
GUARDS = ('envelope_positive__top10_gain_pp', 'envelope_positive__coverage_log_error_reduction',
          'all__H_all_MSE_gain_percent')


def gates_for(summary):
    required = [summary[k]['full'] for k in ('aux_vs_control', 'aux_vs_original')]
    primary = all(r[PRIMARY]['positive'] == 6 for r in required)
    guards = all(r[k]['negative'] == r[k]['not_estimable'] == 0 for r in required for k in GUARDS)
    information = summary['aux_vs_shuffled']['full'][PRIMARY]['positive'] == 6
    return dict(primary_cost_gate=primary, tail_coverage_all_harm_guards=guards,
        true_vs_shuffled_gate=information, auxiliary_cost_contribution=primary and guards and information,
        new_policy_evaluated=False, independent_confirmation=False, deployment_changed=False,
        submission_ready=False, stage5c_executed=False, smc_enabled=False)


def aggregates(rows, cfg):
    contrasts = {}
    for key, (new, old) in COMPARISONS.items():
        proxy = [dict(r, folds=[dict(f, metrics=dict(fractional=f['metrics'][new], mean=f['metrics'][old]))
                               for f in r['folds']]) for r in rows]
        contrasts[key] = paired_contrasts(proxy, cfg)
    summary = summarize_contrasts(contrasts)
    event = {}
    for pair in cfg['pairs']:
        event[pair] = {}
        for arm in cfg['arms']:
            metrics = [f['event_metrics'][arm] for r in rows if r['pair'] == pair for f in r['folds']]
            supported = [m['AUROC'] for m in metrics if m.get('AUROC') is not None]
            event[pair][arm] = dict(dependent_views=len(metrics),
                median_supported_AUROC=float(np.median(supported)) if supported else None,
                unavailable_ranking=sum(m.get('AP') is None for m in metrics),
                BCE_better_than_prior=sum(m['BCE'] < m['prior_BCE'] for m in metrics))
    return dict(contrasts=contrasts, summary=summary, event_diagnostics=event, gates=gates_for(summary))


def main():
    cfg, _ = run.registration(); run.check_freeze()
    rows = json.loads((run.PUBLIC/'readout.json').read_text())['rows']
    assert len(rows) == 36 and sum(len(r['folds']) for r in rows) == 144
    result = aggregates(rows, cfg)
    run.immutable_json(run.PUBLIC/'aggregate_metrics.json', result)
    freeze = json.loads((run.PUBLIC/'prediction_freeze.json').read_text())
    fits, endpoints = [], []
    for ref in freeze['receipts']:
        receipt = json.loads((ROOT/ref['path']).read_text()); h = receipt['identity']; f = receipt['fit']
        row = dict(tag=h['input']['tag'], seed=h['seed'], arm=h['arm'], fit=f)
        fits.append(row)
        endpoints.append(dict(tag=row['tag'], arm=row['arm'], seed=row['seed'], updates=f['step'],
            first_cost_loss=f['trace'][0]['cost_loss'], last_cost_loss=f['trace'][-1]['cost_loss'],
            first_true_cap_BCE=f['trace'][0]['true_cap_BCE'], last_true_cap_BCE=f['trace'][-1]['true_cap_BCE'],
            fit_seconds=f['seconds']))
    run.immutable_json(run.PUBLIC/'training_metrics.json', fits)
    run.immutable_json(run.PUBLIC/'compute_receipt.json', dict(heads=len(fits),
        updates=sum(r['fit']['step'] for r in fits), summed_fit_seconds=sum(r['fit']['seconds'] for r in fits),
        unknown_draws=sum(r['fit']['unknown_rows_sampled'] for r in fits), parameters=12899,
        threads=4, interop=1, workers=0, real_torch=True, remote_jobs_submitted=0,
        excludes_loading_inference_preflight_and_verification=True))
    with (run.PUBLIC/'training_loss_endpoints.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(endpoints[0])); writer.writeheader(); writer.writerows(endpoints)
    lines = ['# Cap-Event Auxiliary Cost Results', '', '## Material Passport',
        'fresh_run:432 native-Torch heads,864000 fixed updates and144 source-held readouts.',
        'cached_verified:source forecasters, nested risk producers and original cost estimators.',
        'not_run:new trajectory training, policy evaluation, independent selection/calibration/confirmation.', '',
        '| Contrast / inputs | Easy-harm MSE: positive / negative / overlap / missing | Point range (%) |', '|---|---|---|']
    for key, pairs in result['summary'].items():
        for pair, metrics in pairs.items():
            m = metrics[PRIMARY]
            lines.append(f"| {key}/{pair} | {[m[k] for k in ('positive','negative','overlap','not_estimable')]} | {m['point_range']} |")
    lines += ['', '## All Assignment Intervals', '', '| Contrast / inputs / assignment | Point (%) | 95% locality CI |', '|---|---:|---|']
    for key, pairs in result['contrasts'].items():
        for pair, assignments in pairs.items():
            for assignment, metrics in assignments.items():
                m = metrics[PRIMARY]
                lines.append(f"| {key}/{pair}/{assignment} | {m.get('point','not_estimable')} | {m.get('CI','not_estimable')} |")
    lines += ['', 'Three seeds averaged within locality, then3000 paired resamples of four localities.',
        'Six assignments overlap; source development has prior exposure. Intervals are exploratory, not multiplicity-adjusted.',
        'Full/motion changes forecasts and event populations; not a matched feature ablation. Unknown support is retained.',
        'Primary requires positive full-input cost intervals against control and original, with tail/coverage/all-harm guards.',
        'Task-information additionally requires improvement against locality-shuffled auxiliary. Event classification alone cannot pass.',
        'New matched controls have399 inputs and two learned harm costs; the frozen original is a stronger historical comparator, not an identical control.',
        'Event probability never multiplies expected harm. The zero-reference guard is unchanged. No policy lift claimed.', '',
        '```json', json.dumps(result['gates'], indent=2), '```', '',
        'Obs8/pred12 native annotation steps, detector pixels; no metric/seconds, human-gold, physical-safety, true3D or foundation claim.']
    (run.PUBLIC/'results.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(gates=result['gates'], full={k: v['full'][PRIMARY] for k, v in result['summary'].items()}), indent=2))


if __name__ == '__main__': main()
