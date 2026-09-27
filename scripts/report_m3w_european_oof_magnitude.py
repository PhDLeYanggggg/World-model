"""Fixed matched contrasts for the OOF magnitude experiment."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import run_m3w_european_oof_magnitude as run
from scripts.report_m3w_european_support_fractional import paired_contrasts
from scripts.report_m3w_european_frozen_harm_readout import summarize_contrasts
from scripts.report_m3w_european_cap_auxiliary_cost import PRIMARY, GUARDS

COMPARISONS = {
    'scaled_true_vs_scaled_cost': ('cap_aux_scaled', 'cost_only_scaled'),
    'scaled_true_vs_raw_true': ('cap_aux_scaled', 'cap_aux_raw'),
    'scaled_true_vs_scaled_shuffled': ('cap_aux_scaled', 'shuffled_aux_scaled'),
    'scaled_cost_vs_raw_cost': ('cost_only_scaled', 'cost_only_raw'),
    'scaled_shuffled_vs_raw_shuffled': ('shuffled_aux_scaled', 'shuffled_aux_raw')}


def aggregate(rows, cfg):
    contrasts = {}
    for key, (new, old) in COMPARISONS.items():
        proxy = [dict(row, folds=[dict(fold, metrics=dict(fractional=fold['metrics'][new],
            mean=fold['metrics'][old])) for fold in row['folds']]) for row in rows]
        contrasts[key] = paired_contrasts(proxy, cfg)
    summary = summarize_contrasts(contrasts)
    main = [summary[k]['full'] for k in ('scaled_true_vs_scaled_cost', 'scaled_true_vs_raw_true')]
    primary = all(r[PRIMARY]['positive'] == 6 for r in main)
    guard = all(r[k]['negative'] == r[k]['not_estimable'] == 0 for r in main for k in GUARDS)
    information = summary['scaled_true_vs_scaled_shuffled']['full'][PRIMARY]['positive'] == 6
    gates = dict(primary_cost_gate=primary, guard_gate=guard, information_gate=information,
        advance_gate=primary and guard and information, deployment_changed=False,
        independent_confirmation=False, submission_ready=False, stage5c_executed=False, smc_enabled=False)
    return dict(contrasts=contrasts, summary=summary, gates=gates)


def main():
    cfg, _ = run.registration(); run.check_freeze()
    readout = json.loads((run.PUBLIC/'readout.json').read_text())
    result = aggregate(readout['rows'], cfg); run.immutable_json(run.PUBLIC/'aggregate_metrics.json', result)
    frozen = json.loads((run.PUBLIC/'inner_prediction_freeze.json').read_text())
    fits = [json.loads((ROOT/r['path']).read_text()) for r in frozen['references']+frozen['inner_heads']]
    run.immutable_json(run.PUBLIC/'compute_receipt.json', dict(fresh_heads=len(fits),
        updates=sum(r['fit']['step'] for r in fits), recorded_fit_seconds=sum(r['fit']['seconds'] for r in fits),
        unknown_draws=sum(r['fit']['unknown_rows_sampled'] for r in fits), cached_inner_controls=432,
        threads=4, interop=1, workers=0, excludes_preflight_IO_and_readout=True, remote_jobs_submitted=0))
    lines = ['# Honest OOF Magnitude Readout', '',
        'fresh_run:1008 native Torch nuisance/auxiliary heads;2016000 optimizer updates;432 two-slope readouts.',
        'cached_verified:432 cost-only inner controls,144 outer heads per arm,forecasts and causal features.',
        'not_run:new trajectory or policy,independent selection,reserved calibration,confirmation.', '',
        '| Contrast/family | Positive / negative / overlap / missing CIs | Primary range (%) |', '|---|---|---|']
    for key, families in result['summary'].items():
        for family, metrics in families.items():
            m = metrics[PRIMARY]
            lines.append(f"| {key}/{family} | {[m[k] for k in ('positive','negative','overlap','not_estimable')]} | {m['point_range']} |")
    lines += ['', '## Assignment Intervals', '', '| Contrast/family/assignment | Point (%) | 95% CI |', '|---|---:|---|']
    for key, families in result['contrasts'].items():
        for family, assignments in families.items():
            for name, metrics in assignments.items():
                m = metrics[PRIMARY]
                lines.append(f"| {key}/{family}/{name} | {m.get('point')} | {m.get('CI')} |")
    lines += ['', '## Boundaries',
        'Primary is positive-envelope expected easy-harm cost MSE,not FDE/ADE or easy degradation.',
        'Three seeds averaged within locality;3000 paired four-locality resamples per assignment.',
        'Assignments overlap;source-held rows were exposed historically. All CIs are descriptive and unadjusted.',
        'Single-site reference to two-site OOF head to three-site outer head introduces training-size and easy-cut drift.',
        'No labels from inner/outer held localities entered their producer lineage. OOF targets use producer-specific easy cuts.',
        'Origin least squares is followed by nested causal-envelope clipping;it is not a conformal or safety guarantee.',
        'Obs8/pred12 native steps,detector pixels. No metric/seconds,true3D,foundation or human-gold claims.', '',
        '```json', json.dumps(result['gates'], indent=2), '```']
    (run.PUBLIC/'results.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(gates=result['gates'], full={k:v['full'][PRIMARY] for k,v in result['summary'].items()}), indent=2))


if __name__ == '__main__': main()
