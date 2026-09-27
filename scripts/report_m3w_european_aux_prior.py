"""Fixed intercept-repair contrasts on exposed source-held cost readout."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import run_m3w_european_aux_prior as run
from scripts.report_m3w_european_support_fractional import paired_contrasts
from scripts.report_m3w_european_frozen_harm_readout import summarize_contrasts
from scripts.report_m3w_european_cap_auxiliary_cost import PRIMARY, GUARDS

COMPARISONS = {
    'repair_vs_cost_only': ('new_true', 'cost_only'),
    'repair_vs_old_true': ('new_true', 'old_true'),
    'repair_vs_matched_shuffled': ('new_true', 'new_shuffled'),
    'shuffled_repair_vs_old_shuffled': ('new_shuffled', 'old_shuffled'),
    'old_true_vs_cost_only': ('old_true', 'cost_only')}


def aggregate(rows, cfg, fitting=False):
    contrasts = {}
    for key, (new, old) in COMPARISONS.items():
        proxy = []
        for row in rows:
            folds = []
            for fold in row['folds']:
                values = {k:v['training'] for k,v in fold['training'].items()} if fitting else fold['metrics']
                folds.append(dict(fold, metrics=dict(fractional=values[new], mean=values[old])))
            proxy.append(dict(row, folds=folds))
        contrasts[key] = paired_contrasts(proxy, cfg)
    return dict(contrasts=contrasts, summary=summarize_contrasts(contrasts))


def gates(summary):
    required = [summary[k]['full'] for k in ('repair_vs_cost_only', 'repair_vs_old_true')]
    primary = all(r[PRIMARY]['positive'] == 6 for r in required)
    guard = all(r[k]['negative'] == r[k]['not_estimable'] == 0 for r in required for k in GUARDS)
    information = summary['repair_vs_matched_shuffled']['full'][PRIMARY]['positive'] == 6
    return dict(primary_cost_gate=primary, tail_coverage_all_harm_guards=guard,
        true_vs_shuffled_gate=information, repair_advance_gate=primary and guard and information,
        independent_confirmation=False, deployment_changed=False, new_policy_evaluated=False,
        submission_ready=False, stage5c_executed=False, smc_enabled=False)


def main():
    cfg, _ = run.registration(); run.check_freeze()
    rows = json.loads((run.PUBLIC/'readout.json').read_text())['rows']
    assert len(rows) == 36 and sum(len(r['folds']) for r in rows) == 144
    result = aggregate(rows, cfg); result['gates'] = gates(result['summary'])
    result['fitting_secondary'] = aggregate(rows, cfg, fitting=True)
    run.immutable_json(run.PUBLIC/'aggregate_metrics.json', result)
    freeze = json.loads((run.PUBLIC/'prediction_freeze.json').read_text())
    fits = [json.loads((ROOT/r['path']).read_text()) for r in freeze['receipts']]
    run.immutable_json(run.PUBLIC/'compute_receipt.json', dict(heads=len(fits),
        updates=sum(r['fit']['step'] for r in fits), summed_fit_seconds=sum(r['fit']['seconds'] for r in fits),
        unknown_draws=sum(r['fit']['unknown_rows_sampled'] for r in fits),
        threads=4, interop=1, workers=0, real_torch=True, remote_jobs_submitted=0,
        excludes_loading_inference_preflight_and_verification=True))
    endpoints = [dict(tag=r['identity']['input']['tag'], arm=r['identity']['arm'],
        initial_prior=r['fit']['initial_prior'], inherited_prior=r['fit']['prevalence'],
        first=r['fit']['trace'][0], last=r['fit']['trace'][-1]) for r in fits]
    run.immutable_json(run.PUBLIC/'training_endpoints.json', endpoints)
    lines = ['# Fitting Cap-Prior Intercept Repair', '',
        'fresh_run:288 native Torch heads,576000 optimizer updates,144 exposed source-held readouts.',
        'cached_verified:original cost-only,true/shuffled controls,forecasts,features and nested labels.',
        'not_run:new forecasting or policy,independent selection,reserved calibration,confirmation.', '',
        '| Contrast / family | Positive / negative / overlap / missing CIs | Primary point range (%) |', '|---|---|---|']
    for key, families in result['summary'].items():
        for family, metrics in families.items():
            m = metrics[PRIMARY]
            lines.append(f"| {key}/{family} | {[m[k] for k in ('positive','negative','overlap','not_estimable')]} | {m['point_range']} |")
    lines += ['', '## All Primary Intervals', '', '| Contrast / family / assignment | Point (%) | 95% locality CI |', '|---|---:|---|']
    for key, families in result['contrasts'].items():
        for family, assignments in families.items():
            for assignment, metrics in assignments.items():
                m = metrics[PRIMARY]
                lines.append(f"| {key}/{family}/{assignment} | {m.get('point','not_estimable')} | {m.get('CI','not_estimable')} |")
    lines += ['', '## Interpretation Boundary',
        'Primary is expected easy-harm cost MSE on positive envelopes,not trajectory ADE/FDE or deployment lift.',
        'Three seeds averaged within locality;3000 paired four-locality resamples per assignment.',
        'Six assignments overlap and reuse previously exposed source development;intervals are descriptive and unadjusted.',
        'All 2000-update final checkpoints were frozen before readout. No checkpoint,threshold or role selection.',
        'Only auxiliary intercept changed. Objective,shared/cost initialization,draws,targets and budgets are matched.',
        'Source-held reference costs are frozen. Auxiliary probabilities never multiply expected harm.',
        'Full/motion families also change forecasts/event populations;they are not matched feature ablations.',
        'Fitting secondary summaries average dependent fitting views,not an independent generalization test.', '',
        '```json', json.dumps(result['gates'], indent=2), '```', '',
        'Obs8/pred12 native annotation steps,detector pixels. No metric/seconds,physical-safety,human-gold,true3D or foundation claims.']
    (run.PUBLIC/'results.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(gates=result['gates'],full={k:v['full'][PRIMARY] for k,v in result['summary'].items()}), indent=2))


if __name__ == '__main__': main()
