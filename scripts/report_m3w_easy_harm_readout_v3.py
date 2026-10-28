"""Report the complete frozen loss comparison, retaining all failed contrasts."""
import argparse
import csv
import io
import json
from pathlib import Path

from scripts import run_m3w_easy_harm_readout_v3 as run

HOME = run.PUBLIC


def validate():
    cfg, reg = run.registration()
    if reg != json.loads((HOME / 'registration.json').read_text()):
        raise ValueError('Changed registered readout')
    run.admission(reg)
    complete = json.loads((HOME / 'complete.json').read_text())
    summary = json.loads((HOME / 'summary.json').read_text())
    if (complete['summary_sha256'] != run.sha(HOME / 'summary.json')
            or complete['training_freeze_sha256'] != run.sha(run.HOME / 'training_freeze.json')
            or not complete['exact_inference_replay'] or not complete['strong_control_replay_exact']
            or any(complete[k] is not False for k in
                   ('new_training','independent_roles_read','deployment_changed','local_checkpoint_cache'))):
        raise ValueError('Complete exact non-deployed readout required')
    rows, seen = [], set()
    for ref in complete['groups']:
        path = run.ROOT / ref['path']
        if (not path.resolve().is_relative_to((HOME / 'groups').resolve())
                or ref['path'] in seen or run.sha(path) != ref['sha256']):
            raise ValueError('Missing, duplicate or altered group')
        row = json.loads(path.read_text())
        if (row['readout_registration_sha256'] != run.sha(HOME / 'registration.json')
                or row['training_freeze_sha256'] != complete['training_freeze_sha256']
                or not row['exact_inference_replay'] or not row['strong_control_replay_exact']
                or row['new_training'] or row['independent_roles_read'] or row['deployment_changed']):
            raise ValueError('Changed group evidence role or provenance')
        seen.add(ref['path'])
        rows.append(row)
    _, old_reg = run.previous.registration()
    recomputed = run.original.api.summarize(rows, old_reg['expected'], cfg)
    if recomputed != summary:
        raise ValueError('Paired-locality summary does not reproduce')
    checks = sum(r['independent_scalar_checks'] for r in rows)
    checks += run.original.scalar.verify_summary(rows, cfg, summary)
    if checks != complete['independent_scalar_checks']:
        raise ValueError('Scalar validation count differs')
    return cfg, complete, summary, rows


def interval(value):
    if value['mean'] is None or value['CI95'] is None:
        return 'undefined; support retained'
    return f"{value['mean']:+.6f} [{value['CI95'][0]:+.6f}, {value['CI95'][1]:+.6f}]"


def tables(summary):
    contrasts = ['| Comparator | All signed-score MSE delta | Full paired lower utility delta (pp) | Same-count paired lower utility delta (pp) |',
                 '|---|---:|---:|---:|']
    for comparator in run.original.api.COMPARATORS:
        contrasts.append('| '+comparator+' | '+' | '.join(interval(summary['contrasts'][comparator+'_'+metric])
            for metric in ('all_MSE','full_paired_lower_percent','matched_paired_lower_percent'))+' |')
    safety = ['| Policy | Selected occurrences | Unknown selected | Undefined easy risk | Known-risk violations | Completion-upper violations | Finite supported views /72 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for name, row in summary['safety'].items():
        safety.append('| '+name+' | '+' | '.join(str(row[k]) for k in
            ('selected_occurrences','unknown_selected_occurrences','undefined_easy_risk',
             'known_easy_violations','easy_upper_violations','finite_completion_supported'))+' |')
    return '\n'.join(contrasts), '\n'.join(safety)


def artifacts(cfg, complete, summary, rows):
    contrasts, safety = tables(summary)
    total = sum(r['result']['rows'] for r in rows)
    unknown = sum(r['result']['unknown_rows'] for r in rows)
    verdict = ('passes the registered development advancement rule; independent confirmation remains required'
               if summary['advance_to_transfer_design'] else
               'fails the registered advancement rule; no deployment or independent-evaluation promotion')
    failures = '\n'.join('- '+r for r in summary['failure_reasons']) or '- None for this development rule.'
    text = f'''# Easy-Harm Deviance: Complete Fixed-Final Readout

## Result and Scope

The one-factor cost-loss repair **{verdict}**.
Result source: fresh_run fixed-final exposed-development inference and paired
locality bootstrap; cached_verified training and frozen forecast producers.
This report assembles those completed results. It is not independent confirmation.

All 72 registered source/head views and 12 localities are included, with three
cost-head seeds and 3,000 paired-locality bootstrap draws. There are {total:,}
row occurrences, including {unknown:,} unknown-outcome occurrences. Repeated views
and overlapping windows are not independent samples or additional datasets.

## What Changed

Only the fifth positive-easy-harm moment loss changes from quadratic error to
normalized continuous-cost deviance. The remaining losses, inputs, sampling,
2,000-step budget, frozen producers and decision rule are unchanged. The 144 final
fits comprise 72 controls and 72 candidates. Of the controls, 54 reproduce historical
states exactly; 18 use explicitly registered original-trainer replays. This
does not erase the failed historical bitwise reproduction. Recovery freshly
trained 17 missing candidates, not 144 new fits in this reporting round.

## All Strong Comparisons

Every entry is candidate minus comparator. Negative MSE is better; positive
paired lower utility is better. Utility is a past-normalized, bounded evaluation
quantity, not raw FDE, t+50 improvement or physical safety. The four primary
comparisons are full/same-count versus quadratic and original.

{contrasts}

Intervals use paired differences after averaging repeated views within each
locality. They are nominal development intervals, not independent confirmatory
tests or simultaneous guarantees over the many exploratory experiments.
Undefined contrasts remain undefined; no unfavorable comparator is removed.

## Selected-Risk and Missing-Outcome Checks

The 2% budget is positive easy harm divided by easy reference error on the
selected population. It is **not** net easy ADE degradation. Missing outcomes
are retained in completion bounds and never imputed safe. Empty or unsupported
cohorts do not pass by default. All matched policies are shown below.

{safety}

## Gate Failures

{failures}

## Verification and Limits

Fresh verification job 38287008 completed with exit code 0:0 and rehashed all 144 checkpoint
payloads plus 145 metadata files. Historical join accounting remains unavailable;
the explicit execution amendment did not fabricate it. The original evaluator
and hash-checked cache-free SSH reader were reused unchanged. Exact repeated
inference, exact strong-control replay and {complete['independent_scalar_checks']:,}
independent scalar checks passed. Readout time was {complete['seconds']:.2f}s.
This is algorithm/artifact evidence, not proof of scientific generalization.

No new training, threshold search, checkpoint selection, independent label access
or deployment change occurred. The unchanged scaffold is 2.5D, not true 3D or a
foundation world model; no metric or seconds-level conversion is established.
Silver/inferred labels are not human gold. Stage5C and SMC remain off.

## Next Decision

Use the complete primary-contrast and selected-risk results, not a single better
MSE or favorable comparator, to decide the next separately registered experiment.
Matched decision-focused controls remain unrun. Their objective must be separated
from risk calibration, preserve the same forecast producer and fit-only roles,
and retain unknown outcomes. No automatic new tuning follows this report.
'''
    stream = io.StringIO(newline='')
    writer = csv.writer(stream, lineterminator='\n')
    writer.writerow(['contrast','mean','ci_low','ci_high','primary','evidence_role'])
    for key, value in summary['contrasts'].items():
        ci = value['CI95'] or [None,None]
        writer.writerow([key,value['mean'],*ci,key in summary['primary_contrasts'],'exposed_development'])
    receipt = dict(result_source='fresh_run_report_and_summary_reverification',
        summary_sha256=run.sha(HOME/'summary.json'),complete_sha256=run.sha(HOME/'complete.json'),
        registration_sha256=run.sha(HOME/'registration.json'),
        groups=len(rows),localities=12,row_occurrences=total,unknown_row_occurrences=unknown,
        bootstrap_resamples=3000,summary_recomputed_exact=True,
        advance=summary['advance_to_transfer_design'],failure_reasons=summary['failure_reasons'],
        historical_accounting='unavailable_not_inferred',independent_confirmation=False,
        deployment_changed=False,submission_ready=False,stage5c_executed=False,smc_enabled=False)
    return {'conclusions.md':text,'contrasts.csv':stream.getvalue(),
            'report_verification.json':json.dumps(receipt,indent=2,allow_nan=False)+'\n'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    content = artifacts(*validate())
    for name, raw in content.items():
        path = HOME / name
        if args.check:
            if path.read_bytes() != raw.encode():
                raise ValueError('Changed report: '+name)
        elif path.exists():
            if path.read_bytes() != raw.encode():
                raise ValueError('Preserve existing differing report: '+name)
        else:
            path.write_bytes(raw.encode())
    print(json.dumps(dict(status='checked' if args.check else 'reported',
                         scientific_inference='cached_verified_complete',new_training=False)))


if __name__ == '__main__':
    main()
