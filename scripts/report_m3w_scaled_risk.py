"""Verify the complete scaling readout and report risk and utility without rescue."""
import argparse
import csv
import io
import json
import statistics

from scripts import run_m3w_scaled_risk_readout as run

HOME=run.PUBLIC


def validate():
    cfg,reg=run.registration()
    if reg!=json.loads((HOME/'registration.json').read_text()):raise ValueError('Readout binding drift')
    run.heads.admission()
    complete=json.loads((HOME/'complete.json').read_text());summary=json.loads((HOME/'summary.json').read_text())
    if (complete['summary_sha256']!=run.sha(HOME/'summary.json')
            or complete['training_freeze_sha256']!=run.sha(run.manager.PUBLIC/'training_freeze.json')
            or not complete['exact_inference_replay'] or not complete['strong_control_replay_exact']
            or complete['new_training'] or complete['independent_roles_read'] or complete['deployment_changed']):
        raise ValueError('Complete unchanged result required')
    rows=[];seen=set()
    for ref in complete['groups']:
        path=run.ROOT/ref['path']
        if (path.is_symlink() or not path.resolve().is_relative_to((HOME/'groups').resolve())
                or ref['path'] in seen or run.sha(path)!=ref['sha256']):raise ValueError('Changed group')
        row=json.loads(path.read_text())
        if (row['readout_registration_sha256']!=run.sha(HOME/'registration.json')
                or row['training_freeze_sha256']!=complete['training_freeze_sha256']
                or not row['exact_inference_replay'] or not row['strong_control_replay_exact']
                or not row['cached_neural_control_predictions_exact']
                or row['new_training'] or row['independent_roles_read'] or row['deployment_changed']):
            raise ValueError('Changed group provenance')
        seen.add(ref['path']);rows.append(row)
    if run.api.summarize(rows,reg['expected'],cfg)!=summary:raise ValueError('Summary replay failed')
    checks=sum(row['independent_scalar_checks'] for row in rows)+run.scalar.verify_summary(rows,cfg,summary)
    if checks!=complete['independent_scalar_checks']:raise ValueError('Scalar count differs')
    return cfg,complete,summary,rows


def diagnostics(rows):
    result={}
    for arm in run.heads.NEURAL_ARMS:
        ds=[r['training_diagnostics'][arm] for r in rows]
        ratios=[r['result']['diagnostics'][arm]['predicted_observed_easy_harm_ratio'] for r in rows]
        valid=[v for v in ratios if v is not None]
        result[arm]=dict(views=len(rows),dual_cap_views=sum(d['dual_cap_reached'] for d in ds),
            empty_selection_views=sum(d['empty_selection'] for d in ds),
            TRAIN_soft_all_excess_views=sum(d['final_TRAIN_monitor']['soft_all_excess']>0 for d in ds),
            TRAIN_soft_easy_excess_views=sum(d['final_TRAIN_monitor']['soft_easy_excess']>0 for d in ds),
            mean_soft_action=statistics.mean(d['development_soft_action_mean'] for d in ds),
            mean_hard_intervention=statistics.mean(d['hard_intervention_rate'] for d in ds),
            mean_soft_hard_disagreement=statistics.mean(d['soft_at_least_half_hard_disagreement'] for d in ds),
            underprediction_views=sum(v<1 for v in valid),undefined_harm_ratio_views=len(rows)-len(valid),
            median_predicted_actual_easy_harm=statistics.median(valid) if valid else None)
    return result


def interval(value):
    if value['mean'] is None or value['CI95'] is None:return 'undefined; support retained'
    return f"{value['mean']:+.6f} [{value['CI95'][0]:+.6f}, {value['CI95'][1]:+.6f}]"


def artifacts(cfg,complete,summary,rows):
    ds=diagnostics(rows)
    utility=['| Control | Signed-score MSE delta | Full paired lower utility delta (pp) | Same-count delta (pp) |',
             '|---|---:|---:|---:|']
    for name in run.api.COMPARATORS:
        utility.append('| '+name+' | '+' | '.join(interval(summary['contrasts'][name+'_'+field])
            for field in ('all_MSE','full_paired_lower_percent','matched_paired_lower_percent'))+' |')
    risks=['| Policy | Selected occurrences | Known easy violations | Upper violations | Finite support /72 |',
           '|---|---:|---:|---:|---:|']
    for name,v in summary['safety'].items():
        risks.append('| '+name+' | '+' | '.join(str(v[k]) for k in ('selected_occurrences','known_easy_violations',
            'easy_upper_violations','finite_completion_supported'))+' |')
    mechanics=['| Arm | TRAIN all excess | TRAIN easy excess | Dual-cap views | Empty views | Mean soft/hard disagreement |',
               '|---|---:|---:|---:|---:|---:|']
    for name,v in ds.items():
        mechanics.append(f"| {name} | {v['TRAIN_soft_all_excess_views']} | {v['TRAIN_soft_easy_excess_views']} | {v['dual_cap_views']} | {v['empty_selection_views']} | {v['mean_soft_hard_disagreement']:.6f} |")
    verdict='passes development advancement; independent confirmation still required' if summary['advance_to_transfer_design'] else 'fails the registered development advancement rule'
    failures='\n'.join('- '+s for s in summary['failure_reasons']) or '- None under this development rule.'
    total=sum(r['result']['rows'] for r in rows);unknown=sum(r['result']['unknown_rows'] for r in rows)
    content=f'''# Fixed TRAIN Constraint Scaling: Complete Results

## Verdict

The candidate **{verdict}**. No deployment change follows. This is exposed
development evidence, not independent confirmation or submission readiness.

## What Ran

144 fresh fits: 72 unit controls and 72 budget-scaled heads, each 2,000 updates.
All 72 unit controls exactly reproduce the prior task-risk model, optimizer,
sampler, duals and traces. The quadratic/utility heads are 144 cached-verified
controls, not newly trained. Four forest controls remain visible. Forecast
producers, features, initialization, draws, loss components, temperature, hard
rule and risk definition stay fixed; only positive TRAIN constraint units change
in the primal and dual updates. Zero-support scaling is not imputed safe.

All 72 views over 12 localities complete, with {total:,} repeated row occurrences
and {unknown:,} unknown-outcome occurrences. Observation8/prediction12 and
dataset-local coordinates are unchanged; these are not t+50 or metric results.

## Full and Same-Count Comparisons

Candidate minus control. Positive paired lower utility is favorable; negative
MSE is favorable. Units are normalized utility percentage points, not ADE/FDE.
All eight full/matched contrasts against unit, quadratic, utility and original
forest are primary. A weak control cannot rescue a failed strong control.

{chr(10).join(utility)}

Intervals use 3,000 paired-locality draws after averaging repeated views within
locality. They are nominal, not simultaneous or independent confirmation.

## Risk and Support

The 2% limit concerns positive selected easy harm/reference error, not net easy
ADE degradation. Unknown outcomes remain in completion bounds. Empty or
unsupported cohorts do not pass. Original-unit constraints are reported below.

{chr(10).join(risks)}

## Mechanism Checks

{chr(10).join(mechanics)}

TRAIN counts refer to fixed 128-query monitors, not the full TRAIN distribution.
The soft-at-least-0.5 comparison is diagnostic, not calibrated confidence or a
new threshold. A zero dual at the final step is not evidence that its constraint
always held. Total losses differ across arms and must not be ranked directly.
Scaling changes conditioning and effective dual capacity, not the feasible set.
Fewer violations with no intervention do not establish a useful world model.
The unit-control comparison isolates the scaling intervention; it does not prove
that optimization scale is the sole cause of any remaining generalization gap.

## Advancement Failures

{failures}

## Verification and Reproduction

All model, input, split and role bindings verify. Repeated inference is exact,
all forest controls replay, and all three prior neural prediction hashes match.
Independent scalar checks total {complete['independent_scalar_checks']:,}; readout
took {complete['seconds']:.2f}s. The summary replays exactly from frozen aggregates.
No checkpoint selection, threshold sweep or independent-role access occurred.
Training is native arm64 Torch, four compute threads and zero loader workers.
Models/checkpoints and raw data stay outside Git; light receipts and metrics remain.

```bash
.venv-pytorch/bin/python -m scripts.report_m3w_scaled_risk --check
```

This checks existing results, not new training. The full method remains a 2.5D
world-state scaffold; this cost-head repair is not new trajectory dynamics,
foundation capability, true3D, physical safety or seconds-level evidence.
Silver labels are not human gold. Stage5C and SMC remain disabled.
'''
    stream=io.StringIO(newline='');writer=csv.writer(stream,lineterminator='\n')
    writer.writerow(['contrast','mean','ci_low','ci_high','primary','role'])
    for k,v in summary['contrasts'].items():writer.writerow([k,v['mean'],*(v['CI95'] or [None,None]),k in summary['primary_contrasts'],'exposed_development'])
    receipt=dict(result_source='fresh_run_aggregate_reverification',summary_sha256=run.sha(HOME/'summary.json'),
        complete_sha256=run.sha(HOME/'complete.json'),training_freeze_sha256=complete['training_freeze_sha256'],
        advance=summary['advance_to_transfer_design'],independent_confirmation=False,deployment_changed=False,
        submission_ready=False,stage5c_executed=False,smc_enabled=False)
    return {'conclusions.md':content,'contrasts.csv':stream.getvalue(),
        'decision_diagnostics.json':json.dumps(ds,indent=2,allow_nan=False)+'\n',
        'report_verification.json':json.dumps(receipt,indent=2,allow_nan=False)+'\n'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check',action='store_true');a=p.parse_args()
    for name,text in artifacts(*validate()).items():
        path=HOME/name
        if a.check or path.exists():
            if path.read_bytes()!=text.encode():raise ValueError('Preserve conflicting report')
        else:
            with path.open('xb') as f:f.write(text.encode())
    print(json.dumps(dict(status='checked' if a.check else 'reported',scientific_readout='cached_verified')))


if __name__=='__main__':main()
