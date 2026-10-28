"""Reverify and report matched decision-objective results without promotion by MSE."""
import argparse
import csv
import io
import json
import statistics

from scripts import run_m3w_task_risk_readout as run

HOME=run.PUBLIC


def validate():
    cfg,reg=run.registration()
    if reg!=json.loads((HOME/'registration.json').read_text()):raise ValueError('Readout binding drift')
    run.heads.local_admission()
    complete=json.loads((HOME/'complete.json').read_text());summary=json.loads((HOME/'summary.json').read_text())
    if (complete['summary_sha256']!=run.sha(HOME/'summary.json')
            or complete['training_freeze_sha256']!=run.sha(run.heads.manager.PUBLIC/'training_freeze.json')
            or not complete['exact_inference_replay'] or not complete['strong_control_replay_exact']
            or complete['new_training'] or complete['independent_roles_read'] or complete['deployment_changed']):
        raise ValueError('Complete unchanged readout required')
    rows=[];seen=set()
    for ref in complete['groups']:
        p=run.ROOT/ref['path']
        if (p.is_symlink() or not p.resolve().is_relative_to((HOME/'groups').resolve())
                or ref['path'] in seen or run.sha(p)!=ref['sha256']):
            raise ValueError('Changed or duplicate readout group')
        r=json.loads(p.read_text())
        if (r['readout_registration_sha256']!=run.sha(HOME/'registration.json')
                or r['training_freeze_sha256']!=complete['training_freeze_sha256']
                or not r['exact_inference_replay'] or not r['strong_control_replay_exact']
                or r['new_training'] or r['independent_roles_read'] or r['deployment_changed']):
            raise ValueError('Changed group role/provenance')
        seen.add(ref['path']);rows.append(r)
    if run.api.summarize(rows,reg['expected'],cfg)!=summary:raise ValueError('Summary failed exact replay')
    checks=sum(r['independent_scalar_checks'] for r in rows)+run.scalar.verify_summary(rows,cfg,summary)
    if checks!=complete['independent_scalar_checks']:raise ValueError('Scalar verification count mismatch')
    return cfg,complete,summary,rows


def diagnostics(rows):
    output={}
    for arm in run.fitting.ARMS:
        ds=[r['training_diagnostics'][arm] for r in rows]
        harms=[r['result']['diagnostics'][arm] for r in rows]
        ratios=[d['predicted_observed_easy_harm_ratio'] for d in harms
                if d['predicted_observed_easy_harm_ratio'] is not None]
        output[arm]=dict(views=len(rows),dual_cap_views=sum(d['dual_cap_reached'] for d in ds),
            empty_selection_views=sum(d['empty_selection'] for d in ds),
            mean_final_dual=[statistics.mean(d['dual'][i] for d in ds) for i in range(2)],
            mean_soft_action=statistics.mean(d['development_soft_action_mean'] for d in ds),
            mean_hard_intervention=statistics.mean(d['hard_intervention_rate'] for d in ds),
            mean_soft_hard_disagreement=statistics.mean(d['soft_at_least_half_hard_disagreement'] for d in ds),
            selected_easy_harm_underprediction_views=sum(v<1 for v in ratios),
            undefined_selected_harm_ratio_views=len(rows)-len(ratios),
            median_predicted_actual_selected_harm_ratio=statistics.median(ratios) if ratios else None,
            median_final_TRAIN_decision_regret=statistics.median(d['final_TRAIN_monitor']['decision_regret'] for d in ds),
            calibrated_probability_claim=False,causal_attribution_proven=False)
    return output


def interval(v):
    if v['mean'] is None or v['CI95'] is None:return 'undefined; support retained'
    return f"{v['mean']:+.6f} [{v['CI95'][0]:+.6f}, {v['CI95'][1]:+.6f}]"


def artifacts(cfg,complete,summary,rows):
    d=diagnostics(rows);total=sum(r['result']['rows'] for r in rows);unknown=sum(r['result']['unknown_rows'] for r in rows)
    table=['| Comparator | Signed-score MSE delta | Full paired lower utility delta (pp) | Same-count delta (pp) |',
           '|---|---:|---:|---:|']
    for name in run.api.COMPARATORS:
        table.append('| '+name+' | '+' | '.join(interval(summary['contrasts'][name+'_'+field])
                     for field in ('all_MSE','full_paired_lower_percent','matched_paired_lower_percent'))+' |')
    risk=['| Policy | Selected occurrences | Known easy-risk violations | Upper-bound violations | Finite supported views /72 |',
          '|---|---:|---:|---:|---:|']
    for name,v in summary['safety'].items():
        risk.append('| '+name+' | '+' | '.join(str(v[k]) for k in ('selected_occurrences',
            'known_easy_violations','easy_upper_violations','finite_completion_supported'))+' |')
    detail=['| Arm | Dual-cap views | Empty selections | Mean soft/hard disagreement | Median predicted/actual easy harm | Underprediction views |',
            '|---|---:|---:|---:|---:|---:|']
    for arm,v in d.items():
        ratio=v['median_predicted_actual_selected_harm_ratio']
        detail.append(f"| {arm} | {v['dual_cap_views']} | {v['empty_selection_views']} | {v['mean_soft_hard_disagreement']:.6f} | {ratio} | {v['selected_easy_harm_underprediction_views']} |")
    verdict='passes the registered development rule; confirmation still required' if summary['advance_to_transfer_design'] else 'fails the registered development advancement rule'
    failures='\n'.join('- '+s for s in summary['failure_reasons']) or '- None under this development rule.'
    text=f'''# Matched Decision-Risk Training: Complete Results

## Verdict

The task-risk candidate **{verdict}**. Deployment is unchanged. This is exposed
source-development evidence, not independent confirmation or submission readiness.

## What Actually Ran

Fresh TRAIN fits:216, comprising72 contemporary quadratic controls,72 utility
heads and72 utility-plus-risk heads, each2,000updates. All share input packets,
initialization, known-query sampling and producer forecasts. Three head seeds
are not three independent forecasters. No historical quadratic checkpoint is
substituted into this comparison. Native arm64 CPU executes the registered
optimizer; the real-data pilot proves original-control and interrupted/direct
resume equality in this environment. No cross-platform identity is claimed.

CREATE preflight stopped before submission because the10GiB reserve would be
violated. Its4.12GiB free space was not bypassed. The local runner streams TRAIN
packets without a disk cache. Original transport/path failures and amendments
remain recorded. This is actual local training, not a CREATE or NumPy-fallback claim.

Fresh fixed-final readout includes all72 views over12 localities, {total:,} repeated
row occurrences and {unknown:,} unknown-outcome occurrences. Observation8/prediction12
and dataset-local coordinates retain their original definition. They are not
raw-frame t+50 improvement, seconds-level prediction or metric-world accuracy.

## All Comparators

Entries are task-risk minus comparator; negative MSE and positive paired lower
utility are favorable. The six primary tests are full/matched against quadratic,
task-utility and original. A better forest comparison cannot override a failed
contemporary control, and a lower training loss cannot override risk.

{chr(10).join(table)}

Nominal95%intervals use3,000 paired-locality bootstrap draws, averaging repeated
views within locality first. They are not independent confirmatory or simultaneous
guarantees. All strong controls remain visible.

## Risk and Missingness

The2%budget means positive selected easy harm/reference error, not net easy ADE.
Unknown outcomes retain completion bounds; empty or unsupported cohorts do not
pass by default. All full and same-count policies are retained below.

{chr(10).join(risk)}

## Training/Decision Diagnostics

{chr(10).join(detail)}

Soft-action0.5 disagreement is descriptive only, not a tuned deployment threshold
or a calibrated confidence claim. Raw total losses across arms are incomparable
because task and dual terms differ. Dual caps, underprediction and soft/hard
mismatch describe mechanisms to investigate, not identified causal explanations.

## Failed Gates

{failures}

## Verification and Next Decision

All checkpoint/receipt/input/role bindings verify; repeated inference and strong
control replay are exact. There are{complete['independent_scalar_checks']:,}
independent scalar checks; readout elapsed{complete['seconds']:.2f}s. The reporter
recomputes the locality summary exactly from frozen group aggregates.

No validation checkpoint selection, threshold search, new forecast producer,
independent selection/calibration/confirmation access or deployment promotion
occurred. A failed result requires a new registered hypothesis, not post-hoc
policy tuning. Missing-label risk and independent generalization remain open.
The scaffold remains2.5D, not true3D or a foundation model. Silver labels are
not human gold. Stage5C and SMC remain disabled.
'''
    stream=io.StringIO(newline='');writer=csv.writer(stream,lineterminator='\n')
    writer.writerow(['contrast','mean','ci_low','ci_high','primary','evidence_role'])
    for key,v in summary['contrasts'].items():
        writer.writerow([key,v['mean'],*(v['CI95'] or [None,None]),key in summary['primary_contrasts'],'exposed_development'])
    receipt=dict(result_source='fresh_run_report_and_summary_reverification',
        summary_sha256=run.sha(HOME/'summary.json'),complete_sha256=run.sha(HOME/'complete.json'),
        training_freeze_sha256=complete['training_freeze_sha256'],groups=72,localities=12,
        advance=summary['advance_to_transfer_design'],failure_reasons=summary['failure_reasons'],
        independent_confirmation=False,deployment_changed=False,submission_ready=False,
        stage5c_executed=False,smc_enabled=False)
    return {'conclusions.md':text,'contrasts.csv':stream.getvalue(),
        'decision_diagnostics.json':json.dumps(d,indent=2,allow_nan=False)+'\n',
        'report_verification.json':json.dumps(receipt,indent=2,allow_nan=False)+'\n'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check',action='store_true');a=p.parse_args()
    for name,raw in artifacts(*validate()).items():
        path=HOME/name
        if a.check or path.exists():
            if path.read_bytes()!=raw.encode():raise ValueError('Preserve conflicting report: '+name)
        else:
            with path.open('xb') as f:f.write(raw.encode())
    print(json.dumps(dict(status='checked' if a.check else 'reported',scientific_readout='cached_verified')))


if __name__=='__main__':main()
