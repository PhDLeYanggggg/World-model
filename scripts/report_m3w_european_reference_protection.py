"""All-setting reports for matched shared versus reference-protected continuation."""
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts import run_m3w_european_reference_protection as run
from scripts.report_m3w_european_easy_harm_sampling import span
from scripts.report_m3w_european_selected_risk_learning import fmt


def summarize(rows,seeds):
    if len(rows)!=18 or len(seeds)!=6: raise ValueError('All source roles and seeds required')
    views={}
    for key in run.POLICIES:
        v=[r['views'][key] for r in rows]
        views[key]=dict(all_vs_raw=span([s['ADE_vs_raw']['all']['equal_scene_gain_percent'] for s in v]),
            hard_vs_raw=span([s['ADE_vs_raw']['hard']['equal_scene_gain_percent'] for s in v]),
            all_vs_reference=span([s['ADE_vs_reference']['all']['equal_scene_gain_percent'] for s in v]),
            easy_worst=span([max(0.,-s['easy_vs_CV']['worst_scene_gain_percent']) for s in v]),
            risk_passes=sum(s['risk']['feasible'] for s in v),
            easy_passes=sum(all(a['easy_degradation_percent'] is not None and a['easy_degradation_percent']<=2
                and a['zero_CV_harm']==0 for a in s['risk']['by_locality'].values()) for s in v),
            all_harm_violations=sum(not a['events']['all']['passes'] for s in v for a in s['risk']['by_locality'].values()),
            easy_harm_violations=sum(not a['events']['easy']['passes'] for s in v for a in s['risk']['by_locality'].values()),
            switch_rate=span([s['switch_rate'] for s in v]))
    contrasts={}
    for key in next(iter(seeds.values())):
        values=[r[key] for r in seeds.values()]
        if any(v.get('status')=='not_estimable' for v in values):
            contrasts[key]=dict(status='not_estimable'); continue
        contrasts[key]=dict(point_range=span([v['gain_percent'] for v in values]),
            positive_CI=sum(v['CI'][0]>0 for v in values),negative_CI=sum(v['CI'][1]<0 for v in values),
            overlap_CI=sum(v['CI'][0]<=0<=v['CI'][1] for v in values))
    return dict(views=views,contrasts=contrasts)


def matched_fit_summary(rows):
    result={}
    for pair in ('full','motion_only'):
        group=[r for r in rows if r['pair']==pair]
        if len(group)!=18: raise ValueError('Eighteen matched pairs required')
        ratios=[]; component=[]
        for row in group:
            a,b=row['arms']['protected']['fit']['trace'][-1],row['arms']['continued']['fit']['trace'][-1]
            ratios.append(a['moment_mse']/b['moment_mse'])
            component.append(np.divide(a['component_mse'],b['component_mse']).tolist())
        result[pair]=dict(fixed_batch_MSE_protected_over_continued_median=float(np.median(ratios)),
            protected_lower_count=sum(v<1 for v in ratios),
            component_ratio_medians=np.median(component,axis=0).tolist(),not_validation=True)
    return result


def main():
    cfg,identity=run.registration(); done=run.checked_training(identity)
    checks=json.loads((run.PUBLIC/'completion_checks.json').read_text()); assert checks['all_passed']
    rows={p:[] for p in cfg['pairs']}
    for ref in checks['groups']:
        assert run.artifact(ROOT/ref['path'])==ref
        r=json.loads((ROOT/ref['path']).read_text()); rows[r['pair']].append(r)
    assert run.artifact(ROOT/checks['seeds']['path'])==checks['seeds']
    seeds=json.loads((run.PUBLIC/'seed_averaged_metrics.json').read_text())
    aggregate={p:summarize(v,seeds[p]) for p,v in rows.items()}
    run.immutable_json(run.PUBLIC/'aggregate_metrics.json',aggregate)
    fits=[]
    for ref in done['decisions']:
        d=json.loads((ROOT/ref['path']).read_text()); arms={}
        for h in d['heads']:
            head=json.loads((ROOT/h['path']).read_text()); arms[head['identity']['arm']]=dict(fit=head['fit'],
                checkpoint=head['artifacts']['checkpoint'])
        fits.append(dict(group=d['group']['group'],pair=d['pair'],arms=arms))
    run.immutable_json(run.PUBLIC/'training_metrics.json',fits)
    run.immutable_json(run.PUBLIC/'fitting_diagnostics.json',matched_fit_summary(fits))
    v=aggregate['full']['views']; c=aggregate['full']['contrasts']
    gates=dict(real_torch_training_complete=all(s['fit']['complete'] for r in fits for s in r['arms'].values()),
        reference_moments_preserved=True,matched_draws_budget_initialization=True,B_only_fitting=True,
        decisions_frozen_before_readout=True,protected_joint_complete_risk=v['protected_joint']['risk_passes']==18,
        protected_joint_net_easy=v['protected_joint']['easy_passes']==18,
        protected_over_continued_all_six_positive=c['protected_joint_vs_continued_joint__all']['positive_CI']==6,
        protected_over_mean_all_six_positive=c['protected_joint_vs_mean_joint__all']['positive_CI']==6,
        protected_over_raw_all_six_positive=c['protected_joint_vs_raw_neural__all']['positive_CI']==6,
        protected_joint_over_individual_all_six_positive=c['protected_joint_vs_protected_dual__all']['positive_CI']==6,
        independent_calibration=False,independent_confirmation=False,risk_certificate=False,
        new_forecaster_training=False,deployment_changed=False,submission_ready=False,stage5c_executed=False,smc_enabled=False)
    run.immutable_json(run.PUBLIC/'gates.json',gates)
    queue=json.loads((run.PRIVATE/'create_queue.json').read_text())
    events=[json.loads(s) for s in (run.PRIVATE/'events.jsonl').read_text().splitlines()]
    run.immutable_json(run.PUBLIC/'compute_receipt.json',dict(heads=72,updates=sum(s['fit']['step'] for r in fits for s in r['arms'].values()),
        summed_fit_seconds=sum(s['fit']['seconds'] for r in fits for s in r['arms'].values()),
        arm_fit_seconds={a:sum(r['arms'][a]['fit']['seconds'] for r in fits) for a in cfg['arms']},
        parameters={a:sorted({r['arms'][a]['fit']['parameters'] for r in fits}) for a in cfg['arms']},
        trainable_parameters={a:sorted({r['arms'][a]['fit']['trainable_parameters'] for r in fits}) for a in cfg['arms']},
        native_arm64=True,torch_cpu_threads=4,interop_threads=1,workers=0,
        training_pids=sorted({e['pid'] for e in events if e['state']=='training'}),
        unknown_training_draws=sum(s['fit']['unknown_training_draws'] for r in fits for s in r['arms'].values()),
        create_queue_returncode=queue['response']['returncode'],create_observed_utc=queue['completed_utc'],
        remote_M3W_inventory='not_run',jobs_submitted=0,remote_modified=False))
    lines=['# Reference Protection Results','','## Material Passport',
        'Fresh 72 Torch continuations / 144,000 additional updates and 360 new readout views; 216 old views cached_verified.',
        'All six source A/B assignments, three seeds, full/motion-only forecast pairs. C is historically opened development.',
        'No favorable seed, checkpoint, threshold or source assignment is selected.','','## Policies','',
        '| Pair / policy | All ADE vs raw (%) | Hard vs raw (%) | All vs R (%) | Worst net easy degradation (%) | Risk /18 | Net easy /18 | Switch fraction |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for pair,a in aggregate.items():
        for key,v in a['views'].items():
            lines.append(f"| {pair} / {key} | {fmt(v['all_vs_raw'])} | {fmt(v['hard_vs_raw'])} | {fmt(v['all_vs_reference'])} | {fmt(v['easy_worst'])} | {v['risk_passes']} | {v['easy_passes']} | {fmt(v['switch_rate'])} |")
    lines+=['','Complete risk requires all/easy positive harm <=2%, net easy <=2%, and no zero-CV harm in every supported C locality.',
        'Reference-only abstention is not learned improvement. A frozen denominator is not a calibrated denominator.','',
        '## Three-Seed Contrasts','',
        '| Pair / A-B roles | All-ADE contrast | Point (%) | 95% locality-bootstrap CI (%) |','|---|---|---:|---:|']
    for pair,groups in seeds.items():
        for group,cs in groups.items():
            for key,v in cs.items():
                if key.endswith('__all'): lines.append(f"| {pair} / {group} | {key} | {v.get('gain_percent',float('nan')):+.5f} | {fmt(v.get('CI'))} |")
    lines+=['','Three seeds are averaged within locality, then four C localities are resampled 3,000 times.',
        'Overlapping source roles and unadjusted multiple contrasts are not independent confirmatory evidence.',
        'The count-matched hash control does not match realized risk. Allocation is greedy, not optimal or collision-aware.',
        'Protected inference carries a second frozen network; equal update count is not equal total storage or inference cost.',
        'Obs8/pred12 annotation steps at raw stride12; image pixels, detector-derived labels.',
        'No metric/seconds, human-gold, physical-safety, true3D, foundation or submission-ready claim.',
        'No selection/calibration/confirmation access. No deployment. Stage5C and SMC remain off.']
    (run.PUBLIC/'results.md').write_text('\n'.join(lines)+'\n')
    (run.PUBLIC/'world_model_gate.md').write_text('# Reference Protection Gates\n\nEngineering completion is not scientific success.\n\n'+
        '\n'.join(f'- {k}: {str(v).lower()}' for k,v in gates.items())+'\n')
    print(json.dumps(dict(gates=gates,full_primary=c['protected_joint_vs_continued_joint__all']),indent=2))


if __name__=='__main__': main()
