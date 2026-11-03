"""Verified source-control readout; a supported secondary never passes full primary."""
import csv
import json
import os
from pathlib import Path
import sys

for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key,'4')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from src.evaluation.m3w_recording_lineage import sha256

PUBLIC=ROOT/'outputs/publication_readiness_2026_09/moving_source_forecast_supported_v1'
PRIVATE=ROOT/'data/stage_cvpr2027_experiments/moving_source_forecast_supported_v1'


def read(path):return json.loads(path.read_text())


def number(value):return 'not estimable' if value is None else f'{value:.4f}'


def macro(groups,family,key):
    sites=sorted({r['site'] for r in groups});values=[]
    for site in sites:
        local=[r['scores'][family][key] for r in groups if r['site']==site]
        if len(local)!=6 or any(v is None or not np.isfinite(v) for v in local):return None
        values.append(float(np.mean(local)))
    return float(np.mean(values))


def interval_text(value):return f'{value["point"]:+.4f} [{value["CI95"][0]:+.4f}, {value["CI95"][1]:+.4f}]'


def baseline_readout(population):
    from src.data_unification.m3w_moving_forecast_data import prepare
    from src.world_model.m3w_european_source_forecast import BASELINES
    cfg=read(ROOT/'configs/m3w_moving_source_forecast_supported_v1.json')
    data,receipt,_=prepare(ROOT,cfg['admission'])
    if receipt!=population:raise ValueError('Current baseline population differs from frozen complete pack')
    rows=[]
    for site in sorted(set(data['sites'])):
        use=data['sites']==site;known=use&data['valid'].any(1);end=use&data['valid'][:,-1]
        reference=float(data['baseline_ade'][known,1].mean())
        for k,name in enumerate(BASELINES):
            a=float(data['baseline_ade'][known,k].mean())
            rows.append(dict(group=str(site),model=name,available_ADE=a,
                endpoint_FDE=float(data['baseline_fde'][end,k].mean()) if end.any() else None,
                gain_vs_CV_percent=100*(reference-a)/reference,known_rows=int(known.sum()),endpoint_rows=int(end.sum())))
    out=[]
    for name in BASELINES:
        local=[r for r in rows if r['model']==name]
        f=[r['endpoint_FDE'] for r in local]
        out.append(dict(model=name,available_ADE=float(np.mean([r['available_ADE'] for r in local])),
            endpoint_FDE=float(np.mean(f)) if all(v is not None for v in f) else None,
            gain_vs_CV_percent=float(np.mean([r['gain_vs_CV_percent'] for r in local]))))
    return dict(result_source='fresh_baseline_scalar_recalculation_same_hash_verified_RAM_population',
                groups=rows,models=out,test_best_is_diagnostic_only=True)


def figures(analysis,trials):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    destination=PUBLIC/'figures';destination.mkdir(exist_ok=True)
    previews=PRIVATE/'figure_previews';previews.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':9,'svg.fonttype':'none'})
    names=['Supported-group paired lower','Supported-group known']
    vals=[analysis['supported_group_completion_lower_secondary'],analysis['supported_group_known_secondary']]
    fig,ax=plt.subplots(figsize=(7,2.5))
    for i,v in enumerate(vals):
        ax.errorbar(v['point'],i,xerr=[[v['point']-v['CI95'][0]],[v['CI95'][1]-v['point']]],
                    fmt='o',capsize=4,color=('#1766a0','#b15c25')[i])
    ax.axvline(0,color='0.45',lw=1);ax.set_yticks(range(2),names);ax.invert_yaxis()
    ax.set_xlabel('EqMotion minus PCT, percentage points [95% grouped bootstrap CI]')
    ax.set_title('23 prediction-supported groups; original 24-group primary not estimable')
    fig.tight_layout();fig.savefig(destination/'paired_secondary.svg');fig.savefig(previews/'paired_secondary.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,3.5))
    for family,color in (('pct','#1766a0'),('eqmotion','#b15c25')):
        runs=[r['fit']['losses'] for r in trials if r['identity']['family']==family]
        steps=sorted(set.intersection(*[set(v['step'] for v in run) for run in runs]))
        matrix=np.array([[next(v['loss'] for v in run if v['step']==s) for s in steps] for run in runs])
        ax.plot(steps,np.median(matrix,axis=0),color=color,label=family+' median of 9 fits')
        ax.fill_between(steps,matrix.min(0),matrix.max(0),color=color,alpha=.14)
    ax.set_yscale('log');ax.set_xlabel('Optimizer updates');ax.set_ylabel('Fitting-baseline-relative objective (log scale)')
    ax.set_title('Training loss only; band is run range, not held-data confidence')
    ax.legend(frameon=False);fig.tight_layout();fig.savefig(destination/'training_loss.svg');fig.savefig(previews/'training_loss.png',dpi=170);plt.close(fig)


def main():
    analysis=read(PUBLIC/'analysis.json');training=read(PUBLIC/'training.json');population=read(PUBLIC/'population.json')
    fit_exec=read(PUBLIC/'train_execution.json');evaluate_exec=read(PUBLIC/'evaluate_execution.json');verify=read(PUBLIC/'verify_execution.json')
    refits=read(PUBLIC/'full_refit_verification.json');pilot=read(PUBLIC/'pilot.json')
    if (training['complete_primary_models']!=18 or not training['matched_sampling_all9pairs_exact']
            or not verify['real_Torch_training'] or refits['full_primary_refits']!=2
            or analysis['primary']['status']!='not_estimable' or analysis['all_declared_group_primary_passed']):
        raise ValueError('Require complete real fits, exact replays and honest primary qualification')
    trials=[]
    for item in training['checkpoints']:
        path=ROOT/item['path']
        if sha256(path)!=item['sha256']:raise ValueError('Changed complete trained checkpoint')
        trials.append(read(path.parent/'complete.json'))
    for item in analysis['prediction_receipts']:
        if sha256(ROOT/item['path'])!=item['sha256']:raise ValueError('Changed fully replayed prediction')
    baseline=baseline_readout(population)
    (PUBLIC/'baseline_metrics.json').write_text(json.dumps(baseline,indent=2,sort_keys=True,allow_nan=False)+'\n')
    rows=analysis['groups'];lower=analysis['supported_group_completion_lower_secondary'];known=analysis['supported_group_known_secondary']
    summaries={name:dict(available_ADE=macro(rows,name,'available_ADE'),endpoint_FDE=macro(rows,name,'endpoint_FDE'),
                         gain_vs_CV_percent=macro(rows,name,'gain_vs_CV_percent')) for name in ('PCT','eqmotion')}
    model_rows=[*baseline['models'],*[dict(model=k,**v) for k,v in summaries.items()]]
    figure_rows=trials;figures(analysis,figure_rows)
    result=dict(result_source='verified_scalar_readout_with_fresh_same_population_baseline_recalculation',
        analysis_sha256=sha256(PUBLIC/'analysis.json'),training_sha256=sha256(PUBLIC/'training.json'),
        verification_sha256=sha256(PUBLIC/'verify_execution.json'),models=model_rows,
        original24group_primary='not_estimable_not_passed',secondary23group_lower=lower,secondary23group_known=known,
        three_seed_breakdown=analysis['by_seed'],new_policy_evaluated=False,new_deployment=False,
        ccf_a_candidate=False,stage5c_executed=False,smc_enabled=False)
    (PUBLIC/'readout_sources.json').write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    with (PUBLIC/'metrics.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=['model','available_ADE','endpoint_FDE','gain_vs_CV_percent']);writer.writeheader();writer.writerows(model_rows)
    lines=['# Admitted Moving-Source Forecast Controls','','## Material Passport','',
        'Fresh_run: 18 real Torch CPU producer fits, three actual training seeds,',
        'complete excluded-source inference and the two first-job full refits.',
        'Cached_verified: every saved checkpoint/prediction and a separate full',
        'prediction replay. This report recomputes baseline scalars on the exact',
        'same hash-verified in-memory population; writing prose is not training.','',
        '## Scope And Primary','',
        '**The original 24-group primary is not estimable and has not passed.**',
        'All 64 admitted recordings are processed, but 58 recordings/23 groups',
        'have fixed clip-contained 8/12 queries. Three short scene2 clips give',
        'one wholly empty group. Other zero-query clips remain in the ledger.',
        'The support repair removes no data row and fabricates no zero errors.','',
        f'Rows: {population["rows"]:,}; known {population["known_rows"]:,}; complete {population["complete_rows"]:,};',
        f'wholly unknown {population["wholly_unknown_future_rows"]:,}; endpoint labels {population["endpoint_rows"]:,}.',
        'The raw input inventory includes current-visible partial-history agents.',
        'Both comparator interfaces consume the same complete-past nearest8neighbors.',
        'The separate48shared observed-motion features are not used by either model',
        'and have no claimed downstream contribution in this experiment.','',
        '## Supported-Group Secondary Results','',
        f'Paired completion-lower EqMotion-minus-PCT difference: **{interval_text(lower)} pp**.',
        f'Known-label paired difference: **{interval_text(known)} pp**.',
        'Intervals use3,000source-group resamples after the two excluded producers',
        'and three actual seed averages. They are not window-level IID intervals.',
        'The legacy interval field says locality; these are conservative source',
        'components, not certified independent geographic sites. Available-label',
        'ADE on partial rows is not full-future ground truth. Unknown rows use',
        'the unchanged paired-disagreement bound, not imputed positions.','',
        '| Model | Equal supported-group pixel ADE | Full-supported-group endpoint FDE | Gain vs CV (%) |',
        '|---|---:|---:|---:|']
    for row in model_rows:lines.append(f'| {row["model"]} | {number(row["available_ADE"])} | {number(row["endpoint_FDE"])} | {number(row["gain_vs_CV_percent"])} |')
    lines += ['', 'Baselines are reported separately. A held-data best baseline is diagnostic,',
        'never an inference choice; each producer chooses its actual rollout floor',
        'using fitting groups only. PCT and EqMotion are architecture-family controls,',
        'not capacity-matched or an isolated equivariance effect. EqMotion emits one',
        'fixed head, not published best-of20metrics.','',
        '## Actual Training And Reproduction','',
        f'Complete18-model train invocation PID{fit_exec["pid"]}: {fit_exec["seconds"]:.3f}s, peak RSS {fit_exec["peak_RSS_bytes"]:,}bytes.',
        f'Original excluded-source inference PID{evaluate_exec["pid"]}: {evaluate_exec["seconds"]:.3f}s.',
        f'Full verification/refit invocation PID{verify["pid"]}: {verify["seconds"]:.3f}s.',
        'Primary budget is72,000updates;200of them reuse exact first-job pilot prefixes.',
        'Direct-pilot checks and8,000full-refit replay updates are additional verification',
        'work, not extra primary fitting budget. All nine paired families have identical',
        'train IDs,loss factors,draw coverage and sampler RNG. Held rows are never drawn.',
        'All original predictions and both complete first-job checkpoints replay exactly.',
        'CPU4/interop1/workers0; no NumPy model fallback, MPS claim or resource probing.','',
        '## What This Does Not Establish','',
        'No controller, matched intervention policy, selected-risk calibration, full2%',
        'operational/easy contract, physical consistency or independent confirmation',
        'has passed here. No JEPA/scene/goal/shared-motion lift is demonstrated.',
        'DUT backgrounds and audit future images are not predictive inputs.',
        'Native annotation steps/pixels are not seconds/metres or equal durations',
        'to SDD/European views. This is not true3D,foundation or a new dynamics',
        'architecture contribution. Deployment stays unchanged; Stage5C/SMCoff.',
        'The full research goal and submission-readiness standard remain unmet.']
    (PUBLIC/'readout.md').write_text('\n'.join(lines)+'\n')
    card=['# Model And Data Card','',
        'Task: image-plane native8/12forecasting on fit-admitted MovingDroneCrowd++.',
        'Population:64processed recordings,58with queries,24declared/23supported groups.',
        'All195,286past-eligible rows remain,including partial and unknown labels.',
        'Architectures: existing PCT and pinned-author fixed-head EqMotion controls.',
        'Initialization: random seeds17/29/43,one fitting fold per producer,other two excluded.',
        'Source normalization/baseline/loss factors follow fitting groups only.',
        'Inputs: past positions,valid past masks,selected complete-past neighbors and',
        'causal rollout/time requests. Labels and target-validity masks never enter inputs.',
        'Raw current-visible input construction is preserved; incomplete-past agents',
        'are not neural neighbor tokens under the matched comparator rule.',
        'Source snapshots/role and dependency hashes are in registration.json/job_roster.json.',
        'The 48 shared-motion fields are unused prospective critic features, not image modality.',
        'Training uses native arm64 TorchCPU4,atomic checkpoints and resume,no workers.',
        'All weights/predictions remain private; Git contains code/scalars/source hashes.',
        'Source use: attributed local non-commercial CC-BY-NC4.0 release; no raw redistribution.',
        'The fixed-head adapter is not published best-of20reproduction; no new EqMotion novelty.',
        'Original24-groupprimary not estimable;23-group results are secondary development evidence.',
        'No independent geographic-site certification or transport of risk guarantees.',
        'No deployable M3W upgrade,full2%gate,metric/seconds,true3D or foundation claim.',
        'Stage5Cexecution andSMCremain off.']
    (PUBLIC/'model_data_card.md').write_text('\n'.join(card)+'\n')
    failure=['# Failure Analysis And Next Experiment','',
        'The original controller preparation failed on one absent row-group name,',
        'not a Torch hang. Its failure,full population and primary specification remain.',
        'The forward support adapter retains every row and makes the full24-group',
        'primary not estimable; it does not promote a positive supported secondary.',
        '',f'Supported lower difference: {interval_text(lower)} pp.',
        f'Supported known difference: {interval_text(known)} pp.',
        'An interval crossing zero is inconclusive; a negative interval is a failed',
        'forecast-family improvement. A positive interval is a proposal-control result,',
        'not selected-risk safety or a new architecture contribution.','',
        'Inspect per-group/per-seed scores before attributing failure to architecture.',
        'Moving-camera head centers,variable track support and native-time mismatch',
        'are documented potential difficulties,not experimentally isolated causes.',
        'Do not call the dataset metric,video seconds verified or sensor-time causal.',
        '', 'Next use the frozen source-excluded proposal bank for the actual matched',
        'cost-learning experiment: existing past/local context versus the all-current',
        '48-column observed-motion context,capacity/budget matched,three seeds,',
        'disjoint predictor/critic/held groups and same-predictor intervention controls.',
        'The shared context is not physical camera odometry. Test its incremental',
        'information rather than repeating failed width/duration or test thresholds.',
        'Keep the full original2%risk and empty-support not-estimable rules; do not',
        'open DUT/DroneCrowd outcomes before actual policy/producer admission.']
    (PUBLIC/'failure_analysis.md').write_text('\n'.join(failure)+'\n')
    print(json.dumps(dict(secondary_lower=lower['point'],secondary_known=known['point'],full24primary='not_estimable',
                          report=str(PUBLIC/'readout.md'),deployment_changed=False)))


if __name__=='__main__':main()
