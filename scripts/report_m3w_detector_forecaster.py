"""Verified SOURCE detector-token trajectory experiment artifacts."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts import run_m3w_detector_forecaster as run
from src.evaluation import m3w_metric_bundle as bundle


def number(value):return 'not_estimable' if value is None else f'{value:+.6f}'


def interval(value):
    return 'not_estimable' if value['CI95'] is None else number(value['point'])+' ['+', '.join(number(v) for v in value['CI95'])+']'


def support_profile(complete):
    rows=[];seen=set()
    for ref in complete['groups']:
        path=ROOT/ref['path']
        if run.sha(path)!=ref['sha256']:raise ValueError('Verified readout group changed')
        row=json.loads(path.read_text());key=row['trial'],row['representation']
        if key in seen:raise ValueError('Repeated model readout group')
        seen.add(key)
        if row['representation']=='history':rows.append(row)
    if len(seen)!=72 or len(rows)!=18:raise ValueError('Full fixed72arm/18producer readout required')
    profile={}
    for kind in ('single','complement'):
        selected=[r['result'] for r in rows if r['kind']==kind]
        if len(selected)!=9:raise ValueError('All three folds and three seeds required')
        records={record for r in selected for site in r['sites'].values() for record in site['records']}
        profile[kind]=dict(producer_views=len(selected),unique_recordings=len(records),
            indexed_row_views=sum(r['rows'] for r in selected),
            complete_future_row_views=sum(r['complete_future_rows'] for r in selected),
            partial_future_row_views=sum(r['partial_future_rows'] for r in selected),
            missing_future_row_views=sum(r['missing_future_rows'] for r in selected),
            stationary_observed_row_views=sum(r['stationary_observed_rows'] for r in selected),
            stationary_known_nonzero_error_row_views={arm:sum(r['stationary_known_nonzero_error_rows'][arm] for r in selected)
                for arm in run.api.REPRESENTATIONS},
            repeated_producer_seed_row_views_not_independent_samples=True)
    return profile


def evidence():
    check=json.loads((run.PUBLIC/'verification.json').read_text());summary=json.loads((run.PUBLIC/'summary.json').read_text())
    if (not check['exact_fresh_inference_replay'] or check['groups']!=72
            or check['summary_sha256']!=run.sha(run.PUBLIC/'summary.json')
            or check['complete_sha256']!=run.sha(run.PUBLIC/'complete.json')
            or not check['all18disabled_legacy_predictions_exact']):raise ValueError('Complete fresh exact readout replay required')
    models=run.models();traces=[]
    for (trial,arm),row in sorted(models.items()):
        loss=row['fit']['losses']
        if row['fit']['step']!=4000 or loss[-1]['step']!=4000:raise ValueError('All complete fixed4000step traces required')
        traces.append(dict(trial=trial,kind=row['original_producer']['kind'],seed=row['original_producer']['seed'],
            representation=arm,losses=loss,checkpoint_sha256=row['checkpoint']['sha256'],minibatch_not_validation=True))
    run.once(run.PUBLIC/'training_traces.json',dict(rows=traces,models=72,primary_optimizer_updates=288000,
        result_source='cached_verified_completed_training_traces',
        legacy_mean_past_normalized_ADE_trace_field_is_restored_pixel_ADE_in_this_SOURCE_adapter=True))
    frozen=json.loads((run.PUBLIC/'training_freeze.json').read_text());complete=json.loads((run.PUBLIC/'complete.json').read_text())
    records={}
    for arm in run.api.REPRESENTATIONS:
        values=[r for r in traces if r['representation']==arm]
        records[arm]=dict(models=len(values),first_median_minibatch_loss=float(np.median([r['losses'][0]['loss'] for r in values])),
            final_median_minibatch_loss=float(np.median([r['losses'][-1]['loss'] for r in values])))
    value=dict(result_source='fresh_verified_producer_experiment_aggregate_receipt',
        training={k:frozen[k] for k in ('pid','seconds','peak_RSS_bytes','count','primary_optimizer_updates','extra_full_refit_updates','pilot_updates','disabled_exact_endpoints')},
        readout={k:complete[k] for k in ('pid','seconds','peak_RSS_bytes')},
        replay=json.loads((run.PRIVATE/'verification_runtime.json').read_text()),
        checkpoint_bytes=sum(r['checkpoint']['bytes'] for r in models.values()),loss_summary=records,
        support_profile=support_profile(complete),
        summary_sha256=run.sha(run.PUBLIC/'summary.json'),verification_sha256=run.sha(run.PUBLIC/'verification.json'),
        proposal_gates=summary['proposal_input_gates'],proposal_input_evidence_gate=summary['proposal_input_evidence_gate'],
        learned_policy_safety_evaluated=False,controller_held_scored=False,independent_roles_opened=False,
        deployment_changed=False,submission_ready=False,stage5c_executed=False,smc_enabled=False)
    run.once(run.PUBLIC/'evidence.json',value);return value,summary


def render(value,summary):
    c=value['training'];lines=['# Detector Tokens In Trajectory Dynamics: Full Matched Experiment','',
        '## Material Passport','',
        'Fresh_run:72real Transformer trajectory producers,288,000primary updates,',
        '4,000additional full-refit updates,900pilot updates,72fixed readout groups',
        'and a separate complete inference replay. Cached_verified: original18',
        'producer designs/checkpoints,318,969SOURCE rows and163record past-input',
        'store. Fresh legacy inference uses exactly the same allowed IDs/batches.',
        'Not_run: learned routing on the new proposal bank,controller-held outcomes,',
        'independent selection/calibration/confirmation,DroneCrowd or deployment.','',
        f"Training PID{c['pid']}: {c['seconds']:.6f}s, peak RSS{c['peak_RSS_bytes']:,}B.",
        f"Readout PID{value['readout']['pid']}: {value['readout']['seconds']:.6f}s; exact replay PID{value['replay']['pid']}: {value['replay']['seconds']:.6f}s.",
        'All18full disabled states and predictions reproduce the original model.',
        'First full history refit is exact. Native arm64CPU4/workers0,no resource',
        'probing,NumPy training fallback or reduction in original4,000step budgets.','',
        '## Matched Trajectory Accuracy','',
        'Relative to each producer own-fit-selected strongest causal baseline.',
        'Primary is single-fold history,the original controller-consumer design.',
        'Complement designs are retained secondary controls,not substituted winners.',
        'Intervals use3,000locality resamples after fold/seed averaging over all12',
        'SOURCE localities. These are exposed development outcomes,not independent',
        'confirmation. ADE retains the original available-label mask;partial rows',
        'do not reveal full-grid ground truth. FDE needs the last requested label.','',
        '| Producer design | Detector input | ADE gain %, CI95 | FDE gain %, CI95 | Hard ADE gain %, CI95 | Easy ADE gain %, CI95 |',
        '|---|---|---|---|---|---|']
    for kind,d in summary['kinds'].items():
        for arm,m in d['representations'].items():
            lines.append('| '+kind+' | '+arm+' | '+' | '.join(interval(v) for v in
                (m['all']['ADE'],m['all']['FDE'],m['hard']['ADE'],m['easy']['ADE']))+' |')
    lines+=['','## Label Support And Repeated Views','',
        'These counts repeat original producer-fold/seed views,not independent',
        'samples. The recording union is reported separately.','',
        '| Design | Unique recordings | Indexed row views | Complete future | Partial future | Missing future | Stationary observed |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for kind,r in value['support_profile'].items():
        lines.append('| '+kind+' | '+' | '.join(str(r[k]) for k in ('unique_recordings','indexed_row_views',
            'complete_future_row_views','partial_future_row_views','missing_future_row_views','stationary_observed_row_views'))+' |')
    lines+=['','The inherited zero ego-motion budget prevents stationary-to-moving',
        'corrections. Known stationary misses remain counted for every input arm;',
        'see support_profile in evidence.json. No future label determines inputs.','',
        '## All Causal Physical Baselines','',
        'These are descriptive paired comparisons,not selection on readout labels.','',
        '| Producer design | Physical baseline | ADE gain %, CI95 | FDE gain %, CI95 |',
        '|---|---|---|---|']
    for kind,d in summary['kinds'].items():
        for k,metrics in d['physical_baselines'].items():
            lines.append('| '+kind+' | '+run.source.BASELINES[int(k)]+' | '+interval(metrics['ADE'])+' | '+interval(metrics['FDE'])+' |')
    lines+=['','## Primary History Minus Controls','',
        '| Control | Known ADE change pp, CI95 | Completion-lower pp, CI95 | Completion-upper pp, CI95 |',
        '|---|---|---|---|']
    for arm,m in summary['kinds']['single']['history_contrasts']['all'].items():
        lines.append('| '+arm+' | '+' | '.join(interval(m[k]) for k in ('known_pp','lower_pp','upper_pp'))+' |')
    lines+=['','Completion bounds keep causal forecast-disagreement mass for wholly',
        'unknown future outcomes;the denominator remains fixed known reference',
        'mass,not an identified full-population percentage. Hard/easy membership',
        'uses observed evaluation errors and is a diagnostic subset,not missing-',
        'membership identification. Required undefined views are not dropped.','',
        '## Actual Training Loss','',
        '| Input | Models | First median minibatch objective | Final median minibatch objective |',
        '|---|---:|---:|---:|']
    for arm,r in value['loss_summary'].items():
        lines.append(f"| {arm} | {r['models']} | {r['first_median_minibatch_loss']:.6g} | {r['final_median_minibatch_loss']:.6g} |")
    lines+=['','Losses are real training minibatches,not validation/independent losses.',
        'The inherited trace key mean_past_normalized_ADE is a legacy name:here',
        'its raw error is restored pixel ADE before own-fit loss factors,not metres.',
        'Neither lower loss nor oracle physical-plus-neural opportunity constitutes',
        'a safe intervention policy. The approved2%complete policy requirement is',
        'unchanged and not evaluated by this raw-proposal experiment.','',
        '## Proposal Gates','',*['- '+k+': '+json.dumps(v) for k,v in value['proposal_gates'].items()],
        '- proposal_input_evidence_gate: '+json.dumps(value['proposal_input_evidence_gate']),
        '- learned_policy_safety_evaluated: false','- deployment_changed: false','- submission_ready: false','',
        '## Limits','',
        'The branch bundles detector shape/confidence,body-relative motion and IoU;',
        'it does not isolate a confidence-only mechanism. No new scene images,JEPA',
        'or generative model is trained. SOURCE is supplied detector-silver oblique',
        'midpoint pixels,obs8/pred12/stride12 raw frames,not human gold or verified',
        'live sensor-as-of. No metric/seconds,true3D,foundation,Stage5C or SMC claim.',
        'Independent and controller-held outcomes remain closed. Config/checkpoints',
        'and exact local replay do not mean a public from-scratch anonymous asset',
        'package is complete. See model_data_card.md and reproduction_zh.md.','',
        '[Loss and accuracy figure](figures/summary.svg), [actual traces](training_traces.json),',
        '[machine-readable evidence](evidence.json), [failure analysis](failure_analysis.md).','']
    return '\n'.join(lines)


def figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    check=json.loads((run.PUBLIC/'verification.json').read_text())
    if not check['exact_fresh_inference_replay'] or check['summary_sha256']!=run.sha(run.PUBLIC/'summary.json'):raise ValueError('Exact verified result required')
    s=json.loads((run.PUBLIC/'summary.json').read_text());traces=json.loads((run.PUBLIC/'training_traces.json').read_text())['rows']
    colors=['#666666','#bb4b36','#8a6996','#2864a0'];plt.rcParams.update({'font.size':10,'svg.hashsalt':run.NAME,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(14,4.8))
    for i,arm in enumerate(run.api.REPRESENTATIONS):
        for j,kind in enumerate(('single','complement')):
            p=s['kinds'][kind]['representations'][arm]['all']['ADE']
            if p['CI95'] is None:axes[j].text(i,0,'not estimable',rotation=90,ha='center',fontsize=8)
            else:axes[j].errorbar(i,p['point'],yerr=np.array([[p['point']-p['CI95'][0]],[p['CI95'][1]-p['point']]]),fmt='o',color=colors[i],capsize=4)
        rows=[r for r in traces if r['representation']==arm];steps=[x['step'] for x in rows[0]['losses']]
        if len(rows)!=18 or any([x['step'] for x in r['losses']]!=steps for r in rows):raise ValueError('All18complete traces per input required')
        values=np.array([[x['loss'] for x in r['losses']] for r in rows]);low,mid,high=np.quantile(values,[.1,.5,.9],0)
        axes[2].plot(steps,mid,label=arm,color=colors[i]);axes[2].fill_between(steps,low,high,color=colors[i],alpha=.10)
    for j,kind in enumerate(('Single-fold primary','Complement secondary')):
        axes[j].axhline(0,color='#888888',lw=.7);axes[j].set_title(kind)
        axes[j].set_xticks(range(4),run.api.REPRESENTATIONS,rotation=20);axes[j].set_ylabel('Known masked ADE gain vs fitted baseline (%)')
    axes[2].set_yscale('log');axes[2].set_title('Actual training objective');axes[2].set_xlabel('Optimizer update')
    axes[2].set_ylabel('Minibatch loss, median / p10-p90');axes[2].legend(frameon=False,fontsize=8)
    for ax in axes:ax.grid(axis='y',color='#dddddd',lw=.6);ax.set_axisbelow(True)
    fig.suptitle('Detector tokens in trajectory dynamics: SOURCE development only',y=.98)
    fig.text(.5,.035,'3,000 locality bootstrap draws;loss spread is not a CI. No independent readout or learned-policy safety certification.',ha='center',fontsize=8)
    fig.subplots_adjust(left=.07,right=.98,top=.83,bottom=.22,wspace=.35)
    folder=run.PUBLIC/'figures';folder.mkdir(exist_ok=True);path=folder/'summary.svg';fig.savefig(path,metadata={'Date':None})
    path.write_text('\n'.join(v.rstrip() for v in path.read_text().splitlines())+'\n')
    fig.savefig(run.PRIVATE/'summary.png',dpi=150);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=('report','figures','bundle'));args=parser.parse_args()
    if args.phase=='figures':figures()
    elif args.phase=='bundle':print(json.dumps(bundle.pack(ROOT,run.PUBLIC)))
    else:
        value,summary=evidence();text=render(value,summary);path=run.PUBLIC/'readout.md'
        if path.exists() and path.read_text()!=text:raise ValueError('Frozen report differs')
        path.write_text(text);print(json.dumps(value['proposal_gates']))
