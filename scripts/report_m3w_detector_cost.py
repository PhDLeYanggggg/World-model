"""Verified detector-history experiment receipts, actual loss traces and figures."""
import argparse
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts import run_m3w_detector_cost as run
from scripts.report_m3w_completion_cost import number,interval
from scripts.report_m3w_forecast_amplitude import risk_causes
from src.evaluation import m3w_metric_bundle as bundle


def traces():
    rows=[]
    for (context,seed,bank,arm),ref in sorted(run.models().items()):
        state=run.parent.api.base.read_checkpoint(ROOT/ref['checkpoint']['path'])
        if state['step']!=2000 or state['trace'][-1]['step']!=2000:raise ValueError('Every full trace required')
        rows.append(dict(context=context,seed=seed,bank=bank,representation=arm,
            checkpoint_sha256=ref['checkpoint']['sha256'],trace=state['trace'],
            minibatch_not_validation=True,unknown_rows_drawn=state['unknown_rows_drawn'],
            unknown_rows_with_loss=state['unknown_rows_with_loss']))
    result=dict(result_source='fresh_trace_extraction_from_cached_verified_frozen_states',
        heads=144,primary_optimizer_updates=288000,rows=rows,new_optimizer_updates=0,
        validation_or_independent_loss_claim=False)
    run.once(run.PUBLIC/'training_traces.json',result)
    return result


def evidence():
    summary=json.loads((run.PUBLIC/'summary.json').read_text())
    check=json.loads((run.PUBLIC/'verification.json').read_text())
    complete=json.loads((run.PUBLIC/'complete.json').read_text())
    if not check['exact_replay'] or check['groups']!=288 or check['summary_sha256']!=run.sha(run.PUBLIC/'summary.json'):
        raise ValueError('Separate complete replay required')
    rows=[]
    for ref in complete['groups']:
        path=ROOT/ref['path']
        if run.sha(path)!=ref['sha256']:raise ValueError('Original aggregate changed')
        rows.append(json.loads(path.read_text()))
    if len(rows)!=288:raise ValueError('All original groups required')
    frozen=json.loads((run.PUBLIC/'training_freeze.json').read_text())
    model_refs=run.models();loss=traces()
    diagnostics={}
    for arm in run.api.REPRESENTATIONS:
        group=[r for r in rows if r['representation']==arm]
        diagnostics[arm]=dict(risk_causes={bank:risk_causes([p[bank] for r in group for p in r['result']['recordings'].values()])
            for bank in ('physical_only','physical_neural')},
            mean_all_legal_normalized_cost_MSE={bank:np.mean([
                r['result']['calibration'][bank]['normalized_moment_MSE'] for r in group],0).tolist()
                for bank in ('physical_only','physical_neural')},
            diagnostic_MSE_not_risk_calibration=True,
            loss_first_last_median={bank:{which:float(np.median([r['trace'][index]['loss'] for r in loss['rows']
                if r['representation']==arm and r['bank']==bank])) for which,index in (('first',0),('last',-1))}
                for bank in ('physical_only','physical_neural')})
    native={}
    for source in sorted({r['source'] for r in rows}):
        native[source]={}
        for arm in run.api.REPRESENTATIONS:
            for bank in ('floor','physical_only','physical_neural'):
                vals=[]
                for row in rows:
                    if row['source']!=source or row['representation']!=arm:continue
                    r=row['result'];p=r['policies'][bank]['policy']
                    if r['known_rows']<=0:raise ValueError('Cannot omit undefined locality')
                    vals.append((r['full_known_reference_mass']-p['selected_known_benefit_mass']+p['selected_known_harm_mass'])/r['known_rows'])
                native[source][arm+'_'+bank+'_mean_ADE_pixels']=statistics.fmean(vals)
    value=dict(result_source='fresh_aggregate_diagnosis_of_verified_detector_experiment',
        summary_sha256=run.sha(run.PUBLIC/'summary.json'),training_freeze_sha256=run.sha(run.PUBLIC/'training_freeze.json'),
        compute={key:frozen[key] for key in ('pid','seconds','peak_RSS_bytes','count','optimizer_updates','extra_full_refit_updates','pilot_updates','disabled_exact_endpoints')},
        primary_checkpoint_bytes=sum(r['checkpoint']['bytes'] for r in model_refs.values()),
        readout={key:complete[key] for key in ('pid','seconds','peak_RSS_bytes')},
        replay=json.loads((run.PRIVATE/'verification_runtime.json').read_text()),
        diagnostics=diagnostics,native_ADE_descriptive_not_pooled_claim=native,
        gates=dict(summary['fitting_gates'],fitting_advancement_gate=summary['fitting_advancement_gate'],
            independent_confirmation=False,deployment_changed=False,submission_ready=False,stage5c_executed=False,smc_enabled=False))
    run.once(run.PUBLIC/'evidence.json',value);run.once(run.PUBLIC/'gates.json',value['gates'])
    return value,summary


def render(value,summary):
    c=value['compute'];lines=['# Past Detector Histories: Matched Cost Learning','',
        '## Material Passport','',
        'Fresh_run:144 real Torch cost heads,288,000 primary optimizer updates,',
        '2,000 extra full-refit updates,360 pilot updates,288 fitting groups and a',
        'separate complete inference replay. Cached_verified: frozen trajectory',
        'producer seed43, original fitting packets, normalizers and raw-replayed',
        'detector features. Not_run: new trajectory/world-dynamics training, held',
        'or independent evaluation, deployment. No NumPy training fallback.','',
        f"Training PID{c['pid']}, exit0: {c['seconds']:.6f}s, peak RSS{c['peak_RSS_bytes']:,}B.",
        f"Readout PID{value['readout']['pid']}: {value['readout']['seconds']:.6f}s; replay PID{value['replay']['pid']}: {value['replay']['seconds']:.6f}s.",
        'All36disabled endpoints exactly reproduce old known-only model/Adam/RNG/',
        'draws/traces. The first full history/mixed refit is exact. Four representations',
        'share the original398fields and64-field zero-initial branch;30,405parameters.',
        'Extra input statistics use own-context causal fitting rows, never future support.','',
        '## Complete Policy And Risk','',
        '| Representation | Bank | Known ADE gain %, fitting CI95 | Risk support | Undefined easy | Known all/easy violations | Selected / unknown |',
        '|---|---|---|---:|---:|---:|---:|']
    for arm,d in summary['representations'].items():
        for bank in ('physical_only','physical_neural'):
            r=d['safety'][bank]
            lines.append(f"| {arm} | {bank} | {interval(d['accuracy'][bank])} | {r['supported_record_views']}/{r['required_record_views']} | {r['undefined_easy_record_views']} | {r['known_all_risk_violations']}/{r['known_easy_risk_violations']} | {r['selected']} / {r['selected_unknown']} |")
    lines+=['','Known-label gains are not complete-outcome safety. All444recording views',
        'per representation repeat six contexts/head seeds over74unique recordings.',
        'Undefined selected/easy risk is not a passed zero. No recording is removed.','',
        '## Primary History Minus Information Controls','',
        '| Control | Full known change pp, CI95 | Full completion-lower pp, CI95 | Matched completion-lower pp, CI95 |',
        '|---|---|---|---|']
    primary=summary['representations']['history']
    for arm in ('disabled','current','noise'):
        d=primary['representation_contrasts'][arm]['physical_neural']
        lines.append('| '+arm+' | '+' | '.join(interval(d[k]) for k in ('full_known_pp','full_lower_pp','matched_lower_pp'))+' |')
    lines+=['','Missing-outcome disagreement mass remains in completion bounds, with fixed',
        'known floor reference mass. Count-matched controls compare utility only.',
        'All intervals use3,000locality bootstrap draws after averaging contexts and',
        'controller seeds across12localities. They are fitting-conditional, not',
        'independent confirmation or whole-pipeline OOF. Noise can encode identity;',
        'it controls capacity/keys, not useful visual evidence.','',
        '## Neural Candidate Bank Increment','',
        '| Representation | Mixed minus physical known pp, CI95 | Completion-lower pp, CI95 |',
        '|---|---|---|']
    for arm,d in summary['representations'].items():
        lines.append('| '+arm+' | '+interval(d['neural_bank_increment']['full_known_pp'])+' | '+interval(d['neural_bank_increment']['full_lower_pp'])+' |')
    lines+=['','## Actual Minibatch Loss','',
        '| Representation | Physical first / final median | Mixed first / final median |','|---|---|---|']
    for arm,d in value['diagnostics'].items():
        lines.append('| '+arm+' | '+' | '.join(f"{d['loss_first_last_median'][b]['first']:.6g} / {d['loss_first_last_median'][b]['last']:.6g}" for b in ('physical_only','physical_neural'))+' |')
    lines+=['','First/final minibatches are not fixed validation losses. Falling objective',
        'does not certify selected-risk calibration. All traces are retained.','',
        '## Gates','',*['- '+k+': '+json.dumps(v) for k,v in value['gates'].items()],'',
        'The primary remains history/mixed; no secondary replaces it after scoring.',
        'SOURCE is oblique detector-silver midpoint pixels,obs8/pred12/stride12 raw',
        'frames. Acquisition-as-of is not certified. No metric,seconds,human-gold,',
        'true3D,foundation,Stage5C or SMC claim. Independent roles and DroneCrowd stay closed.','',
        '[Actual loss and policy figures](figures/summary.svg), [machine-readable evidence](evidence.json),',
        '[complete training traces](training_traces.json), [reproduction](reproduction_zh.md).','']
    return '\n'.join(lines)


def figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    s=json.loads((run.PUBLIC/'summary.json').read_text());t=json.loads((run.PUBLIC/'training_traces.json').read_text())
    plt.rcParams.update({'font.size':10,'svg.hashsalt':run.NAME,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(14,4.8));colors=['#656565','#bb4b36','#8a6996','#2864a0']
    for i,arm in enumerate(run.api.REPRESENTATIONS):
        d=s['representations'][arm];p=d['accuracy']['physical_neural'];risk=d['safety']['physical_neural']
        axes[0].errorbar(i,p['point'],yerr=np.array([[p['point']-p['CI95'][0]],[p['CI95'][1]-p['point']]]),fmt='o',color=colors[i],capsize=4)
        axes[1].bar(i,100*risk['supported_record_views']/risk['required_record_views'],color=colors[i])
        rows=[r for r in t['rows'] if r['representation']==arm and r['bank']=='physical_neural']
        if len(rows)!=18:raise ValueError('All18mixed traces required per representation')
        steps=[v['step'] for v in rows[0]['trace']]
        if any([v['step'] for v in r['trace']]!=steps for r in rows):raise ValueError('Unmatched loss checkpoints')
        loss=np.array([[v['loss'] for v in r['trace']] for r in rows]);low,mid,hi=np.quantile(loss,[.1,.5,.9],axis=0)
        axes[2].plot(steps,mid,label=arm,color=colors[i]);axes[2].fill_between(steps,low,hi,color=colors[i],alpha=.10)
    axes[0].axhline(0,color='#888888',lw=.7);axes[0].set_ylabel('Mixed known ADE gain over floor (%)')
    axes[1].set_ylabel('Complete recording risk views supported (%)');axes[1].set_ylim(0,105)
    axes[1].axhline(100,color='#888888',ls=':',lw=.8)
    for ax in axes[:2]:ax.set_xticks(range(4),run.api.REPRESENTATIONS,rotation=20)
    axes[2].set_yscale('log');axes[2].set_xlabel('Optimizer update');axes[2].set_ylabel('Minibatch objective, median / p10-p90')
    axes[2].legend(frameon=False,fontsize=8)
    for ax in axes:ax.grid(axis='y',color='#dddddd',lw=.6);ax.set_axisbelow(True)
    fig.suptitle('Matched detector-history inputs: fixed fitting-only comparison',y=.98)
    fig.text(.5,.035,'Accuracy CI: 3,000 locality-bootstrap draws. Loss spread is descriptive, not a CI. No held or independent evaluation.',ha='center',fontsize=8)
    fig.subplots_adjust(left=.07,right=.985,top=.84,bottom=.22,wspace=.35)
    folder=run.PUBLIC/'figures';folder.mkdir(exist_ok=True)
    fig.savefig(folder/'summary.svg',metadata={'Date':None});fig.savefig(run.PRIVATE/'summary.png',dpi=150);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=('traces','report','figures','bundle'));args=parser.parse_args()
    if args.phase=='traces':print(json.dumps(dict(heads=traces()['heads'])))
    elif args.phase=='figures':figures();print('Verified aggregate SVG and private preview created')
    elif args.phase=='bundle':print(json.dumps(bundle.pack(ROOT,run.PUBLIC)))
    else:
        value,summary=evidence();text=render(value,summary);path=run.PUBLIC/'readout.md'
        if path.exists() and path.read_text()!=text:raise ValueError('Frozen readout changed')
        path.write_text(text);print(json.dumps(value['gates']))
