"""All six source assignments and all three seeds; no favorable-role selection."""
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts import run_m3w_european_support_fractional as run
from scripts.report_m3w_european_harm_tail_crossfit import summarize_population


def differences(fold):
    a,b=[fold['metrics'][arm] for arm in ('fractional','mean')]
    def gain(x,y): return 100*(y-x)/y if x is not None and y is not None and y>0 else None
    out={}
    for subset in ('all','envelope_positive'):
        x,y=a[subset],b[subset]
        if any(v.get('status')=='not_estimable' for v in (x,y)):
            for k in ('harm_MSE_gain_percent','top10_gain_pp','AUROC_delta','coverage_log_error_reduction'):
                out[subset+'__'+k]=None
            continue
        out[subset+'__harm_MSE_gain_percent']=gain(x['harm_MSE'],y['harm_MSE'])
        for metric,key,factor in (('top10_harm_mass_share','top10_gain_pp',100),('AUROC','AUROC_delta',1)):
            v,w=[z['scores']['moment'][metric] for z in (x,y)]
            out[subset+'__'+key]=factor*(v-w) if v is not None and w is not None else None
        v,w=x['harm_coverage'],y['harm_coverage']
        out[subset+'__coverage_log_error_reduction']=float(abs(np.log(w))-abs(np.log(v))) if v is not None and w is not None and min(v,w)>0 else None
    x,y=a['all']['component_MSE'],b['all']['component_MSE']
    for index,label in enumerate(('D_all','H_all','D_easy','H_easy')):
        out['all__'+label+'_MSE_gain_percent']=gain(x[index],y[index]) if x is not None and y is not None else None
    return out


def paired_contrasts(rows,cfg):
    out={}
    for pair in cfg['pairs']:
        out[pair]={}
        for a,b in sorted({(r['producer'],r['controller']) for r in rows}):
            group=[r for r in rows if r['pair']==pair and (r['producer'],r['controller'])==(a,b)]
            assert len(group)==3 and sorted(r['seed'] for r in group)==cfg['seeds']
            sites=sorted({f['held'] for r in group for f in r['folds']}); assert len(sites)==4
            cs={}
            for key in differences(group[0]['folds'][0]):
                values=[]
                for site in sites:
                    numbers=[differences(next(f for f in r['folds'] if f['held']==site))[key] for r in group]
                    if any(v is None or not np.isfinite(v) for v in numbers): break
                    values.append(float(np.mean(numbers)))
                if len(values)!=4: cs[key]=dict(status='not_estimable',reason='missing_support_not_dropped'); continue
                v=np.array(values); rng=np.random.default_rng(cfg['bootstrap_seed'])
                draws=v[rng.integers(0,4,size=(cfg['bootstrap_resamples'],4))].mean(1)
                cs[key]=dict(point=float(v.mean()),CI=np.quantile(draws,[.025,.975]).tolist(),
                    locality_points=dict(zip(sites,values)),seeds=3,localities=4)
            out[pair][f'producer{a}_controller{b}']=cs
    return out


def main():
    cfg,identity=run.registration(); done=run.checked_training(identity)
    checks=json.loads((run.PUBLIC/'completion_checks.json').read_text()); rows=[]
    for ref in checks['groups']:
        assert run.artifact(ROOT/ref['path'])==ref; rows.append(json.loads((ROOT/ref['path']).read_text()))
    cs=paired_contrasts(rows,cfg); populations={}; summary={}
    for pair in cfg['pairs']:
        group=[r for r in rows if r['pair']==pair]
        populations[pair]={subset:{arm:summarize_population([f['metrics'][arm][subset] for r in group for f in r['folds']])
            for arm in ('mean','fractional')} for subset in ('all','envelope_positive')}
        summary[pair]={}
        for key in next(iter(cs[pair].values())):
            vals=[v[key] for v in cs[pair].values()]; good=[v for v in vals if 'CI' in v]
            summary[pair][key]=dict(positive=sum(v['CI'][0]>0 for v in good),negative=sum(v['CI'][1]<0 for v in good),
                overlap=sum(v['CI'][0]<=0<=v['CI'][1] for v in good),not_estimable=len(vals)-len(good),
                point_range=[min(v['point'] for v in good),max(v['point'] for v in good)] if good else None)
    run.immutable_json(run.PUBLIC/'aggregate_metrics.json',dict(populations=populations,contrasts=cs,summary=summary))
    fits=[json.loads((ROOT/ref['path']).read_text()) for ref in done['heads']]
    run.immutable_json(run.PUBLIC/'training_metrics.json',[dict(group=r['input']['group']['group'],pair=r['input']['pair'],
        held=r['input']['held'],fit=r['fit']) for r in fits])
    run.immutable_json(run.PUBLIC/'compute_receipt.json',dict(heads=len(fits),updates=sum(r['fit']['step'] for r in fits),
        summed_fit_seconds=sum(r['fit']['seconds'] for r in fits),unknown_draws=sum(r['fit']['unknown_rows_sampled'] for r in fits),
        target_roundoff_clamps=sum(r['fit']['target_roundoff_clamps'] for r in fits),threads=4,interop=1,workers=0,
        real_torch=True,new_forecaster_training=False,remote_jobs_submitted=0,
        create_followup=run.artifact(run.PRIVATE/'create_followup.json'),remote_M3W_inventory='not_run_unverified_project_path'))
    full=summary['full']; primary=full['envelope_positive__harm_MSE_gain_percent']
    guards=[full['envelope_positive__'+k] for k in ('top10_gain_pp','coverage_log_error_reduction')]
    advance=primary['positive']==6 and all(v['negative']==v['not_estimable']==0 for v in guards)
    gates=dict(training_complete=True,matched_initialization_draws_steps=True,locality_exclusion=True,
        primary_six_positive=primary['positive']==6,tail_and_coverage_guards=all(v['negative']==v['not_estimable']==0 for v in guards),
        development_advance_gate=advance,new_policy_evaluated=False,independent_confirmation=False,
        deployment_changed=False,submission_ready=False,stage5c_executed=False,smc_enabled=False)
    run.immutable_json(run.PUBLIC/'gates.json',gates)
    lines=['# Support-Fractional Harm Results','','## Material Passport',
        '144 fresh Torch heads / 288,000 updates; matched cached mean heads; no new forecast or policy.',
        'Previously opened source development. Three-site easy definitions vary across folds.','',
        '## Population Diagnostics','',
        '| Pair / subset / arm | Dependent views | Weak support | Event AUROC | AP/prevalence | Top10 harm share | Harm coverage |',
        '|---|---:|---:|---:|---:|---:|---:|']
    def fmt(x): return 'not_estimable' if x is None else f'{x:.5f}'
    for pair,subsets in populations.items():
        for subset,arms in subsets.items():
            for arm,v in arms.items():
                s=v['scores']['moment']; vals=[s['AUROC']['median'],s['AP_over_prevalence']['median'],s['top10_harm_mass_share']['median'],v['harm_coverage']['median']]
                lines.append(f"| {pair} / {subset} / {arm} | {v['population_views']} | {v['weak_support']} | "+' | '.join(map(fmt,vals))+' |')
    lines+=['','## Paired Contrasts','','All signs favor the fractional auxiliary when positive. MSE gains are percentages; tail differences are pp.',
        'Coverage contrast is reduction in absolute log(predicted/actual harm), not percentage coverage.',
        'Three seeds averaged within locality, then 3,000 resamples of four localities; exploratory, no multiplicity correction.','',
        '| Pair / source roles | Contrast | Point | 95% CI |','|---|---|---:|---:|']
    for pair,groups in cs.items():
        for group,items in groups.items():
            for key,v in items.items():
                lines.append(f"| {pair} / {group} | {key} | {fmt(v.get('point'))} | {v.get('CI','not_estimable')} |")
    lines+=['','## Gates','',*['- '+k+': '+str(v).lower() for k,v in gates.items()],
        '','The fractional score is expected severity fraction, not a failure probability or risk certificate.',
        'Image pixels, annotation steps, detector-derived labels; no metric/seconds, human-gold, true3D or foundation claim.',
        'Reserved calibration/confirmation remain closed. No deployment. Stage5C and SMC remain off.']
    (run.PUBLIC/'results.md').write_text('\n'.join(lines)+'\n'); print(json.dumps(dict(primary=primary,gates=gates),indent=2))


if __name__=='__main__': main()
