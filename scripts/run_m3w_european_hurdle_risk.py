"""Matched product-MSE versus occurrence/severity supervision for causal risk."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time

if platform.system()=='Darwin' and platform.machine()!='arm64':
    raise RuntimeError('Native arm64 required before Torch import')
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key,'4')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts import run_m3w_european_geometric_cost as geo
from scripts.run_m3w_native_forecast import immutable_json,json_write,array_hash
from src.world_model.m3w_hurdle_risk import fit,predict,initialize,factor_targets
from src.evaluation.m3w_native_metrics import native_errors,paired_scene_metrics
from src.evaluation.m3w_producer_transport import score_diagnosis

old=geo.old
PUBLIC=ROOT/'outputs/publication_readiness_2026_09/european_hurdle_risk_v1'
PRIVATE=ROOT/'data/stage_cvpr2027_experiments/european_hurdle_risk_v1'
CONFIG='configs/m3w_european_hurdle_risk_v1.json'
FILES=(CONFIG,'scripts/run_m3w_european_hurdle_risk.py','src/world_model/m3w_hurdle_risk.py',
    'tests/test_m3w_hurdle_risk.py','scripts/diagnose_m3w_event_support.py',
    'outputs/publication_readiness_2026_09/european_hurdle_risk_v1/registration.md',
    'outputs/publication_readiness_2026_09/european_hurdle_risk_v1/fitting_support.json')


def beat(state,**kw):
    row=dict(pid=os.getpid(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),state=state,**kw)
    json_write(PRIVATE/'heartbeat.json',row)
    with (PRIVATE/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)


def assert_identity(identity):
    geo.assert_identity(identity['geometric_identity'])
    if old.digest(geo.PUBLIC/'analysis.json')!=identity['geometric_analysis_sha256']:
        raise ValueError('Frozen geometric results changed')
    for file,h in identity['bindings'].items():
        if old.digest(ROOT/file)!=h:raise ValueError('Frozen hurdle binding changed: '+file)


def load():
    reg=json.loads((ROOT/CONFIG).read_text())
    greg,previous,data,gid,originals=geo.load()
    verification=json.loads((geo.PUBLIC/'verification.json').read_text())
    if not verification['all_passed'] or verification['identity']!=gid or verification['analysis_sha256']!=old.digest(geo.PUBLIC/'analysis.json'):
        raise ValueError('Verified complete geometric experiment required')
    for k in ('seeds','candidates','head_training'):
        if reg[k]!=greg[k]:raise ValueError('Unmatched fitting design')
    if reg['risk_budget']!=greg['risk_budget'] or any(reg[k] for k in (
            'new_forecaster_training','threshold_refit','calibration_refit','reserved_roles_opened',
            'deployment_changed','stage5c_executed','smc_enabled')):
        raise ValueError('Fixed source-only experiment required')
    identity=dict(geometric_identity=gid,geometric_analysis_sha256=verification['analysis_sha256'],
        bindings={f:old.digest(ROOT/f) for f in FILES})
    immutable_json(PRIVATE/'identity.json',identity)
    immutable_json(PUBLIC/'matrix.json',dict(identity=identity,new_heads=72,updates=144000,views=144,
        matched_controls=72,objective_contrasts=36,neural_damping_contrasts=72))
    return reg,previous,data,identity,originals


def checked(key,identity):
    r=json.loads((PRIVATE/'heads'/key/'complete.json').read_text())
    if r['identity']['experiment']!=identity or not r['fit']['complete'] or r['fit']['step']!=2000:
        raise ValueError('Incomplete or changed hurdle head')
    for a in r['artifacts'].values():
        if old.digest(ROOT/a['path'])!=a['sha256']:raise ValueError('Changed head artifact')
    return r


def jobs(previous,data,identity):
    return old.jobs(previous,data,identity['geometric_identity']['old_identity'])


def assemble(candidate,fold,seed,design,data,identity):
    return old.assemble(candidate,fold,seed,design,data,identity['geometric_identity']['old_identity'])


def train(reg,previous,data,identity,originals,*,resume,pilot):
    for name,candidate,fold,seed,design in jobs(previous,data,identity):
        a=assemble(candidate,fold,seed,design,data,identity)
        envelope=geo.rollout_envelope(a['b'],a['p']);fitting,held=design['train_ids'],design['held_ids']
        for event in reg['events']:
            y,pr=old.task_data(a,design,data,event);parent=originals[name+'_'+event]
            if parent['identity']['target_sha256']!=array_hash(y) or parent['identity']['lineage']!=a['lineage']:
                raise ValueError('Original input/target identity changed')
            for arm in reg['training_arms']:
                key=name+'_'+event+'_'+arm;directory=PRIVATE/'heads'/key
                hid=dict(experiment=identity,lineage=a['lineage'],event=event,arm=arm,
                    target_sha256=array_hash(y),envelope_train_sha256=array_hash(envelope[fitting]),
                    cost_scale=pr['cost_scale'],original_checkpoint=parent['artifacts']['checkpoint'])
                if (directory/'complete.json').exists():
                    if checked(key,identity)['identity']!=hid:raise ValueError('Fitting lineage changed')
                    continue
                if shutil.disk_usage(PRIVATE).free<10*1024**3:raise OSError('Below10GiB; keep completed work')
                beat('head_training',head=key)
                model,report=fit(a['x'][fitting],y,data['sites'][fitting],envelope[fitting],pr,
                    seed=seed,arm=arm,settings=reg['head_training'],identity=hid,directory=directory,
                    resume=resume,stop_at=100 if pilot else None,heartbeat=lambda **kw:beat(head=key,**kw))
                if pilot:
                    immutable_json(PRIVATE/'pilot.json',dict(identity=hid,fit=report,checkpoint=geo.artifact(directory/'checkpoint.pt')))
                    return
                pred=predict(model,a['x'][held],envelope[held],pr,components=True)
                path=directory/'scores.npz';tmp=path.with_suffix('.tmp.npz')
                np.savez(tmp,ids=held,scores=pred[:,:2],factors=pred[:,2:]);os.replace(tmp,path)
                immutable_json(directory/'complete.json',dict(identity=hid,fit=report,
                    artifacts=dict(checkpoint=geo.artifact(directory/'checkpoint.pt'),scores=geo.artifact(path))))
                assert_identity(identity);beat('head_complete',head=key,steps=report['step'])


def factors(r,ids):
    with np.load(ROOT/r['artifacts']['scores']['path'],allow_pickle=False) as z:
        pos=np.searchsorted(z['ids'],ids);np.testing.assert_array_equal(z['ids'][pos],ids)
        return z['factors'][pos].copy()


def replay(reg,previous,data,identity,originals):
    checks=[]
    for name,candidate,fold,seed,design in jobs(previous,data,identity):
        a=assemble(candidate,fold,seed,design,data,identity);ids=design['held_ids'][:reg['replay_rows']]
        envelope=geo.rollout_envelope(a['b'][ids],a['p'][ids])
        for event in reg['events']:
            parent=torch.load(ROOT/originals[name+'_'+event]['artifacts']['checkpoint']['path'],map_location='cpu',weights_only=False)
            for arm in reg['training_arms']:
                key=name+'_'+event+'_'+arm;r=checked(key,identity)
                state=torch.load(ROOT/r['artifacts']['checkpoint']['path'],map_location='cpu',weights_only=False)
                if state['identity']!=r['identity'] or state['step']!=2000:raise ValueError('Wrong training state')
                np.testing.assert_array_equal(state['draws'],parent['draws'])
                for f in ('mean','std','constant','weights','known'):
                    np.testing.assert_array_equal(state['preprocess'][f],parent['preprocess'][f])
                model=initialize(state['preprocess'],state['prior'],reg['head_training']['width'],seed)
                model.load_state_dict(state['model'])
                pred=predict(model,a['x'][ids],envelope,state['preprocess'],components=True)
                np.testing.assert_array_equal(pred[:,:2],old.scores(r,ids))
                np.testing.assert_array_equal(pred[:,2:],factors(r,ids))
                if np.any(pred[:,1]>envelope+1e-4):raise ValueError('Causal bound violated')
                checks.append(dict(head=key,checkpoint=r['artifacts']['checkpoint'],rows=len(ids),exact=True,
                    sampler_exact=True,total_draws=int(state['draws'].sum()),parameters=r['fit']['parameters']))
        beat('head_group_replayed',group=name)
    assert_identity(identity)
    immutable_json(PUBLIC/'head_replay.json',dict(identity=identity,checks=checks,all_passed=len(checks)==72))


def calibration_diagnostic(factor,labels,envelope,sites,selected):
    truth=factor_targets(labels,envelope);out={}
    for site in sorted(set(sites)):
        out[str(site)]={}
        for name,mask in [('population',(sites==site)&truth['known']),('selected',(sites==site)&truth['known']&selected)]:
            n=int(mask.sum());p=factor[mask,0];t=truth['positive'][mask]
            q=(sites==site)&truth['positive']&(selected if name=='selected' else True)
            bins=[];ece=0.
            for i in range(10):
                use=(p>=i/10)&(p<(i+1)/10 if i<9 else p<=1)
                if use.any():
                    a,b=float(p[use].mean()),float(t[use].mean())
                    ece+=use.sum()*abs(a-b)/max(n,1)
                    bins.append(dict(bin=i,rows=int(use.sum()),predicted=a,actual=b))
            out[str(site)][name]=dict(rows=n,positive_rows=int(t.sum()),probability_mean=float(p.mean()) if n else None,
                positive_rate=float(t.mean()) if n else None,brier=float(np.square(p-t).mean()) if n else None,
                ece=float(ece) if n else None,positive_fraction_mse=float(np.square(factor[q,1]-truth['fraction'][q]).mean()) if q.any() else None,bins=bins)
    return out


def evaluate(reg,previous,data,identity,originals,verify):
    receipt=json.loads((PUBLIC/'head_replay.json').read_text())
    if receipt['identity']!=identity or not receipt['all_passed'] or len(receipt['checks'])!=72:
        raise ValueError('Every head must finish and replay before outcome readout')
    prior=json.loads((geo.PUBLIC/'analysis.json').read_text())
    views,replacements,versus,objective_pairs,envelope_pairs={},{},{},{},{}
    pairs={};matches=0
    for name,candidate,fold,seed,design in jobs(previous,data,identity):
        a=assemble(candidate,fold,seed,design,data,identity)
        ids=design['held_ids'];sites=data['sites'][ids];roster=sorted(set(sites))
        cv,cvf=np.asarray(data['baseline_ade'][ids,1]),np.asarray(data['baseline_fde'][ids,1])
        ade,fde=native_errors(a['p'][ids].astype(float)+data['origin'][ids,None],data['target_eval'][ids],data['valid'][ids],np.ones(len(ids)))
        moving=np.linalg.norm(np.diff(data['history'][ids],axis=1),axis=2).sum(1)>0
        envelope=geo.rollout_envelope(a['b'][ids],a['p'][ids])
        masks=dict(all=np.ones(len(ids),bool),easy=(cv>0)&(cv<=design['easy_cut']),hard=cv>=design['hard_cut'])
        def metric(cost,reference,mask):
            return paired_scene_metrics(cost[mask],reference[mask],sites[mask],expected_scenes=roster,
                dataset='EuropeanSquares_released_detector_tracks',coordinate_unit='image_pixel',
                bootstrap_resamples=reg['bootstrap_resamples'],seed=reg['bootstrap_seed'])
        utility=old.scores(originals[name+'_utility'],ids);local={}
        for event in reg['events']:
            records=dict(old=originals[name+'_'+event],envelope=geo.checked(name+'_'+event,identity['geometric_identity']))
            records.update({arm:checked(name+'_'+event+'_'+arm,identity) for arm in reg['training_arms']})
            for variant in reg['variants']:
                risk=old.scores(records[variant],ids)
                d=score_diagnosis(cv,ade,utility,risk,moving,sites,easy_cut=design['easy_cut'],event=event,
                    budget=reg['risk_budget'],resamples=reg['bootstrap_resamples'],seed=reg['bootstrap_seed'])
                bits=d.pop('reasons')==5;selected=np.where(bits,ade,cv)
                metrics={s:metric(selected,cv,m) for s,m in masks.items()};zero=np.isfinite(cv)&(cv==0)
                key=f'{candidate}_fold{fold}_seed{seed}_{event}_{variant}'
                row=dict(**d,ADE_vs_CV=metrics,FDE_vs_CV=metric(np.where(bits,fde,cvf),cvf,masks['all']),
                    raw_ADE_vs_CV=metric(ade,cv,masks['all']),zero_CV=dict(rows=int(zero.sum()),harmed_rows=int((selected[zero]>0).sum())),
                    decision_sha256=array_hash(ids,bits),safety_observed_pass=bool(metrics['easy']['worst_scene_gain_percent'] is not None
                        and metrics['easy']['worst_scene_gain_percent']>=-2 and not (selected[zero]>0).any()))
                if variant in reg['training_arms']:
                    label=old.event_labels(cv,np.maximum(ade-cv,0),easy_cut=design['easy_cut'],event=event)
                    row['factor_reliability']=calibration_diagnostic(factors(records[variant],ids),label,envelope,sites,bits)
                else:
                    prior_arm='old' if variant=='old' else 'risk_only'
                    p=prior['views'][f'{candidate}_fold{fold}_seed{seed}_{event}_{prior_arm}']
                    if row['decision_sha256']!=p['decision_sha256'] or metrics!=p['ADE_vs_CV']:
                        raise ValueError('Original or envelope control changed')
                    matches+=1
                views[key]=row;local[(event,variant)]=selected
            for variant in reg['variants'][1:]:
                key=f'{candidate}_fold{fold}_seed{seed}_{event}_{variant}'
                replacements[key]={s:metric(local[(event,variant)],local[(event,'old')],m) for s,m in masks.items()}
            key=f'{candidate}_fold{fold}_seed{seed}_{event}'
            objective_pairs[key]={s:metric(local[(event,'hurdle')],local[(event,'product_mse')],m) for s,m in masks.items()}
            envelope_pairs[key]={s:metric(local[(event,'hurdle')],local[(event,'envelope')],m) for s,m in masks.items()}
        if candidate=='neural':pairs[(fold,seed)]=local
        else:
            for event in reg['events']:
                for arm in reg['variants']:
                    versus[f'fold{fold}_seed{seed}_{event}_{arm}']={s:metric(pairs[(fold,seed)][(event,arm)],local[(event,arm)],m) for s,m in masks.items()}
            del pairs[(fold,seed)]
        beat('group_evaluated',group=name,verify=verify)
    if len(views)!=144 or len(replacements)!=108 or len(versus)!=72 or matches!=72 or len(objective_pairs)!=36:
        raise ValueError('Incomplete registered matrix')
    assert_identity(identity)
    result=dict(identity=identity,result_source='fresh_run_72_risk_heads_cached_verified_forecasts_utility_controls',
        views=views,replacements=replacements,neural_vs_damping=versus,objective_pairs=objective_pairs,
        envelope_pairs=envelope_pairs,original_views_reproduced=matches,new_heads=72,new_updates=144000,
        forecasts_unchanged=True,threshold_refit=False,reserved_roles_opened=False,deployment_changed=False,
        stage5c_executed=False,smc_enabled=False)
    immutable_json(PUBLIC/'analysis.json',result)
    if verify:immutable_json(PUBLIC/'verification.json',dict(identity=identity,analysis_sha256=old.digest(PUBLIC/'analysis.json'),
        all_passed=True,metrics_recomputed=True,original_views_reproduced=72))
    beat('evaluation_complete',views=144,verify=verify)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for flag in ('prepare','pilot','train','resume','replay','evaluate','verify'):parser.add_argument('--'+flag,action='store_true')
    args=parser.parse_args()
    if not any((args.prepare,args.pilot,args.train,args.replay,args.evaluate,args.verify)):parser.error('Explicit phase required')
    PRIVATE.mkdir(parents=True,exist_ok=True)
    with (PRIVATE/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.set_num_threads(4);torch.set_num_interop_threads(1)
        reg,previous,data,identity,originals=load();beat('verified_start',threads=4,workers=0,architecture=platform.machine())
        if args.pilot or args.train:train(reg,previous,data,identity,originals,resume=args.resume,pilot=args.pilot)
        if args.replay or args.verify:replay(reg,previous,data,identity,originals)
        if args.evaluate or args.verify:evaluate(reg,previous,data,identity,originals,verify=args.verify)
        beat('phase_complete')


if __name__=='__main__':main()
