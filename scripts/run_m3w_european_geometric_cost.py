"""Fit matched causal-envelope cost heads; keep forecasts and source roles fixed."""
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
from scripts import run_m3w_european_nested_calibration as old
from scripts.run_m3w_native_forecast import immutable_json,json_write,array_hash
from src.world_model.m3w_geometric_cost_head import rollout_envelope,initialize_head,fit,predict
from src.evaluation.m3w_producer_transport import score_diagnosis
from src.evaluation.m3w_native_metrics import native_errors,paired_scene_metrics

PUBLIC=ROOT/'outputs/publication_readiness_2026_09/european_geometric_cost_v1'
PRIVATE=ROOT/'data/stage_cvpr2027_experiments/european_geometric_cost_v1'
TRANSPORT=ROOT/'outputs/publication_readiness_2026_09/european_producer_transport_v1'
CONFIG='configs/m3w_european_geometric_cost_v1.json'
FILES=(CONFIG,'scripts/run_m3w_european_geometric_cost.py',
    'src/world_model/m3w_geometric_cost_head.py','tests/test_m3w_geometric_cost_head.py',
    'src/evaluation/m3w_producer_transport.py','src/evaluation/m3w_opportunity_diagnosis.py',
    'outputs/publication_readiness_2026_09/european_geometric_cost_v1/registration.md',
    'scripts/diagnose_m3w_european_score_scaling.py',
    'outputs/publication_readiness_2026_09/european_score_scaling_v1/diagnosis.json')


def artifact(path):
    return dict(path=str(path.relative_to(ROOT)),sha256=old.digest(path))


def beat(state,**kw):
    row=dict(pid=os.getpid(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),state=state,**kw)
    json_write(PRIVATE/'heartbeat.json',row)
    with (PRIVATE/'events.jsonl').open('a') as f:
        f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)


def assert_identity(identity):
    old.assert_identity(identity['old_identity'])
    if old.digest(TRANSPORT/'analysis.json')!=identity['transport_analysis_sha256']:
        raise ValueError('Starting transport evidence changed')
    for path,sha in identity['bindings'].items():
        if old.digest(ROOT/path)!=sha:
            raise ValueError('Frozen geometric-cost binding changed: '+path)


def load():
    reg=json.loads((ROOT/CONFIG).read_text())
    previous,_,data,old_id=old.load()
    v=json.loads((TRANSPORT/'verification.json').read_text())
    if not v['all_passed'] or v['analysis_sha256']!=old.digest(TRANSPORT/'analysis.json'):
        raise ValueError('Verified previous diagnosis required')
    for key in ('seeds','candidates','head_training'):
        if reg[key]!=previous[key]:
            raise ValueError('Matched original fitting design required')
    if reg['risk_budget']!=previous['predicted_risk_budget'] or any(reg[k] for k in (
            'new_forecaster_training','threshold_refit','calibration_refit','reserved_roles_opened',
            'deployment_changed','stage5c_executed','smc_enabled')):
        raise ValueError('Source-only fixed-budget head ablation required')
    identity=dict(old_identity=old_id,transport_analysis_sha256=v['analysis_sha256'],
        bindings={p:old.digest(ROOT/p) for p in FILES})
    originals=old.all_heads(previous,data,old_id)
    immutable_json(PRIVATE/'identity.json',identity)
    immutable_json(PUBLIC/'matrix.json',dict(identity=identity,new_heads=54,updates=108000,
        policy_views=144,head_replacement_comparisons=108,neural_vs_damping_comparisons=72,
        fixed_candidates=2,forecast_training=False,reserved_roles_opened=False))
    return reg,previous,data,identity,originals


def checked(key,identity):
    r=json.loads((PRIVATE/'heads'/key/'complete.json').read_text())
    if r['identity']['experiment']!=identity or r['fit']['step']!=2000 or not r['fit']['complete']:
        raise ValueError('Incomplete or changed fitted head')
    for a in r['artifacts'].values():
        if old.digest(ROOT/a['path'])!=a['sha256']:
            raise ValueError('Changed head artifact')
    return r


def train(reg,previous,data,identity,originals,*,resume,pilot):
    for name,candidate,fold,seed,design in old.jobs(previous,data,identity['old_identity']):
        a=old.assemble(candidate,fold,seed,design,data,identity['old_identity'])
        envelope=rollout_envelope(a['b'],a['p'])
        fitting,held=design['train_ids'],design['held_ids']
        for task in reg['tasks']:
            key=name+'_'+task;directory=PRIVATE/'heads'/key
            y,pr=old.task_data(a,design,data,task)
            parent=originals[key]
            if parent['identity']['target_sha256']!=array_hash(y) or parent['identity']['lineage']!=a['lineage']:
                raise ValueError('Original input/target identity changed')
            hid=dict(experiment=identity,lineage=a['lineage'],task=task,
                target_sha256=array_hash(y),envelope_train_sha256=array_hash(envelope[fitting]),
                cost_scale=pr['cost_scale'],original_checkpoint=parent['artifacts']['checkpoint'])
            if (directory/'complete.json').exists():
                if checked(key,identity)['identity']!=hid:
                    raise ValueError('Changed head fitting lineage')
                continue
            if shutil.disk_usage(PRIVATE).free<10*1024**3:
                raise OSError('Below10GiB; preserve completed artifacts')
            beat('head_training',head=key)
            model,report=fit(a['x'][fitting],y,data['sites'][fitting],envelope[fitting],pr,
                seed=seed,task=task,settings=reg['head_training'],identity=hid,directory=directory,
                resume=resume,stop_at=100 if pilot else None,heartbeat=lambda **kw:beat(head=key,**kw))
            if pilot:
                immutable_json(PRIVATE/'pilot.json',dict(identity=hid,fit=report,
                    checkpoint=artifact(directory/'checkpoint.pt')))
                return
            pred=predict(model,a['x'][held],envelope[held],pr)
            path=directory/'scores.npz';tmp=path.with_suffix('.tmp.npz')
            np.savez(tmp,ids=held,scores=pred);os.replace(tmp,path)
            immutable_json(directory/'complete.json',dict(identity=hid,fit=report,
                artifacts=dict(checkpoint=artifact(directory/'checkpoint.pt'),scores=artifact(path))))
            assert_identity(identity);beat('head_complete',head=key,steps=report['step'])


def replay(reg,previous,data,identity,originals):
    checks=[]
    for name,candidate,fold,seed,design in old.jobs(previous,data,identity['old_identity']):
        a=old.assemble(candidate,fold,seed,design,data,identity['old_identity'])
        ids=design['held_ids'][:reg['replay_rows']]
        envelope=rollout_envelope(a['b'][ids],a['p'][ids])
        for task in reg['tasks']:
            key=name+'_'+task;r=checked(key,identity)
            state=torch.load(ROOT/r['artifacts']['checkpoint']['path'],map_location='cpu',weights_only=False)
            parent=torch.load(ROOT/originals[key]['artifacts']['checkpoint']['path'],map_location='cpu',weights_only=False)
            if state['identity']!=r['identity'] or state['step']!=2000:
                raise ValueError('Wrong saved training state')
            np.testing.assert_array_equal(state['draws'],parent['draws'])
            for field in ('mean','std','constant','weights','known'):
                np.testing.assert_array_equal(state['preprocess'][field],parent['preprocess'][field])
            model=initialize_head(reg['head_training']['width'],state['preprocess'],seed,task,state['mean_envelope'])
            model.load_state_dict(state['model'])
            fresh=predict(model,a['x'][ids],envelope,state['preprocess'])
            np.testing.assert_array_equal(fresh,old.scores(r,ids))
            if task=='utility':
                if np.any(fresh.sum(1)>envelope+1e-4):raise ValueError('Utility envelope violated')
            elif np.any(fresh[:,1]>envelope+1e-4):raise ValueError('Risk envelope violated')
            checks.append(dict(head=key,checkpoint=r['artifacts']['checkpoint'],rows=len(ids),exact=True,
                sampler_exact=True,total_draws=int(state['draws'].sum()),parameters=r['fit']['parameters']))
        beat('head_group_replayed',group=name)
    assert_identity(identity)
    immutable_json(PUBLIC/'head_replay.json',dict(identity=identity,checks=checks,all_passed=True))


def evaluate(reg,previous,data,identity,originals,verify):
    replay_report=json.loads((PUBLIC/'head_replay.json').read_text())
    if replay_report['identity']!=identity or len(replay_report['checks'])!=54 or not replay_report['all_passed']:
        raise ValueError('All matched fitted heads must replay before readout')
    prior=json.loads((TRANSPORT/'analysis.json').read_text())
    views,contrasts,versus={}, {}, {}
    original_matches=0
    pair_costs={}
    for name,candidate,fold,seed,design in old.jobs(previous,data,identity['old_identity']):
        a=old.assemble(candidate,fold,seed,design,data,identity['old_identity'])
        ids=design['held_ids'];sites=data['sites'][ids];roster=sorted(set(sites))
        cv,cvf=np.asarray(data['baseline_ade'][ids,1]),np.asarray(data['baseline_fde'][ids,1])
        ade,fde=native_errors(a['p'][ids].astype(float)+data['origin'][ids,None],
            data['target_eval'][ids],data['valid'][ids],np.ones(len(ids)))
        moving=np.linalg.norm(np.diff(data['history'][ids],axis=1),axis=2).sum(1)>0
        masks=dict(all=np.ones(len(ids),bool),easy=(cv>0)&(cv<=design['easy_cut']),hard=cv>=design['hard_cut'])
        def metric(cost,reference,mask):
            return paired_scene_metrics(cost[mask],reference[mask],sites[mask],expected_scenes=roster,
                dataset='EuropeanSquares_released_detector_tracks',coordinate_unit='image_pixel',
                bootstrap_resamples=reg['bootstrap_resamples'],seed=reg['bootstrap_seed'])
        scores={family:{task:old.scores((originals[name+'_'+task] if family=='old'
                else checked(name+'_'+task,identity)),ids) for task in reg['tasks']} for family in ('old','new')}
        local={}
        for event in ('all','easy'):
            for variant in reg['variants']:
                u=scores['new' if variant in ('utility_only','both') else 'old']['utility']
                risk=scores['new' if variant in ('risk_only','both') else 'old'][event]
                d=score_diagnosis(cv,ade,u,risk,moving,sites,easy_cut=design['easy_cut'],
                    event=event,budget=reg['risk_budget'],resamples=reg['bootstrap_resamples'],seed=reg['bootstrap_seed'])
                bits=d.pop('reasons')==5
                selected=np.where(bits,ade,cv);selected_fde=np.where(bits,fde,cvf)
                metrics={s:metric(selected,cv,m) for s,m in masks.items()}
                zero=np.isfinite(cv)&(cv==0)
                key=f'{candidate}_fold{fold}_seed{seed}_{event}_{variant}'
                views[key]=dict(**d,ADE_vs_CV=metrics,FDE_vs_CV=metric(selected_fde,cvf,masks['all']),
                    raw_ADE_vs_CV=metric(ade,cv,masks['all']),
                    zero_CV=dict(rows=int(zero.sum()),harmed_rows=int((selected[zero]>0).sum())),
                    decision_sha256=array_hash(ids,bits),
                    safety_observed_pass=bool(metrics['easy']['worst_scene_gain_percent'] is not None
                        and metrics['easy']['worst_scene_gain_percent']>=-2 and not (selected[zero]>0).any()))
                local[(event,variant)]=selected
                if variant=='old':
                    producer='full4' if candidate=='neural' else 'damping097'
                    p=prior['views'][f'fold{fold}_seed{seed}_{producer}_{event}']
                    if views[key]['decision_sha256']!=p['decision_sha256'] or metrics!=p['ADE_vs_CV']:
                        raise ValueError('Matched old policy does not reproduce exactly')
                    original_matches+=1
        for event in ('all','easy'):
            for variant in reg['variants'][1:]:
                key=f'{candidate}_fold{fold}_seed{seed}_{event}_{variant}'
                contrasts[key]={s:metric(local[(event,variant)],local[(event,'old')],m) for s,m in masks.items()}
        if candidate=='neural':
            pair_costs[(fold,seed)]=local
        else:
            for event in ('all','easy'):
                for variant in reg['variants']:
                    versus[f'fold{fold}_seed{seed}_{event}_{variant}']={s:metric(
                        pair_costs[(fold,seed)][(event,variant)],local[(event,variant)],m) for s,m in masks.items()}
            del pair_costs[(fold,seed)]
        beat('group_evaluated',group=name,verify=verify)
    if len(views)!=144 or len(contrasts)!=108 or len(versus)!=72 or original_matches!=36:
        raise ValueError('Incomplete registered comparison matrix')
    assert_identity(identity)
    result=dict(identity=identity,result_source='fresh_run_54_envelope_heads_cached_verified_forecasts_controls',
        views=views,replacements=contrasts,neural_vs_damping=versus,original_views_reproduced=original_matches,
        new_heads=54,new_updates=108000,forecasts_unchanged=True,threshold_refit=False,
        reserved_roles_opened=False,deployment_changed=False,stage5c_executed=False,smc_enabled=False)
    immutable_json(PUBLIC/'analysis.json',result)
    if verify:
        immutable_json(PUBLIC/'verification.json',dict(identity=identity,analysis_sha256=old.digest(PUBLIC/'analysis.json'),
            all_passed=True,metrics_recomputed=True,original_views_reproduced=36))
    beat('evaluation_complete',views=144,verify=verify)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for flag in ('prepare','pilot','train','resume','replay','evaluate','verify'):
        parser.add_argument('--'+flag,action='store_true')
    args=parser.parse_args()
    if not any((args.prepare,args.pilot,args.train,args.replay,args.evaluate,args.verify)):
        parser.error('Explicit experiment phase required')
    PRIVATE.mkdir(parents=True,exist_ok=True)
    with (PRIVATE/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.set_num_threads(4);torch.set_num_interop_threads(1)
        reg,previous,data,identity,originals=load()
        beat('verified_start',threads=4,workers=0,architecture=platform.machine())
        if args.pilot or args.train:
            train(reg,previous,data,identity,originals,resume=args.resume,pilot=args.pilot)
        if args.replay or args.verify:
            replay(reg,previous,data,identity,originals)
        if args.evaluate or args.verify:
            evaluate(reg,previous,data,identity,originals,verify=args.verify)
        beat('phase_complete')


if __name__=='__main__':
    main()
