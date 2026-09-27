"""Source-separated fixed-floor signed-excess objective contrast."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_m3w_european_fixed_floor_tail as parent
from src.world_model import m3w_fixed_floor_excess as api
import numpy as np
torch,inter=parent.torch,parent.inter
floor_api=parent.parent
PUBLIC=parent.PUBLIC.parent/'european_fixed_floor_excess_v1'
PRIVATE=parent.PRIVATE.parent/'european_fixed_floor_excess_v1'
CONFIG='configs/m3w_european_fixed_floor_excess_v1.json'
FILES=[CONFIG,'scripts/run_m3w_european_fixed_floor_excess.py','src/world_model/m3w_fixed_floor_excess.py',
       'tests/test_m3w_fixed_floor_excess.py',str(PUBLIC.relative_to(ROOT)/'protocol.md')]
artifact,digest,immutable_json=parent.artifact,parent.digest,parent.immutable_json


def beat(state,**kw):
    r=dict(state=state,pid=os.getpid(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw)
    inter.json_write(PRIVATE/'heartbeat.json',r)
    with (PRIVATE/'events.jsonl').open('a') as f: f.write(json.dumps(r)+'\n')
    print(json.dumps(r),flush=True)


def load(create=False):
    cfg=json.loads((ROOT/CONFIG).read_text()); sp=parent.PUBLIC/'verification.json'
    assert digest(sp)==cfg['parent_seal_sha256']; seal=json.loads(sp.read_text())
    for p,h in seal['source_bindings'].items(): assert digest(ROOT/p)==h,p
    for p,h in seal['artifacts'].items(): assert digest(parent.PUBLIC/p)==h,p
    pc,data,jobs,oid,pid,pbound=parent.load()
    assert cfg['head_training']==pc['head_training'] and cfg['new_heads']==cfg['groups']==108
    assert cfg['risk_budget']==.02 and cfg['new_forecasters']==0
    assert not any(cfg[k] for k in ('threshold_search','independent_roles_read','deployment_changed','stage5c_executed','smc_enabled'))
    bound=dict(parent_seal=artifact(sp),bindings={p:digest(ROOT/p) for p in FILES},
        rosters=oid['rosters'],source_rows=len(data['sites']),new_risk_heads=108,
        matched_controls='cached_verified',changed_factor='fixed_floor_signed_excess_loss_only',independent_roles_read=False)
    if create: immutable_json(PUBLIC/'registration.json',bound)
    else:
        assert json.loads((PUBLIC/'registration.json').read_text())==bound
        inter.committed(PUBLIC/'registration.json')
    return cfg,data,jobs,oid,pid,pbound,bound


def control(c,data,pair,pid,pbound,bound):
    name,fit,held,y,pr,old,hid=parent.prepare(c,data,pair,pid,pbound)
    path=parent.PRIVATE/'heads'/(name+'_mse')/'complete.json'
    rec=parent.done(path,hid)
    state=torch.load(ROOT/rec['artifacts']['checkpoint']['path'],map_location='cpu',weights_only=False)
    with np.load(ROOT/rec['artifacts']['scores']['path'],allow_pickle=False) as z:
        np.testing.assert_array_equal(z['ids'],c['ids'][held]); mse=z['scores'].copy()
    ident=dict(experiment=bound,control=artifact(path),roles_and_targets=hid)
    return name,fit,held,y,pr,old,state,mse,ident


def actions(c,data,held,old,mse,new):
    eligible=c['moving'][held]&old['support']&(old['scores'][:,5]>old['scores'][:,6])
    take=eligible&(new[:,1]<=.02*new[:,0])&(new[:,3]<=.02*new[:,2])
    ids=c['ids'][held]; score=api.signed(mse.astype(float)).max(1)
    matched=parent.api.match_counts(take,eligible,score,parent.query_keys(data,ids),data['frames'][ids],ids)
    return dict(ids=ids,eligible=eligible,excess=take,mse=eligible&(mse[:,1]<=.02*mse[:,0])&(mse[:,3]<=.02*mse[:,2]),
                ridge=old['floor_safe'],mse_matched_count=matched)


def done(path,identity):
    r=json.loads(path.read_text()); assert r['identity']==identity and r['fit']['complete'] and r['fit']['step']==2000
    for a in r['artifacts'].values(): assert artifact(ROOT/a['path'])==a
    return r


def training_quality(model,mse_state,c,fit,y,pr):
    old=api.initialize(pr,mse_state['settings']['width'],mse_state['seed'],mse_state['mean_envelope'])
    old.load_state_dict(mse_state['model']); result={}; known=pr['known']
    for arm,m in [('mse',old),('excess',model)]:
        p=api.predict(m,c['x'][fit],c['env'][fit],pr).astype(float)
        squared=((api.signed(p[known])-api.signed(y[known]))/pr['cost_scale'])**2
        result[arm]=dict(zip(('all_normalized_excess_MSE','easy_normalized_excess_MSE'),
                            map(float,(squared*pr['weights'][known,None]).sum(0))))
    return result


def train(cfg,data,jobs,oid,pid,pbound,bound,*,resume=False,pilot=False,replay=False):
    refs=[]; decisions=[]
    for c in floor_api.contexts(data,jobs,oid):
        for pair in range(6):
            name,fit,held,y,pr,old,ms,mse,identity=control(c,data,pair,pid,pbound,bound)
            home=PRIVATE/'heads'/name; path=home/'complete.json'
            if path.exists():
                r=done(path,identity)
                if not resume and not replay: raise ValueError('Use --resume for existing results')
            else:
                if replay: raise FileNotFoundError(path)
                if shutil.disk_usage(PRIVATE).free<10*2**30: raise OSError('Keep10GiB; retain completed checkpoints')
                model,info=api.fit(c['x'][fit],y,data['sites'][c['ids'][fit]],c['env'][fit],pr,
                    seed=ms['seed'],settings=cfg['head_training'],identity=identity,directory=home,
                    heartbeat=lambda **kw:beat(head=name,**kw),resume=resume,
                    stop_at=cfg['pilot_updates'] if pilot else None)
                if pilot:
                    immutable_json(PRIVATE/'pilot.json',dict(fit=info,checkpoint=artifact(home/'checkpoint.pt'))); return
                p=api.predict(model,c['x'][held],c['env'][held],pr)
                floor_api.parent.write_arrays(home/'scores.npz',dict(ids=c['ids'][held],scores=p),False)
                q=training_quality(model,ms,c,fit,y,pr)
                r=dict(identity=identity,fit=info,training_quality=q,held_labels_read=False,
                       artifacts=dict(checkpoint=artifact(home/'checkpoint.pt'),scores=artifact(home/'scores.npz')))
                immutable_json(path,r)
            state=torch.load(home/'checkpoint.pt',map_location='cpu',weights_only=False); api.assert_matched(state,ms)
            with np.load(home/'scores.npz',allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'],c['ids'][held]); p=z['scores'].copy()
            if replay:
                model=api.initialize(pr,cfg['head_training']['width'],ms['seed'],state['mean_envelope']); model.load_state_dict(state['model'])
                np.testing.assert_array_equal(api.predict(model,c['x'][held],c['env'][held],pr),p)
                assert training_quality(model,ms,c,fit,y,pr)==r['training_quality']
            a=actions(c,data,held,old,mse,p)
            with np.load(parent.PRIVATE/'decisions'/(name+'.npz'),allow_pickle=False) as z:
                for k in ('ids','eligible','mse','ridge'): np.testing.assert_array_equal(a[k],z[k])
                a['tail4']=z['tail4'].copy()
            ap=PRIVATE/'decisions'/(name+'.npz'); floor_api.parent.write_arrays(ap,a,replay)
            rec=ap.with_suffix('.json'); immutable_json(rec,dict(identity=identity,arrays=artifact(ap)))
            refs.append(artifact(path)); decisions.append(artifact(rec)); beat('head_replayed' if replay else 'head_complete',head=name,count=len(refs))
    assert len(refs)==108
    immutable_json(PUBLIC/'decision_freeze.json',dict(identity=bound,heads=refs,decisions=decisions,
        new_heads=108,updates=216000,matched_sampler_pairs=108,held_labels_used_for_selection=False))
    if replay: immutable_json(PUBLIC/'prediction_replay.json',dict(exact=True,new_heads=108,matched_controls=108,decisions=108,fitting_score_replay=True))


def causal_replay(data,jobs,oid,pid,pbound,bound):
    causal={k:data[k] for k in ('sites','history','origin','geometry','recordings','frames')}
    c=next(floor_api.contexts(causal,jobs,oid))
    name,_,held,_,old,_=parent.inherited(c,causal,0,pid)
    state=torch.load(PRIVATE/'heads'/name/'checkpoint.pt',map_location='cpu',weights_only=False)
    assert state['identity']['experiment']==bound
    model=api.initialize(state['preprocess'],state['settings']['width'],state['seed'],state['mean_envelope'])
    model.load_state_dict(state['model']); p=api.predict(model,c['x'][held],c['env'][held],state['preprocess'])
    with np.load(PRIVATE/'heads'/name/'scores.npz',allow_pickle=False) as z: np.testing.assert_array_equal(p,z['scores'])
    with np.load(parent.PRIVATE/'heads'/(name+'_mse')/'scores.npz',allow_pickle=False) as z: mse=z['scores'].copy()
    a=actions(c,causal,held,old,mse,p)
    with np.load(PRIVATE/'decisions'/(name+'.npz'),allow_pickle=False) as z:
        for k,v in a.items(): np.testing.assert_array_equal(v,z[k])
    immutable_json(PUBLIC/'causal_replay.json',dict(exact=True,new_heads=1,actions=1,future_fields_removed=True))


def evaluate(cfg,data,jobs,oid,pid,pbound,bound,replay=False):
    inter.committed(PUBLIC/'decision_freeze.json'); freeze=json.loads((PUBLIC/'decision_freeze.json').read_text())
    assert freeze['identity']==bound
    for r in freeze['heads']+freeze['decisions']: assert artifact(ROOT/r['path'])==r
    rows=[]; quality=[]
    for c in floor_api.contexts(data,jobs,oid):
        cv,cf,(floor,ff),(neural,nf)=floor_api.costs(c,data,np.arange(len(c['ids'])))
        for pair in range(6):
            name,_,held,_,pr,old,ms,mse,identity=control(c,data,pair,pid,pbound,bound)
            rec=done(PRIVATE/'heads'/name/'complete.json',identity)
            with np.load(PRIVATE/'heads'/name/'scores.npz',allow_pickle=False) as z: p=z['scores'].copy()
            with np.load(PRIVATE/'decisions'/(name+'.npz'),allow_pickle=False) as z: a={k:z[k].copy() for k in z.files}
            regenerated=actions(c,data,held,old,mse,p)
            for k,v in regenerated.items(): np.testing.assert_array_equal(v,a[k])
            for site in c['pairs'][pair][1]:
                local=data['sites'][a['ids']]==site; ix=held[local]; known=np.isfinite(cv[ix])
                base=dict(site=site,seed=c['job']['old_identity']['seed'],group=name)
                for policy in ('floor','ridge','mse','tail4','excess','mse_matched_count'):
                    take=np.zeros(len(ix),bool) if policy=='floor' else a[policy][local]
                    m=floor_api.metric(cv[ix],floor[ix],neural[ix],cf[ix],ff[ix],nf[ix],
                        np.where(take,neural[ix],floor[ix]),np.where(take,nf[ix],ff[ix]),take,
                        data['valid'][c['ids'][ix]],c['job']['design']['easy_cut'],c['job']['design']['hard_cut'])
                    rows.append(dict(**base,policy=policy,metric=m))
                y=floor_api.api.targets(cv[ix],floor[ix],neural[ix],c['job']['design']['easy_cut'])[:,[2,1,3,4]]
                target=api.signed(y)/pr['cost_scale']
                for arm,pred in [('mse',mse[local]),('excess',p[local])]:
                    q=api.signed(pred.astype(float))/pr['cost_scale']; take=a[arm][local]&known
                    metric={key:float(np.mean((q[known,i]-target[known,i])**2)) for i,key in enumerate(('all_normalized_excess_MSE','easy_normalized_excess_MSE'))}
                    for i,kind in enumerate(('all','easy')):
                        metric[kind+'_fit_MSE']=rec['training_quality'][arm][kind+'_normalized_excess_MSE']
                        metric[kind+'_selected_predicted_excess']=float(q[take,i].mean()) if take.any() else None
                        metric[kind+'_selected_observed_excess']=float(target[take,i].mean()) if take.any() else None
                    quality.append(dict(**base,policy=arm,metric=metric))
        beat('held_scored',group=c['name'])
    sites=sorted(set(data['sites']))
    def grouped(rr):
        return {p:{k:inter.paired_localities([r for r in rr if r['policy']==p],sites,k,cfg['bootstrap_resamples'],cfg['bootstrap_seed'])
                   for k in rr[0]['metric']} for p in sorted(set(r['policy'] for r in rr))}
    index={(r['group'],r['site'],r['policy']):r['metric'] for r in rows}; contrasts=[]
    for r in rows:
        if r['policy']!='excess': continue
        t=r['metric']
        for control_name in ('mse','mse_matched_count','ridge','tail4'):
            m=index[(r['group'],r['site'],control_name)]
            risk=None if m['selected_positive_harm_ratio'] is None or t['selected_positive_harm_ratio'] is None else 100*(m['selected_positive_harm_ratio']-t['selected_positive_harm_ratio'])
            contrasts.append(dict(site=r['site'],seed=r['seed'],policy=control_name,metric=dict(
                positive_harm_reduction_pp=risk,ADE_gain_percent=100*(1-t['error_sum']/m['error_sum']) if m['error_sum']>0 else None,
                intervention_difference_pp=100*(t['intervention_rate']-m['intervention_rate']))))
    summary=grouped(rows); paired=grouped(contrasts); primary=paired['mse_matched_count']['positive_harm_reduction_pp']
    chosen=[r['metric'] for r in rows if r['policy']=='excess']; s=summary['excess']
    gates=dict(primary_equal_count_harm_reduction=primary['ci95'] is not None and primary['ci95'][0]>0,
        equal_count_ADE_advantage=paired['mse_matched_count']['ADE_gain_percent']['ci95'][0]>0,
        floor_ADE_advantage=s['all_gain_floor']['ci95'][0]>0,
        nonzero_each_locality=all(v is not None and v>0 for v in s['intervention_rate']['by_site'].values()),
        every_view_easy_preserved=all(m['easy_gain_CV'] is not None and m['easy_gain_CV']>=-2 for m in chosen),
        no_zero_CV_harm=all(m['zero_CV_harmed']==0 for m in chosen),
        every_view_defined_risk_within_budget=all(m['selected_positive_harm_ratio'] is not None and m['selected_positive_harm_ratio']<=.02 for m in chosen))
    gates['exploratory_joint_screen_pass']=all(gates.values())
    gates.update(independent_confirmation=False,calibration_certificate=False,deployment_changed=False,stage5c_executed=False,smc_enabled=False)
    doc=dict(identity=bound,result_source='fresh_signed_excess_training_readout_cached_verified_controls',
        summary=summary,paired=paired,quality=grouped(quality),gates=gates,
        by_seed={str(s):grouped([r for r in rows if r['seed']==s]) for s in cfg['seeds']},
        worst_views={p:dict(worst_easy_gain_CV=min(r['metric']['easy_gain_CV'] for r in rows if r['policy']==p and r['metric']['easy_gain_CV'] is not None),
            risk_violating_views=sum(r['metric']['selected_positive_harm_ratio'] is not None and r['metric']['selected_positive_harm_ratio']>.02 for r in rows if r['policy']==p),
            undefined_risk_views=sum(r['metric']['selected_positive_harm_ratio'] is None for r in rows if r['policy']==p)) for p in summary},
        new_heads=108,updates=216000,independent_roles_read=False)
    immutable_json(PRIVATE/'details.json',dict(rows=rows,quality=quality,contrasts=contrasts))
    immutable_json(PUBLIC/'summary.json',doc); immutable_json(PUBLIC/'gates.json',gates)
    if replay: immutable_json(PUBLIC/'evaluation_replay.json',dict(exact=True,summary=artifact(PUBLIC/'summary.json'),details=artifact(PRIVATE/'details.json')))


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--phase',required=True,
        choices=['register','pilot','train','replay','evaluate','replay_evaluate','causal_replay']); p.add_argument('--resume',action='store_true'); args=p.parse_args()
    PUBLIC.mkdir(parents=True,exist_ok=True); PRIVATE.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    with (PRIVATE/'run.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); beat('phase_started',phase=args.phase)
        cfg,data,jobs,oid,pid,pbound,bound=load(args.phase=='register')
        if args.phase in ('pilot','train','replay'):
            train(cfg,data,jobs,oid,pid,pbound,bound,resume=args.resume,pilot=args.phase=='pilot',replay=args.phase=='replay')
        elif args.phase in ('evaluate','replay_evaluate'): evaluate(cfg,data,jobs,oid,pid,pbound,bound,replay=args.phase=='replay_evaluate')
        elif args.phase=='causal_replay': causal_replay(data,jobs,oid,pid,pbound,bound)
        beat('phase_complete',phase=args.phase)


if __name__=='__main__': main()
