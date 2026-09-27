"""Matched bounded neural risk heads and same-query coverage controls."""
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
from scripts import run_m3w_european_fixed_floor_probe as parent
from scripts.replay_m3w_dimensionless_training import exact
from src.world_model import m3w_fixed_floor_tail as api
import numpy as np
torch,inter=parent.torch,parent.inter
PUBLIC=parent.PUBLIC.parent/'european_fixed_floor_tail_v1'
PRIVATE=parent.PRIVATE.parent/'european_fixed_floor_tail_v1'
CONFIG='configs/m3w_european_fixed_floor_tail_v1.json'
FILES=[CONFIG,'scripts/run_m3w_european_fixed_floor_tail.py','src/world_model/m3w_fixed_floor_tail.py',
       'tests/test_m3w_fixed_floor_tail.py',str(PUBLIC.relative_to(ROOT)/'protocol.md')]
artifact,digest,immutable_json=parent.artifact,parent.digest,parent.immutable_json


def beat(state,**kw):
    row=dict(state=state,pid=os.getpid(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw)
    inter.json_write(PRIVATE/'heartbeat.json',row)
    with (PRIVATE/'events.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)


def load(create=False):
    cfg=json.loads((ROOT/CONFIG).read_text()); sp=parent.PUBLIC/'verification.json'
    assert digest(sp)==cfg['parent_seal_sha256']; seal=json.loads(sp.read_text())
    for p,h in seal['source_bindings'].items(): assert digest(ROOT/p)==h,p
    for p,h in seal['artifacts'].items(): assert digest(parent.PUBLIC/p)==h,p
    _,data,jobs,old_identity,parent_identity=parent.load()
    assert cfg['head_count']==216 and cfg['risk_budget']==.02 and cfg['groups']==108
    assert cfg['arms']==['mse','tail4'] and cfg['positive_harm_tail_quantile']==.9 and cfg['tail_multiplier']==4
    assert not any(cfg[k] for k in ('threshold_search','new_forecaster_training','independent_roles_read',
                                   'deployment_changed','stage5c_executed','smc_enabled'))
    bound=dict(parent_seal=artifact(sp),bindings={p:digest(ROOT/p) for p in FILES},
        rosters=old_identity['rosters'],source_rows=len(data['sites']),changed_factor='risk_head_loss_weighting_only',
        upstream_fitting_roles='cached_verified',independent_roles_read=False)
    if create: immutable_json(PUBLIC/'registration.json',bound)
    else:
        assert json.loads((PUBLIC/'registration.json').read_text())==bound
        inter.committed(PUBLIC/'registration.json')
    return cfg,data,jobs,old_identity,parent_identity,bound


def inherited(c,data,pair,parent_identity):
    fit_sites,held_sites=c['pairs'][pair]
    parent.api.assert_roles(c['producer_sites'],c['controller_sites'],fit_sites,held_sites)
    sites=data['sites'][c['ids']]; fit=np.flatnonzero(np.isin(sites,fit_sites)); held=np.flatnonzero(np.isin(sites,held_sites))
    name=c['name']+f'_pair{pair}'; home=parent.PRIVATE/'fits'/name
    record=json.loads((home/'complete.json').read_text()); assert record['identity']==parent_identity
    for ref in record['artifacts'].values(): assert artifact(ROOT/ref['path'])==ref
    state=torch.load(home/'ridge.pt',map_location='cpu',weights_only=False); assert state['identity']==parent_identity
    with np.load(home/'scores.npz',allow_pickle=False) as z: old={k:z[k].copy() for k in z.files}
    np.testing.assert_array_equal(old['ids'],c['ids'][held])
    return name,fit,held,state['model'],old,artifact(home/'complete.json')


def prepare(c,data,pair,parent_identity,bound):
    name,fit,held,ridge,old,ref=inherited(c,data,pair,parent_identity)
    cv,_,(floor,_),(neural,_)=parent.costs(c,data,fit)
    y=parent.api.targets(cv,floor,neural,c['job']['design']['easy_cut'])[:,[2,1,3,4]]
    known=np.isfinite(cv); sites=data['sites'][c['ids'][fit]]; weights=np.zeros(len(fit))
    for s in sorted(set(sites)):
        ix=(sites==s)&known; weights[ix]=.5/ix.sum()
    assert int(known.sum())==ridge['known_rows'] and sorted(set(sites))==ridge['training_sites']
    pr=dict(mean=ridge['mean'],std=ridge['std'],cost_scale=ridge['scale'],clip=ridge['clip'],
        known=known,weights=weights,constant=(y[known]*weights[known,None]).sum(0),training_sites=ridge['training_sites'])
    identity=dict(experiment=bound,parent_fit=ref,group=name,training_sites=pr['training_sites'],
        held_sites=c['pairs'][pair][1],producer_sites=c['producer_sites'],controller_sites=c['controller_sites'],
        fit_ids_sha256=inter.array_hash(c['ids'][fit]),features_sha256=inter.array_hash(c['x'][fit]),
        train_labels_sha256=inter.array_hash(y),held_ids_sha256=inter.array_hash(c['ids'][held]),
        seed=c['job']['old_identity']['seed']+1009*c['controller']+109*pair)
    return name,fit,held,y,pr,old,identity


def done(path,identity):
    d=json.loads(path.read_text()); assert d['identity']==identity and d['fit']['complete'] and d['fit']['step']==2000
    for ref in d['artifacts'].values(): assert artifact(ROOT/ref['path'])==ref
    return d


def matched(a,b):
    for k in ('settings','preprocess','seed','step','draws','sampler_rng','torch_rng','initial_model','tail','mean_envelope'):
        exact(a[k],b[k])
    assert a['arm']=='mse' and b['arm']=='tail4'


def query_keys(data,ids):
    return np.char.add(np.char.add(data['sites'][ids].astype(str),'|'),data['recordings'][ids].astype(str))


def action_arrays(c,data,held,old,scores):
    utility=old['scores'][:,5]-old['scores'][:,6]
    eligible=c['moving'][held]&old['support']&(utility>0)
    arrays=dict(ids=c['ids'][held],ridge=old['floor_safe'],eligible=eligible)
    for arm,p in scores.items():
        arrays[arm]=eligible&(p[:,1]<=.02*p[:,0])&(p[:,3]<=.02*p[:,2])
    p=scores['mse'].astype(float)
    score=np.maximum(p[:,1]-.02*p[:,0],p[:,3]-.02*p[:,2])
    ids=arrays['ids']
    arrays['mse_matched_count']=api.match_counts(arrays['tail4'],eligible,score,query_keys(data,ids),data['frames'][ids],ids)
    return arrays


def train(cfg,data,jobs,old_identity,parent_identity,bound,*,resume=False,pilot=False,replay=False):
    refs=[]; decisions=[]; count=0
    for c in parent.contexts(data,jobs,old_identity):
        for pair in range(6):
            name,fit,held,y,pr,old,identity=prepare(c,data,pair,parent_identity,bound)
            scores={}; states={}
            for arm in cfg['arms']:
                home=PRIVATE/'heads'/(name+'_'+arm); path=home/'complete.json'
                if path.exists():
                    d=done(path,identity)
                    if not resume and not replay: raise ValueError('Completed heads require --resume')
                else:
                    if replay: raise FileNotFoundError(path)
                    if shutil.disk_usage(PRIVATE).free<10*2**30: raise OSError('Keep10GiB; retain completed checkpoints')
                    model,fit_info=api.fit(c['x'][fit],y,data['sites'][c['ids'][fit]],c['env'][fit],pr,
                        seed=identity['seed'],arm=arm,settings=cfg['head_training'],identity=identity,directory=home,
                        heartbeat=lambda **kw:beat(head=name+'_'+arm,**kw),resume=resume,
                        stop_at=cfg['pilot_updates'] if pilot else None)
                    if pilot:
                        immutable_json(PRIVATE/'pilot.json',dict(fit=fit_info,checkpoint=artifact(home/'checkpoint.pt')))
                        return
                    p=api.predict(model,c['x'][held],c['env'][held],pr)
                    parent.parent.write_arrays(home/'scores.npz',dict(ids=c['ids'][held],scores=p),False)
                    assert fit_info['unknown_rows_sampled']==0
                    d=dict(identity=identity,arm=arm,fit=fit_info,held_labels_read=False,
                        artifacts=dict(checkpoint=artifact(home/'checkpoint.pt'),scores=artifact(home/'scores.npz')))
                    immutable_json(path,d)
                state=torch.load(home/'checkpoint.pt',map_location='cpu',weights_only=False); states[arm]=state
                with np.load(home/'scores.npz',allow_pickle=False) as z:
                    np.testing.assert_array_equal(z['ids'],c['ids'][held]); scores[arm]=z['scores'].copy()
                if replay:
                    model=api.initialize(pr,cfg['head_training']['width'],identity['seed'],state['mean_envelope'])
                    model.load_state_dict(state['model'])
                    np.testing.assert_array_equal(api.predict(model,c['x'][held],c['env'][held],pr),scores[arm])
                refs.append(artifact(path)); count+=1; beat('head_replayed' if replay else 'head_complete',head=name+'_'+arm,complete=count)
            matched(states['mse'],states['tail4'])
            arrays=action_arrays(c,data,held,old,scores)
            path=PRIVATE/'decisions'/(name+'.npz')
            parent.parent.write_arrays(path,arrays,replay)
            rec=path.with_suffix('.json'); immutable_json(rec,dict(identity=identity,arrays=artifact(path),counts_matched_current_query=True))
            decisions.append(artifact(rec))
    assert count==216 and len(decisions)==108
    immutable_json(PUBLIC/'decision_freeze.json',dict(identity=bound,heads=refs,decisions=decisions,
        heads_count=count,updates=count*cfg['head_training']['steps'],matched_sampler_pairs=108,held_labels_read_for_fits=False))
    if replay: immutable_json(PUBLIC/'prediction_replay.json',dict(all_exact=True,heads=216,decisions=108,matched_sampler_pairs=108))


def verify_causal_first(data,jobs,old_identity,parent_identity,bound):
    causal={k:data[k] for k in ('sites','history','origin','geometry','recordings','frames')}
    c=next(parent.contexts(causal,jobs,old_identity))
    name,_,held,ridge,old,_=inherited(c,causal,0,parent_identity); scores={}
    for arm in ('mse','tail4'):
        home=PRIVATE/'heads'/(name+'_'+arm)
        s=torch.load(home/'checkpoint.pt',map_location='cpu',weights_only=False)
        assert s['identity']['experiment']==bound
        model=api.initialize(s['preprocess'],s['settings']['width'],s['seed'],s['mean_envelope'])
        model.load_state_dict(s['model']); p=api.predict(model,c['x'][held],c['env'][held],s['preprocess'])
        with np.load(home/'scores.npz',allow_pickle=False) as z: np.testing.assert_array_equal(p,z['scores'])
        scores[arm]=p
    parent.parent.write_arrays(PRIVATE/'decisions'/(name+'.npz'),action_arrays(c,causal,held,old,scores),True)
    immutable_json(PUBLIC/'causal_replay.json',dict(exact=True,heads=2,decisions=1,future_fields_removed=True))


def evaluate(cfg,data,jobs,old_identity,parent_identity,bound,replay=False):
    inter.committed(PUBLIC/'decision_freeze.json'); freeze=json.loads((PUBLIC/'decision_freeze.json').read_text())
    assert freeze['identity']==bound
    for r in freeze['heads']+freeze['decisions']: assert artifact(ROOT/r['path'])==r
    rows=[]; quality=[]; matched_queries=0
    for c in parent.contexts(data,jobs,old_identity):
        cv,cf,(floor,ff),(neural,nf)=parent.costs(c,data,np.arange(len(c['ids'])))
        for pair in range(6):
            name,_,held,_,old,_=inherited(c,data,pair,parent_identity)
            path=PRIVATE/'decisions'/(name+'.json'); rec=json.loads(path.read_text()); assert rec['identity']['experiment']==bound
            assert artifact(ROOT/rec['arrays']['path'])==rec['arrays']
            with np.load(ROOT/rec['arrays']['path'],allow_pickle=False) as z: arrays={k:z[k].copy() for k in z.files}
            np.testing.assert_array_equal(arrays['ids'],c['ids'][held]); scores={}
            for arm in cfg['arms']:
                with np.load(PRIVATE/'heads'/(name+'_'+arm)/'scores.npz',allow_pickle=False) as z: scores[arm]=z['scores'].copy()
            regenerated=action_arrays(c,data,held,old,scores)
            for k,v in arrays.items(): np.testing.assert_array_equal(v,regenerated[k])
            matched_queries+=len(set(zip(query_keys(data,arrays['ids']).tolist(),data['frames'][arrays['ids']].tolist())))
            for site in c['pairs'][pair][1]:
                local=data['sites'][arrays['ids']]==site; ix=held[local]
                b,d,n=cv[ix],floor[ix],neural[ix]; bf,df,nfe=cf[ix],ff[ix],nf[ix]
                base=dict(site=site,seed=c['job']['old_identity']['seed'],group=name)
                policies=dict(floor=np.zeros(len(ix),bool),ridge=arrays['ridge'][local],
                    mse=arrays['mse'][local],tail4=arrays['tail4'][local],mse_matched_count=arrays['mse_matched_count'][local])
                for policy,take in policies.items():
                    m=parent.metric(b,d,n,bf,df,nfe,np.where(take,n,d),np.where(take,nfe,df),take,
                                    data['valid'][c['ids'][ix]],c['job']['design']['easy_cut'],c['job']['design']['hard_cut'])
                    rows.append(dict(**base,policy=policy,metric=m))
                y=parent.api.targets(b,d,n,c['job']['design']['easy_cut'])[:,[2,1,3,4]]
                known=np.isfinite(b)
                for arm in cfg['arms']:
                    p=scores[arm][local].astype(float); take=arrays[arm][local]&known
                    def ratio(a,b): return float(a/b) if b>0 else None
                    m={key+'_MSE':float(np.mean((p[known,col]-y[known,col])**2)) for col,key in enumerate(('reference','harm','easy_reference','easy_harm'))}
                    m.update(predicted_selected_harm_ratio=ratio(p[take,1].sum(),p[take,0].sum()),
                        actual_selected_harm_ratio=ratio(y[take,1].sum(),y[take,0].sum()),
                        predicted_over_actual_reference=ratio(p[take,0].sum(),y[take,0].sum()),
                        zero_predicted_harm_fraction=ratio((p[take,1]==0).sum(),take.sum()))
                    quality.append(dict(**base,policy=arm,metric=m))
        beat('held_scored',group=c['name'])
    sites=sorted(set(data['sites']))
    def aggregate(rr):
        return {k:inter.paired_localities(rr,sites,k,cfg['bootstrap_resamples'],cfg['bootstrap_seed']) for k in rr[0]['metric']}
    def by_policy(rr): return {p:aggregate([r for r in rr if r['policy']==p]) for p in sorted(set(r['policy'] for r in rr))}
    index={(r['group'],r['site'],r['policy']):r['metric'] for r in rows}
    contrasts=[]
    for r in rows:
        if r['policy']!='tail4': continue
        t=r['metric']
        for control in ('mse','mse_matched_count','ridge'):
            m=index[(r['group'],r['site'],control)]
            risk=None if m['selected_positive_harm_ratio'] is None or t['selected_positive_harm_ratio'] is None else 100*(m['selected_positive_harm_ratio']-t['selected_positive_harm_ratio'])
            contrasts.append(dict(site=r['site'],seed=r['seed'],policy=control,metric=dict(
                positive_harm_reduction_pp=risk,ADE_gain_percent=100*(1-t['error_sum']/m['error_sum']) if m['error_sum']>0 else None,
                intervention_difference_pp=100*(t['intervention_rate']-m['intervention_rate']))))
    summary=by_policy(rows); paired=by_policy(contrasts); primary=paired['mse_matched_count']['positive_harm_reduction_pp']
    t=summary['tail4']; chosen=[r['metric'] for r in rows if r['policy']=='tail4']
    gates=dict(primary_equal_count_harm_reduction=primary['ci95'] is not None and primary['ci95'][0]>0,
        equal_count_ADE_advantage=paired['mse_matched_count']['ADE_gain_percent']['ci95'][0]>0,
        floor_ADE_advantage=t['all_gain_floor']['ci95'][0]>0,
        nonzero_each_locality=all(v is not None and v>0 for v in t['intervention_rate']['by_site'].values()),
        every_view_easy_preserved=all(v['easy_gain_CV'] is not None and v['easy_gain_CV']>=-2 for v in chosen),
        no_zero_CV_harm=all(v['zero_CV_harmed']==0 for v in chosen),
        every_view_defined_risk_within_budget=all(v['selected_positive_harm_ratio'] is not None and v['selected_positive_harm_ratio']<=.02 for v in chosen))
    gates['exploratory_joint_screen_pass']=all(gates.values())
    gates.update(calibration_certificate=False,independent_confirmation=False,deployment_changed=False,stage5c_executed=False,smc_enabled=False)
    doc=dict(identity=bound,result_source='fresh_risk_head_training_readout_cached_verified_fixed_forecasters_floor_utility',
        summary=summary,paired=paired,quality=by_policy(quality),gates=gates,matched_current_query_views=matched_queries,
        by_seed={str(s):by_policy([r for r in rows if r['seed']==s]) for s in cfg['seeds']},
        worst_views={p:dict(worst_easy_gain_CV=min(r['metric']['easy_gain_CV'] for r in rows if r['policy']==p and r['metric']['easy_gain_CV'] is not None),
            risk_violating_views=sum(r['metric']['selected_positive_harm_ratio'] is not None and r['metric']['selected_positive_harm_ratio']>.02 for r in rows if r['policy']==p),
            undefined_risk_views=sum(r['metric']['selected_positive_harm_ratio'] is None for r in rows if r['policy']==p),
            zero_CV_harmed_views=sum(r['metric']['zero_CV_harmed']>0 for r in rows if r['policy']==p)) for p in summary},
        new_risk_heads=216,updates=432000,new_forecasters=0,independent_roles_read=False)
    immutable_json(PRIVATE/'details.json',dict(rows=rows,quality=quality,contrasts=contrasts))
    immutable_json(PUBLIC/'summary.json',doc); immutable_json(PUBLIC/'gates.json',gates)
    if replay: immutable_json(PUBLIC/'evaluation_replay.json',dict(exact=True,summary=artifact(PUBLIC/'summary.json'),details=artifact(PRIVATE/'details.json')))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',required=True,choices=['register','pilot','train','replay','evaluate','replay_evaluate','causal_replay'])
    p.add_argument('--resume',action='store_true'); args=p.parse_args()
    PUBLIC.mkdir(parents=True,exist_ok=True); PRIVATE.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    with (PRIVATE/'run.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); beat('phase_started',phase=args.phase)
        cfg,data,jobs,old_identity,parent_identity,bound=load(args.phase=='register')
        if args.phase in ('pilot','train','replay'):
            train(cfg,data,jobs,old_identity,parent_identity,bound,resume=args.resume,pilot=args.phase=='pilot',replay=args.phase=='replay')
        elif args.phase in ('evaluate','replay_evaluate'):
            evaluate(cfg,data,jobs,old_identity,parent_identity,bound,replay=args.phase=='replay_evaluate')
        elif args.phase=='causal_replay': verify_causal_first(data,jobs,old_identity,parent_identity,bound)
        beat('phase_complete',phase=args.phase)


if __name__=='__main__': main()
