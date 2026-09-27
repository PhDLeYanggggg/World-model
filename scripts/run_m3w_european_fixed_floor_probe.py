"""Frozen-floor 4/4/2/2 incremental learning with pre-readout decision freeze."""
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
from scripts import run_m3w_european_score_support_calibration as parent
from src.world_model import m3w_fixed_floor_probe as api
from src.world_model.m3w_floor_relative import matched_features
from src.world_model.m3w_score_support_calibration import decide, support_limit
import numpy as np
from sklearn.metrics import roc_auc_score
torch,inter=parent.torch,parent.inter
PUBLIC=parent.PUBLIC.parent/'european_fixed_floor_probe_v1'
PRIVATE=parent.PRIVATE.parent/'european_fixed_floor_probe_v1'
CONFIG='configs/m3w_european_fixed_floor_probe_v1.json'
FILES=[CONFIG,'scripts/run_m3w_european_fixed_floor_probe.py',
    'src/world_model/m3w_fixed_floor_probe.py','tests/test_m3w_fixed_floor_probe.py',
    str(PUBLIC.relative_to(ROOT)/'protocol.md')]
artifact,digest,immutable_json=parent.artifact,parent.digest,parent.immutable_json


def beat(state,**kw):
    r=dict(state=state,pid=os.getpid(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw)
    inter.json_write(PRIVATE/'heartbeat.json',r)
    with (PRIVATE/'events.jsonl').open('a') as f: f.write(json.dumps(r)+'\n')
    print(json.dumps(r),flush=True)


def load(create=False):
    cfg=json.loads((ROOT/CONFIG).read_text()); sp=parent.PUBLIC/'verification.json'
    assert digest(sp)==cfg['parent_seal_sha256']
    seal=json.loads(sp.read_text())
    for p,h in seal['source_bindings'].items(): assert digest(ROOT/p)==h,p
    for p,h in seal['artifacts'].items(): assert digest(parent.PUBLIC/p)==h,p
    _,data,jobs,_,_,identity=parent.load()
    assert cfg['linear_heads']==216 and cfg['groups']==108 and cfg['risk_budget']==.02
    assert not any(cfg[k] for k in ('independent_roles_read','new_forecaster_training','threshold_search',
                                   'deployment_changed','stage5c_executed','smc_enabled'))
    bound=dict(parent_seal=artifact(sp),bindings={p:digest(ROOT/p) for p in FILES},
        rosters=identity['rosters'],source_rows=len(data['sites']),independent_roles_read=False,
        forecaster_and_floor='cached_verified')
    if create: immutable_json(PUBLIC/'registration.json',bound)
    else:
        assert json.loads((PUBLIC/'registration.json').read_text())==bound
        inter.committed(PUBLIC/'registration.json')
    return cfg,data,jobs,identity,bound


def contexts(data,jobs,identity):
    for c in parent.contexts(data,jobs,identity['rosters']):
        if c['arm']!='dimensionless': continue
        z=parent.read_scores(c,identity)
        d=dict(c,name=c['name'].replace('_dimensionless','_damped'))
        dz=parent.read_scores(d,identity)
        mask=decide(dz['excess'],dz['guard'],dz['supported'],0.)
        nr=decide(z['excess'],z['guard'],z['supported'],0.)
        ids=c['ids']; cv=inter.baseline_numpy(data['history'][ids],1)-data['origin'][ids,None]
        damping=inter.baseline_numpy(data['history'][ids],3)-data['origin'][ids,None]
        floor=np.where(mask[:,None,None],damping,cv)
        x,env=matched_features(data['geometry'][ids],cv,floor,c['prediction'],mask)
        yield dict(c,x=x,env=env,floor=floor,floor_mask=mask,neural_mask=nr,moving=z['moving'])


def costs(c,data,pos):
    ids=c['ids'][pos]
    floor=inter.native_errors(c['floor'][pos].astype(float)+data['origin'][ids,None],
        data['target_eval'][ids],data['valid'][ids],np.ones(len(ids)))
    neural=parent.candidate_errors(c,data,pos)
    return data['baseline_ade'][ids,1],data['baseline_fde'][ids,1],floor,neural


def score(model,c,idx):
    p,distance=api.predict(model,c['x'][idx],c['env'][idx]); p=p.astype(np.float32)
    arrays=dict(ids=c['ids'][idx],scores=p,support=distance<=model['support_limit'])
    for offset,ref in enumerate(('cv','floor')):
        positive,safe=api.decisions(p[:,offset*5:offset*5+5],c['moving'][idx],arrays['support'])
        arrays[ref+'_positive']=positive; arrays[ref+'_safe']=safe
    return arrays


def fit_all(cfg,data,jobs,identity,bound,*,resume=False,replay=False,limit=None):
    refs=[]; total=0
    for c in contexts(data,jobs,identity):
        sites=data['sites'][c['ids']]
        for pair,(fit_sites,held_sites) in enumerate(c['pairs']):
            api.assert_roles(c['producer_sites'],c['controller_sites'],fit_sites,held_sites)
            fit=np.flatnonzero(np.isin(sites,fit_sites)); held=np.flatnonzero(np.isin(sites,held_sites))
            name=c['name']+f'_pair{pair}'; home=PRIVATE/'fits'/name; path=home/'complete.json'
            if path.exists() and not replay:
                if not resume: raise ValueError('Use --resume for completed fits')
                r=json.loads(path.read_text()); assert r['identity']==bound
                for v in r['artifacts'].values(): assert artifact(ROOT/v['path'])==v
            else:
                if shutil.disk_usage(PRIVATE).free<10*2**30: raise OSError('Preserve10GiB; resume completed groups')
                # Only these two fitting sources reach new supervised targets or preprocessing.
                cv,_,(floor,_),(neural,_)=costs(c,data,fit)
                y=np.column_stack([api.targets(cv,r,neural,c['job']['design']['easy_cut']) for r in (cv,floor)])
                start=time.monotonic()
                model=api.fit_ridge(c['x'][fit],y,cv,sites[fit],ridge_lambda=cfg['ridge_lambda'],clip=cfg['feature_clip'])
                _,dist=api.predict(model,c['x'][fit],c['env'][fit]); weights=np.zeros(len(fit))
                for s in fit_sites:
                    ix=(sites[fit]==s)&np.isfinite(cv); weights[ix]=.5/ix.sum()
                model['support_limit']=support_limit(dist,weights)
                arrays=score(model,c,held)
                if replay:
                    old=torch.load(home/'ridge.pt',map_location='cpu',weights_only=False)
                    assert old['identity']==bound
                    for k,v in model.items(): np.testing.assert_array_equal(old['model'][k],v) if isinstance(v,np.ndarray) else None
                    assert old['model']['support_limit']==model['support_limit']
                else:
                    home.mkdir(parents=True,exist_ok=True)
                    temp=home/'ridge.tmp'; torch.save(dict(identity=bound,model=model),temp); os.replace(temp,home/'ridge.pt')
                parent.write_arrays(home/'scores.npz',arrays,replay)
                if not replay:
                    immutable_json(path,dict(identity=bound,name=name,producer_sites=c['producer_sites'],
                        controller_sites=c['controller_sites'],fitting_sites=fit_sites,held_sites=held_sites,
                        known_training_rows=model['known_rows'],unknown_training_rows=model['unknown_rows'],
                        training_seconds=time.monotonic()-start,held_labels_read=False,linear_heads=2,
                        artifacts=dict(checkpoint=artifact(home/'ridge.pt'),scores=artifact(home/'scores.npz'))))
            refs.append(artifact(path)); total+=1; beat('fit_replayed' if replay else 'fit_frozen',group=name,complete=total)
            if limit and total>=limit:
                assert replay
                immutable_json(PUBLIC/'fit_replay.json',dict(groups=total,exact=True,registration=artifact(PUBLIC/'registration.json')))
                return
    assert total==cfg['groups']
    immutable_json(PUBLIC/'decision_freeze.json',dict(identity=bound,groups=refs,linear_heads=216,
        new_forecasters=0,held_labels_read_for_new_fits=False,independent_roles_read=False))


def prediction_replay(data,jobs,identity,bound):
    count=0
    # No future arrays in the real feature construction/prediction path.
    causal={k:data[k] for k in ('sites','history','origin','geometry')}
    for c in contexts(causal,jobs,identity):
        for pair,(_,held_sites) in enumerate(c['pairs']):
            home=PRIVATE/'fits'/(c['name']+f'_pair{pair}')
            state=torch.load(home/'ridge.pt',map_location='cpu',weights_only=False); assert state['identity']==bound
            held=np.flatnonzero(np.isin(causal['sites'][c['ids']],held_sites))
            parent.write_arrays(home/'scores.npz',score(state['model'],c,held),True); count+=1
        beat('causal_prediction_replayed',group=c['name'],complete=count)
    immutable_json(PUBLIC/'prediction_replay.json',dict(groups=count,exact=True,future_fields_removed=True))


def safe_ratio_gain(ref,value,mask):
    den=float(np.sum(ref[mask])); return 100*(1-float(np.sum(value[mask]))/den) if den>0 else None


def metric(cv,floor,neural,cf,ff,nf,selected,selected_fde,take,valid,easy,hard):
    known=np.isfinite(cv); e=known&(cv>0)&(cv<=easy); h=known&(cv>=hard)
    endpoint=np.isfinite(cf); complete=known&valid.all(1); partial=known&~valid.all(1)
    harm=np.maximum(selected-floor,0); selected_known=take&known
    m=dict(rows=len(cv),known_rows=int(known.sum()),unknown_rows=int((~known).sum()),
        intervention_rate=float(take.mean()),known_intervention_rate=float(take[known].mean()),
        unknown_interventions=int(take[~known].sum()),error_sum=float(selected[known].sum()),
        floor_error_sum=float(floor[known].sum()),CV_error_sum=float(cv[known].sum()),
        zero_CV_harmed=int(((cv==0)&(selected>0)).sum()),
        selected_positive_harm_ratio=float(harm[selected_known].sum()/floor[selected_known].sum()) if floor[selected_known].sum()>0 else None,
        p95_ratio_to_floor=float(np.quantile(selected[known],.95)/np.quantile(floor[known],.95)) if np.quantile(floor[known],.95)>0 else None)
    for name,mask in dict(all=known,easy=e,hard=h,complete=complete,partial=partial).items():
        m[name+'_gain_floor']=safe_ratio_gain(floor,selected,mask)
        m[name+'_gain_CV']=safe_ratio_gain(cv,selected,mask)
    m['FDE_gain_floor']=safe_ratio_gain(ff,selected_fde,endpoint)
    m['FDE_gain_CV']=safe_ratio_gain(cf,selected_fde,endpoint)
    return m


def evaluate(cfg,data,jobs,identity,bound,replay=False):
    inter.committed(PUBLIC/'decision_freeze.json')
    freeze=json.loads((PUBLIC/'decision_freeze.json').read_text()); assert freeze['identity']==bound
    for r in freeze['groups']: assert artifact(ROOT/r['path'])==r
    rows=[]; decompositions=[]; learnability=[]
    for c in contexts(data,jobs,identity):
        cv,cf,(floor,ff),(neural,nf)=costs(c,data,np.arange(len(c['ids'])))
        for pair,(_,held_sites) in enumerate(c['pairs']):
            name=c['name']+f'_pair{pair}'; home=PRIVATE/'fits'/name
            state=torch.load(home/'ridge.pt',map_location='cpu',weights_only=False); assert state['identity']==bound
            with np.load(home/'scores.npz',allow_pickle=False) as z: arrays={k:z[k].copy() for k in z.files}
            ids=arrays['ids']; pos=np.searchsorted(c['ids'],ids); np.testing.assert_array_equal(c['ids'][pos],ids)
            with np.load(parent.PRIVATE/'decisions'/(name+'.npz'),allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'],ids); calibrated=z['excess_calibrated_supported'].copy()
            with np.load(parent.PRIVATE/'decisions'/(name.replace('_dimensionless','_damped')+'.npz'),allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'],ids)
                np.testing.assert_array_equal(z['excess_calibrated_supported'],c['floor_mask'][pos])
            for site in held_sites:
                local=data['sites'][ids]==site; ix=pos[local]; rowids=ids[local]
                b,d,n=cv[ix],floor[ix],neural[ix]; bf,df,nfe=cf[ix],ff[ix],nf[ix]
                known=np.isfinite(b); oracle=known&(n<d)
                policies=dict(cv=(b,bf,np.zeros(len(ix),bool)),floor=(d,df,np.zeros(len(ix),bool)),
                    raw_neural=(n,nfe,np.ones(len(ix),bool)))
                for label,take in [('parent_original',c['neural_mask'][ix]),('parent_rebased',c['neural_mask'][ix]),
                    ('parent_calibrated_rebased',calibrated[local]),('oracle_diagnostic',oracle),
                    *[(k,arrays[k][local]) for k in ('cv_positive','cv_safe','floor_positive','floor_safe')]]:
                    fallback=b if label=='parent_original' else d; fallback_fde=bf if label=='parent_original' else df
                    policies[label]=(np.where(take,n,fallback),np.where(take,nfe,fallback_fde),take)
                base=dict(site=site,group=name,seed=c['job']['old_identity']['seed'])
                for label,(a,f,take) in policies.items():
                    m=metric(b,d,n,bf,df,nfe,a,f,take,data['valid'][rowids],c['job']['design']['easy_cut'],c['job']['design']['hard_cut'])
                    rows.append(dict(**base,policy=label,metric=m))
                for label,take in [('parent_rebased',c['neural_mask'][ix]),('parent_calibrated_rebased',calibrated[local]),
                    ('floor_safe',arrays['floor_safe'][local]),('floor_positive',arrays['floor_positive'][local])]:
                    m=api.accounting(b,d,n,take); den=d[known].sum()
                    m={k:(100*v/den if k!='unknown_rows' and den>0 else v) for k,v in m.items()}
                    decompositions.append(dict(**base,policy=label,metric=m))
                for offset,ref in enumerate(('cv','floor')):
                    y=api.targets(b,b if ref=='cv' else d,n,c['job']['design']['easy_cut'])
                    pred=arrays['scores'][local,offset*5:offset*5+5]
                    constant=dict(state['model'],coef=np.zeros_like(state['model']['coef']))
                    cp=api.predict(constant,c['x'][ix],c['env'][ix])[0][:,offset*5:offset*5+5]
                    m={}
                    for col,key in enumerate(('benefit','harm','reference','easy_reference','easy_harm')):
                        mse=float(np.mean((pred[known,col]-y[known,col])**2))
                        control=float(np.mean((cp[known,col]-y[known,col])**2))
                        m[key+'_MSE_skill_percent']=100*(1-mse/control) if control>0 else None
                    truth=(d[known]>n[known]); scores=pred[known,0]-pred[known,1]
                    m['floor_gain_AUROC']=float(roc_auc_score(truth,scores)) if len(set(truth))==2 else None
                    learnability.append(dict(**base,policy=ref,metric=m))
        beat('held_scored',group=c['name'])
    sites=sorted(set(data['sites']))
    def aggregate(rr):
        return {k:inter.paired_localities(rr,sites,k,cfg['bootstrap_resamples'],cfg['bootstrap_seed']) for k in rr[0]['metric']}
    def by_policy(rr): return {p:aggregate([r for r in rr if r['policy']==p]) for p in sorted(set(r['policy'] for r in rr))}
    index={(r['group'],r['site'],r['policy']):r for r in rows}
    comparisons=[]
    for r in rows:
        if r['policy']!='floor_safe': continue
        other=index[(r['group'],r['site'],'cv_safe')]['metric']['error_sum']
        comparisons.append(dict(site=r['site'],seed=r['seed'],metric=dict(gain=100*(1-r['metric']['error_sum']/other) if other>0 else None)))
    primary=aggregate(comparisons)['gain']; summary=by_policy(rows)
    selected=[r['metric'] for r in rows if r['policy']=='floor_safe']
    gates=dict(primary_floor_target_beats_matched_cv_target=primary['ci95'][0]>0,
        floor_safe_beats_floor=summary['floor_safe']['all_gain_floor']['ci95'][0]>0,
        every_view_easy_preserved=all(r['easy_gain_CV'] is not None and r['easy_gain_CV']>=-2 for r in selected),
        no_zero_reference_harm=all(r['zero_CV_harmed']==0 for r in selected),
        no_observed_incremental_risk_violation=all(r['selected_positive_harm_ratio'] is None or r['selected_positive_harm_ratio']<=.02 for r in selected),
        independent_confirmation=False,calibration_certificate=False,deployment_changed=False,stage5c_executed=False,smc_enabled=False)
    doc=dict(identity=bound,result_source='fresh_run_linear_fits_readout_cached_verified_forecaster_floor',
        primary=primary,summary=summary,decomposition=by_policy(decompositions),learnability=by_policy(learnability),
        by_seed={str(s):by_policy([r for r in rows if r['seed']==s]) for s in cfg['seeds']},gates=gates,
        worst_views={p:dict(worst_easy_gain_CV=min(r['metric']['easy_gain_CV'] for r in rows if r['policy']==p and r['metric']['easy_gain_CV'] is not None),
            zero_reference_harmed_views=sum(r['metric']['zero_CV_harmed']>0 for r in rows if r['policy']==p),
            risk_violating_views=sum(r['metric']['selected_positive_harm_ratio'] is not None and r['metric']['selected_positive_harm_ratio']>.02 for r in rows if r['policy']==p)) for p in summary},
        groups=108,linear_heads=216,new_forecasters=0,neural_training_updates=0,
        floor_matches_parent_calibrated_supported=True,independent_roles_read=False)
    immutable_json(PRIVATE/'details.json',dict(rows=rows,decomposition=decompositions,learnability=learnability))
    immutable_json(PUBLIC/'summary.json',doc); immutable_json(PUBLIC/'gates.json',gates)
    if replay: immutable_json(PUBLIC/'evaluation_replay.json',dict(exact=True,summary=artifact(PUBLIC/'summary.json'),details=artifact(PRIVATE/'details.json')))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',required=True,choices=['register','fit','replay_fit','replay_predict','evaluate','replay_evaluate'])
    p.add_argument('--resume',action='store_true'); args=p.parse_args()
    PUBLIC.mkdir(parents=True,exist_ok=True); PRIVATE.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    with (PRIVATE/'run.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); beat('phase_started',phase=args.phase)
        cfg,data,jobs,identity,bound=load(args.phase=='register')
        if args.phase in ('fit','replay_fit'): fit_all(cfg,data,jobs,identity,bound,resume=args.resume,replay=args.phase=='replay_fit',limit=1 if args.phase=='replay_fit' else None)
        elif args.phase=='replay_predict': prediction_replay(data,jobs,identity,bound)
        elif args.phase in ('evaluate','replay_evaluate'): evaluate(cfg,data,jobs,identity,bound,replay=args.phase=='replay_evaluate')
        beat('phase_complete',phase=args.phase)


if __name__=='__main__': main()
