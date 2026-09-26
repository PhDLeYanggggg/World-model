"""Direct easy-membership learning under the frozen source-development protocol."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import time
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts import run_m3w_european_frozen_harm_readout as parent
from src.world_model import m3w_easy_membership_probe as method
from src.evaluation import m3w_easy_membership_metrics as metric
import numpy as np
import torch
base=parent.parent.base; tail=parent.parent.previous
PUBLIC=ROOT/'outputs/publication_readiness_2026_09/european_easy_membership_v1'
PRIVATE=ROOT/'data/stage_cvpr2027_experiments/european_easy_membership_v1'
CONFIG='configs/m3w_european_easy_membership_v1.json'
artifact,digest,immutable_json,array_hash=parent.artifact,parent.digest,parent.immutable_json,parent.array_hash
FILES=[CONFIG,'src/world_model/m3w_easy_membership_probe.py','src/evaluation/m3w_easy_membership_metrics.py',
    'tests/test_m3w_easy_membership_probe.py','tests/test_m3w_easy_membership_metrics.py',
    'scripts/run_m3w_european_easy_membership.py',str(PUBLIC.relative_to(ROOT)/'registration.md')]


def beat(state,**kw):
    row=dict(pid=os.getpid(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),state=state,**kw)
    base.base.cross.json_write(PRIVATE/'heartbeat.json',row)
    with (PRIVATE/'events.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)


def registration(create=False):
    cfg=json.loads((ROOT/CONFIG).read_text()); _,pid=parent.registration(); parent.checked_training(pid)
    v=json.loads((parent.PUBLIC/'verification.json').read_text()); assert v['all_passed']
    for f,h in v['artifacts'].items(): assert digest(parent.PUBLIC/f)==h
    for f,h in v['source_bindings'].items(): assert digest(ROOT/f)==h
    assert (cfg['new_heads'],cfg['updates'])==(288,576000)
    assert cfg['head_training']==json.loads((ROOT/base.CONFIG).read_text())['head_training']
    assert not any(cfg[k] for k in ('new_forecaster_training','new_policy_evaluation','threshold_refit','selection_access',
        'reserved_calibration_access','confirmation_access','deployment_changed','stage5c_executed','smc_enabled'))
    identity=dict(source=pid['source'],parent_verification=artifact(parent.PUBLIC/'verification.json'),
        bindings={f:digest(ROOT/f) for f in FILES})
    path=PUBLIC/'registration_lock.json'
    if create: immutable_json(path,identity)
    else:
        assert json.loads(path.read_text())==identity; base.previous.require_committed(path)
    return cfg,identity


def measure(p,y,env,sites):
    return {s:metric.summarize(p,y,sites,mask) for s,mask in
        (('all',np.ones(len(y),bool)),('envelope_positive',env>0))}


def train(cfg,identity,resume=False,pilot=False):
    heads=[]
    for g,data,pairs in base.contexts(identity['source']):
        name=g['group']; seed=int(name.split('_seed')[1].split('_')[0]); bi=pairs['B']['ids']; sites=data['sites'][bi]; cv=data['baseline_ade'][bi,1]
        for pair in cfg['pairs']:
            x,env,by,*_=base.pair_inputs(g,data,pairs,pair)
            for held in sorted(set(sites)):
                if shutil.disk_usage(PRIVATE).free<10*1024**3: raise OSError('10 GiB reserve; keep resumable checkpoints')
                tr,te,pr,cut,oldy,_=tail.fold_inputs(x,env,by,cv,sites,held)
                y=method.labels(cv[tr],cut); assert np.array_equal(np.isfinite(y),pr['known'])
                tag=name+'_'+pair+'_'+held; olddir=parent.sources(tag)['mean_features']; _,control=base.restore(olddir)
                assert held not in pr['training_sites'] and held not in g['producer_roster']
                initial=[]
                for arm in cfg['arms']:
                    inp=PRIVATE/'inputs'/(tag+'_'+arm+'.json')
                    inputs=dict(group=g,pair=pair,held=held,arm=arm,easy_cut=cut,training_sites=pr['training_sites'],feature_dimension=x.shape[1],
                        train_x_sha256=array_hash(x[tr]),held_x_sha256=array_hash(x[te]),train_y_sha256=array_hash(y),
                        train_ids_sha256=array_hash(bi[tr]),held_ids_sha256=array_hash(bi[te]),original_control=artifact(olddir/'complete.json'),
                        held_targets_used_for_fit=False)
                    immutable_json(inp,inputs); hid=dict(experiment=identity,input=artifact(inp),seed=seed)
                    directory=PRIVATE/'heads'/tag/arm; receipt=directory/'complete.json'
                    if receipt.exists():
                        r=json.loads(receipt.read_text()); assert r['identity']==hid
                        for a in r['artifacts'].values(): assert artifact(ROOT/a['path'])==a
                    else:
                        beat('fit',group=name,pair=pair,held=held,arm=arm)
                        model,fit=method.fit(x[tr],y,sites[tr],pr,arm=arm,seed=seed,settings=cfg['head_training'],identity=hid,
                            directory=directory,heartbeat=lambda **kw:beat(group=name,pair=pair,held=held,arm=arm,**kw),resume=resume,stop_at=100 if pilot else None)
                        if pilot:
                            immutable_json(PRIVATE/'pilot.json',dict(fit=fit,checkpoint=artifact(directory/'checkpoint.pt'),
                                projected_linear_fit_seconds=fit['seconds']/100*cfg['updates'],MLP_cost_not_yet_measured=True)); return
                        train_score=method.predict(model,x[tr],pr); score=method.predict(model,x[te],pr)
                        base.previous.parent.atomic_npz(directory/'scores.npz',ids=bi[te],scores=score)
                        restored,state=method.restore(directory)
                        np.testing.assert_array_equal(method.predict(restored,x[te][:4096],pr),score[:4096])
                        for k in ('draws','fixed_ids'): np.testing.assert_array_equal(state[k],control[k])
                        assert torch.equal(state['sampler_rng'],control['sampler_rng'])
                        fitmetrics=measure(train_score,y,env[tr],sites[tr]); constant=np.full(tr.sum(),fit['prevalence'])
                        immutable_json(directory/'fit_metrics.json',dict(model=fitmetrics,train_constant=measure(constant,y,env[tr],sites[tr])))
                        r=dict(identity=hid,input=inputs,fit=fit,result_source='fresh_run_membership_probe_cached_verified_causal_inputs',
                            artifacts=dict(checkpoint=artifact(directory/'checkpoint.pt'),scores=artifact(directory/'scores.npz'),metrics=artifact(directory/'fit_metrics.json')))
                        immutable_json(receipt,r)
                    initial.append(r['fit']['trace'][0]); heads.append(artifact(receipt)); beat('head_frozen',completed=len(heads),group=name,pair=pair,held=held,arm=arm)
                assert initial[0]==initial[1]
    assert len(heads)==288
    immutable_json(PRIVATE/'training_complete.json',dict(identity=identity,heads=heads,updates=576000,all_passed=True))
    immutable_json(PUBLIC/'prediction_freeze.json',dict(identity=identity,manifest=artifact(PRIVATE/'training_complete.json'),heads=288,updates=576000,
        current_outcome_readout=False,policy_changed=False))


def checked_training(identity):
    d=json.loads((PRIVATE/'training_complete.json').read_text()); assert d['identity']==identity and d['all_passed']
    for ref in d['heads']:
        assert artifact(ROOT/ref['path'])==ref
        for a in json.loads((ROOT/ref['path']).read_text())['artifacts'].values(): assert artifact(ROOT/a['path'])==a
    return d


def evaluate(cfg,identity):
    checked_training(identity); base.previous.require_committed(PUBLIC/'prediction_freeze.json'); refs=[]
    for g,data,pairs in base.contexts(identity['source']):
        name=g['group']; bi=pairs['B']['ids']; sites=data['sites'][bi]; cv=data['baseline_ade'][bi,1]
        for pair in cfg['pairs']:
            path=PUBLIC/'groups'/(name+'_'+pair+'.json'); receipt=PRIVATE/'readout'/path.name
            if path.exists():
                assert artifact(path)==json.loads(receipt.read_text()); refs.append(artifact(path)); continue
            x,env,by,*_=base.pair_inputs(g,data,pairs,pair); folds=[]
            for held in sorted(set(sites)):
                tr,te,pr,cut,oldy,_=tail.fold_inputs(x,env,by,cv,sites,held)
                y=method.labels(cv[te],cut); tag=name+'_'+pair+'_'+held; stats={}; fitting={}; source={}
                prevalence=float(pr['weights']@np.nan_to_num(method.labels(cv[tr],cut),nan=0.))
                stats['train_constant']=measure(np.full(te.sum(),prevalence),y,env[te],sites[te])
                with np.load(parent.sources(tag)['mean_features']/'scores.npz',allow_pickle=False) as z:
                    np.testing.assert_array_equal(z['ids'],bi[te]); p=z['scores']
                    ratio=np.divide(p[:,2],p[:,0],out=np.zeros(len(p)),where=p[:,0]>0)
                assert (ratio<=1+1e-6).all(); ratio=np.clip(ratio,0,1)
                stats['reference_ratio']=measure(ratio,y,env[te],sites[te])
                for arm in cfg['arms']:
                    directory=PRIVATE/'heads'/tag/arm; source[arm]=artifact(directory/'complete.json')
                    r=json.loads((directory/'complete.json').read_text()); assert r['fit']['prevalence']==prevalence
                    with np.load(directory/'scores.npz',allow_pickle=False) as z:
                        np.testing.assert_array_equal(z['ids'],bi[te]); stats[arm]=measure(z['scores'],y,env[te],sites[te])
                    fitting[arm]=json.loads((directory/'fit_metrics.json').read_text())
                folds.append(dict(held=held,easy_cut=cut,train_prevalence=prevalence,metrics=stats,training=fitting,heads=source,
                    target_sha256=array_hash(y),held_ids_sha256=array_hash(bi[te]),reference_ratio_is_not_event_probability=True))
            row=dict(group=name,producer=g['producer'],controller=g['controller'],seed=int(name.split('_seed')[1].split('_')[0]),
                pair=pair,folds=folds,result_source='fresh_run_membership_metrics_cached_verified_input_lineage')
            immutable_json(path,row); immutable_json(receipt,artifact(path)); refs.append(artifact(path)); beat('readout',completed=len(refs),group=name,pair=pair)
    assert len(refs)==36
    immutable_json(PUBLIC/'completion_checks.json',dict(identity=identity,groups=refs,all_passed=True))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--phase',required=True,choices=['register','pilot','train','evaluate']); ap.add_argument('--resume',action='store_true')
    args=ap.parse_args(); PUBLIC.mkdir(parents=True,exist_ok=True); PRIVATE.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    with (PRIVATE/'lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); cfg,identity=registration(args.phase=='register')
        if args.phase in ('train','pilot'): train(cfg,identity,args.resume,args.phase=='pilot')
        elif args.phase=='evaluate': evaluate(cfg,identity)


if __name__=='__main__': main()
