"""Frozen, label-assisted factor attribution; not an inference policy."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import time
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts import run_m3w_european_membership_cost as parent
from src.evaluation.m3w_cost_factor_attribution import diagnose
import numpy as np
import torch
PUBLIC=ROOT/'outputs/publication_readiness_2026_09/european_cost_attribution_v1'
PRIVATE=ROOT/'data/stage_cvpr2027_experiments/european_cost_attribution_v1'
artifact,digest,immutable_json=parent.artifact,parent.digest,parent.immutable_json
FILES=['src/evaluation/m3w_cost_factor_attribution.py','tests/test_m3w_cost_factor_attribution.py',
       str(Path(__file__).relative_to(ROOT)),str(PUBLIC.relative_to(ROOT)/'protocol.md')]


def registration(create=False):
    _,pid=parent.registration(); parent.checked_training(pid)
    v=json.loads((parent.PUBLIC/'verification.json').read_text()); assert v['all_passed']
    for p,h in v['artifacts'].items(): assert digest(parent.PUBLIC/p)==h
    for p,h in v['source_bindings'].items(): assert digest(ROOT/p)==h
    identity=dict(parent=pid,parent_verification=artifact(parent.PUBLIC/'verification.json'),
                  bindings={p:digest(ROOT/p) for p in FILES})
    path=PUBLIC/'registration_lock.json'
    if create: immutable_json(path,identity)
    else:
        assert json.loads(path.read_text())==identity; parent.base.previous.require_committed(path)
    return identity


def beat(**kw):
    item=dict(pid=os.getpid(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw)
    parent.base.base.cross.json_write(PRIVATE/'heartbeat.json',item)
    with (PRIVATE/'events.jsonl').open('a') as f: f.write(json.dumps(item)+'\n')
    print(json.dumps(item),flush=True)


def experts(model,x,env,pr):
    values=[]; model.eval()
    with torch.no_grad():
        for start in range(0,len(x),4096):
            z=torch.from_numpy(parent.method.standardized(x[start:start+4096],pr))
            d=torch.tensor(env[start:start+4096]/pr['cost_scale'],dtype=torch.float32)
            values.append((d[:,None]*model.network(z).sigmoid()).numpy()*pr['cost_scale'])
    return np.concatenate(values)


def readouts(identity,verify=False):
    refs=[]; started=time.monotonic()
    for g,data,pairs in parent.base.contexts(identity['parent']['source']):
        name=g['group']; bi=pairs['B']['ids']; sites=data['sites'][bi]; cv=data['baseline_ade'][bi,1]
        for pair in ('full','motion_only'):
            path=PUBLIC/'groups'/(name+'_'+pair+'.json'); rec=PRIVATE/'readout'/path.name
            if path.exists() and not verify:
                assert artifact(path)==json.loads(rec.read_text()); refs.append(artifact(path)); continue
            x,env,by,*_=parent.base.pair_inputs(g,data,pairs,pair)
            old=json.loads((parent.PUBLIC/'groups'/path.name).read_text()); folds=[]
            for held in sorted(set(sites)):
                if shutil.disk_usage(PRIVATE).free<10*2**30: raise OSError('Preserve 10 GiB reserve')
                tr,te,pr,cut,y,_=parent.tail.fold_inputs(x,env,by,cv,sites,held)
                tag=name+'_'+pair+'_'+held; directory=parent.PRIVATE/'heads'/tag/'conditional'
                model,state=parent.method.restore(directory)
                for k in ('mean','std','known','weights'): np.testing.assert_array_equal(state['preprocess'][k],pr[k])
                member=parent.parent.PRIVATE/'heads'/tag/'mlp'
                with np.load(member/'scores.npz',allow_pickle=False) as z:
                    np.testing.assert_array_equal(z['ids'],bi[te]); prob=z['scores'].copy()
                train_experts=experts(model,x[tr],env[tr],pr)
                severity_edge=parent.tail.diagnostic.quantiles(train_experts[:,0],pr['weights'],(.99,))[0]
                est=experts(model,x[te],env[te],pr)
                with np.load(directory/'scores.npz',allow_pickle=False) as z:
                    np.testing.assert_array_equal(z['ids'],bi[te]); saved=z['scores'].copy()
                np.testing.assert_allclose(prob*est[:,0],saved[:,3],rtol=2e-6,atol=2e-5)
                np.testing.assert_allclose(prob*est[:,0]+(1-prob)*est[:,1],saved[:,1],rtol=2e-6,atol=2e-5)
                original=parent.parent.parent.sources(tag)['mean_features']
                with np.load(original/'scores.npz',allow_pickle=False) as z:
                    np.testing.assert_array_equal(z['ids'],bi[te]); ref=z['scores'].copy()
                target=parent.tail.diagnostic.event_targets(by[te],cv[te],cut)
                out=diagnose(prob,est[:,0],est[:,1],env[te],cv[te],target[:,1],cut,ref[:,3],ref[:,1],severity_edge)
                oldfold=next(f for f in old['folds'] if f['held']==held)
                for sub,oldsub in (('all','all'),('disagreement','envelope_positive')):
                    for value,arm in (('MSE','conditional'),('original_MSE','original_mean')):
                        np.testing.assert_allclose(out['subsets'][sub]['easy'][value],oldfold['metrics'][arm][oldsub]['harm_MSE'],rtol=2e-5,atol=1e-8)
                folds.append(dict(held=held,easy_cut=cut,training_sites=pr['training_sites'],
                    expert_sha256=parent.array_hash(est),membership_sha256=parent.array_hash(prob),
                    target_sha256=parent.array_hash(target),held_ids_sha256=parent.array_hash(bi[te]),
                    conditional_checkpoint=artifact(directory/'checkpoint.pt'),diagnosis=out))
            row=dict(identity=identity,group=name,producer=g['producer'],controller=g['controller'],
                seed=int(name.split('_seed')[1].split('_')[0]),pair=pair,folds=folds,
                result_source='fresh_run_attribution_cached_verified_models_and_source',label_assisted=True,
                policy_changed=False,independent_confirmation=False)
            if verify: assert json.loads(path.read_text())==row
            else: immutable_json(path,row); immutable_json(rec,artifact(path))
            refs.append(artifact(path)); beat(state='verified' if verify else 'computed',groups=len(refs),group=name,pair=pair)
    assert len(refs)==36
    if verify:
        immutable_json(PUBLIC/'replay_receipt.json',dict(identity=identity,groups=refs,all_passed=True,
            held_folds=144,full_expert_replays=144,training_percentile_replays=144))
    else:
        immutable_json(PUBLIC/'completion_checks.json',dict(identity=identity,groups=refs,all_passed=True,
            new_fits=0,threshold_changes=0,policy_changed=False,independent_confirmation=False))
    print(json.dumps(dict(groups=len(refs),seconds=time.monotonic()-started,verify=verify)))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--phase',choices=['register','run','verify'],required=True); args=ap.parse_args()
    PUBLIC.mkdir(parents=True,exist_ok=True); PRIVATE.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    with (PRIVATE/'lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); identity=registration(args.phase=='register')
        if args.phase!='register': readouts(identity,verify=args.phase=='verify')


if __name__=='__main__': main()
