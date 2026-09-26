"""Freeze risk representations, fit matched harm readouts, keep reference moments."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import time
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts import run_m3w_european_support_fractional as parent
from src.world_model import m3w_frozen_harm_readout as method
import numpy as np
import torch
PUBLIC=ROOT/'outputs/publication_readiness_2026_09/european_frozen_harm_readout_v1'
PRIVATE=ROOT/'data/stage_cvpr2027_experiments/european_frozen_harm_readout_v1'
CONFIG='configs/m3w_european_frozen_harm_readout_v1.json'
artifact,digest,immutable_json,array_hash=parent.artifact,parent.digest,parent.immutable_json,parent.array_hash
FILES=[CONFIG,'src/world_model/m3w_frozen_harm_readout.py','tests/test_m3w_frozen_harm_readout.py',
    'scripts/run_m3w_european_frozen_harm_readout.py',str(PUBLIC.relative_to(ROOT)/'registration.md')]


def beat(state,**kw):
    row=dict(pid=os.getpid(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),state=state,**kw)
    parent.base.base.cross.json_write(PRIVATE/'heartbeat.json',row)
    with (PRIVATE/'events.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)


def registration(create=False):
    cfg=json.loads((ROOT/CONFIG).read_text()); pc,pid=parent.registration(); parent.checked_training(pid)
    v=json.loads((parent.PUBLIC/'verification.json').read_text()); assert v['all_passed']
    for f,h in v['artifacts'].items(): assert digest(parent.PUBLIC/f)==h
    for f,h in v['source_bindings'].items(): assert digest(ROOT/f)==h
    diagnosis=json.loads((PUBLIC/'membership_diagnosis.json').read_text())
    assert diagnosis['parent_verification']==artifact(parent.PUBLIC/'verification.json')
    for f,h in diagnosis['bindings'].items(): assert digest(ROOT/f)==h
    for ref in diagnosis['groups']: assert artifact(ROOT/ref['path'])==ref
    assert (cfg['new_heads'],cfg['updates'])==(288,576000)
    assert cfg['head_training']=={k:v for k,v in pc['head_training'].items() if k!='width'}
    assert not any(cfg[k] for k in ('new_forecaster_training','new_policy_evaluation','threshold_refit',
        'selection_access','reserved_calibration_access','confirmation_access','deployment_changed','stage5c_executed','smc_enabled'))
    identity=dict(parent=pid,source=pid['source'],bindings={f:digest(ROOT/f) for f in FILES},
        parent_verification=artifact(parent.PUBLIC/'verification.json'),membership_diagnosis=artifact(PUBLIC/'membership_diagnosis.json'))
    path=PUBLIC/'registration_lock.json'
    if create: immutable_json(path,identity)
    else:
        assert json.loads(path.read_text())==identity; parent.base.previous.require_committed(path)
    return cfg,identity


def sources(tag):
    return {'mean_features':parent.previous.PRIVATE/'heads'/tag,'fractional_features':parent.PRIVATE/'heads'/tag}


def train(cfg,identity,resume=False,pilot=False):
    heads=[]
    for g,data,pairs in parent.base.contexts(identity['source']):
        name=g['group']; seed=int(name.split('_seed')[1].split('_')[0]); bi=pairs['B']['ids']; sites=data['sites'][bi]
        for pair in cfg['pairs']:
            bx,env,by,*_=parent.base.pair_inputs(g,data,pairs,pair)
            for held in sorted(set(sites)):
                if shutil.disk_usage(PRIVATE).free<10*1024**3: raise OSError('10 GiB disk reserve; keep resumable checkpoint')
                tr,te,source_pr,cut,y,masks=parent.previous.fold_inputs(bx,env,by,data['baseline_ade'][bi,1],sites,held)
                tag=name+'_'+pair+'_'+held; old,original=parent.base.restore(sources(tag)['mean_features'])
                train_reference=parent.base.method.predict(old,bx[tr],env[tr],source_pr)
                with np.load(sources(tag)['mean_features']/'scores.npz',allow_pickle=False) as z:
                    np.testing.assert_array_equal(z['ids'],bi[te]); held_reference=z['scores'].copy()
                initial=[]
                for arm in cfg['arms']:
                    source=sources(tag)[arm]; encoder,saved=parent.base.restore(source)
                    for k in ('mean','std','known','weights'): np.testing.assert_array_equal(saved['preprocess'][k],source_pr[k])
                    assert held not in source_pr['training_sites'] and held not in g['producer_roster']
                    before={k:v.clone() for k,v in encoder.state_dict().items()}
                    h=method.extract(encoder,bx,source_pr); pr=method.preprocess(h[tr],source_pr)
                    assert all(torch.equal(before[k],v) for k,v in encoder.state_dict().items())
                    assert all(v.grad is None for v in encoder.parameters())
                    inp=PRIVATE/'inputs'/(tag+'_'+arm+'.json')
                    inputs=dict(group=g,pair=pair,held=held,arm=arm,easy_cut=cut,source_checkpoint=artifact(source/'checkpoint.pt'),
                        original_mean=artifact(sources(tag)['mean_features']/'complete.json'),train_latent_sha256=array_hash(h[tr]),
                        held_latent_sha256=array_hash(h[te]),train_y_sha256=array_hash(y),train_ids_sha256=array_hash(bi[tr]),
                        held_ids_sha256=array_hash(bi[te]),held_targets_used_for_fit=False)
                    immutable_json(inp,inputs); hid=dict(experiment=identity,input=artifact(inp),seed=seed)
                    directory=PRIVATE/'heads'/tag/arm; receipt=directory/'complete.json'
                    if receipt.exists():
                        r=json.loads(receipt.read_text()); assert r['identity']==hid
                        for ref in r['artifacts'].values(): assert artifact(ROOT/ref['path'])==ref
                    else:
                        beat('fit',group=name,pair=pair,held=held,arm=arm)
                        model,fit=method.fit(h[tr],y,sites[tr],env[tr],pr,seed=seed,settings=cfg['head_training'],
                            identity=hid,directory=directory,resume=resume,stop_at=100 if pilot else None,
                            heartbeat=lambda **kw:beat(group=name,pair=pair,held=held,arm=arm,**kw))
                        if pilot:
                            immutable_json(PRIVATE/'pilot.json',dict(fit=fit,checkpoint=artifact(directory/'checkpoint.pt'),
                                projected_fit_seconds=fit['seconds']/100*cfg['updates'])); return
                        train_score=method.predict(model,h[tr],env[tr],pr,train_reference)
                        edges=parent.previous.training_edges(train_score,env[tr],pr,cfg)
                        diag=parent.previous.diagnostic.summarize(train_score,y,env[tr],sites[tr],edges)
                        score=method.predict(model,h[te],env[te],pr,held_reference)
                        parent.base.previous.parent.atomic_npz(directory/'scores.npz',ids=bi[te],scores=score)
                        restored,state=method.restore(directory)
                        np.testing.assert_array_equal(method.predict(restored,h[te][:4096],env[te][:4096],pr,held_reference[:4096]),score[:4096])
                        np.testing.assert_array_equal(state['draws'],original['draws'])
                        np.testing.assert_array_equal(state['fixed_ids'],original['fixed_ids'])
                        assert torch.equal(state['sampler_rng'],original['sampler_rng'])
                        immutable_json(directory/'fit_diagnosis.json',dict(edges=edges,training=diag))
                        r=dict(identity=hid,input=inputs,fit=fit,result_source='fresh_run_readout_cached_verified_frozen_encoder',
                            artifacts=dict(checkpoint=artifact(directory/'checkpoint.pt'),scores=artifact(directory/'scores.npz'),
                                diagnosis=artifact(directory/'fit_diagnosis.json')))
                        immutable_json(receipt,r)
                    initial.append(r['fit']['trace'][0]); heads.append(artifact(receipt))
                    beat('head_frozen',group=name,pair=pair,held=held,arm=arm,completed=len(heads))
                assert initial[0]==initial[1]
    assert len(heads)==288
    immutable_json(PRIVATE/'training_complete.json',dict(identity=identity,heads=heads,updates=576000,all_passed=True))
    immutable_json(PUBLIC/'prediction_freeze.json',dict(identity=identity,manifest=artifact(PRIVATE/'training_complete.json'),
        heads=288,updates=576000,current_held_readout=False,policy_changed=False))


def checked_training(identity):
    d=json.loads((PRIVATE/'training_complete.json').read_text()); assert d['identity']==identity and d['all_passed']
    for ref in d['heads']:
        assert artifact(ROOT/ref['path'])==ref
        for r in json.loads((ROOT/ref['path']).read_text())['artifacts'].values(): assert artifact(ROOT/r['path'])==r
    return d


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--phase',required=True,choices=['register','pilot','train']); ap.add_argument('--resume',action='store_true')
    args=ap.parse_args(); PUBLIC.mkdir(parents=True,exist_ok=True); PRIVATE.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    with (PRIVATE/'lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); cfg,identity=registration(args.phase=='register')
        if args.phase!='register': train(cfg,identity,args.resume,args.phase=='pilot')


if __name__=='__main__': main()
