"""One fixed support-fractional loss intervention with matched cached controls."""
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
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(k,'4')
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts import run_m3w_european_harm_tail_crossfit as previous
from src.world_model import m3w_support_fractional_harm as method
import numpy as np
import torch
base=previous.parent
PUBLIC=ROOT/'outputs/publication_readiness_2026_09/european_support_fractional_v1'
PRIVATE=ROOT/'data/stage_cvpr2027_experiments/european_support_fractional_v1'
CONFIG='configs/m3w_european_support_fractional_v1.json'
artifact,digest,immutable_json,array_hash=previous.artifact,previous.digest,previous.immutable_json,previous.array_hash
FILES=[CONFIG,'src/world_model/m3w_support_fractional_harm.py','tests/test_m3w_support_fractional_harm.py',
    'scripts/run_m3w_european_support_fractional.py',
    'outputs/publication_readiness_2026_09/european_support_fractional_v1/registration.md']


def beat(state,**kw):
    row=dict(pid=os.getpid(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),state=state,**kw)
    base.base.cross.json_write(PRIVATE/'heartbeat.json',row)
    with (PRIVATE/'events.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)


def registration(create=False):
    cfg=json.loads((ROOT/CONFIG).read_text()); pc,pid=previous.registration()
    v=json.loads((previous.PUBLIC/'verification.json').read_text()); assert v['all_passed']
    for f,h in v['artifacts'].items(): assert digest(previous.PUBLIC/f)==h
    for f,h in v['source_bindings'].items(): assert digest(ROOT/f)==h
    previous.checked_training(pid)
    assert cfg['head_training']==pc['head_training'] and cfg['coefficient']==1.
    assert (cfg['new_heads'],cfg['updates'])==(144,288000)
    assert not any(cfg[k] for k in ('new_forecaster_training','new_policy_evaluation','threshold_refit',
        'selection_access','reserved_calibration_access','confirmation_access','deployment_changed','stage5c_executed','smc_enabled'))
    identity=dict(parent=pid,source=pid['parent']['parent']['parent'],
        bindings={f:digest(ROOT/f) for f in FILES},previous_verification=artifact(previous.PUBLIC/'verification.json'))
    path=PUBLIC/'registration_lock.json'
    if create: immutable_json(path,identity)
    else:
        assert json.loads(path.read_text())==identity
        base.previous.require_committed(path)
    return cfg,identity


def train(cfg,identity,resume=False,pilot=False):
    heads=[]
    for g,data,pairs in base.contexts(identity['source']):
        name=g['group']; seed=int(name.split('_seed')[1].split('_')[0]); bi=pairs['B']['ids']; sites=data['sites'][bi]
        for pair in cfg['pairs']:
            bx,env,by,*_=base.pair_inputs(g,data,pairs,pair)
            for held in sorted(set(sites)):
                if shutil.disk_usage(PRIVATE).free<10*1024**3: raise OSError('10 GiB reserve; preserve checkpoints')
                tag=name+'_'+pair+'_'+held; control=previous.PRIVATE/'heads'/tag
                _,cs=base.restore(control)
                tr,te,pr,cut,y,masks=previous.fold_inputs(bx,env,by,data['baseline_ade'][bi,1],sites,held)
                for k in ('mean','std','known','weights'): np.testing.assert_array_equal(pr[k],cs['preprocess'][k])
                assert pr['cost_scale']==cs['preprocess']['cost_scale']
                inp=PRIVATE/'inputs'/(tag+'.json')
                inputs=dict(group=g,pair=pair,held=held,training_sites=pr['training_sites'],easy_cut=cut,
                    train_x_sha256=array_hash(bx[tr]),train_y_sha256=array_hash(y),held_x_sha256=array_hash(bx[te]),
                    train_ids_sha256=array_hash(bi[tr]),held_ids_sha256=array_hash(bi[te]),
                    control=artifact(control/'complete.json'),held_labels_used_for_fit=False)
                immutable_json(inp,inputs); hid=dict(experiment=identity,input=artifact(inp),seed=seed)
                directory=PRIVATE/'heads'/tag; receipt=directory/'complete.json'
                if receipt.exists():
                    r=json.loads(receipt.read_text()); assert r['identity']==hid
                    for ref in r['artifacts'].values(): assert artifact(ROOT/ref['path'])==ref
                else:
                    beat('fit',group=name,pair=pair,held=held)
                    model,fit=method.fit(bx[tr],y,sites[tr],env[tr],masks,pr,seed=seed,
                        settings=cfg['head_training'],identity=hid,directory=directory,resume=resume,
                        coefficient=cfg['coefficient'],stop_at=100 if pilot else None,
                        heartbeat=lambda **kw:beat(group=name,pair=pair,held=held,**kw))
                    if pilot:
                        immutable_json(PRIVATE/'pilot.json',dict(fit=fit,checkpoint=artifact(directory/'checkpoint.pt'),
                            projected_fit_seconds=fit['seconds']/100*cfg['updates'])); return
                    train_score=base.method.predict(model,bx[tr],env[tr],pr)
                    edges=previous.training_edges(train_score,env[tr],pr,cfg)
                    diagnosis=previous.diagnostic.summarize(train_score,y,env[tr],sites[tr],edges)
                    score=base.method.predict(model,bx[te],env[te],pr)
                    base.previous.parent.atomic_npz(directory/'scores.npz',ids=bi[te],scores=score)
                    restored,state=base.restore(directory)
                    np.testing.assert_array_equal(base.method.predict(restored,bx[te][:4096],env[te][:4096],pr),score[:4096])
                    np.testing.assert_array_equal(cs['draws'],state['draws']); assert torch.equal(cs['sampler_rng'],state['sampler_rng'])
                    np.testing.assert_array_equal(cs['fixed_ids'],state['fixed_ids'])
                    assert cs['trace'][0]['moment_mse']==state['trace'][0]['moment_mse']
                    immutable_json(directory/'fit_diagnosis.json',dict(edges=edges,training=diagnosis))
                    r=dict(identity=hid,input=inputs,fit=fit,result_source='fresh_run',matched_draws=True,
                        artifacts=dict(checkpoint=artifact(directory/'checkpoint.pt'),scores=artifact(directory/'scores.npz'),
                            diagnosis=artifact(directory/'fit_diagnosis.json')))
                    immutable_json(receipt,r)
                heads.append(artifact(receipt)); beat('head_frozen',group=name,pair=pair,held=held,completed=len(heads))
    assert len(heads)==144
    immutable_json(PRIVATE/'training_complete.json',dict(identity=identity,heads=heads,updates=288000,all_passed=True))
    immutable_json(PUBLIC/'prediction_freeze.json',dict(identity=identity,manifest=artifact(PRIVATE/'training_complete.json'),
        heads=144,updates=288000,current_held_readout=False,policy_changed=False))


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
