"""Locality-excluded harm fitting; no decision or forecast changes."""
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
from scripts import run_m3w_european_reference_protection as previous
from src.evaluation import m3w_harm_tail_diagnostics as diagnostic
import numpy as np
import torch
parent=previous.parent
PUBLIC=ROOT/'outputs/publication_readiness_2026_09/european_harm_tail_crossfit_v1'
PRIVATE=ROOT/'data/stage_cvpr2027_experiments/european_harm_tail_crossfit_v1'
CONFIG='configs/m3w_european_harm_tail_crossfit_v1.json'
artifact,digest,immutable_json,array_hash=previous.artifact,previous.digest,previous.immutable_json,previous.array_hash
FILES=[CONFIG,'src/evaluation/m3w_harm_tail_diagnostics.py','tests/test_m3w_harm_tail_diagnostics.py',
    'scripts/run_m3w_european_harm_tail_crossfit.py','tests/test_m3w_harm_tail_crossfit.py',
    'outputs/publication_readiness_2026_09/european_harm_tail_crossfit_v1/registration.md']


def beat(state,**kw):
    row=dict(pid=os.getpid(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),state=state,**kw)
    parent.base.cross.json_write(PRIVATE/'heartbeat.json',row)
    with (PRIVATE/'events.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)


def registration(create=False):
    cfg=json.loads((ROOT/CONFIG).read_text()); _,pid=previous.registration()
    v=json.loads((previous.PUBLIC/'verification.json').read_text()); assert v['all_passed']
    for f,h in v['artifacts'].items(): assert digest(previous.PUBLIC/f)==h
    for f,h in v['source_bindings'].items(): assert digest(ROOT/f)==h
    previous.checked_training(pid)
    assert cfg['head_training']==json.loads((ROOT/parent.CONFIG).read_text())['head_training']
    assert (cfg['new_heads'],cfg['updates'])==(144,288000)
    assert not any(cfg[k] for k in ('new_forecaster_training','threshold_refit','selection_access',
        'reserved_calibration_access','confirmation_access','deployment_changed','stage5c_executed','smc_enabled'))
    identity=dict(parent=pid,bindings={f:digest(ROOT/f) for f in FILES},
        previous_verification=artifact(previous.PUBLIC/'verification.json'))
    path=PUBLIC/'registration_lock.json'
    if create: immutable_json(path,identity)
    else:
        assert json.loads(path.read_text())==identity
        parent.previous.require_committed(path)
    return cfg,identity


def fold_inputs(x,env,base,cv,sites,held):
    sites=np.asarray(sites); train=sites!=held; test=sites==held
    if not test.any() or len(set(sites[train]))!=3: raise ValueError('Three fit localities and one held locality required')
    pr=parent.base.ordinary.preprocess(x[train],base[train,:2],cv[train],sites[train],held)
    cut=pr['positive_easy_cut']; y=diagnostic.event_targets(base[train],cv[train],cut)
    masks=np.column_stack((np.ones(train.sum(),bool),env[train]>0,env[train]>0))
    assert held not in pr['training_sites'] and np.array_equal(pr['known'],np.isfinite(y).all(1))
    return train,test,pr,cut,y,masks


def training_edges(pred,env,pr,cfg):
    return {k:diagnostic.quantiles(v,pr['weights'],cfg['score_bin_quantiles'])
        for k,v in diagnostic.score_columns(pred,env).items()}


def train(cfg,identity,resume=False,pilot=False):
    heads=[]
    for g,data,pairs in parent.contexts(identity['parent']['parent']['parent']):
        name=g['group']; seed=int(name.split('_seed')[1].split('_')[0]); bi=pairs['B']['ids']
        sites=data['sites'][bi]; cv=data['baseline_ade'][bi,1]
        assert len(set(sites))==4 and not set(sites)&set(g['producer_roster'])
        for pair in cfg['pairs']:
            bx,env,by,*_=parent.pair_inputs(g,data,pairs,pair)
            for held in sorted(set(sites)):
                if shutil.disk_usage(PRIVATE).free<10*1024**3: raise OSError('10 GiB reserve; preserve checkpoints')
                tr,te,pr,cut,y,masks=fold_inputs(bx,env,by,cv,sites,held)
                tag=name+'_'+pair+'_'+held
                inputs=dict(group=g,pair=pair,held=held,training_sites=pr['training_sites'],
                    training_ids_sha256=array_hash(bi[tr]),held_ids_sha256=array_hash(bi[te]),
                    training_x_sha256=array_hash(bx[tr]),held_x_sha256=array_hash(bx[te]),
                    training_y_sha256=array_hash(y),easy_cut=cut,cut_scope='three_fit_localities_only',
                    causal_subset='envelope_positive_no_B_fitted_selector',held_targets_used_for_fit=False)
                inp=PRIVATE/'inputs'/(tag+'.json'); immutable_json(inp,inputs)
                hid=dict(experiment=identity,input=artifact(inp),seed=seed)
                directory=PRIVATE/'heads'/tag; receipt=directory/'complete.json'
                if receipt.exists():
                    r=json.loads(receipt.read_text()); assert r['identity']==hid
                    for ref in r['artifacts'].values(): assert artifact(ROOT/ref['path'])==ref
                else:
                    beat('fit',group=name,pair=pair,held=held)
                    model,fit=parent.method.fit(bx[tr],y,sites[tr],env[tr],masks,pr,arm='mean',seed=seed,
                        settings=cfg['head_training'],identity=hid,directory=directory,resume=resume,
                        stop_at=100 if pilot else None,
                        heartbeat=lambda **kw:beat(group=name,pair=pair,held=held,**kw))
                    if pilot:
                        immutable_json(PRIVATE/'pilot.json',dict(fit=fit,checkpoint=artifact(directory/'checkpoint.pt'),
                            projected_fit_seconds=fit['seconds']/100*cfg['updates'])); return
                    train_pred=parent.method.predict(model,bx[tr],env[tr],pr)
                    edges=training_edges(train_pred,env[tr],pr,cfg)
                    diag=diagnostic.summarize(train_pred,y,env[tr],sites[tr],edges)
                    score=parent.method.predict(model,bx[te],env[te],pr)
                    parent.previous.parent.atomic_npz(directory/'scores.npz',ids=bi[te],scores=score)
                    frozen,state=parent.restore(directory)
                    np.testing.assert_array_equal(parent.method.predict(frozen,bx[te][:4096],env[te][:4096],pr),score[:4096])
                    immutable_json(directory/'fit_diagnosis.json',dict(edges=edges,training=diag))
                    r=dict(identity=hid,fit=fit,input=inputs,result_source='fresh_run',
                        artifacts=dict(checkpoint=artifact(directory/'checkpoint.pt'),scores=artifact(directory/'scores.npz'),
                            diagnosis=artifact(directory/'fit_diagnosis.json')))
                    immutable_json(receipt,r)
                heads.append(artifact(receipt)); beat('head_frozen',group=name,pair=pair,held=held,completed=len(heads))
    assert len(heads)==144
    immutable_json(PRIVATE/'training_complete.json',dict(identity=identity,heads=heads,updates=288000,all_passed=True))
    immutable_json(PUBLIC/'prediction_freeze.json',dict(identity=identity,manifest=artifact(PRIVATE/'training_complete.json'),
        heads=144,updates=288000,held_outcomes_read=False,C_readout=False,policy_changed=False))


def checked_training(identity):
    d=json.loads((PRIVATE/'training_complete.json').read_text()); assert d['identity']==identity and d['all_passed']
    for ref in d['heads']:
        assert artifact(ROOT/ref['path'])==ref
        for a in json.loads((ROOT/ref['path']).read_text())['artifacts'].values(): assert artifact(ROOT/a['path'])==a
    return d


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--phase',required=True,choices=['register','pilot','train']); ap.add_argument('--resume',action='store_true')
    args=ap.parse_args(); PUBLIC.mkdir(parents=True,exist_ok=True); PRIVATE.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    with (PRIVATE/'lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); cfg,identity=registration(args.phase=='register')
        if args.phase!='register': train(cfg,identity,args.resume,args.phase=='pilot')


if __name__=='__main__': main()
