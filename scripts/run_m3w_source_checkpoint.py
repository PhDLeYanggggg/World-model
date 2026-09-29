"""Registered training-source validation versus fixed-final cost-head control."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import sys
import time
if platform.system()=='Darwin' and platform.machine()!='arm64':raise RuntimeError('Native arm64 required')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts import manage_m3w_boundary_diagnostic as parent
from scripts.export_m3w_easy_hurdle_create import closure
from src.world_model import m3w_source_checkpoint as api

NAME='european_source_checkpoint_v1'
PUBLIC=parent.PUBLIC.parent/NAME;PRIVATE=parent.PRIVATE.parent/NAME
CONFIG=ROOT/'configs'/('m3w_'+NAME+'.json')
base=parent.parent.base;digest=parent.digest;immutable=parent.immutable


def registration():
    cfg=json.loads(CONFIG.read_text());parent.registration()
    assert digest(parent.PUBLIC/'local_compute_receipt.json')==cfg['diagnostic_receipt_sha256']
    paths=closure(ROOT,['scripts.run_m3w_source_checkpoint'])
    paths += [CONFIG,PUBLIC/'protocol.md',ROOT/'tests/test_m3w_source_checkpoint.py']
    return cfg,dict(bindings={str(p.relative_to(ROOT)):digest(p) for p in paths},
                    diagnostic_receipt_sha256=cfg['diagnostic_receipt_sha256'],independent_roles_read=False)


def beat(**kw):
    row=dict(pid=os.getpid(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw)
    base.inter.json_write(PRIVATE/'heartbeat.json',row)
    with (PRIVATE/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)


def guard(cfg):
    if shutil.disk_usage(PRIVATE).free<cfg['disk_reserve_bytes']+16*2**20:raise OSError('Preserve10GiB reserve and completed work')


def train(cfg,data,jobs,oid,pilot=False,resume=False,replay=False):
    refs=[];started=time.monotonic();partitions=[]
    causal={k:data[k] for k in parent.parent.old.parent.CAUSAL_KEYS}
    for c in base.floor_api.contexts(causal,jobs,oid):
        for site in parent.parent.sources(c):
            guard(cfg);name=c['name']+'_fit_'+site
            home=PRIVATE/('replay' if replay else 'heads')/name;cp=home/'checkpoint.pt.gz';docpath=home/'complete.json'
            _,ids,x,env,y,_,upstream=parent.parent.training_arrays(c,data,site)
            identity=dict(registration_sha256=digest(PUBLIC/'registration.json'),context=c['name'],source=site,upstream=upstream)
            if docpath.exists() and resume and not replay:
                doc=json.loads(docpath.read_text());assert doc['identity']==identity
                assert base.artifact(cp)==doc['checkpoint']
            else:
                s=api.fit(x,env,y,data['sites'][ids],data['recordings'][ids],data['frames'][ids],source=site,
                    settings=cfg['head_training'],seed=upstream['seed'],path=cp,identity=identity,
                    heartbeat=lambda **kw:beat(state='training',group=name,**kw),resume=resume and not replay,
                    stop_at=cfg['pilot_updates'] if pilot else None)
                if pilot:
                    elapsed=time.monotonic()-started;size=cp.stat().st_size;projected=size*72*2+32*2**20
                    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
                    immutable(PUBLIC/'pilot.json',dict(group=name,steps=s['step'],seconds=elapsed,checkpoint_bytes=size,
                        estimated_full_seconds=elapsed*20*72,estimate_not_measurement=True,peak_RSS_bytes=rss,
                        projected_storage_bytes=projected,local_feasible=elapsed*20*72<12*3600 and rss<40*2**30,
                        storage_sufficient=shutil.disk_usage(PRIVATE).free-projected>cfg['disk_reserve_bytes']))
                    return
                if replay:
                    old=api.core.read_checkpoint(PRIVATE/'heads'/name/'checkpoint.pt.gz')
                    for k in s:
                        if k!='seconds':api.core.exact(s[k],old[k])
                    immutable(PUBLIC/'fit_replay.json',dict(group=name,updates=s['step'],exact_except_elapsed=True,
                        all72_retrained=False,seconds=time.monotonic()-started));return
                doc=dict(identity=identity,checkpoint=base.artifact(cp),step=s['step'],best_step=s['best_step'],
                    best_validation_score=s['best_score'],final_validation_score=s['trace'][-1]['validation_signed_MSE'],
                    partition=s['partition'],trace=s['trace'])
                immutable(docpath,doc)
            refs.append(base.artifact(docpath));partitions.append(dict(group=name,**doc['partition'],best_step=doc['best_step']))
            beat(state='fit_complete',fits=len(refs),group=name,best_step=doc['best_step'])
    assert len(refs)==72
    immutable(PUBLIC/'training_freeze.json',dict(groups=refs,partitions=partitions,unique_fits=72,parameter_updates=144000,
        seconds=time.monotonic()-started,independent_roles_read=False))


def views(data,jobs,oid):
    base.inter.committed(PUBLIC/'training_freeze.json')
    fits={Path(r['path']).parent.name:r for r in json.loads((PUBLIC/'training_freeze.json').read_text())['groups']}
    causal={k:data[k] for k in parent.parent.old.parent.CAUSAL_KEYS}
    for c in base.floor_api.contexts(causal,jobs,oid):
        for pair,(fitting,outer) in enumerate(c['pairs']):
            for site in fitting:
                target=next(s for s in fitting if s!=site)
                assert not {site,target}&(set(c['producer_sites'])|set(c['controller_sites'])|set(outer))
                ref=fits[c['name']+'_fit_'+site];assert base.artifact(ROOT/ref['path'])==ref
                doc=json.loads((ROOT/ref['path']).read_text());cp=doc['checkpoint'];assert base.artifact(ROOT/cp['path'])==cp
                state=api.core.read_checkpoint(ROOT/cp['path']);assert state['step']==2000 and state['identity']==doc['identity']
                at=np.flatnonzero(causal['sites'][c['ids']]==target);ids=c['ids'][at]
                predictions,support=api.paired_predictions(state,c['x'][at],c['env'][at])
                actions=api.decisions(predictions,c['moving'][at],support,causal['recordings'][ids],causal['frames'][ids],ids)
                meta=dict(view=c['name']+f'_pair{pair}_from_'+site,site=target,source=site,seed=state['seed'],
                    ids_hash=base.inter.array_hash(ids),score_hashes={k:base.inter.array_hash(v) for k,v in predictions.items()},
                    action_hashes={k:base.inter.array_hash(v) for k,v in actions.items()},best_step=state['best_step'])
                yield c,at,ids,predictions,actions,state['preprocess'],meta


def decide(cfg,data,jobs,oid):
    start=time.monotonic();rows=[]
    for _,_,_,_,_,_,meta in views(data,jobs,oid):
        rows.append(meta)
        if len(rows)%12==0:beat(state='causal_action_freeze',views=len(rows))
    assert len(rows)==216
    immutable(PUBLIC/'decision_freeze.json',dict(rows=rows,seconds=time.monotonic()-start,causal_only=True,threshold_search=False))


def evaluate(cfg,data,jobs,oid,replay=False):
    base.inter.committed(PUBLIC/'decision_freeze.json');start=time.monotonic()
    frozen={r['view']:r for r in json.loads((PUBLIC/'decision_freeze.json').read_text())['rows']}
    rows=[];quality=[]
    for c,at,ids,p,actions,pr,meta in views(data,jobs,oid):
        assert meta==frozen[meta['view']]
        cv,cf,(floor,ff),(neural,nf)=base.floor_api.costs(c,data,at)
        y=api.core.targets(cv,floor,neural,c['job']['design']['easy_cut']);known=np.isfinite(y).all(1)
        w,_=api.core.weights(data['sites'][ids],data['recordings'][ids],data['frames'][ids],known)
        for arm in ('final','validation'):
            delta=(api.core.signed(p[arm][known])-api.core.signed(y[known]))/pr['scale']/pr['rms'][5:]
            quality.append(dict(view=meta['view'],site=meta['site'],seed=meta['seed'],arm=arm,
                mean_signed_MSE=float((w[known,None]*delta**2).sum()/3)))
        for arm,take in {**actions,'floor':np.zeros(len(ids),bool)}.items():
            metric=base.floor_api.metric(cv,floor,neural,cf,ff,nf,np.where(take,neural,floor),np.where(take,nf,ff),take,
                data['valid'][ids],c['job']['design']['easy_cut'],c['job']['design']['hard_cut'])
            easy=known&(cv>0)&(cv<=c['job']['design']['easy_cut']);chosen=known&take
            ref=float(floor[chosen&easy].sum());harm=np.maximum(neural-floor,0)
            metric['selected_easy_positive_harm_ratio']=float(harm[chosen&easy].sum())/ref if ref>0 else None
            rows.append(dict(view=meta['view'],site=meta['site'],seed=meta['seed'],policy=arm,metric=metric))
        if len(quality)%24==0:beat(state='internal_transfer_readout',views=len(quality)//2)
    assert len(quality)==432 and len(rows)==1080
    immutable(PUBLIC/'readout.json',dict(rows=rows,quality=quality,independent_confirmation=False,deployment_changed=False))
    immutable(PUBLIC/('evaluation_replay.json' if replay else 'evaluation_runtime.json'),dict(exact=replay,
        seconds=time.monotonic()-start,all216_frozen_actions_reverified=True))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['register','pilot','train','replay_fit','decide','evaluate','replay_evaluate'])
    p.add_argument('--resume',action='store_true');a=p.parse_args();PRIVATE.mkdir(parents=True,exist_ok=True)
    cfg,reg=registration()
    if a.phase=='register':immutable(PUBLIC/'registration.json',reg);print('Registered source-only checkpoint selection');return
    assert json.loads((PUBLIC/'registration.json').read_text())==reg;base.inter.committed(PUBLIC/'registration.json')
    api.torch.set_num_threads(cfg['cpu_threads']);api.torch.set_num_interop_threads(cfg['interop_threads'])
    with (PRIVATE/'process.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guard(cfg);beat(state='loading',phase=a.phase)
        _,_,data,jobs,oid,_,_,_=parent.parent.old.load()
        if a.phase in ('pilot','train','replay_fit'):train(cfg,data,jobs,oid,pilot=a.phase=='pilot',resume=a.resume,replay=a.phase=='replay_fit')
        elif a.phase=='decide':decide(cfg,data,jobs,oid)
        else:evaluate(cfg,data,jobs,oid,replay=a.phase=='replay_evaluate')


if __name__=='__main__':main()
