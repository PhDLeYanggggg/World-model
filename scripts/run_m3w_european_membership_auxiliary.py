"""Matched original nested cost control versus auxiliary membership supervision."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
if platform.system()=='Darwin' and platform.machine()!='arm64':
    raise RuntimeError('Use native .venv-pytorch arm64, not Rosetta/Intel OpenMP')
from scripts import run_m3w_european_cost_attribution as attribution
from src.world_model import m3w_membership_auxiliary as method
import numpy as np
import torch
parent=attribution.parent; base,tail=parent.base,parent.tail
PUBLIC=ROOT/'outputs/publication_readiness_2026_09/european_membership_auxiliary_v1'
PRIVATE=ROOT/'data/stage_cvpr2027_experiments/european_membership_auxiliary_v1'
CONFIG='configs/m3w_european_membership_auxiliary_v1.json'
artifact,digest,immutable_json,array_hash=parent.artifact,parent.digest,parent.immutable_json,parent.array_hash
FILES=[CONFIG,'src/world_model/m3w_membership_auxiliary.py','tests/test_m3w_membership_auxiliary.py',
    'scripts/run_m3w_european_membership_auxiliary.py','scripts/report_m3w_european_membership_auxiliary.py',
    'tests/test_m3w_membership_auxiliary_reporting.py',str(PUBLIC.relative_to(ROOT)/'registration.md')]


def beat(state,**kw):
    row=dict(pid=os.getpid(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),state=state,**kw)
    base.base.cross.json_write(PRIVATE/'heartbeat.json',row)
    with (PRIVATE/'events.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)


def registration(create=False):
    cfg=json.loads((ROOT/CONFIG).read_text()); pid=attribution.registration()
    v=json.loads((attribution.PUBLIC/'verification.json').read_text()); assert v['all_passed']
    for f,h in v['artifacts'].items(): assert digest(attribution.PUBLIC/f)==h
    for f,h in v['source_bindings'].items(): assert digest(ROOT/f)==h
    assert (cfg['new_heads'],cfg['updates'],cfg['auxiliary_coefficient'])==(288,576000,1.)
    assert cfg['head_training']==json.loads((ROOT/parent.CONFIG).read_text())['head_training']
    assert not any(cfg[k] for k in ('new_policy_evaluation','threshold_refit','selection_access',
        'reserved_calibration_access','confirmation_access','deployment_changed','stage5c_executed','smc_enabled'))
    identity=dict(source=pid['parent']['source'],parent_verification=artifact(attribution.PUBLIC/'verification.json'),
        bindings={f:digest(ROOT/f) for f in FILES})
    path=PUBLIC/'registration_lock.json'
    if create: immutable_json(path,identity)
    else:
        assert json.loads(path.read_text())==identity; base.previous.require_committed(path)
    return cfg,identity


def train(cfg,identity,resume=False,pilot=False):
    heads=[]
    for g,data,pairs in base.contexts(identity['source']):
        name=g['group']; seed=int(name.split('_seed')[1].split('_')[0]); bi=pairs['B']['ids']; sites=data['sites'][bi]; cv=data['baseline_ade'][bi,1]
        for pair in cfg['pairs']:
            x,env,by,*_=base.pair_inputs(g,data,pairs,pair)
            for held in sorted(set(sites)):
                if shutil.disk_usage(PRIVATE).free<10*2**30: raise OSError('10 GiB reserve; keep resumable checkpoints')
                tr,te,pr,cut,y,_=tail.fold_inputs(x,env,by,cv,sites,held); easy=parent.parent.method.labels(cv[tr],cut)
                assert held not in pr['training_sites'] and held not in g['producer_roster']
                tag=name+'_'+pair+'_'+held; tref,href,oldstate,olddir=parent.reference(tag,x,tr,te,pr,env,bi)
                for arm in cfg['arms']:
                    inp=PRIVATE/'inputs'/(tag+'_'+arm+'.json')
                    inputs=dict(group=g,pair=pair,held=held,arm=arm,easy_cut=cut,training_sites=pr['training_sites'],
                        train_x_sha256=array_hash(x[tr]),held_x_sha256=array_hash(x[te]),train_y_sha256=array_hash(y),
                        train_easy_sha256=array_hash(easy),train_ids_sha256=array_hash(bi[tr]),held_ids_sha256=array_hash(bi[te]),
                        original_mean=artifact(olddir/'complete.json'),held_targets_used_for_fit=False,
                        membership_probability_multiplied_into_cost=False)
                    immutable_json(inp,inputs); hid=dict(experiment=identity,input=artifact(inp),seed=seed)
                    directory=PRIVATE/'heads'/tag/arm; receipt=directory/'complete.json'
                    if receipt.exists():
                        r=json.loads(receipt.read_text()); assert r['identity']==hid
                        for a in r['artifacts'].values(): assert artifact(ROOT/a['path'])==a
                    else:
                        beat('fit',group=name,pair=pair,held=held,arm=arm)
                        model,fit=method.fit(x[tr],y,easy,sites[tr],env[tr],pr,arm=arm,seed=seed,settings=cfg['head_training'],
                            identity=hid,directory=directory,resume=resume,stop_at=100 if pilot else None,
                            heartbeat=lambda **kw:beat(group=name,pair=pair,held=held,arm=arm,**kw))
                        if pilot:
                            immutable_json(PRIVATE/'pilot.json',dict(fit=fit,checkpoint=artifact(directory/'checkpoint.pt'),
                                projected_cost_only_fit_seconds=fit['seconds']/100*cfg['updates'],auxiliary_cost_not_yet_measured=True)); return
                        tf,tp=method.predict(model,x[tr],env[tr],pr); hf,hp=method.predict(model,x[te],env[te],pr)
                        train_scores=parent.compose(tf[:,[1,3]],tref); score=parent.compose(hf[:,[1,3]],href)
                        edges=tail.training_edges(train_scores,env[tr],pr,cfg)
                        diag=dict(edges=edges,training=tail.diagnostic.summarize(train_scores,y,env[tr],sites[tr],edges))
                        base.previous.parent.atomic_npz(directory/'scores.npz',ids=bi[te],scores=score,membership=hp)
                        restored,state=method.restore(directory)
                        replay,rp=method.predict(restored,x[te][:4096],env[te][:4096],pr)
                        np.testing.assert_array_equal(replay[:,[1,3]],score[:4096][:,[1,3]]); np.testing.assert_array_equal(rp,hp[:4096])
                        for k in ('draws','fixed_ids','loss_scales'): np.testing.assert_array_equal(state[k],oldstate[k])
                        assert torch.equal(state['sampler_rng'],oldstate['sampler_rng'])
                        immutable_json(directory/'fit_diagnosis.json',diag)
                        r=dict(identity=hid,input=inputs,fit=fit,result_source='fresh_run_auxiliary_head_cached_verified_producers_and_reference',
                            artifacts=dict(checkpoint=artifact(directory/'checkpoint.pt'),scores=artifact(directory/'scores.npz'),diagnosis=artifact(directory/'fit_diagnosis.json')))
                        immutable_json(receipt,r)
                    heads.append(artifact(receipt)); beat('head_frozen',completed=len(heads),group=name,pair=pair,held=held,arm=arm)
    assert len(heads)==288
    immutable_json(PRIVATE/'training_complete.json',dict(identity=identity,heads=heads,updates=576000,all_passed=True))
    immutable_json(PUBLIC/'prediction_freeze.json',dict(identity=identity,manifest=artifact(PRIVATE/'training_complete.json'),heads=288,
        updates=576000,current_outcome_readout=False,policy_changed=False))


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
                tr,te,pr,cut,y,_=tail.fold_inputs(x,env,by,cv,sites,held)
                target=tail.diagnostic.event_targets(by[te],cv[te],cut); tag=name+'_'+pair+'_'+held
                olddir=parent.parent.parent.sources(tag)['mean_features']
                homes={'original_mean':olddir,'conditional':parent.PRIVATE/'heads'/tag/'conditional',
                       **{a:PRIVATE/'heads'/tag/a for a in cfg['arms']}}
                scores={}; fits={}; metrics={}; sources={}; membership={}
                for arm,directory in homes.items():
                    sources[arm]=artifact(directory/'complete.json'); r=json.loads((directory/'complete.json').read_text()); assert r['input']['easy_cut']==cut
                    with np.load(directory/'scores.npz',allow_pickle=False) as z:
                        np.testing.assert_array_equal(z['ids'],bi[te]); scores[arm]=z['scores'].copy()
                        if arm in cfg['arms']:
                            membership[arm]=parent.parent.measure(z['membership'],parent.parent.method.labels(cv[te],cut),env[te],sites[te])
                    fits[arm]=json.loads((directory/'fit_diagnosis.json').read_text())
                for arm in scores:
                    np.testing.assert_array_equal(scores[arm][:,[0,2]],scores['original_mean'][:,[0,2]])
                    metrics[arm]=parent.measure(scores[arm],target,env[te],sites[te],fits[arm]['edges'])
                folds.append(dict(held=held,easy_cut=cut,heads=sources,metrics=metrics,training=fits,membership=membership,
                    control_prediction_max_abs_error=float(np.max(np.abs(scores['cost_only']-scores['original_mean']))),
                    target_sha256=array_hash(target),held_ids_sha256=array_hash(bi[te])))
            row=dict(group=name,producer=g['producer'],controller=g['controller'],seed=int(name.split('_seed')[1].split('_')[0]),pair=pair,folds=folds,
                result_source='fresh_run_auxiliary_cost_metrics_cached_verified_reference',policy_changed=False)
            immutable_json(path,row); immutable_json(receipt,artifact(path)); refs.append(artifact(path)); beat('readout',completed=len(refs),group=name,pair=pair)
    assert len(refs)==36
    immutable_json(PUBLIC/'completion_checks.json',dict(identity=identity,groups=refs,all_passed=True))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--phase',required=True,choices=['register','pilot','train','evaluate']); ap.add_argument('--resume',action='store_true'); args=ap.parse_args()
    PUBLIC.mkdir(parents=True,exist_ok=True); PRIVATE.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    with (PRIVATE/'lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); cfg,identity=registration(args.phase=='register')
        if args.phase in ('train','pilot'): train(cfg,identity,args.resume,args.phase=='pilot')
        elif args.phase=='evaluate': evaluate(cfg,identity)


if __name__=='__main__': main()
