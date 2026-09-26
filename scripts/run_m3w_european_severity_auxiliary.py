"""Matched severity-weighted auxiliary task with immutable source roles."""
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
    raise RuntimeError('Use native arm64 .venv-pytorch before importing Torch')
from scripts import run_m3w_european_task_gradients as diagnosis
from src.world_model import m3w_severity_auxiliary as method
import numpy as np
import torch
parent=diagnosis.parent; base,tail=parent.base,parent.tail
PUBLIC=ROOT/'outputs/publication_readiness_2026_09/european_severity_auxiliary_v1'
PRIVATE=ROOT/'data/stage_cvpr2027_experiments/european_severity_auxiliary_v1'
CONFIG='configs/m3w_european_severity_auxiliary_v1.json'
artifact,digest,immutable_json,array_hash=parent.artifact,parent.digest,parent.immutable_json,parent.array_hash
FILES=[CONFIG,'src/world_model/m3w_severity_auxiliary.py','tests/test_m3w_severity_auxiliary.py',
    'scripts/run_m3w_european_severity_auxiliary.py','scripts/report_m3w_european_severity_auxiliary.py',
    'tests/test_m3w_severity_auxiliary_reporting.py',str(PUBLIC.relative_to(ROOT)/'registration.md')]


def registration(create=False):
    cfg=json.loads((ROOT/CONFIG).read_text()); _,pid=diagnosis.registration()
    v=json.loads((diagnosis.PUBLIC/'verification.json').read_text()); assert v['all_passed']
    for p,h in v['artifacts'].items(): assert digest(diagnosis.PUBLIC/p)==h
    for p,h in v['source_bindings'].items(): assert digest(ROOT/p)==h
    assert (cfg['new_heads'],cfg['updates'],cfg['auxiliary_coefficient'])==(144,288000,1.)
    assert cfg['head_training']==json.loads((ROOT/parent.CONFIG).read_text())['head_training']
    assert not any(cfg[k] for k in ('new_policy_evaluation','threshold_refit','selection_access',
        'reserved_calibration_access','confirmation_access','deployment_changed','stage5c_executed','smc_enabled'))
    identity=dict(parent=pid,source=pid['parent']['source'],parent_verification=artifact(diagnosis.PUBLIC/'verification.json'),
        bindings={p:digest(ROOT/p) for p in FILES})
    path=PUBLIC/'registration_lock.json'
    if create: immutable_json(path,identity)
    else:
        assert json.loads(path.read_text())==identity; base.previous.require_committed(path)
    return cfg,identity


def beat(state,**kw):
    row=dict(pid=os.getpid(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),state=state,**kw)
    base.base.cross.json_write(PRIVATE/'heartbeat.json',row)
    with (PRIVATE/'events.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)


def support(cfg,identity):
    rows=[]
    for g,data,pairs in base.contexts(identity['source']):
        for pair in cfg['pairs']:
            for held in sorted(set(data['sites'][pairs['B']['ids']])):
                x,env,y,e,sites,ids,pr=diagnosis.fitting_inputs(g,data,pairs,pair,held)
                tag=g['group']+'_'+pair+'_'+held
                inp=json.loads((parent.PRIVATE/'heads'/tag/'membership_aux'/'complete.json').read_text())['input']
                for key,a in (('train_x_sha256',x),('train_y_sha256',y),('train_easy_sha256',e),('train_ids_sha256',ids)):
                    assert inp[key]==array_hash(a)
                s=method.support(y,e,pr,sites,data['recordings'][ids],data['agents'][ids])
                rows.append(dict(group=g['group'],pair=pair,held=held,support=s,train_ids_sha256=array_hash(ids),
                    train_targets_sha256=array_hash(y),result_source='fresh_run_fitting_only_support_cached_verified_labels'))
            beat('support',views=len(rows),group=g['group'],pair=pair)
    assert len(rows)==144
    allowed=all(r['support']['mean_identifiable'] for r in rows) and all(r['support']['two_sided_identifiable'] for r in rows if r['pair']=='full')
    immutable_json(PUBLIC/'support_report.json',dict(identity=identity,rows=rows,training_allowed=allowed,
        statistical_power_established=False,held_labels_used=False))
    lines=['# Fitting-Only Severity Support','','Known labels only; no held outcomes used. Mass concentration is not independent sample size.',
        'Training numerically allowed: '+str(allowed), '', '| Pair | Dependent views | Weak-support flags | Minimum easy-harm tracks | Minimum easy-harm localities | Median harm-weighted easy fraction |',
        '|---|---:|---:|---:|---:|---:|']
    for pair in cfg['pairs']:
        rs=[r['support'] for r in rows if r['pair']==pair]
        lines.append(f"| {pair} | {len(rs)} | {sum(r['weak_support'] for r in rs)} | {min(r['strata']['easy']['tracks']['positive_groups'] for r in rs)} | {min(r['strata']['easy']['localities']['positive_groups'] for r in rs)} | {np.median([r['harm_weighted_easy_fraction'] for r in rs])} |")
    lines+=['','No data-role, window, primary endpoint or risk-tolerance changes. Weak flags do not license selective fold removal.']
    (PUBLIC/'support_report.md').write_text('\n'.join(lines)+'\n'); print(json.dumps(dict(training_allowed=allowed,views=len(rows))))


def train(cfg,identity,resume=False,pilot=False):
    support_doc=json.loads((PUBLIC/'support_report.json').read_text()); assert support_doc['identity']==identity and support_doc['training_allowed']
    base.previous.require_committed(PUBLIC/'support_report.json'); heads=[]
    for g,data,pairs in base.contexts(identity['source']):
        name=g['group']; seed=int(name.split('_seed')[1].split('_')[0]); bi=pairs['B']['ids']; sites=data['sites'][bi]
        for pair in cfg['pairs']:
            bx,be,*_=pairs['B'][pair]
            for held in sorted(set(sites)):
                if shutil.disk_usage(PRIVATE).free<10*2**30: raise OSError('Preserve10GiB and resumable checkpoints')
                x,env,y,e,ss,ids,pr=diagnosis.fitting_inputs(g,data,pairs,pair,held)
                tr,te=sites!=held,sites==held; tag=name+'_'+pair+'_'+held
                tref,href,oldstate,olddir=parent.parent.reference(tag,bx,tr,te,pr,be,bi)
                inputs=dict(group=g,pair=pair,held=held,arm='severity_aux',easy_cut=pr['positive_easy_cut'],training_sites=pr['training_sites'],
                    train_x_sha256=array_hash(x),held_x_sha256=array_hash(bx[te]),train_y_sha256=array_hash(y),
                    train_easy_sha256=array_hash(e),train_ids_sha256=array_hash(ids),held_ids_sha256=array_hash(bi[te]),
                    original_mean=artifact(olddir/'complete.json'),held_targets_used_for_fit=False,
                    membership_probability_multiplied_into_cost=False,loss_weights_are_detached_fitting_labels=True)
                inp=PRIVATE/'inputs'/(tag+'.json'); immutable_json(inp,inputs)
                hid=dict(experiment=identity,input=artifact(inp),seed=seed)
                directory=PRIVATE/'heads'/tag; receipt=directory/'complete.json'
                if receipt.exists():
                    r=json.loads(receipt.read_text()); assert r['identity']==hid
                    for a in r['artifacts'].values(): assert artifact(ROOT/a['path'])==a
                else:
                    beat('fit',group=name,pair=pair,held=held)
                    model,fit=method.fit(x,y,e,ss,env,pr,seed=seed,settings=cfg['head_training'],identity=hid,
                        directory=directory,resume=resume,stop_at=100 if pilot else None,
                        heartbeat=lambda **kw:beat(group=name,pair=pair,held=held,**kw))
                    if pilot:
                        immutable_json(PRIVATE/'pilot.json',dict(fit=fit,checkpoint=artifact(directory/'checkpoint.pt'),
                            projected_fit_seconds=fit['seconds']/100*cfg['updates'],excludes_loading_inference_verification=True)); return
                    tf,tp=method.predict(model,x,env,pr); hf,hp=method.predict(model,bx[te],be[te],pr)
                    train_scores=parent.parent.compose(tf[:,[1,3]],tref); score=parent.parent.compose(hf[:,[1,3]],href)
                    edges=tail.training_edges(train_scores,env,pr,cfg)
                    diag=dict(edges=edges,training=tail.diagnostic.summarize(train_scores,y,env,ss,edges))
                    base.previous.parent.atomic_npz(directory/'scores.npz',ids=bi[te],scores=score,weighted_membership=hp)
                    restored,state=method.restore(directory); replay,rp=method.predict(restored,bx[te][:4096],be[te][:4096],pr)
                    np.testing.assert_array_equal(replay[:,[1,3]],score[:4096][:,[1,3]]); np.testing.assert_array_equal(rp,hp[:4096])
                    for k in ('draws','fixed_ids','loss_scales'): np.testing.assert_array_equal(state[k],oldstate[k])
                    assert torch.equal(state['sampler_rng'],oldstate['sampler_rng'])
                    immutable_json(directory/'fit_diagnosis.json',diag)
                    r=dict(identity=hid,input=inputs,fit=fit,result_source='fresh_run_severity_head_cached_verified_producers_and_reference',
                        artifacts=dict(checkpoint=artifact(directory/'checkpoint.pt'),scores=artifact(directory/'scores.npz'),diagnosis=artifact(directory/'fit_diagnosis.json')))
                    immutable_json(receipt,r)
                heads.append(artifact(receipt)); beat('head_frozen',completed=len(heads),group=name,pair=pair,held=held)
    assert len(heads)==144
    immutable_json(PRIVATE/'training_complete.json',dict(identity=identity,heads=heads,updates=288000,all_passed=True))
    immutable_json(PUBLIC/'prediction_freeze.json',dict(identity=identity,manifest=artifact(PRIVATE/'training_complete.json'),
        heads=144,updates=288000,current_outcome_readout=False,policy_changed=False))


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
            x,env,by,*_=base.pair_inputs(g,data,pairs,pair)
            previous=json.loads((parent.PUBLIC/'groups'/path.name).read_text()); folds=[]
            for held in sorted(set(sites)):
                tr,te,pr,cut,y,_=tail.fold_inputs(x,env,by,cv,sites,held); tag=name+'_'+pair+'_'+held
                target=tail.diagnostic.event_targets(by[te],cv[te],cut)
                old=next(f for f in previous['folds'] if f['held']==held)
                assert old['target_sha256']==array_hash(target) and old['held_ids_sha256']==array_hash(bi[te])
                metrics={a:old['metrics'][a] for a in ('original_mean','cost_only','membership_aux')}
                fits={a:old['training'][a] for a in metrics}; heads={a:old['heads'][a] for a in metrics}
                directory=PRIVATE/'heads'/tag; r=json.loads((directory/'complete.json').read_text())
                assert r['input']['easy_cut']==cut
                with np.load(directory/'scores.npz',allow_pickle=False) as z:
                    np.testing.assert_array_equal(z['ids'],bi[te]); score=z['scores'].copy()
                refdir=parent.parent.parent.parent.sources(tag)['mean_features']
                with np.load(refdir/'scores.npz',allow_pickle=False) as z:
                    np.testing.assert_array_equal(z['ids'],bi[te]); np.testing.assert_array_equal(score[:,[0,2]],z['scores'][:,[0,2]])
                fits['severity_aux']=json.loads((directory/'fit_diagnosis.json').read_text())
                metrics['severity_aux']=parent.parent.measure(score,target,env[te],sites[te],fits['severity_aux']['edges'])
                heads['severity_aux']=artifact(directory/'complete.json')
                folds.append(dict(held=held,easy_cut=cut,heads=heads,metrics=metrics,training=fits,
                    target_sha256=array_hash(target),held_ids_sha256=array_hash(bi[te])))
            row=dict(group=name,producer=g['producer'],controller=g['controller'],seed=int(name.split('_seed')[1].split('_')[0]),
                pair=pair,folds=folds,result_source='fresh_run_severity_cost_readout_cached_verified_controls',policy_changed=False)
            immutable_json(path,row); immutable_json(receipt,artifact(path)); refs.append(artifact(path)); beat('readout',completed=len(refs),group=name,pair=pair)
    assert len(refs)==36
    immutable_json(PUBLIC/'completion_checks.json',dict(identity=identity,groups=refs,all_passed=True))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--phase',required=True,choices=['register','support','pilot','train','evaluate']); ap.add_argument('--resume',action='store_true'); args=ap.parse_args()
    PUBLIC.mkdir(parents=True,exist_ok=True); PRIVATE.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    with (PRIVATE/'lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); cfg,identity=registration(args.phase=='register')
        if args.phase=='support': support(cfg,identity)
        elif args.phase in ('pilot','train'): train(cfg,identity,args.resume,args.phase=='pilot')
        elif args.phase=='evaluate': evaluate(cfg,identity)


if __name__=='__main__': main()
