"""Fixed-budget source-only reference protection against shared continuation."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import time
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts import run_m3w_european_easy_harm_sampling as prior
from src.world_model import m3w_reference_protection as method
import numpy as np
import torch
parent=prior.parent
PUBLIC=ROOT/'outputs/publication_readiness_2026_09/european_reference_protection_v1'
PRIVATE=ROOT/'data/stage_cvpr2027_experiments/european_reference_protection_v1'
CONFIG='configs/m3w_european_reference_protection_v1.json'
artifact,digest,immutable_json,array_hash=prior.artifact,prior.digest,prior.immutable_json,prior.array_hash
CONTROLS={'reference':'reference','raw_neural':'raw_neural','raw_ridge':'raw_ridge',
    'mean_dual':'mean_dual','mean_joint':'mean_joint','sampling_joint':'corrected_joint'}
NEW_POLICIES=tuple(a+'_'+m for a in method.ARMS for m in ('all','dual','scene','joint','hash_matched'))
POLICIES=tuple(CONTROLS)+NEW_POLICIES
FILES=[CONFIG,'src/world_model/m3w_reference_protection.py','tests/test_m3w_reference_protection.py',
    'scripts/run_m3w_european_reference_protection.py','scripts/evaluate_m3w_european_reference_protection.py',
    'outputs/publication_readiness_2026_09/european_reference_protection_v1/registration.md']


def beat(state,**kw):
    row=dict(pid=os.getpid(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),state=state,**kw)
    parent.base.cross.json_write(PRIVATE/'heartbeat.json',row)
    with (PRIVATE/'events.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)


def registration(create=False):
    cfg=json.loads((ROOT/CONFIG).read_text()); _,pid=prior.registration()
    v=json.loads((prior.PUBLIC/'verification.json').read_text()); assert v['all_passed']
    for f,h in v['artifacts'].items(): assert digest(prior.PUBLIC/f)==h
    for f,h in v['source_bindings'].items(): assert digest(ROOT/f)==h
    transport=json.loads((prior.PUBLIC/'matched_transport_audit.json').read_text())
    assert artifact(ROOT/transport['source_binding']['path'])==transport['source_binding']
    prior.checked_training(pid)
    assert cfg['policies']==list(POLICIES) and cfg['arms']==list(method.ARMS) and cfg['seeds']==[17,29,43]
    assert (cfg['new_heads'],cfg['updates'],cfg['views'])==(72,144000,576)
    assert cfg['head_training']==json.loads((ROOT/prior.CONFIG).read_text())['head_training']
    assert not any(cfg[k] for k in ('new_forecaster_training','threshold_refit','selection_access',
        'reserved_calibration_access','confirmation_access','deployment_changed','stage5c_executed','smc_enabled'))
    identity=dict(parent=pid,prior_verification=artifact(prior.PUBLIC/'verification.json'),
        bindings={f:digest(ROOT/f) for f in FILES})
    path=PUBLIC/'registration_lock.json'
    if create: immutable_json(path,identity)
    else:
        assert json.loads(path.read_text())==identity
        parent.previous.require_committed(path)
    return cfg,identity


def actions(old,moments,utility,moving,env,queries,ids):
    value={k:old[v] for k,v in CONTROLS.items()}
    for arm,score in moments.items():
        for mode in ('all','dual','scene','joint'):
            bits=parent.method.decisions(utility,score,moving,env,queries,ids,mode)
            np.testing.assert_array_equal(bits,parent.method.scalar_decisions(utility,score,moving,env,queries,ids,mode))
            value[arm+'_'+mode]=bits
        value[arm+'_hash_matched']=parent.method.matched_hash(utility,moving,env,queries,ids,value[arm+'_joint'])
    assert set(value)==set(POLICIES)
    return value


def train(cfg,identity,resume=False,pilot=False):
    heads=[]; decisions=[]
    for g,data,pairs in parent.contexts(identity['parent']['parent']):
        name=g['group']; seed=int(name.split('_seed')[1].split('_')[0]); bi,ci=pairs['B']['ids'],pairs['C']['ids']
        queries=parent.query_groups(data['sites'][ci],data['recordings'][ci],data['frames'][ci])
        for pair in cfg['pairs']:
            if shutil.disk_usage(PRIVATE).free<10*1024**3: raise OSError('10 GiB reserve; keep checkpoint')
            bx,be,by,masks,pr,cx,ce,old,*_,meta=parent.pair_inputs(g,data,pairs,pair)
            previous_input=parent.PRIVATE/'inputs'/(name+'_'+pair+'.json')
            assert json.loads(previous_input.read_text())==meta
            warm=parent.PRIVATE/'heads'/(name+'_'+pair+'_mean')
            initial,control=parent.restore(warm)
            input_path=PRIVATE/'inputs'/(name+'_'+pair+'.json')
            immutable_json(input_path,dict(parent_input=artifact(previous_input),warm_start=artifact(warm/'checkpoint.pt'),
                initial_sampler_rng_sha256=array_hash(control['sampler_rng'].numpy()),
                C_labels_used_for_fit=False,optimizer_reset_both_arms=True))
            scores={}; states={}
            for arm in cfg['arms']:
                beat('source_arm',group=name,pair=pair,arm=arm)
                directory=PRIVATE/'heads'/(name+'_'+pair+'_'+arm)
                hid=dict(experiment=identity,input=artifact(input_path),arm=arm,seed=seed)
                if (directory/'complete.json').exists():
                    receipt=json.loads((directory/'complete.json').read_text()); assert receipt['identity']==hid
                    for r in receipt['artifacts'].values(): assert artifact(ROOT/r['path'])==r
                else:
                    model,fit=method.fit(bx,by,data['sites'][bi],be,pr,initial,control,
                        arm=arm,seed=seed,settings=cfg['head_training'],identity=hid,directory=directory,
                        heartbeat=lambda **v:beat(group=name,pair=pair,arm=arm,**v),resume=resume,
                        stop_at=100 if pilot else None)
                    if pilot:
                        immutable_json(PRIVATE/'pilot.json',dict(fit=fit,checkpoint=artifact(directory/'checkpoint.pt'),
                            projected_fit_seconds=fit['seconds']/100*cfg['updates'])); return
                    score=parent.method.predict(model,cx,ce,pr)
                    if arm=='protected':
                        with np.load(warm/'scores.npz',allow_pickle=False) as z:
                            np.testing.assert_array_equal(z['ids'],ci)
                            np.testing.assert_array_equal(score[:,[0,2]],z['scores'][:,[0,2]])
                    parent.previous.parent.atomic_npz(directory/'scores.npz',ids=ci,scores=score)
                    restored,state=method.restore(directory)
                    np.testing.assert_array_equal(parent.method.predict(restored,cx[:4096],ce[:4096],pr),score[:4096])
                    receipt=dict(identity=hid,fit=fit,result_source='fresh_run_warm_start_continuation',
                        artifacts=dict(checkpoint=artifact(directory/'checkpoint.pt'),scores=artifact(directory/'scores.npz')))
                    immutable_json(directory/'complete.json',receipt)
                heads.append(artifact(directory/'complete.json'))
                _,states[arm]=method.restore(directory)
                with np.load(directory/'scores.npz',allow_pickle=False) as z:
                    np.testing.assert_array_equal(z['ids'],ci); scores[arm]=z['scores'].copy()
            np.testing.assert_array_equal(states['continued']['draws'],states['protected']['draws'])
            assert torch.equal(states['continued']['sampler_rng'],states['protected']['sampler_rng'])
            previous_decision=prior.PRIVATE/'decisions'/(name+'_'+pair+'.npz')
            with np.load(previous_decision,allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'],ci); previous={v:z[v].copy() for v in CONTROLS.values()}
            bits=actions(previous,scores,old['neural__utility'],old['moving'],ce,queries,ci)
            path=PRIVATE/'decisions'/(name+'_'+pair+'.npz'); parent.previous.save_arrays(path,dict(bits,ids=ci))
            receipt=path.with_suffix('.json')
            immutable_json(receipt,dict(identity=identity,group=g,pair=pair,input=artifact(input_path),
                heads=heads[-2:],array=artifact(path),old_decisions=artifact(previous_decision),C_outcomes_used=False))
            decisions.append(artifact(receipt)); beat('pair_frozen',group=name,pair=pair)
    assert len(heads)==72 and len(decisions)==36
    immutable_json(PRIVATE/'training_complete.json',dict(identity=identity,heads=heads,decisions=decisions,all_passed=True,updates=144000))
    immutable_json(PUBLIC/'decision_freeze.json',dict(identity=identity,manifest=artifact(PRIVATE/'training_complete.json'),
        heads=72,updates=144000,views=576,C_outcomes_opened=False,selection_access=False,confirmation_access=False))


def checked_training(identity):
    value=json.loads((PRIVATE/'training_complete.json').read_text()); assert value['identity']==identity and value['all_passed']
    for ref in value['heads']:
        assert artifact(ROOT/ref['path'])==ref
        receipt=json.loads((ROOT/ref['path']).read_text())
        for r in receipt['artifacts'].values(): assert artifact(ROOT/r['path'])==r
    for ref in value['decisions']:
        assert artifact(ROOT/ref['path'])==ref
        receipt=json.loads((ROOT/ref['path']).read_text())
        for k in ('input','array','old_decisions'): assert artifact(ROOT/receipt[k]['path'])==receipt[k]
    return value


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--phase',required=True,choices=['register','pilot','train','evaluate'])
    parser.add_argument('--resume',action='store_true'); args=parser.parse_args()
    PUBLIC.mkdir(parents=True,exist_ok=True); PRIVATE.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    with (PRIVATE/'lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); cfg,identity=registration(args.phase=='register')
        if args.phase in ('pilot','train'): train(cfg,identity,args.resume,args.phase=='pilot')
        elif args.phase=='evaluate':
            from scripts.evaluate_m3w_european_reference_protection import evaluate
            evaluate(sys.modules[__name__],cfg,identity)


if __name__=='__main__': main()
