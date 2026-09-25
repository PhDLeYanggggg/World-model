"""Single-factor corrected harm-mass sampling on existing source-only risk inputs."""
import argparse
import fcntl
import json
from pathlib import Path
import os
import shutil
import sys
import time
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import run_m3w_european_selected_risk_learning as parent
from scripts.audit_m3w_easy_harm_training_support import check_parent, PUBLIC, PRIVATE
from src.world_model import m3w_easy_harm_sampling as method
import numpy as np
import torch

artifact, digest, immutable_json, array_hash = parent.artifact, parent.digest, parent.immutable_json, parent.array_hash
CONFIG = 'configs/m3w_european_easy_harm_sampling_v1.json'
CONTROL_KEYS = ('reference','raw_neural','raw_ridge','mean_all','mean_dual','mean_scene','mean_joint',
                'selected_dual','selected_joint')
POLICIES = CONTROL_KEYS+('corrected_all','corrected_dual','corrected_scene','corrected_joint','corrected_hash_matched')
FILES = [CONFIG, 'src/world_model/m3w_easy_harm_sampling.py',
    'scripts/audit_m3w_easy_harm_training_support.py', 'scripts/run_m3w_european_easy_harm_sampling.py',
    'scripts/evaluate_m3w_european_easy_harm_sampling.py', 'tests/test_m3w_easy_harm_sampling.py',
    'outputs/publication_readiness_2026_09/european_easy_harm_sampling_v1/registration.md']


def beat(state, **kw):
    value = dict(pid=os.getpid(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),state=state,**kw)
    parent.base.cross.json_write(PRIVATE/'heartbeat.json',value)
    with (PRIVATE/'events.jsonl').open('a') as f: f.write(json.dumps(value)+'\n')
    print(json.dumps(value),flush=True)


def registration(create=False):
    cfg = json.loads((ROOT/CONFIG).read_text()); identity = check_parent()
    assert cfg['seeds'] == [17,29,43] and cfg['policies'] == list(POLICIES)
    assert (cfg['new_heads'],cfg['updates'],cfg['views'],cfg['harm_mixture']) == (36,72000,504,.5)
    assert cfg['head_training'] == json.loads((ROOT/parent.CONFIG).read_text())['head_training']
    assert not any(cfg[k] for k in ('new_forecaster_training','threshold_refit','selection_access',
        'reserved_calibration_access','confirmation_access','deployment_changed','stage5c_executed','smc_enabled'))
    value = dict(parent=identity, parent_verification=artifact(parent.PUBLIC/'final_verification.json'),
        support_audit=artifact(PUBLIC/'training_support.json'),bindings={f:digest(ROOT/f) for f in FILES})
    path = PUBLIC/'registration_lock.json'
    if create: immutable_json(path,value)
    else:
        assert json.loads(path.read_text()) == value
        parent.previous.require_committed(path)
    return cfg,value


def expectation_check(y, sites, pr, control):
    q,w = method.probabilities(y, sites, pr['weights']); p = pr['weights']
    target = np.where(pr['known'][:,None], y/pr['cost_scale'], 0)
    scale = control['loss_scales']; mean = (p[:,None]*target).sum(0); checks = []
    for offset in (0.,.5,1.):
        prediction = mean+offset*scale; error = (prediction-target)/scale
        loss = (error**2).mean(1); grad = error/(2*scale)
        a,b = np.dot(p,loss),np.dot(q*w,loss)
        ga,gb = (p[:,None]*grad).sum(0), ((q*w)[:,None]*grad).sum(0)
        np.testing.assert_allclose(a,b,rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(ga,gb,rtol=1e-12,atol=1e-12)
        checks.append(dict(offset=offset,loss_absolute_difference=float(abs(a-b)),
            output_gradient_max_absolute_difference=float(abs(ga-gb).max())))
    return dict(checks=checks, sampling_sha256=array_hash(q), ratio_sha256=array_hash(w),
        changed_estimand=False, no_clipping_or_self_normalization=True,
        unbiased_optimizer_updates_claimed=False)


def actions(old, score, utility, moving, env, groups, ids):
    result = {k:old[k] for k in CONTROL_KEYS}
    for mode in ('all','dual','scene','joint'):
        value = parent.method.decisions(utility,score,moving,env,groups,ids,mode)
        np.testing.assert_array_equal(value,parent.method.scalar_decisions(utility,score,moving,env,groups,ids,mode))
        result['corrected_'+mode] = value
    result['corrected_hash_matched'] = parent.method.matched_hash(utility,moving,env,groups,ids,result['corrected_joint'])
    return result


def train(cfg, identity, resume=False, pilot=False):
    heads = []; refs = []
    for g,data,pairs in parent.contexts(identity['parent']):
        name = g['group']; seed = int(name.split('_seed')[1].split('_')[0]); bi,ci = pairs['B']['ids'],pairs['C']['ids']
        queries = parent.query_groups(data['sites'][ci],data['recordings'][ci],data['frames'][ci])
        for pair in cfg['pairs']:
            if shutil.disk_usage(PRIVATE).free < 10*1024**3: raise OSError('10 GiB reserve; keep checkpoints')
            beat('source_pair',group=name,pair=pair)
            bx,be,by,masks,pr,cx,ce,old,_,_,meta = parent.pair_inputs(g,data,pairs,pair)
            old_input = parent.PRIVATE/'inputs'/(name+'_'+pair+'.json')
            assert json.loads(old_input.read_text()) == meta
            old_directory = parent.PRIVATE/'heads'/(name+'_'+pair+'_mean')
            _,control = parent.restore(old_directory)
            check = expectation_check(by,data['sites'][bi],pr,control)
            receipt = PRIVATE/'inputs'/(name+'_'+pair+'.json')
            immutable_json(receipt,dict(parent_inputs=artifact(old_input),expectation=check,
                uniform_control=artifact(old_directory/'complete.json'),C_labels_used_for_fit=False))
            directory = PRIVATE/'heads'/(name+'_'+pair); hid = dict(experiment=identity,input=artifact(receipt),seed=seed)
            if (directory/'complete.json').exists():
                r = json.loads((directory/'complete.json').read_text()); assert r['identity'] == hid
                for a in r['artifacts'].values(): assert artifact(ROOT/a['path']) == a
            else:
                model,fit = method.fit(bx,by,data['sites'][bi],be,masks,pr,seed=seed,settings=cfg['head_training'],
                    identity=hid,directory=directory,resume=resume,stop_at=100 if pilot else None,control=control,
                    heartbeat=lambda **v:beat(group=name,pair=pair,**v))
                if pilot:
                    immutable_json(PRIVATE/'pilot.json',dict(fit=fit,checkpoint=artifact(directory/'checkpoint.pt'),
                        projected_fit_seconds=fit['seconds']/100*cfg['updates'])); return
                score = parent.method.predict(model,cx,ce,pr)
                parent.previous.parent.atomic_npz(directory/'scores.npz',ids=ci,scores=score)
                restored,state = parent.restore(directory)
                np.testing.assert_array_equal(parent.method.predict(restored,cx[:4096],ce[:4096],pr),score[:4096])
                r = dict(identity=hid,fit=fit,result_source='fresh_run',artifacts=dict(
                    checkpoint=artifact(directory/'checkpoint.pt'),scores=artifact(directory/'scores.npz')))
                immutable_json(directory/'complete.json',r)
            heads.append(artifact(directory/'complete.json'))
            with np.load(directory/'scores.npz',allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'],ci); score = z['scores'].copy()
            old_decision = parent.PRIVATE/'decisions'/(name+'_'+pair+'.npz')
            with np.load(old_decision,allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'],ci); previous = {k:z[k].copy() for k in CONTROL_KEYS}
            bits = actions(previous,score,old['neural__utility'],old['moving'],ce,queries,ci)
            path = PRIVATE/'decisions'/(name+'_'+pair+'.npz')
            parent.previous.save_arrays(path,dict(bits,ids=ci))
            rp = path.with_suffix('.json'); immutable_json(rp,dict(identity=identity,group=g,pair=pair,
                input=artifact(receipt),head=heads[-1],array=artifact(path),old_decisions=artifact(old_decision),
                C_outcomes_used_for_fit=False)); refs.append(artifact(rp)); beat('pair_frozen',group=name,pair=pair)
    assert len(heads) == 36 and len(refs) == 36
    immutable_json(PRIVATE/'training_complete.json',dict(identity=identity,heads=heads,decisions=refs,new_updates=72000,all_passed=True))
    immutable_json(PUBLIC/'decision_freeze.json',dict(identity=identity,manifest=artifact(PRIVATE/'training_complete.json'),
        new_heads=36,updates=72000,views=504,source_C_readout=False,selection_access=False,confirmation_access=False))


def checked_training(identity):
    value = json.loads((PRIVATE/'training_complete.json').read_text()); assert value['identity'] == identity and value['all_passed']
    for ref in value['heads']:
        assert artifact(ROOT/ref['path']) == ref
        row = json.loads((ROOT/ref['path']).read_text())
        for a in row['artifacts'].values(): assert artifact(ROOT/a['path']) == a
    for ref in value['decisions']:
        assert artifact(ROOT/ref['path']) == ref
        row = json.loads((ROOT/ref['path']).read_text())
        for k in ('input','head','array','old_decisions'): assert artifact(ROOT/row[k]['path']) == row[k]
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--phase',required=True,choices=['register','pilot','train','evaluate'])
    parser.add_argument('--resume',action='store_true'); args=parser.parse_args()
    PUBLIC.mkdir(parents=True,exist_ok=True); PRIVATE.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    with (PRIVATE/'lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB); cfg,identity=registration(args.phase=='register')
        if args.phase in ('pilot','train'): train(cfg,identity,args.resume,args.phase=='pilot')
        elif args.phase == 'evaluate':
            from scripts.evaluate_m3w_european_easy_harm_sampling import evaluate
            evaluate(sys.modules[__name__],cfg,identity)


if __name__ == '__main__': main()
