"""Frozen six-arm inference after complete paired TRAIN and explicit replay audit."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Use the native arm64 environment before numerical imports')

import joblib
import numpy as np

from scripts import run_m3w_temporal_auxiliary_readout as legacy
from scripts import read_m3w_easy_harm_final_heads as heads
from scripts import verify_m3w_easy_harm_readout as scalar
from src.world_model import m3w_easy_harm_deviance_readout as api
from src.world_model import m3w_easy_harm_deviance_training as fitting

ROOT = heads.ROOT
PUBLIC = heads.manager.PUBLIC/'readout'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/european_easy_harm_deviance_v1/readout'
sha, once, parent, controls = legacy.sha, legacy.once, legacy.parent, legacy.controls


def registration():
    calculation = json.loads((PUBLIC/'calculation_registration.json').read_text())
    for rel,digest in calculation['bindings'].items():
        if sha(ROOT/rel) != digest: raise ValueError('Calculation contract changed')
    cfg = json.loads(heads.manager.CONFIG.read_text())
    train = json.loads(heads.manager.REG.read_text())
    files = parent.forest.closure(ROOT,['scripts.run_m3w_easy_harm_readout'])
    files += [PUBLIC/'calculation_protocol.md',ROOT/'tests/test_m3w_easy_harm_readout_runner.py',ROOT/'tests/test_m3w_easy_harm_final_heads.py',
              ROOT/'tests/test_m3w_easy_harm_deviance_readout.py']
    result = dict(bindings={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(files))},
        training_registration_sha256=sha(heads.manager.REG),
        execution_amendment_sha256=sha(heads.manager.PUBLIC/'control_execution_amendment.json'),
        calculation_registration_sha256=sha(PUBLIC/'calculation_registration.json'),
        controls_registration_sha256=sha(controls.PUBLIC/'registration.json'),
        expected=[dict(group=r['group'],source=r['source'],head_seed=r['seed']) for r in train['expected']],
        arms=list(api.ARMS),new_validation_predictions=False,independent_roles_read=False,
        checkpoint_selection=False,threshold_search=False,risk_budget=.02)
    return cfg,result


def beat(**value):
    row = dict(pid=os.getpid(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**value)
    tmp = PRIVATE/'heartbeat.tmp';tmp.write_text(json.dumps(row)+'\n');os.replace(tmp,PRIVATE/'heartbeat.json')
    with (PRIVATE/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)


def run(cfg, reg, documents, inventory, resume):
    started = time.monotonic(); rows,refs = [],[]; checks = 0
    source_docs = parent.parent.docs()
    cost_docs = controls.read_rows(controls.diagnostic.fitted_run.PUBLIC)
    poisson_docs = controls.read_rows(controls.prior.PUBLIC)
    additive_docs = controls.prior.prior_rows()
    with heads.Reader(inventory) as neural, controls.FrozenReader(controls.diagnostic.SERVER) as cr, \
         controls.FrozenReader(controls.reader.SERVER) as pr, controls.FrozenReader(controls.prior.previous.SERVER) as ar:
        beat(state='owned_final_heads_and_strong_controls_connected')
        _,_,data,jobs,oid,_,_,_ = parent.inner.old.load()
        quality,provenance = controls.prior.prior.past_quality(data)
        assert provenance == json.loads((controls.prior.prior.PUBLIC/'feature_provenance.json').read_text())
        for context in parent.parent.contexts(data,jobs,oid):
            for site in parent.inner.sources(context):
                group = context['name']+'_fit_'+site
                at,ids,x,env,y,_,upstream = parent.inner.training_arrays(context,data,site)
                tr,val,partition = parent.forest.parent.api.source_partition(data['recordings'][ids],data['frames'][ids],site)
                tid,vid = ids[tr],ids[val]
                assert not set(data['recordings'][tid]) & set(data['recordings'][vid])
                temporal,_ = legacy.training.previous.api.targets(context['floor'][at][tr].astype(float)+data['origin'][tid,None],
                    context['prediction'][at][tr].astype(float)+data['origin'][tid,None],data['target_eval'][tid],data['valid'][tid])
                input_hashes = {k:api.previous.forest.fingerprint(np.asarray(v)) for k,v in dict(x=x[tr],envelope=env[tr],
                    primary=y[tr],temporal=temporal,sites=data['sites'][tid],recordings=data['recordings'][tid],frames=data['frames'][tid]).items()}
                preprocessing = api.previous.forest.core.preprocess(x[tr],env[tr],y[tr],data['sites'][tid],
                    data['recordings'][tid],data['frames'][tid],training_site=site)
                for seed in cfg['head_seeds']:
                    if time.monotonic()-started > cfg['hard_runtime_limit_seconds']:
                        raise TimeoutError('12-hour readout limit; exact group outputs remain resumable')
                    beat(state='group_readout',group=group,head_seed=seed)
                    source,anchor = source_docs[group,seed],cost_docs[group,seed]
                    assert sha(ROOT/source['checkpoint']['path']) == source['checkpoint']['sha256']
                    state = joblib.load(ROOT/source['checkpoint']['path'])
                    assert state['identity']['upstream'] == upstream
                    fitting.core.exact(preprocessing,state['preprocess'])
                    trained=[]
                    for arm in fitting.ARMS:
                        doc=documents[group,site,seed,arm]; model=neural.load(ROOT/doc['checkpoint']['path'])
                        identity=dict(group=group,source=site,seed=seed,upstream=upstream,partition=partition,
                            training_ids_sha256=api.previous.forest.fingerprint(tid),source_checkpoint=source['checkpoint'],
                            registration_sha256=sha(legacy.training.PUBLIC/'registration.json'))
                        for k,v in dict(identity=identity,step=2000,settings=cfg['head_training'],preprocess=preprocessing,
                            arm=arm,seed=seed,input_hashes=input_hashes,
                            experiment_sha256=sha(heads.manager.REG)).items(): fitting.core.exact(model[k],v)
                        assert doc['input_hashes'] == input_hashes and doc['draw_hash'] == model['draw_hash']
                        trained.append(model)
                    fitting.assert_matched(*trained)
                    pred,support = legacy.frozen_controls(state,source,anchor,poisson_docs[group,seed],additive_docs[group,seed],
                        (cr,pr,ar),site,seed,partition,tid,vid,x[val],env[val],y[val],quality)
                    moving=context['moving'][at][val]
                    previous,old_actions=controls.api.diagnostic.model.evaluate(state=state,predictions=pred,targets=y[val],
                        envelope=env[val],moving=moving,support=support,sites=data['sites'][vid],recordings=data['recordings'][vid],
                        frames=data['frames'][vid],ids=vid)
                    assert previous == anchor['result']
                    h=parent.base.inter.array_hash
                    for k,a in old_actions.items(): assert h(a)==anchor['action_hashes'][k]
                    for arm,model in zip(fitting.ARMS,trained):
                        pred[arm],_,sp=fitting.parent.predict(model,x[val],env[val])
                        again,_,sr=fitting.parent.predict(model,x[val],env[val])
                        np.testing.assert_array_equal(sp,support);np.testing.assert_array_equal(sr,support)
                        np.testing.assert_array_equal(pred[arm],again)
                    arguments=(preprocessing,pred,y[val],env[val],moving,support,data['sites'][vid],data['recordings'][vid],data['frames'][vid],vid)
                    result,actions=api.evaluate(*arguments)
                    count=scalar.verify(*arguments,result,actions);checks+=count
                    row=dict(group=group,source=site,head_seed=seed,partition=partition,result=result,
                        independent_scalar_checks=count,exact_inference_replay=True,strong_control_replay_exact=True,
                        training_ids_hash=h(tid),validation_ids_hash=h(vid),targets_hash=h(y[val]),
                        prediction_hashes={k:h(v) for k,v in pred.items()},action_hashes={k:h(v) for k,v in actions.items()},
                        input_hashes=input_hashes,training_freeze_sha256=inventory['training_freeze_sha256'],
                        readout_registration_sha256=sha(PUBLIC/'registration.json'),
                        result_source='fresh_run_fixed_final_exposed_development_readout',model_source='cached_verified',
                        new_training=False,independent_roles_read=False,deployment_changed=False)
                    path=PUBLIC/'groups'/(group+'_head'+str(seed)+'.json')
                    if path.exists() and not resume: raise RuntimeError('Explicit readout resume required')
                    once(path,row);rows.append(row);refs.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path)))
                    beat(state='group_verified',groups=len(rows),seconds=time.monotonic()-started)
        bytes_fetched=neural.fetched_bytes
    summary=api.summarize(rows,reg['expected'],cfg)
    checks+=scalar.verify_summary(rows,cfg,summary)
    once(PUBLIC/'summary.json',summary)
    once(PUBLIC/'complete.json',dict(groups=refs,summary_sha256=sha(PUBLIC/'summary.json'),
        training_freeze_sha256=inventory['training_freeze_sha256'],independent_scalar_checks=checks,
        exact_inference_replay=True,strong_control_replay_exact=True,neural_checkpoint_bytes_streamed=bytes_fetched,
        seconds=time.monotonic()-started,local_checkpoint_cache=False,new_training=False,
        independent_roles_read=False,deployment_changed=False))
    beat(state='complete',groups=len(rows),advance=summary['advance_to_transfer_design'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['register','preflight','run']);p.add_argument('--resume',action='store_true');args=p.parse_args()
    cfg,reg=registration()
    if args.phase=='register':
        once(PUBLIC/'registration.json',reg);print(json.dumps(dict(registered=True,actual_readout='not_run')));return
    assert reg==json.loads((PUBLIC/'registration.json').read_text())
    parent.base.inter.committed(PUBLIC/'registration.json')
    docs,inventory=heads.local_admission()
    parent.base.inter.committed(heads.manager.PUBLIC/'training_freeze.json')
    if args.phase=='preflight':
        print(json.dumps(dict(complete_heads=len(docs),actual_readout='not_run',independent_roles_read=False)));return
    if (PUBLIC/'complete.json').exists(): raise RuntimeError('Completed readout is immutable')
    PRIVATE.mkdir(parents=True,exist_ok=True)
    fitting.torch.set_num_threads(4);fitting.torch.set_num_interop_threads(1)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(cfg,reg,docs,inventory,args.resume)


if __name__=='__main__':
    main()
