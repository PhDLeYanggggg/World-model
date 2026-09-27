"""Fit exact nonnegative score intercepts on two fitting sources only."""
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
if platform.system()=='Darwin' and platform.machine()!='arm64':
    raise RuntimeError('Native arm64 required before Torch import')
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import diagnose_m3w_selected_risk as diagnosis
from src.world_model import m3w_signed_bias_probe as api
parent,base=diagnosis.parent,diagnosis.base
PUBLIC=parent.PUBLIC.parent/'european_signed_bias_probe_v1'
PRIVATE=parent.PRIVATE.parent/'european_signed_bias_probe_v1'
FILES=['src/world_model/m3w_signed_bias_probe.py','scripts/probe_m3w_signed_bias.py',
       'tests/test_m3w_signed_bias_probe.py',str(PUBLIC.relative_to(ROOT)/'protocol.md')]


def beat(state,**kw):
    row=dict(state=state,pid=os.getpid(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw)
    base.inter.json_write(PRIVATE/'heartbeat.json',row)
    with (PRIVATE/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)


def binding():
    diagnosis.identity()
    return dict(parent_seal=base.artifact(parent.PUBLIC/'verification.json'),
        motivation_readout=base.artifact(diagnosis.PUBLIC/'summary.json'),
        bindings={p:base.digest(ROOT/p) for p in FILES},kind='fitting_only_analytic_intercepts',
        heads=216,new_neural_training=False,new_policy=False,independent_roles_read=False)


def reduce(records):
    rows=[dict(group=r['group'],seed=r['seed'],arm=a,**v) for r in records for a,v in r['fits'].items()]
    result={}
    for arm in parent.api.ARMS:
        rr=[r for r in rows if r['arm']==arm]
        offsets=np.array([r['nonnegative_offset'] for r in rr]);raw=np.array([r['unconstrained_offset'] for r in rr])
        loss_before=np.array([r['fitting_objective_before'] for r in rr]);loss_after=np.array([r['fitting_objective_after'] for r in rr])
        result[arm]=dict(heads=len(rr),positive_offsets_per_axis=(offsets>0).sum(0).tolist(),
            offsets_mean=offsets.mean(0).tolist(),offsets_median=np.median(offsets,axis=0).tolist(),
            offsets_min=offsets.min(0).tolist(),offsets_max=offsets.max(0).tolist(),
            unconstrained_mean=raw.mean(0).tolist(),loss_before_mean=float(loss_before.mean()),
            loss_after_mean=float(loss_after.mean()),median_relative_loss_reduction_percent=float(np.median(100*(loss_before-loss_after)/loss_before)),
            exact_quadratic_fit=True,unknown_rows_used=0,
            by_seed={str(seed):dict(heads=sum(r['seed']==seed for r in rr),
                offset_mean=np.mean([r['nonnegative_offset'] for r in rr if r['seed']==seed],axis=0).tolist()) for seed in (17,29,43)})
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--register',action='store_true')
    p.add_argument('--resume',action='store_true');p.add_argument('--replay',action='store_true');a=p.parse_args()
    PUBLIC.mkdir(parents=True,exist_ok=True);PRIVATE.mkdir(parents=True,exist_ok=True);ident=binding()
    if a.register:base.immutable_json(PUBLIC/'registration.json',ident);print(json.dumps(ident));return
    assert json.loads((PUBLIC/'registration.json').read_text())==ident
    base.inter.committed(PUBLIC/'registration.json');base.torch.set_num_threads(4);base.torch.set_num_interop_threads(1)
    start=time.monotonic()
    with (PRIVATE/'run.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);beat('started',replay=a.replay)
        _,data,jobs,oid,_,_,_,_,_,pid=parent.load();causal={k:data[k] for k in parent.CAUSAL_KEYS}
        frozen=json.loads((parent.PUBLIC/'training_freeze.json').read_text())
        refs={Path(r['path']).parent.name:r for r in frozen['groups']};records=[];artifacts=[]
        for c in base.floor_api.contexts(causal,jobs,oid):
            bank=parent.masks(c,causal)
            for pair in range(6):
                name=c['name']+f'_pair{pair}';path=PRIVATE/'fits'/(name+'.json')
                if shutil.disk_usage(PRIVATE).free<10*2**30:raise OSError('Keep10GiB and completed fits')
                if path.exists() and a.resume and not a.replay:
                    out=json.loads(path.read_text());assert out['identity']==ident
                    assert out['parent_fit']==refs[name] and base.artifact(ROOT/refs[name]['path'])==refs[name]
                else:
                    if path.exists() and not a.replay:raise ValueError('Existing probe fit requires resume/replay')
                    ref=refs[name];assert base.artifact(ROOT/ref['path'])==ref
                    doc=json.loads((ROOT/ref['path']).read_text());assert doc['identity']['experiment']==pid
                    roles=doc['identity']['roles_and_targets']
                    base.floor_api.api.assert_roles(roles['producer_sites'],roles['controller_sites'],roles['training_sites'],roles['held_sites'])
                    fit=np.flatnonzero(np.isin(data['sites'][c['ids']],roles['training_sites']));ids=c['ids'][fit]
                    states={arm:parent.api.head.read_checkpoint(ROOT/r['path']) for arm,r in doc['artifacts'].items()}
                    for r in doc['artifacts'].values():assert base.artifact(ROOT/r['path'])==r
                    pr=states['subset_pointwise']['preprocess']
                    u=parent.api.head.descriptors(data['geometry'][ids],c['floor'][fit],c['prediction'][fit],c['x'][fit],pr)
                    predictions={}
                    for arm,s in states.items():
                        model=parent.previous.descriptor.model_from(s)
                        pred=parent.api.head.predict(model,c['x'][fit],u,c['env'][fit],pr,s['descriptor_preprocess'])
                        predictions[arm]=parent.api.head.parent.signed(pred.astype(float))/pr['cost_scale']
                    # Only fitting positions are passed to the outcome API.
                    cv,_,(floor,_),(neural,_)=base.floor_api.costs(c,data,fit)
                    truth=diagnosis.api.signed_cost(cv,floor,neural,c['job']['design']['easy_cut'],pr['cost_scale'])
                    fits={arm:api.fit_intercept(pred,truth,data['sites'][ids],data['recordings'][ids],data['frames'][ids],bank[fit],arm=='subset_aggregate') for arm,pred in predictions.items()}
                    out=dict(identity=ident,parent_fit=ref,group=name,seed=c['job']['old_identity']['seed'],roles=roles,
                        fitting_ids_hash=base.inter.array_hash(ids),fitting_targets_hash=base.inter.array_hash(truth),
                        fitting_predictions_hash={arm:base.inter.array_hash(pred) for arm,pred in predictions.items()},
                        fits=fits,held_labels_used=False,policy_changed=False)
                    base.immutable_json(path,out)
                records.append(out);artifacts.append(base.artifact(path));beat('fit_replayed' if a.replay else 'fit_complete',group=name,complete=len(records))
        assert len(records)==108
        summary=dict(identity=ident,result_source='fresh_run_analytic_fit',neural_heads='cached_verified',summary=reduce(records),
            training_loss_is_downstream_lift=False,held_evaluation='not_run',policy_changed=False,
            independent_confirmation=False,stage5c_executed=False,smc_enabled=False)
        base.immutable_json(PUBLIC/'summary.json',summary)
        base.immutable_json(PUBLIC/('replay.json' if a.replay else 'completion.json'),dict(identity=ident,groups=artifacts,
            summary=base.artifact(PUBLIC/'summary.json'),exact_replay=a.replay,seconds=time.monotonic()-start,
            pid=os.getpid(),peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        beat('complete',seconds=time.monotonic()-start)


if __name__=='__main__':main()
