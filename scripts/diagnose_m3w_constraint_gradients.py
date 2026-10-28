"""Frozen 216-head TRAIN-only gradient audit; no optimizer step or held-out input."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

from scripts import read_m3w_task_risk_final_heads as heads

old=heads.execution; base=old.base; ROOT=base.ROOT
PUBLIC=ROOT/'outputs/publication_readiness_2026_09/european_task_risk_gradient_v1'
PRIVATE=ROOT/'data/stage_cvpr2027_experiments/european_task_risk_gradient_v1'
REG=PUBLIC/'registration.json'


def registration():
    files=[Path(__file__).resolve(),ROOT/'src/world_model/m3w_constraint_scale.py',
           ROOT/'tests/test_m3w_constraint_scale.py',PUBLIC/'protocol.md']
    value=dict(parent_training_freeze_sha256=base.sha(heads.manager.PUBLIC/'training_freeze.json'),
        parent_registration_sha256=base.sha(heads.manager.REG),
        bindings={str(p.relative_to(ROOT)):base.sha(p) for p in files},
        views=216, fixed_monitor_queries=128, role='TRAIN_only_posthoc_mechanism_diagnostic',
        new_training=False, independent_roles_read=False, validation_read=False,
        threshold_search=False, deployment_changed=False)
    base.once(REG,value)
    return value


def execute(resume):
    reg=registration()
    for rel in [*reg['bindings'],str(REG.relative_to(ROOT))]:
        result=subprocess.run(['git','show','HEAD:'+rel],cwd=ROOT,capture_output=True)
        if result.returncode or result.stdout!=(ROOT/rel).read_bytes():
            raise ValueError('Commit diagnostic before execution')
    if (PUBLIC/'complete.json').exists(): raise ValueError('Complete diagnostic is immutable')
    from src.world_model import m3w_constraint_scale as api
    from src.world_model.m3w_false_safe_diagnostic import moving_from_features
    api.torch.set_num_threads(4);api.torch.set_num_interop_threads(1)
    docs,inventory=heads.local_admission(); train_reg,cfg=old.check_registration()
    manifest=json.loads((base.PRIVATE/'create_train_input_manifest.json').read_text())
    PRIVATE.mkdir(parents=True,exist_ok=True); started=time.monotonic(); previous=None; rows=[]; refs=[]
    def beat(**kw):
        value=dict(pid=os.getpid(),utc=datetime.now(timezone.utc).isoformat(),**kw)
        tmp=PRIVATE/'heartbeat.tmp';tmp.write_text(json.dumps(value)+'\n');os.replace(tmp,PRIVATE/'heartbeat.json')
        print(json.dumps(value),flush=True)
    with (PRIVATE/'process.lock').open('a') as lock, heads.Reader(inventory) as reader:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for ref in manifest['heads']:
            group=ref['group']; identity=ref['identity']
            if group!=previous:
                beat(state='streaming_TRAIN',group=group)
                args=old.load_packet(ref,train_reg)
                moving=moving_from_features(args[0]);_,support=api.old.parent.forest.causal_inputs(args[0],args[1],args[-1])
                eligible=moving & support;previous=group
            for arm in api.old.ARMS:
                key=group,identity['source'],identity['seed'],arm; doc=docs[key]
                path=PUBLIC/'groups'/(group+'_head'+str(identity['seed'])+'_'+arm+'.json')
                bindings=dict(registration_sha256=base.sha(REG),checkpoint_sha256=doc['checkpoint']['sha256'])
                if path.exists():
                    if not resume: raise ValueError('Explicit diagnostic resume required')
                    row=json.loads(path.read_text())
                    if any(row[k]!=v for k,v in bindings.items()): raise ValueError('Changed completed diagnostic')
                else:
                    state=reader.load(ROOT/doc['checkpoint']['path'])
                    row=dict(**api.frozen_monitor(state,args,eligible),**bindings)
                    base.once(path,row)
                rows.append(row);refs.append(dict(path=str(path.relative_to(ROOT)),sha256=base.sha(path)))
            beat(state='diagnosed',identities=len(rows)//3,total=72)
        summary={}
        for arm in api.old.ARMS:
            ds=[r for r in rows if r['arm']==arm]
            def median(k):
                vs=[r[k] for r in ds if r[k] is not None]
                return statistics.median(vs) if vs else None
            summary[arm]=dict(views=len(ds),
                median_weighted_easy_gradient_over_data=median('weighted_easy_gradient_over_data_gradient'),
                median_weighted_risk_gradient_over_data=median('weighted_risk_gradient_over_data_gradient'),
                median_regret_easy_gradient_cosine=median('regret_easy_gradient_cosine'),
                median_data_risk_gradient_cosine=median('data_risk_gradient_cosine'),
                mean_soft_action=statistics.mean(r['soft_action_mean'] for r in ds),
                mean_hard_action=statistics.mean(r['hard_action_mean'] for r in ds),
                median_budget_scales=[statistics.median(r['TRAIN_budget_scales'][i] for r in ds) for i in range(2)],
                median_scaled_excess=[statistics.median(r['scaled_excess'][i] for r in ds) for i in range(2)],
                positive_hard_excess_views=[sum(r['hard_excess'][i]>0 for r in ds) for i in range(2)],
                positive_soft_excess_views=[sum(r['raw_excess'][i]>0 for r in ds) for i in range(2)])
        base.once(PUBLIC/'summary.json',summary)
        base.once(PUBLIC/'complete.json',dict(groups=refs,registration_sha256=base.sha(REG),
            summary_sha256=base.sha(PUBLIC/'summary.json'),seconds=time.monotonic()-started,
            result_source='fresh_run_frozen_TRAIN_autograd_diagnostic',parent_heads='cached_verified',
            no_optimizer_step=True,independent_roles_read=False,validation_read=False,new_training=False))
        beat(state='complete',views=len(rows))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['register','run']);p.add_argument('--resume',action='store_true');a=p.parse_args()
    if a.action=='register':registration();print('registered TRAIN-only diagnostic')
    else:execute(a.resume)


if __name__=='__main__':main()
