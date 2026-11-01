"""Full SOURCE feasibility after removing only frozen gain-sign eligibility."""
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

if platform.system()=='Darwin' and platform.machine()!='arm64':raise RuntimeError('Native arm64 required before Torch')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_m3w_frozen_cost_support as previous
from scripts.report_m3w_feature_cost_evidence import aggregate_groups
from src.evaluation import m3w_reference_support_preflight as api
from src.evaluation.m3w_metric_bundle import pack

parent=previous.parent;sha,once,write=parent.sha,parent.once,parent.write
NAME='european_reference_support_preflight_v1'
PUBLIC=ROOT/'outputs/publication_readiness_2026_09'/NAME
PRIVATE=ROOT/'data/stage_cvpr2027_experiments'/NAME
REG=PUBLIC/'registration.json'
FILES=(Path(__file__),ROOT/'src/world_model/m3w_reference_action_support.py',ROOT/'src/evaluation/m3w_reference_support_preflight.py',
    ROOT/'tests/test_m3w_reference_support_preflight.py',PUBLIC/'protocol.md')


def admission(register=False):
    values,reg=previous.admission();checked=json.loads((previous.PUBLIC/'verification.json').read_text())
    if not checked['exact72_group_numeric_recalculation'] or checked['summary_sha256']!=sha(previous.PUBLIC/'summary.json'):
        raise ValueError('Verified full support diagnosis required')
    once(REG,dict(bindings={str(p.relative_to(ROOT)):sha(p) for p in FILES},
        previous_complete_sha256=sha(previous.PUBLIC/'complete.json'),previous_summary_sha256=sha(previous.PUBLIC/'summary.json'),
        previous_verification_sha256=sha(previous.PUBLIC/'verification.json'),contexts=reg['contexts'],seeds=reg['seeds'],
        changed_factor='remove_predicted_gain_sign_only_preserve_causal_reference_and_two2percent_cost_constraints',
        label_aware_preflight_not_model_success=True,previous_diagnostic_outcomes_already_read=True,
        supports=list(api.NAMES),new_training=False,independent_roles_opened=False,
        deployment_changed=False,submission_ready=False,stage5c_executed=False,smc_enabled=False))
    if not register:
        for rel,digest in json.loads(REG.read_text())['bindings'].items():
            if sha(ROOT/rel)!=digest:raise ValueError('Frozen preflight definition changed')
            parent.source.frozen.BASE.engine.committed(ROOT/rel)
        parent.source.frozen.BASE.engine.committed(REG)
    return values,json.loads(REG.read_text())


def groups(values,beat):
    cfg,source_values,store,banks,models,controls=values;old,main,_=parent.resources()
    original=aggregate_groups(parent.NAME)
    for context,items in sorted(source_values[2]['contexts'].items()):
        packets,changed=parent.previous.previous.context_packets(context,items,source_values,store,banks,beat)
        for seed in cfg['seeds']:
            data,ranges,_=parent.source.prepare(context,seed,packets,changed,store,models,old,main,beat)
            cost=parent.api.parent.read_checkpoint(ROOT/controls[context,seed]['path'])
            parent.api.parent.exact(cost['input_hashes'],data['input_hashes']);prediction=parent.api.prior.base.predict(cost,data['causal'])
            for source,packet in sorted(packets.items()):
                key=f'{context}_{source}_seed{seed}_fit.json';ref=original[key];p=prediction[ranges[source]]
                if parent.api.parent.digest(p)!=ref['cost_prediction_sha256'] or parent.api.parent.digest(packet['ids'])!=ref['ids_sha256']:
                    raise ValueError('Original exact inputs and frozen moments required')
                old_choices=parent.api.parent.read_checkpoint(parent.PRIVATE/'progress'/(key+'.pt.gz'))['choices']
                choices={a:old_choices[a] for a in api.CONTROLS}
                if any(parent.api.parent.digest(c)!=ref['action_sha256'][a] for a,c in choices.items()):raise ValueError('Original actions changed')
                yield context,seed,source,key,packet,p,choices,ref
            del data,cost,prediction
        del packets,changed


def run(verify=False,resume=False):
    values,reg=admission();parent.api.torch.set_num_threads(4);parent.api.torch.set_num_interop_threads(1);start=time.monotonic()
    if (PUBLIC/'complete.json').exists() and not verify:raise ValueError('Completed preflight immutable')
    def beat(**kw):
        if time.monotonic()-start>43200:raise TimeoutError('12h phase boundary;resume original preflight')
        if shutil.disk_usage(ROOT).free<10*2**30:raise OSError('10GiB reserve')
        v=dict(pid=os.getpid(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw);write(PRIVATE/'heartbeat.json',v);print(json.dumps(v),flush=True)
    beat(state='runtime_ready',architecture=platform.machine(),threads=4,interop_threads=1,workers=0)
    rows=[];refs=[];ids=set();records=set()
    for context,seed,source,key,packet,p,choices,ref in groups(values,beat):
        path=PUBLIC/'groups'/key
        if path.exists() and not(resume or verify):raise ValueError('Explicit partial resume required')
        result=api.diagnose(packet,p,choices)
        for a in api.CONTROLS:
            if result['controls'][a]!=ref['result']['policies'][a]['policy']:raise ValueError('Full frozen control score changed')
        row=dict(context=context,source=source,seed=seed,result=result,registration_sha256=sha(REG),
            cost_prediction_sha256=ref['cost_prediction_sha256'],ids_sha256=ref['ids_sha256'],
            original_control_scores_exact=True,result_source='fresh_label_aware_preflight_cached_verified_costs_and_controls')
        if verify:
            if row!=json.loads(path.read_text()):raise ValueError('Exact preflight replay differs')
        else:once(path,row)
        rows.append(row);refs.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path)))
        ids.update(map(int,packet['ids']));records.update((source,str(r)) for r in packet['recordings'])
        beat(state='preflight_group_complete',groups=len(rows),total=72,context=context,source=source,seed=seed)
    if len(ids)!=149510 or len(records)!=74:raise ValueError('All original targets and recordings required')
    summary=api.summarize(rows,reg['contexts'],values[0]);summary.update(unique_target_rows=len(ids),unique_recordings=len(records))
    runtime=dict(pid=os.getpid(),seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if verify:
        if summary!=json.loads((PUBLIC/'summary.json').read_text()):raise ValueError('Canonical full preflight differs')
        once(PUBLIC/'verification.json',dict(exact72_numeric_replay=True,all_original_controls_exact=True,
            summary_sha256=sha(PUBLIC/'summary.json'),complete_sha256=sha(PUBLIC/'complete.json')))
    else:
        once(PUBLIC/'summary.json',summary);once(PUBLIC/'complete.json',dict(**runtime,groups=refs,
            registration_sha256=sha(REG),summary_sha256=sha(PUBLIC/'summary.json')));pack(ROOT,PUBLIC)
    write(PRIVATE/('verify_runtime.json' if verify else 'run_runtime.json'),runtime)
    beat(state='all72_preflight_verified' if verify else 'all72_preflight_complete')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('register','run','verify'));p.add_argument('--resume',action='store_true');a=p.parse_args()
    for d in (PUBLIC/'groups',PRIVATE):d.mkdir(parents=True,exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if a.phase=='register':admission(True)
        else:run(a.phase=='verify',a.resume)
