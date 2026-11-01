"""Exact post-hoc SOURCE support diagnosis,not another model or threshold search."""
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
from scripts import run_m3w_free_allocation_actor as parent
from scripts.report_m3w_feature_cost_evidence import aggregate_groups
from src.evaluation import m3w_frozen_cost_support as api
from src.evaluation.m3w_metric_bundle import pack

NAME='european_frozen_cost_support_v1'
PUBLIC=ROOT/'outputs/publication_readiness_2026_09'/NAME
PRIVATE=ROOT/'data/stage_cvpr2027_experiments'/NAME
REG=PUBLIC/'registration.json'
sha,once,write=parent.sha,parent.once,parent.write
FILES=(Path(__file__),ROOT/'src/evaluation/m3w_frozen_cost_support.py',ROOT/'tests/test_m3w_frozen_cost_support.py',PUBLIC/'protocol.md')


def admission(register=False):
    values=parent.admission();checked=json.loads((parent.PUBLIC/'verification.json').read_text())
    if not checked['all72_frozen_actions_inference_and_scores_exact'] or checked['summary_sha256']!=sha(parent.PUBLIC/'summary.json'):
        raise ValueError('Complete verified parent required')
    once(REG,dict(bindings={str(p.relative_to(ROOT)):sha(p) for p in FILES},
        parent_summary_sha256=sha(parent.PUBLIC/'summary.json'),parent_complete_sha256=sha(parent.PUBLIC/'complete.json'),
        parent_verification_sha256=sha(parent.PUBLIC/'verification.json'),parent_registration_sha256=sha(parent.REG),
        contexts=json.loads(parent.REG.read_text())['contexts'],seeds=[17,29,43],supports=list(api.SUPPORTS),
        controls=list(api.CONTROLS),post_hoc_parent_outcomes_already_read=True,
        scope='all72_original_exposed_SOURCE_fitting_groups',future_labels_used_by_diagnostic_oracle=True,
        new_training=False,held_scored=False,independent_roles_opened=False,deployment_changed=False,
        submission_ready=False,stage5c_executed=False,smc_enabled=False))
    reg=json.loads(REG.read_text())
    if not register:
        for rel,digest in reg['bindings'].items():
            if sha(ROOT/rel)!=digest:raise ValueError('Registered diagnostic changed')
            parent.source.frozen.BASE.engine.committed(ROOT/rel)
        parent.source.frozen.BASE.engine.committed(REG)
    return values,reg


def run(verify=False,resume=False,pilot=False):
    values,reg=admission();cfg,source_values,store,banks,models,controls=values
    parent.api.torch.set_num_threads(4);parent.api.torch.set_num_interop_threads(1)
    start=time.monotonic()
    def beat(**kw):
        if time.monotonic()-start>43200:raise TimeoutError('12h phase boundary;resume same diagnostic')
        if shutil.disk_usage(ROOT).free<10*2**30:raise OSError('10GiB reserve')
        v=dict(pid=os.getpid(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw)
        write(PRIVATE/'heartbeat.json',v);print(json.dumps(v),flush=True)
    beat(state='runtime_ready',architecture=platform.machine(),threads=4,interop_threads=1,workers=0)
    if (PUBLIC/'complete.json').exists() and not verify:raise ValueError('Completed diagnostic immutable')
    old,main,_=parent.resources();original=aggregate_groups(parent.NAME);rows=[];refs=[];ids=set();recordings=set()
    parent_refs={Path(r['path']).name:r for r in json.loads((parent.PUBLIC/'complete.json').read_text())['groups']}
    for ci,(context,items) in enumerate(sorted(source_values[2]['contexts'].items())):
        packets,changed=parent.previous.previous.context_packets(context,items,source_values,store,banks,beat)
        for seed in ([17] if pilot else reg['seeds']):
            data,ranges,_=parent.source.prepare(context,seed,packets,changed,store,models,old,main,beat)
            cost=parent.api.parent.read_checkpoint(ROOT/controls[context,seed]['path'])
            parent.api.parent.exact(cost['input_hashes'],data['input_hashes'])
            prediction=parent.api.prior.base.predict(cost,data['causal'])
            for name,packet in sorted(packets.items()):
                key=f'{context}_{name}_seed{seed}_fit.json';p=prediction[ranges[name]];ref=original[key]
                if parent.api.parent.digest(p)!=ref['cost_prediction_sha256'] or parent.api.parent.digest(packet['ids'])!=ref['ids_sha256']:
                    raise ValueError('Exact original moment predictions and row alignment required')
                saved=parent.api.parent.read_checkpoint(parent.PRIVATE/'progress'/(key+'.pt.gz'))
                choices={a:saved['choices'][a] for a in api.CONTROLS}
                for a,c in choices.items():
                    if parent.api.parent.digest(c)!=ref['action_sha256'][a]:raise ValueError('Parent policy action changed')
                path=PUBLIC/'groups'/key
                if path.exists() and not (resume or verify or pilot):raise ValueError('Explicit partial resume required')
                result=api.diagnose(packet,p,choices)
                for a in api.CONTROLS:
                    if result['controls'][a]!=ref['result']['policies'][a]['policy']:
                        raise ValueError('Exact original full policy score required')
                    for rec,value in result['recordings'].items():
                        if value['controls'][a]['actual']!=ref['result']['recordings'][rec][a]:
                            raise ValueError('Exact original per-recording score required')
                row=dict(context=context,source=name,seed=seed,result=result,
                    registration_sha256=sha(REG),parent_group_sha256=parent_refs[key]['sha256'],
                    cost_prediction_sha256=ref['cost_prediction_sha256'],ids_sha256=ref['ids_sha256'],
                    frozen_control_scores_exact=True,result_source='fresh_label_aware_SOURCE_diagnosis_cached_verified_models_and_actions')
                if not pilot:
                    if verify:
                        if row!=json.loads(path.read_text()):raise ValueError('Exact diagnosis replay differs')
                    else:once(path,row)
                rows.append(row);refs.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path) if not pilot else None))
                ids.update(map(int,packet['ids']));recordings.update((name,str(r)) for r in packet['recordings'])
                beat(state='diagnostic_group_complete',groups=len(rows),total=4 if pilot else 72,context=context,source=name,seed=seed)
            del data,prediction,cost
        del packets,changed
        if pilot:break
    runtime=dict(pid=os.getpid(),seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if pilot:
        once(PUBLIC/'pilot.json',dict(**runtime,groups=len(rows),original_controls_exact=True,registration_sha256=sha(REG),
            results_not_saved_or_used_for_model_selection=True,projected72group_seconds=(time.monotonic()-start)*18))
    else:
        if len(ids)!=149510 or len(recordings)!=74:raise ValueError('Original full cohort required')
        summary=api.summarize(rows,reg['contexts'],cfg);summary.update(unique_target_rows=len(ids),unique_recordings=len(recordings))
        if verify:
            if summary!=json.loads((PUBLIC/'summary.json').read_text()):raise ValueError('Exact canonical summary differs')
            once(PUBLIC/'verification.json',dict(exact72_group_numeric_recalculation=True,all_original_control_scores_exact=True,
                summary_sha256=sha(PUBLIC/'summary.json'),complete_sha256=sha(PUBLIC/'complete.json')))
        else:
            once(PUBLIC/'summary.json',summary);once(PUBLIC/'complete.json',dict(**runtime,groups=refs,registration_sha256=sha(REG),summary_sha256=sha(PUBLIC/'summary.json')))
            pack(ROOT,PUBLIC)
    write(PRIVATE/('verify_runtime.json' if verify else 'pilot_runtime.json' if pilot else 'run_runtime.json'),runtime)
    beat(state='diagnostic_verified' if verify else 'pilot_complete' if pilot else 'complete72_diagnosis')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('register','pilot','run','verify'));p.add_argument('--resume',action='store_true');a=p.parse_args()
    for d in (PUBLIC/'groups',PRIVATE):d.mkdir(parents=True,exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if a.phase=='register':admission(True)
        else:run(a.phase=='verify',a.resume,a.phase=='pilot')
