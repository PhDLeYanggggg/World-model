"""Check all remaining recording views for necessary query-pool easy support."""
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
from scripts import run_m3w_reference_support_preflight as previous
from scripts.report_m3w_feature_cost_evidence import aggregate_groups
from src.evaluation import m3w_query_easy_support as api
from src.evaluation.m3w_metric_bundle import pack

NAME='european_query_easy_support_v1';PUBLIC=ROOT/'outputs/publication_readiness_2026_09'/NAME
PRIVATE=ROOT/'data/stage_cvpr2027_experiments'/NAME;REG=PUBLIC/'registration.json'
sha,once,write=previous.sha,previous.once,previous.write
FILES=(Path(__file__),ROOT/'src/evaluation/m3w_query_easy_support.py',ROOT/'tests/test_m3w_query_easy_support.py',PUBLIC/'protocol.md')


def admission(register=False):
    values,_=previous.admission();checked=json.loads((previous.PUBLIC/'verification.json').read_text())
    if not checked['exact72_numeric_replay'] or checked['summary_sha256']!=sha(previous.PUBLIC/'summary.json'):raise ValueError('Verified full preflight required')
    once(REG,dict(bindings={str(p.relative_to(ROOT)):sha(p) for p in FILES},
        previous_complete_sha256=sha(previous.PUBLIC/'complete.json'),previous_summary_sha256=sha(previous.PUBLIC/'summary.json'),
        previous_verification_sha256=sha(previous.PUBLIC/'verification.json'),solver_seconds_per_query=10.,
        contexts=json.loads(previous.REG.read_text())['contexts'],seeds=[17,29,43],
        scope='all72_original_groups_preserved_cached_pointwise_witnesses_plus_all_remaining_record_views',
        future_easy_labels_used_by_diagnostic_only=True,post_hoc_parent_outcomes_read=True,
        new_training=False,independent_roles_opened=False,deployment_changed=False,submission_ready=False,stage5c_executed=False,smc_enabled=False))
    if not register:
        for rel,digest in json.loads(REG.read_text())['bindings'].items():
            if sha(ROOT/rel)!=digest:raise ValueError('Frozen query-support diagnosis changed')
            previous.parent.source.frozen.BASE.engine.committed(ROOT/rel)
        previous.parent.source.frozen.BASE.engine.committed(REG)
    return values,json.loads(REG.read_text())


def summarize(rows):
    if len(rows)!=72 or len({(r['context'],r['source'],r['seed']) for r in rows})!=72:raise ValueError('All72views required')
    records=[dict(context=r['context'],source=r['source'],seed=r['seed'],recording=k,**v) for r in rows for k,v in r['recordings'].items()]
    cached=[r for r in records if r['result_source']=='cached_verified_pointwise_witness']
    fresh=[r for r in records if r['result_source']=='fresh_query_feasibility']
    blocked=[r for r in fresh if r['diagnosis']['complete_easy_support_obstruction']]
    unknown=[r for r in fresh if not r['diagnosis']['necessary_condition_conclusive']]
    return dict(groups=72,required_record_views=len(records),cached_verified_pointwise_record_views=len(cached),fresh_query_record_views=len(fresh),
        necessary_query_easy_supported_record_views=len(cached)+sum(r['diagnosis']['necessary_query_easy_support'] for r in fresh),
        blocked_record_views=blocked,inconclusive_record_views=unknown,
        algebraic_or_singleton_blocked_record_views=sum(r['diagnosis']['obstruction_algebraic_or_singleton_only'] for r in fresh),
        frozen_cost_family_has_necessary_support_obstruction=bool(blocked),
        all_original_rows_and_recordings_retained=True,positive_safe_model_gain_proven=False,
        oracle_labels_used_only_in_diagnostic=True,new_training=False,independent_roles_opened=False,
        deployment_changed=False,submission_ready=False,stage5c_executed=False,smc_enabled=False)


def run(verify=False,resume=False):
    values,reg=admission();previous.parent.api.torch.set_num_threads(4);previous.parent.api.torch.set_num_interop_threads(1);start=time.monotonic()
    if (PUBLIC/'complete.json').exists() and not verify:raise ValueError('Completed query diagnosis immutable')
    def beat(**kw):
        if time.monotonic()-start>43200:raise TimeoutError('12hphase boundary;resume exact remaining groups')
        if shutil.disk_usage(ROOT).free<10*2**30:raise OSError('10GiBreserve')
        d=dict(pid=os.getpid(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw);write(PRIVATE/'heartbeat.json',d);print(json.dumps(d),flush=True)
    beat(state='runtime_ready',threads=4,interop_threads=1,workers=0,architecture=platform.machine())
    preflight=aggregate_groups(previous.NAME);rows=[];refs=[];ids=set();records=set()
    for context,seed,source,key,packet,p,_,original in previous.groups(values,beat):
        old=preflight[key]
        if old['cost_prediction_sha256']!=original['cost_prediction_sha256'] or old['ids_sha256']!=original['ids_sha256']:raise ValueError('Original full inputs differ')
        details={}
        for rec in sorted(set(packet['recordings'])):
            prior=old['result']['recordings'][str(rec)]['reference_pointwise']
            if prior['budget_feasible']:
                details[str(rec)]=dict(result_source='cached_verified_pointwise_witness',necessary_query_easy_support=True)
            else:
                ix=packet['recordings']==rec;sub={k:packet[k][ix] for k in ('y','allowed','recordings','frames','ids')}
                d=api.recording_diagnosis(sub,p[ix],seconds=reg['solver_seconds_per_query'],
                    heartbeat=lambda **kw:beat(state='query_support',context=context,source=source,seed=seed,recording=str(rec),**kw))
                details[str(rec)]=dict(result_source='fresh_query_feasibility',diagnosis=d)
        row=dict(context=context,source=source,seed=seed,recordings=details,
            registration_sha256=sha(REG),cost_prediction_sha256=original['cost_prediction_sha256'],ids_sha256=original['ids_sha256'])
        path=PUBLIC/'groups'/key
        if path.exists() and not(verify or resume):raise ValueError('Explicit partial resume required')
        if verify:
            if row!=json.loads(path.read_text()):raise ValueError('Exact query diagnosis replay differs')
        else:once(path,row)
        rows.append(row);refs.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path)))
        ids.update(map(int,packet['ids']));records.update((source,str(r)) for r in packet['recordings'])
        beat(state='query_support_group_complete',groups=len(rows),total=72,context=context,source=source,seed=seed)
    if len(ids)!=149510 or len(records)!=74:raise ValueError('Complete original target cohort required')
    summary=summarize(rows);summary.update(unique_target_rows=len(ids),unique_recordings=len(records))
    runtime=dict(pid=os.getpid(),seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if verify:
        if summary!=json.loads((PUBLIC/'summary.json').read_text()):raise ValueError('Complete canonical diagnosis differs')
        once(PUBLIC/'verification.json',dict(exact72_groups_and_remaining_query_results=True,
            summary_sha256=sha(PUBLIC/'summary.json'),complete_sha256=sha(PUBLIC/'complete.json')))
    else:
        once(PUBLIC/'summary.json',summary);once(PUBLIC/'complete.json',dict(**runtime,groups=refs,registration_sha256=sha(REG),summary_sha256=sha(PUBLIC/'summary.json')));pack(ROOT,PUBLIC)
    write(PRIVATE/('verify_runtime.json' if verify else 'run_runtime.json'),runtime);beat(state='all72_verified' if verify else 'all72_complete')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('register','run','verify'));p.add_argument('--resume',action='store_true');a=p.parse_args()
    for d in (PUBLIC/'groups',PRIVATE):d.mkdir(parents=True,exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if a.phase=='register':admission(True)
        else:run(a.phase=='verify',a.resume)
