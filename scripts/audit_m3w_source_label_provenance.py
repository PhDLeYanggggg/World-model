"""Trace frozen fitting-query future support directly to the original raw export."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
import time
import zipfile

if platform.system()=='Darwin' and platform.machine()!='arm64':raise RuntimeError('Native arm64 .venv-pytorch required')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import build_m3w_detector_history_store as old
import numpy as np
from src.evaluation import m3w_source_label_provenance as api

NAME='european_source_label_provenance_v1';PUBLIC=ROOT/'outputs/publication_readiness_2026_09'/NAME
PRIVATE=ROOT/'data/stage_cvpr2027_experiments'/NAME
CONFIG=ROOT/'configs/m3w_source_label_provenance_v1.json';REG=PUBLIC/'registration.json'
FILES=(Path(__file__),CONFIG,ROOT/'src/evaluation/m3w_source_label_provenance.py',
    ROOT/'tests/test_m3w_source_label_provenance.py',ROOT/'tests/test_m3w_source_label_audit_runner.py',PUBLIC/'protocol.md')
sha,write=old.digest,old.save_event


def once(path,value):
    text=json.dumps(value,indent=2,sort_keys=True)+'\n'
    if path.exists():
        if path.read_text()!=text:raise ValueError('Immutable label-audit receipt differs:'+str(path))
        return
    path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w') as stream:
        stream.write(text);stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


def admission(register=False):
    cfg=json.loads(CONFIG.read_text());_,previous,packed,records=old.admission()
    flags=('controller_held_outcomes_opened','independent_roles_opened','write_label_repairs',
        'write_inference_features','rescore_proposals','training_executed','stage5c_executed','smc_enabled')
    if any(cfg[k] for k in flags) or (cfg['history_steps'],cfg['future_steps'],cfg['raw_frame_stride'])!=(8,12,12):
        raise ValueError('Fixed label-only scope and original8/12raw-stride12 task required')
    complete=json.loads((old.PUBLIC/'complete.json').read_text());check=json.loads((old.PUBLIC/'verification.json').read_text())
    ref=complete['arrays']['ids'];path=ROOT/ref['path']
    if sha(path)!=ref['sha256'] or check['complete_sha256']!=sha(old.PUBLIC/'complete.json') or not check['all_arrays_exact']:
        raise ValueError('Verified original fitting-query whitelist required')
    ids=np.load(path,mmap_mode='r',allow_pickle=False)
    if len(ids)!=cfg['expected_rows'] or np.any(np.diff(ids)<=0):raise ValueError('Original sorted149510query IDs required')
    arrays={}
    for k in ('recordings','agents','frames','origin','target_eval','valid'):
        path=old.PACKED/(k+'.npy')
        if sha(path)!=packed['arrays'][k]:raise ValueError('Frozen packed audit field changed:'+k)
        arrays[k]=np.load(path,mmap_mode='r',allow_pickle=False)
    if len(records)!=cfg['expected_recordings'] or set(arrays['recordings'][ids])!=set(records):
        raise ValueError('Exact74original fitting recordings required')
    for ri,item in records.items():
        take=ids[arrays['recordings'][ids]==ri]
        if len(take)!=item['rows'] or not np.array_equal(take,np.flatnonzero(arrays['recordings']==ri)):
            raise ValueError('No new,removed or rewindowed query allowed')
    once(REG,dict(bindings={str(p.relative_to(ROOT)):sha(p) for p in FILES},
        original_input_complete_sha256=sha(old.PUBLIC/'complete.json'),whitelist_sha256=ref['sha256'],
        packed_receipt_sha256=sha(old.PACKED/'receipt.json'),source_roles_sha256=sha(old.ROLES),
        packed_label_fields={k:packed['arrays'][k] for k in arrays},archive=previous['archive'],
        recordings={str(ri):dict(source_member=r['name'],rows=r['rows'],raw_sha256=r['raw_sha256']) for ri,r in records.items()},
        scope='existing_controller_fitting_queries_only_outcome_provenance_not_inputs',
        future_labels_read_for_audit_only=True,controller_held_outcomes_opened=False,independent_roles_opened=False,
        label_changes_written=False,forecast_scores_computed=False,training_executed=False))
    if not register:
        bindings=json.loads(REG.read_text())['bindings']
        for p in FILES:
            if sha(p)!=bindings[str(p.relative_to(ROOT))]:
                raise ValueError('Registered label audit changed')
        for p in (*FILES,REG):
            subprocess.run(['git','diff','--quiet','HEAD','--',str(p.relative_to(ROOT))],cwd=ROOT,check=True)
            subprocess.run(['git','ls-files','--error-unmatch',str(p.relative_to(ROOT))],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
    return cfg,arrays,ids,records


def execute(phase,resume=False):
    cfg,arrays,ids,records=admission();verify=phase=='verify';pilot=phase=='pilot';start=time.monotonic()
    finish=PUBLIC/('pilot.json' if pilot else 'audit.json')
    if finish.exists() and not verify:raise ValueError('Completed label audit immutable; use verify')
    if verify and not finish.exists():raise ValueError('Complete original audit required for replay')
    if not (pilot or verify) and not json.loads((PUBLIC/'pilot.json').read_text())['engineering_pass']:
        raise ValueError('Real raw-label alignment pilot required')
    def beat(**event):
        if time.monotonic()-start>cfg['hard_runtime_limit_seconds']:raise TimeoutError('12h boundary; resume unchanged audit')
        if shutil.disk_usage(ROOT).free<cfg['disk_reserve_bytes']:raise OSError('Preserve10GiB reserve')
        value=dict(pid=os.getpid(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**event)
        write(PRIVATE/'heartbeat.json',value)
        with (PRIVATE/'events.jsonl').open('a') as f:f.write(json.dumps(value)+'\n')
        print(json.dumps(value),flush=True)
    beat(state='runtime_ready',architecture=platform.machine(),workers=0,training_executed=False)
    archive=json.loads(REG.read_text())['archive'];path=ROOT/archive['path']
    if sha(path)!=archive['sha256']:raise ValueError('Original raw archive changed')
    results=[];refs=[];parsed=cached=0;selected=list(records.items())[:1] if pilot else list(records.items())
    with zipfile.ZipFile(path) as z:
        for ri,item in selected:
            take=ids[arrays['recordings'][ids]==ri];rp=PUBLIC/'records'/f'record{ri}.json'
            binding=dict(recording=ri,source_member=item['name'],locality=item['receipt']['locality_group'],
                registration_sha256=sha(REG),raw_sha256=item['raw_sha256'],ids_sha256=hashlib.sha256(take.tobytes()).hexdigest())
            if rp.exists() and not(verify or pilot):
                if not resume:raise ValueError('Partial audit requires explicit --resume')
                value=json.loads(rp.read_text())
                if value['binding']!=binding:raise ValueError('Resumed label-audit scope changed')
                cached+=1
            else:
                beat(state='parse_original_fitting_record',recording=ri,done=len(results),total=len(selected))
                with z.open(item['name']) as f:raw,_=old.read_raw_csv(f)
                if hashlib.sha256(raw.tobytes()).hexdigest()!=item['raw_sha256']:raise ValueError('Structured raw rows changed')
                value=dict(binding=binding,**api.audit_rows(raw,arrays['agents'][take],arrays['frames'][take],
                    arrays['origin'][take],arrays['target_eval'][take],arrays['valid'][take]))
                parsed+=1;del raw
                if verify:
                    if value!=json.loads(rp.read_text()):raise ValueError('Fresh raw-label receipt replay differs')
                elif not pilot:once(rp,value)
            results.append(value)
            if not pilot:refs.append(dict(path=str(rp.relative_to(ROOT)),sha256=sha(rp)))
            beat(state='record_complete',recording=ri,done=len(results),total=len(selected))
    if pilot:
        once(finish,dict(pid=os.getpid(),seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            engineering_pass=True,recording=selected[0][0],result=results[0],training_executed=False))
    else:
        sites=sorted({r['binding']['locality'] for r in results})
        if len(sites)!=cfg['expected_localities'] or sum(r['rows'] for r in results)!=cfg['expected_rows']:
            raise ValueError('Complete12locality149510query audit required')
        value=dict(result_source='fresh_run',registration_sha256=sha(REG),records=refs,
            totals=api.combine(results),by_locality={s:api.combine([r for r in results if r['binding']['locality']==s]) for s in sites},
            inference_features_written=False,label_changes_written=False,forecast_scores_computed=False,
            controller_held_outcomes_opened=False,independent_roles_opened=False,training_executed=False,
            metric_claim=False,seconds_claim=False,deployment_changed=False,stage5c_executed=False,smc_enabled=False)
        if verify:
            if value!=json.loads(finish.read_text()):raise ValueError('Complete raw-label audit replay differs')
            once(PUBLIC/'verification.json',dict(result_source='fresh_run',all74records_reparsed=True,all_scalar_receipts_exact=True,
                audit_sha256=sha(finish),rows=cfg['expected_rows']))
        else:once(finish,value)
    runtime=dict(pid=os.getpid(),seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        parsed_records=parsed,cached_verified_records=cached)
    write(PRIVATE/(phase+'_runtime.json'),runtime);beat(state=phase+'_complete',**{k:v for k,v in runtime.items() if k!='pid'})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=('register','pilot','run','verify'))
    parser.add_argument('--resume',action='store_true');args=parser.parse_args();PRIVATE.mkdir(parents=True,exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.phase=='register':admission(register=True)
        else:execute(args.phase,args.resume)
