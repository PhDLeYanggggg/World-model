"""Build and fully replay past-only inputs for the frozen selection localities."""
import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import signal
import subprocess
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.data_unification import m3w_model_selection_inputs as api
from src.evaluation.m3w_european_squares_raw_v2 import read_raw_csv
from src.evaluation.m3w_european_squares_intake import past_scene,safe_zip_member
from scripts.build_m3w_european_squares_source import store_array

PUBLIC=ROOT/'outputs/publication_readiness_2026_09/european_model_selection_inputs_v1'
PRIVATE=ROOT/'data/stage_cvpr2027_experiments/european_model_selection_inputs_v1'
CONFIG=ROOT/'configs/m3w_model_selection_inputs_v1.json';REG=PUBLIC/'registration.json'
ROLES=ROOT/'outputs/publication_readiness_2026_09/european_squares_roles_v1/roles.json'
MANIFEST=ROOT/'outputs/publication_readiness_2026_09/european_squares_intake_v1/trajectory_manifest.json'
FILES=(Path(__file__),CONFIG,ROOT/'src/data_unification/m3w_model_selection_inputs.py',
    ROOT/'tests/test_m3w_model_selection_inputs.py',PUBLIC/'protocol.md')
DEPENDENCIES=('src/data_unification/m3w_european_squares_source.py','src/evaluation/m3w_european_squares_raw_v2.py',
    'src/evaluation/m3w_european_squares_intake.py','src/evaluation/m3w_european_squares_roles.py',
    'scripts/build_m3w_european_squares_source.py','scripts/fetch_m3w_european_squares.py')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n');temp.replace(path)


def once(path,value):
    if path.exists():
        if json.loads(path.read_text())!=value:raise ValueError('Immutable artifact changed: '+str(path))
    else:write(path,value)


def admission(register=False):
    cfg=json.loads(CONFIG.read_text());roles=json.loads(ROLES.read_text())
    selected=sorted((r for r in roles['recordings'] if r['role']=='model_selection_reserved'),key=lambda r:r['source_member'])
    if len(selected)!=28 or len({r['locality_group'] for r in selected})!=6:
        raise ValueError('All original28recordings/six selection localities required')
    if (cfg['per_recording_query_cap']!=256 or (cfg['observation_steps'],cfg['prediction_steps'],cfg['raw_frame_stride'])!=(8,12,12)
            or any(cfg[k] for k in ('future_target_arrays_constructed','forecast_errors_opened','normalization_fitted',
                'training_access','risk_calibration_access','confirmation_access','stage5c_executed','smc_enabled'))):
        raise ValueError('Input-only role, original grid and budgets required')
    for entry in roles['dependency_bindings'].values():
        if sha(ROOT/entry['path'])!=entry['sha256']:raise ValueError('Original role evidence changed')
    for row in selected:api.require_input_role(roles,row['source_member'])
    source=json.loads(MANIFEST.read_text())['private_file']
    value=dict(bindings={str(p.relative_to(ROOT)):sha(p) for p in FILES},
        dependencies={p:sha(ROOT/p) for p in DEPENDENCIES},roles_sha256=sha(ROLES),archive_manifest_sha256=sha(MANIFEST),
        archive=source,members=selected,scope='model_selection_past_inputs_only_no_predictive_outcomes',
        source_training_accessor_unchanged=True,model_selection_forecast_readout_authorized=False,
        risk_calibration_access=False,confirmation_access=False,deployment_changed=False,
        stage5c_executed=False,smc_enabled=False)
    if register:once(REG,value)
    if json.loads(REG.read_text())!=value:raise ValueError('Input conversion registration changed')
    if not register:
        for path in (*FILES,REG):
            old=subprocess.run(['git','show','HEAD:'+str(path.relative_to(ROOT))],cwd=ROOT,capture_output=True,check=True).stdout
            if hashlib.sha256(old).hexdigest()!=sha(path):raise ValueError('Commit input definition before raw access')
    return cfg,roles,selected,source


def prefix_checks(rows,values,queries):
    checks=0
    for qi in sorted({0,len(queries)//2,len(queries)-1}) if len(queries) else []:
        ref=past_scene(rows,int(queries[qi]));lo,hi=values['query_offsets'][qi:qi+2]
        if set(ref)!=set(map(int,values['agent_id'][lo:hi])):raise ValueError('Visible-prefix agent mismatch')
        for i in range(lo,hi):
            item=ref[int(values['agent_id'][i])]
            for field,other in (('history_xy','box_center'),('history_boxes','boxes'),('history_valid','valid'),
                    ('velocity_causal_fd','velocity_causal_fd'),('velocity_valid','velocity_valid')):
                api.np.testing.assert_array_equal(values[field][i],item[other])
        checks+=1
    return checks


def main(phase,resume=False):
    cfg,roles,selected,source=admission(phase=='register')
    if phase=='register':return
    start=time.monotonic();signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt('Resume unchanged input conversion')))
    def guard():
        if shutil.disk_usage(ROOT).free<cfg['disk_reserve_bytes']:raise OSError('Keep10GiB disk reserve; no silent query reduction')
        if time.monotonic()-start>cfg['hard_runtime_limit_seconds']:raise TimeoutError('12h phase boundary; resume unchanged')
        if sum(p.stat().st_size for p in PRIVATE.rglob('*.npy'))>cfg['private_cache_cap_bytes']:raise OSError('1GiB input-cache cap')
    def beat(state,**event):
        guard();value=dict(pid=os.getpid(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),phase=phase,state=state,**event)
        write(PRIVATE/'heartbeat.json',value);print(json.dumps(value),flush=True)
    final=PUBLIC/('pilot.json' if phase=='pilot' else 'verification.json' if phase=='verify' else 'analysis.json')
    if final.exists():raise ValueError('Completed conversion phase immutable')
    if phase=='verify' and not (PUBLIC/'analysis.json').exists():raise ValueError('Full conversion required before replay')
    if phase=='build' and not json.loads((PUBLIC/'pilot.json').read_text())['engineering_pass']:
        raise ValueError('Real largest-record pilot required')
    if phase=='pilot':selected=[min(selected,key=lambda r:(-r['raw_rows'],r['source_member']))]
    beat('verify_archive',bytes=source['bytes'])
    archive=ROOT/source['path']
    if archive.stat().st_size!=source['bytes'] or sha(archive)!=source['sha256']:raise ValueError('Raw archive changed')
    receipts=[];raw_parsed=0;cached=0
    with zipfile.ZipFile(archive) as z:
        for index,row in enumerate(selected):
            guard();member=row['source_member'];api.require_input_role(roles,member)
            directory=PRIVATE/'records'/hashlib.sha256(member.encode()).hexdigest();path=directory/'receipt.json'
            if path.exists() and phase!='verify':
                if not resume:raise ValueError('Explicit resume required for existing input checkpoint')
                old=json.loads(path.read_text())
                if old['registration_sha256']!=sha(REG) or old['rows_sha256']!=row['rows_sha256']:raise ValueError('Cached recording identity changed')
                values,query=api.load_record(directory,roles,member,registration_sha256=sha(REG))
                if old['input_statistics']!=api.input_statistics(values):raise ValueError('Cached input statistics changed')
                receipts.append(old);cached+=1;beat('cached_verified_record',records=index+1,total=len(selected));continue
            info=z.getinfo(member)
            if safe_zip_member(info)!=member:raise ValueError('Unexpected ZIP member')
            beat('parsing_record',record=index+1,total=len(selected),member=member)
            with z.open(info) as f:rows,_=read_raw_csv(f)
            raw_parsed+=1
            if len(rows)!=row['raw_rows'] or hashlib.sha256(rows.tobytes()).hexdigest()!=row['rows_sha256']:
                raise ValueError('Raw row identity changed')
            query,support=api.source.choose_query_frames(rows,maximum=cfg['per_recording_query_cap'])
            if support['all_past_eligible_targets']!=row['past_eligible_k8_stride12']:raise ValueError('Original past support changed')
            values=api.build_inputs(rows,query);checks=prefix_checks(rows,values,query);del rows
            arrays={};directory.mkdir(parents=True,exist_ok=True)
            for key,value in dict(**{'input_'+k:v for k,v in values.items()},index_query_frames=query).items():
                dest=directory/(key+'.npy')
                if phase=='verify':api.np.testing.assert_array_equal(api.np.load(dest,allow_pickle=False,mmap_mode='r'),value)
                else:
                    guard()
                    if shutil.disk_usage(ROOT).free-value.nbytes<cfg['disk_reserve_bytes']:raise OSError('Preserve disk reserve')
                    store_array(dest,value)
                arrays[key]=dict(name=dest.name,shape=list(value.shape),dtype=str(value.dtype),bytes=dest.stat().st_size,sha256=sha(dest))
            receipt=dict(source_member=member,locality_group=row['locality_group'],rows_sha256=row['rows_sha256'],
                role='model_selection_inputs_only',registration_sha256=sha(REG),arrays=arrays,support=support,
                independent_prefix_query_checks=checks,input_statistics=api.input_statistics(values),
                cached_visible_scoped_agents=int(len(api.np.unique(values['agent_id']))),
                future_target_arrays_constructed=False,forecast_errors_opened=False,training_access=False)
            if phase=='verify':
                if receipt!=json.loads(path.read_text()):raise ValueError('Complete raw/input replay differs')
            else:once(path,receipt)
            receipts.append(receipt);beat('record_complete',records=index+1,total=len(selected),**receipt['input_statistics'])
    sites={s:dict(recordings=sum(r['locality_group']==s for r in receipts),
        target_rows=sum(r['input_statistics']['target_rows'] for r in receipts if r['locality_group']==s),
        visible_rows=sum(r['input_statistics']['visible_rows'] for r in receipts if r['locality_group']==s)) for s in sorted({r['locality_group'] for r in receipts})}
    core=dict(registration_sha256=sha(REG),roles_sha256=sha(ROLES),recordings=len(receipts),localities=sites,
        target_rows=sum(r['input_statistics']['target_rows'] for r in receipts),
        visible_rows=sum(r['input_statistics']['visible_rows'] for r in receipts),
        query_frames=sum(r['input_statistics']['queries'] for r in receipts),
        independent_prefix_query_checks=sum(r['independent_prefix_query_checks'] for r in receipts),
        cache_bytes=sum(v['bytes'] for r in receipts for v in r['arrays'].values()),
        record_receipts=[dict(path=str((PRIVATE/'records'/hashlib.sha256(r['source_member'].encode()).hexdigest()/'receipt.json').relative_to(ROOT)),
            sha256=sha(PRIVATE/'records'/hashlib.sha256(r['source_member'].encode()).hexdigest()/'receipt.json'),
            source_member=r['source_member'],locality_group=r['locality_group'],input_statistics=r['input_statistics']) for r in receipts],
        future_target_arrays_constructed=False,forecast_errors_opened=False,normalization_fitted=False,
        calibration_or_confirmation_members_deserialized=0,original_training_accessor_changed=False)
    metadata=dict(pid=os.getpid(),seconds=time.monotonic()-start,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if platform.system()=='Darwin' else 1024),
        parsed_raw_records=raw_parsed,cached_verified_records=cached,source_role='model_selection_inputs_only',
        original_role_manifest_unchanged=sha(ROLES)==json.loads(REG.read_text())['roles_sha256'],
        predictive_generalization_result='not_run',deployment_changed=False,stage5c_executed=False,smc_enabled=False)
    if not metadata['original_role_manifest_unchanged']:raise ValueError('Frozen role manifest changed')
    if phase=='verify':
        if json.loads((PUBLIC/'analysis.json').read_text())['cohort']!=core:raise ValueError('Full conversion summary differs')
        once(final,dict(**metadata,all28_raw_reparsed_all_input_arrays_exact=True,analysis_sha256=sha(PUBLIC/'analysis.json')))
    elif phase=='pilot':once(final,dict(**metadata,cohort=core,engineering_pass=True,largest_raw_record_pilot=True))
    else:
        if len(receipts)!=28 or len(sites)!=6:raise ValueError('All original selection recordings required')
        once(final,dict(**metadata,cohort=core))
    beat('complete',records=len(receipts),seconds=time.monotonic()-start)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=('register','pilot','build','verify'))
    parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    PRIVATE.mkdir(parents=True,exist_ok=True);PUBLIC.mkdir(parents=True,exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);main(args.phase,args.resume)
