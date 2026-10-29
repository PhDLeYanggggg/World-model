"""Recover original causal detection histories on the exact fitting cohort."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import sys
import time
import zipfile

if platform.system()=='Darwin' and platform.machine()!='arm64':
    raise RuntimeError('Use native arm64 .venv-pytorch')
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key,'4')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.fetch_m3w_european_squares import digest,save
from src.evaluation.m3w_european_squares_raw_v2 import read_raw_csv
from src.evaluation.m3w_european_squares_roles import require_source_training
from src.world_model.m3w_detector_history_features import lookup_history,features,NAMES

NAME='european_detector_history_store_v1'
PUBLIC=ROOT/'outputs/publication_readiness_2026_09'/NAME
PRIVATE=ROOT/'data/stage_cvpr2027_experiments'/NAME
CONFIG=ROOT/'configs/m3w_detector_history_store_v1.json';REG=PUBLIC/'registration_v2.json'
SOURCE=ROOT/'outputs/publication_readiness_2026_09/european_squares_source_v1/analysis.json'
ROLES=ROOT/'outputs/publication_readiness_2026_09/european_squares_roles_v1/roles.json'
PACKED=ROOT/'data/stage_cvpr2027_experiments/european_source_forecast_v1/packed'
AUDIT=ROOT/'outputs/publication_readiness_2026_09/european_static_scene_audit_v1'
FILES=(Path(__file__),CONFIG,ROOT/'src/world_model/m3w_detector_history_features.py',
       ROOT/'tests/test_m3w_detector_history_features.py',ROOT/'tests/test_m3w_detector_history_builder.py',PUBLIC/'protocol.md')


def admission(register=False):
    cfg=json.loads(CONFIG.read_text());old=json.loads((AUDIT/'audit.json').read_text())
    check=json.loads((AUDIT/'verification.json').read_text())
    if not check['exact_replay'] or check['audit_sha256']!=digest(AUDIT/'audit.json'):
        raise ValueError('Verified fitting-row identity required')
    roles=json.loads(ROLES.read_text());manifest=json.loads(SOURCE.read_text())
    packed=json.loads((PACKED/'receipt.json').read_text())
    counts={r['recording']:r['rows'] for r in old['rows']}
    records={r['source_member']:r for r in manifest['record_receipts']}
    whitelist={}
    for ri,n in sorted(counts.items()):
        name=packed['records'][ri];access=require_source_training(roles,name)
        entry=records[name];folder=ROOT/entry['directory']
        if digest(folder/'receipt.json')!=entry['receipt_sha256']:
            raise ValueError('Original source input receipt changed')
        receipt=json.loads((folder/'receipt.json').read_text())
        if receipt['rows_sha256']!=access['rows_sha256']:
            raise ValueError('Raw source-row binding changed')
        whitelist[ri]=dict(name=name,rows=n,folder=folder,receipt=receipt,raw_sha256=access['rows_sha256'])
    if len(whitelist)!=cfg['expected_recordings'] or sum(counts.values())!=cfg['expected_rows']:
        raise ValueError('Original74record149510row fitting roster required')
    source_file=json.loads((SOURCE.parent.parent/'european_squares_intake_v1/trajectory_manifest.json').read_text())['private_file']
    identity=dict(bindings={str(p.relative_to(ROOT)):digest(p) for p in FILES},
        prior_failed_registration_sha256=digest(PUBLIC/'registration.json'),
        execution_amendment='rename_runtime_receipt_local_that_shadowed_runtime_function_before_raw_access',
        roles_sha256=digest(ROLES),source_manifest_sha256=digest(SOURCE),
        fitting_audit_sha256=digest(AUDIT/'audit.json'),packed_receipt_sha256=digest(PACKED/'receipt.json'),
        archive=source_file,recordings={str(ri):dict(source_member=r['name'],rows=r['rows'],raw_rows_sha256=r['raw_sha256']) for ri,r in whitelist.items()},
        feature_names=list(NAMES),future_target_arrays_read=False,independent_roles_opened=False)
    save(REG,identity)
    if not register:
        import subprocess
        for p in (*FILES,REG):
            subprocess.run(['git','diff','--quiet','HEAD','--',str(p.relative_to(ROOT))],cwd=ROOT,check=True)
            subprocess.run(['git','ls-files','--error-unmatch',str(p.relative_to(ROOT))],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
    return cfg,identity,packed,whitelist


def runtime(cfg):
    start=time.monotonic();PRIVATE.mkdir(parents=True,exist_ok=True)
    def beat(**event):
        if time.monotonic()-start>cfg['hard_runtime_limit_seconds']:
            raise TimeoutError('12h boundary; resume identical full extraction')
        if shutil.disk_usage(ROOT).free<cfg['disk_reserve_bytes']:
            raise OSError('Preserve10GiB reserve')
        if sum(p.stat().st_size for p in PRIVATE.rglob('*') if p.is_file())>cfg['private_cache_cap_bytes']:
            raise OSError('Private feature cache cap exceeded')
        value=dict(pid=os.getpid(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**event)
        save_event(PRIVATE/'heartbeat.json',value)
        with (PRIVATE/'events.jsonl').open('a') as stream:stream.write(json.dumps(value)+'\n')
        print(json.dumps(value),flush=True)
    beat(state='runtime_ready',architecture=platform.machine(),workers=0)
    return start,beat


def save_event(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value)+'\n');os.replace(temp,path)


def input_array(item,name):
    ref=item['receipt']['arrays']['input_'+name];path=item['folder']/ref['name']
    if digest(path)!=ref['sha256']:
        raise ValueError('Original past-input array changed:'+name)
    return np.load(path,mmap_mode='r',allow_pickle=False)


def input_match(item,agents,frames):
    a=input_array(item,'agent_id');f=input_array(item,'query_frame')
    dtype=np.dtype([('frame','<i8'),('agent','<i8')])
    keys=np.empty(len(a),dtype=dtype);keys['frame']=f;keys['agent']=a
    order=np.argsort(keys,kind='stable');query=np.empty(len(agents),dtype=dtype)
    query['frame']=frames;query['agent']=agents;pos=np.searchsorted(keys[order],query)
    if np.any(pos>=len(keys)) or not np.array_equal(keys[order[pos]],query):
        raise ValueError('Packed current agent/frame failed original input alignment')
    ids=order[pos]
    if not input_array(item,'history_valid')[ids].all():
        raise ValueError('Only original complete pasts are eligible')
    return input_array(item,'history_boxes')[ids].copy()


def checkpoint(path,arrays):
    temp=path.with_suffix('.tmp')
    with temp.open('wb') as stream:np.savez_compressed(stream,**arrays)
    os.replace(temp,path)


def execute(phase,resume=False):
    cfg,identity,packed,whitelist=admission();verify=phase=='verify';pilot=phase=='pilot'
    finish=PUBLIC/('pilot.json' if pilot else 'complete.json')
    if finish.exists() and not verify:raise ValueError('Completed extraction immutable')
    if not pilot and not json.loads((PUBLIC/'pilot.json').read_text())['engineering_pass']:
        raise ValueError('Real one-record input alignment pilot required')
    start,beat=runtime(cfg);beat(state='verify_raw_archive',bytes=identity['archive']['bytes'])
    archive=ROOT/identity['archive']['path']
    if digest(archive)!=identity['archive']['sha256']:raise ValueError('Original raw archive changed')
    arrays={}
    for key in ('recordings','agents','frames','history','width'):
        path=PACKED/(key+'.npy')
        if digest(path)!=packed['arrays'][key]:raise ValueError('Packed causal identity changed:'+key)
        arrays[key]=np.load(path,mmap_mode='r',allow_pickle=False)
    receipts=[];blocks=[];new_parse=0;cache_hits=0
    (PRIVATE/'records').mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for ri,item in list(whitelist.items())[:1] if pilot else whitelist.items():
            before=time.monotonic();ids=np.flatnonzero(arrays['recordings']==ri)
            if len(ids)!=item['rows']:raise ValueError('Exact original fitting record row count changed')
            binding=dict(registration_sha256=digest(REG),recording=ri,source_member=item['name'],rows=len(ids),
                ids_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),raw_rows_sha256=item['raw_sha256'])
            path=PRIVATE/'records'/f'record{ri}.npz';refpath=path.with_suffix('.json')
            if path.exists() and refpath.exists() and resume and not verify:
                receipt=json.loads(refpath.read_text())
                if receipt['binding']!=binding or receipt['sha256']!=digest(path):raise ValueError('Resumed past-feature checkpoint changed')
                with np.load(path,allow_pickle=False) as f:out={k:f[k].copy() for k in f.files}
                cache_hits+=1
            else:
                beat(state='parse_fitting_record',recording=ri,done=len(receipts),total=len(whitelist),source_member=item['name'])
                with z.open(item['name']) as stream:raw,_=read_raw_csv(stream)
                if hashlib.sha256(raw.tobytes()).hexdigest()!=item['raw_sha256']:raise ValueError('Full original structured raw rows changed')
                expected=input_match(item,arrays['agents'][ids],arrays['frames'][ids])
                boxes,confidence,grid=lookup_history(raw,arrays['agents'][ids],arrays['frames'][ids],expected)
                np.testing.assert_array_equal((boxes[...,:2]+boxes[...,2:])/2,arrays['history'][ids])
                np.testing.assert_array_equal(boxes[:,-1,2]-boxes[:,-1,0],arrays['width'][ids])
                for at in sorted(set([0,len(ids)//2,len(ids)-1])):
                    q=arrays['frames'][ids[at]]
                    bb,cc,ff=lookup_history(raw[raw['frame']<=q],arrays['agents'][ids[at:at+1]],
                        arrays['frames'][ids[at:at+1]],expected[at:at+1])
                    np.testing.assert_array_equal(bb,boxes[at:at+1]);np.testing.assert_array_equal(cc,confidence[at:at+1])
                    np.testing.assert_array_equal(ff,grid[at:at+1])
                out=dict(ids=ids,features=features(boxes,confidence),confidence=confidence,
                    recordings=arrays['recordings'][ids].copy(),agents=arrays['agents'][ids].copy(),frames=arrays['frames'][ids].copy())
                receipt=dict(binding=binding,arrays={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in out.items()},
                    prefix_checks=len(set([0,len(ids)//2,len(ids)-1])),all_boxes_and_midpoints_exact=True,
                    confidence_range=[float(confidence.min()),float(confidence.max())],zero_area_observations=int((np.prod(boxes[...,2:]-boxes[...,:2],axis=-1)==0).sum()),
                    observed_box_min=boxes.reshape(-1,4).min(0).tolist(),observed_box_max=boxes.reshape(-1,4).max(0).tolist())
                del raw,expected,boxes,confidence,grid
                if verify:
                    previous=json.loads(refpath.read_text());receipt.update(sha256=previous['sha256'])
                    if receipt!=previous or digest(path)!=previous['sha256']:raise ValueError('Raw replay feature receipt changed')
                    with np.load(path,allow_pickle=False) as old:
                        for k,value in out.items():np.testing.assert_array_equal(value,old[k])
                else:
                    checkpoint(path,out);receipt['sha256']=digest(path);save(refpath,receipt)
                new_parse+=1
            receipts.append(dict(path=str(refpath.relative_to(ROOT)),sha256=digest(refpath),rows=len(ids)))
            blocks.append(out)
            beat(state='record_complete',recording=ri,done=len(receipts),total=1 if pilot else len(whitelist),seconds=time.monotonic()-before)
    if pilot:
        save(finish,dict(pid=os.getpid(),seconds=time.monotonic()-start,rows=len(blocks[0]['ids']),
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,engineering_pass=True,
            confidence_and_boxes_exact=True,real_prefix_checks=3,query_future_labels_read=False,
            projected_model_training_or_scientific_gate=False))
    else:
        joined={k:np.concatenate([b[k] for b in blocks]) for k in blocks[0]};order=np.argsort(joined['ids'])
        joined={k:v[order] for k,v in joined.items()}
        if len(joined['ids'])!=cfg['expected_rows'] or len(np.unique(joined['ids']))!=len(joined['ids']):raise ValueError('Complete unique fitting-row roster required')
        refs={};folder=PRIVATE/'store';folder.mkdir(exist_ok=True)
        for key,value in joined.items():
            path=folder/(key+'.npy')
            if verify:np.testing.assert_array_equal(value,np.load(path,mmap_mode='r',allow_pickle=False))
            else:
                temp=path.with_suffix('.tmp')
                with temp.open('wb') as stream:np.save(stream,value,allow_pickle=False)
                os.replace(temp,path)
            refs[key]=dict(path=str(path.relative_to(ROOT)),sha256=digest(path),shape=list(value.shape),bytes=path.stat().st_size)
        result=dict(result_source='fresh_raw_reparse_and_exact_past_box_confidence_alignment',registration_sha256=digest(REG),
            rows=len(joined['ids']),recordings=len(receipts),receipts=receipts,arrays=refs,
            feature_schema=list(NAMES),confidence_quantiles=np.quantile(joined['confidence'],[0,.1,.5,.9,1]).tolist(),
            feature_min=joined['features'].min(0).tolist(),feature_max=joined['features'].max(0).tolist(),
            real_prefix_checks=sum(json.loads((ROOT/r['path']).read_text())['prefix_checks'] for r in receipts),
            query_future_labels_read=False,independent_roles_opened=False,metric_claim=False,seconds_claim=False,
            sensor_asof_certified=False,training_executed=False,deployment_changed=False,stage5c_executed=False,smc_enabled=False)
        if verify:
            if result!=json.loads(finish.read_text()):raise ValueError('Complete feature-store report replay differs')
            save(PUBLIC/'verification.json',dict(raw_reparsed=True,all_arrays_exact=True,rows=cfg['expected_rows'],complete_sha256=digest(finish)))
        else:save(finish,result)
    runtime_record=dict(pid=os.getpid(),seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        newly_parsed_records=new_parse,verified_cache_records=cache_hits)
    save_event(PRIVATE/('verify_runtime.json' if verify else phase+'_runtime.json'),runtime_record)
    beat(state='verified' if verify else phase+'_complete',**{k:v for k,v in runtime_record.items() if k!='pid'})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=('register','pilot','build','verify'))
    parser.add_argument('--resume',action='store_true');args=parser.parse_args();PRIVATE.mkdir(parents=True,exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.phase=='register':admission(register=True)
        else:execute(args.phase,args.resume)
