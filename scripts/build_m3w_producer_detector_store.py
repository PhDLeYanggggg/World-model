"""Complete causal detector inputs for original SOURCE producer training roles."""
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

if platform.system()=='Darwin' and platform.machine()!='arm64':raise RuntimeError('Native arm64 required')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts import build_m3w_detector_history_store as old
from src.world_model.m3w_producer_detector_data import coverage

NAME='european_producer_detector_store_v1';PUBLIC=ROOT/'outputs/publication_readiness_2026_09'/NAME
PRIVATE=ROOT/'data/stage_cvpr2027_experiments'/NAME
CONFIG=ROOT/'configs/m3w_producer_detector_store_v1.json';REG=PUBLIC/'registration.json'
FILES=(Path(__file__),CONFIG,ROOT/'src/world_model/m3w_producer_detector_data.py',
    ROOT/'tests/test_m3w_producer_detector_data.py',PUBLIC/'protocol.md')
sha,once,write=old.digest,old.save,old.save_event


def admission(register=False):
    cfg=json.loads(CONFIG.read_text());_,previous,packed,_=old.admission()
    manifest=json.loads(old.SOURCE.read_text());roles=json.loads(old.ROLES.read_text())
    controller=json.loads((old.PUBLIC/'complete.json').read_text());check=json.loads((old.PUBLIC/'verification.json').read_text())
    if (not check['raw_reparsed'] or not check['all_arrays_exact'] or check['complete_sha256']!=sha(old.PUBLIC/'complete.json')
            or cfg['future_targets_read'] or cfg['independent_roles_opened']):raise ValueError('Verified old inputs and original roles required')
    arrays={}
    for key in ('sites','recordings','agents','frames','history','width'):
        path=old.PACKED/(key+'.npy')
        if sha(path)!=packed['arrays'][key]:raise ValueError('Original packed past array changed')
        arrays[key]=np.load(path,mmap_mode='r',allow_pickle=False)
    for ref in controller['arrays'].values():
        if sha(ROOT/ref['path'])!=ref['sha256']:raise ValueError('Old verified detector cache changed')
    old_ids=np.load(ROOT/controller['arrays']['ids']['path'],mmap_mode='r',allow_pickle=False)
    cov=coverage(arrays['sites'],arrays['recordings'],old_ids,packed['identity']['folds'])
    if (cov['rows']!=cfg['expected_rows'] or cov['records']!=cfg['expected_recordings']
            or cov['cached_rows']!=cfg['controller_rows'] or cov['cached_records']!=cfg['controller_recordings']):
        raise ValueError('Complete original producer rows required; no controller-only substitute')
    entries={r['source_member']:r for r in manifest['record_receipts']};items={}
    for ri,name in enumerate(packed['records']):
        access=old.require_source_training(roles,name);entry=entries[name];folder=ROOT/entry['directory']
        if sha(folder/'receipt.json')!=entry['receipt_sha256']:raise ValueError('Original input receipt changed')
        receipt=json.loads((folder/'receipt.json').read_text())
        if receipt['rows_sha256']!=access['rows_sha256'] or receipt['role']!='source_training':raise ValueError('Source-only rows required')
        ids=np.flatnonzero(arrays['recordings']==ri)
        if len(ids)!=receipt['target_rows'] or not len(ids):raise ValueError('Complete nonempty past-support record required')
        items[ri]=dict(name=name,folder=folder,receipt=receipt,raw_sha256=access['rows_sha256'],ids=ids)
    once(REG,dict(bindings={str(p.relative_to(ROOT)):sha(p) for p in FILES},
        original_source_registration_sha256=sha(old.REG),packed_receipt_sha256=sha(old.PACKED/'receipt.json'),
        controller_complete_sha256=sha(old.PUBLIC/'complete.json'),controller_verification_sha256=sha(old.PUBLIC/'verification.json'),
        archive=previous['archive'],recordings={str(ri):dict(source_member=v['name'],rows=len(v['ids']),raw_sha256=v['raw_sha256']) for ri,v in items.items()},
        scope='all_original_SOURCE_producer_inputs_not_new_readout',future_target_arrays_read=False,independent_roles_opened=False))
    if not register:
        import subprocess
        for p in (*FILES,REG):
            subprocess.run(['git','diff','--quiet','HEAD','--',str(p.relative_to(ROOT))],cwd=ROOT,check=True)
            subprocess.run(['git','ls-files','--error-unmatch',str(p.relative_to(ROOT))],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
    return cfg,packed,arrays,items,cov,controller


def runtime(cfg):
    started=time.monotonic()
    def beat(**event):
        if time.monotonic()-started>cfg['hard_runtime_limit_seconds']:raise TimeoutError('12h boundary; resume same full inputs')
        if shutil.disk_usage(ROOT).free<cfg['disk_reserve_bytes']:raise OSError('Preserve10GiB reserve')
        if sum(p.stat().st_size for p in PRIVATE.rglob('*') if p.is_file())>cfg['private_cache_cap_bytes']:raise OSError('Input cache cap exceeded')
        value=dict(pid=os.getpid(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**event);write(PRIVATE/'heartbeat.json',value)
        with (PRIVATE/'events.jsonl').open('a') as f:f.write(json.dumps(value)+'\n')
        print(json.dumps(value),flush=True)
    beat(state='runtime_ready',architecture=platform.machine(),workers=0,future_targets_read=False)
    return started,beat


def execute(phase,resume=False):
    cfg,packed,arrays,items,cov,controller=admission();verify=phase=='verify';pilot=phase=='pilot'
    finish=PUBLIC/('pilot.json' if pilot else 'complete.json')
    if finish.exists() and not verify:raise ValueError('Completed input phase immutable')
    start,beat=runtime(cfg);once(PUBLIC/'coverage.json',cov)
    archive=json.loads(REG.read_text())['archive'];path=ROOT/archive['path']
    if sha(path)!=archive['sha256']:raise ValueError('Original raw archive changed')
    cached={json.loads((ROOT/r['path']).read_text())['binding']['recording']:r for r in controller['receipts']}
    selected=[(ri,v) for ri,v in items.items() if ri not in cached][:1] if pilot else list(items.items())
    blocks=[];refs=[];parsed=reused=resumed=0;(PRIVATE/'records').mkdir(exist_ok=True)
    with zipfile.ZipFile(path) as z:
        for ri,item in selected:
            ids=item['ids'];cp=PRIVATE/'records'/f'record{ri}.npz';rp=cp.with_suffix('.json')
            binding=dict(registration_sha256=sha(REG),recording=ri,source_member=item['name'],rows=len(ids),raw_sha256=item['raw_sha256'])
            if resume and not verify and cp.exists() and rp.exists():
                rec=json.loads(rp.read_text())
                if rec['binding']!=binding or sha(cp)!=rec['sha256']:raise ValueError('Resumed input checkpoint changed')
                with np.load(cp,allow_pickle=False) as f:out={k:f[k].copy() for k in f.files}
                resumed+=1
            else:
                if not verify and ri in cached:
                    oldref=cached[ri];oldpath=ROOT/oldref['path'];r=json.loads(oldpath.read_text())
                    if sha(oldpath)!=oldref['sha256'] or sha(oldpath.with_suffix('.npz'))!=r['sha256']:raise ValueError('Old verified input block changed')
                    with np.load(oldpath.with_suffix('.npz'),allow_pickle=False) as f:out={k:f[k].copy() for k in f.files}
                    checks=r['prefix_checks'];reused+=1
                else:
                    beat(state='parse_original_source_record',recording=ri,done=len(blocks),total=len(selected))
                    with z.open(item['name']) as f:raw,_=old.read_raw_csv(f)
                    if hashlib.sha256(raw.tobytes()).hexdigest()!=item['raw_sha256']:raise ValueError('Original structured raw rows changed')
                    expected=old.input_match(item,arrays['agents'][ids],arrays['frames'][ids])
                    boxes,confidence,grid=old.lookup_history(raw,arrays['agents'][ids],arrays['frames'][ids],expected)
                    np.testing.assert_array_equal((boxes[...,:2]+boxes[...,2:])/2,arrays['history'][ids])
                    np.testing.assert_array_equal(boxes[:,-1,2]-boxes[:,-1,0],arrays['width'][ids])
                    probes=sorted(set([0,len(ids)//2,len(ids)-1]));checks=len(probes)
                    for at in probes:
                        q=arrays['frames'][ids[at]];args=(arrays['agents'][ids[at:at+1]],arrays['frames'][ids[at:at+1]],expected[at:at+1])
                        bb,cc,ff=old.lookup_history(raw[raw['frame']<=q],*args)
                        np.testing.assert_array_equal(bb,boxes[at:at+1]);np.testing.assert_array_equal(cc,confidence[at:at+1]);np.testing.assert_array_equal(ff,grid[at:at+1])
                    out=dict(ids=ids,features=old.features(boxes,confidence),confidence=confidence,
                        recordings=arrays['recordings'][ids].copy(),agents=arrays['agents'][ids].copy(),frames=arrays['frames'][ids].copy())
                    parsed+=1;del raw,boxes,confidence,grid,expected
                np.testing.assert_array_equal(out['ids'],ids)
                for key in ('recordings','agents','frames'):np.testing.assert_array_equal(out[key],arrays[key][ids])
                rec=dict(binding=binding,prefix_checks=checks,
                    arrays={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in out.items()})
                if verify:
                    prev=json.loads(rp.read_text());rec['sha256']=prev['sha256']
                    if rec!=prev or sha(cp)!=prev['sha256']:raise ValueError('Full raw replay input receipt changed')
                    with np.load(cp,allow_pickle=False) as f:
                        for k,v in out.items():np.testing.assert_array_equal(v,f[k])
                else:old.checkpoint(cp,out);rec['sha256']=sha(cp);once(rp,rec)
            blocks.append(out);refs.append(dict(path=str(rp.relative_to(ROOT)),sha256=sha(rp),rows=len(ids)))
            beat(state='record_complete',recording=ri,done=len(blocks),total=len(selected))
    if pilot:
        once(finish,dict(pid=os.getpid(),seconds=time.monotonic()-start,rows=len(blocks[0]['ids']),
            recording=selected[0][0],engineering_pass=True,exact_raw_past_alignment=True,scoring_or_training_run=False))
    else:
        joined={k:np.concatenate([b[k] for b in blocks]) for k in blocks[0]};order=np.argsort(joined['ids'])
        joined={k:v[order] for k,v in joined.items()};np.testing.assert_array_equal(joined['ids'],np.arange(cfg['expected_rows']))
        old_ids=np.load(ROOT/controller['arrays']['ids']['path'],mmap_mode='r',allow_pickle=False)
        for key,ref in controller['arrays'].items():np.testing.assert_array_equal(joined[key][old_ids],np.load(ROOT/ref['path'],mmap_mode='r',allow_pickle=False))
        dest=PRIVATE/'store';dest.mkdir(exist_ok=True);saved={}
        for key,v in joined.items():
            cp=dest/(key+'.npy')
            if verify:np.testing.assert_array_equal(v,np.load(cp,mmap_mode='r',allow_pickle=False))
            else:
                temp=cp.with_suffix('.tmp')
                with temp.open('wb') as f:np.save(f,v,allow_pickle=False)
                os.replace(temp,cp)
            saved[key]=dict(path=str(cp.relative_to(ROOT)),sha256=sha(cp),bytes=cp.stat().st_size,shape=list(v.shape))
        result=dict(result_source='full_original_SOURCE_producer_past_input_alignment',rows=cfg['expected_rows'],recordings=len(refs),
            arrays=saved,receipts=refs,feature_schema=list(old.NAMES),registration_sha256=sha(REG),
            prefix_checks=sum(json.loads((ROOT/r['path']).read_text())['prefix_checks'] for r in refs),
            controller_overlap_exact=True,future_targets_read=False,independent_roles_opened=False,
            sensor_asof_certified=False,training_run=False,deployment_changed=False,stage5c_executed=False,smc_enabled=False)
        if verify:
            if result!=json.loads(finish.read_text()):raise ValueError('Full producer input report replay changed')
            once(PUBLIC/'verification.json',dict(raw_reparsed=True,all_arrays_exact=True,rows=cfg['expected_rows'],complete_sha256=sha(finish)))
        else:once(finish,result)
    record=dict(pid=os.getpid(),seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        parsed_records=parsed,cached_controller_records=reused,resumed_records=resumed)
    write(PRIVATE/(phase+'_runtime.json'),record);beat(state=phase+'_complete',**{k:v for k,v in record.items() if k!='pid'})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=('register','pilot','build','verify'))
    parser.add_argument('--resume',action='store_true');args=parser.parse_args();PRIVATE.mkdir(parents=True,exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.phase=='register':admission(register=True)
        else:execute(args.phase,args.resume)
