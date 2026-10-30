"""Matched detector inputs on all nine actual active dimensionless producers."""
import argparse
import fcntl
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

if platform.system()=='Darwin' and platform.machine()!='arm64':raise RuntimeError('Native arm64 before Torch')
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ.setdefault(key,'4')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts import run_m3w_european_dimensionless_refit as parent
from scripts import run_m3w_european_source_forecast as source
from scripts import build_m3w_producer_detector_store as store
from src.world_model import m3w_active_detector_forecaster as api
from src.world_model.m3w_detector_cost import representation
from src.world_model.m3w_producer_detector_data import normalize
from src.world_model.m3w_multiaction_cost import exact

NAME='european_active_detector_forecaster_v1';PUBLIC=ROOT/'outputs/publication_readiness_2026_09'/NAME
PRIVATE=ROOT/'data/stage_cvpr2027_experiments'/NAME
CONFIG=ROOT/'configs/m3w_active_detector_forecaster_v1.json';REG=PUBLIC/'registration_v2.json'
sha,once,write=source.digest,source.immutable_json,source.json_write
FILES=(Path(__file__),CONFIG,ROOT/'src/world_model/m3w_active_detector_forecaster.py',
    ROOT/'tests/test_m3w_active_detector_forecaster.py',ROOT/'tests/test_m3w_active_detector_runner.py',ROOT/'src/world_model/m3w_detector_forecaster.py',
    ROOT/'src/world_model/m3w_detector_cost.py',ROOT/'src/world_model/m3w_producer_detector_data.py',PUBLIC/'protocol.md')


def committed(path):
    subprocess.run(['git','diff','--quiet','HEAD','--',str(path.relative_to(ROOT))],cwd=ROOT,check=True)
    subprocess.run(['git','ls-files','--error-unmatch',str(path.relative_to(ROOT))],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)


def admission(register=False):
    cfg=json.loads(CONFIG.read_text());_,reg,data,jobs,identity=parent.load();refs=parent.endpoints(jobs)
    if (cfg['training']!=reg['training'] or cfg['seeds']!=reg['seeds'] or len(jobs)!=9 or cfg['parent_models']!=9
            or cfg['architecture']!=dict(width=64,heads=4,layers=2)
            or cfg['representations']!=list(api.REPRESENTATIONS) or cfg['new_models']!=36
            or any(cfg[k] for k in ('held_controller_outcomes_opened','independent_roles_opened','deployment_changed','stage5c_executed','smc_enabled'))):
        raise ValueError('Complete original active9by4design and unchanged roles/budget required')
    if len(jobs)!=9 or len(data['sites'])!=318969:raise ValueError('All9active parents and318969original rows required')
    controls={}
    for job,ref in zip(jobs,refs):
        r=json.loads((ROOT/ref['path']).read_text());cp=r['checkpoint']
        if sha(ROOT/cp['path'])!=cp['sha256'] or r['fit']['step']!=4000 or r['fit']['parameters']!=88514:
            raise ValueError('Complete actual dimensionless parent required')
        controls[job['key']]=r
    complete=json.loads((store.PUBLIC/'complete.json').read_text());verify=json.loads((store.PUBLIC/'verification.json').read_text())
    if (not verify['all_arrays_exact'] or verify['complete_sha256']!=sha(store.PUBLIC/'complete.json')
            or not verify['raw_reparsed'] or complete['rows']!=318969 or complete['recordings']!=163 or complete['future_targets_read']):
        raise ValueError('Verified complete past-only detector store required')
    arrays={}
    for k,ref in complete['arrays'].items():
        if sha(ROOT/ref['path'])!=ref['sha256']:raise ValueError('Detector input cache changed')
        arrays[k]=np.load(ROOT/ref['path'],mmap_mode='r',allow_pickle=False)
    np.testing.assert_array_equal(arrays['ids'],np.arange(len(data['sites'])))
    for k in ('recordings','agents','frames'):np.testing.assert_array_equal(arrays[k],data[k])
    job_refs=[dict(key=j['key'],identity=j['old_identity'],checkpoint=controls[j['key']]['checkpoint']) for j in jobs]
    once(REG,dict(bindings={str(p.relative_to(ROOT)):sha(p) for p in FILES},
        prior_draft_registration_sha256=sha(PUBLIC/'registration.json'),
        pre_training_amendment='resume_fixture_supplies_representation_roster_no_numeric_or_role_change',
        active_parent_registration_sha256=sha(parent.PUBLIC/'registration.json'),
        active_parent_prediction_freeze_sha256=sha(parent.PUBLIC/'prediction_freeze.json'),
        active_geometry=identity['geometry'],parent_jobs=job_refs,
        producer_store_complete_sha256=sha(store.PUBLIC/'complete.json'),
        producer_store_verification_sha256=sha(store.PUBLIC/'verification.json'),
        models=36,parent_models=9,primary='single_history',steps_per_model=4000,
        primary_optimizer_updates=144000,extra_full_refit_updates=4000,pilot_updates=900,
        causal_input_fields=64,additional_parameters=512,actual_active_backbone=True,
        complete_original_active_family=True,active_complement_family_exists=False,
        controller_held_outcomes_opened=False,independent_roles_opened=False,deployment_changed=False))
    if not register:
        for rel,digest in json.loads(REG.read_text())['bindings'].items():
            if sha(ROOT/rel)!=digest:raise ValueError('Registered active producer code changed')
            committed(ROOT/rel)
        committed(REG)
    return cfg,data,jobs,arrays,controls


def runtime(cfg):
    torch.set_num_threads(4);torch.set_num_interop_threads(1);start=time.monotonic()
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt('Resume last atomic active-producer checkpoint')))
    def guard():
        if time.monotonic()-start>cfg['hard_runtime_limit_seconds']:raise TimeoutError('12h boundary; resume full36design')
        if shutil.disk_usage(ROOT).free<cfg['disk_reserve_bytes']:raise OSError('Preserve10GiB reserve')
        if sum(p.stat().st_size for p in PRIVATE.rglob('checkpoint.pt'))>cfg['checkpoint_cap_bytes']:raise OSError('Active checkpoint cap exceeded')
    def beat(**event):
        guard();value=dict(pid=os.getpid(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**event)
        write(PRIVATE/'heartbeat.json',value)
        with (PRIVATE/'events.jsonl').open('a') as f:f.write(json.dumps(value)+'\n')
        print(json.dumps(value),flush=True)
    beat(state='runtime_ready',architecture=platform.machine(),device='cpu',threads=4,workers=0,resource_probing=False)
    return start,beat,guard


def prepare_arm(data,arrays,job,arm):
    extra=representation(arrays['features'],arrays['recordings'],arrays['agents'],arrays['frames'],arm)
    norm=normalize(extra,data['sites'],job['old_identity']['fit_sites'])
    prepared=dict(data,geometry=api.DetectorGeometry(data['geometry'],extra,norm['mean'],norm['std'],norm['clip']))
    serial=dict(norm,mean=norm['mean'].tolist(),std=norm['std'].tolist())
    return prepared,serial


def directory(key,arm,folder='trials'):return PRIVATE/folder/(key+'_'+arm)


def fit(data,job,arm,norm,cfg,path,beat,*,resume=False,stop_at=None):
    seed=job['old_identity']['seed'];torch.manual_seed(seed)
    model=api.ActiveDetectorForecaster(job['design']['baseline_index'],arm,**cfg['architecture'])
    identity=dict(parent_trial=job['old_identity'],normalizer=norm,representation=arm,registration_sha256=sha(REG))
    with api.installed():
        result=api.native.fit_trial(model,data,job['design'],seed=seed,settings=cfg['training'],identity=identity,
            directory=path,resume=resume,stop_at=stop_at,heartbeat=lambda **kw:beat(trial=job['key'],representation=arm,**kw))
    return result,torch.load(path/'checkpoint.pt',map_location='cpu',weights_only=False)


def pilot(resume=False):
    cfg,data,jobs,arrays,controls=admission()
    if (PUBLIC/'pilot.json').exists():raise ValueError('Completed active pilot immutable')
    start,beat,guard=runtime(cfg);job=jobs[0];cells=[]
    for arm in cfg['representations']:
        prepared,norm=prepare_arm(data,arrays,job,arm);path=directory(job['key'],arm,'pilot')
        if path.exists() and not resume:raise ValueError('Existing active pilot requires --resume')
        if not (path/'checkpoint.pt').exists():fit(prepared,job,arm,norm,cfg,path,beat,stop_at=cfg['pilot_resume_at'])
        result,state=fit(prepared,job,arm,norm,cfg,path,beat,resume=True,stop_at=cfg['pilot_steps'])
        direct=directory(job['key'],arm,'pilot_direct')
        _,other=fit(prepared,job,arm,norm,cfg,direct,beat,resume=resume and (direct/'checkpoint.pt').exists(),stop_at=cfg['pilot_steps'])
        api.assert_recovery(state,other)
        if arm=='disabled':
            legacy=directory(job['key'],'parent','pilot');torch.manual_seed(job['old_identity']['seed'])
            api.native.fit_trial(parent.make_model(job),data,job['design'],seed=job['old_identity']['seed'],
                settings=cfg['training'],identity=dict(actual_parent_pilot=True),directory=legacy,
                resume=resume and (legacy/'checkpoint.pt').exists(),stop_at=cfg['pilot_steps'],heartbeat=beat)
            api.assert_neutral(torch.load(legacy/'checkpoint.pt',map_location='cpu',weights_only=False),state)
        elif not torch.count_nonzero(state['model']['detector_weight']):raise ValueError('No actual detector-projection update')
        cells.append(dict(representation=arm,seconds=result['seconds'],step=result['step'],parameters=result['parameters'],
            held_rows_sampled=result['held_rows_sampled'],exact_recovery=True))
        del prepared,state,other
    once(PUBLIC/'pilot.json',dict(pid=os.getpid(),seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        cells=cells,actual_parent_disabled_exact=True,engineering_pass=True,real_optimizer_updates=900,
        projected_primary_optimizer_seconds=float(np.mean([r['seconds'] for r in cells])*40*36),
        projection_excludes_loading_preparation_full_refit_and_readout=True))
    beat(state='active_producer_pilot_complete',updates=900)


def train(resume=False):
    cfg,data,jobs,arrays,controls=admission()
    if (PUBLIC/'training_freeze.json').exists():raise ValueError('Completed active36model freeze immutable')
    if not json.loads((PUBLIC/'pilot.json').read_text())['engineering_pass']:raise ValueError('Actual active-parent pilot required')
    start,beat,guard=runtime(cfg);refs=[];neutral=0;refit=False
    for job in jobs:
        for arm in cfg['representations']:
            rp=PUBLIC/'fits'/(job['key']+'_'+arm+'.json');path=directory(job['key'],arm)
            if rp.exists():
                if not resume:raise ValueError('Completed active receipts require explicit --resume')
                row=json.loads(rp.read_text())
                if (row['trial']!=job['key'] or row['representation']!=arm or row['parent_trial']!=job['old_identity']
                        or row['registration_sha256']!=sha(REG) or row['fit']['step']!=4000
                        or sha(ROOT/row['checkpoint']['path'])!=row['checkpoint']['sha256']):
                    raise ValueError('Completed active receipt changed')
                if arm=='disabled':neutral+=1
                if arm=='history' and job['key']==jobs[0]['key']:
                    previous=json.loads((PUBLIC/'first_full_refit.json').read_text())
                    if not previous['scientific_state_exact'] or previous['checkpoint_sha256']!=row['checkpoint']['sha256']:
                        raise ValueError('Active full refit changed')
                    refit=True
                refs.append(dict(path=str(rp.relative_to(ROOT)),sha256=sha(rp)))
                beat(state='completed_active_receipt_verified',models=len(refs),total=36);continue
            prepared,norm=prepare_arm(data,arrays,job,arm)
            result,state=fit(prepared,job,arm,norm,cfg,path,beat,resume=resume and (path/'checkpoint.pt').exists())
            old=torch.load(ROOT/controls[job['key']]['checkpoint']['path'],map_location='cpu',weights_only=False)
            if result['step']!=4000 or result['held_rows_sampled'] or result['parameters']!=89026:
                raise ValueError('Complete unchanged active budget required')
            for field in ('draws','train_ids','factors','sampler_rng','torch_rng'):exact(state[field],old[field])
            if arm=='disabled':api.assert_neutral(old,state);neutral+=1
            elif not torch.count_nonzero(state['model']['detector_weight']):raise ValueError('Inactive active-family detector branch')
            if arm=='history' and not refit:
                repeat=directory(job['key'],arm,'full_refit')
                _,again=fit(prepared,job,arm,norm,cfg,repeat,beat,resume=resume and (repeat/'checkpoint.pt').exists())
                api.assert_recovery(state,again);refit=True
                once(PUBLIC/'first_full_refit.json',dict(trial=job['key'],updates=4000,scientific_state_exact=True,
                    checkpoint_sha256=sha(path/'checkpoint.pt'),refit_checkpoint_sha256=sha(repeat/'checkpoint.pt')))
            row=dict(trial=job['key'],representation=arm,parent_trial=job['old_identity'],normalizer=norm,fit=result,
                checkpoint=dict(path=str(path.relative_to(ROOT)/'checkpoint.pt'),sha256=sha(path/'checkpoint.pt'),bytes=(path/'checkpoint.pt').stat().st_size),
                registration_sha256=sha(REG),result_source='fresh_native_arm64_Torch_active_producer_training',
                controller_held_outcomes_opened=False,independent_roles_opened=False)
            once(rp,row);refs.append(dict(path=str(rp.relative_to(ROOT)),sha256=sha(rp)))
            beat(state='active_endpoint_complete',trial=job['key'],representation=arm,models=len(refs),total=36,disabled_exact=neutral)
            del prepared,state,old
    if len(refs)!=36 or neutral!=9 or not refit:raise ValueError('Complete36models9neutralandfullrefit required')
    once(PUBLIC/'training_freeze.json',dict(pid=os.getpid(),seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        models=refs,count=36,disabled_exact_endpoints=9,primary_optimizer_updates=144000,extra_full_refit_updates=4000,pilot_updates=900,
        first_full_refit_sha256=sha(PUBLIC/'first_full_refit.json'),registration_sha256=sha(REG),
        controller_held_outcomes_opened=False,independent_roles_opened=False,deployment_changed=False,stage5c_executed=False,smc_enabled=False))
    beat(state='all36active_producers_complete',primary_optimizer_updates=144000)


def models():
    committed(PUBLIC/'training_freeze.json');frozen=json.loads((PUBLIC/'training_freeze.json').read_text());reg=json.loads(REG.read_text());result={}
    if frozen['count']!=36 or frozen['disabled_exact_endpoints']!=9 or frozen['registration_sha256']!=sha(REG):raise ValueError('Complete committed active36freeze required')
    if sha(PUBLIC/'first_full_refit.json')!=frozen['first_full_refit_sha256']:raise ValueError('Active full-refit receipt changed')
    for ref in frozen['models']:
        path=ROOT/ref['path'];committed(path);r=json.loads(path.read_text())
        if sha(path)!=ref['sha256'] or sha(ROOT/r['checkpoint']['path'])!=r['checkpoint']['sha256'] or r['fit']['step']!=4000:
            raise ValueError('Frozen active endpoint changed')
        key=r['trial'],r['representation']
        if key in result:raise ValueError('Repeated active model key')
        result[key]=r
    if set(result)!={(j['key'],a) for j in reg['parent_jobs'] for a in api.REPRESENTATIONS}:raise ValueError('All9active parents and4arms required')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=('register','pilot','train'))
    parser.add_argument('--resume',action='store_true');args=parser.parse_args();PRIVATE.mkdir(parents=True,exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.phase=='register':admission(register=True)
        elif args.phase=='pilot':pilot(args.resume)
        else:train(args.resume)
