"""Matched true SOURCE Transformer training with past detector token features."""
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

if platform.system()=='Darwin' and platform.machine()!='arm64':raise RuntimeError('Native arm64 .venv-pytorch required before Torch')
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ.setdefault(key,'4')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts import run_m3w_european_source_forecast as source
from scripts import build_m3w_producer_detector_store as inputs
from src.world_model import m3w_detector_forecaster as api
from src.world_model.m3w_detector_cost import representation
from src.world_model.m3w_producer_detector_data import normalize

NAME='european_detector_forecaster_v1';PUBLIC=ROOT/'outputs/publication_readiness_2026_09'/NAME
PRIVATE=ROOT/'data/stage_cvpr2027_experiments'/NAME
CONFIG=ROOT/'configs/m3w_detector_forecaster_v1.json';REG=PUBLIC/'registration_v2.json'
FILES=(Path(__file__),CONFIG,ROOT/'src/world_model/m3w_detector_forecaster.py',
    ROOT/'tests/test_m3w_detector_forecaster.py',ROOT/'tests/test_m3w_detector_forecaster_runner.py',
    ROOT/'src/world_model/m3w_producer_detector_data.py',ROOT/'src/world_model/m3w_detector_cost.py',
    ROOT/'src/world_model/m3w_detector_history_features.py',PUBLIC/'protocol.md')
sha,once,write=source.digest,source.immutable_json,source.json_write


def committed(path):
    subprocess.run(['git','diff','--quiet','HEAD','--',str(path.relative_to(ROOT))],cwd=ROOT,check=True)
    subprocess.run(['git','ls-files','--error-unmatch',str(path.relative_to(ROOT))],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)


def admission(register=False):
    cfg=json.loads(CONFIG.read_text());original,manifest,receipts,identity=source.load_source()
    if (cfg['training']!=original['training'] or cfg['architecture']!=original['architecture']
            or cfg['seeds']!=original['seeds'] or cfg['representations']!=list(api.REPRESENTATIONS)
            or cfg['kinds']!=['single','complement'] or cfg['held_controller_readout_opened'] or cfg['independent_roles_opened']):
        raise ValueError('Complete original18producer designs and matched four input arms required')
    data=source.prepare(manifest,receipts,identity);jobs=source.trials(original,data,identity)
    report=json.loads((inputs.PUBLIC/'complete.json').read_text());verification=json.loads((inputs.PUBLIC/'verification.json').read_text())
    if (report['rows']!=len(data['sites']) or report['recordings']!=163 or report['future_targets_read']
            or not report['controller_overlap_exact'] or not verification['all_arrays_exact']
            or not verification['raw_reparsed'] or verification['complete_sha256']!=sha(inputs.PUBLIC/'complete.json')):
        raise ValueError('Complete raw-replayed producer past input store required')
    arrays={}
    for key,ref in report['arrays'].items():
        path=ROOT/ref['path']
        if sha(path)!=ref['sha256']:raise ValueError('Verified producer input array changed')
        arrays[key]=np.load(path,mmap_mode='r',allow_pickle=False)
    np.testing.assert_array_equal(arrays['ids'],np.arange(len(data['sites'])))
    for key in ('recordings','agents','frames'):np.testing.assert_array_equal(arrays[key],data[key])
    old={}
    analysis=json.loads((source.PUBLIC/'analysis.json').read_text())
    for r in analysis['training']:
        ti=r['identity'];key=f"{ti['kind']}{ti['fold']}_seed{ti['seed']}"
        if key in old or sha(ROOT/r['checkpoint'])!=r['checkpoint_sha256']:raise ValueError('Original producer endpoint changed')
        old[key]=r
    if len(jobs)!=18 or len(old)!=18:raise ValueError('All original18producer endpoints required')
    for key,_,ti in jobs:
        if old[key]['identity']!=ti:raise ValueError('Original producer fitting roles or design changed')
    once(REG,dict(bindings={str(p.relative_to(ROOT)):sha(p) for p in FILES},
        prior_draft_registration_sha256=sha(PUBLIC/'registration.json'),
        pre_training_amendment='resume_test_fixture_uses_its_own_temporary_root_no_numeric_or_role_change',
        source_analysis_sha256=sha(source.PUBLIC/'analysis.json'),source_matrix_sha256=sha(source.PUBLIC/'matrix.json'),
        producer_input_complete_sha256=sha(inputs.PUBLIC/'complete.json'),producer_input_verification_sha256=sha(inputs.PUBLIC/'verification.json'),
        source_identity=identity,producer_jobs=[dict(key=k,identity=ti) for k,_,ti in jobs],
        representations=cfg['representations'],models=72,updates_per_model=4000,primary_optimizer_updates=288000,
        additional_parameters=512,scope='original_SOURCE_training_only_old_controller_fitting_readout_only',
        future_support_input=False,held_controller_readout_opened=False,independent_roles_opened=False,
        deployment_changed=False,stage5c_executed=False,smc_enabled=False))
    if not register:
        for rel,digest in json.loads(REG.read_text())['bindings'].items():
            if sha(ROOT/rel)!=digest:raise ValueError('Registered detector producer code changed')
            committed(ROOT/rel)
        for path in (REG,inputs.PUBLIC/'complete.json',inputs.PUBLIC/'verification.json'):committed(path)
    return cfg,data,jobs,arrays,old


def runtime(cfg):
    torch.set_num_threads(4);torch.set_num_interop_threads(1);start=time.monotonic()
    PRIVATE.mkdir(parents=True,exist_ok=True)
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt('Resume last atomic producer checkpoint')))
    def guard():
        if time.monotonic()-start>cfg['hard_runtime_limit_seconds']:raise TimeoutError('12h boundary; resume full72model design')
        if shutil.disk_usage(ROOT).free<cfg['disk_reserve_bytes']:raise OSError('Preserve10GiB disk reserve')
        if sum(p.stat().st_size for p in PRIVATE.rglob('checkpoint.pt'))>cfg['checkpoint_cap_bytes']:raise OSError('Private producer checkpoint cap exceeded')
    def beat(**event):
        guard();r=dict(pid=os.getpid(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**event)
        write(PRIVATE/'heartbeat.json',r)
        with (PRIVATE/'events.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
        print(json.dumps(r),flush=True)
    beat(state='runtime_ready',architecture=platform.machine(),torch_threads=4,workers=0,device='cpu',resource_probing=False)
    return start,beat,guard


def prepare_arm(data,arrays,ti,arm):
    extra=representation(arrays['features'],arrays['recordings'],arrays['agents'],arrays['frames'],arm)
    norm=normalize(extra,data['sites'],ti['fit_sites'])
    normalized=dict(data,geometry=api.DetectorGeometry(data['geometry'],extra,norm['mean'],norm['std'],norm['clip']))
    serial=dict(norm,mean=norm['mean'].tolist(),std=norm['std'].tolist())
    return normalized,serial


def directory(key,arm,folder='trials'):return PRIVATE/folder/(key+'_'+arm)


def fit(data,fold,ti,arm,norm,cfg,path,beat,*,resume=False,stop_at=None):
    torch.manual_seed(ti['seed']);model=api.DetectorForecaster(cfg['architecture'],ti['baseline_index'],arm)
    identity=dict(original_producer=ti,representation=arm,normalizer=norm,registration_sha256=sha(REG))
    with api.installed():
        result=api.native.fit_trial(model,data,fold,seed=ti['seed'],settings=cfg['training'],identity=identity,
            directory=path,resume=resume,stop_at=stop_at,heartbeat=lambda **kw:beat(representation=arm,**kw))
    state=torch.load(path/'checkpoint.pt',map_location='cpu',weights_only=False)
    return result,state


def pilot(resume=False):
    cfg,data,jobs,arrays,old=admission()
    if (PUBLIC/'pilot.json').exists():raise ValueError('Completed real producer pilot immutable')
    start,beat,guard=runtime(cfg);key,fold,ti=jobs[0];cells=[]
    for arm in cfg['representations']:
        prepared,norm=prepare_arm(data,arrays,ti,arm);path=directory(key,arm,'pilot')
        if (path/'checkpoint.pt').exists() and not resume:raise ValueError('Existing pilot requires explicit resume')
        if not (path/'checkpoint.pt').exists():fit(prepared,fold,ti,arm,norm,cfg,path,beat,stop_at=cfg['pilot_resume_at'])
        result,state=fit(prepared,fold,ti,arm,norm,cfg,path,beat,resume=True,stop_at=cfg['pilot_steps'])
        direct_path=directory(key,arm,'pilot_direct')
        _,direct=fit(prepared,fold,ti,arm,norm,cfg,direct_path,beat,
            resume=(direct_path/'checkpoint.pt').exists() and resume,stop_at=cfg['pilot_steps'])
        api.assert_recovery(state,direct)
        if arm=='disabled':
            legacy=directory(key,'legacy','pilot');torch.manual_seed(ti['seed'])
            api.native.fit_trial(api.SourceForecaster(cfg['architecture'],ti['baseline_index']),data,fold,
                seed=ti['seed'],settings=cfg['training'],identity=dict(real_legacy_pilot=True),directory=legacy,
                stop_at=cfg['pilot_steps'],resume=(legacy/'checkpoint.pt').exists() and resume,heartbeat=beat)
            api.assert_neutral(torch.load(legacy/'checkpoint.pt',map_location='cpu',weights_only=False),state)
        elif not torch.count_nonzero(state['model']['detector_weight']):raise ValueError('Detector branch did not update')
        cells.append(dict(representation=arm,steps=result['step'],seconds=result['seconds'],parameters=result['parameters'],
            held_rows_sampled=result['held_rows_sampled'],exact_recovery=True,real_gradient_updates=True))
        del prepared
    once(PUBLIC/'pilot.json',dict(pid=os.getpid(),seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        cells=cells,engineering_pass=True,disabled_legacy_exact=True,real_optimizer_updates=900,
        projected_primary_optimizer_seconds=float(np.mean([c['seconds'] for c in cells])*40*72),
        projection_excludes_input_preparation_refit_and_readout=True,independent_roles_opened=False))
    beat(state='real_producer_pilot_complete',real_optimizer_updates=900)


def train(resume=False):
    cfg,data,jobs,arrays,old=admission()
    if (PUBLIC/'training_freeze.json').exists():raise ValueError('Completed72model freeze immutable')
    if not json.loads((PUBLIC/'pilot.json').read_text())['engineering_pass']:raise ValueError('Actual full-context producer pilot required')
    start,beat,guard=runtime(cfg);refs=[];neutral=0;refit=False
    for key,fold,ti in jobs:
        for arm in cfg['representations']:
            rp=PUBLIC/'fits'/(key+'_'+arm+'.json')
            if rp.exists():
                if not resume:raise ValueError('Existing producer receipts require explicit resume')
                row=json.loads(rp.read_text())
                if (row['trial']!=key or row['representation']!=arm or row['original_producer']!=ti
                        or row['registration_sha256']!=sha(REG) or row['fit']['step']!=4000
                        or sha(ROOT/row['checkpoint']['path'])!=row['checkpoint']['sha256']):
                    raise ValueError('Completed producer receipt or checkpoint changed')
                if arm=='disabled':neutral+=1
                if arm=='history' and key==jobs[0][0]:
                    r=json.loads((PUBLIC/'first_full_refit.json').read_text())
                    if not r['scientific_state_exact'] or r['primary_checkpoint_sha256']!=row['checkpoint']['sha256']:
                        raise ValueError('Existing first full refit differs')
                    refit=True
                refs.append(dict(path=str(rp.relative_to(ROOT)),sha256=sha(rp)))
                beat(state='completed_producer_verified',trial=key,representation=arm,models=len(refs),total=72)
                continue
            prepared,norm=prepare_arm(data,arrays,ti,arm);path=directory(key,arm)
            result,state=fit(prepared,fold,ti,arm,norm,cfg,path,lambda **kw:beat(trial=key,**kw),
                resume=(path/'checkpoint.pt').exists() and resume)
            if result['step']!=4000 or result['held_rows_sampled'] or result['parameters']!=old[key]['fit']['parameters']+512:
                raise ValueError('Complete matched producer endpoint with no held sampling required')
            original=torch.load(ROOT/old[key]['checkpoint'],map_location='cpu',weights_only=False)
            from src.world_model.m3w_multiaction_cost import exact
            for field in ('draws','train_ids','factors','sampler_rng','torch_rng'):exact(original[field],state[field])
            if arm=='disabled':
                api.assert_neutral(original,state);neutral+=1
            elif not torch.count_nonzero(state['model']['detector_weight']):raise ValueError('Inactive past detector branch')
            if arm=='history' and not refit:
                repeat=directory(key,arm,'full_refit')
                _,again=fit(prepared,fold,ti,arm,norm,cfg,repeat,lambda **kw:beat(trial=key,full_refit=True,**kw),
                    resume=(repeat/'checkpoint.pt').exists() and resume)
                api.assert_recovery(state,again);refit=True
                once(PUBLIC/'first_full_refit.json',dict(trial=key,representation=arm,updates=4000,scientific_state_exact=True,
                    primary_checkpoint_sha256=sha(path/'checkpoint.pt'),refit_checkpoint_sha256=sha(repeat/'checkpoint.pt')))
            cp=path/'checkpoint.pt';receipt=dict(trial=key,representation=arm,original_producer=ti,normalizer=norm,
                fit=result,checkpoint=dict(path=str(cp.relative_to(ROOT)),sha256=sha(cp),bytes=cp.stat().st_size),
                registration_sha256=sha(REG),result_source='fresh_native_arm64_Torch_trajectory_producer_training',
                held_controller_readout_opened=False,independent_roles_opened=False)
            once(rp,receipt);refs.append(dict(path=str(rp.relative_to(ROOT)),sha256=sha(rp)))
            beat(state='producer_endpoint_complete',trial=key,representation=arm,models=len(refs),total=72,disabled_exact=neutral)
            del prepared,state
    if len(refs)!=72 or neutral!=18 or not refit:raise ValueError('Complete72models18neutralandfullrefit required')
    once(PUBLIC/'training_freeze.json',dict(pid=os.getpid(),seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        models=refs,count=72,primary_optimizer_updates=288000,extra_full_refit_updates=4000,pilot_updates=900,
        disabled_exact_endpoints=18,first_full_refit_sha256=sha(PUBLIC/'first_full_refit.json'),registration_sha256=sha(REG),
        held_controller_readout_opened=False,independent_roles_opened=False,deployment_changed=False,stage5c_executed=False,smc_enabled=False))
    beat(state='all72producer_training_complete',primary_optimizer_updates=288000)


def models():
    committed(PUBLIC/'training_freeze.json');frozen=json.loads((PUBLIC/'training_freeze.json').read_text());reg=json.loads(REG.read_text())
    if frozen['count']!=72 or frozen['disabled_exact_endpoints']!=18 or frozen['registration_sha256']!=sha(REG):raise ValueError('Complete committed72producer endpoints required')
    if sha(PUBLIC/'first_full_refit.json')!=frozen['first_full_refit_sha256']:raise ValueError('Full refit receipt changed')
    result={}
    for ref in frozen['models']:
        p=ROOT/ref['path'];committed(p);r=json.loads(p.read_text())
        if sha(p)!=ref['sha256'] or sha(ROOT/r['checkpoint']['path'])!=r['checkpoint']['sha256'] or r['fit']['step']!=4000:
            raise ValueError('Frozen producer endpoint changed')
        key=r['trial'],r['representation']
        if key in result:raise ValueError('Unique producer model key required')
        result[key]=r
    expected={(r['key'],arm) for r in reg['producer_jobs'] for arm in reg['representations']}
    if set(result)!=expected:raise ValueError('Full original18jobs by four inputs required')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=('register','pilot','train'))
    parser.add_argument('--resume',action='store_true');args=parser.parse_args();PRIVATE.mkdir(parents=True,exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.phase=='register':admission(register=True)
        elif args.phase=='pilot':pilot(args.resume)
        else:train(args.resume)
