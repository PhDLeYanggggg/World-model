"""Loss-only matched refit of the actual frozen dimensionless producer."""
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

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 required before Torch import')
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key,'4')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from scripts import run_m3w_european_dimensionless_refit as parent
from scripts.run_m3w_native_forecast import json_write
from src.world_model.m3w_reference_harm_forecast import fit_trial, assert_training_equivalent
from src.world_model.m3w_native_forecast import fit_trial as native_fit

NAME='european_dimensionless_harm_forecast_v1'
CONFIG='configs/m3w_dimensionless_harm_forecast_v1.json'
PUBLIC=ROOT/'outputs/publication_readiness_2026_09'/NAME
PRIVATE=ROOT/'data/stage_cvpr2027_experiments'/NAME
FILES=(CONFIG,'scripts/run_m3w_dimensionless_harm_forecast.py',
       'src/world_model/m3w_reference_harm_forecast.py',
       'tests/test_m3w_reference_harm_forecast.py',
       str((PUBLIC/'protocol.md').relative_to(ROOT)))
digest,immutable_json=parent.digest,parent.immutable_json


def beat(state,**kw):
    event=dict(state=state,pid=os.getpid(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw)
    json_write(PRIVATE/'heartbeat.json',event)
    with (PRIVATE/'events.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
    print(json.dumps(event),flush=True)


def load():
    cfg=json.loads((ROOT/CONFIG).read_text())
    if cfg['objective'] != dict(selected_weight=2.,cv_weight=2.) or cfg['models']!=9 or cfg['steps']!=4000:
        raise ValueError('Fixed comparison required')
    if shutil.disk_usage(ROOT).free < cfg['minimum_free_disk_GiB']*2**30:
        raise OSError('Preserve10GiB and existing work')
    _,pc,data,jobs,pidentity=parent.load()
    refs=parent.endpoints(jobs)
    controls={}
    for job,ref in zip(jobs,refs):
        if parent.artifact(ROOT/ref['path']) != ref:raise ValueError('Control receipt changed')
        controls[job['key']]=json.loads((ROOT/ref['path']).read_text())
    predictions=json.loads((parent.PUBLIC/'prediction_freeze.json').read_text())
    for job,ref in zip(jobs,predictions['predictions']):
        if job['key']!=ref['key'] or parent.artifact(ROOT/ref['dimensionless']['path'])!=ref['dimensionless']:
            raise ValueError('Control prediction differs')
        controls[job['key']]['prediction']=ref['dimensionless']
    identity=dict(parent_registration=parent.artifact(parent.PUBLIC/'registration.json'),
        parent_identity=pidentity,controls=controls,bindings={p:digest(ROOT/p) for p in FILES},
        objective=cfg['objective'],source_role='previously_opened_source_development',runtime=pc['runtime'])
    immutable_json(PRIVATE/'identity.json',identity)
    immutable_json(PUBLIC/'registration.json',dict(identity=identity,
        jobs=[dict(key=j['key'],trial=j['old_identity']) for j in jobs],
        primary_updates_planned=36000,neutral_updates_planned=4000,engineering_updates_planned=80,
        independent_roles_opened=False,result_source='cached_verified_assets_training_not_run_at_registration'))
    return cfg,pc,data,jobs,identity


def assert_identity(identity):
    for path,sha in identity['bindings'].items():
        if digest(ROOT/path)!=sha:raise ValueError('Frozen code changed')
    if parent.artifact(parent.PUBLIC/'registration.json')!=identity['parent_registration']:
        raise ValueError('Parent changed')


def fit(job,pc,data,identity,path,weights,*,resume=False,stop_at=None,native=False):
    model=parent.make_model(job)
    ti=dict(experiment=identity,parent_trial=job['old_identity'])
    fn=native_fit if native else fit_trial
    result=fn(model,data,job['design'],seed=job['old_identity']['seed'],settings=pc['training'],
        identity=ti,directory=path,resume=resume,stop_at=stop_at,
        heartbeat=lambda **kw:beat(trial=job['key'],**kw),**({} if native else weights))
    assert_identity(identity)
    return dict(identity=ti,fit=result,checkpoint=parent.artifact(path/'checkpoint.pt'),objective=weights)


def saved(ref):return torch.load(ROOT/ref['path'],map_location='cpu',weights_only=False)


def checked(path):
    doc=json.loads(path.read_text())
    for ref in doc['checkpoints']:
        if parent.artifact(ROOT/ref['path'])!=ref:raise ValueError('Pilot/neutral receipt changed')
    return doc


def pilot(cfg,pc,data,jobs,identity):
    path=PUBLIC/'pilot.json'
    if path.exists():return checked(path)
    started=time.monotonic();out={};job=jobs[0]
    for mode in ('direct','resumed','neutral','native'):
        home=PRIVATE/'pilot'/mode
        w=dict(selected_weight=0.,cv_weight=0.) if mode in ('native','neutral') else cfg['objective']
        if mode=='resumed':fit(job,pc,data,identity,home,w,stop_at=cfg['pilot_resume_at'])
        out[mode]=fit(job,pc,data,identity,home,w,stop_at=cfg['pilot_steps'],resume=mode=='resumed',native=mode=='native')
    assert_training_equivalent(saved(out['direct']['checkpoint']),saved(out['resumed']['checkpoint']))
    assert_training_equivalent(saved(out['neutral']['checkpoint']),saved(out['native']['checkpoint']))
    immutable_json(path,dict(result_source='fresh_run',pid=os.getpid(),updates=80,
        exact_resume=True,exact_neutral=True,seconds=time.monotonic()-started,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        checkpoints=[v['checkpoint'] for v in out.values()]))
    beat('pilot_complete',updates=80)


def neutral(cfg,pc,data,jobs,identity,resume):
    path=PUBLIC/'neutral_replay.json'
    if path.exists():return checked(path)
    job=next(j for j in jobs if j['key']==cfg['full_neutral_control'])
    doc=fit(job,pc,data,identity,PRIVATE/'neutral'/job['key'],dict(selected_weight=0.,cv_weight=0.),resume=resume)
    ref=identity['controls'][job['key']]['checkpoint']
    assert_training_equivalent(saved(doc['checkpoint']),saved(ref))
    immutable_json(path,dict(result_source='fresh_run_replay_against_cached_verified',pid=os.getpid(),
        exact_full_neutral=True,fit=doc['fit'],checkpoints=[doc['checkpoint'],ref]))
    beat('full_neutral_exact',updates=4000)


def complete(job,identity):
    doc=json.loads((PRIVATE/'trials'/job['key']/'complete.json').read_text())
    if (doc['identity']!=dict(experiment=identity,parent_trial=job['old_identity'])
            or not doc['fit']['complete'] or doc['fit']['step']!=4000
            or doc['fit']['held_rows_sampled']!=0 or parent.artifact(ROOT/doc['checkpoint']['path'])!=doc['checkpoint']):
        raise ValueError('Changed or incomplete endpoint')
    return doc


def train(cfg,pc,data,jobs,identity,resume):
    checked(PUBLIC/'pilot.json');checked(PUBLIC/'neutral_replay.json')
    start=time.monotonic();finals={};new=0
    for job in jobs:
        path=PRIVATE/'trials'/job['key']
        if not (path/'complete.json').exists():
            beat('primary_fit_started',trial=job['key'])
            doc=fit(job,pc,data,identity,path,cfg['objective'],resume=resume)
            immutable_json(path/'complete.json',doc)
            new+=doc['fit']['new_updates']
        finals[job['key']]=complete(job,identity)
        beat('primary_fit_complete',trial=job['key'])
    immutable_json(PUBLIC/'training_freeze.json',dict(result_source='fresh_run',identity=identity,
        fits=finals,models=9,primary_updates=36000,summed_fit_seconds=sum(v['fit']['seconds'] for v in finals.values()),
        readout_not_yet_run=True,independent_roles_opened=False,deployment_changed=False))
    json_write(PRIVATE/'training_runtime.json',dict(pid=os.getpid(),seconds=time.monotonic()-start,
        new_updates_this_invocation=new,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    beat('training_complete',models=9,updates=36000)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ('prepare','pilot','neutral','train','resume'):p.add_argument('--'+flag,action='store_true')
    a=p.parse_args()
    if not any((a.prepare,a.pilot,a.neutral,a.train)):p.error('Explicit phase required')
    for path in (PRIVATE,PUBLIC):
        if any(v.is_symlink() for v in (path,*path.parents)):raise ValueError('Symlinked destination')
        path.mkdir(parents=True,exist_ok=True)
    with (PRIVATE/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.set_num_threads(4);torch.set_num_interop_threads(1)
        cfg,pc,data,jobs,identity=load()
        beat('verified_current_producer_assets',rows=len(data['sites']),models=len(jobs))
        if a.pilot:pilot(cfg,pc,data,jobs,identity)
        if a.neutral:neutral(cfg,pc,data,jobs,identity,a.resume)
        if a.train:train(cfg,pc,data,jobs,identity,a.resume)
        beat('phase_complete')


if __name__=='__main__':main()
