"""Four matched causal detector representations, fixed complete Torch fits."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import sys
import time

if platform.system()=='Darwin' and platform.machine()!='arm64':
    raise RuntimeError('Use native arm64 .venv-pytorch before importing Torch')
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key,'4')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts import run_m3w_completion_cost as parent
from src.world_model import m3w_detector_cost as api
from src.world_model.m3w_detector_feature_store import DetectorFeatureStore

NAME='european_detector_cost_v1';PUBLIC=ROOT/'outputs/publication_readiness_2026_09'/NAME
PRIVATE=ROOT/'data/stage_cvpr2027_experiments'/NAME
STORE=ROOT/'outputs/publication_readiness_2026_09/european_detector_history_store_v1'
CONFIG=ROOT/'configs/m3w_detector_cost_v1.json';REG=PUBLIC/'registration.json'
FILES=(Path(__file__),CONFIG,ROOT/'src/world_model/m3w_detector_cost.py',
    ROOT/'src/world_model/m3w_detector_feature_store.py',ROOT/'tests/test_m3w_detector_cost.py',
    ROOT/'tests/test_m3w_detector_feature_store.py',ROOT/'tests/test_m3w_detector_cost_runner.py',PUBLIC/'protocol.md')
sha,once,write=parent.sha,parent.once,parent.json_write


def admission(register=False):
    cfg=json.loads(CONFIG.read_text());pc,values=parent.admission();refs=parent.models()
    store=DetectorFeatureStore(ROOT,STORE)
    if (cfg['training']!=pc['training'] or cfg['representations']!=list(api.REPRESENTATIONS)
            or cfg['banks']!=pc['banks'] or cfg['seeds']!=pc['seeds'] or cfg['supervision']!='known_only'
            or cfg['risk_budget']!=.02 or cfg['held_scored'] or cfg['independent_roles_opened']):
        raise ValueError('Matched fixed detector-input ablation required')
    once(REG,dict(bindings={str(p.relative_to(ROOT)):sha(p) for p in FILES},
        verified_store_complete_sha256=store.complete_sha256,store_verification_sha256=sha(STORE/'verification.json'),
        parent_freeze_sha256=sha(parent.PUBLIC/'training_freeze.json'),
        contexts=sorted(values[2]['contexts']),seeds=cfg['seeds'],banks=cfg['banks'],representations=cfg['representations'],
        fits=144,steps_per_fit=2000,primary_optimizer_updates=288000,parameters=30405,
        original_fields=398,extra_fields=64,scope='original_fitting_only',
        held_scored=False,independent_roles_opened=False,deployment_changed=False,stage5c_executed=False,smc_enabled=False))
    if not register:
        for rel,digest in json.loads(REG.read_text())['bindings'].items():
            if sha(ROOT/rel)!=digest:raise ValueError('Registered detector-cost code changed')
            parent.parent.parent.committed(ROOT/rel)
        for path in (REG,STORE/'complete.json',STORE/'verification.json'):
            parent.parent.parent.committed(path)
    return cfg,values,refs,store


def runtime(cfg):
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    PRIVATE.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt('Resume last atomic checkpoint')))
    def guard():
        if shutil.disk_usage(ROOT).free<cfg['disk_reserve_bytes']:raise OSError('Preserve10GiB reserve')
        if sum(p.stat().st_size for p in PRIVATE.rglob('*.pt.gz'))>cfg['checkpoint_cap_bytes']:raise OSError('Private checkpoint cap exceeded')
        if time.monotonic()-start>cfg['hard_runtime_limit_seconds']:raise TimeoutError('12h boundary; resume the complete fixed roster')
    def beat(**event):
        guard();value=dict(pid=os.getpid(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**event)
        write(PRIVATE/'heartbeat.json',value)
        with (PRIVATE/'events.jsonl').open('a') as stream:stream.write(json.dumps(value)+'\n')
        print(json.dumps(value),flush=True)
    guard();beat(state='runtime_ready',architecture=platform.machine(),cpu_threads=4,workers=0,device='cpu',resource_probing=False)
    return start,beat,guard


def checkpoint(context,seed,bank,representation,folder='heads'):
    return PRIVATE/folder/f'{context}_seed{seed}_{bank}_{representation}.pt.gz'


def data_for(packets,store,arm,original):
    extra=np.concatenate([store.read(packets[s]['ids'],arm) for s in sorted(packets)])
    sites=np.concatenate([packets[s]['sites'] for s in sorted(packets)])
    return api.training_data(original,extra,sites,arm)


def fit(data,cfg,context,seed,bank,arm,path,beat,guard,*,stop_at=None,resume=False):
    identity=dict(registration_sha256=sha(REG),context=context,representation=arm)
    with api.installed(arm):
        return parent.api.fit(data,mode='known_only',bank=bank,settings=cfg['training'],seed=seed,
            identity=identity,path=path,heartbeat=lambda **kw:beat(context=context,representation=arm,**kw),
            guard=guard,stop_at=stop_at,resume=resume)


def pilot(resume=False):
    cfg,values,refs,store=admission()
    if (PUBLIC/'pilot.json').exists():raise ValueError('Completed real pilot immutable')
    start,beat,guard=runtime(cfg);context,items=sorted(values[2]['contexts'].items())[0]
    packets=parent.packets(context,items,values,beat);original=parent.api.training_data(packets,parent.parent)
    times=[];cells=[]
    for arm in cfg['representations']:
        data=data_for(packets,store,arm,original)
        for bank in cfg['banks']:
            path=checkpoint(context,17,bank,arm,'pilot')
            if path.exists() and not resume:raise ValueError('Existing pilot requires explicit resume')
            if not path.exists():fit(data,cfg,context,17,bank,arm,path,beat,guard,stop_at=10)
            state=fit(data,cfg,context,17,bank,arm,path,beat,guard,stop_at=20,resume=True)
            direct_path=checkpoint(context,17,bank,arm,'pilot_direct')
            direct=fit(data,cfg,context,17,bank,arm,direct_path,beat,guard,stop_at=20,resume=direct_path.exists() and resume)
            for key in state:
                if key!='seconds':parent.api.base.exact(state[key],direct[key])
            if arm=='disabled':
                legacy=parent.api.fit(original,mode='known_only',bank=bank,settings=cfg['training'],seed=17,
                    identity=dict(real_pilot_legacy=True),path=checkpoint(context,17,bank,'legacy','pilot'),
                    stop_at=20,resume=resume,heartbeat=beat,guard=guard)
                api.assert_neutral(legacy,state)
            elif not torch.count_nonzero(state['model']['detector_weight']):
                raise ValueError('No active detector-branch gradient update')
            times.append(state['seconds']);cells.append(dict(bank=bank,representation=arm,exact20step_recovery=True,
                parameters=state['parameters'],unknown_rows_drawn=state['unknown_rows_drawn'],
                unknown_rows_with_loss=state['unknown_rows_with_loss']))
    once(PUBLIC/'pilot.json',dict(pid=os.getpid(),seconds=time.monotonic()-start,peak_RSS_bytes=parent.rss_bytes(),
        cells=cells,engineering_pass=True,legacy_disabled_neutral_both_banks=True,real_optimizer_updates=360,
        projected_primary_optimizer_seconds=float(np.mean(times)*100*144),
        projection_excludes_preparation_refit_and_readout=True,held_scored=False,independent_roles_opened=False))
    beat(state='pilot_complete',real_optimizer_updates=360)


def train(resume=False):
    cfg,values,refs,store=admission()
    if (PUBLIC/'training_freeze.json').exists():raise ValueError('Completed144fit freeze immutable')
    if not json.loads((PUBLIC/'pilot.json').read_text())['engineering_pass']:raise ValueError('Actual complete-context pilot required')
    start,beat,guard=runtime(cfg);receipts=[];neutral=0;refit=False
    for context,items in sorted(values[2]['contexts'].items()):
        packets=parent.packets(context,items,values,beat);original=parent.api.training_data(packets,parent.parent)
        for arm in cfg['representations']:
            data=data_for(packets,store,arm,original)
            for seed in cfg['seeds']:
                for bank in cfg['banks']:
                    path=checkpoint(context,seed,bank,arm)
                    state=fit(data,cfg,context,seed,bank,arm,path,beat,guard,resume=path.exists() and resume)
                    old=parent.api.base.read_checkpoint(ROOT/refs[context,seed,bank,'known_only']['checkpoint']['path'])
                    for key in ('draw_hash','sampler_rng','torch_rng','row_draws'):
                        parent.api.base.exact(state[key],old[key])
                    if arm=='disabled':api.assert_neutral(old,state);neutral+=1
                    elif not torch.count_nonzero(state['model']['detector_weight']):raise ValueError('Inactive detector branch')
                    row=dict(context=context,seed=seed,bank=bank,representation=arm,step=state['step'],
                        parameters=state['parameters'],seconds=state['seconds'],unknown_rows_drawn=state['unknown_rows_drawn'],
                        unknown_rows_with_loss=state['unknown_rows_with_loss'],checkpoint=dict(path=str(path.relative_to(ROOT)),
                        sha256=sha(path),bytes=path.stat().st_size),registration_sha256=sha(REG),held_scored=False)
                    rp=PUBLIC/'fits'/f'{context}_seed{seed}_{bank}_{arm}.json';once(rp,row)
                    receipts.append(dict(path=str(rp.relative_to(ROOT)),sha256=sha(rp)))
                    if arm=='history' and bank=='physical_neural' and not refit:
                        direct=checkpoint(context,seed,bank,arm,'full_refit')
                        repeated=fit(data,cfg,context,seed,bank,arm,direct,beat,guard,resume=direct.exists() and resume)
                        for key in state:
                            if key!='seconds':parent.api.base.exact(state[key],repeated[key])
                        once(PUBLIC/'first_full_refit.json',dict(context=context,seed=seed,bank=bank,representation=arm,
                            scientific_state_exact=True,primary_checkpoint_sha256=sha(path),refit_checkpoint_sha256=sha(direct),
                            primary_seconds=state['seconds'],refit_seconds=repeated['seconds']))
                        refit=True
                    beat(state='endpoint_complete',context=context,representation=arm,seed=seed,bank=bank,
                         fits=len(receipts),total=144,disabled_exact=neutral)
            del data
        del original,packets
    if len(receipts)!=144 or neutral!=36 or not refit:raise ValueError('Complete144fits/36neutral/fullrefit required')
    once(PUBLIC/'training_freeze.json',dict(pid=os.getpid(),seconds=time.monotonic()-start,peak_RSS_bytes=parent.rss_bytes(),
        registration_sha256=sha(REG),fits=receipts,count=144,optimizer_updates=288000,
        disabled_exact_endpoints=36,extra_full_refit_updates=2000,pilot_updates=360,
        first_full_refit_sha256=sha(PUBLIC/'first_full_refit.json'),held_scored=False,
        independent_roles_opened=False,deployment_changed=False,stage5c_executed=False,smc_enabled=False))
    beat(state='all144training_complete',optimizer_updates=288000)


def models():
    parent.parent.parent.committed(PUBLIC/'training_freeze.json')
    freeze=json.loads((PUBLIC/'training_freeze.json').read_text());reg=json.loads(REG.read_text())
    if freeze['count']!=144 or freeze['disabled_exact_endpoints']!=36 or freeze['registration_sha256']!=sha(REG):
        raise ValueError('Committed complete144endpoint freeze required')
    if sha(PUBLIC/'first_full_refit.json')!=freeze['first_full_refit_sha256'] or not json.loads((PUBLIC/'first_full_refit.json').read_text())['scientific_state_exact']:
        raise ValueError('Complete real refit verification changed')
    result={}
    for ref in freeze['fits']:
        path=ROOT/ref['path'];parent.parent.parent.committed(path);row=json.loads(path.read_text())
        if sha(path)!=ref['sha256'] or sha(ROOT/row['checkpoint']['path'])!=row['checkpoint']['sha256'] or row['step']!=2000:
            raise ValueError('Frozen detector endpoint changed')
        key=row['context'],row['seed'],row['bank'],row['representation']
        if key in result:raise ValueError('Repeated endpoint')
        result[key]=row
    expected={(c,s,b,r) for c in reg['contexts'] for s in reg['seeds'] for b in reg['banks'] for r in reg['representations']}
    if set(result)!=expected:raise ValueError('Full original detector training roster required')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=('register','pilot','train'))
    parser.add_argument('--resume',action='store_true');args=parser.parse_args();PRIVATE.mkdir(parents=True,exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.phase=='register':admission(register=True)
        elif args.phase=='pilot':pilot(args.resume)
        else:train(args.resume)
