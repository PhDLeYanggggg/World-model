"""Fixed active-producer fitting-only readout, then complete exact replay."""
import argparse
import fcntl
import json
from pathlib import Path
import platform
import sys
import time
if platform.system()=='Darwin' and platform.machine()!='arm64':raise RuntimeError('Native arm64 before Torch')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts import run_m3w_active_detector_forecaster as run
from src.evaluation import m3w_active_detector_readout as api

REG=run.PUBLIC/'readout_registration.json'
FILES=(Path(__file__),ROOT/'src/evaluation/m3w_active_detector_readout.py',ROOT/'tests/test_m3w_active_detector_readout.py',
    ROOT/'src/evaluation/m3w_detector_forecast_readout.py',ROOT/'src/evaluation/m3w_native_metrics.py',run.PUBLIC/'readout_protocol.md')


def admission(register=False):
    cfg,data,jobs,arrays,controls=run.admission();learned=run.models()
    old_complete=run.store.old.PUBLIC/'complete.json';ref=json.loads(old_complete.read_text())['arrays']['ids']
    if run.sha(ROOT/ref['path'])!=ref['sha256']:raise ValueError('Original query whitelist changed')
    ids=np.load(ROOT/ref['path'],mmap_mode='r',allow_pickle=False)
    if len(ids)!=149510 or np.any(np.diff(ids)<=0):raise ValueError('Same149510sorted query IDs required')
    run.once(REG,dict(bindings={str(p.relative_to(ROOT)):run.sha(p) for p in FILES},
        training_freeze_sha256=run.sha(run.PUBLIC/'training_freeze.json'),query_whitelist_sha256=ref['sha256'],
        controller_input_complete_sha256=run.sha(old_complete),primary='single_history',groups=36,
        active_partial_geometry=True,controller_held_outcomes_opened=False,independent_roles_opened=False,deployment_changed=False))
    if not register:
        for rel,digest in json.loads(REG.read_text())['bindings'].items():
            if run.sha(ROOT/rel)!=digest:raise ValueError('Registered active readout changed')
            run.committed(ROOT/rel)
        run.committed(REG)
    return cfg,data,jobs,arrays,controls,learned,ids


def execute(verify=False,resume=False):
    cfg,data,jobs,arrays,controls,learned,whitelist=admission();finish=run.PUBLIC/'complete.json'
    if finish.exists() and not verify:raise ValueError('Completed active readout immutable')
    start,beat,guard=run.runtime(cfg);rows=[];refs=[]
    for job in jobs:
        key=job['key'];ti=job['old_identity'];ids=np.asarray(whitelist[~np.isin(data['sites'][whitelist],ti['fit_sites'])])
        if not len(ids) or np.isin(ids,job['design']['train_ids']).any():raise ValueError('Producer-excluded old fitting queries only')
        predictions={};relative={}
        for arm in cfg['representations']:
            prepared,norm=run.prepare_arm(data,arrays,job,arm);ref=learned[key,arm]
            if norm!=ref['normalizer'] or ref['parent_trial']!=ti:raise ValueError('Frozen active normalizer or parent changed')
            state=torch.load(ROOT/ref['checkpoint']['path'],map_location='cpu',weights_only=False)
            if state['step']!=4000 or state['identity']['normalizer']!=norm:raise ValueError('Complete active checkpoint required')
            torch.manual_seed(ti['seed']);model=run.api.ActiveDetectorForecaster(job['design']['baseline_index'],arm,**cfg['architecture'])
            model.load_state_dict(state['model'])
            with run.api.installed():p=run.api.native.predict(model,prepared,ids,128)
            relative[arm]=p;predictions[arm]=p.astype(float)+data['origin'][ids,None]
            beat(state='active_proposal_inference',trial=key,representation=arm,rows=len(ids),controller_held_scored=False)
            del model,prepared,state
        parent=run.parent.make_model(job)
        parent.load_state_dict(torch.load(ROOT/controls[key]['checkpoint']['path'],map_location='cpu',weights_only=False)['model'])
        previous=run.api.native.predict(parent,data,ids,128);np.testing.assert_array_equal(previous,relative['disabled']);del parent
        physical=np.stack([run.source.baseline_numpy(data['history'][ids],k) for k in range(6)],1)
        result=api.site_readout(ids,data['sites'][ids],data['recordings'][ids],data['history'][ids],data['target_eval'][ids],
            data['valid'][ids],physical,job['design']['baseline_index'],predictions,easy_cut=ti['easy_cut'],hard_cut=ti['hard_cut'])
        for arm in cfg['representations']:
            path=run.PUBLIC/'groups'/(key+'_'+arm+'.json')
            if path.exists() and not(verify or resume):raise ValueError('Explicit partial readout --resume required')
            value=dict(trial=key,kind='single',seed=ti['seed'],representation=arm,result=result,
                ids_sha256=run.source.array_hash(ids),prediction_sha256={a:run.source.array_hash(v) for a,v in relative.items()},
                disabled_legacy_exact=True,actual_parent_checkpoint_sha256=controls[key]['checkpoint']['sha256'],
                training_freeze_sha256=run.sha(run.PUBLIC/'training_freeze.json'),readout_registration_sha256=run.sha(REG),
                result_source='fresh_run_actual_active_producer_excluded_SOURCE_development',
                controller_held_scored=False,independent_roles_opened=False)
            if verify:
                if value!=json.loads(path.read_text()):raise ValueError('Active fresh prediction/metric replay differs')
            else:run.once(path,value)
            rows.append(value);refs.append(dict(path=str(path.relative_to(ROOT)),sha256=run.sha(path)))
        beat(state='active_readout_progress',trial=key,groups=len(rows),total=36)
    summary=api.summarize(rows,json.loads(run.REG.read_text())['parent_jobs'],cfg)
    if verify:
        if summary!=json.loads((run.PUBLIC/'summary.json').read_text()) or refs!=json.loads(finish.read_text())['groups']:
            raise ValueError('Complete active summary replay differs')
        run.once(run.PUBLIC/'verification.json',dict(all36groups_exact=True,all9fresh_parent_predictions_exact=True,
            summary_sha256=run.sha(run.PUBLIC/'summary.json'),complete_sha256=run.sha(finish)))
    else:
        run.once(run.PUBLIC/'summary.json',summary)
        run.once(finish,dict(pid=run.os.getpid(),seconds=time.monotonic()-start,peak_RSS_bytes=run.resource.getrusage(run.resource.RUSAGE_SELF).ru_maxrss,
            groups=refs,summary_sha256=run.sha(run.PUBLIC/'summary.json'),new_optimizer_updates=0,
            controller_held_scored=False,independent_roles_opened=False))
    run.write(run.PRIVATE/('verification_runtime.json' if verify else 'readout_runtime.json'),
        dict(pid=run.os.getpid(),seconds=time.monotonic()-start,peak_RSS_bytes=run.resource.getrusage(run.resource.RUSAGE_SELF).ru_maxrss))
    beat(state='all36active_readouts_verified' if verify else 'all36active_readouts_complete',proposal_input_evidence_gate=summary['proposal_input_evidence_gate'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=('register','run','verify'))
    parser.add_argument('--resume',action='store_true');args=parser.parse_args();run.PRIVATE.mkdir(parents=True,exist_ok=True)
    with (run.PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.phase=='register':admission(register=True)
        else:execute(args.phase=='verify',args.resume)
