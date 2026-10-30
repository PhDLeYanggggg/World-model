"""Fixed producer-excluded fitting-ID readout, then fresh exact inference replay."""
import argparse
import fcntl
import json
from pathlib import Path
import platform
import sys
import time

if platform.system()=='Darwin' and platform.machine()!='arm64':raise RuntimeError('Native arm64 required before Torch')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts import run_m3w_detector_forecaster as run
from src.evaluation import m3w_detector_forecast_readout as api

REG=run.PUBLIC/'readout_registration.json'
FILES=(Path(__file__),ROOT/'src/evaluation/m3w_detector_forecast_readout.py',
    ROOT/'tests/test_m3w_detector_forecast_readout.py',run.PUBLIC/'readout_protocol.md')


def admission(register=False):
    cfg,data,jobs,arrays,old=run.admission();learned=run.models()
    old_complete=run.inputs.old.PUBLIC/'complete.json';r=json.loads(old_complete.read_text())['arrays']['ids']
    if run.sha(ROOT/r['path'])!=r['sha256']:raise ValueError('Original controller fitting ID whitelist changed')
    ids=np.load(ROOT/r['path'],mmap_mode='r',allow_pickle=False)
    if len(ids)!=149510 or np.any(np.diff(ids)<=0):raise ValueError('Original149510sorted fitting IDs required')
    run.once(REG,dict(bindings={str(p.relative_to(ROOT)):run.sha(p) for p in FILES},
        training_freeze_sha256=run.sha(run.PUBLIC/'training_freeze.json'),
        producer_input_complete_sha256=run.sha(run.inputs.PUBLIC/'complete.json'),
        old_controller_complete_sha256=run.sha(old_complete),controller_ids_sha256=r['sha256'],
        primary='single_history',groups=72,training_run=False,scope='old_controller_fitting_ids_producer_excluded_only',
        controller_held_outcomes_opened=False,independent_roles_opened=False,deployment_changed=False))
    if not register:
        for rel,digest in json.loads(REG.read_text())['bindings'].items():
            if run.sha(ROOT/rel)!=digest:raise ValueError('Fixed producer readout changed')
            run.committed(ROOT/rel)
        run.committed(REG)
    return cfg,data,jobs,arrays,old,learned,ids


def execute(verify=False,resume=False):
    cfg,data,jobs,arrays,old,learned,whitelist=admission();finish=run.PUBLIC/'complete.json'
    if finish.exists() and not verify:raise ValueError('Completed producer readout immutable; use verify')
    start,beat,guard=run.runtime(cfg);rows=[];refs=[]
    for key,fold,ti in jobs:
        ids=np.asarray(whitelist[~np.isin(data['sites'][whitelist],ti['fit_sites'])])
        if not len(ids) or np.isin(ids,fold['train_ids']).any():raise ValueError('Producer-excluded fitting readout only')
        predictions={};relative={}
        for arm in cfg['representations']:
            guard();prepared,norm=run.prepare_arm(data,arrays,ti,arm);ref=learned[key,arm]
            if norm!=ref['normalizer'] or ref['original_producer']!=ti:raise ValueError('Own-fit normalizer or producer role changed')
            state=torch.load(ROOT/ref['checkpoint']['path'],map_location='cpu',weights_only=False)
            if state['identity']['normalizer']!=norm or state['step']!=4000:raise ValueError('Original final producer state required')
            torch.manual_seed(ti['seed']);model=run.api.DetectorForecaster(cfg['architecture'],ti['baseline_index'],arm)
            model.load_state_dict(state['model'])
            with run.api.installed():p=run.api.native.predict(model,prepared,ids,batch_size=128)
            relative[arm]=p;predictions[arm]=p.astype(float)+data['origin'][ids,None]
            beat(state='source_proposal_inference',trial=key,representation=arm,rows=len(ids),controller_held_scored=False)
            del prepared,state,model
        torch.manual_seed(ti['seed']);legacy=run.api.SourceForecaster(cfg['architecture'],ti['baseline_index'])
        state=torch.load(ROOT/old[key]['checkpoint'],map_location='cpu',weights_only=False);legacy.load_state_dict(state['model'])
        previous=run.api.native.predict(legacy,data,ids,batch_size=128)
        np.testing.assert_array_equal(previous,relative['disabled']);del legacy,state
        physical=np.stack([run.source.baseline_numpy(data['history'][ids],k) for k in range(6)],1)
        result=api.site_readout(ids,data['sites'][ids],data['recordings'][ids],data['history'][ids],
            data['target_eval'][ids],data['valid'][ids],physical,ti['baseline_index'],predictions,
            easy_cut=ti['easy_cut'],hard_cut=ti['hard_cut'])
        for arm in cfg['representations']:
            path=run.PUBLIC/'groups'/(key+'_'+arm+'.json')
            if path.exists() and not(verify or resume):raise ValueError('Explicit partial readout resume required')
            value=dict(trial=key,kind=ti['kind'],fold=ti['fold'],seed=ti['seed'],representation=arm,result=result,
                ids_sha256=run.source.array_hash(ids),prediction_sha256={a:run.source.array_hash(p) for a,p in relative.items()},
                original_checkpoint_sha256=old[key]['checkpoint_sha256'],disabled_legacy_exact=True,
                registration_sha256=run.sha(REG),training_freeze_sha256=run.sha(run.PUBLIC/'training_freeze.json'),
                result_source='fresh_producer_excluded_SOURCE_development_inference',controller_held_scored=False,independent_roles_opened=False)
            if verify:
                if value!=json.loads(path.read_text()):raise ValueError('Exact producer prediction/metric replay differs')
            else:run.once(path,value)
            rows.append(value);refs.append(dict(path=str(path.relative_to(ROOT)),sha256=run.sha(path)))
        beat(state='proposal_readout_progress',trial=key,groups=len(rows),total=72)
    registration=json.loads(run.REG.read_text());summary=api.summarize(rows,registration['producer_jobs'],cfg)
    if verify:
        if summary!=json.loads((run.PUBLIC/'summary.json').read_text()) or refs!=json.loads(finish.read_text())['groups']:
            raise ValueError('Complete producer summary replay differs')
        run.once(run.PUBLIC/'verification.json',dict(exact_fresh_inference_replay=True,groups=72,
            summary_sha256=run.sha(run.PUBLIC/'summary.json'),complete_sha256=run.sha(finish),
            all18disabled_legacy_predictions_exact=True))
    else:
        run.once(run.PUBLIC/'summary.json',summary)
        run.once(finish,dict(pid=run.os.getpid(),seconds=time.monotonic()-start,groups=refs,
            peak_RSS_bytes=run.resource.getrusage(run.resource.RUSAGE_SELF).ru_maxrss,
            summary_sha256=run.sha(run.PUBLIC/'summary.json'),new_optimizer_updates=0,
            controller_held_scored=False,independent_roles_opened=False))
    run.write(run.PRIVATE/('verification_runtime.json' if verify else 'readout_runtime.json'),
        dict(pid=run.os.getpid(),seconds=time.monotonic()-start,peak_RSS_bytes=run.resource.getrusage(run.resource.RUSAGE_SELF).ru_maxrss))
    beat(state='producer_readout_verified' if verify else 'producer_readout_complete',groups=72,
        proposal_input_evidence_gate=summary['proposal_input_evidence_gate'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=('register','run','verify'))
    parser.add_argument('--resume',action='store_true');args=parser.parse_args();run.PRIVATE.mkdir(parents=True,exist_ok=True)
    with (run.PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.phase=='register':admission(register=True)
        else:execute(args.phase=='verify',args.resume)
