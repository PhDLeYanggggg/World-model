"""Matched topology-only training with immutable controls and pre-readout freeze."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 required before Torch import')
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(k, '4')
import numpy as np
import torch
from scripts import run_m3w_european_partial_neighbor_refit as old
from src.world_model.m3w_agent_track_context import AgentTrackSourceForecaster
from src.world_model.m3w_native_forecast import fit_trial, predict
from src.evaluation.m3w_native_metrics import native_errors
from src.evaluation import m3w_agent_track_refit as metrics

BASE = old.BASE
PUBLIC = BASE/'european_agent_track_refit_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/european_agent_track_refit_v1'
CONFIG = 'configs/m3w_european_agent_track_refit_v1.json'
FILES = [CONFIG, 'scripts/run_m3w_european_agent_track_refit.py',
    'src/world_model/m3w_agent_track_context.py', 'src/evaluation/m3w_agent_track_refit.py',
    'tests/test_m3w_agent_track_refit.py', str(PUBLIC.relative_to(ROOT)/'protocol.md')]
digest, artifact, immutable_json, committed = old.digest, old.artifact, old.immutable_json, old.committed


def beat(state, **kw):
    row = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), state=state, **kw)
    old.parent.json_write(PRIVATE/'heartbeat.json', row)
    with (PRIVATE/'events.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
    print(json.dumps(row), flush=True)


def load(create=False):
    cfg = json.loads((ROOT/CONFIG).read_text())
    seal_path = old.PUBLIC/'verification.json'
    assert digest(seal_path) == cfg['partial_refit_seal_sha256']
    seal = json.loads(seal_path.read_text())
    for p,h in seal['source_bindings'].items(): assert digest(ROOT/p) == h
    for p,h in seal['artifacts'].items(): assert digest(old.PUBLIC/p) == h
    _, reg, data, receipts, jobs, old_id = old.load()
    data = old.prepared(data, receipts, old_id)
    frozen = json.loads((old.PUBLIC/'prediction_freeze.json').read_text())
    controls = []
    for j,ref,train_ref in zip(jobs, frozen['predictions'], frozen['training']):
        assert j['key'] == ref['key']
        assert artifact(ROOT/train_ref['path']) == train_ref
        train = json.loads((ROOT/train_ref['path']).read_text())
        for r in (train['checkpoint'], ref['partial'], ref['legacy']): assert artifact(ROOT/r['path']) == r
        assert train['identity']['parent_trial'] == j['old_identity']
        j['flat_checkpoint'] = train['checkpoint']; j['flat_prediction'] = ref['partial']
        controls.append(dict(key=j['key'], training=train_ref, checkpoint=train['checkpoint'],
                             prediction=ref['partial'], original_prediction=ref['legacy']))
    assert cfg['steps'] == reg['training']['steps'] == 4000 and cfg['seeds'] == [17,29,43]
    assert cfg['producer_folds'] == [0,1,2] and len(jobs) == cfg['new_models'] == 9
    identity = dict(parent_seal=artifact(seal_path), geometry=artifact(old.PRIVATE/'geometry.json'),
        controls=controls, bindings={p:digest(ROOT/p) for p in FILES}, training=reg['training'])
    path = PUBLIC/'registration.json'
    if create: immutable_json(path, identity)
    else:
        assert json.loads(path.read_text()) == identity; committed(path)
    return cfg, reg, data, jobs, identity


def make_model(job):
    torch.manual_seed(job['old_identity']['seed'])
    return AgentTrackSourceForecaster(job['design']['baseline_index'], width=64, heads=4, layers=2)


def train(cfg, reg, data, jobs, phase, resume):
    for j in (jobs[:1] if phase == 'pilot' else jobs):
        home = PRIVATE/'grouped'/j['key']; done = home/'complete.json'
        identity = dict(registration=artifact(PUBLIC/'registration.json'), parent_trial=j['old_identity'],
                        arm='grouped_track', geometry=artifact(old.PRIVATE/'geometry.json'))
        if done.exists():
            record=json.loads(done.read_text()); assert record['identity'] == identity
            assert artifact(ROOT/record['checkpoint']['path']) == record['checkpoint']
            beat('cached_verified_endpoint', trial=j['key']); continue
        if shutil.disk_usage(PRIVATE).free < 10*2**30: raise OSError('Preserve10GiB; resume checkpoint')
        model = make_model(j); control = old.make_model(j, reg, 'partial')
        for k,v in control.state_dict().items(): torch.testing.assert_close(v, model.state_dict()[k], rtol=0, atol=0)
        del control
        fit=fit_trial(model,data,j['design'],seed=j['old_identity']['seed'],settings=reg['training'],
            identity=identity,directory=home,resume=resume,stop_at=cfg['pilot_updates'] if phase=='pilot' else None,
            heartbeat=lambda **kw:beat(trial=j['key'], **kw))
        cp=home/'checkpoint.pt'
        if phase=='pilot':
            immutable_json(PRIVATE/'pilot.json',dict(fit=fit,checkpoint=artifact(cp),same_budget_resume=True)); return
        state=torch.load(cp,map_location='cpu',weights_only=False)
        control=torch.load(ROOT/j['flat_checkpoint']['path'],map_location='cpu',weights_only=False)
        old.state_matches(state,control,parameters=False)
        assert fit['complete'] and fit['held_rows_sampled']==0 and fit['parameters']==88514
        immutable_json(done,dict(identity=identity,checkpoint=artifact(cp),fit=fit,
            sampling_exact_vs_flat=True,result_source='fresh_native_torch_training'))
        beat('trained_endpoint',trial=j['key'],fit_seconds=fit['seconds'])


def endpoints(jobs):
    refs=[]
    for j in jobs:
        path=PRIVATE/'grouped'/j['key']/'complete.json'; d=json.loads(path.read_text())
        assert d['fit']['step']==4000 and d['fit']['held_rows_sampled']==0 and d['sampling_exact_vs_flat']
        assert d['identity']['registration']==artifact(PUBLIC/'registration.json')
        assert d['identity']['parent_trial']==j['old_identity']
        assert artifact(ROOT/d['checkpoint']['path'])==d['checkpoint']
        refs.append(artifact(path))
    return refs


def produce(reg,data,jobs,verify=False):
    training=endpoints(jobs); predictions=[]
    for j in jobs:
        ids=j['design']['held_ids']; home=PRIVATE/'grouped'/j['key']
        model=make_model(j)
        model.load_state_dict(torch.load(home/'checkpoint.pt',map_location='cpu',weights_only=False)['model'])
        p=predict(model,data,ids,128)
        flat=old.make_model(j,reg,'partial')
        flat.load_state_dict(torch.load(ROOT/j['flat_checkpoint']['path'],map_location='cpu',weights_only=False)['model'])
        with np.load(ROOT/j['flat_prediction']['path'],allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'],ids)
            np.testing.assert_array_equal(predict(flat,data,ids,128),z['prediction'])
        path=home/'prediction.npz'
        if verify or path.exists():
            with np.load(path,allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'],ids); np.testing.assert_array_equal(z['prediction'],p)
        else:
            temp=path.with_suffix('.tmp')
            with temp.open('wb') as f: np.savez(f,ids=ids,prediction=p)
            os.replace(temp,path)
        predictions.append(dict(key=j['key'],grouped=artifact(path),flat=j['flat_prediction'],
            original=j['old_prediction'],rows=len(ids),flat_fresh_inference_exact=True))
        beat('prediction_replayed' if verify else 'prediction_frozen',trial=j['key'],rows=len(ids))
    immutable_json(PUBLIC/'prediction_freeze.json',dict(registration=artifact(PUBLIC/'registration.json'),
        training=training,predictions=predictions,new_updates=36000,comparison_targets_used=False,
        independent_roles_read=False))
    if verify: immutable_json(PUBLIC/'prediction_replay.json',dict(all_exact=True,model_pairs=9,
        prediction_freeze_sha256=digest(PUBLIC/'prediction_freeze.json')))


def evaluate(cfg,data,jobs,verify=False):
    path=PUBLIC/'prediction_freeze.json'; committed(path)
    frozen=json.loads(path.read_text()); rows=[]; proxies=[]; slices=[]
    for j,ref in zip(jobs,frozen['predictions']):
        assert j['key']==ref['key']; ids=j['design']['held_ids']; pred={}
        for arm in ('grouped','flat','original'):
            assert artifact(ROOT/ref[arm]['path'])==ref[arm]
            with np.load(ROOT/ref[arm]['path'],allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'],ids); pred[arm]=z['prediction'].copy()
        errors={k:native_errors(v.astype(float)+data['origin'][ids,None],data['target_eval'][ids],
            data['valid'][ids],np.ones(len(ids))) for k,v in pred.items()}
        cv=data['baseline_ade'][ids,1]
        subsets=old.metrics.masks(cv,j['design']['easy_cut'],j['design']['hard_cut'])
        support=metrics.causal_slices(data['geometry'][ids]); sites=data['sites'][ids]
        assert not set(sites)&set(j['old_identity']['fit_sites'])
        for site in sorted(set(sites)):
            take=sites==site
            for ep,ei in [('ADE',0),('FDE',1)]:
                reference,ep_cv=old.metrics.endpoint_references(data,ids,j['design']['baseline_index'],ep)
                for name,m in subsets.items():
                    result=metrics.slice_metrics(errors['grouped'][ei][take],errors['flat'][ei][take],
                        reference[take],ep_cv[take],m[take])
                    secondary=old.metrics.slice_metrics(errors['grouped'][ei][take],errors['original'][ei][take],
                        reference[take],ep_cv[take],m[take])
                    result['gain_vs_original_percent']=secondary.get('gain_vs_legacy_percent')
                    rows.append(dict(trial=j['key'],fold=j['old_identity']['fold'],seed=j['old_identity']['seed'],
                        site=str(site),subset=name,endpoint=ep,metric=result))
            for name,m in support.items():
                result=metrics.slice_metrics(errors['grouped'][0][take],errors['flat'][0][take],
                    data['baseline_ade'][ids,j['design']['baseline_index']][take],cv[take],m[take])
                slices.append(dict(trial=j['key'],site=str(site),subset=name,metric=result))
            def smooth(p): return float(np.linalg.norm(np.diff(p[take],n=2,axis=1),axis=2).mean())
            proxies.append(dict(trial=j['key'],site=str(site),rows=int(take.sum()),
                recordings=int(len(np.unique(data['recordings'][ids][take]))),finite_output=bool(np.isfinite(pred['grouped'][take]).all()),
                raw_step_acceleration_grouped=smooth(pred['grouped']),raw_step_acceleration_flat=smooth(pred['flat'])))
        beat('source_scored',trial=j['key'])
    roster=sorted(set(data['sites'])); assert len(rows)==576 and len(slices)==360 and len(roster)==12
    def aggregate(rr,key): return metrics.paired_localities(rr,roster,key,cfg['bootstrap_draws'],cfg['bootstrap_seed'])
    summaries={}
    for ep in ('ADE','FDE'):
        for subset in ('all','positive_easy','hard','zero_CV'):
            rr=[r for r in rows if r['endpoint']==ep and r['subset']==subset]
            summaries[ep+'_'+subset]={k:aggregate(rr,k) for k in
                ('gain_vs_flat_percent','gain_vs_original_percent','gain_vs_reference_percent','gain_vs_CV_percent','absolute_harm_vs_flat')}
    doc=dict(result_source='fresh_source_development_readout',rows=rows,summaries=summaries,
        per_seed={str(seed):aggregate([r for r in rows if r['endpoint']=='ADE' and r['subset']=='all' and r['seed']==seed],
            'gain_vs_flat_percent') for seed in cfg['seeds']},
        causal_slices=slices,causal_slice_summaries={name:aggregate([r for r in slices if r['subset']==name],
            'gain_vs_flat_percent') for name in support},motion_proxies=proxies,expected_localities=roster,
        independent_confirmation=False,deployment_changed=False)
    immutable_json(PUBLIC/'evaluation.json',doc)
    immutable_json(PUBLIC/'gates.json',metrics.gates(doc,cfg['easy_degradation_guard_percent']))
    if verify: immutable_json(PUBLIC/'evaluation_replay.json',dict(all_exact=True,rows=len(rows),
        evaluation_sha256=digest(PUBLIC/'evaluation.json')))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',required=True,choices=['register','pilot','train','predict','evaluate','replay','verify_eval'])
    p.add_argument('--resume',action='store_true'); args=p.parse_args()
    PRIVATE.mkdir(parents=True,exist_ok=True); PUBLIC.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    with (PRIVATE/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        beat('started',phase=args.phase,threads=4,workers=0,architecture=platform.machine())
        cfg,reg,data,jobs,identity=load(args.phase=='register')
        if args.phase in ('pilot','train'): train(cfg,reg,data,jobs,args.phase,args.resume)
        if args.phase in ('predict','replay'): produce(reg,data,jobs,args.phase=='replay')
        if args.phase in ('evaluate','verify_eval'): evaluate(cfg,data,jobs,args.phase=='verify_eval')
        beat('complete',phase=args.phase)


if __name__=='__main__': main()
