"""Fixed two-arm source repair; reuse verified encodings, never tune held outputs."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_m3w_source_pretrained_temporal import load_config as load_parent, context, FeatureCorpus
from scripts.run_m3w_source_crossfit import immutable_json,save_arrays,array_hash,metrics
from src.world_model.m3w_source_motion_candidate import fit_motion_candidate
from src.world_model.m3w_source_cost_dynamics import forecast
from src.world_model.m3w_source_crossfit import assemble_oof,cost_labels
from src.world_model.m3w_source_temporal_centered import ARMS,CenteredTemporalDynamics
from src.world_model.m3w_offline_visual_data import json_write
from src.evaluation.m3w_experiment_contract import file_digest
import numpy as np
import torch


def load_config(path):
    reg=json.loads(path.read_text())
    if (str(path)!=reg['registration_path'] or reg['role']!='training_only_temporal_centering'
            or reg['arms']!=list(ARMS) or reg['seeds']!=[17,29,43]
            or reg['sites']!=['coupa','deathCircle','gates','hyang']
            or reg['models']!=24 or reg['updates']!=240000 or reg['selection'] or reg['new_deployment']):
        raise ValueError('Fixed source-only centering contract required')
    for p,sha in reg['bindings'].items():
        if file_digest(ROOT/p)!=sha:raise ValueError('Changed dependency: '+p)
    return reg


def setup(reg):
    parent=load_parent(Path(reg['parent_registration']))
    old=ROOT/parent['reports']; ver=json.loads((old/'verification.json').read_text())
    for p,sha in ver['artifact_hashes'].items():
        if file_digest(ROOT/p)!=sha:raise ValueError('Changed verified parent asset: '+p)
    assert parent['training']==reg['training']
    data=context(parent)
    cached=FeatureCorpus(data,ROOT/parent['output'],json.loads((old/'preparation.json').read_text()))
    reference=json.loads((old/'report.json').read_text())
    return parent,data,cached,reference


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registration',type=Path,required=True);p.add_argument('--replay',action='store_true')
    p.add_argument('--trial');p.add_argument('--stop-at',type=int)
    args=p.parse_args();torch.set_num_threads(4);torch.set_num_interop_threads(1)
    reg=load_config(args.registration)
    if args.stop_at is not None and (not args.trial or args.replay):raise ValueError('Named training pilot required')
    parent,data,cached,reference=setup(reg)
    private,public=ROOT/reg['output'],ROOT/reg['reports']
    identity=dict(registration_sha256=file_digest(args.registration),data=data.identity,assignment=data.assignment_hash,
        feature_store_sha256=file_digest(ROOT/parent['output']/'feature_store.npz'),
        torch=torch.__version__,numpy=np.__version__,torch_threads=4,interop_threads=1,num_workers=0)
    immutable_json(private/'identity.json',identity)
    def beat(**kw):
        event=dict(pid=os.getpid(),timestamp_unix=time.time(),**kw)
        json_write(private/'heartbeat.json',event);print(json.dumps(event),flush=True)
    beat(state='verified_inputs')
    allowed={f'{site}_{arm}_seed{seed}' for site in reg['sites'] for arm in ARMS for seed in reg['seeds']}
    if args.trial and args.trial not in allowed:raise ValueError('Unknown trial')
    trials=[];checks=[];replays=[];updates=0
    for site in reg['sites']:
        train,weights,held,outer,scale=data.configure(site)
        old=next(t for t in reference['trials'] if t['site']==site)
        fold=old['identity']['fold'];hard=old['identity']['training_hard_cut']
        assert array_hash(train)==fold['training_ids_sha256'] and array_hash(held)==fold['held_ids_sha256']
        assert scale==fold['cost_scale']
        assert array_hash(data.normalizer['mean'],data.normalizer['std'],data.normalizer['constant'])==fold['normalizer_sha256']
        for illegal in (np.array([0]),outer[:1]):
            try:cached.inputs(illegal)
            except ValueError:pass
            else:raise AssertionError('Prohibited input role')
        try:cached.inputs(held[:1],training=True)
        except ValueError:pass
        else:raise AssertionError('Held input reached trainer')
        before=cached.inputs(held[:8]); labels=data.target.copy()
        try:
            data.target[:]=np.nan; after=cached.inputs(held[:8])
        finally:data.target[:]=labels
        assert all(torch.equal(a,b) for x,y in zip(before,after) for a,b in zip(x,y))
        checks.append(dict(site=site,fold=fold,target_poison_rows=8))
        for seed in reg['seeds']:
            control=next(t for t in reference['trials'] if t['site']==site and t['seed']==seed and t['arm']=='sequence')
            original=torch.load(ROOT/control['checkpoint_path'],map_location='cpu',weights_only=False)
            for arm in ARMS:
                key=f'{site}_{arm}_seed{seed}'
                if args.trial and key!=args.trial:continue
                ti=dict(identity,site=site,seed=seed,arm=arm,fold=fold,training_hard_cut=hard)
                cp=private/'checkpoints'/f'{key}.pt';pp=private/'predictions'/f'{key}.npz';rp=private/'trials'/f'{key}.json'
                saved_receipt=json.loads(rp.read_text()) if rp.exists() else None
                if saved_receipt:
                    assert saved_receipt['identity']==ti and file_digest(cp)==saved_receipt['checkpoint_sha256']
                    assert file_digest(pp)==saved_receipt['prediction_sha256']
                    if not args.replay:trials.append(saved_receipt);continue
                elif args.replay:raise ValueError('Missing completed trial')
                torch.manual_seed(seed);model=CenteredTemporalDynamics(arm)
                if not args.replay:
                    beat(state='fit_or_resume',trial=key)
                    result=fit_motion_candidate(model,lambda q:cached.inputs(q,training=True),data.loss_targets,
                        train,weights,scale=scale,seed=seed,config=reg['training'],identity=ti,checkpoint=cp,
                        heartbeat=lambda **kw:beat(trial=key,**kw),suppress_zero_targets=False,stop_at=args.stop_at)
                    updates+=result['new_updates']
                    if not result['complete']:beat(state='pilot_complete_no_forecast',trial=key,step=result['step']);return
                saved=torch.load(cp,map_location='cpu',weights_only=False)
                assert saved['identity']==ti and saved['step']==10000 and not saved['suppress_zero_targets']
                assert torch.equal(saved['sampler_rng'],original['sampler_rng'])
                np.testing.assert_array_equal(saved['train_ids'],original['train_ids'])
                np.testing.assert_array_equal(saved['draw_counts'],original['draw_counts'])
                model.load_state_dict(saved['model'])
                arrays=dict(train_ids=train,held_ids=held,
                    train_prediction=forecast(model,cached.inputs,train,'mask_only'),
                    held_prediction=forecast(model,cached.inputs,held,'mask_only'))
                if args.replay:
                    with np.load(pp,allow_pickle=False) as a:
                        for name,value in arrays.items():np.testing.assert_array_equal(a[name],value)
                    replays.append(key);beat(state='exact_replay',trial=key);continue
                save_arrays(pp,arrays)
                trial=dict(identity=ti,trial=key,site=site,seed=seed,arm=arm,fit=result,
                    result_source='fresh_run_torch_temporal_centering',parameters=sum(v.numel() for v in model.parameters()),
                    training=metrics(data,train,arrays['train_prediction'],scale,hard),
                    held=metrics(data,held,arrays['held_prediction'],scale,hard),
                    checkpoint_path=str(cp.relative_to(ROOT)),checkpoint_sha256=file_digest(cp),
                    prediction_path=str(pp.relative_to(ROOT)),prediction_sha256=file_digest(pp),sampler_matches_control=True)
                immutable_json(rp,trial);trials.append(trial);beat(state='trial_complete',trial=key)
    if args.trial:return
    immutable_json(public/'input_checks.json',dict(identity=identity,folds=checks,main_outer_scored=0,
        sensor_asof_certified=False,target_in_features=False,all_rows_retained=True))
    if args.replay:
        assert len(replays)==24
        immutable_json(public/'replay.json',dict(exact_replays=replays,new_updates=0))
        beat(state='replay_complete',models=24);return
    assert len(trials)==24 and sum(t['fit']['step'] for t in trials)==240000
    archives=[]
    for arm in ARMS:
        for seed in reg['seeds']:
            pieces=[]
            for trial in trials:
                if trial['arm']!=arm or trial['seed']!=seed:continue
                with np.load(ROOT/trial['prediction_path'],allow_pickle=False) as a:
                    pieces.append(dict(ids=a['held_ids'].copy(),prediction=a['held_prediction'].copy(),
                        cost_scale=np.full(len(a['held_ids']),trial['identity']['fold']['cost_scale'])))
            prediction,scale=assemble_oof(cached.ids,pieces)
            labels=cost_labels(prediction,data.target[cached.ids-data.nmain],scale)
            path=private/'oof'/f'{arm}_seed{seed}.npz'
            save_arrays(path,dict(ids=cached.ids,prediction=prediction,cost_scale=scale,**labels))
            archives.append(dict(arm=arm,seed=seed,path=str(path.relative_to(ROOT)),sha256=file_digest(path)))
    immutable_json(public/'report.json',dict(identity=identity,trials=trials,oof_labels=archives,
        models=24,optimizer_updates=240000,rows=len(cached.ids),new_policy=False,new_deployment=False,
        main_rows_scored=0,outer_rows_scored=0,stage5c_executed=False,smc_enabled=False))
    beat(state='training_complete',models=24,new_updates=updates)


if __name__=='__main__':main()
