"""Matched coordinate-frame conditioning, without changing scientific roles."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Use the native arm64 environment')
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '4'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.build_m3w_observed_motion import load_registration
from scripts.run_m3w_track_event_sampling import check_saved
from scripts.run_m3w_objective_alignment import paired_interval
from src.evaluation.m3w_experiment_contract import file_digest
from src.world_model.m3w_offline_visual_data import json_write
from src.world_model.m3w_offline_visual_forecast import OfflineVisualForecast, forecast_metrics
from src.world_model.m3w_objective_alignment import supported_standardization
from src.world_model.m3w_observed_motion import feature_variant
from src.world_model.m3w_past_frame import past_frame, rotate_features, frame_prediction, fit_frame
from src.world_model.m3w_track_event_sampling import EVENT_NAMES, training_event_labels

ARMS = ('original','guard_only','past_frame')


def summarize(trials, reg):
    lookup = {(t['variant'],t['arm'],t['seed'],t['fold']):t for t in trials}
    def matrix(v,a,key):
        return np.array([[lookup[v,a,s,f]['vs_CV'][key] for f in range(3)] for s in reg['seeds']])
    output = {}
    for v in reg['variants']:
        for arm in ARMS:
            rows = [t for t in trials if t['variant'] == v and t['arm'] == arm]
            output[v+'_'+arm] = dict(
                vs_CV=paired_interval(matrix(v,arm,'primary_ADE'),matrix(v,arm,'reference_ADE'),reg['bootstrap_resamples']),
                vs_guard=paired_interval(matrix(v,arm,'primary_ADE'),matrix(v,'guard_only','primary_ADE'),reg['bootstrap_resamples']),
                easy_degradation_percent=[t['vs_CV']['easy_degradation_percent'] for t in rows],
                easy_absolute_harm=[t['vs_CV']['easy_absolute_harm'] for t in rows],
                train_equal_scene_gain_percent=[t['training_gain_percent'] for t in rows],
                safe_positive_fits=sum(t['vs_CV']['improvement_percent'] > 0 and
                    t['vs_CV']['easy_degradation_percent'] is not None and t['vs_CV']['easy_degradation_percent'] <= 2 for t in rows),
                fresh_fit_seconds=sum(t['fit']['fit_seconds'] for t in rows if arm != 'original'),
                quarter_turn_mean_prediction_gap=[t['frame_stress']['mean_gap'] for t in rows],
                quarter_turn_max_coordinate_gap=max(t['frame_stress']['max_coordinate_gap'] for t in rows))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration',type=Path,required=True)
    parser.add_argument('--trial'); parser.add_argument('--stop-at',type=int)
    parser.add_argument('--replay',action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    reg = json.loads(args.registration.read_text())
    if reg['role'] != 'fit_only_exploratory' or reg['arms'] != list(ARMS):
        raise ValueError('Fixed fit-only comparison required')
    for path,digest in reg['bindings'].items():
        if file_digest(ROOT/path) != digest:
            raise ValueError('Bound file changed: '+path)
    old,parent,contract,source,receipt = load_registration(ROOT/reg['control_registration'])
    if reg['training'] != old['training'] or reg['seeds'] != old['seeds'] or receipt['rows'] != 11966:
        raise ValueError('Changed matched budget/seeds/cohort')
    controls,inputs = ROOT/old['output'],ROOT/old['output']/'inputs'
    old_identity = dict(registration_sha256=file_digest(ROOT/reg['control_registration']),
        parent_protocol_sha256=contract.digest,source_manifest_sha256=file_digest(source/'data_manifest.json'))
    manifest = json.loads((inputs/'manifest.json').read_text())
    if manifest['identity'] != old_identity:
        raise ValueError('Feature lineage changed')
    for name,digest in manifest['arrays'].items():
        if file_digest(inputs/name) != digest:
            raise ValueError('Changed feature array')
    old_identity['motion_manifest_sha256'] = file_digest(inputs/'manifest.json')
    identity = dict(registration_sha256=file_digest(args.registration),source_identity=old_identity)
    output,reports = ROOT/reg['output'],ROOT/reg['reports']
    if not output.is_relative_to(ROOT/'data/stage_cvpr2027_experiments'):
        raise ValueError('Private artifacts must stay under data')
    if (output/'identity.json').exists() and json.loads((output/'identity.json').read_text()) != identity:
        raise ValueError('Experiment identity changed')
    json_write(output/'identity.json',identity)
    keys = {f'{v}_{a}_seed{s}_fold{f}' for v in reg['variants'] for a in ARMS for s in reg['seeds'] for f in range(3)}
    if args.trial and args.trial not in keys:
        raise ValueError('Unknown trial')
    if args.stop_at is not None and (not args.trial or '_original_' in args.trial or not 0 < args.stop_at <= 4000):
        raise ValueError('Pilot requires a registered fresh fit')
    def heartbeat(value):
        value = dict(pid=os.getpid(),time_unix=time.time(),**value)
        json_write(output/'heartbeat.json',value); print(json.dumps(value),flush=True)
    a = {n:np.load(source/(n+'.npy'),mmap_mode='r') for n in
         ('geometry','targets','baselines','folds','scale','image_rows','coverage')}
    if not np.array_equal(a['geometry'][:,308:].reshape(a['baselines'].shape),a['baselines']):
        raise ValueError('Typed baseline block differs from independent baseline array')
    metadata = json.loads((source/'rows.json').read_text())
    if not np.array_equal(a['folds'],[m['fold'] for m in metadata]):
        raise ValueError('Row/fold alignment changed')
    tracks = np.array([f"{m['recording']}:{m['agent']}" for m in metadata])
    records = np.array([m['recording'] for m in metadata])
    observed = torch.cat([torch.from_numpy(a['coverage'][a['image_rows'][i:i+128],None].astype(np.float32)/9.)
                          .mean((2,3,4)) for i in range(0,receipt['rows'],128)])
    motion,quality = [np.load(inputs/(n+'.npy'),mmap_mode='r') for n in ('motion','quality')]
    cv = receipt['baseline_names'].index('constant_velocity_causal_fd')
    base,target = torch.from_numpy(a['baselines'][:,cv].copy()),torch.from_numpy(a['targets'].copy())
    q,valid,kinds = past_frame(np.concatenate((a['geometry'],np.zeros((receipt['rows'],105))),1))
    anchor_counts = {str(k):int(n) for k,n in zip(*np.unique(kinds,return_counts=True))}
    trials,replays,updates = [],[],0
    for fold in range(3):
        train,held = np.flatnonzero(a['folds'] != fold),np.flatnonzero(a['folds'] == fold)
        allowed = np.zeros(len(target),bool); allowed[train] = True
        train_scenes = np.unique(a['folds'][train])
        train_errors = np.linalg.norm(a['baselines'][train].astype(float)-a['targets'][train,None],axis=-1).mean(-1)
        strongest = int(np.argmin(np.mean([train_errors[a['folds'][train] == f].mean(0) for f in train_scenes],axis=0)))
        for variant in reg['variants']:
            original = feature_variant(a['geometry'],motion,quality,variant)
            for arm in ARMS:
                features = rotate_features(original,q).astype(np.float32) if arm == 'past_frame' else original
                normalized,normalizer = supported_standardization(features[train],features)
                x = torch.from_numpy(normalized)
                rotations = torch.from_numpy(q.astype(np.float32)) if arm == 'past_frame' else torch.eye(2).repeat(len(x),1,1)
                supported = torch.from_numpy(valid.copy()) if arm != 'original' else torch.ones(len(x),dtype=torch.bool)
                def batch(ids):
                    if not allowed[ids.numpy()].all():
                        raise ValueError('Held labels requested during training')
                    return x[ids],observed[ids],base[ids],target[ids],rotations[ids],supported[ids]
                def infer(model,xx,obs,bb,qq,vv):
                    model.eval()
                    with torch.no_grad():
                        return np.concatenate([frame_prediction(model,xx[i:i+128],obs[i:i+128],bb[i:i+128],
                            qq[i:i+128],vv[i:i+128]).numpy() for i in range(0,len(xx),128)])
                for seed in reg['seeds']:
                    key = f'{variant}_{arm}_seed{seed}_fold{fold}'
                    if args.trial and args.trial != key:
                        continue
                    trial_identity = dict(identity,variant=variant,arm=arm,seed=seed,fold=fold)
                    cp,pp,rp = [output/d/(key+e) for d,e in [('checkpoints','.pt'),('predictions','.npz'),('trials','.json')]]
                    is_control = arm == 'original'
                    if is_control:
                        old_key = f'{variant}_row_log_seed{seed}_fold{fold}'
                        cp,pp = controls/'checkpoints'/(old_key+'.pt'),controls/'predictions'/(old_key+'.npz')
                    saved = json.loads(rp.read_text()) if rp.exists() else None
                    if saved:
                        check_saved(saved,trial_identity,cp,pp)
                        if not args.replay:
                            trials.append(saved); continue
                    torch.manual_seed(seed); model = OfflineVisualForecast(original.shape[1])
                    if is_control:
                        expected = dict(old_identity,variant=variant,objective='row_log',seed=seed,fold=fold)
                        control = json.loads((controls/'trials'/(old_key+'.json')).read_text())
                        check_saved(control,expected,cp,pp)
                        state = torch.load(cp,map_location='cpu',weights_only=False)
                        if state['identity'] != expected or state['config'] != reg['training'] or state['step'] != 4000:
                            raise ValueError('Control state changed')
                        model.load_state_dict(state['model']); fit = dict(control['fit'],new_updates_this_invocation=0,reused_not_fresh=True)
                    elif args.replay:
                        if saved is None:
                            raise ValueError('Replay needs completed trial')
                        state = torch.load(cp,map_location='cpu',weights_only=False)
                        if state['identity'] != trial_identity or state['config'] != reg['training'] or state['step'] != 4000 or not np.array_equal(state['train_ids'],train):
                            raise ValueError('Replay state changed')
                        model.load_state_dict(state['model'])
                    else:
                        fit = fit_frame(model,batch,torch.from_numpy(train),config=reg['training'],seed=seed,
                            identity=trial_identity,checkpoint=cp,heartbeat=lambda v:heartbeat(dict(trial=key,**v)),stop_at=args.stop_at)
                        updates += fit['new_updates_this_invocation']
                        if not fit['complete']:
                            heartbeat(dict(state='pilot_saved_no_held_evaluation',trial=key,step=fit['step'])); return
                    pred = infer(model,x[held],observed[held],base[held],rotations[held],supported[held])
                    if is_control or args.replay:
                        with np.load(pp) as prior:
                            if not np.array_equal(prior['prediction'],pred) or not np.array_equal(prior['held_indices'],held):
                                raise ValueError('Exact prediction replay failed')
                    if args.replay:
                        replays.append(dict(trial=key,prediction_exact=True)); continue
                    if not is_control:
                        pp.parent.mkdir(parents=True,exist_ok=True); np.savez(pp,prediction=pred,held_indices=held)
                    # Coordinate re-expression is target-free, with fixed training moments.
                    gaps = []
                    for matrix in (np.array([[0.,-1.],[1.,0.]]),-np.eye(2),np.array([[0.,1.],[-1.,0.]])):
                        r = np.broadcast_to(matrix,(len(held),2,2))
                        reexpressed = rotate_features(original[held],r)
                        qr,vr,_ = past_frame(reexpressed)
                        shifted = (rotate_features(reexpressed,qr) if arm == 'past_frame' else reexpressed).astype(np.float32)
                        z = np.clip((shifted-normalizer['mean'])/normalizer['std'],-10,10)
                        z[:,normalizer['constant']] = 0
                        qr = qr if arm == 'past_frame' else np.broadcast_to(np.eye(2),(len(held),2,2))
                        vv = vr if arm != 'original' else np.ones(len(held),bool)
                        rotated_base = np.einsum('ntd,ndk->ntk',a['baselines'][held,cv],r)
                        pp_rotated = infer(model,torch.from_numpy(z.astype(np.float32)),observed[held],
                            torch.from_numpy(rotated_base.astype(np.float32)),torch.from_numpy(qr.astype(np.float32)),torch.from_numpy(vv))
                        restored = np.einsum('ntd,nkd->ntk',pp_rotated,r)
                        gaps.append(restored-pred)
                    gap = np.asarray(gaps)
                    train_pred = infer(model,x[train],observed[train],base[train],rotations[train],supported[train])
                    error = np.linalg.norm(train_pred.astype(float)-a['targets'][train],axis=-1).mean(1)
                    te = np.mean([error[a['folds'][train] == f].mean() for f in train_scenes])
                    tb = np.mean([train_errors[a['folds'][train] == f,cv].mean() for f in train_scenes])
                    events = training_event_labels(a['geometry'][held],a['targets'][held])
                    slices = {}
                    groups = [(n,events == j) for j,n in enumerate(EVENT_NAMES)]
                    groups += [(n,records[held] == n) for n in np.unique(records[held])]
                    groups += [(n,kinds[held] == n) for n in np.unique(kinds[held])]
                    for name,mask in groups:
                        ids = held[mask]
                        slices[name] = dict(forecast_metrics(pred[mask],a['targets'][ids],a['baselines'][ids,cv],a['scale'][ids],parent['easy_threshold']),
                            tracks=len(np.unique(tracks[ids]))) if len(ids) else dict(rows=0)
                    result = dict(identity=trial_identity,trial=key,variant=variant,arm=arm,seed=seed,fold=fold,
                        result_source='cached_verified_exact_control' if is_control else 'fresh_run_native_torch_cached_verified_inputs',
                        fit=fit,train_rows=len(train),held_rows=len(held),training_gain_percent=float(100*(1-te/tb)),slices=slices,
                        vs_CV=forecast_metrics(pred,a['targets'][held],a['baselines'][held,cv],a['scale'][held],parent['easy_threshold']),
                        vs_train_selected_strongest=forecast_metrics(pred,a['targets'][held],a['baselines'][held,strongest],a['scale'][held],parent['easy_threshold']),
                        train_selected_strongest=receipt['baseline_names'][strongest],
                        frame_stress=dict(mean_gap=float(np.linalg.norm(gap,axis=-1).mean()),max_coordinate_gap=float(np.abs(gap).max()),
                                          rows=len(held),quarter_turns=3,labels_used=False),
                        checkpoint_path=str(cp.relative_to(ROOT)),prediction_path=str(pp.relative_to(ROOT)),
                        checkpoint_sha256=file_digest(cp),prediction_sha256=file_digest(pp))
                    json_write(rp,result); trials.append(result)
                    heartbeat(dict(state='held_fit_evaluated',trial=key,gain=result['vs_CV']['improvement_percent']))
    if args.replay:
        json_write(reports/'replay.json',dict(identity=identity,trials=replays,status='exact',new_optimizer_updates=0))
        heartbeat(dict(state='replay_complete',trials=len(replays))); return
    if args.trial:
        return
    if len(trials) != 54:
        raise ValueError('Incomplete matrix')
    report_path = reports/'report.json'
    if report_path.exists() and updates == 0:
        previous = json.loads(report_path.read_text())
        if previous['identity'] != identity or previous['trials'] != trials:
            raise ValueError('Completed report changed')
        heartbeat(dict(state='completed_resume_verified',trials=54,new_optimizer_updates=0)); return
    sums = summarize(trials,reg)
    json_write(report_path,dict(identity=identity,complete=True,fresh_fits=36,cached_verified_controls=18,
        fresh_optimizer_updates=144000,rows=11966,anchor_counts=anchor_counts,trials=trials,summary=sums,
        primary='past_normalized_ADE',aggregation='equal_physical_scene_and_seed',
        observation_mode='offline_annotated_not_strict_sensor_as_of',deployment=False,
        sealed_roles_opened=False,new_source_admitted=False,stage5c_executed=False,smc_enabled=False))
    lines = ['# Past-Only Coordinate Frame Comparison','',
        '36 new fits and 18 exact-replayed original controls. Same model/loss/budget and all11,966 fit rows.',
        'No final-test or policy selection. Quarter-turn gaps measure coordinate consistency, not accuracy.','',
        '| Features / arm | Gain vs CV (%) | Gain vs guard (%) | Safe positive fits | Maximum rotation gap |',
        '| --- | ---: | ---: | ---: | ---: |']
    for key,value in sums.items():
        lines.append(f'| {key} | {value["vs_CV"]["gain_percent"]:.5f} | {value["vs_guard"]["gain_percent"]:.5f} | {value["safe_positive_fits"]}/9 | {value["quarter_turn_max_coordinate_gap"]:.3g} |')
    lines += ['','Inputs are re-expressed vector summaries, not a raw-image rotation audit. No new physical-time/metric/deployment claim.','']
    (reports/'report.md').write_text('\n'.join(lines))
    heartbeat(dict(state='complete',trials=54,fresh_fits=36))


if __name__ == '__main__':
    main()
