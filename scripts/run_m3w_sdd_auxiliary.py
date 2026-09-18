"""Fixed, paired SDD auxiliary training; no development or final-test selection."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Use native arm64 .venv-pytorch')
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '4'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts.run_m3w_objective_alignment import paired_interval
from src.evaluation.m3w_experiment_contract import file_digest
from src.world_model.m3w_offline_visual_data import json_write, verify_registration
from src.world_model.m3w_offline_visual_forecast import OfflineVisualForecast, forecast_metrics
from src.world_model.m3w_objective_alignment import supported_standardization
from src.world_model.m3w_sdd_auxiliary import (
    MODALITIES, SCHEDULES, fit_two_phase, load_registration, predict, source_entries,
)
from src.world_model.m3w_track_event_sampling import EVENT_NAMES, training_event_labels


class Corpus:
    def __init__(self, reg):
        parent, contract = verify_registration(ROOT, ROOT/reg['main_registration'])
        main_dir = ROOT/parent['output']/'inputs'
        mr = json.loads((main_dir/'data_manifest.json').read_text())
        expected = dict(registration_sha256=file_digest(ROOT/reg['main_registration']),
                        parent_protocol_sha256=contract.digest)
        if mr['identity'] != expected or mr['rows'] != 11966:
            raise ValueError('Main cohort identity changed')
        for name, digest in mr['arrays'].items():
            if file_digest(main_dir/name) != digest:
                raise ValueError('Main input changed: '+name)
        self.main = {k:np.load(main_dir/(k+'.npy'), mmap_mode='r', allow_pickle=False)
                     for k in ('geometry','targets','baselines','scale','folds','image_rows','rgb','coverage')}
        self.main_rows = json.loads((main_dir/'rows.json').read_text())
        if not np.array_equal(self.main['folds'], [r['fold'] for r in self.main_rows]):
            raise ValueError('Main row/fold alignment changed')
        self.cv = mr['baseline_names'].index('constant_velocity_causal_fd')
        self.baseline_names = mr['baseline_names']
        self.easy_threshold = parent['easy_threshold']
        self.manifest_path = ROOT/reg['output']/'inputs/manifest.json'
        ar = json.loads(self.manifest_path.read_text())
        if (ar['identity']['registration_sha256'] != file_digest(ROOT/reg['registration_path'])
                or ar['rows'] != 229333 or ar['source_role'] != 'supervised_auxiliary_training'
                or ar['original_val_test_opened']):
            raise ValueError('Full auxiliary identity/role changed')
        roster = {e['annotation_key'] for e in source_entries(ROOT, reg)}
        if {r['recording'] for r in ar['records']} != roster or len(ar['records']) != 40:
            raise ValueError('Auxiliary recording roster differs from train-40')
        parts = {k:[] for k in ('geometry','baseline','target','valid')}
        self.images, self.ends, self.record_ids, self.local_ids = [], [], [], []
        total = 0
        for rid, receipt in enumerate(ar['records']):
            d = self.manifest_path.parent/receipt['recording']
            for name, digest in receipt['arrays'].items():
                if file_digest(d/name) != digest:
                    raise ValueError('Auxiliary input changed: '+str(d/name))
            for k in parts:
                parts[k].append(np.load(d/(k+'.npy'), allow_pickle=False))
            self.images.append({k:np.load(d/(k+'.npy'), mmap_mode='r', allow_pickle=False)
                               for k in ('image_rows','rgb','coverage')})
            self.record_ids.extend([rid]*receipt['rows'])
            self.local_ids.extend(range(receipt['rows']))
            total += receipt['rows']; self.ends.append(total)
        self.aux = {k:np.concatenate(v) for k,v in parts.items()}
        self.record_ids = np.asarray(self.record_ids); self.local_ids = np.asarray(self.local_ids)
        self.identity = dict(main_manifest_sha256=file_digest(main_dir/'data_manifest.json'),
                             auxiliary_manifest_sha256=file_digest(self.manifest_path))
        self.records = np.array([r['recording'] for r in self.main_rows])
        self.tracks = np.array([r['recording']+':'+str(r['agent']) for r in self.main_rows])
        self.auxiliary_rows = total
        self.fold = None

    def set_fold(self, fold):
        self.fold = fold
        self.train = np.flatnonzero(self.main['folds'] != fold)
        self.held = np.flatnonzero(self.main['folds'] == fold)
        self.x, normalizer = supported_standardization(self.main['geometry'][self.train], self.main['geometry'])
        self.normalizer = normalizer
        z = (self.aux['geometry']-normalizer['mean'])/normalizer['std']
        self.aux_clip_fraction = float((np.abs(z[:, ~normalizer['constant']]) > 10).mean())
        self.ax = np.clip(z, -10, 10).astype(np.float32)
        self.ax[:, normalizer['constant']] = 0
        self.allowed = np.zeros(11966, bool); self.allowed[self.train] = True
        errors = np.linalg.norm(self.main['baselines'][self.train].astype(float)-
                               self.main['targets'][self.train, None], axis=-1).mean(-1)
        groups = self.main['folds'][self.train]
        self.strongest = int(np.argmin(np.mean([errors[groups == g].mean(0) for g in np.unique(groups)], axis=0)))
        self.hard_cut = float(np.quantile(errors[:, self.cv], .75))

    def batch(self, ids, modality, *, auxiliary=False, training=False):
        ids = np.asarray(ids, dtype=int)
        source = self.aux if auxiliary else self.main
        count = self.auxiliary_rows if auxiliary else 11966
        if ids.ndim != 1 or (len(ids) and (ids.min() < 0 or ids.max() >= count)):
            raise ValueError('Invalid source row request')
        if training and not auxiliary and not self.allowed[ids].all():
            raise ValueError('Held fit rows requested by trainer')
        if auxiliary:
            cov = np.empty((len(ids), 8, 1, 32, 32), np.float32)
            rgb = np.zeros((len(ids), 8, 3, 32, 32), np.float32) if modality != 'geometry' else None
            records = self.record_ids[ids]
            for record in np.unique(records):
                loc = np.flatnonzero(records == record); store = self.images[int(record)]
                image_rows = store['image_rows'][self.local_ids[ids[loc]]]
                cov[loc] = store['coverage'][image_rows, None].astype(np.float32)/9.
                if modality == 'past_rgb':
                    rgb[loc] = store['rgb'][image_rows].astype(np.float32)/255.
            geometry, baseline, target, valid = self.ax[ids],source['baseline'][ids],source['target'][ids],source['valid'][ids]
        else:
            rows = source['image_rows'][ids]
            cov = source['coverage'][rows, None].astype(np.float32)/9.
            rgb = (source['rgb'][rows].astype(np.float32)/255. if modality == 'past_rgb'
                   else np.zeros((len(ids), 8, 3, 32, 32), np.float32) if modality == 'mask_only' else None)
            geometry, baseline, target = self.x[ids],source['baselines'][ids,self.cv],source['targets'][ids]
            valid = np.ones((len(ids), 12), bool)
        value = dict(geometry=geometry, baseline=baseline, target=target, valid=valid,
                     observed=cov.mean((2,3,4)))
        if modality != 'geometry':
            value.update(rgb=rgb, coverage=cov)
        return {k:torch.from_numpy(np.asarray(v).copy()) for k,v in value.items()}

    def infer(self, model, ids, modality):
        model.eval()
        with torch.no_grad():
            return np.concatenate([predict(model, self.batch(ids[i:i+128],modality),modality).numpy()
                                   for i in range(0,len(ids),128)])


def summarize(trials, reg):
    look = {(t['schedule'],t['modality'],t['seed'],t['fold']):t for t in trials}
    def matrix(s,m,field):
        return np.array([[look[s,m,seed,fold]['vs_CV'][field] for fold in range(3)] for seed in reg['seeds']])
    output = {}
    for schedule in SCHEDULES:
        for modality in MODALITIES:
            ts = [t for t in trials if t['schedule']==schedule and t['modality']==modality]
            model = matrix(schedule,modality,'primary_ADE')
            output[schedule+'_'+modality] = dict(
                vs_CV=paired_interval(model,matrix(schedule,modality,'reference_ADE'),reg['bootstrap_resamples']),
                vs_no_aux_same_modality=paired_interval(model,matrix('no_aux',modality,'primary_ADE'),reg['bootstrap_resamples']),
                vs_same_source_mask=paired_interval(model,matrix(schedule,'mask_only','primary_ADE'),reg['bootstrap_resamples']),
                positive_fits=sum(t['vs_CV']['improvement_percent']>0 for t in ts),
                safe_positive_fits=sum(t['vs_CV']['improvement_percent']>0 and
                    t['vs_CV']['easy_degradation_percent'] is not None and t['vs_CV']['easy_degradation_percent']<=2 for t in ts),
                easy_degradation_percent=[t['vs_CV']['easy_degradation_percent'] for t in ts],
                easy_absolute_harm=[t['vs_CV']['easy_absolute_harm'] for t in ts],
                fit_seconds=sum(t['fit']['fit_seconds'] for t in ts))
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registration',type=Path,required=True); p.add_argument('--trial')
    p.add_argument('--stop-at',type=int); p.add_argument('--replay',action='store_true')
    args = p.parse_args()
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    reg = load_registration(ROOT,args.registration)
    if str(args.registration) != reg['registration_path']:
        raise ValueError('Use the registered relative config path')
    keys = {f'{s}_{m}_seed{seed}_fold{f}' for s in SCHEDULES for m in MODALITIES for seed in reg['seeds'] for f in range(3)}
    if args.trial and args.trial not in keys:
        raise ValueError('Unregistered trial')
    if args.stop_at is not None and (not args.trial or args.replay):
        raise ValueError('Pilot requires one explicit training trial')
    output, reports = ROOT/reg['output'],ROOT/reg['reports']
    def heartbeat(value):
        v = dict(pid=os.getpid(),timestamp_unix=time.time(),**value)
        json_write(output/'training_heartbeat.json',v); print(json.dumps(v),flush=True)
    heartbeat(dict(state='verifying_inputs'))
    data = Corpus(reg)
    identity = dict(registration_sha256=file_digest(args.registration), torch=torch.__version__,
                    numpy=np.__version__, **data.identity)
    if (output/'training_identity.json').exists() and json.loads((output/'training_identity.json').read_text()) != identity:
        raise ValueError('Training identity changed')
    json_write(output/'training_identity.json',identity)
    trials, replays, updates = [],[],0
    for fold in range(3):
        data.set_fold(fold)
        for modality in MODALITIES:
            for seed in reg['seeds']:
                for schedule in SCHEDULES:
                    key = f'{schedule}_{modality}_seed{seed}_fold{fold}'
                    if args.trial and args.trial != key:
                        continue
                    cp, pp, rp = [output/d/(key+ext) for d,ext in
                                  (('checkpoints','.pt'),('predictions','.npz'),('trials','.json'))]
                    trial_identity = dict(identity,schedule=schedule,modality=modality,seed=seed,fold=fold)
                    saved = json.loads(rp.read_text()) if rp.exists() else None
                    if saved:
                        if (saved['identity'] != trial_identity or file_digest(cp) != saved['checkpoint_sha256']
                                or file_digest(pp) != saved['prediction_sha256']):
                            raise ValueError('Completed fit identity changed')
                        if not args.replay:
                            trials.append(saved); continue
                    torch.manual_seed(seed); model = OfflineVisualForecast(476)
                    if args.replay:
                        if not saved:
                            raise ValueError('Replay requires complete trials')
                        state = torch.load(cp,map_location='cpu',weights_only=False)
                        if state['identity'] != trial_identity or state['step'] != 6000:
                            raise ValueError('Replay checkpoint identity mismatch')
                        model.load_state_dict(state['model'])
                    else:
                        def main_batch(ids):
                            return data.batch(data.train[ids],modality,training=True)
                        def aux_batch(ids):
                            return data.batch(ids,modality,auxiliary=True,training=True)
                        fit = fit_two_phase(model,main_batch,aux_batch,len(data.train),data.auxiliary_rows,
                            modality=modality,schedule=schedule,config=reg['training'],seed=seed,
                            identity=trial_identity,checkpoint=cp,
                            heartbeat=lambda v:heartbeat(dict(trial=key,**v)),stop_at=args.stop_at)
                        updates += fit['new_updates_this_invocation']
                        if not fit['complete']:
                            heartbeat(dict(state='pilot_saved_no_held_evaluation',trial=key,step=fit['step'],
                                           fit_seconds=fit['fit_seconds'])); return
                    prediction = data.infer(model,data.held,modality)
                    if args.replay:
                        with np.load(pp,allow_pickle=False) as prior:
                            np.testing.assert_array_equal(prediction,prior['prediction'])
                            np.testing.assert_array_equal(data.held,prior['held_indices'])
                        replays.append(dict(trial=key,prediction_exact=True))
                        heartbeat(dict(state='replay_trial_complete',trial=key)); continue
                    pp.parent.mkdir(parents=True,exist_ok=True)
                    temporary = pp.with_suffix('.tmp.npz')
                    np.savez(temporary,prediction=prediction,held_indices=data.held); os.replace(temporary,pp)
                    main = data.main; ids = data.held
                    metric = lambda pred,ix,ref:forecast_metrics(pred,main['targets'][ix],main['baselines'][ix,ref],
                                                                  main['scale'][ix],data.easy_threshold)
                    events = training_event_labels(main['geometry'][ids],main['targets'][ids])
                    ref_errors = np.linalg.norm(main['baselines'][ids,data.cv].astype(float)-main['targets'][ids],axis=-1).mean(1)
                    groups = [(n,events == j) for j,n in enumerate(EVENT_NAMES)]
                    groups += [(r,data.records[ids] == r) for r in np.unique(data.records[ids])]
                    groups += [('hard_train_q75',ref_errors >= data.hard_cut)]
                    slices = {}
                    for name,mask in groups:
                        if mask.any():
                            slices[name] = dict(metric(prediction[mask],ids[mask],data.cv),
                                                tracks=len(np.unique(data.tracks[ids[mask]])))
                        else:
                            slices[name] = dict(rows=0)
                    fit_pred = data.infer(model,data.train,modality)
                    err = np.linalg.norm(fit_pred.astype(float)-main['targets'][data.train],axis=-1).mean(1)
                    ref = np.linalg.norm(main['baselines'][data.train,data.cv].astype(float)-main['targets'][data.train],axis=-1).mean(1)
                    folds = main['folds'][data.train]
                    train_gain = 100*(1-np.mean([err[folds==f].mean() for f in np.unique(folds)])/
                                      np.mean([ref[folds==f].mean() for f in np.unique(folds)]))
                    result = dict(identity=trial_identity,trial=key,schedule=schedule,modality=modality,
                        seed=seed,fold=fold,fit=fit,train_rows=len(data.train),held_rows=len(ids),
                        result_source='fresh_run_torch_with_hash_verified_inputs',slices=slices,
                        train_equal_scene_gain_percent=float(train_gain),auxiliary_clipped_feature_fraction=data.aux_clip_fraction,
                        constant_feature_columns=np.flatnonzero(data.normalizer['constant']).tolist(),
                        train_selected_strongest=data.baseline_names[data.strongest],
                        vs_CV=metric(prediction,ids,data.cv),
                        vs_train_selected_strongest=metric(prediction,ids,data.strongest),
                        checkpoint_sha256=file_digest(cp),prediction_sha256=file_digest(pp),
                        checkpoint_path=str(cp.relative_to(ROOT)),prediction_path=str(pp.relative_to(ROOT)))
                    json_write(rp,result); trials.append(result)
                    heartbeat(dict(state='held_fit_evaluated',trial=key,gain=result['vs_CV']['improvement_percent'],
                                   easy=result['vs_CV']['easy_degradation_percent']))
    if args.replay:
        json_write(reports/'replay.json',dict(identity=identity,trials=replays,new_updates=0,all_exact=True))
        heartbeat(dict(state='replay_complete',trials=len(replays))); return
    if args.trial:
        return
    if len(trials) != 54:
        raise ValueError('Incomplete fixed comparison')
    path = reports/'report.json'
    if path.exists() and updates == 0:
        old = json.loads(path.read_text())
        if old['identity'] != identity or old['trials'] != trials:
            raise ValueError('Completed report changed')
        heartbeat(dict(state='completed_resume_verified',trials=54,new_optimizer_updates=0)); return
    summary = summarize(trials,reg)
    result = dict(identity=identity,trials=trials,summary=summary,complete=True,
        total_fits=54,registered_updates=324000,main_rows=11966,auxiliary_rows=data.auxiliary_rows,
        primary='past_normalized_ADE_equal_physical_scene',independent_confirmation=False,
        original_SDD_val_test_opened=False,sealed_main_roles_opened=False,
        new_deployment=False,stage5c_executed=False,smc_enabled=False)
    json_write(path,result)
    lines = ['# Matched SDD Auxiliary Training','',
        '54 real fits; three seeds, three exposed fit scenes. No independent confirmation or deployment.','',
        '| Source / modality | Gain vs CV (%) | Gain vs no-aux same input (%) | Gain vs same-source mask (%) | Safe positive fits |',
        '| --- | ---: | ---: | ---: | ---: |']
    for key,s in summary.items():
        lines.append(f"| {key} | {s['vs_CV']['gain_percent']:.5f} | {s['vs_no_aux_same_modality']['gain_percent']:.5f} | {s['vs_same_source_mask']['gain_percent']:.5f} | {s['safe_positive_fits']}/9 |")
    lines += ['','All outputs and negative fits retained. Bootstrap is conditional on three previously exposed physical sites.',
              'Partial auxiliary labels never change the complete-label main primary. No metric/seconds claim.','']
    (reports/'report.md').write_text('\n'.join(lines))
    heartbeat(dict(state='complete',trials=54,new_optimizer_updates=updates))


if __name__ == '__main__':
    main()

