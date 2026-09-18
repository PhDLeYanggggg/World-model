"""Registered exposure/source-supervision controls; all fit-only outcomes retained."""
from __future__ import annotations

import argparse
import csv
import io
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

from scripts.run_m3w_sdd_auxiliary import Corpus
from scripts.run_m3w_objective_alignment import paired_interval
from src.evaluation.m3w_experiment_contract import file_digest
from src.world_model.m3w_auxiliary_mechanism import (
    NEW_ARMS, ALL_ARMS, load_mechanism_registration, arm_config, residual_donors,
    permuted_batch, donor_evidence,
)
from src.world_model.m3w_offline_visual_data import json_write
from src.world_model.m3w_offline_visual_forecast import OfflineVisualForecast, forecast_metrics
from src.world_model.m3w_sdd_auxiliary import fit_two_phase
from src.world_model.m3w_track_event_sampling import EVENT_NAMES, training_event_labels


def verify_controls(path, data):
    report = json.loads(path.read_text())
    if not report['complete'] or len(report['trials']) != 54:
        raise ValueError('All prior paired controls required')
    for key, value in data.identity.items():
        if report['identity'][key] != value:
            raise ValueError('Parent input alignment changed')
    controls = []
    for trial in report['trials']:
        for kind in ('checkpoint', 'prediction'):
            if file_digest(ROOT/trial[kind+'_path']) != trial[kind+'_sha256']:
                raise ValueError('Prior control artifact changed')
        fold = trial['fold']
        with np.load(ROOT/trial['prediction_path'], allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved['held_indices'], np.flatnonzero(data.main['folds']==fold))
        controls.append(dict(trial, arm=trial['schedule'],
            parent_result_source=trial['result_source'],
            result_source='cached_verified_parent_training_predictions'))
    return controls


def evaluate(model, data, modality, prediction):
    main, ids = data.main, data.held
    def metric(pred, ix, ref):
        return forecast_metrics(pred, main['targets'][ix], main['baselines'][ix,ref],
                                main['scale'][ix], data.easy_threshold)
    events = training_event_labels(main['geometry'][ids], main['targets'][ids])
    reference_error = np.linalg.norm(main['baselines'][ids,data.cv].astype(float)-main['targets'][ids],axis=-1).mean(1)
    groups = [(n,events==j) for j,n in enumerate(EVENT_NAMES)]
    groups += [(r,data.records[ids]==r) for r in np.unique(data.records[ids])]
    groups += [('hard_train_q75',reference_error>=data.hard_cut)]
    slices = {}
    for name, mask in groups:
        slices[name] = (dict(metric(prediction[mask],ids[mask],data.cv),
                            tracks=len(np.unique(data.tracks[ids[mask]]))) if mask.any() else dict(rows=0))
    fit_pred = data.infer(model,data.train,modality)
    err = np.linalg.norm(fit_pred.astype(float)-main['targets'][data.train],axis=-1).mean(1)
    ref = np.linalg.norm(main['baselines'][data.train,data.cv].astype(float)-main['targets'][data.train],axis=-1).mean(1)
    folds = main['folds'][data.train]
    train_gain = 100*(1-np.mean([err[folds==f].mean() for f in np.unique(folds)])/
                        np.mean([ref[folds==f].mean() for f in np.unique(folds)]))
    return dict(vs_CV=metric(prediction,ids,data.cv),
        vs_train_selected_strongest=metric(prediction,ids,data.strongest),
        train_selected_strongest=data.baseline_names[data.strongest], slices=slices,
        train_equal_scene_gain_percent=float(train_gain), train_rows=len(data.train), held_rows=len(ids))


def summaries(trials, reg):
    look = {(t['arm'],t['modality'],t['seed'],t['fold']):t for t in trials}
    def matrix(arm, modality, field='primary_ADE'):
        return np.array([[look[arm,modality,s,f]['vs_CV'][field] for f in range(3)] for s in reg['seeds']])
    groups = {}
    for arm in ALL_ARMS:
        for modality in reg['modalities']:
            ts = [t for t in trials if t['arm']==arm and t['modality']==modality]
            groups[arm+'_'+modality] = dict(
                vs_CV=paired_interval(matrix(arm,modality),matrix(arm,modality,'reference_ADE'),2000),
                vs_main4k=paired_interval(matrix(arm,modality),matrix('main4k',modality),2000),
                vs_permuted_source=paired_interval(matrix(arm,modality),matrix('sdd_permuted',modality),2000),
                vs_same_schedule_mask=paired_interval(matrix(arm,modality),matrix(arm,'mask_only'),2000),
                positive_fits=sum(t['vs_CV']['improvement_percent']>0 for t in ts),
                safe_positive_fits=sum(t['vs_CV']['improvement_percent']>0 and
                    t['vs_CV']['easy_degradation_percent'] is not None and t['vs_CV']['easy_degradation_percent']<=2 for t in ts),
                easy_degradation_percent=[t['vs_CV']['easy_degradation_percent'] for t in ts],
                easy_absolute_harm=[t['vs_CV']['easy_absolute_harm'] for t in ts],
                train_gain_percent=[t['train_equal_scene_gain_percent'] for t in ts],
                fit_seconds=sum(t['fit']['fit_seconds'] for t in ts))
    return groups


def write_report(report, reports):
    json_write(reports/'report.json',report)
    lines = ['# Auxiliary Source Mechanism: Complete Fixed Comparison','',
        'Result source: 54 fresh Torch fits, 54 cached_verified controls. No independent confirmation.',
        'Main cohort: all 11,966 fit windows; primary: past-normalized ADE, equal physical scene.',
        'No held-score model selection, no sealed-role access and no deployment.','',
        '| Schedule / input | Gain vs CV (%) | Gain vs main4k (%) | Gain vs shuffled source (%) | Pixel gain vs mask (%) | Safe positive fits |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for key, s in report['summary'].items():
        lines.append(f"| {key} | {s['vs_CV']['gain_percent']:.5f} | {s['vs_main4k']['gain_percent']:.5f} | {s['vs_permuted_source']['gain_percent']:.5f} | {s['vs_same_schedule_mask']['gain_percent']:.5f} | {s['safe_positive_fits']}/9 |")
    lines += ['','All seed/site failures and absolute easy harm are retained in report.json and fit_metrics.csv.',
        'The 2,000-draw site intervals resample only three already exposed physical sites.',
        'main4k matches main exposure, not total updates; source permutation matches source draws and total updates.',
        'The label ablation preserves recording/support strata, not every dependency or semantic property.',
        'No metric, seconds, true-3D, foundation, Stage5C or SMC claim.','']
    (reports/'conclusions.md').write_text('\n'.join(lines))
    stream = io.StringIO()
    fields = ['trial','result_source','arm','modality','seed','fold','gain_vs_CV','ADE','FDE',
              'easy_degradation','easy_absolute_harm','train_gain','fit_seconds']
    writer = csv.DictWriter(stream,fieldnames=fields,lineterminator='\n'); writer.writeheader()
    for t in report['trials']:
        m=t['vs_CV']
        writer.writerow(dict(trial=t['trial'],result_source=t['result_source'],arm=t['arm'],
            modality=t['modality'],seed=t['seed'],fold=t['fold'],gain_vs_CV=m['improvement_percent'],
            ADE=m['primary_ADE'],FDE=m['FDE'],easy_degradation=m['easy_degradation_percent'],
            easy_absolute_harm=m['easy_absolute_harm'],train_gain=t['train_equal_scene_gain_percent'],fit_seconds=t['fit']['fit_seconds']))
    (reports/'fit_metrics.csv').write_text(stream.getvalue())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration',type=Path,required=True)
    parser.add_argument('--trial'); parser.add_argument('--stop-at',type=int)
    parser.add_argument('--replay',action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    reg,parent = load_mechanism_registration(ROOT,args.registration)
    if str(args.registration) != reg['registration_path']:
        raise ValueError('Use registered relative path')
    keys = {f'{a}_{m}_seed{s}_fold{f}' for a in NEW_ARMS for m in reg['modalities'] for s in reg['seeds'] for f in range(3)}
    if args.trial and args.trial not in keys:
        raise ValueError('Unregistered trial')
    if args.stop_at is not None and (not args.trial or args.replay):
        raise ValueError('Pilot requires an explicit fresh trial')
    output,reports = ROOT/reg['output'],ROOT/reg['reports']
    def heartbeat(value):
        event = dict(pid=os.getpid(),timestamp_unix=time.time(),**value)
        json_write(output/'training_heartbeat.json',event); print(json.dumps(event),flush=True)
    heartbeat(dict(state='verifying_inputs_and_controls'))
    data = Corpus(parent)
    controls = verify_controls(ROOT/reg['control_report'],data)
    identity = dict(registration_sha256=file_digest(args.registration),torch=torch.__version__,
                    numpy=np.__version__,**data.identity)
    ip = output/'training_identity.json'
    if ip.exists() and json.loads(ip.read_text()) != identity:
        raise ValueError('Runtime or input identity changed')
    json_write(ip,identity)
    manifest = json.loads(data.manifest_path.read_text())
    tracks = np.concatenate([np.load(data.manifest_path.parent/r['recording']/'query_keys.npy')[:,1]
                             for r in manifest['records']])
    donor_maps, checks = {}, {}
    for seed in reg['seeds']:
        donor = residual_donors(data.record_ids,data.aux['valid'],seed)
        donor_maps[seed] = donor
        checks[str(seed)] = donor_evidence(donor,data.record_ids,data.aux['valid'],tracks)
        path = output/'donors'/f'seed{seed}.npy'
        path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():
            np.testing.assert_array_equal(np.load(path,allow_pickle=False),donor)
        else:
            temporary = path.with_suffix('.tmp.npy'); np.save(temporary,donor); os.replace(temporary,path)
    json_write(reports/'input_checks.json',dict(identity=identity,main_rows=11966,source_rows=data.auxiliary_rows,
        source_records=40,cached_verified_controls=54,donors=checks,
        main_primary_changed=False,sealed_roles_opened=False,source_val_test_opened=False))
    trials,replays,updates = [],[],0
    for fold in range(3):
        data.set_fold(fold)
        for modality in reg['modalities']:
            for seed in reg['seeds']:
                for arm in NEW_ARMS:
                    key = f'{arm}_{modality}_seed{seed}_fold{fold}'
                    if args.trial and args.trial != key:
                        continue
                    cp,pp,rp = [output/d/(key+ext) for d,ext in
                                (('checkpoints','.pt'),('predictions','.npz'),('trials','.json'))]
                    config,schedule = arm_config(arm,parent)
                    total = config['pretraining_updates']+config['main_updates']
                    ti = dict(identity,arm=arm,modality=modality,seed=seed,fold=fold,
                              donor_sha256=checks[str(seed)]['sha256'] if arm=='sdd_permuted' else None)
                    saved = json.loads(rp.read_text()) if rp.exists() else None
                    if saved:
                        if (saved['identity'] != ti or file_digest(cp) != saved['checkpoint_sha256']
                                or file_digest(pp) != saved['prediction_sha256']):
                            raise ValueError('Completed artifact changed')
                        if not args.replay:
                            trials.append(saved); continue
                    torch.manual_seed(seed); model = OfflineVisualForecast(476)
                    if args.replay:
                        if not saved:
                            raise ValueError('Replay requires complete trial')
                        state = torch.load(cp,map_location='cpu',weights_only=False)
                        if state['identity'] != ti or state['step'] != total:
                            raise ValueError('Replay checkpoint identity mismatch')
                        model.load_state_dict(state['model'])
                    else:
                        def main_batch(ids):
                            return data.batch(data.train[ids],modality,training=True)
                        def aux_batch(ids):
                            batch = data.batch(ids,modality,auxiliary=True,training=True)
                            return permuted_batch(batch,ids,donor_maps[seed],data.aux)
                        fit = fit_two_phase(model,main_batch,aux_batch,len(data.train),data.auxiliary_rows,
                            modality=modality,schedule=schedule,config=config,seed=seed,identity=ti,
                            checkpoint=cp,heartbeat=lambda v:heartbeat(dict(trial=key,**v)),stop_at=args.stop_at)
                        updates += fit['new_updates_this_invocation']
                        if not fit['complete']:
                            heartbeat(dict(state='pilot_saved_no_held_evaluation',trial=key,step=fit['step'],
                                           fit_seconds=fit['fit_seconds'])); return
                    prediction = data.infer(model,data.held,modality)
                    if args.replay:
                        with np.load(pp,allow_pickle=False) as old:
                            np.testing.assert_array_equal(prediction,old['prediction'])
                            np.testing.assert_array_equal(data.held,old['held_indices'])
                        replays.append(dict(trial=key,prediction_exact=True))
                        heartbeat(dict(state='replay_trial_complete',trial=key)); continue
                    pp.parent.mkdir(parents=True,exist_ok=True)
                    temporary = pp.with_suffix('.tmp.npz')
                    np.savez(temporary,prediction=prediction,held_indices=data.held); os.replace(temporary,pp)
                    result = dict(identity=ti,trial=key,arm=arm,modality=modality,seed=seed,fold=fold,
                        result_source='fresh_run_torch',fit=fit,
                        auxiliary_clipped_feature_fraction=data.aux_clip_fraction,
                        **evaluate(model,data,modality,prediction),
                        checkpoint_path=str(cp.relative_to(ROOT)),checkpoint_sha256=file_digest(cp),
                        prediction_path=str(pp.relative_to(ROOT)),prediction_sha256=file_digest(pp))
                    json_write(rp,result); trials.append(result)
                    heartbeat(dict(state='held_fit_evaluated',trial=key,gain=result['vs_CV']['improvement_percent'],
                                   easy=result['vs_CV']['easy_degradation_percent']))
    if args.replay:
        json_write(reports/'replay.json',dict(identity=identity,trials=replays,new_updates=0,all_exact=True))
        heartbeat(dict(state='replay_complete',trials=len(replays))); return
    if args.trial:
        return
    if len(trials) != 54:
        raise ValueError('Incomplete fixed matrix')
    path = reports/'report.json'
    if path.exists() and updates == 0:
        old = json.loads(path.read_text())
        if old['identity'] != identity or old['trials'] != controls+trials:
            raise ValueError('Completed report changed')
        heartbeat(dict(state='completed_resume_verified',trials=54,new_optimizer_updates=0)); return
    result = dict(identity=identity,complete=True,trials=controls+trials,
        summary=summaries(controls+trials,reg),new_fits=54,cached_verified_fits=54,
        new_registered_updates=270000,actual_new_fit_steps=sum(t['fit']['step'] for t in trials),
        summed_new_fit_seconds=sum(t['fit']['fit_seconds'] for t in trials),
        unsupported_batches=sum(t['fit']['unsupported_batches'] for t in trials),
        primary='past_normalized_ADE_equal_physical_scene',independent_confirmation=False,
        sealed_roles_opened=False,new_deployment=False,stage5c_executed=False,smc_enabled=False)
    write_report(result,reports)
    heartbeat(dict(state='complete',trials=54,cached_controls=54,new_optimizer_updates=updates))


if __name__ == '__main__':
    main()
