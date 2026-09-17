"""Fixed proper-score probe of observed motion; no forecasting/policy selection."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time
import warnings

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 required')
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '4'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
import sklearn
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.build_m3w_observed_motion import load_registration
from src.evaluation.m3w_experiment_contract import file_digest
from src.evaluation.m3w_motion_start_probe import (
    ARMS, probe_features, join_rows, group_brier, paired_agent_interval,
)
from src.evaluation.m3w_stationary_start_probe import score_probabilities
from src.world_model.m3w_offline_visual_data import json_write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--replay', action='store_true')
    args = parser.parse_args()
    reg = json.loads(args.registration.read_text())
    if reg['role'] != 'fit_only_information_probe' or reg['arms'] != list(ARMS):
        raise ValueError('Registered fit-only information probe required')
    for name, sha in reg['bindings'].items():
        if file_digest(ROOT/name) != sha:
            raise ValueError('Bound artifact changed: '+name)
    old, parent, contract, source, manifest = load_registration(ROOT/reg['motion_registration'])
    source_probe = ROOT/reg['stationary_cache']
    old_identity = json.loads((source_probe/'run_identity.json').read_text())
    if (old_identity['parent_protocol_sha256'] != contract.digest or
            file_digest(ROOT/old_identity['registration_path']) != old_identity['registration_sha256']):
        raise ValueError('Stationary protocol lineage changed')
    receipt = json.loads((source_probe/'fit_rows.receipt.json').read_text())
    if receipt['cache_sha256'] != file_digest(source_probe/'fit_rows.npz'):
        raise ValueError('Stationary rows changed')
    with np.load(source_probe/'fit_rows.npz', allow_pickle=False) as z:
        neighbors, labels = z['features'].copy(), z['targets'].copy()
        rows = json.loads(str(z['rows_json']))
    motion_path = ROOT/old['output']/'inputs'
    motion_manifest = json.loads((motion_path/'manifest.json').read_text())
    if motion_manifest['identity'] != dict(registration_sha256=file_digest(ROOT/reg['motion_registration']),
            parent_protocol_sha256=contract.digest, source_manifest_sha256=file_digest(source/'data_manifest.json')):
        raise ValueError('Motion source lineage changed')
    for name, sha in motion_manifest['arrays'].items():
        if file_digest(motion_path/name) != sha:
            raise ValueError('Motion feature changed')
    all_rows = json.loads((source/'rows.json').read_text())
    geometry = np.load(source/'geometry.npy', mmap_mode='r')
    targets = np.load(source/'targets.npy', mmap_mode='r')
    ids = join_rows(rows, all_rows, geometry, targets, labels)
    if len(ids) != 365 or manifest['rows'] != 11966:
        raise ValueError('Registered cohort changed')
    motion = np.load(motion_path/'motion.npy', mmap_mode='r')[ids]
    quality = np.load(motion_path/'quality.npy', mmap_mode='r')[ids]
    features = {arm: probe_features(neighbors, motion, quality, arm) for arm in ARMS}
    folds = np.array([r['fit_fold'] for r in rows])
    agents = np.array([f"{r['recording_id']}:{r['agent_id']}" for r in rows])
    runs = np.array([f"{r['recording_id']}:{r['agent_id']}:{r['first_row']}" for r in rows])
    identity = dict(registration_sha256=file_digest(args.registration), numpy=np.__version__,
        sklearn=sklearn.__version__, source_manifest_sha256=file_digest(source/'data_manifest.json'),
        motion_manifest_sha256=file_digest(motion_path/'manifest.json'),
        stationary_cache_sha256=receipt['cache_sha256'])
    output, reports = ROOT/reg['output'], ROOT/reg['reports']
    if not output.is_relative_to(ROOT/'data/stage_cvpr2027_experiments'):
        raise ValueError('Private outputs required')
    if (output/'identity.json').exists() and json.loads((output/'identity.json').read_text()) != identity:
        raise ValueError('Resume identity changed')
    json_write(output/'identity.json', identity)
    predictions, trials, replays, new_fits = {}, [], [], 0
    begin = time.monotonic()
    for fold in (0, 1):
        train, held = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        if set(agents[train]) & set(agents[held]):
            raise ValueError('Agent exposure crosses folds')
        prior = float((labels[train].sum()+1)/(len(train)+2))
        for arm in ARMS:
            for model_name, settings in reg['models'].items():
                for seed in reg['seeds']:
                    key = f'{arm}_{model_name}_seed{seed}_fold{fold}'
                    cp, pp, rp = [output/d/(key+ext) for d, ext in
                                   [('models', '.joblib'), ('predictions', '.npz'), ('trials', '.json')]]
                    trial_id = dict(run=identity, fold=fold, arm=arm, model=model_name, seed=seed)
                    saved = json.loads(rp.read_text()) if rp.exists() else None
                    if saved:
                        if (saved['identity'] != trial_id or file_digest(cp) != saved['model_sha256']
                                or file_digest(pp) != saved['prediction_sha256']):
                            raise ValueError('Saved trial changed')
                        with np.load(pp) as a:
                            if not np.array_equal(a['held_indices'], ids[held]):
                                raise ValueError('Prediction row alignment changed')
                            probability = a['probability'].copy()
                        if args.replay:
                            model = joblib.load(cp)
                            fresh = model.predict_proba(features[arm][held])[:, 1]
                            if not np.array_equal(fresh, probability):
                                raise ValueError('Checkpoint probability replay differs')
                            replays.append(dict(trial=key, exact=True))
                        trials.append(saved)
                        predictions[fold, arm, model_name, seed] = probability
                        continue
                    if args.replay:
                        raise ValueError('Cannot replay missing fit')
                    json_write(output/'heartbeat.json', dict(pid=os.getpid(), trial=key,
                        state='fitting', elapsed_seconds=time.monotonic()-begin))
                    model = (make_pipeline(StandardScaler(), LogisticRegression(**settings, random_state=seed))
                             if model_name == 'logistic' else
                             ExtraTreesClassifier(**settings, random_state=seed, n_jobs=1))
                    started = time.monotonic()
                    with warnings.catch_warnings(record=True) as notices:
                        warnings.simplefilter('always')
                        model.fit(features[arm][train], labels[train])
                    seconds = time.monotonic()-started
                    probability = model.predict_proba(features[arm][held])[:, 1]
                    train_probability = model.predict_proba(features[arm][train])[:, 1]
                    cp.parent.mkdir(parents=True, exist_ok=True); pp.parent.mkdir(parents=True, exist_ok=True)
                    tmp = cp.with_suffix('.tmp'); joblib.dump(model, tmp); os.replace(tmp, cp)
                    tmp = pp.with_suffix('.tmp.npz')
                    np.savez(tmp, held_indices=ids[held], probability=probability)
                    os.replace(tmp, pp)
                    reference = np.full(len(held), prior)
                    trial = dict(identity=trial_id, trial=key, result_source='fresh_run_classifier_fit',
                        arm=arm, model=model_name, seed=seed, fold=fold, feature_width=features[arm].shape[1],
                        fit_seconds=seconds, train_rows=len(train), held_rows=len(held),
                        train_agents=len(np.unique(agents[train])), held_agents=len(np.unique(agents[held])),
                        warnings=[str(w.message) for w in notices],
                        train=score_probabilities(labels[train], train_probability, prior=prior),
                        held=score_probabilities(labels[held], probability, prior=prior),
                        agent_balanced=group_brier(labels[held], probability, reference, agents[held]),
                        run_balanced=group_brier(labels[held], probability, reference, runs[held]),
                        model_path=str(cp.relative_to(ROOT)), model_sha256=file_digest(cp),
                        prediction_path=str(pp.relative_to(ROOT)), prediction_sha256=file_digest(pp))
                    json_write(rp, trial); trials.append(trial); new_fits += 1
                    predictions[fold, arm, model_name, seed] = probability
                    print(json.dumps(dict(trial=key, brier_lift=trial['held']['brier_lift_over_prior'],
                                          fit_seconds=seconds)), flush=True)
    summary = []
    for fold in (0, 1):
        held = np.flatnonzero(folds == fold)
        for arm in ARMS:
            for model in reg['models']:
                selected = [t for t in trials if (t['fold'], t['arm'], t['model']) == (fold, arm, model)]
                p = np.stack([predictions[fold, arm, model, s] for s in reg['seeds']])
                q = np.stack([predictions[fold, 'quality', model, s] for s in reg['seeds']])
                pri = np.full_like(p, selected[0]['held']['train_only_prior'])
                summary.append(dict(fold=fold, arm=arm, model=model,
                    held_brier_lift_mean=float(np.mean([t['held']['brier_lift_over_prior'] for t in selected])),
                    train_brier_lift_mean=float(np.mean([t['train']['brier_lift_over_prior'] for t in selected])),
                    auc_mean=float(np.mean([t['held']['auroc'] for t in selected])),
                    positive_brier_seeds=sum(t['held']['brier_lift_over_prior'] > 0 for t in selected),
                    window_lift_over_quality=float(np.mean((q-labels[held])**2-(p-labels[held])**2)),
                    agent_paired_vs_prior=paired_agent_interval(labels[held], p, pri, agents[held]),
                    agent_paired_vs_quality=paired_agent_interval(labels[held], p, q, agents[held])))
    report = dict(identity=identity, result_source='fresh_run_information_probes_cached_verified_inputs',
        classifiers=48, torch_training=False, rows=len(ids), agents=len(np.unique(agents)),
        runs=len(np.unique(runs)), summary=summary, trials=trials,
        unsupported_fold=dict(fold=2, status='not_run_no_exact_static_histories'),
        independent_confirmation=False, deployment=False, forecasting_primary_changed=False,
        sealed_roles_opened=False, stage5c_executed=False, smc_enabled=False)
    if (reports/'report.json').exists():
        if json.loads((reports/'report.json').read_text()) != report:
            raise ValueError('Resume changed completed report')
    else:
        json_write(reports/'report.json', report)
    if args.replay:
        json_write(reports/'replay.json', dict(identity=identity, models=len(replays),
            trials=replays, report_sha256=file_digest(reports/'report.json')))
    json_write(output/'heartbeat.json', dict(pid=os.getpid(), state='complete', new_fits=new_fits,
        replayed=len(replays), elapsed_seconds=time.monotonic()-begin))
    print(json.dumps(dict(complete=True, new_fits=new_fits, replayed=len(replays), summary=summary)), flush=True)


if __name__ == '__main__':
    main()
