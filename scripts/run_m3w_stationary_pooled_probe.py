"""One-factor adaptive fit-only repair: pooled rather than ordered context."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if platform.system() == 'Darwin' and platform.machine() != 'arm64':
        raise SystemExit('Use arm64 runtime')
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[name] = '4'
    import numpy as np
    import sklearn
    import joblib
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import ExtraTreesClassifier
    from src.evaluation.m3w_experiment_contract import ExperimentContract, file_digest
    from src.evaluation.m3w_stationary_start_probe import score_probabilities
    from src.evaluation.m3w_stationary_pooled_context import pool_context, NAMES, GEOMETRY_COLUMNS
    from scripts.run_m3w_stationary_start_probe import atomic_json, digest
    registration = json.loads(args.registration.read_text())
    for path, sha in registration['bindings'].items():
        if file_digest(ROOT/path) != sha:
            raise ValueError('Pooled repair registration binding changed')
    parent_reg = json.loads((ROOT/registration['source_registration']).read_text())
    contract = ExperimentContract(json.loads((ROOT/parent_reg['parent_protocol']).read_text()), ROOT)
    source_identity = json.loads((args.source/'run_identity.json').read_text())
    source_receipt = json.loads((args.source/'fit_rows.receipt.json').read_text())
    cache = args.source/'fit_rows.npz'
    if source_identity['parent_protocol_sha256'] != contract.digest or source_receipt['identity_sha256'] != digest(source_identity) or source_receipt['cache_sha256'] != file_digest(cache):
        raise ValueError('Source fit-only data identity changed')
    if source_identity['registration_sha256'] != file_digest(ROOT/registration['source_registration']):
        raise ValueError('Source registration mismatch')
    with np.load(cache, allow_pickle=False) as values:
        x, y, rows = pool_context(values['features']), values['targets'], json.loads(str(values['rows_json']))
    if any(contract.protocol['assignments'][r['recording_id']] != 'fit' or r['data_role'] != 'fit' for r in rows):
        raise ValueError('Only original fit rows may enter this adaptive probe')
    identity = {'registration_sha256': file_digest(args.registration), 'source_identity': source_identity,
                'source_cache_sha256': file_digest(cache), 'sklearn_version': sklearn.__version__}
    output = args.output.resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError('Workspace-local outputs only')
    if args.resume:
        if json.loads((output/'run_identity.json').read_text()) != identity:
            raise ValueError('Pooled fit identity changed')
    else:
        output.mkdir(parents=True, exist_ok=False)
        atomic_json(output/'run_identity.json', identity)
    began, trials = time.monotonic(), []
    folds = np.array([r['fit_fold'] for r in rows])
    for fold in sorted(set(contract.protocol['fit_folds'].values())):
        train, held = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        if not len(held) or not len(train):
            trials.append({'fold': fold, 'status': 'not_run_no_stationary_held_or_train_rows'})
            continue
        train_scenes, held_scenes = {rows[i]['physical_scene'] for i in train}, {rows[i]['physical_scene'] for i in held}
        if train_scenes & held_scenes:
            raise ValueError('Physical-scene overlap')
        prior = float((y[train].sum()+1)/(len(train)+2))
        for seed in contract.protocol['seeds']:
            for feature_name, columns in [('geometry', GEOMETRY_COLUMNS), ('geometry_motion', list(range(x.shape[1])))]:
                for model_name, settings in parent_reg['models'].items():
                    name = f'fold{fold}_seed{seed}_{feature_name}_{model_name}'
                    report_path, model_path = output/(name+'.json'), output/(name+'.joblib')
                    if report_path.exists():
                        report = json.loads(report_path.read_text())
                        if report['run_identity_sha256'] != digest(identity) or file_digest(model_path) != report['model_sha256']:
                            raise ValueError('Saved pooled probe changed')
                        trials.append({**report, 'status': 'cached_verified'})
                        continue
                    atomic_json(output/'heartbeat.json', {'pid': os.getpid(), 'state': 'training', 'trial': name})
                    if len(np.unique(y[train])) < 2:
                        raise ValueError('Unexpected single-class repair train split; no unregistered fallback')
                    model = (make_pipeline(StandardScaler(), LogisticRegression(**settings, random_state=seed))
                             if model_name == 'logistic' else ExtraTreesClassifier(**settings, random_state=seed, n_jobs=4))
                    model.fit(x[train][:, columns], y[train])
                    joblib.dump(model, model_path)
                    p = model.predict_proba(x[held][:, columns])[:, 1]
                    report = {'run_identity_sha256': digest(identity), 'model_sha256': file_digest(model_path),
                        'fold': fold, 'seed': seed, 'model': model_name, 'features': feature_name, 'feature_dimension': len(columns),
                        'status': 'fresh_run', 'train_physical_scenes': sorted(train_scenes), 'held_physical_scenes': sorted(held_scenes),
                        **score_probabilities(y[held], p, prior=prior)}
                    atomic_json(report_path, report)
                    atomic_json(output/(name+'.predictions.json'), {'held_row_indices': held.tolist(), 'probability': p.tolist()})
                    trials.append(report)
    result = {'identity': identity, 'result_source': 'fresh_run_adaptive_fit_only_feature_pooling_repair',
        'feature_names': NAMES, 'trials': trials, 'change': 'ordered_context_to_pooled_context_only',
        'original_models_hyperparameters_labels_folds_unchanged': True,
        'development_labels_opened': False, 'adaptive_followup_not_independent_confirmation': True,
        'new_deployment': False, 'stage5c_executed': False, 'smc_enabled': False}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.report, result)
    atomic_json(output/'completion.json', {'identity_sha256': digest(identity), 'report_sha256': file_digest(args.report)})
    atomic_json(output/'heartbeat.json', {'pid': os.getpid(), 'state': 'complete', 'elapsed_seconds': time.monotonic()-began})
    print(json.dumps({'status': 'complete', 'trials': len(trials), 'feature_dimension': x.shape[1]}))


if __name__ == '__main__':
    main()
