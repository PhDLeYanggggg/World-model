"""Source-verified, fit-fold-only stationary-start audit and fixed context probes."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report-dir', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if platform.system() == 'Darwin' and platform.machine() != 'arm64':
        raise SystemExit('Use arm64 environment')
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
    from src.data_unification.m3w_causal_recordings import clean_points
    from src.evaluation.m3w_recording_lineage import read_track
    from src.evaluation.m3w_stationary_start_probe import (
        past_features, stationary_label, stationary_run, score_probabilities, GEOMETRY_COLUMNS, FEATURE_NAMES,
    )
    registration_path, output = args.registration.resolve(), args.output.resolve()
    if not output.is_relative_to(ROOT) or not registration_path.is_relative_to(ROOT):
        raise ValueError('Workspace-local outputs and registration required')
    registration = json.loads(registration_path.read_text())
    contract = ExperimentContract(json.loads((ROOT/registration['parent_protocol']).read_text()), ROOT)
    if registration['parent_protocol_sha256'] != contract.digest or registration['role'] != 'fit_only_diagnostic':
        raise ValueError('Parent/role mismatch')
    for path, sha in registration['bindings'].items():
        if file_digest(ROOT/path) != sha:
            raise ValueError('Frozen diagnostic source changed: ' + path)
    identity = {'registration_path': str(registration_path.relative_to(ROOT)),
                'registration_sha256': file_digest(registration_path), 'parent_protocol_sha256': contract.digest,
                'sklearn_version': sklearn.__version__, 'numpy_version': np.__version__, 'threads': 4}
    if args.resume:
        if json.loads((output/'run_identity.json').read_text()) != identity:
            raise ValueError('Resume identity changed')
    else:
        output.mkdir(parents=True, exist_ok=False)
        atomic_json(output/'run_identity.json', identity)
    began = time.monotonic()

    def heartbeat(state, **fields):
        atomic_json(output/'heartbeat.json', {'pid': os.getpid(), 'state': state,
                    'elapsed_seconds': time.monotonic()-began, **fields})

    cache, receipt = output/'fit_rows.npz', output/'fit_rows.receipt.json'
    if receipt.exists():
        meta = json.loads(receipt.read_text())
        if meta['identity_sha256'] != digest(identity) or meta['cache_sha256'] != file_digest(cache):
            raise ValueError('Cached fit rows changed')
        with np.load(cache, allow_pickle=False) as arrays:
            x, y, rows = arrays['features'], arrays['targets'], json.loads(str(arrays['rows_json']))
        audit = meta['audit']
    else:
        rows, features, labels, audit = [], [], [], {}
        for recording, role in contract.protocol['assignments'].items():
            if role != 'fit':
                continue
            reader, indices = contract.open_recording(recording, purpose='fit')
            source = reader.metadata['files'][0]
            if file_digest(ROOT/source['path']) != source['sha256']:
                raise ValueError('Canonical source changed')
            raw, malformed = read_track(ROOT/source['path'])
            cleaned, duplicate = clean_points(raw)
            if not np.array_equal(cleaned, reader.points):
                raise ValueError('Canonical source and cached coordinate rows differ')
            count_before = len(rows)
            all_cv_error = 0.
            for number, index in enumerate(indices):
                row = reader.index[index]
                history = reader.points[int(row['history_start']):int(row['current_row'])+1]
                if np.any(np.diff(history[:, 2:4], axis=0) != 0):
                    continue
                inference = reader.get_inputs(int(index))
                feature = past_features(inference)
                # Feature vector is sealed before target access; labels remain separate.
                label = reader.get_labels(int(index))
                future = label['future_xy_dataset_local']
                if not np.array_equal(label['future_frame_ids'], history[-1, 0]+inference['prediction_frame_offsets']):
                    raise ValueError('Label/native request grid mismatch')
                label_values = stationary_label(history[:, 2:4], future)
                run = stationary_run(reader.points, int(row['current_row']))
                info = {**reader.identity(int(index)), 'fit_fold': contract.protocol['fit_folds'][recording],
                        'physical_scene': reader.metadata['physical_scene'], 'data_role': 'fit',
                        'native_frame_step': int(history[-1, 0]-history[-2, 0]),
                        'history_contiguous': bool(np.all(np.diff(history[:, 0]) == history[-1, 0]-history[-2, 0])),
                        'source_current_row': int(row['current_row']), 'source_history_start': int(row['history_start']),
                        **run, **label_values, 'complete_aligned_neighbors': int(feature[1])}
                rows.append(info); features.append(feature); labels.append(int(label_values['changed']))
                all_cv_error += label_values['native_ade'] / float(inference['causal_features'][2])
                if number % 100 == 0:
                    heartbeat('extracting_fit_only', recording=recording, window=number)
            part = rows[count_before:]
            runs = {(r['agent_id'], r['first_row']) for r in part}
            audit[recording] = {'fit_windows': len(indices), 'stationary_windows': len(part),
                'positive_windows': sum(r['changed'] for r in part), 'agents': len({r['agent_id'] for r in part}),
                'stationary_runs': len(runs), 'runs_with_requested_future_change': len({(r['agent_id'], r['first_row']) for r in part if r['changed']}),
                'runs_starting_at_track_entry': len({(r['agent_id'], r['first_row']) for r in part if r['starts_at_track_entry']}),
                'native_ade_quantiles': np.quantile([r['native_ade'] for r in part], [0, .25, .5, .75, 1]).tolist() if part else [],
                'whole_run_rows_quantiles_label_only': np.quantile([r['whole_run_rows_label_only'] for r in part], [0, .5, 1]).tolist() if part else [],
                'first_change_step_counts': {str(i): sum(r['first_change_step'] == i for r in part) for i in range(1, 13)},
                'windows_no_complete_neighbor': sum(r['complete_aligned_neighbors'] == 0 for r in part),
                'history_gaps': sum(not r['history_contiguous'] for r in part),
                'normalized_cv_error_sum_stationary': all_cv_error,
                'source_path': source['path'], 'source_sha256': source['sha256'], 'cached_source_exact_match': True,
                'all_cached_point_rows': len(cleaned), 'malformed_source_rows': malformed, 'duplicate_source_rows_removed': duplicate,
                'annotation_physical_motion_verified': False}
            print(json.dumps({'recording': recording, **{k: audit[recording][k] for k in ('fit_windows', 'stationary_windows', 'positive_windows', 'agents', 'stationary_runs')}}), flush=True)
        x, y = np.asarray(features), np.asarray(labels)
        tmp = cache.with_suffix('.tmp.npz')
        np.savez(tmp, features=x, targets=y, rows_json=np.array(json.dumps(rows)))
        os.replace(tmp, cache)
        atomic_json(receipt, {'identity_sha256': digest(identity), 'cache_sha256': file_digest(cache), 'audit': audit})
    trials = []
    folds = np.asarray([r['fit_fold'] for r in rows])
    for fold in sorted(set(contract.protocol['fit_folds'].values())):
        train, held = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        if not len(held) or not len(train):
            trials.append({'fold': fold, 'status': 'not_run_no_stationary_held_or_train_rows', 'train_rows': len(train), 'held_rows': len(held)})
            continue
        fit_names, held_names = sorted({rows[i]['recording_id'] for i in train}), sorted({rows[i]['recording_id'] for i in held})
        if set(contract.scene_set(fit_names)) & set(contract.scene_set(held_names)):
            raise ValueError('Physical scene crosses probe train/held roles')
        prior = float((y[train].sum()+1)/(len(train)+2))
        pri = {'fold': fold, 'model': 'ego_only_train_prior', 'seed': None, 'features': 'constant_stationary_ego',
               'status': 'fresh_run', 'train_recordings': fit_names, 'held_recordings': held_names,
               **score_probabilities(y[held], np.full(len(held), prior), prior=prior)}
        trials.append(pri)
        for seed in contract.protocol['seeds']:
            for feature_name, columns in [('geometry', GEOMETRY_COLUMNS), ('geometry_motion', list(range(x.shape[1])))]:
                for model_name, settings in registration['models'].items():
                    name = f'fold{fold}_seed{seed}_{feature_name}_{model_name}'
                    trial_file, model_file = output/(name+'.json'), output/(name+'.joblib')
                    trial_identity = {'run': identity, 'fit_rows_sha256': file_digest(cache), 'fold': fold,
                        'seed': seed, 'features': feature_name, 'columns': columns, 'model': model_name, 'settings': settings,
                        'train_recordings': fit_names, 'held_recordings': held_names}
                    if trial_file.exists():
                        trial = json.loads(trial_file.read_text())
                        if trial['identity'] != trial_identity or (trial['model_sha256'] is not None and file_digest(model_file) != trial['model_sha256']):
                            raise ValueError('Saved fit-fold probe changed')
                        trials.append({**trial, 'status': 'cached_verified'})
                        continue
                    heartbeat('training_probe', trial=name)
                    if len(np.unique(y[train])) < 2:
                        probability, model_hash = np.full(len(held), prior), None
                        fit_status = 'not_run_single_class_train_prior_fallback'
                    else:
                        if model_name == 'logistic':
                            model = make_pipeline(StandardScaler(), LogisticRegression(**settings, random_state=seed))
                        elif model_name == 'extra_trees':
                            model = ExtraTreesClassifier(**settings, random_state=seed, n_jobs=4)
                        else:
                            raise ValueError('Unregistered model family')
                        model.fit(x[train][:, columns], y[train])
                        joblib.dump(model, model_file)
                        model_hash = file_digest(model_file)
                        # Held labels have never entered the estimator or preprocessing.
                        probability = model.predict_proba(x[held][:, columns])[:, 1]
                        fit_status = 'fresh_run'
                    trial = {'identity': trial_identity, 'fold': fold, 'seed': seed, 'features': feature_name,
                        'model': model_name, 'status': fit_status, 'model_sha256': model_hash,
                        'train_rows': len(train), 'train_positives': int(y[train].sum()),
                        'train_recordings': fit_names, 'held_recordings': held_names,
                        'held_agents': len({(rows[i]['recording_id'], rows[i]['agent_id']) for i in held}),
                        'held_stationary_runs': len({(rows[i]['recording_id'], rows[i]['agent_id'], rows[i]['first_row']) for i in held}),
                        **score_probabilities(y[held], probability, prior=prior)}
                    atomic_json(trial_file, trial)
                    atomic_json(output/(name+'.predictions.json'), {'held_row_indices': held.tolist(), 'probability': probability.tolist()})
                    trials.append(trial)
    report = {'result_source': 'fresh_run_source_audit_and_fit_only_probes_or_verified_resume', 'identity': identity,
              'feature_names': FEATURE_NAMES, 'geometry_columns': GEOMETRY_COLUMNS, 'audit': audit, 'trials': trials,
              'development_labels_opened': False, 'primary_protocol_or_metric_changed': False,
              'physical_motion_or_sensor_causality_verified': False, 'independent_confirmation': False,
              'new_deployment': False, 'stage5c_executed': False, 'smc_enabled': False}
    args.report_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.report_dir/'metrics.json', report)
    atomic_json(output/'completion.json', {'identity_sha256': digest(identity), 'report_sha256': file_digest(args.report_dir/'metrics.json')})
    heartbeat('all_fit_only_probes_complete', trials=len(trials))
    print(json.dumps({'status': 'complete', 'stationary_rows': len(rows), 'trials': len(trials), 'development_opened': False}), flush=True)


if __name__ == '__main__':
    main()
