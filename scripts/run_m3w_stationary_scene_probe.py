"""Fit-only static-scene start/trajectory probes; no deployment or final labels."""
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
    from PIL import Image
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
    from src.evaluation.m3w_experiment_contract import ExperimentContract, file_digest
    from src.evaluation.m3w_stationary_start_probe import score_probabilities
    from src.evaluation.m3w_stationary_scene_context import (
        read_obstacles, scene_context, encode_future_label, forecast_metrics, NAMES, FEATURE_SETS,
    )
    from scripts.run_m3w_stationary_start_probe import atomic_json, digest
    from scripts.summarize_m3w_stationary_probe import group_balanced_brier

    registration = json.loads(args.registration.read_text())
    if registration['role'] != 'fit_only_static_proxy_diagnostic':
        raise ValueError('Invalid role')
    for p, sha in registration['bindings'].items():
        if file_digest(ROOT/p) != sha:
            raise ValueError('Frozen binding changed: '+p)
    contract = ExperimentContract(json.loads((ROOT/registration['parent_protocol']).read_text()), ROOT)
    if contract.digest != registration['parent_protocol_sha256']:
        raise ValueError('Parent changed')
    source_identity = json.loads((args.source/'run_identity.json').read_text())
    source_receipt = json.loads((args.source/'fit_rows.receipt.json').read_text())
    source_cache = args.source/'fit_rows.npz'
    if (file_digest(source_cache) != source_receipt['cache_sha256']
            or digest(source_identity) != source_receipt['identity_sha256']
            or source_identity['parent_protocol_sha256'] != contract.digest
            or source_identity['registration_sha256'] != file_digest(ROOT/registration['source_registration'])):
        raise ValueError('Source stationary rows changed')
    with np.load(source_cache, allow_pickle=False) as arrays:
        source_rows = json.loads(str(arrays['rows_json']))
        source_y = arrays['targets'].copy()
    if any(contract.protocol['assignments'][r['recording_id']] != 'fit' or r['data_role'] != 'fit' for r in source_rows):
        raise ValueError('Only fit rows allowed')
    identity = {'registration_sha256': file_digest(args.registration), 'parent_protocol_sha256': contract.digest,
                'source_rows_sha256': file_digest(source_cache), 'numpy': np.__version__, 'sklearn': sklearn.__version__}
    output = args.output.resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError('Workspace-local output required')
    if args.resume:
        if json.loads((output/'run_identity.json').read_text()) != identity:
            raise ValueError('Resume identity mismatch')
    else:
        output.mkdir(parents=True, exist_ok=False)
        atomic_json(output/'run_identity.json', identity)
    began = time.monotonic()
    def beat(state, **kwargs):
        atomic_json(output/'heartbeat.json', {'pid': os.getpid(), 'state': state,
                    'elapsed_seconds': time.monotonic()-began, **kwargs})

    cache, receipt = output/'scene_rows.npz', output/'scene_rows.receipt.json'
    if receipt.exists():
        rec = json.loads(receipt.read_text())
        if rec['identity_sha256'] != digest(identity) or rec['cache_sha256'] != file_digest(cache):
            raise ValueError('Scene cache mismatch')
        assets = rec['assets']
    else:
        features, targets, native, basis, scales, parent_scales, valid, rows = [], [], [], [], [], [], [], []
        assets, readers, lookups, maps = {}, {}, {}, {}
        for recording in sorted({r['recording_id'] for r in source_rows}):
            reader, indices = contract.open_recording(recording, purpose='fit')
            readers[recording] = reader
            lookups[recording] = {(reader.identity(int(i))['agent_id'], reader.identity(int(i))['frame_id']): int(i) for i in indices}
            directory = ROOT/registration['maps'][recording]
            obstacles = read_obstacles(directory/'map.xml'); maps[recording] = obstacles
            files = {name: {'sha256': file_digest(directory/name), 'bytes': (directory/name).stat().st_size}
                     for name in ('map.xml', 'map.png', 'reference.png', 'H.txt', 'info.txt')}
            with Image.open(directory/'reference.png') as im:
                size = list(im.size)
            matrix = np.loadtxt(directory/'H.txt')
            inverse = np.linalg.inv(matrix)
            positions = reader.points[:, 2:4]
            projection = np.c_[positions, np.ones(len(positions))] @ inverse.T
            finite = np.abs(projection[:, 2]) > 1e-12
            uv = projection[finite, :2]/projection[finite, 2:3]
            inside = (uv[:, 0]>=0)&(uv[:, 0]<size[0])&(uv[:, 1]>=0)&(uv[:, 1]<size[1])
            assets[recording] = {'files': files, 'reference_size': size,
                'line_count': len(obstacles['lines']), 'circle_count': len(obstacles['circles']),
                'finite_H_projection_rows': int(finite.sum()), 'all_fit_recording_points': len(positions),
                'projected_within_reference_fraction': float(inside.mean()),
                'projection_overlay_or_geometric_truth_verified': False,
                'reference_image_used_as_input': False, 'destinations_or_groups_used': False,
                'map_status': 'supplied_static_proxy_authorship_and_geometric_accuracy_unverified',
                'image_clock_status': 'unverified_image_capture_and_annotation_mapping',
                'asset_admission': 'local_fit_only_diagnostic_no_redistribution_or_official_modality_claim'}
        for i, old in enumerate(source_rows):
            rid = old['recording_id']; reader = readers[rid]
            item = lookups[rid][(old['agent_id'], old['frame_id'])]
            cur = int(reader.index[item]['current_row'])
            if cur != old['source_current_row']:
                raise ValueError('Source row alignment changed')
            inputs = reader.get_inputs(item)
            center = reader.points[cur, 2:4]
            f, transform = scene_context(inputs, center, maps[rid])
            # Freeze all feature construction before opening this row's supervision.
            features.append(f.copy()); basis.append(transform['basis']); scales.append(transform['scale'])
            parent_scales.append(float(inputs['causal_features'][2])); valid.append(transform['defined'])
            label = reader.get_labels(item)
            target = label['future_xy_dataset_local']
            if not np.array_equal(label['future_frame_ids'], old['frame_id']+inputs['prediction_frame_offsets']):
                raise ValueError('Future request grid mismatch')
            if int(np.any(target != center)) != source_y[i]:
                raise ValueError('Frozen start target changed')
            native.append(target-center); targets.append(encode_future_label(target, center, transform))
            rows.append(old)
            if i%50 == 0:
                beat('extracting_past_static_context', row=i)
        tmp = cache.with_suffix('.tmp.npz')
        np.savez(tmp, features=np.asarray(features), targets=np.asarray(targets), native=np.asarray(native),
                 basis=np.asarray(basis), scale=np.asarray(scales), parent_scale=np.asarray(parent_scales),
                 valid=np.asarray(valid), start=source_y, rows_json=np.array(json.dumps(rows)))
        os.replace(tmp, cache)
        atomic_json(receipt, {'identity_sha256': digest(identity), 'cache_sha256': file_digest(cache), 'assets': assets})
    with np.load(cache, allow_pickle=False) as arrays:
        x, target, native, basis, scale, parent_scale, valid, y = [arrays[k].copy() for k in
            ('features', 'targets', 'native', 'basis', 'scale', 'parent_scale', 'valid', 'start')]
        rows = json.loads(str(arrays['rows_json']))
    folds = np.array([r['fit_fold'] for r in rows])
    easy_threshold = contract.protocol['development_evaluation']['easy_threshold']
    trials, controls = [], []
    for fold in sorted(set(contract.protocol['fit_folds'].values())):
        train, held = np.flatnonzero(folds!=fold), np.flatnonzero(folds==fold)
        if not len(held) or not len(train):
            controls.append({'fold': fold, 'status': 'not_run_no_stationary_held_or_fit_rows'})
            continue
        if {rows[i]['physical_scene'] for i in train} & {rows[i]['physical_scene'] for i in held}:
            raise ValueError('Scene overlap')
        fit_reg = train[valid[train]]
        if not len(fit_reg):
            raise ValueError('No defined training reference frames; regression not run')
        prior = float((y[train].sum()+1)/(len(train)+2))
        group = [(rows[i]['recording_id'], rows[i]['agent_id'], rows[i]['first_row']) for i in held]
        def score_reg(p):
            return forecast_metrics(p, native[held], parent_scale[held], easy_threshold, group)
        def restore(p):
            prediction = np.einsum('nki,nji->nkj', p.reshape(-1, 12, 2), basis[held])*scale[held, None, None]
            prediction[~valid[held]] = 0.
            return prediction
        constant = np.tile(target[fit_reg].mean(0), (len(held), 1, 1))
        controls.append({'fold': fold, 'held': sorted({rows[i]['recording_id'] for i in held}),
            'train_rows': len(train), 'held_rows': len(held), 'valid_train_frames': len(fit_reg),
            'undefined_held_frames': int((~valid[held]).sum()), 'prior': prior,
            'cv': score_reg(np.zeros_like(native[held])), 'train_mean_trajectory': score_reg(restore(constant))})
        for seed in contract.protocol['seeds']:
            for feature_name, columns in FEATURE_SETS.items():
                for family in ('linear', 'extra_trees'):
                    name = f'fold{fold}_seed{seed}_{feature_name}_{family}'
                    trial_path = output/(name+'.json')
                    model_path = output/(name+'.joblib')
                    trial_id = {'identity': identity, 'cache_sha256': file_digest(cache),
                                'fold': fold, 'seed': seed, 'feature_name': feature_name, 'columns': columns, 'family': family}
                    if trial_path.exists():
                        trial = json.loads(trial_path.read_text())
                        if trial['trial_identity'] != trial_id or file_digest(model_path) != trial['checkpoint_sha256']:
                            raise ValueError('Stored trial changed')
                        trials.append({**trial, 'status': 'cached_verified'})
                        continue
                    beat('fitting_start_and_trajectory', trial=name)
                    start_time = time.monotonic()
                    if family == 'linear':
                        classifier = make_pipeline(StandardScaler(), LogisticRegression(C=1., max_iter=1000, random_state=seed))
                        regressor = make_pipeline(StandardScaler(), Ridge(alpha=1.))
                    else:
                        params = dict(n_estimators=256, min_samples_leaf=10, max_features=1., n_jobs=4, random_state=seed)
                        classifier, regressor = ExtraTreesClassifier(**params), ExtraTreesRegressor(**params)
                    classifier.fit(x[train][:, columns], y[train])
                    regressor.fit(x[fit_reg][:, columns], target[fit_reg].reshape(-1, 24))
                    fit_seconds = time.monotonic()-start_time
                    probability = classifier.predict_proba(x[held][:, columns])[:, 1]
                    p = restore(regressor.predict(x[held][:, columns]))
                    switch = probability >= registration['diagnostic_probability_gate']
                    safe = p*switch[:, None, None]
                    # No winner or threshold is chosen with these held-fold outcomes.
                    tmp_model = model_path.with_suffix('.tmp.joblib')
                    joblib.dump({'classifier': classifier, 'regressor': regressor}, tmp_model)
                    os.replace(tmp_model, model_path)
                    trial = {'trial_identity': trial_id, 'status': 'fresh_run', 'fold': fold, 'seed': seed,
                        'feature_name': feature_name, 'family': family, 'feature_dimension': len(columns),
                        'checkpoint_sha256': file_digest(model_path), 'fit_seconds': fit_seconds,
                        'classification': score_probabilities(y[held], probability, prior=prior),
                        'run_balanced_brier': group_balanced_brier(y[held], probability, prior, group),
                        'agent_balanced_brier': group_balanced_brier(y[held], probability, prior,
                            [(rows[i]['recording_id'], rows[i]['agent_id']) for i in held]),
                        'trajectory_unrestricted': score_reg(p), 'trajectory_fixed_gate': score_reg(safe),
                        'fixed_gate_switch_rate': float(switch.mean()), 'new_deployment': False}
                    atomic_json(trial_path, trial)
                    np.savez(output/(name+'.predictions.npz'), held=held, probability=probability, prediction=p, guarded=safe)
                    trials.append(trial)
                    print(json.dumps({'trial': name, 'brier_lift': trial['classification']['brier_lift_over_prior'],
                        'trajectory_gain_pct': trial['trajectory_unrestricted']['gain_vs_cv_pct'],
                        'guarded_gain_pct': trial['trajectory_fixed_gate']['gain_vs_cv_pct']}), flush=True)
    report = {'result_source': 'fresh_run_fit_only_scene_proxy_classification_and_regression_or_verified_resume',
        'identity': identity, 'cache_sha256': file_digest(cache), 'features': NAMES, 'assets': assets,
        'rows': len(rows), 'undefined_static_frames': int((~valid).sum()),
        'controls': controls, 'trials': trials, 'models_per_trial': 2,
        'independent_confirmation': False, 'parent_protocol_or_metric_changed': False,
        'development_labels_opened': False, 'scene_images_encoded': False,
        'physical_geometry_or_asof_annotation_verified': False, 'new_deployment': False,
        'stage5c_executed': False, 'smc_enabled': False}
    args.report_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.report_dir/'metrics.json', report)
    atomic_json(output/'completion.json', {'identity_sha256': digest(identity), 'report_sha256': file_digest(args.report_dir/'metrics.json')})
    beat('complete', model_pairs=len(trials), fitted_models=2*len(trials))
    print(json.dumps({'status': 'complete', 'fitted_models': 2*len(trials), 'elapsed_seconds': time.monotonic()-began}), flush=True)


if __name__ == '__main__':
    main()
