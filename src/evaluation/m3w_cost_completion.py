"""Verify a completed ridge cost head and its OOF dependencies without Torch.

An existing fit report binds features but not target bytes. For old runs, an
independently retained file-binding snapshot is needed to establish that targets
have not changed together with their receipt. A self-consistent current archive
alone is not such an anchor, nor is this verifier an optimal-fit proof.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.evaluation.m3w_experiment_contract import file_digest
from src.world_model.m3w_oof_identity import oof_feature_identity


def verify_completed_ridge(contract, directory, *, source_paths, expected_run_identity=None,
                           expected_files=None):
    contract._assert_frozen()
    directory = Path(directory).resolve()
    if not directory.is_relative_to(contract.root):
        raise ValueError('Completed cost head must remain in workspace')
    bindings = {}

    def checked(path):
        path = Path(path).resolve()
        if not path.is_relative_to(contract.root):
            raise ValueError('Dependency path escapes workspace')
        relative = str(path.relative_to(contract.root))
        value = file_digest(path)
        if expected_files is not None and expected_files.get(relative) != value:
            raise ValueError('Completed dependency binding changed or missing: '+relative)
        bindings[relative] = value
        return path

    def read(name):
        return json.loads(checked(directory / name).read_text())

    run, report, artifact = (read(p) for p in ('run_identity.json', 'fit_report.json', 'artifact.json'))
    if expected_run_identity is not None and run != expected_run_identity:
        raise ValueError('Completed run settings changed')
    if (set(source_paths) != {'code_sha256', 'oof_identity_source_sha256', 'script_sha256'} or
            any(run[k] != file_digest(Path(path)) for k, path in source_paths.items()) or
            run['protocol_sha256'] != contract.digest):
        raise ValueError('Completed source or protocol identity changed')
    mapping, folds = run['fold_models'], contract.protocol['fit_folds']
    if set(mapping) != {str(f) for f in folds.values()}:
        raise ValueError('Complete held-fold mapping required')
    if set(run['predictor_sha256']) != set(mapping.values()):
        raise ValueError('Producer hash set differs from held-fold mapping')
    if (not np.isfinite(run['alpha']) or run['alpha'] <= 0 or
            type(run['batch_size']) is not int or run['batch_size'] < 1):
        raise ValueError('Invalid original fitting settings')
    groups, recordings = [], set()
    for fold, producer in sorted(mapping.items()):
        held = sorted(r for r, f in folds.items() if str(f) == fold)
        if contract.artifacts[producer]['kind'] != 'forecaster':
            raise ValueError('OOF producer is not a forecaster')
        if run['predictor_sha256'][producer] != contract.artifacts[producer]['sha256']:
            raise ValueError('OOF producer identity changed')
        contract.assert_prediction_use(producer, held, purpose='oof_risk_training')
        for parent in contract._closure(producer):
            checked(contract._path(contract.artifacts[parent]['path']))
        for name in held:
            contract.open_recording(name, purpose='fit')
        receipt = read(f'fold_{fold}.json')
        cache = checked(directory / f'fold_{fold}.npz')
        if receipt['run_identity'] != run or file_digest(cache) != receipt['cache_sha256']:
            raise ValueError('Completed OOF cache or receipt changed')
        meta = receipt['group_metadata']
        expected_meta = dict(predictor_id=producer, predictor_sha256=run['predictor_sha256'][producer],
            protocol_sha256=contract.digest, metric=contract.protocol['task']['primary_metric'],
            baseline_name=run['baseline'], feature_source='past_and_frozen_rollouts_only',
            target_source='held_fold_realized_relative_costs')
        if meta != expected_meta:
            raise ValueError('Completed OOF provenance changed')
        with np.load(cache, allow_pickle=False) as arrays:
            x, y = arrays['features'], arrays['targets']
            rows = json.loads(arrays['identities_json'].item())
        if (x.ndim != 2 or not len(x) or x.shape[1] < 1 or y.shape != (len(x), 2) or
                len(rows) != len(x) or not np.isfinite(x).all() or not np.isfinite(y).all() or (y < 0).any()):
            raise ValueError('Malformed completed OOF arrays')
        if {r['recording_id'] for r in rows} != set(held):
            raise ValueError('Completed OOF rows do not cover the held fold')
        for row in rows:
            if (row.get('data_role') != 'fit' or row.get('protocol') != contract.protocol['task']['prediction_unit'] or
                    row.get('physical_scene') != contract.protocol['records'][row['recording_id']]['physical_scene']):
                raise ValueError('Completed OOF row provenance changed')
        recordings.update(held)
        groups.append(dict(meta, features=x, targets=y, identities=rows))
    identity = oof_feature_identity(groups)
    x = np.concatenate([g['features'] for g in groups])
    checkpoint = checked(directory / 'linear_cost_head.npz')
    checkpoint_sha = file_digest(checkpoint)
    expected_report = dict(protocol_sha256=contract.digest, alpha=run['alpha'],
        fit_recordings=sorted(recordings), parents=sorted(set(mapping.values())),
        normalization_source='fit_OOF_rows_only', baseline_name=run['baseline'],
        metric=contract.protocol['task']['primary_metric'], oof_feature_identity=identity,
        training_rows=len(x), oof_total_folds=len(groups), feature_dimension=x.shape[1],
        calibrated_policy=False, test_evaluated=False, checkpoint_sha256=checkpoint_sha,
        code_sha256=run['code_sha256'])
    if any(report.get(k) != v for k, v in expected_report.items()):
        raise ValueError('Completed fitting report provenance changed')
    expected_artifact = dict(id=directory.name, kind='risk_head', path=str(checkpoint.relative_to(contract.root)),
        sha256=checkpoint_sha, protocol_sha256=contract.digest, lineage_complete=True,
        fit_recordings=sorted(recordings), selection_recordings=[], calibration_recordings=[],
        parents=sorted(set(mapping.values())))
    if artifact != expected_artifact:
        raise ValueError('Completed head artifact provenance changed')
    with np.load(checkpoint, allow_pickle=False) as weights:
        expected_shapes = dict(mean=(x.shape[1],), scale=(x.shape[1],), coef=(2, x.shape[1]), intercept=(2,))
        if set(weights.files) != set(expected_shapes) or any(
                weights[k].shape != shape or not np.isfinite(weights[k]).all() for k, shape in expected_shapes.items()):
            raise ValueError('Malformed completed ridge checkpoint')
        if (not np.array_equal(weights['mean'], x.mean(0)) or
                not np.array_equal(weights['scale'], np.maximum(x.std(0), 1e-6))):
            raise ValueError('Completed fit-only normalization changed')
    contract._assert_frozen()
    for relative, expected in bindings.items():
        if file_digest(contract.root / relative) != expected:
            raise ValueError('Completed dependency changed during verification')
    return dict(status='completed_ridge_dependencies_verified', protocol_sha256=contract.digest,
        artifact_manifest_digest=contract._artifact_digest(), run_identity=run,
        file_bindings=bindings, training_rows=len(x), feature_dimension=x.shape[1],
        oof_feature_identity=identity, normalization_exact=True,
        target_source_status=('externally_bound_bytes_verified' if expected_files is not None
                              else 'receipt_verified_not_independently_anchored'),
        new_training=False, new_inference=False, independent_calibration=False)
