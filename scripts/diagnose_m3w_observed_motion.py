"""Fixed-output motion diagnostics, never a threshold or checkpoint selection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.stats import rankdata
from scripts.build_m3w_observed_motion import load_registration
from src.evaluation.m3w_experiment_contract import file_digest
from src.world_model.m3w_offline_visual_data import json_write
from src.world_model.m3w_observed_motion import feature_variant
from src.world_model.m3w_objective_alignment import supported_standardization


def auc(labels, scores):
    positives, negatives = int(labels.sum()), int((~labels).sum())
    if not positives or not negatives:
        return None
    return float((rankdata(scores)[labels].sum() - positives*(positives+1)/2) / (positives*negatives))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registration', type=Path, required=True)
    args = p.parse_args()
    reg, parent, _, data, receipt = load_registration(args.registration)
    output, reports = ROOT/reg['output'], ROOT/reg['reports']
    report = json.loads((reports/'report.json').read_text())
    if not report['complete'] or len(report['trials']) != 54:
        raise ValueError('Require all fixed fits before post-fit diagnosis')
    a = {k: np.load(data/(k+'.npy'), mmap_mode='r') for k in ('geometry', 'targets', 'baselines', 'folds', 'scale')}
    m = np.load(output/'inputs/motion.npy', mmap_mode='r')
    q = np.load(output/'inputs/quality.npy', mmap_mode='r')
    receipt_motion = json.loads((output/'inputs/manifest.json').read_text())
    for name, digest in receipt_motion['arrays'].items():
        if file_digest(output/'inputs'/name) != digest:
            raise ValueError('Changed input motion')
    meta = json.loads((data/'rows.json').read_text())
    record = np.array([r['recording'] for r in meta])
    stationary = np.all(a['geometry'][:, :16] == 0, axis=1)
    future_static = np.all(a['targets'] == 0, axis=(1, 2))
    features = feature_variant(a['geometry'], m, q, 'directed')
    domains, support, trials = {}, {}, []
    for rid in sorted(set(record)):
        ids = np.flatnonzero(record == rid)
        static_ids = ids[stationary[ids]]
        labels = ~future_static[static_ids]
        scores = m[static_ids, -1, 7]
        domains[rid] = dict(rows=len(ids), stationary_rows=len(static_ids),
            stationary_agents=len(set(meta[i]['agent'] for i in static_ids)),
            image_pair_available_fraction=float(q[ids, :, 4].mean()),
            center_consistency_mean=float(q[ids, :, 2].mean()),
            ring_consistency_mean=float(q[ids, :, 3].mean()),
            static_future_movement_rows=int(labels.sum()),
            stationary_motion_magnitude_quantiles=np.quantile(scores, [0, .25, .5, .75, .95, 1]).tolist() if len(scores) else None,
            motion_magnitude_departure_AUROC_descriptive=auc(labels, scores) if len(scores) else None,
            high_flow_consistency_is_not_identity_or_future_direction_confidence=True)
    for fold in range(3):
        train, held = np.flatnonzero(a['folds'] != fold), np.flatnonzero(a['folds'] == fold)
        _, normalizer = supported_standardization(features[train], features[held])
        standardized = (features[held] - normalizer['mean']) / normalizer['std']
        standardized[:, normalizer['constant']] = 0
        motion_z = standardized[:, -70:]
        support[fold] = dict(held_rows=len(held),
            any_motion_column_clipped_fraction=float((np.abs(motion_z) > 10).any(1).mean()),
            stationary_motion_clipped_fraction=float((np.abs(motion_z[stationary[held]]) > 10).any(1).mean()) if stationary[held].any() else None,
            constant_motion_columns=int(normalizer['constant'][-70:].sum()))
    cv = receipt['baseline_names'].index('constant_velocity_causal_fd')
    for trial in report['trials']:
        path = output/'predictions'/(trial['trial']+'.npz')
        if file_digest(path) != trial['prediction_sha256']:
            raise ValueError('Frozen prediction changed')
        with np.load(path) as cached:
            ids, prediction = cached['held_indices'], cached['prediction'].astype(float)
        target = a['targets'][ids].astype(float)
        ref = np.linalg.norm(a['baselines'][ids, cv].astype(float)-target, axis=-1).mean(1)
        error = np.linalg.norm(prediction-target, axis=-1).mean(1)
        extra = np.maximum(error-ref, 0)
        slices = {}
        for name, mask in [('stationary_history', stationary[ids]), ('moving_history', ~stationary[ids]),
                           ('stationary_past_and_future', stationary[ids] & future_static[ids]),
                           ('stationary_past_future_movement', stationary[ids] & ~future_static[ids])]:
            slices[name] = dict(rows=int(mask.sum()),
                mean_harm=float((error[mask]-ref[mask]).mean()) if mask.any() else None,
                positive_harm_share=float(extra[mask].sum()/extra.sum()) if extra.sum() else None)
        native = {}
        for rid in sorted(set(record[ids])):
            mask = record[ids] == rid
            predicted, reference = (error[mask]*a['scale'][ids[mask]]).mean(), (ref[mask]*a['scale'][ids[mask]]).mean()
            native[rid] = dict(model_ADE=float(predicted), CV_ADE=float(reference), gain_percent=float(100*(1-predicted/reference)),
                               unit='dataset_local_unverified_diagnostic_not_pooled')
        trials.append(dict(trial=trial['trial'], variant=trial['variant'], objective=trial['objective'],
            seed=trial['seed'], fold=trial['fold'], slices=slices, native_by_recording=native))
    result = dict(result_source='fresh_run_diagnosis_hash_verified_fixed_predictions',
        code_sha256=file_digest(Path(__file__)), report_sha256=file_digest(reports/'report.json'),
        adaptive_fit_only_diagnostic=True, domains=domains, support=support, trials=trials,
        diagnosis_labels_not_inputs=True, thresholds_or_models_selected=False,
        overlapping_window_AUROC_not_independent_evidence=True,
        development_calibration_confirmation_opened=False)
    json_write(reports/'failure_diagnosis.json', result)
    print(json.dumps(dict(domains=domains, support=support, trials=len(trials))))


if __name__ == '__main__':
    main()
