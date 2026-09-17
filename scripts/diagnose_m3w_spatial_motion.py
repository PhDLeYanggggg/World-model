"""Post-fit decomposition of fixed predictions; no model/threshold selection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts.build_m3w_observed_motion import load_registration
from src.evaluation.m3w_experiment_contract import file_digest
from src.world_model.m3w_offline_visual_data import json_write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', type=Path, required=True)
    args = parser.parse_args()
    reg, _, _, data, receipt = load_registration(args.registration)
    output, reports = ROOT/reg['output'], ROOT/reg['reports']
    report = json.loads((reports/'report.json').read_text())
    if not report['complete'] or len(report['trials']) != 36:
        raise ValueError('Require all 36 fixed fits before diagnosis')
    arrays = {k: np.load(data/(k+'.npy'), mmap_mode='r')
              for k in ('geometry', 'targets', 'baselines', 'folds', 'scale')}
    rows = json.loads((data/'rows.json').read_text())
    recording = np.array([row['recording'] for row in rows])
    stationary = np.all(arrays['geometry'][:, :16] == 0, axis=1)
    future_stationary = np.all(arrays['targets'] == 0, axis=(1, 2))
    cv = receipt['baseline_names'].index('constant_velocity_causal_fd')
    domains, trials = {}, []
    for rid in sorted(set(recording)):
        ids = np.flatnonzero(recording == rid)
        static_ids = ids[stationary[ids]]
        domains[rid] = dict(rows=len(ids), stationary_rows=len(static_ids),
            stationary_source_ids=len(set(rows[i]['agent'] for i in static_ids)),
            stationary_past_future_movement_rows=int((~future_stationary[static_ids]).sum()))
    for trial in report['trials']:
        path = output/'predictions'/(trial['trial']+'.npz')
        if file_digest(path) != trial['prediction_sha256']:
            raise ValueError('Frozen prediction changed')
        with np.load(path) as cached:
            ids = cached['held_indices']
            prediction = cached['prediction'].astype(float)
        target = arrays['targets'][ids].astype(float)
        reference = np.linalg.norm(arrays['baselines'][ids, cv].astype(float)-target, axis=-1).mean(1)
        error = np.linalg.norm(prediction-target, axis=-1).mean(1)
        positive_harm = np.maximum(error-reference, 0)
        slices = {}
        for name, mask in (
            ('stationary_history', stationary[ids]),
            ('moving_history', ~stationary[ids]),
            ('stationary_past_and_future', stationary[ids] & future_stationary[ids]),
            ('stationary_past_future_movement', stationary[ids] & ~future_stationary[ids]),
        ):
            slices[name] = dict(rows=int(mask.sum()),
                model_ADE=float(error[mask].mean()) if mask.any() else None,
                CV_ADE=float(reference[mask].mean()) if mask.any() else None,
                mean_harm=float((error[mask]-reference[mask]).mean()) if mask.any() else None,
                positive_harm_share=float(positive_harm[mask].sum()/positive_harm.sum())
                    if positive_harm.sum() else None)
        native = {}
        for rid in sorted(set(recording[ids])):
            mask = recording[ids] == rid
            scale = arrays['scale'][ids[mask]]
            pred_ade, cv_ade = (error[mask]*scale).mean(), (reference[mask]*scale).mean()
            native[rid] = dict(model_ADE=float(pred_ade), CV_ADE=float(cv_ade),
                gain_percent=float(100*(1-pred_ade/cv_ade)),
                unit='dataset_local_unverified_diagnostic_not_pooled')
        trials.append(dict(trial=trial['trial'], variant=trial['variant'],
            seed=trial['seed'], fold=trial['fold'], slices=slices, native_by_recording=native))
    result = dict(result_source='fresh_run_diagnosis_hash_verified_fixed_predictions',
        code_sha256=file_digest(Path(__file__)), report_sha256=file_digest(reports/'report.json'),
        adaptive_fit_only_diagnostic=True, domains=domains, trials=trials,
        diagnosis_labels_not_inputs=True, thresholds_or_models_selected=False,
        independent_scene_clusters=3, development_calibration_confirmation_opened=False)
    json_write(reports/'failure_diagnosis.json', result)
    summary = {}
    for variant in reg['variants']:
        own = [t for t in trials if t['variant'] == variant]
        hotel = [t['slices']['stationary_history']['positive_harm_share'] for t in own if t['fold'] == 1]
        native = {rid: float(np.mean([t['native_by_recording'][rid]['gain_percent']
                  for t in own if rid in t['native_by_recording']])) for rid in domains}
        summary[variant] = dict(hotel_stationary_positive_harm_share_range=[min(hotel), max(hotel)],
                               native_by_recording_seed_mean_gain=native)
    print(json.dumps(dict(domains=domains, variants=summary)))


if __name__ == '__main__':
    main()
