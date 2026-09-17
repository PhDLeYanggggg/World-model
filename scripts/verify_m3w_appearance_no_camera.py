"""Replay every no-camera repair checkpoint and preserve aggregate evidence."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--report-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if platform.system() == 'Darwin' and platform.machine() != 'arm64':
        raise SystemExit('Native arm64 interpreter required')
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[key] = '4'
    import numpy as np
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    from src.evaluation.m3w_experiment_contract import file_digest
    from src.world_model.m3w_past_appearance_probe import PastAppearanceProbe, pixel_delta_to_native
    from scripts.run_m3w_appearance_no_camera import no_camera_geometry
    from scripts.run_m3w_stationary_start_probe import atomic_json

    reg = json.loads(args.registration.read_text())
    for p, digest in reg['bindings'].items():
        if file_digest(ROOT / p) != digest:
            raise ValueError('Changed registered source: ' + p)
    original_reg = json.loads((ROOT / reg['original_registration']).read_text())
    for p, digest in original_reg['bindings'].items():
        if file_digest(ROOT / p) != digest:
            raise ValueError('Changed original registered source: ' + p)
    complete = json.loads((args.study / 'completion.json').read_text())
    metrics_path = args.report_dir / 'metrics.json'
    if (complete['identity']['registration_sha256'] != file_digest(args.registration)
            or complete['report_sha256'] != file_digest(metrics_path)):
        raise ValueError('Changed completion/metrics')
    metrics = json.loads(metrics_path.read_text())
    if metrics['completed_models'] != 6 or metrics['total_updates'] != 6000:
        raise ValueError('Incomplete registered budget')
    with np.load(ROOT / reg['cache'], allow_pickle=False) as a:
        x = a['geometry'].copy()
        images = torch.tensor(a['rgb'].astype(np.float32) / 255 - .5)
        mask = torch.tensor(a['mask'].astype(np.float32))
        xy, h = torch.tensor(a['image_xy']), torch.tensor(a['homography'])
    maximum_difference = 0.0
    for trial in metrics['trials']:
        name = f'fold{trial["fold"]}_seed{trial["seed"]}_past_rgb'
        checkpoint, prediction_file = args.study / (name + '.pt'), args.study / (name + '.npz')
        receipt = args.study / (name + '.json')
        if (file_digest(checkpoint) != trial['checkpoint_sha256']
                or file_digest(prediction_file) != trial['predictions_sha256']
                or file_digest(receipt) != complete['receipts'][name]):
            raise ValueError('Changed saved model/predictions')
        record = json.loads(receipt.read_text())
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if (state['identity'] != record['identity'] or state['step'] != 1000
                or state['identity']['feature_treatment'] != 'zero_normalized_camera_28_32'):
            raise ValueError('Changed model identity or budget')
        original = state['identity']['original_trial_identity']
        geometry = torch.tensor(no_camera_geometry(x, original['normalization_mean'], original['normalization_std']))
        model = PastAppearanceProbe(32)
        model.load_state_dict(state['model'])
        model.eval()
        with np.load(prediction_file, allow_pickle=False) as a:
            held, saved, saved_probability, saved_guarded = [a[k].copy() for k in ('held', 'prediction', 'probability', 'guarded')]
        predictions, probabilities = [], []
        with torch.no_grad():
            for start in range(0, len(held), 32):
                ids = held[start:start + 32]
                delta, logits = model(geometry[ids], images[ids], mask[ids], 'past_rgb')
                predictions.append(pixel_delta_to_native(delta, xy[ids], h[ids]).numpy())
                probabilities.append(torch.sigmoid(logits).numpy())
        p, prob = np.concatenate(predictions), np.concatenate(probabilities)
        switch = (prob >= original_reg['diagnostic_probability_gate']) & mask[held].bool().all(1).numpy()
        guarded = p * switch[:, None, None]
        difference = max(float(np.abs(p - saved).max()), float(np.abs(prob - saved_probability).max()),
                         float(np.abs(guarded - saved_guarded).max()))
        if difference > 1e-10:
            raise ValueError('Checkpoint replay mismatch: ' + name)
        maximum_difference = max(maximum_difference, difference)
    grouped = {}
    for fold in (0, 1):
        trials = [t for t in metrics['trials'] if t['fold'] == fold]
        grouped[str(fold)] = {
            'original_guarded_gain_mean': float(np.mean([t['original_guarded']['gain_vs_cv_pct'] for t in trials])),
            'no_camera_guarded_gain_mean': float(np.mean([t['guarded']['gain_vs_cv_pct'] for t in trials])),
            'no_camera_unrestricted_gain_mean': float(np.mean([t['unrestricted']['gain_vs_cv_pct'] for t in trials])),
            'no_camera_easy_absolute_harm_mean': float(np.mean([t['guarded']['easy_absolute_harm'] for t in trials])),
            'no_camera_brier_lift_mean': float(np.mean([t['classification']['brier_lift_over_prior'] for t in trials])),
            'no_camera_switch_rate_mean': float(np.mean([t['switch_rate'] for t in trials])),
        }
    output = args.output.resolve()
    if not output.is_relative_to(ROOT) or output.exists():
        raise ValueError('Use new workspace output file')
    result = {'result_source': 'cached_verified_checkpoint_replay', 'checkpoints': 6,
              'maximum_prediction_difference': maximum_difference, 'seed_means': grouped,
              'registration_sha256': file_digest(args.registration), 'metrics_sha256': file_digest(metrics_path),
              'verification_code_sha256': file_digest(Path(__file__)), 'new_training': False,
              'model_selection': False, 'new_deployment': False}
    atomic_json(output, result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
