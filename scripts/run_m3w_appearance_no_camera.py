"""Single-factor, fixed-budget past-RGB retraining without camera features."""
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


def no_camera_geometry(features, mean, std):
    import numpy as np
    x, mean, std = np.asarray(features), np.asarray(mean), np.asarray(std)
    if (x.ndim != 2 or x.shape[1] != 32 or mean.shape != (32,) or std.shape != (32,)
            or not np.isfinite(x).all() or not np.isfinite(mean).all()
            or not np.isfinite(std).all() or np.any(std <= 0)):
        raise ValueError('Finite original geometry/normalization required')
    result = np.clip((x - mean) / std, -10, 10).astype(np.float32)
    result[:, 28:32] = 0
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report-dir', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if platform.system() == 'Darwin' and platform.machine() != 'arm64':
        raise SystemExit('Native arm64 interpreter required')
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[key] = '4'
    import numpy as np
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    from src.evaluation.m3w_experiment_contract import ExperimentContract, file_digest
    from src.evaluation.m3w_stationary_scene_context import forecast_metrics
    from src.evaluation.m3w_stationary_start_probe import score_probabilities
    from src.world_model.m3w_past_appearance_probe import PastAppearanceProbe, fit_probe, pixel_delta_to_native
    from scripts.run_m3w_stationary_start_probe import atomic_json

    reg = json.loads(args.registration.read_text())
    for p, digest in reg['bindings'].items():
        if file_digest(ROOT / p) != digest:
            raise ValueError('Changed repair source: ' + p)
    old_reg = json.loads((ROOT / reg['original_registration']).read_text())
    for p, digest in old_reg['bindings'].items():
        if file_digest(ROOT / p) != digest:
            raise ValueError('Changed original source: ' + p)
    old_report = json.loads((ROOT / reg['original_report']).read_text())
    old_study = ROOT / reg['original_study']
    old_completion = json.loads((old_study / 'completion.json').read_text())
    if (not old_report['complete_registered_budget']
            or old_completion['report_sha256'] != file_digest(ROOT / reg['original_report'])):
        raise ValueError('Original comparison incomplete/changed')
    parent = ExperimentContract(json.loads((ROOT / old_reg['parent_protocol']).read_text()), ROOT)
    if parent.digest != old_reg['parent_protocol_sha256']:
        raise ValueError('Changed parent protocol')
    with np.load(ROOT / reg['cache'], allow_pickle=False) as a:
        x = a['geometry'].copy()
        data = {'images': torch.tensor(a['rgb'].astype(np.float32) / 255 - .5),
                'mask': torch.tensor(a['mask'].astype(np.float32)),
                'image_xy': torch.tensor(a['image_xy']), 'homography': torch.tensor(a['homography'])}
        rows = json.loads(str(a['rows_json']))
    source = ROOT / old_reg['source_cache']
    if file_digest(source) != old_reg['source_cache_sha256']:
        raise ValueError('Changed original source labels')
    with np.load(source, allow_pickle=False) as a:
        native, scale, y = [a[k].copy() for k in ('native', 'parent_scale', 'start')]
        old_rows = json.loads(str(a['rows_json']))
    if any(r['data_role'] != 'fit' or parent.protocol['assignments'][r['recording_id']] != 'fit'
           or any(r[k] != old_rows[i][k] for k in r) for i, r in enumerate(rows)):
        raise ValueError('Input/label alignment or fit role mismatch')
    data.update({'target_native': torch.from_numpy(native), 'parent_scale': torch.from_numpy(scale),
                 'start': torch.from_numpy(y.astype(np.float32))})
    output, reports = args.output.resolve(), args.report_dir.resolve()
    if not output.is_relative_to(ROOT / 'data/stage_cvpr2027_experiments') or not reports.is_relative_to(ROOT):
        raise ValueError('Workspace paths required; weights/rows remain ignored')
    identity = {'registration_sha256': file_digest(args.registration),
                'torch': str(torch.__version__), 'numpy': np.__version__}
    if args.resume:
        if json.loads((output / 'identity.json').read_text()) != identity:
            raise ValueError('Resume identity changed')
    else:
        if reports.exists():
            raise ValueError('Use new report directory')
        output.mkdir(parents=True, exist_ok=False)
        atomic_json(output / 'identity.json', identity)
    trials, hashes, fresh, cached = [], {}, 0, 0
    started = time.monotonic()
    for old in old_report['trials']:
        if old['arm'] != 'past_rgb':
            continue
        name = f'fold{old["fold"]}_seed{old["seed"]}_past_rgb'
        if (file_digest(old_study / (name + '.pt')) != old['checkpoint_sha256']
                or file_digest(old_study / (name + '.npz')) != old['prediction_sha256']
                or old_completion['checkpoint_hashes'][name] != old['checkpoint_sha256']):
            raise ValueError('Changed original trial')
        train = np.asarray(old['trial_identity']['train_rows'], dtype=int)
        held = np.flatnonzero(np.array([r['fit_fold'] for r in rows]) == old['fold'])
        if (not np.array_equal(train, np.setdiff1d(np.arange(len(rows)), held))
                or {rows[i]['physical_scene'] for i in train} & {rows[i]['physical_scene'] for i in held}):
            raise ValueError('Training/held scene partition mismatch')
        mean, std = x[train].mean(0), np.maximum(x[train].std(0), 1e-6)
        if (not np.array_equal(mean, old['trial_identity']['normalization_mean'])
                or not np.array_equal(std, old['trial_identity']['normalization_std'])):
            raise ValueError('Original training normalization not reproduced')
        geometry = no_camera_geometry(x, mean, std)
        data['geometry'] = torch.from_numpy(geometry)
        trial_id = {'run': identity, 'seed': old['seed'], 'fold': old['fold'], 'arm': 'past_rgb',
                    'feature_treatment': 'zero_normalized_camera_28_32',
                    'original_trial_identity': old['trial_identity']}
        checkpoint, receipt, predictions = output / (name + '.pt'), output / (name + '.json'), output / (name + '.npz')
        if receipt.exists():
            saved = json.loads(receipt.read_text())
            if (saved['identity'] != trial_id or saved['metrics']['checkpoint_sha256'] != file_digest(checkpoint)
                    or saved['metrics']['predictions_sha256'] != file_digest(predictions)):
                raise ValueError('Changed resumed model/result')
            trials.append(saved['metrics'])
            hashes[name] = file_digest(receipt)
            cached += 1
            continue
        prior = float((y[train].sum() + 1) / (len(train) + 2))
        torch.manual_seed(old['seed'])
        model = PastAppearanceProbe(32)
        with torch.no_grad():
            model.start.bias.fill_(float(np.log(prior / (1 - prior))))
        def heartbeat(step, loss, elapsed):
            record = {'state': 'training', 'pid': os.getpid(), 'trial': name,
                      'step': step, 'target_steps': 1000, 'loss': loss, 'fit_seconds': elapsed}
            atomic_json(output / 'heartbeat.json', record)
            print(json.dumps(record), flush=True)
        fit = fit_probe(model, data, train, old_reg['training'], trial_id, checkpoint, heartbeat=heartbeat)
        if not fit['complete']:
            raise ValueError('Incomplete registered fit')
        model.eval()
        p, prob = [], []
        with torch.no_grad():
            for start in range(0, len(held), 32):
                ids = held[start:start + 32]
                delta, logit = model(data['geometry'][ids], data['images'][ids], data['mask'][ids], 'past_rgb')
                p.append(pixel_delta_to_native(delta, data['image_xy'][ids], data['homography'][ids]).numpy())
                prob.append(torch.sigmoid(logit).numpy())
        p, prob = np.concatenate(p), np.concatenate(prob)
        switch = (prob >= old_reg['diagnostic_probability_gate']) & data['mask'][held].bool().all(1).numpy()
        guarded = p * switch[:, None, None]
        groups = [(old_rows[i]['recording_id'], old_rows[i]['agent_id'], old_rows[i]['first_row']) for i in held]
        def score(prediction):
            return forecast_metrics(prediction, native[held], scale[held],
                                    parent.protocol['development_evaluation']['easy_threshold'], groups)
        np.savez(predictions, held=held, prediction=p, probability=prob, guarded=guarded)
        metrics = {'fold': old['fold'], 'seed': old['seed'], 'steps': fit['step'],
                   'train_rows': len(train), 'held_rows': len(held), 'fit_seconds': fit['fit_seconds'],
                   'loss_first': fit['losses'][0], 'loss_last': fit['losses'][-1],
                   'parameters': sum(v.numel() for v in model.parameters()),
                   'original_unrestricted': old['trajectory_unrestricted'],
                   'original_guarded': old['trajectory_fixed_gate'],
                   'unrestricted': score(p), 'guarded': score(guarded),
                   'classification': score_probabilities(y[held], prob, prior=prior),
                   'switch_rate': float(switch.mean()),
                   'checkpoint_sha256': file_digest(checkpoint), 'predictions_sha256': file_digest(predictions)}
        atomic_json(receipt, {'identity': trial_id, 'metrics': metrics})
        trials.append(metrics)
        hashes[name] = file_digest(receipt)
        fresh += 1
        print(json.dumps({'completed': name, 'guarded_gain': metrics['guarded']['gain_vs_cv_pct'],
                          'unrestricted_gain': metrics['unrestricted']['gain_vs_cv_pct']}), flush=True)
    if len(trials) != 6:
        raise ValueError('Missing a registered trial')
    if (output / 'completion.json').exists():
        complete = json.loads((output / 'completion.json').read_text())
        if (complete['identity'] != identity or complete['receipts'] != hashes
                or complete['report_sha256'] != file_digest(reports / 'metrics.json')):
            raise ValueError('Changed original completion report')
        print(json.dumps({'result_source': 'cached_verified', 'fresh_models': fresh, 'cached_models': cached}), flush=True)
        return
    if reports.exists():
        raise ValueError('No report overwrite permitted')
    reports.mkdir(parents=True)
    report = {'result_source': 'fresh_run_matched_retraining_without_learned_camera_inputs', 'identity': identity,
              'completed_models': 6, 'total_updates': sum(t['steps'] for t in trials),
              'fit_seconds_sum': sum(t['fit_seconds'] for t in trials), 'trials': trials,
              'elapsed_seconds': time.monotonic() - started, 'fresh_models': fresh, 'cached_models': cached,
              'new_primary_metric': False, 'model_or_threshold_selection': False,
              'new_deployment': False, 'stage5c_executed': False, 'smc_enabled': False}
    atomic_json(reports / 'metrics.json', report)
    lines = ['# Past-RGB Retraining Without Camera Inputs', '',
             'All six fixed-budget fits, no seed/threshold selection. Positive gain means better than CV.', '',
             '| Held scene | Seed | Original guarded gain % | No-camera guarded gain % | No-camera unrestricted gain % | Switch rate |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for t in trials:
        lines.append(f'| {"ETH" if t["fold"] == 0 else "Hotel"} | {t["seed"]} | '
                     f'{t["original_guarded"]["gain_vs_cv_pct"]:.5f} | {t["guarded"]["gain_vs_cv_pct"]:.5f} | '
                     f'{t["unrestricted"]["gain_vs_cv_pct"]:.5f} | {t["switch_rate"]:.4%} |')
    lines += ['', 'Easy percentage ratios undefined at zero CV floor; absolute harm/native errors in metrics.json.',
              'Fit-only adaptive research, not independent confirmation, metric calibration or deployment.', '']
    (reports / 'results.md').write_text('\n'.join(lines))
    atomic_json(output / 'completion.json', {'identity': identity, 'receipts': hashes,
                'report_sha256': file_digest(reports / 'metrics.json')})
    atomic_json(output / 'heartbeat.json', {'state': 'complete', 'pid': os.getpid(), 'models': 6,
                'elapsed_seconds': time.monotonic() - started})
    print(json.dumps({'complete': True, 'models': 6}), flush=True)


if __name__ == '__main__':
    main()
