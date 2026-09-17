"""Fixed-model counterfactual feature-support diagnostic, never a model search."""
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
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report-dir', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if platform.system() == 'Darwin' and platform.machine() != 'arm64':
        raise SystemExit('Native arm64 interpreter required')
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[name] = '4'
    import numpy as np
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    from src.evaluation.m3w_experiment_contract import ExperimentContract, file_digest
    from src.evaluation.m3w_appearance_support_control import fit_box, project_box
    from src.evaluation.m3w_stationary_scene_context import forecast_metrics
    from src.evaluation.m3w_stationary_start_probe import score_probabilities
    from src.world_model.m3w_past_appearance_probe import PastAppearanceProbe, pixel_delta_to_native
    from scripts.run_m3w_stationary_start_probe import atomic_json

    reg = json.loads(args.registration.read_text())
    for path, digest in reg['bindings'].items():
        if file_digest(ROOT / path) != digest:
            raise ValueError('Changed bound source: ' + path)
    original_reg = json.loads((ROOT / reg['original_registration']).read_text())
    for path, digest in original_reg['bindings'].items():
        if file_digest(ROOT / path) != digest:
            raise ValueError('Changed original bound source: ' + path)
    original = json.loads((ROOT / reg['original_report']).read_text())
    if not original['complete_registered_budget'] or len(original['trials']) != 18:
        raise ValueError('Incomplete original experiment')
    parent = ExperimentContract(json.loads((ROOT / original_reg['parent_protocol']).read_text()), ROOT)
    if parent.digest != original_reg['parent_protocol_sha256']:
        raise ValueError('Changed parent protocol')
    old_study = ROOT / reg['original_study']
    completion = json.loads((old_study / 'completion.json').read_text())
    if completion['report_sha256'] != file_digest(ROOT / reg['original_report']):
        raise ValueError('Changed original completion')
    with np.load(ROOT / reg['cache'], allow_pickle=False) as a:
        x = a['geometry'].copy()
        images = torch.tensor(a['rgb'].astype(np.float32) / 255 - .5)
        mask = torch.tensor(a['mask'].astype(np.float32))
        xy, h = torch.tensor(a['image_xy']), torch.tensor(a['homography'])
        rows = json.loads(str(a['rows_json']))
    if file_digest(ROOT / original_reg['source_cache']) != original_reg['source_cache_sha256']:
        raise ValueError('Changed source labels')
    with np.load(ROOT / original_reg['source_cache'], allow_pickle=False) as a:
        target, scale, label = [a[k].copy() for k in ('native', 'parent_scale', 'start')]
        old_rows = json.loads(str(a['rows_json']))
    if any(row['data_role'] != 'fit' or parent.protocol['assignments'][row['recording_id']] != 'fit'
           or any(row[k] != old_rows[i][k] for k in row) for i, row in enumerate(rows)):
        raise ValueError('Row alignment or fit-role mismatch')
    output, reports = args.output.resolve(), args.report_dir.resolve()
    if not output.is_relative_to(ROOT / 'data/stage_cvpr2027_experiments') or not reports.is_relative_to(ROOT):
        raise ValueError('Workspace paths required; row outputs must remain in ignored data')
    identity = {'registration_sha256': file_digest(args.registration),
                'torch': str(torch.__version__), 'numpy': np.__version__}
    if args.resume:
        if json.loads((output / 'identity.json').read_text()) != identity:
            raise ValueError('Resume identity changed')
    else:
        output.mkdir(parents=True, exist_ok=False)
        if reports.exists():
            raise ValueError('Use a new report directory')
        atomic_json(output / 'identity.json', identity)
    started, fresh, cached = time.monotonic(), 0, 0
    trials, receipts = [], {}
    for trial in original['trials']:
        name = f'fold{trial["fold"]}_seed{trial["seed"]}_{trial["arm"]}'
        checkpoint, old_prediction = old_study / (name + '.pt'), old_study / (name + '.npz')
        if (file_digest(checkpoint) != trial['checkpoint_sha256']
                or file_digest(old_prediction) != trial['prediction_sha256']
                or completion['checkpoint_hashes'][name] != trial['checkpoint_sha256']):
            raise ValueError('Changed checkpoint/predictions')
        receipt, prediction_file = output / (name + '.json'), output / (name + '.npz')
        if receipt.exists():
            record = json.loads(receipt.read_text())
            if record['identity'] != identity or record['prediction_sha256'] != file_digest(prediction_file):
                raise ValueError('Changed partial-completion receipt')
            trials.append(record['metrics'])
            receipts[name] = file_digest(receipt)
            cached += 1
            continue
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if state['identity'] != trial['trial_identity'] or state['step'] != 1000:
            raise ValueError('Checkpoint identity/budget mismatch')
        train = np.asarray(state['identity']['train_rows'], dtype=int)
        with np.load(old_prediction, allow_pickle=False) as a:
            held, old_p, old_prob, old_guard = [a[k].copy() for k in ('held', 'prediction', 'probability', 'guarded')]
        expected_held = np.array([i for i, r in enumerate(rows) if r['fit_fold'] == trial['fold']])
        if (not np.array_equal(held, expected_held) or not np.array_equal(train, np.setdiff1d(np.arange(len(x)), held))
                or {rows[i]['physical_scene'] for i in train} & {rows[i]['physical_scene'] for i in held}):
            raise ValueError('Scene/row partition mismatch')
        lower, upper = fit_box(x[train])
        checked, _ = project_box(x[train], lower, upper, np.arange(32))
        if not np.array_equal(checked, x[train]):
            raise ValueError('Training support treatment changed training features')
        camera, camera_outside = project_box(x, lower, upper, np.arange(28, 32))
        clipped, outside = project_box(x, lower, upper, np.arange(32))
        mean = np.asarray(state['identity']['normalization_mean'])
        std = np.asarray(state['identity']['normalization_std'])
        model = PastAppearanceProbe(32)
        model.load_state_dict(state['model'])
        model.eval()
        prior = float((label[train].sum() + 1) / (len(train) + 2))
        groups = [(old_rows[i]['recording_id'], old_rows[i]['agent_id'], old_rows[i]['first_row']) for i in held]
        def score(p):
            return forecast_metrics(p, target[held], scale[held],
                                    parent.protocol['development_evaluation']['easy_threshold'], groups)
        by_treatment, saved = {}, {'held': held}
        replay_difference = None
        for treatment in reg['treatments']:
            treated = camera if treatment == 'jacobian_box' else clipped if treatment == 'all_feature_box' else x
            geometry = torch.tensor(np.clip((treated - mean) / std, -10, 10).astype(np.float32))
            p, prob = [], []
            with torch.no_grad():
                for start in range(0, len(held), 32):
                    ids = held[start:start + 32]
                    delta, logit = model(geometry[ids], images[ids], mask[ids], trial['arm'])
                    p.append(pixel_delta_to_native(delta, xy[ids], h[ids]).numpy())
                    prob.append(torch.sigmoid(logit).numpy())
            p, prob = np.concatenate(p), np.concatenate(prob)
            switch = (prob >= original_reg['diagnostic_probability_gate']) & mask[held].bool().all(1).numpy()
            if treatment == 'support_fallback':
                switch &= ~outside[held]
            guarded = p * switch[:, None, None]
            if treatment == 'original':
                replay_difference = max(float(np.abs(p - old_p).max()), float(np.abs(prob - old_prob).max()),
                                        float(np.abs(guarded - old_guard).max()))
                if replay_difference > 1e-10:
                    raise ValueError('Original checkpoint replay mismatch')
            by_treatment[treatment] = {
                'unrestricted': score(p), 'guarded': score(guarded),
                'classification': score_probabilities(label[held], prob, prior=prior),
                'switch_rate': float(switch.mean()),
                'input_changed_rows': int(np.any(treated[held] != x[held], axis=1).sum()),
                'guarded_gain_change_vs_original_pp': float(score(guarded)['gain_vs_cv_pct'] - trial['trajectory_fixed_gate']['gain_vs_cv_pct']),
            }
            for key, value in (('prediction', p), ('probability', prob), ('guarded', guarded)):
                saved[treatment + '_' + key] = value
        np.savez(prediction_file, **saved)
        metrics = {'fold': trial['fold'], 'seed': trial['seed'], 'arm': trial['arm'],
                   'held_rows': len(held), 'training_rows': len(train),
                   'inside_training_box_rows': int((~outside[held]).sum()),
                   'outside_camera_box_rows': int(camera_outside[held].sum()),
                   'max_original_replay_difference': replay_difference,
                   'checkpoint_sha256': trial['checkpoint_sha256'], 'treatments': by_treatment}
        atomic_json(receipt, {'identity': identity, 'metrics': metrics, 'prediction_sha256': file_digest(prediction_file)})
        trials.append(metrics)
        receipts[name] = file_digest(receipt)
        fresh += 1
        atomic_json(output / 'heartbeat.json', {'state': 'evaluating_frozen_predictors', 'pid': os.getpid(),
                    'completed': len(trials), 'target': 18, 'elapsed_seconds': time.monotonic() - started})
        print(json.dumps({'completed': name, 'inside_box': metrics['inside_training_box_rows'],
                         'guarded_gains': {k: v['guarded']['gain_vs_cv_pct'] for k, v in by_treatment.items()}}), flush=True)
    if len(trials) != 18:
        raise ValueError('Incomplete registered budget')
    grouped = {}
    for fold in (0, 1):
        grouped[str(fold)] = {}
        for arm in original_reg['arms']:
            group = [t for t in trials if t['fold'] == fold and t['arm'] == arm]
            grouped[str(fold)][arm] = {}
            for treatment in reg['treatments']:
                selected = [t['treatments'][treatment] for t in group]
                grouped[str(fold)][arm][treatment] = {
                    'unrestricted_gain_mean': float(np.mean([t['unrestricted']['gain_vs_cv_pct'] for t in selected])),
                    'guarded_gain_mean': float(np.mean([t['guarded']['gain_vs_cv_pct'] for t in selected])),
                    'guarded_easy_absolute_harm_mean': float(np.mean([t['guarded']['easy_absolute_harm'] for t in selected])),
                    'switch_rate_mean': float(np.mean([t['switch_rate'] for t in selected])),
                }
    if (output / 'completion.json').exists():
        complete = json.loads((output / 'completion.json').read_text())
        if (complete['identity'] != identity or complete['receipts'] != receipts
                or complete['report_sha256'] != file_digest(reports / 'metrics.json')):
            raise ValueError('Changed completed report/receipts')
        print(json.dumps({'result_source': 'cached_verified', 'fresh_evaluations': fresh, 'cached_predictors': cached}), flush=True)
        return
    if reports.exists():
        raise ValueError('Use original partial-output/new-report directory; no overwrite')
    reports.mkdir(parents=True)
    report = {'result_source': 'fresh_run_frozen_model_feature_support_control', 'identity': identity,
              'training_performed': False, 'model_or_threshold_selection': False, 'trials': trials,
              'seed_means': grouped, 'completed_predictors': 18, 'treatments_per_predictor': 4,
              'elapsed_seconds': time.monotonic() - started, 'fresh_predictors': fresh, 'cached_predictors': cached,
              'formal_risk_certificate': False, 'new_deployment': False, 'stage5c_executed': False, 'smc_enabled': False}
    atomic_json(reports / 'metrics.json', report)
    lines = ['# Frozen Appearance Support Controls', '',
             'No new training or model selection. All 18 models and four treatments retained.', '',
             '| Held scene | Input | Treatment | Unrestricted gain % | Guarded gain % | Easy absolute harm | Switch rate |',
             '| --- | --- | --- | ---: | ---: | ---: | ---: |']
    for fold, arms in grouped.items():
        for arm, treatments in arms.items():
            for treatment, v in treatments.items():
                lines.append(f'| {"ETH" if fold == "0" else "Hotel"} | {arm} | {treatment} | '
                             f'{v["unrestricted_gain_mean"]:.5f} | {v["guarded_gain_mean"]:.5f} | '
                             f'{v["guarded_easy_absolute_harm_mean"]:.6f} | {v["switch_rate_mean"]:.4%} |')
    lines += ['', 'Easy percentage ratios are undefined at the zero CV floor. No marginal box certifies joint support.',
              'Native ADE/FDE and all seeds remain in metrics.json. No independent test/scene CI or deployment.', '']
    (reports / 'results.md').write_text('\n'.join(lines))
    atomic_json(output / 'completion.json', {'identity': identity, 'receipts': receipts,
                'report_sha256': file_digest(reports / 'metrics.json')})
    atomic_json(output / 'heartbeat.json', {'state': 'complete', 'pid': os.getpid(),
                'completed_predictors': 18, 'elapsed_seconds': time.monotonic() - started})
    print(json.dumps({'complete': True, 'seed_means': grouped}), flush=True)


if __name__ == '__main__':
    main()
