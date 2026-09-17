"""Label-using diagnostic ceilings for frozen candidate selection and scaling."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts.build_m3w_observed_motion import load_registration
from src.evaluation.m3w_experiment_contract import file_digest
from src.evaluation.m3w_candidate_headroom import segment_oracle, budget_envelope
from src.world_model.m3w_offline_visual_data import json_write


def aggregate(loss, reference, folds, repeats):
    loss, reference = np.asarray(loss), np.asarray(reference)
    means = np.stack([loss[:, folds == f].mean(1) for f in range(3)], 1)
    ref = np.array([reference[folds == f].mean() for f in range(3)])
    draws = np.random.default_rng(917).integers(0, 3, size=(repeats, 3))
    gains = 100*(1-means.mean(0)[draws].mean(1)/ref[draws].mean(1))
    return dict(gain_percent=float(100*(1-means.mean()/ref.mean())),
        per_seed_gain_percent=(100*(1-means.mean(1)/ref.mean())).tolist(),
        per_scene_gain_percent=(100*(1-means.mean(0)/ref)).tolist(),
        descriptive_scene_ci95_percent=np.quantile(gains, [.025, .975]).tolist(),
        physical_scene_clusters=3, bootstrap_resamples=repeats,
        method_confidence_interval=False, independent_confirmation=False)


def support_summary(feasible, lower, reference, folds, tracks, mask):
    present = [f for f in range(3) if np.any(mask & (folds == f))]
    if not present:
        return dict(rows=0)
    base = np.mean([reference[mask & (folds == f)].mean() for f in present])
    value = np.mean([feasible[mask & (folds == f)].mean() for f in present])
    optimistic = np.mean([lower[mask & (folds == f)].mean() for f in present])
    return dict(rows=int(mask.sum()), recording_local_tracks=len(np.unique(tracks[mask])),
        supported_scenes=len(present), reference_ADE=float(base), feasible_oracle_ADE=float(value),
        numerical_lower_ADE=float(optimistic),
        feasible_gain_percent=float(100*(1-value/base)) if base > 0 else None,
        optimistic_gain_percent=float(100*(1-optimistic/base)) if base > 0 else None,
        absolute_harm=float(value-base))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    reg = json.loads(args.registration.read_text())
    if reg['role'] != 'fit_only_posthoc_oracle_diagnostic':
        raise ValueError('This evaluator cannot train or deploy a policy')
    for path, digest in reg['bindings'].items():
        if file_digest(ROOT/path) != digest:
            raise ValueError('Changed dependency: '+path)
    control, parent, contract, source, source_manifest = load_registration(ROOT/reg['control_registration'])
    frozen_path = ROOT/reg['frozen_report']
    frozen = json.loads(frozen_path.read_text())
    replay = json.loads((ROOT/reg['frozen_replay']).read_text())
    if (not frozen['complete'] or len(frozen['trials']) != 72 or len(replay['trials']) != 72
            or not all(t['prediction_exact'] for t in replay['trials'])
            or replay['identity'] != frozen['identity']
            or file_digest(source/'data_manifest.json') != frozen['identity']['source_identity']['source_manifest_sha256']
            or contract.digest != frozen['identity']['source_identity']['parent_protocol_sha256']):
        raise ValueError('Frozen source/fold/replay lineage differs')
    a = {n:np.load(source/(n+'.npy'), mmap_mode='r') for n in ('geometry', 'targets', 'baselines', 'folds')}
    folds = a['folds']
    rows = json.loads((source/'rows.json').read_text())
    tracks = np.array([f"{r['recording']}:{r['agent']}" for r in rows])
    if source_manifest['rows'] != 11966 or not np.array_equal(folds, [r['fold'] for r in rows]):
        raise ValueError('Cohort changed')
    cv = source_manifest['baseline_names'].index('constant_velocity_causal_fd')
    base, target = a['baselines'][:, cv].astype(float), a['targets'].astype(float)
    reference = np.linalg.norm(base-target, axis=-1).mean(1)
    output, reports = ROOT/reg['output'], ROOT/reg['reports']
    if not output.is_relative_to(ROOT/'data/stage_cvpr2027_experiments'):
        raise ValueError('Private per-row diagnostics required')
    identity = dict(registration_sha256=file_digest(args.registration), frozen_report_sha256=file_digest(frozen_path),
        source_manifest_sha256=file_digest(source/'data_manifest.json'))
    conditions = [v+'_'+m for v in ('quality_control', 'directed') for m in
                  ('row_uniform', 'scene_uniform', 'scene_track', 'scene_event_track')]
    expected = {(c, s, f) for c in conditions for s in reg['seeds'] for f in range(3)}
    actual = {(t['variant']+'_'+t['mode'], t['seed'], t['fold']) for t in frozen['trials']}
    if actual != expected or len(actual) != len(frozen['trials']):
        raise ValueError('Fixed candidate set changed')
    losses = {c:{k:np.empty((3, len(rows))) for k in ('binary_loss', 'feasible_loss', 'lower_loss')} for c in conditions}
    receipts, replayed, fresh, maximum_gap = [], [], 0, 0.
    started = time.monotonic()
    for t in frozen['trials']:
        pred_path, cp = ROOT/t['prediction_path'], ROOT/t['checkpoint_path']
        if file_digest(pred_path) != t['prediction_sha256'] or file_digest(cp) != t['checkpoint_sha256']:
            raise ValueError('Frozen checkpoint/prediction changed')
        with np.load(pred_path) as pred:
            held, prediction = pred['held_indices'], pred['prediction'].copy()
        if not np.array_equal(held, np.flatnonzero(folds == t['fold'])):
            raise ValueError('Prediction row alignment differs')
        key = t['trial']
        path, rp = output/(key+'.npz'), output/(key+'.json')
        trial_identity = dict(identity, trial=key, prediction_sha256=t['prediction_sha256'])
        saved = json.loads(rp.read_text()) if rp.exists() else None
        if saved and (saved['identity'] != trial_identity or file_digest(path) != saved['sha256']):
            raise ValueError('Changed diagnostic cache')
        if saved and not args.verify:
            with np.load(path) as old:
                values = {k:old[k].copy() for k in old.files}
        else:
            values = segment_oracle(base[held], prediction, target[held], iterations=reg['iterations'])
            values['held_indices'] = held
            if args.verify:
                if not saved:
                    raise ValueError('Missing original receipt')
                with np.load(path) as old:
                    if set(old.files) != set(values) or not all(np.array_equal(old[k], v) for k, v in values.items()):
                        raise ValueError('Recomputed oracle differs')
                replayed.append(dict(trial=key, exact=True))
            else:
                output.mkdir(parents=True, exist_ok=True)
                np.savez(path, **values)
                saved = dict(identity=trial_identity, sha256=file_digest(path), result_source='fresh_run_oracle_computation',
                             rows=len(held), input_result_source='cached_verified_frozen_forecasts')
                json_write(rp, saved)
                fresh += 1
        if not np.allclose(values['reference_loss'], reference[held], rtol=0, atol=1e-12):
            raise ValueError('Baseline reference differs')
        maximum_gap = max(maximum_gap, float(np.max(values['feasible_loss']-values['lower_loss'])))
        c = t['variant']+'_'+t['mode']; seed = reg['seeds'].index(t['seed'])
        for name in losses[c]:
            losses[c][name][seed, held] = values[name]
        receipts.append(saved)
        print(json.dumps(dict(trial=key, processed=len(receipts), fresh=fresh, verified=len(replayed))), flush=True)
    if args.verify:
        json_write(reports/'replay.json', dict(identity=identity, recomputed=len(replayed), trials=replayed,
            maximum_numerical_loss_interval=maximum_gap, model_training_run=False, result_source='fresh_run_exact_recomputation'))
        return
    all_conditions = dict(losses)
    all_conditions['pooled_eight_candidates_per_seed'] = {k:np.minimum.reduce([losses[c][k] for c in conditions])
                                                         for k in ('binary_loss', 'feasible_loss', 'lower_loss')}
    summary, budgets, subsets = {}, {}, {}
    static = np.all(a['geometry'][:, :16] == 0, axis=1)
    masks = dict(all=np.ones(len(rows), bool), static_past=static,
                 static_stays=static & np.all(target == 0, axis=(1, 2)),
                 static_moves=static & ~np.all(target == 0, axis=(1, 2)),
                 moving_past=~static, easy=reference <= parent['easy_threshold'])
    for key, data in all_conditions.items():
        summary[key] = {k:aggregate(v, reference, folds, reg['bootstrap_resamples']) for k, v in data.items()}
        subsets[key] = {name:[support_summary(data['feasible_loss'][i], data['lower_loss'][i], reference, folds, tracks, mask)
                             for i in range(3)] for name, mask in masks.items()}
        budgets[key] = [[budget_envelope(reference, data['feasible_loss'][i], data['lower_loss'][i], folds, fraction)
                        for fraction in reg['coverage_fractions']] for i in range(3)]
    result = dict(identity=identity, complete=True, rows=len(rows), conditions=summary, budget_curves=budgets,
        subsets=subsets, receipts=receipts, seconds=time.monotonic()-started,
        maximum_numerical_loss_interval=maximum_gap, input_result_source='cached_verified_frozen_forecasts',
        result_source='fresh_run_posthoc_oracle_diagnostic', alpha_uses_future_labels=True,
        oracle_is_model_result=False, learned_policy_trained=False, deployment=False,
        oracle_class='union_of_B_to_N_line_segments_one_alpha_per_entire_path',
        excludes_arbitrary_mixtures_new_predictions_and_per_waypoint_scaling=True,
        development_calibration_confirmation_opened=False, stage5c_executed=False, smc_enabled=False)
    destination = reports/'report.json'
    if destination.exists() and fresh == 0:
        original = json.loads(destination.read_text())
        result['seconds'] = original['seconds']
        if result != original:
            raise ValueError('Cached report no longer reproduces')
        print('Completed resume: 72 receipts verified; no new oracle minimizations or model updates')
        return
    json_write(destination, result)
    lines = ['# Frozen Candidate Action-Class Ceiling', '',
        '**Uses future labels. This is not an inference policy, learned improvement or independent confirmation.**', '',
        '| Candidate condition | Binary oracle gain (%) | Whole-path scaling oracle gain (%) | Numerical optimistic gain (%) |',
        '| --- | ---: | ---: | ---: |']
    for key, item in summary.items():
        lines.append(f'| {key} | {item["binary_loss"]["gain_percent"]:.6f} | {item["feasible_loss"]["gain_percent"]:.6f} | {item["lower_loss"]["gain_percent"]:.6f} |')
    lines += ['', 'The pooled diagnostic may choose one of eight fixed candidates per row and seed, then one scalar in [0,1] for the entire path.',
        'It does not bound arbitrary convex mixtures, per-waypoint correction, new predictors or world models.',
        'Budget curves ignore joint constraints and use future outcomes. They are optimistic relaxations, not safety certificates.', '']
    (reports/'report.md').write_text('\n'.join(lines))
    print(json.dumps(dict(complete=True, fresh_diagnostics=fresh, numerical_gap=maximum_gap, seconds=result['seconds'])))


if __name__ == '__main__':
    main()
