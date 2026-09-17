"""Fit-only geometric oracle for the frozen output bound, not model evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from src.evaluation.m3w_experiment_contract import ExperimentContract, file_digest


def oracle_errors(baseline, target, radius):
    baseline, target, radius = (np.asarray(x, dtype=np.float64) for x in (baseline, target, radius))
    if (baseline.shape != target.shape or baseline.shape[-1] != 2
            or radius.shape != baseline.shape[:-1] or (radius < 0).any()
            or not all(np.isfinite(x).all() for x in (baseline, target, radius))):
        raise ValueError('Finite aligned trajectories and nonnegative correction radius required')
    error = np.linalg.norm(baseline-target, axis=-1)
    return error.mean(-1), np.maximum(error-radius, 0).mean(-1)


def summarize(baseline, infimum, zero):
    baseline, infimum, zero = np.asarray(baseline), np.asarray(infimum), np.asarray(zero, dtype=bool)
    total = float(baseline.sum())
    return {'rows': len(baseline), 'cv_ade': float(baseline.mean()),
        'ball_oracle_ade_infimum': float(infimum.mean()),
        'optimistic_improvement_upper_bound_pct': 100*(total-float(infimum.sum()))/total if total else None,
        'zero_budget_rows': int(zero.sum()), 'zero_budget_cv_ade_sum': float(baseline[zero].sum()),
        'zero_budget_share_of_cv_error': float(baseline[zero].sum())/total if total else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    contract = ExperimentContract(json.loads(args.protocol.read_text()), ROOT)
    from src.world_model.m3w_supervised_intervention import ContractForecastDataset, collate_forecasts
    from src.world_model.m3w_baseline_relative_forecaster import past_motion_budget
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    recordings = [r for r, role in contract.protocol['assignments'].items() if role == 'fit']
    results = {}
    all_baseline, all_infimum, all_zero = [], [], []
    for recording in recordings:
        dataset = ContractForecastDataset(contract, [recording], purpose='fit', baseline_name='constant_velocity_causal_fd')
        baseline, infimum, zero = [], [], []
        for start in range(0, len(dataset), 128):
            batch = collate_forecasts([dataset[i] for i in range(start, min(start+128, len(dataset)))])
            if not batch['target_mask'].all():
                raise ValueError('Complete registered fit targets required')
            budget = past_motion_budget(batch['inputs']).numpy()
            b, c = oracle_errors(batch['inputs']['baseline'].numpy(), batch['target'].numpy(), budget)
            baseline.extend(b.tolist())
            infimum.extend(c.tolist())
            zero.extend((budget.max(-1) == 0).tolist())
        results[recording] = summarize(baseline, infimum, zero)
        all_baseline.extend(baseline)
        all_infimum.extend(infimum)
        all_zero.extend(zero)
    result = {'result_source': 'fresh_run_fit_only_geometric_diagnostic',
        'protocol_sha256': contract.digest, 'code_sha256': file_digest(Path(__file__)),
        'bound_code_sha256': file_digest(ROOT/'src/world_model/m3w_baseline_relative_forecaster.py'),
        'data_role': 'fit', 'per_recording': results,
        'agent_window_descriptive_not_primary_aggregation': summarize(all_baseline, all_infimum, all_zero),
        'targets_used_for_diagnostic_only': True, 'development_labels_opened': False,
        'current_model_or_policy_changed': False, 'learned_prediction': False,
        'independent_confirmation': False, 'new_deployment': False}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/'fit_bound_headroom.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    lines = ['# Fit-Only Motion-Bound Headroom', '',
        'Only registered fit rows. The oracle uses future labels for diagnosis, never as inference inputs.',
        'It chooses an arbitrary correction independently at each requested step. This is an optimistic infimum inside each correction ball, not a trained network or a generalization bound.', '',
        '| Recording | Rows | CV normalized ADE | Oracle ADE infimum | Optimistic headroom % | Zero-budget rows | Zero-budget share of CV error |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name, r in results.items():
        lines.append(f"| {name} | {r['rows']} | {r['cv_ade']:.6g} | {r['ball_oracle_ade_infimum']:.6g} | {r['optimistic_improvement_upper_bound_pct']} | {r['zero_budget_rows']} | {r['zero_budget_share_of_cv_error']} |")
    lines += ['', 'Exactly stationary pasts with a stationary causal baseline have zero correction radius. Any later movement remains scored; these rows are not removed.',
        'The upper bound ignores learnability, shared network parameters and temporal smoothness. Large theoretical headroom does not prove a predictor can capture it. Small headroom does not authorize changing this running experiment.',
        'All values retain the registered past normalization. No physical-distance, seconds, calibrated-safety or deployment claim. Neither the training configuration nor the primary metric changes.', '']
    (args.output_dir/'fit_bound_headroom.md').write_text('\n'.join(lines))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
