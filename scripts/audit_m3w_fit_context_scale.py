"""Diagnose fit-only input magnitude; never refit or change a frozen transform."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.data_unification.m3w_causal_recordings import FEATURE_NAMES
from src.evaluation.m3w_experiment_contract import ExperimentContract, file_digest


def context_magnitudes(inputs):
    history = np.asarray(inputs['history_xy'])
    neighbors = np.asarray(inputs['neighbor_xy'])
    hm, nm = np.asarray(inputs['history_mask']), np.asarray(inputs['neighbor_mask'])
    ht, nt = np.asarray(inputs['history_frame_offsets']), np.asarray(inputs['neighbor_frame_offsets'])
    if (hm.dtype != bool or nm.dtype != bool or not hm.all()
            or history.shape != (len(hm), 2) or neighbors.shape != (*nm.shape, 2)
            or ht.shape != hm.shape or nt.shape != nm.shape
            or not np.isfinite(history[hm]).all() or not np.isfinite(neighbors[nm]).all()
            or not np.isfinite(ht[hm]).all() or not np.isfinite(nt[nm]).all()
            or (ht[hm] > 0).any() or (nt[nm] > 0).any()):
        raise ValueError('Finite past-only, aligned context required')
    scale = float(inputs['causal_features'][FEATURE_NAMES.index('normalization_scale')])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Positive past-only scale required')
    eligible = nm.all(-1) & np.isclose(nt, ht[None], atol=1e-6, rtol=1e-6).all(-1)
    observed = neighbors[eligible]
    maximum = float(np.linalg.norm(observed, axis=-1).max()) if len(observed) else 0.
    return np.array([scale, np.linalg.norm(history, axis=-1).max(), maximum, eligible.sum()], float)


def summarize(rows):
    rows = np.asarray(rows, float)
    if rows.ndim != 2 or rows.shape[1] != 4 or not len(rows) or not np.isfinite(rows).all():
        raise ValueError('Nonempty finite context rows required')
    floor = rows[:, 0] <= np.float32(.001)
    return {
        'rows': len(rows), 'scale_floor_rows': int(floor.sum()),
        'scale_quantiles_p0_p50_p95_p99_p100': np.quantile(rows[:, 0], [0, .5, .95, .99, 1]).tolist(),
        'ego_norm_quantiles_p50_p95_p99_p100': np.quantile(rows[:, 1], [.5, .95, .99, 1]).tolist(),
        'neighbor_norm_quantiles_p50_p95_p99_p100': np.quantile(rows[:, 2], [.5, .95, .99, 1]).tolist(),
        'no_complete_aligned_neighbors': int((rows[:, 3] == 0).sum()),
        'neighbor_magnitude_counts': {str(limit): int((rows[:, 2] > limit).sum()) for limit in (10, 100, 1000)},
        'floor_and_neighbor_above_100': int((floor & (rows[:, 2] > 100)).sum()),
        'not_floor_and_neighbor_above_100': int((~floor & (rows[:, 2] > 100)).sum()),
    }


def audit_fit_only(contract):
    reports = {}
    for recording, role in sorted(contract.protocol['assignments'].items()):
        if role != 'fit':
            continue
        reader, indices = contract.open_recording(recording, purpose='fit')
        # get_inputs is the only row-access API: no labels or model forward pass.
        reports[recording] = summarize([context_magnitudes(reader.get_inputs(int(i))) for i in indices])
    if not reports:
        raise ValueError('No fit recordings')
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=ROOT / 'configs/m3w_8to12_continuous_context_v5.json')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/publication_readiness_2026_09/8to12_public_predictors_v5')
    args = parser.parse_args()
    contract = ExperimentContract(json.loads(args.protocol.read_text()), ROOT)
    report = {'result_source': 'fresh_run_fit_only_context_magnitude_audit',
        'protocol_sha256': contract.digest, 'source_sha256': file_digest(Path(__file__)),
        'recordings': audit_fit_only(contract), 'future_label_calls': 0,
        'development_or_confirmation_data_used': False, 'frozen_training_or_metric_changed': False,
        'population': 'registered_supervised_fit_windows_only_not_full_online_observation_population',
        'interpretation': 'input conditioning hypothesis; not measured gradients or predictor accuracy',
        'next_test_if_prediction_fails': 'separate training-input scale from the unchanged evaluation scale in a new registered ablation'}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'fit_context_scale.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    lines = ['# Fit-Only Context Magnitude', '',
        'No labels, development rows, model updates or changes to the active protocol.',
        'Counts use complete supervised fit windows and the registered aligned-neighbor rule.', '',
        '| Recording | Rows | Scale floor | Ego max norm | Neighbor p99 | Neighbor max | Rows neighbor >100 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name, r in report['recordings'].items():
        lines.append(f"| {name} | {r['rows']} | {r['scale_floor_rows']} | {r['ego_norm_quantiles_p50_p95_p99_p100'][-1]:.3f} | "
            f"{r['neighbor_norm_quantiles_p50_p95_p99_p100'][-2]:.3f} | {r['neighbor_norm_quantiles_p50_p95_p99_p100'][-1]:.3f} | {r['neighbor_magnitude_counts']['100']} |")
    lines += ['', 'An ego-derived scale can keep ego history near unit magnitude while placing other agents far outside that range.',
        'This is an input-conditioning hypothesis, not proof that neighbors cause the final error or should be removed.',
        'Changing input conditioning must preserve the frozen evaluation scale and baseline comparison, and be separately tested.',
        'No metric/seconds claim, new deployment, independent confirmation, Stage5C or SMC execution.', '']
    (args.output_dir / 'fit_context_scale.md').write_text('\n'.join(lines))
    print(json.dumps(report))


if __name__ == '__main__':
    main()
