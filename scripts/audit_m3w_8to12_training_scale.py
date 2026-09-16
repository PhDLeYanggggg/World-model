"""Measure fit-only normalization pathologies without changing the frozen study."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.data_unification.m3w_causal_recordings import causal_coordinate_transform
from src.evaluation.m3w_experiment_contract import ExperimentContract, file_digest


def energy_summary(rows):
    rows = np.asarray(rows, dtype=float)
    if rows.ndim != 2 or rows.shape[1] != 3 or not len(rows) or not np.isfinite(rows).all() or (rows < 0).any():
        raise ValueError('Finite nonnegative scale/target-energy/path rows required')
    energy = rows[:, 1]
    top = np.argsort(energy)[-max(1, len(energy) // 100):]
    return {'rows': len(rows), 'scale_quantiles': np.quantile(rows[:, 0], [0, .01, .5, .99, 1]).tolist(),
            'target_square_quantiles': np.quantile(energy, [0, .5, .95, .99, 1]).tolist(),
            'top_approximately_one_percent_rows': len(top),
            'top_energy_share': float(energy[top].sum() / energy.sum()) if energy.sum() > 0 else None,
            'scale_floor_rows': int((rows[:, 0] <= .001).sum()),
            'zero_history_path_rows': int((rows[:, 2] == 0).sum())}


def main():
    protocol = ROOT / 'configs/m3w_8to12_development_v1.json'
    contract = ExperimentContract(json.loads(protocol.read_text()), ROOT)
    recordings = {}
    for name, role in contract.protocol['assignments'].items():
        if role != 'fit':
            continue
        reader, ids = contract.open_recording(name, purpose='fit')
        rows = []
        for i in ids:
            index = reader.index[i]
            history = reader.points[int(index['history_start']):int(index['current_row']) + 1]
            t = causal_coordinate_transform(history[:, 2:4], history[:, 0], int(index['horizon_raw']))
            target = reader.get_labels(int(i))['future_xy_dataset_local']
            y = (target - t['origin_xy']) @ t['rotation'] / t['scale']
            rows.append([t['scale'], float(np.square(y).mean()),
                         float(np.linalg.norm(np.diff(history[:, 2:4], axis=0), axis=1).sum())])
        recordings[name] = energy_summary(rows)
    report = {'result_source': 'fresh_run_fit_only_label_diagnostic', 'protocol_sha256': contract.digest,
              'recordings': recordings, 'development_or_confirmation_labels_used': False,
              'frozen_training_or_metric_changed': False,
              'interpretation': 'target energy concentration; not measured per-example gradient contribution',
              'hypothesis': 'Near-stationary past scale can amplify start-moving labels and dominate squared loss',
              'proposed_next_ablation': 'robust coordinate loss versus MSE on unchanged data/metric; investigate scale floor separately',
              'source_sha256': file_digest(Path(__file__))}
    output = ROOT / 'outputs/publication_readiness_2026_09/8to12_development_v1'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'training_scale_audit.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    lines = ['# Fit-Only Training-Scale Audit', '',
             '`fresh_run`; no development/confirmation labels, training changes or sample deletion.', '',
             '| Fit recording | Rows | At scale floor | Largest ~1% target-energy share | Max target square |',
             '| --- | ---: | ---: | ---: | ---: |']
    for name, r in recordings.items():
        share = 'NA' if r['top_energy_share'] is None else f"{100*r['top_energy_share']:.2f}%"
        lines.append(f"| {name} | {r['rows']} | {r['scale_floor_rows']} | {share} | {r['target_square_quantiles'][-1]:.2f} |")
    lines.extend(['', 'The past-derived scale is max(history path length, last speed times horizon, 0.001).',
                  'Stationary-to-moving cases can therefore have extremely large normalized targets.',
                  'This is a plausible explanation for volatile squared-loss fitting, not proof of the full failure mechanism.',
                  'It measures target energy, not actual per-example gradient or realized prediction error.',
                  'Do not discard these cases or renormalize test errors after seeing scores.',
                  'First compare a registered robust-loss fit on the same inputs, then audit a train-derived scale floor separately.', ''])
    (output / 'training_scale_audit.md').write_text('\n'.join(lines))
    print(json.dumps(report))


if __name__ == '__main__':
    main()
