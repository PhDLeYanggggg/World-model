"""Fresh real-input invariance check with random Torch weights, not forecasting evidence."""
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
    parser.add_argument('--report-dir', type=Path, default=ROOT / 'outputs/publication_readiness_2026_09/supervised_backend')
    args = parser.parse_args()
    if platform.system() == 'Darwin' and platform.machine() != 'arm64':
        raise SystemExit('Refusing Rosetta before Torch import')
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[name] = '4'
    import numpy as np
    import torch
    from src.data_unification.m3w_causal_recordings import RecordingWindows, PROTOCOL_STEPS
    from src.evaluation.m3w_recording_lineage import sha256
    from src.world_model.m3w_supervised_intervention import PastContextForecaster, collate_inputs, pack_inputs, risk_features
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(927)
    model = PastContextForecaster(width=16, heads=2, layers=1).eval()
    manifest_path = ROOT / 'data/stage_cvpr2027_causal/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    began, checks = time.monotonic(), []

    def no_labels(*args, **kwargs):
        raise AssertionError('Future label API must not be used by this engineering check')

    def forward(scene):
        agents = [a['agent_id'] for a in scene['agents']]
        packed = collate_inputs([pack_inputs(a['inputs'], 'constant_velocity_causal_fd') for a in scene['agents']])
        with torch.no_grad():
            forecast = model(packed)
            features = risk_features(packed, forecast)
        return agents, packed, forecast, features

    for record in manifest['recordings']:
        reader = RecordingWindows(ROOT / 'data/stage_cvpr2027_causal' / record['id'])
        reader.get_labels = reader.get_scene_labels = no_labels
        eligible = reader.index[reader.index['protocol'] == PROTOCOL_STEPS]
        frames = np.unique(reader.points[eligible['current_row'], 0]).astype(int)
        for frame in frames[np.linspace(0, len(frames) - 1, min(3, len(frames)), dtype=int)]:
            source = reader.points
            scene = reader.get_scene_inputs(int(frame), 50)
            if not scene['agents']:
                checks.append({'recording': record['id'], 'frame': int(frame), 'status': 'no_raw50_past_support'})
                continue
            before = forward(scene)
            modified = np.asarray(source).copy()
            modified[modified[:, 0] > frame, 2:] = np.nan
            reader.points = modified
            after = forward(reader.get_scene_inputs(int(frame), 50))
            reader.points = source
            stable = (before[0] == after[0] and all(torch.equal(before[1][k], after[1][k]) for k in before[1])
                      and torch.equal(before[2], after[2]) and torch.equal(before[3], after[3]))
            finite = bool(torch.isfinite(before[2]).all() and torch.isfinite(before[3]).all())
            checks.append({'recording': record['id'], 'frame': int(frame), 'agents': len(before[0]),
                           'status': 'checked', 'future_mutation_invariant': stable, 'finite': finite})
    evaluated = [r for r in checks if r['status'] == 'checked']
    if not evaluated or not all(r['future_mutation_invariant'] and r['finite'] for r in evaluated):
        raise SystemExit('Real input invariance check failed')
    report = {'result_source': 'fresh_run', 'data_source': 'cached_verified_raw_position_cache',
              'scope': 'random_weight_forward_engineering_not_predictive_evaluation',
              'architecture': 'arm64' if platform.machine() == 'arm64' else platform.machine(),
              'torch_version': str(torch.__version__), 'device': 'cpu', 'threads': 4,
              'queries_checked': len(evaluated), 'agents_checked': sum(r['agents'] for r in evaluated),
              'queries_without_raw50_past_support': len(checks) - len(evaluated),
              'manifest_sha256': sha256(manifest_path), 'checks': checks,
              'future_label_api_calls': 0, 'real_supervised_training': False, 'real_accuracy_computed': False,
              'dataset_or_scientific_protocol_selected': False, 'stage5c_executed': False, 'smc_enabled': False,
              'elapsed_seconds': time.monotonic() - began,
              'code_sha256': {p: sha256(ROOT / p) for p in ('scripts/check_m3w_supervised_inputs.py',
                              'src/world_model/m3w_supervised_intervention.py')}}
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / 'real_input_checks.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('queries_checked', 'agents_checked', 'queries_without_raw50_past_support',
                                           'future_label_api_calls', 'real_supervised_training', 'elapsed_seconds')}))


if __name__ == '__main__':
    main()
