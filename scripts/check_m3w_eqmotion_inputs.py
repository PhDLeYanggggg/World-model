"""Official-core random-weight input audit; no labels, fitting or accuracy claims."""
from __future__ import annotations

import argparse
import hashlib
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
    parser.add_argument('--device', choices=('cpu', 'mps'), default='cpu')
    args = parser.parse_args()
    if platform.system() == 'Darwin' and platform.machine() != 'arm64':
        raise SystemExit('Use arm64 .venv-pytorch')
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[name] = '4'
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
    import numpy as np
    import torch
    from src.data_unification.m3w_causal_recordings import RecordingWindows
    from src.evaluation.m3w_development_evaluation import scene_requests
    from src.world_model.m3w_supervised_intervention import build_forecaster, pack_inputs, collate_inputs, risk_features
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(927)
    architecture = {'family': 'eqmotion_fixed_head', 'history_steps': 8, 'prediction_steps': 12,
                    'hidden_nf': 16, 'channels': 16, 'layers': 2, 'fixed_head': 0}
    model = build_forecaster(architecture).to(args.device).eval()
    cache = ROOT / 'data/stage_cvpr2027_causal'
    manifest_path = cache / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    task = {'history_steps': 8, 'prediction_unit': 'observation_steps', 'horizon': 12}
    began, calls, checks = time.monotonic(), 0, []

    def no_labels(*_):
        nonlocal calls
        calls += 1
        raise AssertionError('No future label API may be used')

    def forward(scene):
        inputs = collate_inputs([pack_inputs(a['inputs'], 'constant_velocity_causal_fd') for a in scene['agents']])
        inputs = {k: v.to(args.device) for k, v in inputs.items()}
        with torch.no_grad():
            result = model(inputs)
            features = risk_features(inputs, result)
            context = model.prepare_context(inputs)
        return inputs, result, features, context['eligible_neighbors']

    for record in manifest['recordings']:
        reader = RecordingWindows(cache / record['id'])
        reader.get_labels = reader.get_scene_labels = no_labels
        stride = max(1, len(np.unique(reader.frame_values)) // 4)
        for query, scene in enumerate(scene_requests(reader, task, stride=stride)):
            if query >= 3:
                break
            before = forward(scene)
            original = reader.points
            corrupted = np.asarray(original).copy()
            corrupted[corrupted[:, 0] > scene['frame_id'], 2:] = np.nan
            reader.points = corrupted
            after_scene = reader.get_scene_inputs(scene['frame_id'], scene['horizon_raw'], history_steps=8)
            # Match the original requested grid using only the declared past support.
            after_scene['agents'] = [a for a in after_scene['agents'] if len(a['inputs']['prediction_frame_offsets']) == 12]
            try:
                after = forward(after_scene)
            finally:
                reader.points = original
            assert [a['agent_id'] for a in scene['agents']] == [a['agent_id'] for a in after_scene['agents']]
            for key in before[0]:
                torch.testing.assert_close(before[0][key], after[0][key], atol=0, rtol=0)
            for a, b in zip(before[1:], after[1:]):
                torch.testing.assert_close(a, b, atol=0, rtol=0)
            assert torch.isfinite(before[1]).all() and torch.isfinite(before[2]).all()
            checks.append({'recording': record['id'], 'frame': scene['frame_id'],
                           'horizon_raw': scene['horizon_raw'], 'agents': len(scene['agents']),
                           'eligible_neighbor_slots': int(before[3].sum().cpu()),
                           'available_neighbor_slots': int(before[0]['neighbor_mask'].any(-1).sum().cpu()),
                           'future_mutation_invariant': True})
    if not checks:
        raise ValueError('No input requests were checked')
    report = {'result_source': 'fresh_run', 'source_data': 'cached_verified_raw_position_cache',
              'scope': 'random_weights_real_past_inputs_only_not_paper_replication',
              'runtime': {'device': args.device, 'architecture': platform.machine(), 'torch_version': str(torch.__version__),
                          'threads': 4, 'workers': 0, 'implicit_mps_cpu_fallback': False},
              'model_architecture': architecture, 'official_source_identity': model.source_identity,
              'queries_checked': len(checks), 'agents_checked': sum(c['agents'] for c in checks),
              'eligible_neighbor_slots': sum(c['eligible_neighbor_slots'] for c in checks),
              'available_neighbor_slots': sum(c['available_neighbor_slots'] for c in checks),
              'future_label_calls': calls, 'checks': checks, 'real_training': False, 'real_accuracy_evaluated': False,
              'scientific_protocol_approved': False, 'stage5c_executed': False, 'smc_enabled': False,
              'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              'code_sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in
                  ('scripts/check_m3w_eqmotion_inputs.py', 'src/world_model/m3w_eqmotion_adapter.py',
                   'src/world_model/m3w_supervised_intervention.py', 'src/data_unification/m3w_causal_recordings.py',
                   'src/evaluation/m3w_development_evaluation.py')},
              'elapsed_seconds': time.monotonic() - began}
    directory = ROOT / 'outputs/publication_readiness_2026_09/public_baselines'
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f'eqmotion_real_input_checks_{args.device}.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: report[k] for k in ('queries_checked', 'agents_checked', 'eligible_neighbor_slots',
                                            'available_neighbor_slots', 'future_label_calls', 'elapsed_seconds')}))


if __name__ == '__main__':
    main()
