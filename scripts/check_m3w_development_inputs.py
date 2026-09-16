"""Real past-input checks for the five-arm evaluator, without real labels or scores.

Random forecast weights and constant synthetic costs are instrumentation only.
They neither train a policy nor approve a scientific evaluation protocol.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise SystemExit('Use arm64 .venv-pytorch')

import numpy as np
from src.data_unification.m3w_causal_recordings import RecordingWindows
from src.world_model.m3w_supervised_intervention import PastContextForecaster, pack_inputs, collate_inputs, risk_features
from src.evaluation.m3w_development_evaluation import decide_scene
import torch


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(927)
    model = PastContextForecaster(width=16, heads=2, layers=1).eval()
    cache = ROOT / 'data/stage_cvpr2027_causal'
    manifest = json.loads((cache / 'manifest.json').read_text())
    checks, agents, label_calls = [], 0, 0
    began = time.monotonic()

    def forbid_labels(*_):
        nonlocal label_calls
        label_calls += 1
        raise AssertionError('Real labels must not be read by this engineering probe')

    for record in manifest['recordings']:
        reader = RecordingWindows(cache / record['id'])
        reader.get_labels = reader.get_scene_labels = forbid_labels
        frames = np.unique(reader.frame_values)
        for index in np.linspace(min(8, len(frames)-1), len(frames)-1, 3, dtype=int):
            frame = int(frames[index])
            scene = reader.get_scene_inputs(frame, 50)
            if not scene['agents']:
                checks.append({'recording': record['id'], 'frame': frame, 'status': 'no_raw50_past_support'})
                continue
            baseline = 'constant_velocity_causal_fd'
            inputs = collate_inputs([pack_inputs(a['inputs'], baseline) for a in scene['agents']])
            dim = risk_features(inputs, inputs['baseline']).shape[1]
            head = {'mean': np.zeros(dim), 'scale': np.ones(dim), 'coef': np.zeros((2, dim)),
                    'intercept': np.array([.1, .01])}
            scale = float(np.median([a['coordinate_transform']['scale'] for a in scene['agents']]))
            settings = {'baseline': baseline, 'policy': {'pair_weight': .1, 'max_mean_predicted_harm': .1,
                        'max_intervention_fraction': .5, 'min_predicted_gain': 0., 'max_agent_predicted_harm': 1.},
                        'geometry': {'graph_radius': scale, 'proximity_threshold': scale * .1},
                        'device': 'cpu', 'solver_seconds': 2.}
            before = decide_scene(scene, model, head, **settings)
            original = reader.points
            changed = np.asarray(original).copy()
            changed[changed[:, 0] > frame, 2:] = np.nan
            reader.points = changed
            altered = reader.get_scene_inputs(frame, 50)
            after = decide_scene(altered, model, head, **settings)
            reader.points = original
            assert before['agent_ids'] == after['agent_ids']
            for arm in before['arms']:
                np.testing.assert_array_equal(before['arms'][arm]['switch'], after['arms'][arm]['switch'])
                np.testing.assert_array_equal(before['arms'][arm]['prediction'], after['arms'][arm]['prediction'])
                if arm != 'uncontrolled':
                    assert before['arms'][arm]['predicted_constraints_satisfied']
            agents += len(scene['agents'])
            checks.append({'recording': record['id'], 'frame': frame, 'agents': len(scene['agents']),
                           'status': 'checked', 'all_five_arms_future_invariant': True,
                           'guarded_predicted_constraints_satisfied': True})
    report = {'result_source': 'fresh_run', 'source_data': 'cached_verified_raw_position_cache',
              'scope': 'random_forecaster_constant_synthetic_costs_real_inputs_only',
              'queries_checked': sum(r['status'] == 'checked' for r in checks), 'agents_checked': agents,
              'queries_without_past_support': sum(r['status'] != 'checked' for r in checks),
              'future_label_calls': label_calls, 'checks': checks,
              'real_training': False, 'real_accuracy_evaluated': False, 'real_gain_harm_learned': False,
              'scientific_protocol_approved': False, 'risk_calibrated': False,
              'stage5c_executed': False, 'smc_enabled': False,
              'runtime': {'device': 'cpu', 'torch_version': str(torch.__version__), 'threads': 4, 'workers': 0},
              'code_sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in
                  ('scripts/check_m3w_development_inputs.py', 'src/evaluation/m3w_development_evaluation.py',
                   'src/world_model/m3w_supervised_intervention.py', 'src/world_model/m3w_joint_intervention.py',
                   'src/data_unification/m3w_causal_recordings.py')},
              'elapsed_seconds': time.monotonic() - began}
    directory = ROOT / 'outputs/publication_readiness_2026_09/development_evaluation'
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'real_input_checks.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: report[k] for k in ('queries_checked', 'agents_checked', 'queries_without_past_support', 'future_label_calls', 'elapsed_seconds')}))


if __name__ == '__main__':
    main()
