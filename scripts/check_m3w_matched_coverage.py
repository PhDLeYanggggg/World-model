"""Exact-count mechanism and causal-input checks, not a predictive comparison."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from itertools import product
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise SystemExit('Use the arm64 PyTorch environment')

import numpy as np
from src.world_model.m3w_joint_intervention import (
    InterventionProblem, compare_at_independent_coverage, select_interventions,
)


def synthetic_checks():
    rng = np.random.default_rng(20260917)
    differences, count_changes, timings, nonzero = [], 0, [], 0
    for _ in range(80):
        n = 7
        edges = np.array([(i, j) for i in range(n) for j in range(i+1, n)])
        cost = rng.uniform(0, 1, (len(edges), 2, 2))
        cost[:, 0, 0] = 0
        p = InterventionProblem(rng.normal(size=n), rng.uniform(0, .3, n), rng.random(n) > .1,
                                edges, cost, .8, .12, 4)
        start = time.perf_counter()
        matched = compare_at_independent_coverage(p)
        timings.append(time.perf_counter()-start)
        ordinary = select_interventions(p, mode='joint')
        count_changes += int(ordinary['switch'].sum() != matched['reference_count'])
        assert matched['matched']
        nonzero += int(matched['nonzero_matched'])
        objectives = []
        for bits in product((0, 1), repeat=n):
            x = np.array(bits)
            if (x.sum() != matched['reference_count'] or np.any(x > p.supported)
                    or np.mean(x*np.maximum(p.expected_harm, -p.expected_gain)) > .12):
                continue
            pair = cost[np.arange(len(edges)), x[edges[:, 0]], x[edges[:, 1]]].mean()
            objectives.append(-np.mean(x*p.expected_gain)+p.pair_weight*pair)
        differences.append(abs(matched['joint_exact']['objective']-min(objectives)))
    assert max(differences) < 1e-8
    return {'source': 'synthetic_predicted_scores_not_real_losses', 'scenes': len(differences), 'agents_each': 7,
            'ordinary_cap_only_count_mismatches': count_changes, 'exact_count_matched_scenes': len(differences),
            'nonzero_matches': nonzero, 'max_exhaustive_objective_error': max(differences),
            'median_pair_solve_seconds': float(np.median(timings)), 'max_pair_solve_seconds': max(timings)}


def real_input_checks():
    import torch
    from src.data_unification.m3w_causal_recordings import RecordingWindows
    from src.data_unification.m3w_citr_recordings import CITRRecordingWindows
    from src.evaluation.m3w_development_evaluation import decide_scene
    from src.world_model.m3w_supervised_intervention import (
        PastContextForecaster, pack_inputs, collate_inputs, risk_features,
    )
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(2718)
    model = PastContextForecaster(width=16, heads=2, layers=1).eval()
    root = ROOT / 'data/stage_cvpr2027_causal'
    catalog = json.loads((root / 'manifest.json').read_text())
    readers = [(RecordingWindows(root / record['id']), 3) for record in catalog['recordings']]
    citr = root / 'citr_diagnostic'
    readers += [(CITRRecordingWindows(directory), 1) for directory in sorted(citr.glob('citr_*')) if directory.is_dir()]
    if not citr.is_dir():
        raise ValueError('Expected the already converted diagnostic CITR cache')
    rows, label_calls, checked_agents = [], 0, 0

    def no_labels(*_):
        nonlocal label_calls
        label_calls += 1
        raise AssertionError('This instrumentation must not open future labels')

    for reader, count in readers:
        reader.get_labels = reader.get_scene_labels = no_labels
        frames = np.unique(reader.frame_values)
        selected = [len(frames)//2] if count == 1 else np.linspace(min(8, len(frames)-1), len(frames)-1, count, dtype=int)
        metadata_hash = hashlib.sha256((reader.directory / 'metadata.json').read_bytes()).hexdigest()
        for index in selected:
            frame = int(frames[index])
            scene = reader.get_scene_inputs(frame, 50)
            row = {'recording': reader.metadata['id'], 'metadata_sha256': metadata_hash,
                   'physical_scene': reader.metadata['physical_scene'], 'frame': frame}
            if not scene['agents']:
                rows.append({**row, 'status': 'not_run_no_raw50_past_support'})
                continue
            baseline = 'constant_velocity_causal_fd'
            inputs = collate_inputs([pack_inputs(a['inputs'], baseline) for a in scene['agents']])
            dim = risk_features(inputs, inputs['baseline']).shape[1]
            head = {'mean': np.zeros(dim), 'scale': np.ones(dim), 'coef': np.zeros((2, dim)),
                    'intercept': np.array([.1, .01])}
            scale = float(np.median([a['coordinate_transform']['scale'] for a in scene['agents']]))
            settings = dict(baseline=baseline, policy={'pair_weight': .1, 'max_mean_predicted_harm': .1,
                            'max_intervention_fraction': .5, 'min_predicted_gain': 0., 'max_agent_predicted_harm': 1.},
                            geometry={'graph_radius': scale, 'proximity_threshold': scale*.1},
                            device='cpu', solver_seconds=2., include_matched_coverage=True)
            before = decide_scene(scene, model, head, **settings)
            original = reader.points
            changed = np.array(original)
            changed[changed[:, 0] > frame, 2:] = np.nan
            reader.points = changed
            after = decide_scene(reader.get_scene_inputs(frame, 50), model, head, **settings)
            reader.points = original
            assert before['agent_ids'] == after['agent_ids']
            for arm in ('reference', 'joint_exact'):
                for key in ('switch', 'prediction'):
                    np.testing.assert_array_equal(before['coverage_match'][arm][key], after['coverage_match'][arm][key])
            match = before['coverage_match']
            assert match['matched'] == after['coverage_match']['matched']
            if match['matched']:
                assert match['reference_count'] == match['joint_count']
            checked_agents += len(scene['agents'])
            rows.append({**row, 'status': 'input_checked', 'agents': len(scene['agents']),
                         'reference_count': match['reference_count'], 'joint_count': match['joint_count'],
                         'matched': match['matched'], 'nonzero_matched': match['nonzero_matched'],
                         'matching_status': match['status'], 'future_invariant': True})
    return {'source': 'cached_verified_positions_fresh_random_weight_input_check',
            'queries_checked': sum(r['status'] == 'input_checked' for r in rows), 'agents_checked': checked_agents,
            'unsupported_queries': sum(r['status'] != 'input_checked' for r in rows),
            'matched_queries': sum(r.get('matched', False) for r in rows),
            'nonzero_matched_queries': sum(r.get('nonzero_matched', False) for r in rows),
            'solver_or_feasibility_unmatched_queries': sum(r.get('matched') is False for r in rows),
            'future_label_api_calls': label_calls, 'checks': rows,
            'real_model_trained': False, 'real_accuracy_evaluated': False,
            'all_parameters_are_engineering_instrumentation_not_approved_scientific_rules': True,
            'runtime': {'device': 'cpu', 'torch_version': str(torch.__version__), 'threads': 4, 'workers': 0}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir', type=Path, default=ROOT / 'outputs/publication_readiness_2026_09/matched_coverage')
    args = parser.parse_args()
    if not args.report_dir.resolve().is_relative_to(ROOT):
        raise SystemExit('Report must remain inside the workspace')
    began = time.perf_counter()
    report = {'result_source': 'fresh_run_engineering_checks', 'generated_at_utc': datetime.now(timezone.utc).isoformat(),
              'synthetic': synthetic_checks(), 'real_inputs': real_input_checks(),
              'formal_protocol_changed': False, 'real_predictive_improvement': 'not_run',
              'matched_control_is_not_deployment': True, 'stage5c_executed': False, 'smc_enabled': False}
    paths = ['scripts/check_m3w_matched_coverage.py', 'src/world_model/m3w_joint_intervention.py',
             'src/evaluation/m3w_development_evaluation.py', 'src/world_model/m3w_supervised_intervention.py',
             'src/data_unification/m3w_causal_recordings.py', 'src/data_unification/m3w_citr_recordings.py',
             'configs/m3w_independent_experiment.draft.json']
    report['sha256'] = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}
    report['elapsed_seconds'] = time.perf_counter()-began
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir/'checks.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'synthetic': report['synthetic'], 'real_inputs': {k:v for k,v in report['real_inputs'].items() if k!='checks'},
                      'elapsed_seconds': report['elapsed_seconds']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
