from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import hashlib
import json

from src.evaluation.m3w_forecast_supplement import score_raw_prefix, summarize_supplement
from src.evaluation.m3w_development_evaluation import score_scene
from scripts.evaluate_m3w_forecast_supplement import verify_completed_evaluation


def example():
    offsets = np.array([10., 20., 30., 40., 50., 60.])
    scene = {'recording_id': 'a', 'physical_scene': 'one_site', 'frame_id': 100, 'horizon_raw': 60,
        'agents': [{'agent_id': 1, 'inputs': {'prediction_frame_offsets': offsets},
                    'coordinate_transform': {'origin_xy': np.zeros(2), 'rotation': np.eye(2), 'scale': 2.}}]}
    prediction = np.zeros((1, 6, 2))
    arm = {'prediction': prediction, 'switch': np.array([False]), 'mean_pair_proxy': 0.,
           'reason': 'floor', 'predicted_constraints_satisfied': True}
    decision = {'agent_ids': [1], 'baseline': prediction,
                'arms': {name: deepcopy(arm) for name in ('floor', 'independent_count_reference', 'joint_exact_count')}}
    labels = [{'agent_id': 1, 'future_frame_ids': 100+offsets, 'future_xy_dataset_local': np.ones((6, 2)),
               'future_label_mask': np.array([True]*5+[False])}]
    return scene, decision, labels


def test_exact_prefix_keeps_missing_full_path_agent_without_changing_inference():
    s, d, y = example()
    before = deepcopy(s)
    full = score_scene(s, d, y, label_policy='complete_requested_path')
    short = score_raw_prefix(s, d, y, raw_horizon=50)
    assert full[0]['baseline_ade'] is None
    assert short[0]['baseline_ade'] == pytest.approx(np.sqrt(2)/2)
    assert short[0]['baseline_fde'] == pytest.approx(np.sqrt(2)/2)
    assert short[0]['full_inference_horizon_raw'] == 60
    assert short[0]['scale'] == 2.
    np.testing.assert_array_equal(s['agents'][0]['inputs']['prediction_frame_offsets'], before['agents'][0]['inputs']['prediction_frame_offsets'])
    assert d['baseline'].shape == (1, 6, 2)


def test_not_on_native_grid_is_not_interpolated_and_missing_endpoint_is_not_replaced():
    s, d, y = example()
    assert score_raw_prefix(s, d, y, raw_horizon=25) is None
    y[0]['future_label_mask'][4] = False
    r = score_raw_prefix(s, d, y, raw_horizon=50)[0]
    assert r['baseline_ade'] is None and r['baseline_fde'] is None


def test_prefix_slice_does_not_reclassify_missing_full_target_as_easy():
    s, d, y = example()
    full = score_scene(s, d, y, label_policy='complete_requested_path')
    short = score_raw_prefix(s, d, y, raw_horizon=50)
    contract = SimpleNamespace(protocol={'development_evaluation': {'easy_threshold': 1., 'hard_threshold': 2., 'bootstrap_seed': 17},
        'task': {'aggregation': 'equal_physical_scene', 'primary_metric': 'ade'}, 'bootstrap_resamples': 2000})
    query = {'recording_id': 'a', 'frame_id': 100, 'horizon_raw': 60, 'status': 'matched_zero_not_evidence_of_coupling',
             'matched': True, 'nonzero_matched': False}
    result = summarize_supplement(full, short, [query], contract)
    assert result['raw50_prefix']['unknown_full_path_slice_agent_queries'] == 1
    assert result['raw50_prefix']['arms']['floor']['full_path_defined_easy']['ade']['count'] == 0
    assert result['matched_control_comparisons']['matched_nonzero_only']['ade']['count'] == 0
    assert result['zero_count_matches_are_not_coupling_evidence']


def test_primary_completion_requires_unchanged_outputs_and_identity(tmp_path):
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    identity = {'device': 'cpu'}
    (tmp_path / 'run_identity.json').write_text(json.dumps(identity))
    hashes = {}
    for name in ('development_report.json', 'selected_policy.json', 'artifact.json'):
        (tmp_path / name).write_text('{}')
        hashes[name] = digest(tmp_path / name)
    receipt = {'artifacts': hashes, 'run_sha256': hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()}
    (tmp_path / 'completion.json').write_text(json.dumps(receipt))
    verify_completed_evaluation(tmp_path, digest)
    (tmp_path / 'run_identity.json').write_text(json.dumps({'device': 'mps'}))
    with pytest.raises(ValueError, match='identity mismatch'):
        verify_completed_evaluation(tmp_path, digest)
    (tmp_path / 'run_identity.json').write_text(json.dumps(identity))
    (tmp_path / 'selected_policy.json').write_text('{"changed": true}')
    with pytest.raises(ValueError, match='result changed'):
        verify_completed_evaluation(tmp_path, digest)
