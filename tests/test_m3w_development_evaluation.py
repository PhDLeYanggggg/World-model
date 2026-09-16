from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from test_m3w_supervised_intervention import datasets, training_config
from test_m3w_experiment_contract import approve
from src.evaluation.m3w_experiment_contract import ExperimentContract, file_digest
from src.evaluation.m3w_development_evaluation import (
    scene_requests, decide_scene, score_scene, summarize_rows, choose_development,
    validate_plan, evaluate_development,
)


class BaselineCopy(torch.nn.Module):
    def forward(self, inputs):
        return inputs['baseline'].clone()


def zero_head(scene, baseline):
    from src.world_model.m3w_supervised_intervention import pack_inputs, collate_inputs, risk_features
    x = collate_inputs([pack_inputs(a['inputs'], baseline) for a in scene['agents']])
    dim = risk_features(x, x['baseline']).shape[1]
    return {'mean': np.zeros(dim), 'scale': np.ones(dim), 'coef': np.zeros((2, dim)),
            'intercept': np.array([.2, .01])}


POLICY = {'pair_weight': .1, 'max_mean_predicted_harm': .1,
          'max_intervention_fraction': 1., 'min_predicted_gain': 0., 'max_agent_predicted_harm': 1.}
GEOMETRY = {'graph_radius': 5., 'proximity_threshold': .5}


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_scene_schedule_uses_past_not_complete_target_index(tmp_path):
    contract, train, _ = datasets(tmp_path)
    reader = train.readers[0]
    reader.index = np.empty(0, dtype=reader.index.dtype)
    queries = list(scene_requests(reader, contract.protocol['task'], stride=1))
    assert any(s['frame_id'] == 390 for s in queries)
    last = next(s for s in queries if s['frame_id'] == 390)
    assert len(last['agents']) == 1
    assert not reader.get_scene_labels(last)[0]['future_label_mask'].any()


def test_raw_queries_keep_incompatible_native_grids_separate(tmp_path):
    from src.data_unification.m3w_causal_recordings import RecordingWindows, write_recording
    points = np.array([[t, agent, t * .1, agent] for agent, step in ((1, 10), (2, 5)) for t in range(0, 201, step)])
    write_recording(tmp_path / 'mixed', points, {'id': 'mixed', 'physical_scene': 'mixed'})
    reader = RecordingWindows(tmp_path / 'mixed')
    queries = list(scene_requests(reader, {'history_steps': 8, 'prediction_unit': 'raw_annotation_frames', 'horizon': 50}, stride=1))
    at70 = [s for s in queries if s['frame_id'] == 70]
    assert len(at70) == 2 and all(len(s['agents']) == 1 for s in at70)
    assert {len(s['agents'][0]['inputs']['prediction_frame_offsets']) for s in at70} == {5, 10}
    assert all(any(e['reason'] == 'different_native_request_grid' for e in s['excluded_past_support']) for s in at70)


def test_decision_is_future_invariant_and_uses_all_past_supported_agents(tmp_path):
    contract, train, _ = datasets(tmp_path)
    reader = train.readers[0]
    scene = next(s for s in scene_requests(reader, contract.protocol['task'], stride=1) if s['frame_id'] == 290)
    assert len(scene['agents']) == 2
    model = BaselineCopy()
    head = zero_head(scene, train.baseline_name)
    before = decide_scene(scene, model, head, baseline=train.baseline_name, policy=POLICY,
                          geometry=GEOMETRY, device='cpu', solver_seconds=2.)
    points = np.asarray(reader.points).copy()
    points[points[:, 0] > 290, 2:] = np.nan
    reader.points = points
    after_scene = reader.get_scene_inputs(290, scene['horizon_raw'])
    after = decide_scene(after_scene, model, head, baseline=train.baseline_name, policy=POLICY,
                         geometry=GEOMETRY, device='cpu', solver_seconds=2.)
    for arm in before['arms']:
        np.testing.assert_array_equal(before['arms'][arm]['switch'], after['arms'][arm]['switch'])
        np.testing.assert_array_equal(before['arms'][arm]['prediction'], after['arms'][arm]['prediction'])
    assert before['arms']['uncontrolled']['switch'].sum() == 2


def test_missing_future_agents_remain_in_coverage_but_not_fake_fde(tmp_path):
    contract, train, _ = datasets(tmp_path)
    reader = train.readers[0]
    scene = reader.get_scene_inputs(290, 120)
    result = decide_scene(scene, BaselineCopy(), zero_head(scene, train.baseline_name),
                          baseline=train.baseline_name, policy=POLICY, geometry=GEOMETRY,
                          device='cpu', solver_seconds=2.)
    rows = score_scene(scene, result, reader.get_scene_labels(scene), label_policy='complete_requested_path')
    assert len(rows) == 2 and all(r['arms']['uncontrolled']['switch'] for r in rows)
    assert all(r['baseline_fde'] is None for r in rows)
    assert all(r['baseline_ade'] is None for r in rows)
    assert any(r['available_label_steps'] > 0 for r in rows)
    partial = score_scene(scene, result, reader.get_scene_labels(scene), label_policy='available_steps')
    assert any(r['baseline_ade'] is not None for r in partial)
    assert all(r['baseline_fde'] is None for r in partial)


def test_labels_cannot_reorder_agents_or_change_requested_endpoint(tmp_path):
    _, train, _ = datasets(tmp_path)
    reader = train.readers[0]
    scene = reader.get_scene_inputs(100, 120)
    result = decide_scene(scene, BaselineCopy(), zero_head(scene, train.baseline_name),
                          baseline=train.baseline_name, policy=POLICY, geometry=GEOMETRY,
                          device='cpu', solver_seconds=2.)
    labels = reader.get_scene_labels(scene)
    with pytest.raises(ValueError, match='identity'):
        score_scene(scene, result, labels[::-1], label_policy='available_steps')
    labels[0]['future_frame_ids'][-1] += 10
    with pytest.raises(ValueError, match='grid'):
        score_scene(scene, result, labels, label_policy='available_steps')


def test_nonfinite_candidate_forces_floor_without_removing_agent(tmp_path):
    _, train, _ = datasets(tmp_path)
    scene = train.readers[0].get_scene_inputs(100, 120)

    class InvalidCandidate(BaselineCopy):
        def forward(self, inputs):
            result = super().forward(inputs)
            result[0, 0, 0] = float('nan')
            return result

    result = decide_scene(scene, InvalidCandidate(), zero_head(scene, train.baseline_name),
                          baseline=train.baseline_name, policy=POLICY, geometry=GEOMETRY,
                          device='cpu', solver_seconds=2.)
    assert result['nonfinite_candidate_or_scores_floor_count'] == 1
    assert len(result['agent_ids']) == 2
    for arm in result['arms'].values():
        assert not arm['switch'][0]
        np.testing.assert_array_equal(arm['prediction'][0], result['baseline'][0])


def metric_row(scene, recording, frame, baseline, selected, switch=True):
    return {'physical_scene': scene, 'recording_id': recording, 'frame_id': frame, 'horizon_raw': 50,
            'agent_id': 1, 'scale': 2., 'baseline_ade': baseline, 'baseline_fde': baseline,
            'available_label_steps': 5, 'requested_steps': 5,
            'arms': {'joint': {'ade': selected, 'fde': selected, 'switch': switch, 'pair_proxy': 0.,
                                'reason': 'selected', 'constraints_satisfied': True}}}


def summarize(rows, aggregation='equal_physical_scene', easy=2.):
    return summarize_rows(rows, metric='fde', aggregation=aggregation, error_unit='past_normalized',
                          easy_threshold=easy, hard_threshold=3., bootstrap_resamples=2000, bootstrap_seed=17)


def test_cluster_aggregation_does_not_treat_overlapping_windows_as_scenes():
    rows = [metric_row('a', 'a1', i, 10., 9.) for i in range(100)]
    rows += [metric_row('b', 'b1', 0, 1., 2.)]
    macro, micro = summarize(rows), summarize(rows, 'agent_window')
    assert macro['arms']['joint']['all']['improvement_pct'] == pytest.approx(0.)
    assert micro['arms']['joint']['all']['improvement_pct'] > 9
    assert macro['physical_scene_count'] == 2 and macro['agent_query_count'] == 101
    assert macro['arms']['joint']['all']['bootstrap']['resampling_unit'] == 'physical_scene'
    assert macro['arms']['joint']['all']['bootstrap']['independence_verified'] is False


def test_duplicate_rows_rejected_and_one_cluster_ci_not_fabricated():
    row = metric_row('a', 'a1', 0, 1., .8)
    with pytest.raises(ValueError, match='Duplicate'):
        summarize([row, row])
    result = summarize([row])
    assert result['arms']['joint']['all']['bootstrap']['status'] == 'not_run_insufficient_physical_scenes'


def test_raw_metrics_not_pooled_across_unverified_coordinate_scales():
    rows = [metric_row('a', 'a1', 0, 1., .8), metric_row('b', 'b1', 0, 2., 1.8)]
    result = summarize(rows)
    assert result['dataset_local_metrics']['status'] == 'per_recording_only_no_cross_domain_raw_pool'
    assert result['dataset_local_metrics']['recordings']['a1']['arms']['joint']['fde'] == pytest.approx(1.6)
    with pytest.raises(ValueError, match='normalized'):
        summarize_rows(rows, metric='fde', aggregation='equal_physical_scene', error_unit='dataset_local',
                       easy_threshold=2., hard_threshold=3., bootstrap_resamples=2000, bootstrap_seed=17)


def test_selection_respects_easy_guard_and_retains_baseline_on_failure():
    unsafe = summarize([metric_row('a', 'a1', 0, 10., 2.), metric_row('b', 'b1', 0, 1., 2.)])
    choice = choose_development({'unsafe': unsafe}, eligible_arms=['joint'], easy_degradation_max=.02)
    assert choice['selected']['candidate_id'] is None and choice['selected']['arm'] == 'floor'
    assert 'easy_degradation' in choice['rejections'][0]['reasons']
    safe = summarize([metric_row('a', 'a1', 0, 10., 5.), metric_row('b', 'b1', 0, 1., 1.)])
    assert choose_development({'safe': safe}, eligible_arms=['joint'], easy_degradation_max=.02)['selected']['candidate_id'] == 'safe'


def test_selection_requires_matched_sample_identity_and_baseline_floor():
    a = summarize([metric_row('a', 'a1', 0, 1., .9)])
    b = summarize([metric_row('a', 'a1', 1, 1., .8)])
    with pytest.raises(ValueError, match='matched'):
        choose_development({'a': a, 'b': b}, eligible_arms=['joint'], easy_degradation_max=.02)


def test_unapproved_cli_refuses_before_evaluation():
    import subprocess
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([str(root / '.venv-pytorch/bin/python'),
                             str(root / 'scripts/evaluate_m3w_development.py'), '--preflight-only'],
                            cwd=root, text=True, capture_output=True, timeout=30)
    assert result.returncode == 2 and 'approval' in result.stdout
    assert json.loads(result.stdout)['development_evaluated'] is False


def fitted_fixture(tmp_path):
    from src.world_model.m3w_supervised_intervention import (
        ContractForecastDataset, train_forecaster, load_verified_forecaster,
        make_oof_cost_rows, fit_linear_gain_harm,
    )
    initial, _, _ = datasets(tmp_path)
    protocol = deepcopy(initial.protocol)
    protocol['development_evaluation'] = {
        'error_unit': 'past_normalized', 'label_policy': 'complete_requested_path',
        'query_stride': 2, 'geometry_by_recording': {'d': GEOMETRY},
        'easy_threshold': .2, 'hard_threshold': .5, 'eligible_arms': ['independent', 'scene_uniform', 'joint'],
        'solver_seconds': 2., 'bootstrap_seed': 917, 'policies': {'p1': POLICY},
    }
    protocol['risk']['easy_definition'] = {'kind': 'baseline_error_at_most', 'metric': 'ade',
                                         'error_unit': 'past_normalized', 'threshold': .2}
    approve(protocol)
    contract = ExperimentContract(protocol, tmp_path)
    architecture, settings = training_config()
    baseline = 'constant_velocity_causal_fd'
    artifacts = []
    for name, recordings in (('fit_a', ['a']), ('fit_b', ['b']), ('fit_ab', ['a', 'b'])):
        data = ContractForecastDataset(contract, recordings, purpose='fit', baseline_name=baseline)
        result = train_forecaster(data, architecture=architecture, settings=settings, output_dir=tmp_path / name)
        checkpoint = Path(result['checkpoint'])
        artifacts.append({'id': name, 'kind': 'forecaster', 'path': str(checkpoint.relative_to(tmp_path)),
                          'sha256': file_digest(checkpoint), 'protocol_sha256': contract.digest, 'lineage_complete': True,
                          'fit_recordings': recordings, 'selection_recordings': [], 'calibration_recordings': [], 'parents': []})
    contract = ExperimentContract(protocol, tmp_path, artifacts)
    groups = []
    for predictor, held in (('fit_a', 'b'), ('fit_b', 'a')):
        data = ContractForecastDataset(contract, [held], purpose='fit', baseline_name=baseline)
        model = load_verified_forecaster(contract, predictor, device='cpu')
        groups.append(make_oof_cost_rows(contract, predictor, data, model, batch_size=8, device='cpu'))
    head = fit_linear_gain_harm(contract, groups, alpha=1.)
    checkpoint = tmp_path / 'cost.npz'
    np.savez(checkpoint, **{k: head[k] for k in ('mean', 'scale', 'coef', 'intercept')})
    report = {k: v for k, v in head.items() if k not in {'mean', 'scale', 'coef', 'intercept'}}
    report.update(checkpoint_sha256=file_digest(checkpoint),
                  code_sha256=file_digest(Path(__file__).resolve().parents[1] / 'src/world_model/m3w_supervised_intervention.py'))
    path = tmp_path / 'cost.json'
    path.write_text(json.dumps(report))
    artifacts.append({'id': 'cost', 'kind': 'risk_head', 'path': checkpoint.name, 'sha256': file_digest(checkpoint),
                      'protocol_sha256': contract.digest, 'lineage_complete': True, 'fit_recordings': ['a', 'b'],
                      'selection_recordings': [], 'calibration_recordings': [], 'parents': ['fit_a', 'fit_b']})
    plan = {'schema_version': 1, 'candidates': [{'id': 'small', 'forecaster_id': 'fit_ab', 'risk_head_id': 'cost',
                'risk_report_path': path.name, 'risk_report_sha256': file_digest(path),
                'baseline': baseline, 'policy_id': 'p1'}]}
    return ExperimentContract(protocol, tmp_path, artifacts), plan


def test_fitted_forecaster_oof_cost_and_development_evaluation_end_to_end(tmp_path, monkeypatch):
    from src.data_unification.m3w_causal_recordings import RecordingWindows
    from src.evaluation import m3w_development_evaluation as module
    contract, plan = fitted_fixture(tmp_path)
    opened, labels, events = [], [], []
    original_open, original_labels = contract.open_recording, RecordingWindows.get_scene_labels
    original_decide = module.decide_scene

    def checked_open(name, *, purpose, **kwargs):
        assert name == 'd' and purpose == 'development'
        opened.append(name)
        return original_open(name, purpose=purpose, **kwargs)

    def checked_labels(reader, scene):
        assert reader.metadata['id'] == 'd'
        assert events[-1] == ('decision', scene['frame_id'])
        events.append(('labels', scene['frame_id']))
        labels.append(scene['frame_id'])
        return original_labels(reader, scene)

    def checked_decide(scene, *args, **kwargs):
        decision = original_decide(scene, *args, **kwargs)
        events.append(('decision', scene['frame_id']))
        return decision

    monkeypatch.setattr(contract, 'open_recording', checked_open)
    monkeypatch.setattr(RecordingWindows, 'get_scene_labels', checked_labels)
    monkeypatch.setattr(module, 'decide_scene', checked_decide)
    report, rows = evaluate_development(contract, plan)
    summary = report['summaries']['small']
    assert opened == ['d'] and labels
    assert set(summary['arms']) == {'floor', 'uncontrolled', 'independent', 'scene_uniform', 'joint'}
    assert summary['agent_query_count'] > summary['label_coverage']['fde']
    assert summary['arms']['floor']['all']['improvement_pct'] == pytest.approx(0.)
    assert report['selection']['deployable'] is False
    assert not report['calibration_or_confirmation_opened']
    assert {r['recording_id'] for r in rows['small']} == {'d'}


def test_evaluation_family_requires_approved_rules_and_provenance(tmp_path):
    contract, plan = fitted_fixture(tmp_path)
    bad = deepcopy(plan)
    bad['candidates'][0]['policy_id'] = 'unapproved_thresholds'
    with pytest.raises(ValueError, match='approved candidate'):
        validate_plan(contract, bad)
    bad = deepcopy(plan)
    bad['candidates'][0]['risk_report_sha256'] = 'changed'
    with pytest.raises(ValueError, match='identity'):
        validate_plan(contract, bad)
    p = deepcopy(contract.protocol)
    p.pop('development_evaluation')
    approve(p)
    with pytest.raises(ValueError, match='development_evaluation'):
        validate_plan(ExperimentContract(p, tmp_path), plan)


def test_cost_report_cannot_smuggle_test_normalization_before_label_access(tmp_path, monkeypatch):
    from src.data_unification.m3w_causal_recordings import RecordingWindows
    contract, plan = fitted_fixture(tmp_path)
    path = tmp_path / plan['candidates'][0]['risk_report_path']
    report = json.loads(path.read_text())
    report['normalization_source'] = 'confirmation_statistics'
    path.write_text(json.dumps(report))
    plan['candidates'][0]['risk_report_sha256'] = file_digest(path)
    monkeypatch.setattr(RecordingWindows, 'get_scene_labels', lambda *_: pytest.fail('Labels opened before provenance refusal'))
    with pytest.raises(ValueError, match='lineage mismatch'):
        evaluate_development(contract, plan)


def test_easy_definition_cannot_be_changed_independently_of_risk_protocol(tmp_path):
    contract, plan = fitted_fixture(tmp_path)
    protocol = deepcopy(contract.protocol)
    protocol['development_evaluation']['easy_threshold'] = .4
    approve(protocol)
    changed = ExperimentContract(protocol, tmp_path)
    with pytest.raises(ValueError, match='risk definition'):
        validate_plan(changed, plan)


def test_development_cli_freezes_family_and_resumes_hash_verified_cache(tmp_path):
    import subprocess
    contract, plan = fitted_fixture(tmp_path)
    protocol_path, plan_path = tmp_path / 'protocol.json', tmp_path / 'plan.json'
    protocol_path.write_text(json.dumps(contract.protocol))
    plan_path.write_text(json.dumps(plan))
    artifacts = []
    for name, value in contract.artifacts.items():
        path = tmp_path / (name + '.artifact.json')
        path.write_text(json.dumps(value))
        artifacts.append(str(path))
    root = Path(__file__).resolve().parents[1]
    command = [str(root / '.venv-pytorch/bin/python'), str(root / 'scripts/evaluate_m3w_development.py'),
               '--workspace-root', str(tmp_path), '--protocol', str(protocol_path), '--plan', str(plan_path),
               '--artifacts', *artifacts, '--output-dir', str(tmp_path / 'evaluation')]
    first = subprocess.run(command, text=True, capture_output=True, timeout=60)
    assert first.returncode == 0, first.stderr
    output = tmp_path / 'evaluation'
    digest = file_digest(output / 'development_report.json')
    resumed = subprocess.run([*command, '--resume'], text=True, capture_output=True, timeout=60)
    assert resumed.returncode == 0 and 'cached_verified' in resumed.stdout, resumed.stderr
    assert digest == file_digest(output / 'development_report.json')
    assert (output / 'heartbeat.jsonl').is_file()
    assert not (tmp_path / 'confirmation.json').exists() and not (tmp_path / 'calibration.json').exists()
    cache = next(output.glob('*.rows.json'))
    cache.write_text('[]')
    corrupted = subprocess.run([*command, '--resume'], text=True, capture_output=True, timeout=60)
    assert corrupted.returncode != 0 and 'cache identity changed' in corrupted.stderr
