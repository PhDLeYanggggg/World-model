from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from test_m3w_development_evaluation import BaselineCopy, POLICY, GEOMETRY, zero_head, fitted_fixture
from test_m3w_supervised_intervention import datasets
from src.evaluation.m3w_development_evaluation import decide_scene, score_scene, summarize_rows, evaluate_development
from src.world_model.m3w_cost_sensitive_deferral import DeferralHead


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def constant_deferral(scene, baseline, bias=1.):
    dim = len(zero_head(scene, baseline)['mean'])
    head = DeferralHead(np.zeros(dim), np.ones(dim), width=0)
    with torch.no_grad():
        head.network.weight.zero_()
        head.network.bias[:] = torch.tensor([0., bias])
    return head.eval()


def test_deferral_is_opt_in_same_forecast_and_unconstrained(tmp_path):
    _, train, _ = datasets(tmp_path)
    scene = train.readers[0].get_scene_inputs(100, 120)
    kwargs = dict(baseline=train.baseline_name, policy={**POLICY, 'max_intervention_fraction': 0.},
                  geometry=GEOMETRY, device='cpu', solver_seconds=2.)
    ordinary = decide_scene(scene, BaselineCopy(), zero_head(scene, train.baseline_name), **kwargs)
    result = decide_scene(scene, BaselineCopy(), zero_head(scene, train.baseline_name),
                          deferral_head=constant_deferral(scene, train.baseline_name), **kwargs)
    assert 'cost_sensitive_deferral' not in ordinary['arms']
    arm = result['arms']['cost_sensitive_deferral']
    assert arm['switch'].all() and not arm['predicted_constraints_satisfied']
    assert not arm['constraints_enforced'] and not arm['calibrated_risk']
    np.testing.assert_array_equal(arm['prediction'], ordinary['candidate'])
    for name in ordinary['arms']:
        np.testing.assert_array_equal(ordinary['arms'][name]['prediction'], result['arms'][name]['prediction'])
    labels = train.readers[0].get_scene_labels(scene)
    rows = score_scene(scene, result, labels, label_policy='complete_requested_path')
    summary = summarize_rows(rows, metric='ade', aggregation='equal_physical_scene', error_unit='past_normalized',
                             easy_threshold=.2, hard_threshold=.5, bootstrap_resamples=2000, bootstrap_seed=17)
    assert summary['arms']['cost_sensitive_deferral']['control_constraints_enforced'] is False
    assert 'deferral_unconstrained' in summary['matched_comparison']


def test_oof_fingerprint_tracks_rows_features_and_producers_but_not_costs():
    from src.world_model.m3w_oof_identity import oof_feature_identity
    rows = [{'recording_id': 'a', 'agent_id': 1, 'frame_id': i, 'horizon_raw': 50} for i in range(2)]
    group = {'features': np.array([[1., 2.], [3., 4.]], np.float32), 'identities': rows,
             'predictor_id': 'p', 'predictor_sha256': 'a'*64, 'baseline_name': 'b', 'metric': 'ade',
             'protocol_sha256': 'c'*64, 'targets': np.ones((2, 2))}
    before = oof_feature_identity([group])
    changed = deepcopy(group)
    changed['features'], changed['identities'] = changed['features'][::-1], changed['identities'][::-1]
    changed['targets'] *= 100
    assert oof_feature_identity([changed]) == before
    changed['features'][0, 0] += .1
    assert oof_feature_identity([changed]) != before
    changed = {**group, 'predictor_id': 'different'}
    assert oof_feature_identity([changed]) != before
    with pytest.raises(ValueError, match='Duplicated'):
        oof_feature_identity([group, group])


def fitted_comparison(tmp_path, *, seed=1):
    from src.evaluation.m3w_experiment_contract import ExperimentContract, file_digest
    from src.world_model.m3w_supervised_intervention import ContractForecastDataset, load_verified_forecaster, make_oof_cost_rows
    from src.world_model.m3w_cost_sensitive_deferral import make_oof_deferral_rows, train_deferral
    from src.world_model.m3w_oof_identity import oof_feature_identity

    def rules(p):
        p['development_evaluation']['diagnostic_controls'] = ['cost_sensitive_deferral']
        p['comparators'] = {'cost_sensitive_deferral': {'cost_bound': 2., 'width': 0,
            'fit_settings': {'steps': 8, 'batch_size': 8, 'learning_rate': .02,
                             'checkpoint_every': 3, 'heartbeat_every': 3}}}

    contract, plan = fitted_fixture(tmp_path, protocol_updates=rules)
    groups, relative = [], []
    for predictor, recording in (('fit_a', 'b'), ('fit_b', 'a')):
        data = ContractForecastDataset(contract, [recording], purpose='fit', baseline_name=plan['candidates'][0]['baseline'])
        model = load_verified_forecaster(contract, predictor, device='cpu')
        groups.append(make_oof_deferral_rows(contract, predictor, data, model, batch_size=8, device='cpu'))
        relative.append(make_oof_cost_rows(contract, predictor, data, model, batch_size=8, device='cpu'))
    risk_path = tmp_path / 'cost.json'
    risk = json.loads(risk_path.read_text())
    risk['oof_feature_identity'] = oof_feature_identity(relative)
    risk_path.write_text(json.dumps(risk))
    plan['candidates'][0]['risk_report_sha256'] = file_digest(risk_path)
    fitted = train_deferral(contract, groups, seed=seed, output_dir=tmp_path / 'deferral')
    report = tmp_path / 'deferral/fit_report.json'
    artifact = {'id': 'deferral', 'kind': 'policy', 'family': 'cost_sensitive_deferral',
                'path': 'deferral/latest.pt', 'sha256': fitted['checkpoint_sha256'],
                'protocol_sha256': contract.digest, 'lineage_complete': True,
                'parents': fitted['parents'], 'fit_recordings': fitted['fit_recordings'],
                'selection_recordings': [], 'calibration_recordings': []}
    plan['candidates'][0].update(deferral_head_id='deferral', deferral_report_path='deferral/fit_report.json',
                                deferral_report_sha256=file_digest(report))
    return ExperimentContract(contract.protocol, tmp_path, [*contract.artifacts.values(), artifact]), plan


def test_fitted_comparison_decides_all_arms_before_labels(tmp_path, monkeypatch):
    from src.data_unification.m3w_causal_recordings import RecordingWindows
    from src.evaluation import m3w_development_evaluation as module
    contract, plan = fitted_comparison(tmp_path)
    events = []
    decide, labels = module.decide_scene, RecordingWindows.get_scene_labels

    def checked_decide(scene, *args, **kwargs):
        result = decide(scene, *args, **kwargs)
        assert set(result['arms']) == {*module.ARMS, 'cost_sensitive_deferral'}
        events.append(('all_decisions', scene['frame_id']))
        return result

    def checked_labels(reader, scene):
        assert reader.metadata['id'] == 'd' and events[-1] == ('all_decisions', scene['frame_id'])
        events.append(('labels', scene['frame_id']))
        return labels(reader, scene)

    monkeypatch.setattr(module, 'decide_scene', checked_decide)
    monkeypatch.setattr(RecordingWindows, 'get_scene_labels', checked_labels)
    report, rows = evaluate_development(contract, plan)
    summary = report['summaries']['small']
    assert len(events) > 0 and len(events) % 2 == 0
    assert summary['agent_query_count'] > summary['label_coverage']['fde']
    assert summary['arms']['cost_sensitive_deferral']['control_constraints_enforced'] is False
    assert summary['paired_deferral_comparisons']['joint']['all']['ci95'] is None
    assert report['selection']['selected']['arm'] != 'cost_sensitive_deferral'
    assert not report['calibration_or_confirmation_opened']
    assert {r['recording_id'] for r in rows['small']} == {'d'}


@pytest.mark.parametrize('mutation', ['missing_input_identity', 'changed_input_identity', 'report_normalization',
                                      'report_seed', 'candidate_baseline'])
def test_comparator_mismatch_refused_before_any_labels(tmp_path, monkeypatch, mutation):
    from src.data_unification.m3w_causal_recordings import RecordingWindows
    from src.evaluation.m3w_experiment_contract import file_digest
    contract, plan = fitted_comparison(tmp_path)
    candidate = plan['candidates'][0]
    if mutation in {'missing_input_identity', 'changed_input_identity'}:
        path = tmp_path / candidate['risk_report_path']
        report = json.loads(path.read_text())
        if mutation == 'missing_input_identity':
            report.pop('oof_feature_identity')
        else:
            report['oof_feature_identity']['sha256'] = 'changed'
        report_key = 'risk_report_sha256'
    elif mutation in {'report_normalization', 'report_seed'}:
        path = tmp_path / candidate['deferral_report_path']
        report = json.loads(path.read_text())
        report['normalization_source' if mutation == 'report_normalization' else 'seed'] = 'wrong'
        report_key = 'deferral_report_sha256'
    else:
        candidate['baseline'] = 'damped_velocity'
    if mutation != 'candidate_baseline':
        path.write_text(json.dumps(report))
        candidate[report_key] = file_digest(path)
    monkeypatch.setattr(RecordingWindows, 'get_scene_labels', lambda *_: pytest.fail('Premature label access'))
    with pytest.raises(ValueError, match='mismatch'):
        evaluate_development(contract, plan)


def test_actual_deferral_seed_must_match_forecaster(tmp_path, monkeypatch):
    from src.data_unification.m3w_causal_recordings import RecordingWindows
    contract, plan = fitted_comparison(tmp_path, seed=2)
    monkeypatch.setattr(RecordingWindows, 'get_scene_labels', lambda *_: pytest.fail('Premature label access'))
    with pytest.raises(ValueError, match='seed mismatch'):
        evaluate_development(contract, plan)


def test_no_silent_diagnostic_arm_or_policy_selection(tmp_path):
    from src.evaluation.m3w_development_evaluation import validate_plan, choose_development
    contract, plan = fitted_fixture(tmp_path)
    plan['candidates'][0]['deferral_head_id'] = 'not_approved'
    with pytest.raises(ValueError, match='explicit protocol'):
        validate_plan(contract, plan)
    with pytest.raises(ValueError, match='guarded development family'):
        choose_development({'fake': {}}, eligible_arms=['cost_sensitive_deferral'], easy_degradation_max=.02)


def test_full_family_refused_before_first_candidate_labels(tmp_path, monkeypatch):
    from src.data_unification.m3w_causal_recordings import RecordingWindows
    from src.evaluation.m3w_experiment_contract import file_digest
    contract, plan = fitted_comparison(tmp_path)
    second = deepcopy(plan['candidates'][0])
    path = tmp_path / 'bad_report.json'
    report = json.loads((tmp_path / second['deferral_report_path']).read_text())
    report['normalization_source'] = 'test_stats'
    path.write_text(json.dumps(report))
    second.update(id='later_invalid', deferral_report_path=path.name, deferral_report_sha256=file_digest(path))
    plan['candidates'].append(second)
    monkeypatch.setattr(RecordingWindows, 'get_scene_labels', lambda *_: pytest.fail('First candidate labels opened too soon'))
    with pytest.raises(ValueError, match='provenance mismatch'):
        evaluate_development(contract, plan)


def test_paired_interval_uses_scene_not_window_count():
    from src.evaluation.m3w_development_evaluation import paired_control_errors
    from test_m3w_development_evaluation import metric_row
    rows = [metric_row('a', 'a', i, 10., 7.) for i in range(100)] + [metric_row('b', 'b', 0, 10., 11.)]
    for row in rows:
        row['arms']['cost_sensitive_deferral'] = {**row['arms']['joint'], 'fde': 9., 'ade': 9.}
    report = paired_control_errors(rows, 'joint', 'cost_sensitive_deferral', metric='fde',
                                  aggregation='equal_physical_scene', n_bootstrap=2000, seed=12)
    assert report['left_minus_right_error'] == pytest.approx(0.)
    assert report['ci95'] == pytest.approx([-2., 2.])
    assert report['physical_scene_count'] == 2 and report['count'] == 101
    assert not report['coverage_matched'] and not report['risk_matched']


def test_deferral_future_mutation_and_missing_labels_do_not_change_membership(tmp_path):
    _, train, _ = datasets(tmp_path)
    reader = train.readers[0]
    scene = reader.get_scene_inputs(290, 120)
    kwargs = dict(baseline=train.baseline_name, policy=POLICY, geometry=GEOMETRY,
                  device='cpu', solver_seconds=2., deferral_head=constant_deferral(scene, train.baseline_name))
    before = decide_scene(scene, BaselineCopy(), zero_head(scene, train.baseline_name), **kwargs)
    labels = reader.get_scene_labels(scene)
    rows = score_scene(scene, before, labels, label_policy='complete_requested_path')
    assert len(rows) == 2 and all(r['baseline_fde'] is None for r in rows)
    assert all(r['arms']['cost_sensitive_deferral']['switch'] for r in rows)
    points = np.array(reader.points)
    points[points[:, 0] > 290, 2:] = np.nan
    reader.points = points
    after_scene = reader.get_scene_inputs(290, 120)
    after = decide_scene(after_scene, BaselineCopy(), zero_head(scene, train.baseline_name), **kwargs)
    for arm in before['arms']:
        np.testing.assert_array_equal(before['arms'][arm]['prediction'], after['arms'][arm]['prediction'])


def test_nonfinite_forecast_falls_back_for_deferral(tmp_path):
    _, train, _ = datasets(tmp_path)
    scene = train.readers[0].get_scene_inputs(100, 120)

    class Invalid(BaselineCopy):
        def forward(self, inputs):
            result = super().forward(inputs)
            result[0, 0, 0] = float('nan')
            return result

    result = decide_scene(scene, Invalid(), zero_head(scene, train.baseline_name), baseline=train.baseline_name,
                          policy=POLICY, geometry=GEOMETRY, device='cpu', solver_seconds=2.,
                          deferral_head=constant_deferral(scene, train.baseline_name))
    assert result['arms']['cost_sensitive_deferral']['switch'].tolist() == [False, True]


def test_training_and_development_clis_match_budget_and_resume(tmp_path):
    import subprocess
    import sys
    from src.evaluation.m3w_experiment_contract import file_digest
    contract, plan = fitted_comparison(tmp_path)
    root = Path(__file__).resolve().parents[1]
    protocol, mapping = tmp_path / 'protocol.json', tmp_path / 'mapping.json'
    protocol.write_text(json.dumps(contract.protocol))
    mapping.write_text(json.dumps({'0': 'fit_b', '1': 'fit_a'}))
    manifests = []
    for key in ('fit_a', 'fit_b', 'fit_ab'):
        path = tmp_path / f'{key}.json'
        path.write_text(json.dumps(contract.artifacts[key]))
        manifests.append(str(path))
    shared = ['--protocol', str(protocol), '--workspace-root', str(tmp_path), '--artifacts', *manifests,
              '--fold-models', str(mapping), '--baseline', plan['candidates'][0]['baseline'], '--threads', '2']
    for script, folder, extra in (
        ('train_m3w_oof_cost_head.py', 'ridge_cli', ['--alpha', '1']),
        ('train_m3w_deferral_control.py', 'deferral_cli', ['--seed', '1']),
    ):
        command = [sys.executable, str(root / 'scripts' / script), *shared, *extra, '--output-dir', str(tmp_path / folder)]
        subprocess.run(command, check=True, capture_output=True, timeout=40)
    risk, defer = [json.loads((tmp_path / folder / 'fit_report.json').read_text())
                   for folder in ('ridge_cli', 'deferral_cli')]
    assert risk['oof_feature_identity'] == defer['oof_feature_identity']
    assert risk['training_rows'] == defer['training_rows'] == risk['oof_feature_identity']['rows']
    candidate = plan['candidates'][0]
    candidate.update(risk_head_id='ridge_cli', risk_report_path='ridge_cli/fit_report.json',
                     risk_report_sha256=file_digest(tmp_path / 'ridge_cli/fit_report.json'),
                     deferral_head_id='deferral_cli', deferral_report_path='deferral_cli/fit_report.json',
                     deferral_report_sha256=file_digest(tmp_path / 'deferral_cli/fit_report.json'))
    plan_path, output = tmp_path / 'plan.json', tmp_path / 'development'
    plan_path.write_text(json.dumps(plan))
    command = [sys.executable, str(root / 'scripts/evaluate_m3w_development.py'), '--protocol', str(protocol),
               '--workspace-root', str(tmp_path), '--plan', str(plan_path), '--artifacts', *manifests,
               str(tmp_path / 'ridge_cli/artifact.json'), str(tmp_path / 'deferral_cli/artifact.json'),
               '--output-dir', str(output), '--threads', '2']
    subprocess.run(command, check=True, capture_output=True, timeout=40)
    result_path = output / 'development_report.json'
    result = json.loads(result_path.read_text())
    assert result['diagnostic_controls'] == ['cost_sensitive_deferral']
    assert not result['calibration_or_confirmation_opened']
    assert 'cost_sensitive_deferral' in result['summaries']['small']['arms']
    before = file_digest(result_path)
    cached = subprocess.run(command + ['--resume'], check=True, capture_output=True, text=True, timeout=40)
    assert 'cached_verified' in cached.stdout and file_digest(result_path) == before
    cache = next(output.glob('*.rows.json'))
    cache.write_text(cache.read_text() + ' ')
    refused = subprocess.run(command + ['--resume'], capture_output=True, text=True, timeout=40)
    assert refused.returncode != 0 and 'cache identity changed' in refused.stderr
