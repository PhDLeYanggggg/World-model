from copy import deepcopy
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from test_m3w_development_evaluation import fitted_fixture
from test_m3w_experiment_contract import approve
from src.evaluation.m3w_experiment_contract import ExperimentContract, file_digest


SPEC = {'width': 12, 'loss': 'squared_benefit_harm', 'fit_settings': {
    'steps': 12, 'batch_size': 8, 'learning_rate': .01, 'checkpoint_every': 3, 'heartbeat_every': 3}}


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def fitted_inputs(tmp_path):
    from src.world_model.m3w_supervised_intervention import (
        ContractForecastDataset, load_verified_forecaster, make_oof_cost_rows,
    )
    contract, plan = fitted_fixture(tmp_path, protocol_updates=lambda p: p.update(gain_harm_training=deepcopy(SPEC)))
    groups = []
    for producer, held in (('fit_a', 'b'), ('fit_b', 'a')):
        data = ContractForecastDataset(contract, [held], purpose='fit', baseline_name=plan['candidates'][0]['baseline'])
        model = load_verified_forecaster(contract, producer, device='cpu')
        groups.append(make_oof_cost_rows(contract, producer, data, model, batch_size=8, device='cpu'))
    return contract, plan, groups


def test_neural_head_predicts_coherent_costs_and_keeps_magnitudes():
    from src.world_model.m3w_neural_gain_harm import NeuralGainHarm, regression_loss
    head = NeuralGainHarm(np.zeros(3), np.ones(3), width=8)
    scores = head(torch.zeros(2, 3))
    assert torch.all(scores['benefit'] >= 0) and torch.all(scores['harm'] >= 0)
    torch.testing.assert_close(scores['gain'], scores['benefit'] - scores['harm'])
    prediction = {'benefit': torch.tensor([.2, .2], requires_grad=True),
                  'harm': torch.tensor([.1, .1], requires_grad=True)}
    small = regression_loss(prediction, torch.tensor([[.5, 0.], [0., .5]]))
    large = regression_loss(prediction, torch.tensor([[5., 0.], [0., 5.]]))
    assert large > small * 10
    large.backward()
    assert prediction['benefit'].grad is not None


def test_neural_training_resume_equals_uninterrupted(tmp_path):
    from src.world_model.m3w_neural_gain_harm import train_neural_gain_harm
    contract, _, groups = fitted_inputs(tmp_path)
    direct = train_neural_gain_harm(contract, groups, seed=1, output_dir=tmp_path / 'direct')
    first = train_neural_gain_harm(contract, groups, seed=1, output_dir=tmp_path / 'resumed', stop_after=5)
    resumed = train_neural_gain_harm(contract, groups, seed=1, output_dir=tmp_path / 'resumed', resume=True)
    assert not first['training_complete'] and resumed['resumed_from_step'] == 5
    assert direct['training_complete'] and resumed['training_complete']
    assert direct['losses'] == resumed['losses'] and np.isfinite(direct['losses']).all()
    states = [torch.load(r['checkpoint'], weights_only=True) for r in (direct, resumed)]
    for key in states[0]['model']:
        torch.testing.assert_close(states[0]['model'][key], states[1]['model'][key], atol=0, rtol=0)
    expected = np.concatenate([g['features'] for g in groups]).mean(0)
    np.testing.assert_allclose(states[0]['model']['mean'], expected, atol=1e-6)
    stable = file_digest(Path(resumed['checkpoint']))
    reused = train_neural_gain_harm(contract, groups, seed=1, output_dir=tmp_path / 'resumed', resume=True)
    assert reused['result_source'] == 'cached_verified'
    assert file_digest(Path(resumed['checkpoint'])) == stable
    assert reused['normalization_source'] == 'fit_OOF_rows_only' and not reused['calibrated_risk']


@pytest.mark.parametrize('violation', ['missing_fold', 'duplicate_fold', 'wrong_role', 'absolute_cost',
                                     'producer', 'nonfinite', 'negative_cost', 'unaligned'])
def test_neural_fit_refuses_invalid_oof_before_checkpoint(tmp_path, violation):
    from src.world_model.m3w_neural_gain_harm import train_neural_gain_harm
    contract, _, groups = fitted_inputs(tmp_path)
    if violation == 'missing_fold':
        groups.pop()
    elif violation == 'duplicate_fold':
        groups.append(deepcopy(groups[0]))
    elif violation == 'wrong_role':
        groups[0]['identities'][0]['data_role'] = 'confirmation'
    elif violation == 'absolute_cost':
        groups[0]['target_source'] = 'held_fold_absolute_past_normalized_errors'
    elif violation == 'producer':
        groups[0]['predictor_sha256'] = 'changed'
    elif violation == 'nonfinite':
        groups[0]['features'][0, 0] = np.nan
    elif violation == 'negative_cost':
        groups[0]['targets'][0, 0] = -1.
    else:
        groups[0]['targets'] = groups[0]['targets'][:-1]
    with pytest.raises(ValueError):
        train_neural_gain_harm(contract, groups, seed=1, output_dir=tmp_path / 'refused')
    assert not (tmp_path / 'refused/latest.pt').exists()


def test_neural_spec_and_seed_are_not_silently_chosen(tmp_path):
    from src.world_model.m3w_neural_gain_harm import train_neural_gain_harm, training_spec
    contract, _, groups = fitted_inputs(tmp_path)
    with pytest.raises(ValueError, match='seed'):
        train_neural_gain_harm(contract, groups, seed=2, output_dir=tmp_path / 'wrong_seed')
    p = deepcopy(contract.protocol)
    p.pop('gain_harm_training')
    approve(p)
    with pytest.raises(ValueError, match='specification'):
        training_spec(ExperimentContract(p, tmp_path))


@pytest.mark.parametrize('field', ['features', 'targets'])
def test_resume_binds_features_and_label_magnitudes(tmp_path, field):
    from src.world_model.m3w_neural_gain_harm import train_neural_gain_harm
    contract, _, groups = fitted_inputs(tmp_path)
    train_neural_gain_harm(contract, groups, seed=1, output_dir=tmp_path / 'fit', stop_after=3)
    groups[0][field][0, 0] += .5
    with pytest.raises(ValueError, match='resume identity'):
        train_neural_gain_harm(contract, groups, seed=1, output_dir=tmp_path / 'fit', resume=True)


def test_completed_resume_rejects_changed_checkpoint_bytes(tmp_path):
    from src.world_model.m3w_neural_gain_harm import train_neural_gain_harm
    contract, _, groups = fitted_inputs(tmp_path)
    report = train_neural_gain_harm(contract, groups, seed=1, output_dir=tmp_path / 'fit')
    checkpoint = Path(report['checkpoint'])
    with checkpoint.open('ab') as stream:
        stream.write(b'changed_after_completion')
    with pytest.raises(ValueError, match='Completed neural cost checkpoint changed'):
        train_neural_gain_harm(contract, groups, seed=1, output_dir=tmp_path / 'fit', resume=True)


def register_head(contract, report, name='neural'):
    artifact = {'id': name, 'kind': 'risk_head', 'family': 'neural_gain_harm',
                'path': str(Path(report['checkpoint']).relative_to(contract.root)),
                'sha256': report['checkpoint_sha256'], 'protocol_sha256': contract.digest,
                'lineage_complete': True, 'fit_recordings': report['fit_recordings'],
                'selection_recordings': [], 'calibration_recordings': [], 'parents': report['parents']}
    return ExperimentContract(contract.protocol, contract.root, [*contract.artifacts.values(), artifact])


def attach_plan(contract, plan, groups):
    from src.world_model.m3w_neural_gain_harm import train_neural_gain_harm
    from src.world_model.m3w_oof_identity import oof_feature_identity
    report = train_neural_gain_harm(contract, groups, seed=1, output_dir=contract.root / 'neural')
    contract = register_head(contract, report)
    candidate = {**plan['candidates'][0], 'id': 'neural', 'risk_head_id': 'neural',
                 'risk_report_path': 'neural/fit_report.json',
                 'risk_report_sha256': file_digest(contract.root / 'neural/fit_report.json')}
    plan = {'schema_version': 1, 'candidates': [*plan['candidates'], candidate]}
    assert report['oof_feature_identity'] == oof_feature_identity(groups)
    return contract, plan


def test_neural_and_ridge_development_share_forecasts_and_decide_before_labels(tmp_path, monkeypatch):
    from src.data_unification.m3w_causal_recordings import RecordingWindows
    from src.evaluation import m3w_development_evaluation as module
    contract, plan, groups = fitted_inputs(tmp_path)
    contract, plan = attach_plan(contract, plan, groups)
    events, forecasts = [], []
    decide, labels = module.decide_scene, RecordingWindows.get_scene_labels

    def checked_decide(scene, *args, **kwargs):
        result = decide(scene, *args, **kwargs)
        events.append(('decision', scene['frame_id']))
        forecasts.append(result['candidate'].copy())
        return result

    def checked_labels(reader, scene):
        assert reader.metadata['id'] == 'd' and events[-1] == ('decision', scene['frame_id'])
        events.append(('labels', scene['frame_id']))
        return labels(reader, scene)

    monkeypatch.setattr(module, 'decide_scene', checked_decide)
    monkeypatch.setattr(RecordingWindows, 'get_scene_labels', checked_labels)
    result, rows = module.evaluate_development(contract, plan)
    halfway = len(forecasts) // 2
    assert halfway > 0
    for a, b in zip(forecasts[:halfway], forecasts[halfway:]):
        np.testing.assert_array_equal(a, b)
    assert len(rows['small']) == len(rows['neural'])
    assert not result['calibration_or_confirmation_opened']
    assert result['summaries']['small']['physical_scene_count'] == 1
    assert result['summaries']['neural']['arms']['joint']['all']['bootstrap']['status'] == 'not_run_insufficient_physical_scenes'


@pytest.mark.parametrize('violation', ['unfinished', 'changed_report', 'seed_claim', 'architecture'])
def test_neural_family_rejected_before_any_development_labels(tmp_path, monkeypatch, violation):
    from src.evaluation.m3w_development_evaluation import evaluate_development
    from src.data_unification.m3w_causal_recordings import RecordingWindows
    from src.world_model.m3w_neural_gain_harm import train_neural_gain_harm
    contract, plan, groups = fitted_inputs(tmp_path)
    report = train_neural_gain_harm(contract, groups, seed=1, output_dir=tmp_path / 'neural',
                                    stop_after=3 if violation == 'unfinished' else None)
    contract = register_head(contract, report)
    if violation == 'changed_report':
        report['oof_feature_identity']['sha256'] = 'wrong'
    elif violation == 'seed_claim':
        report['seed'] = 2
    elif violation == 'architecture':
        report['spec']['width'] += 1
    path = tmp_path / 'neural/fit_report.json'
    path.write_text(json.dumps(report))
    plan['candidates'][0].update(risk_head_id='neural', risk_report_path='neural/fit_report.json',
                                 risk_report_sha256=file_digest(path))
    monkeypatch.setattr(RecordingWindows, 'get_scene_labels', lambda *a: pytest.fail('Opened development labels'))
    with pytest.raises(ValueError):
        evaluate_development(contract, plan)


def test_neural_decisions_invariant_to_hidden_future(tmp_path):
    from src.world_model.m3w_neural_gain_harm import train_neural_gain_harm, load_verified_neural_gain_harm
    from src.world_model.m3w_supervised_intervention import load_verified_forecaster
    from src.evaluation.m3w_development_evaluation import decide_scene
    from test_m3w_development_evaluation import POLICY, GEOMETRY
    contract, plan, groups = fitted_inputs(tmp_path)
    result = train_neural_gain_harm(contract, groups, seed=1, output_dir=tmp_path / 'neural')
    contract = register_head(contract, result)
    head = load_verified_neural_gain_harm(contract, 'neural')
    forecast = load_verified_forecaster(contract, 'fit_ab', device='cpu')
    reader, _ = contract.open_recording('d', purpose='development')
    before = reader.get_scene_inputs(100, 120)
    points = reader.points.copy()
    points[points[:, 0] > 100, 2:] += 10000
    reader.points = points
    after = reader.get_scene_inputs(100, 120)
    kwargs = dict(baseline=plan['candidates'][0]['baseline'], policy=POLICY, geometry=GEOMETRY,
                  device='cpu', solver_seconds=2.)
    a, b = (decide_scene(s, forecast, head, **kwargs) for s in (before, after))
    for key in ('predicted_gain', 'predicted_harm', 'candidate'):
        np.testing.assert_array_equal(a[key], b[key])
    for arm in a['arms']:
        np.testing.assert_array_equal(a['arms'][arm]['switch'], b['arms'][arm]['switch'])


def test_neural_cli_requires_real_approval_before_torch():
    import subprocess
    import sys
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, str(root / 'scripts/train_m3w_neural_cost_head.py'), '--preflight-only'],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 2
    assert json.loads(result.stdout)['neural_cost_training_started'] is False


def test_neural_cli_reuses_ridge_oof_and_resumes_across_processes(tmp_path):
    import subprocess
    import sys
    contract, plan, groups = fitted_inputs(tmp_path)
    root = Path(__file__).resolve().parents[1]
    protocol, mapping = tmp_path / 'protocol.json', tmp_path / 'mapping.json'
    protocol.write_text(json.dumps(contract.protocol))
    folds = contract.protocol['fit_folds']
    mapping.write_text(json.dumps({str(folds[g['identities'][0]['recording_id']]): g['predictor_id'] for g in groups}))
    paths = []
    for name, artifact in contract.artifacts.items():
        path = tmp_path / f'{name}.artifact.json'
        path.write_text(json.dumps(artifact))
        paths.append(str(path))
    common = ['--workspace-root', str(tmp_path), '--protocol', str(protocol), '--artifacts', *paths, '--threads', '2']
    ridge = [sys.executable, str(root / 'scripts/train_m3w_oof_cost_head.py'), *common,
             '--fold-models', str(mapping), '--baseline', groups[0]['baseline_name'], '--alpha', '1',
             '--output-dir', str(tmp_path / 'ridge_cache')]
    subprocess.run(ridge, check=True, capture_output=True, timeout=40)
    neural = [sys.executable, str(root / 'scripts/train_m3w_neural_cost_head.py'), *common,
              '--oof-cache-dir', str(tmp_path / 'ridge_cache'), '--seed', '1']
    def run(directory, extra=()):
        return subprocess.run([*neural, '--output-dir', str(tmp_path / directory), *extra],
                              check=True, capture_output=True, text=True, timeout=40)
    run('cli_direct')
    run('cli_resume', ['--stop-after', '5'])
    assert not (tmp_path / 'cli_resume/artifact.json').exists()
    run('cli_resume', ['--resume'])
    a, b = [torch.load(tmp_path / d / 'latest.pt', weights_only=True) for d in ('cli_direct', 'cli_resume')]
    assert a['losses'] == b['losses']
    for key in a['model']:
        torch.testing.assert_close(a['model'][key], b['model'][key], atol=0, rtol=0)
    r = json.loads((tmp_path / 'cli_resume/fit_report.json').read_text())
    ridge_report = json.loads((tmp_path / 'ridge_cache/fit_report.json').read_text())
    assert r['oof_feature_identity'] == ridge_report['oof_feature_identity']
    assert r['runtime']['interop_threads'] == 1 and r['runtime']['dataloader_workers'] == 0
    assert json.loads(run('cli_resume', ['--resume']).stdout)['result_source'] == 'cached_verified'
    original = plan['candidates'][0]
    candidates = []
    for folder in ('ridge_cache', 'cli_resume'):
        candidates.append({**original, 'id': folder, 'risk_head_id': folder,
                           'risk_report_path': f'{folder}/fit_report.json',
                           'risk_report_sha256': file_digest(tmp_path / folder / 'fit_report.json')})
    planpath = tmp_path / 'matched_plan.json'
    planpath.write_text(json.dumps({'schema_version': 1, 'candidates': candidates}))
    evaluation = [sys.executable, str(root / 'scripts/evaluate_m3w_development.py'), *common,
                  '--artifacts', *paths, str(tmp_path / 'ridge_cache/artifact.json'),
                  str(tmp_path / 'cli_resume/artifact.json'), '--plan', str(planpath),
                  '--output-dir', str(tmp_path / 'development')]
    subprocess.run(evaluation, check=True, capture_output=True, text=True, timeout=40)
    evaluated = json.loads((tmp_path / 'development/development_report.json').read_text())
    assert set(evaluated['summaries']) == {'ridge_cache', 'cli_resume'}
    assert not evaluated['calibration_or_confirmation_opened']
    reused = subprocess.run([*evaluation, '--resume'], check=True, capture_output=True, text=True, timeout=40)
    assert json.loads(reused.stdout)['status'] == 'cached_verified'
    cache = next((tmp_path / 'ridge_cache').glob('fold_*.npz'))
    with cache.open('ab') as stream:
        stream.write(b'changed')
    broken = subprocess.run([*neural, '--output-dir', str(tmp_path / 'refuse')], capture_output=True, text=True, timeout=40)
    assert broken.returncode != 0 and 'OOF cache bytes' in broken.stderr


@pytest.mark.skipif(os.environ.get('M3W_TEST_MPS') != '1', reason='Explicit MPS execution only; no resource probing')
def test_mps_neural_cost_resume_and_shared_scene_inference(tmp_path):
    from src.world_model.m3w_neural_gain_harm import train_neural_gain_harm, load_verified_neural_gain_harm
    from src.world_model.m3w_supervised_intervention import load_verified_forecaster
    from src.evaluation.m3w_development_evaluation import decide_scene
    from test_m3w_development_evaluation import POLICY, GEOMETRY
    contract, plan, groups = fitted_inputs(tmp_path)
    direct = train_neural_gain_harm(contract, groups, seed=1, output_dir=tmp_path / 'direct', device='mps')
    train_neural_gain_harm(contract, groups, seed=1, output_dir=tmp_path / 'neural', device='mps', stop_after=5)
    resumed = train_neural_gain_harm(contract, groups, seed=1, output_dir=tmp_path / 'neural', device='mps', resume=True)
    a, b = [torch.load(r['checkpoint'], weights_only=True, map_location='cpu') for r in (direct, resumed)]
    difference = max(float((a['model'][k] - b['model'][k]).abs().max()) for k in a['model'])
    assert difference <= 1e-6
    np.testing.assert_allclose(direct['losses'], resumed['losses'], rtol=0, atol=1e-6)
    contract = register_head(contract, resumed)
    head = load_verified_neural_gain_harm(contract, 'neural', device='mps')
    forecast = load_verified_forecaster(contract, 'fit_ab', device='mps')
    reader, _ = contract.open_recording('d', purpose='development')
    scene = reader.get_scene_inputs(100, 120)
    decisions = decide_scene(scene, forecast, head, baseline=plan['candidates'][0]['baseline'],
                             policy=POLICY, geometry=GEOMETRY, device='mps', solver_seconds=2.)
    assert np.isfinite(decisions['predicted_gain']).all()
    assert head.mean.device.type == 'mps' and next(forecast.parameters()).device.type == 'mps'
    (tmp_path / 'mps_evidence.json').write_text(json.dumps({
        'scope': 'synthetic_only_MPS_neural_cost_training_and_past_scene_inference',
        'fresh_run': True, 'updates_per_fit': 12, 'resume_at_update': 5,
        'max_parameter_difference': difference, 'max_loss_difference': float(np.max(np.abs(np.array(direct['losses']) - resumed['losses']))),
        'device': 'mps', 'implicit_cpu_fallback': False, 'real_training': False,
        'source_sha256': file_digest(Path(__file__).resolve().parents[1] / 'src/world_model/m3w_neural_gain_harm.py')}, indent=2))
