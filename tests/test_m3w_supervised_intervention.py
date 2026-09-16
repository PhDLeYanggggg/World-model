from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from test_m3w_experiment_contract import fixture_contract, approve
from src.data_unification.m3w_causal_recordings import write_recording
from src.evaluation.m3w_experiment_contract import ContractError, ExperimentContract, file_digest
from src.world_model.m3w_supervised_intervention import (
    INPUT_KEYS, ContractForecastDataset, PastContextForecaster, GainHarmHead,
    collate_forecasts, cost_targets, forecast_mse, load_verified_forecaster,
    make_oof_cost_rows, risk_features, train_forecaster,
    fit_linear_gain_harm, predict_linear_gain_harm,
)


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def datasets(tmp_path):
    protocol = fixture_contract(tmp_path)
    for phase, name in enumerate(protocol['records']):
        points = np.array([[t * 10, agent, phase + .12 * t + .002 * t * t * agent,
                            .2 * agent + .3 * np.sin(t / 4 + phase + agent)]
                           for agent, length in ((1, 40), (2, 30)) for t in range(length)])
        write_recording(tmp_path / name, points, {'id': name, 'physical_scene': name})
        protocol['records'][name]['metadata_sha256'] = file_digest(tmp_path / name / 'metadata.json')
    approve(protocol)
    contract = ExperimentContract(protocol, tmp_path)
    train = ContractForecastDataset(contract, ['a'], purpose='fit', baseline_name='constant_velocity_causal_fd')
    other = ContractForecastDataset(contract, ['b'], purpose='fit', baseline_name='constant_velocity_causal_fd')
    return contract, train, other


def training_config():
    return {'width': 8, 'heads': 2, 'layers': 1}, {
        'seed': 1, 'steps': 8, 'batch_size': 4, 'learning_rate': .003,
        'checkpoint_every': 2, 'heartbeat_every': 2,
    }


def register(contract, result):
    from pathlib import Path
    path = Path(result['checkpoint'])
    record = {'id': 'fold_a', 'kind': 'forecaster', 'path': str(path.relative_to(contract.root)),
              'sha256': file_digest(path), 'protocol_sha256': contract.digest,
              'lineage_complete': True, 'fit_recordings': result['fit_recordings'],
              'selection_recordings': [], 'calibration_recordings': [], 'parents': []}
    return ExperimentContract(contract.protocol, contract.root, [record])


def test_dataset_cannot_open_confirmation_or_train_on_development(tmp_path):
    contract, train, _ = datasets(tmp_path)
    with pytest.raises(ValueError, match='fit/development'):
        ContractForecastDataset(contract, ['t'], purpose='confirmation', baseline_name=train.baseline_name)
    with pytest.raises(ContractError, match='role'):
        ContractForecastDataset(contract, ['t'], purpose='fit', baseline_name=train.baseline_name)
    dev = ContractForecastDataset(contract, ['d'], purpose='development', baseline_name=train.baseline_name)
    architecture, settings = training_config()
    with pytest.raises(ValueError, match='fit recordings'):
        train_forecaster(dev, architecture=architecture, settings=settings, output_dir=tmp_path / 'bad')


def test_training_schema_and_model_predictions_ignore_changed_future(tmp_path):
    _, train, _ = datasets(tmp_path)
    before = train[0]
    reader = train.readers[0]
    changed = np.asarray(reader.points).copy()
    changed[changed[:, 0] > before['identity']['frame_id'], 2:] += 1000
    reader.points = changed
    after = train[0]
    assert set(before['inputs']) == INPUT_KEYS
    assert not np.array_equal(before['target'], after['target'])
    for key in INPUT_KEYS:
        np.testing.assert_array_equal(before['inputs'][key], after['inputs'][key])
    model = PastContextForecaster(width=8, heads=2, layers=1).eval()
    b, a = collate_forecasts([before]), collate_forecasts([after])
    with torch.no_grad():
        pb, pa = model(b['inputs']), model(a['inputs'])
    torch.testing.assert_close(pb, pa, atol=0, rtol=0)
    torch.testing.assert_close(risk_features(b['inputs'], pb), risk_features(a['inputs'], pa), atol=0, rtol=0)


def test_unknown_inputs_and_post_current_tokens_are_refused(tmp_path):
    _, train, _ = datasets(tmp_path)
    batch = collate_forecasts([train[0]])
    model = PastContextForecaster(width=8, heads=2, layers=1)
    with pytest.raises(ValueError, match='causal input'):
        model({**batch['inputs'], 'future_endpoint': batch['target'][:, -1]})
    batch['inputs']['history'][0, 0, 2] = .1
    with pytest.raises(ValueError, match='post-current'):
        model(batch['inputs'])


def test_masked_neighbor_values_do_not_change_prediction(tmp_path):
    _, train, _ = datasets(tmp_path)
    batch = collate_forecasts([train[0], train[1]])
    model = PastContextForecaster(width=8, heads=2, layers=1).eval()
    with torch.no_grad():
        before = model(batch['inputs'])
        batch['inputs']['neighbors'][~batch['inputs']['neighbor_mask']] = float('nan')
        after = model(batch['inputs'])
    torch.testing.assert_close(before, after, atol=0, rtol=0)


def test_ragged_request_endpoint_is_not_padded_or_last_observed_endpoint():
    baseline = np.zeros((2, 4, 2))
    candidate = np.ones_like(baseline)
    target = np.zeros_like(baseline)
    request = np.array([[True, True, False, False], [True, True, True, True]])
    mask = request.copy()
    mask[1, -1] = False
    costs, valid = cost_targets(baseline, candidate, target, mask, request, metric='fde')
    assert valid.tolist() == [True, False]
    assert costs[0, 0] == 0 and costs[0, 1] == pytest.approx(2 ** .5)
    assert np.isnan(costs[1]).all()
    _, ade_valid = cost_targets(baseline, candidate, target, mask, request, metric='ade')
    assert ade_valid.all()


def test_cost_labels_are_not_best_class_or_raw_unnormalized_errors():
    baseline = np.zeros((2, 3, 2))
    candidate = np.ones_like(baseline)
    target = np.stack([np.ones((3, 2)), np.zeros((3, 2))])
    mask = np.ones((2, 3), bool)
    cost, valid = cost_targets(baseline, candidate, target, mask, mask, metric='ade')
    np.testing.assert_allclose(cost, [[2 ** .5, 0], [0, 2 ** .5]])
    assert valid.all()


def test_gain_harm_heads_are_coherent():
    head = GainHarmHead(5, width=8)
    prediction = head(torch.randn(20, 5))
    assert (prediction['benefit'] >= 0).all() and (prediction['harm'] >= 0).all()
    assert torch.all(prediction['harm'] + 1e-6 >= -prediction['gain'])


def test_loss_ignores_missing_labels_and_does_not_accept_no_labels():
    prediction = torch.ones(2, 3, 2, requires_grad=True)
    target = torch.zeros_like(prediction)
    target[0, 2] = float('nan')
    mask = torch.ones(2, 3, dtype=torch.bool)
    mask[0, 2] = False
    loss = forecast_mse(prediction, target, mask)
    loss.backward()
    assert loss == 1 and torch.isfinite(prediction.grad).all()
    with pytest.raises(ValueError, match='valid label'):
        forecast_mse(prediction, target, mask & False)


def test_checkpoint_resume_reproduces_optimizer_and_sampler(tmp_path):
    _, train, _ = datasets(tmp_path)
    architecture, settings = training_config()
    complete = train_forecaster(train, architecture=architecture, settings=settings, output_dir=tmp_path / 'complete')
    partial = train_forecaster(train, architecture=architecture, settings=settings, output_dir=tmp_path / 'resume', stop_after=3)
    assert not partial['training_complete'] and partial['steps_completed'] == 3
    resumed = train_forecaster(train, architecture=architecture, settings=settings, output_dir=tmp_path / 'resume', resume=True)
    a = torch.load(complete['checkpoint'], weights_only=True)
    b = torch.load(resumed['checkpoint'], weights_only=True)
    assert a['losses'] == b['losses'] and a['cursor'] == b['cursor']
    assert torch.equal(a['order'], b['order']) and torch.equal(a['sampler_rng'], b['sampler_rng'])
    assert torch.equal(a['torch_rng'], b['torch_rng'])
    for key in a['model']:
        torch.testing.assert_close(a['model'][key], b['model'][key], atol=0, rtol=0)
    assert a['optimizer']['param_groups'] == b['optimizer']['param_groups']
    for key, values in a['optimizer']['state'].items():
        for name, value in values.items():
            torch.testing.assert_close(value, b['optimizer']['state'][key][name], atol=0, rtol=0)
    assert (tmp_path / 'resume/heartbeat.jsonl').is_file()
    stable = file_digest(Path(resumed['checkpoint']))
    verified = train_forecaster(train, architecture=architecture, settings=settings, output_dir=tmp_path / 'resume', resume=True)
    assert verified['result_source'] == 'cached_verified' and file_digest(Path(verified['checkpoint'])) == stable


def test_resume_cannot_change_baseline_or_settings(tmp_path):
    _, train, _ = datasets(tmp_path)
    architecture, settings = training_config()
    train_forecaster(train, architecture=architecture, settings=settings, output_dir=tmp_path / 'resume', stop_after=3)
    with pytest.raises(ValueError, match='identity changed'):
        train_forecaster(train, architecture=architecture, settings={**settings, 'learning_rate': .02},
                         output_dir=tmp_path / 'resume', resume=True)


def test_verified_oof_costs_and_in_memory_weight_tampering(tmp_path):
    contract, train, other = datasets(tmp_path)
    architecture, settings = training_config()
    result = train_forecaster(train, architecture=architecture, settings=settings, output_dir=tmp_path / 'fit')
    contract = register(contract, result)
    predictor = load_verified_forecaster(contract, 'fold_a', device='cpu')
    rows = make_oof_cost_rows(contract, 'fold_a', other, predictor, batch_size=4, device='cpu')
    assert len(rows['targets']) == len(other)
    assert all(r['recording_id'] == 'b' for r in rows['identities'])
    assert np.isfinite(rows['features']).all() and np.isfinite(rows['targets']).all()
    with pytest.raises(ContractError, match='exposure'):
        make_oof_cost_rows(contract, 'fold_a', train, predictor, batch_size=4, device='cpu')
    with torch.no_grad():
        next(predictor.parameters()).add_(1)
    with pytest.raises(ValueError, match='weights changed'):
        make_oof_cost_rows(contract, 'fold_a', other, predictor, batch_size=4, device='cpu')


def test_checkpoint_contents_bind_declared_fit_provenance(tmp_path):
    contract, train, _ = datasets(tmp_path)
    architecture, settings = training_config()
    result = train_forecaster(train, architecture=architecture, settings=settings, output_dir=tmp_path / 'fit')
    honest = register(contract, result)
    record = deepcopy(honest.artifacts['fold_a'])
    record['fit_recordings'] = ['b']
    dishonest = ExperimentContract(contract.protocol, contract.root, [record])
    with pytest.raises(ValueError, match='fit provenance'):
        load_verified_forecaster(dishonest, 'fold_a', device='cpu')


def test_partial_fixed_budget_checkpoint_cannot_be_used_as_finished_teacher(tmp_path):
    contract, train, _ = datasets(tmp_path)
    architecture, settings = training_config()
    partial = train_forecaster(train, architecture=architecture, settings=settings, output_dir=tmp_path / 'partial', stop_after=2)
    contract = register(contract, partial)
    with pytest.raises(ValueError, match='incomplete'):
        load_verified_forecaster(contract, 'fold_a', device='cpu')


def test_simple_cost_regression_uses_only_verified_oof_supervision(tmp_path):
    contract, train, other = datasets(tmp_path)
    architecture, settings = training_config()
    result = train_forecaster(train, architecture=architecture, settings=settings, output_dir=tmp_path / 'fit')
    contract = register(contract, result)
    predictor = load_verified_forecaster(contract, 'fold_a', device='cpu')
    rows = make_oof_cost_rows(contract, 'fold_a', other, predictor, batch_size=4, device='cpu')
    head = fit_linear_gain_harm(contract, [rows], alpha=1.)
    assert head['fit_recordings'] == ['b'] and head['parents'] == ['fold_a']
    scores = predict_linear_gain_harm(head, rows['features'])
    assert np.isfinite(scores['gain']).all()
    assert np.all(scores['harm'] + 1e-6 >= -scores['gain'])
    with pytest.raises(ValueError, match='Duplicated'):
        fit_linear_gain_harm(contract, [rows, rows], alpha=1.)


def test_real_draft_refuses_training_before_torch_import():
    from pathlib import Path
    import subprocess
    import sys
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, str(root / 'scripts/train_m3w_causal_forecaster.py'), '--preflight-only'],
                            cwd=root, capture_output=True, text=True, timeout=20)
    assert result.returncode == 2
    message = json.loads(result.stdout)
    assert message['status'] == 'refused' and not message['torch_training_started']


def test_cli_cross_process_resume_with_synthetic_recordings(tmp_path):
    from pathlib import Path
    import subprocess
    import sys
    contract, _, _ = datasets(tmp_path)
    architecture, settings = training_config()
    protocol_path, config_path = tmp_path / 'protocol.json', tmp_path / 'config.json'
    protocol_path.write_text(json.dumps(contract.protocol))
    config_path.write_text(json.dumps({'architecture': architecture,
                                     'fit_settings': {k: v for k, v in settings.items() if k != 'seed'}}))
    script = Path(__file__).resolve().parents[1] / 'scripts/train_m3w_causal_forecaster.py'
    common = [sys.executable, str(script), '--protocol', str(protocol_path), '--workspace-root', str(tmp_path),
              '--config', str(config_path), '--fit-recordings', 'a', '--baseline', 'constant_velocity_causal_fd',
              '--seed', '1', '--threads', '2']
    subprocess.run(common + ['--output-dir', str(tmp_path / 'separate_resume'), '--stop-after', '3'],
                   check=True, capture_output=True, timeout=30)
    subprocess.run(common + ['--output-dir', str(tmp_path / 'separate_resume'), '--resume'],
                   check=True, capture_output=True, timeout=30)
    subprocess.run(common + ['--output-dir', str(tmp_path / 'separate_full')],
                   check=True, capture_output=True, timeout=30)
    a = torch.load(tmp_path / 'separate_resume/latest.pt', weights_only=True)
    b = torch.load(tmp_path / 'separate_full/latest.pt', weights_only=True)
    assert a['losses'] == b['losses']
    for key in a['model']:
        torch.testing.assert_close(a['model'][key], b['model'][key], atol=0, rtol=0)
    record = json.loads((tmp_path / 'separate_resume/artifact.json').read_text())
    assert record['fit_recordings'] == ['a'] and record['selection_recordings'] == []


def test_cost_head_cli_fold_checkpoint_resume_and_tamper_detection(tmp_path):
    from pathlib import Path
    import subprocess
    import sys
    contract, a, b = datasets(tmp_path)
    architecture, settings = training_config()
    manifests = []
    for name, dataset in (('model_a', a), ('model_b', b)):
        result = train_forecaster(dataset, architecture=architecture, settings=settings, output_dir=tmp_path / name)
        record = register(contract, result).artifacts['fold_a']
        record['id'] = name
        path = tmp_path / f'{name}.json'
        path.write_text(json.dumps(record))
        manifests.append(str(path))
    protocol, mapping = tmp_path / 'protocol.json', tmp_path / 'fold_models.json'
    protocol.write_text(json.dumps(contract.protocol))
    mapping.write_text(json.dumps({'0': 'model_b', '1': 'model_a'}))
    script = Path(__file__).resolve().parents[1] / 'scripts/train_m3w_oof_cost_head.py'
    output = tmp_path / 'cost_head'
    command = [sys.executable, str(script), '--protocol', str(protocol), '--workspace-root', str(tmp_path),
               '--artifacts', *manifests, '--fold-models', str(mapping), '--baseline', 'constant_velocity_causal_fd',
               '--alpha', '1', '--threads', '2', '--output-dir', str(output)]
    subprocess.run(command + ['--stop-after-folds', '1'], check=True, capture_output=True, timeout=30)
    assert not (output / 'fit_report.json').exists()
    cache = output / 'fold_0.npz'
    original = cache.read_bytes()
    cache.write_bytes(original + b'changed')
    rejected = subprocess.run(command + ['--resume'], capture_output=True, text=True, timeout=30)
    assert rejected.returncode != 0 and 'changed' in rejected.stderr
    cache.write_bytes(original)
    subprocess.run(command + ['--resume'], check=True, capture_output=True, timeout=30)
    report = json.loads((output / 'fit_report.json').read_text())
    assert report['training_rows'] == len(a) + len(b)
    assert report['oof_folds_reused_cached_verified'] == 1
    assert report['parents'] == ['model_a', 'model_b'] and not report['test_evaluated']
    stable = file_digest(output / 'linear_cost_head.npz')
    repeat = subprocess.run(command + ['--resume'], check=True, capture_output=True, text=True, timeout=30)
    assert 'already_complete_cached_verified' in repeat.stdout
    assert file_digest(output / 'linear_cost_head.npz') == stable
