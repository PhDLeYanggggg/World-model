from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from test_m3w_supervised_intervention import datasets, register, training_config
from test_m3w_experiment_contract import approve
from src.evaluation.m3w_experiment_contract import ContractError, ExperimentContract, file_digest
from src.world_model.m3w_supervised_intervention import (
    ContractForecastDataset, load_verified_forecaster, make_oof_cost_rows, train_forecaster,
)
from src.world_model.m3w_cost_sensitive_deferral import (
    DeferralHead, absolute_cost_targets, comparator_spec, deferral_decision, deferral_surrogate,
    load_verified_deferral, make_oof_deferral_rows, train_deferral, validate_groups,
)


@pytest.fixture(autouse=True)
def limited_threads():
    original = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(original)


def setup(tmp_path, *, width=0, steps=20):
    contract, _, _ = datasets(tmp_path)
    protocol = deepcopy(contract.protocol)
    protocol['comparators'] = {'cost_sensitive_deferral': {
        'cost_bound': 2., 'width': width,
        'fit_settings': {'steps': steps, 'batch_size': 8, 'learning_rate': .02,
                         'checkpoint_every': 3, 'heartbeat_every': 3}}}
    approve(protocol)
    contract = ExperimentContract(protocol, tmp_path)
    a, b = [ContractForecastDataset(contract, [name], purpose='fit', baseline_name='constant_velocity_causal_fd')
            for name in ('a', 'b')]
    architecture, settings = training_config()
    result = train_forecaster(a, architecture=architecture, settings=settings, output_dir=tmp_path / 'producer')
    contract = register(contract, result)
    b = ContractForecastDataset(contract, ['b'], purpose='fit', baseline_name=b.baseline_name)
    model = load_verified_forecaster(contract, 'fold_a', device='cpu')
    group = make_oof_deferral_rows(contract, 'fold_a', b, model, batch_size=8, device='cpu')
    return contract, a, b, model, group


def test_surrogate_uses_opposite_cost_and_preserves_cost_mass():
    logits = torch.tensor([[0., 0.]], requires_grad=True)
    costs = torch.tensor([[.02, .9]], requires_grad=True)
    loss = deferral_surrogate(logits, costs)
    assert loss.item() == pytest.approx(.92)
    loss.backward()
    assert logits.grad[0, 0] < 0 and logits.grad[0, 1] > 0
    assert costs.grad is None
    scaled = deferral_surrogate(logits.detach(), costs.detach() / 2)
    assert scaled.item() == pytest.approx(loss.item() / 2)


def test_rare_large_harm_beats_majority_wins():
    costs = torch.tensor([[.02, .01]] * 9 + [[.02, 1.]])
    mean = costs.mean(0)
    best_p = mean.flip(0) / mean.sum()
    assert best_p.argmax() == 0  # Minimum expected cost: keep baseline.
    assert (costs.argmin(1) == 1).float().mean() == pytest.approx(.9)
    assert (costs.flip(1) / costs.sum(1, keepdim=True)).mean(0).argmax() == 1
    logits = best_p.log().repeat(10, 1).requires_grad_()
    deferral_surrogate(logits, costs).backward()
    torch.testing.assert_close(logits.grad.sum(0), torch.zeros(2), atol=2e-7, rtol=0)


@pytest.mark.parametrize('cost', [-.1, 1.1, float('nan'), float('inf')])
def test_invalid_costs_refused(cost):
    with pytest.raises(ValueError, match='bounded'):
        deferral_surrogate(torch.zeros(1, 2), torch.tensor([[cost, 0.]]))


def test_zero_costs_and_ties_do_not_force_switch():
    z = torch.zeros(3, 2, requires_grad=True)
    loss = deferral_surrogate(z, torch.zeros_like(z))
    loss.backward()
    assert loss == 0 and torch.equal(z.grad, torch.zeros_like(z))
    model = DeferralHead([0.], [1.], width=0)
    with torch.no_grad():
        model.network.weight.zero_()
        model.network.bias.zero_()
    out = deferral_decision(model, torch.zeros(3, 1), support=torch.ones(3, dtype=torch.bool))
    assert not out['use_candidate'].any() and not out['calibrated_risk']
    with torch.no_grad():
        model.network.bias[1] = 10
    out = deferral_decision(model, torch.zeros(3, 1), support=torch.tensor([True, False, True]))
    assert out['use_candidate'].tolist() == [True, False, True]


@pytest.mark.parametrize('metric', ['ade', 'fde'])
def test_absolute_costs_keep_both_errors_and_exact_endpoint(metric):
    b, c, y = np.zeros((2, 4, 2)), np.ones((2, 4, 2)), np.zeros((2, 4, 2))
    y[0, :2, 0] = 3
    request = np.array([[True, True, False, False], [True, True, True, True]])
    mask = request.copy()
    mask[1, -1] = False
    y[~mask] = np.nan
    costs, valid = absolute_cost_targets(b, c, y, mask, request, metric=metric)
    np.testing.assert_allclose(costs[0], [3, 5 ** .5])
    assert valid.tolist() == [True, metric == 'ade']
    if metric == 'fde':
        assert np.isnan(costs[1]).all()


def test_protocol_must_explicitly_bind_cost_bound(tmp_path):
    contract, _, _ = datasets(tmp_path)
    with pytest.raises(ValueError, match='explicitly bind'):
        comparator_spec(contract)


def test_oof_features_match_existing_control_and_future_only_changes_labels(tmp_path):
    contract, a, b, model, group = setup(tmp_path)
    old = make_oof_cost_rows(contract, 'fold_a', b, model, batch_size=8, device='cpu')
    np.testing.assert_array_equal(group['features'], old['features'])
    gain = group['targets'][:, 0] - group['targets'][:, 1]
    np.testing.assert_allclose(old['targets'], np.column_stack([np.maximum(gain, 0), np.maximum(-gain, 0)]), atol=1e-6)
    b.rows = b.rows[:1]
    before = make_oof_deferral_rows(contract, 'fold_a', b, model, batch_size=1, device='cpu')
    reader = b.readers[0]
    changed = np.array(reader.points)
    changed[changed[:, 0] > before['identities'][0]['frame_id'], 2:] += 100
    reader.points = changed
    after = make_oof_deferral_rows(contract, 'fold_a', b, model, batch_size=1, device='cpu')
    np.testing.assert_array_equal(before['features'], after['features'])
    assert not np.array_equal(before['targets'], after['targets'])
    with pytest.raises(ContractError, match='exposure'):
        make_oof_deferral_rows(contract, 'fold_a', a, model, batch_size=8, device='cpu')
    with torch.no_grad():
        next(model.parameters()).add_(1)
    with pytest.raises(ValueError, match='weights changed'):
        make_oof_deferral_rows(contract, 'fold_a', b, model, batch_size=8, device='cpu')


def test_duplicate_wrong_role_and_changed_producer_rejected(tmp_path):
    contract, _, _, _, group = setup(tmp_path)
    with pytest.raises(ValueError, match='Duplicated'):
        validate_groups(contract, [group, group])
    bad = deepcopy(group)
    bad['identities'][0]['data_role'] = 'confirmation'
    with pytest.raises(ValueError, match='fit-only'):
        validate_groups(contract, [bad])
    bad = {**group, 'predictor_sha256': 'changed'}
    with pytest.raises(ValueError, match='identity changed'):
        validate_groups(contract, [bad])
    bad = {**group, 'held_recordings': ['a', 'b']}
    with pytest.raises(ContractError, match='exposure'):
        validate_groups(contract, [bad])


@pytest.mark.parametrize('width', [0, 8])
def test_full_resume_and_normalization_exactly_match(tmp_path, width):
    contract, _, _, _, group = setup(tmp_path, width=width)
    full = train_deferral(contract, [group], seed=1, output_dir=tmp_path / 'full')
    part = train_deferral(contract, [group], seed=1, output_dir=tmp_path / 'resume', stop_after=5)
    assert not part['training_complete'] and part['steps_completed'] == 5
    done = train_deferral(contract, [group], seed=1, output_dir=tmp_path / 'resume', resume=True)
    assert full['losses'] == done['losses'] and done['resumed_from_step'] == 5
    x, y = [torch.load(r['checkpoint'], weights_only=True) for r in (full, done)]
    for k in x['model']:
        torch.testing.assert_close(x['model'][k], y['model'][k], rtol=0, atol=0)
    for k in ('order', 'sampler_rng', 'torch_rng'):
        assert torch.equal(x[k], y[k])
    np.testing.assert_allclose(x['model']['mean'], group['features'].mean(0), atol=1e-5)
    assert (tmp_path / 'resume/heartbeat.jsonl').exists()
    again = train_deferral(contract, [group], seed=1, output_dir=tmp_path / 'resume', resume=True)
    assert again['result_source'] == 'cached_verified' and again['checkpoint_sha256'] == done['checkpoint_sha256']
    modified = deepcopy(group)
    modified['targets'][0, 0] += .1
    with pytest.raises(ValueError, match='identity changed'):
        train_deferral(contract, [modified], seed=1, output_dir=tmp_path / 'resume', resume=True)


def test_actual_head_learns_rare_harm_example(tmp_path):
    contract, _, _, _, group = setup(tmp_path, steps=200)
    # Constructed-cost optimization fixture, not trajectory performance evidence.
    n = len(group['features'])
    group['features'] = np.zeros((n, 1), np.float32)
    group['targets'] = np.tile([.02, .01], (n, 1)).astype(np.float32)
    group['targets'][::10, 1] = 1
    result = train_deferral(contract, [group], seed=1, output_dir=tmp_path / 'head')
    state = torch.load(result['checkpoint'], weights_only=True)
    head = DeferralHead([0.], [1.], width=0)
    head.load_state_dict(state['model'])
    choice = deferral_decision(head, torch.zeros(1, 1), support=torch.ones(1, dtype=torch.bool))
    assert not choice['use_candidate'].item()


def test_cost_clipping_is_visible_and_partial_checkpoint_cannot_load(tmp_path):
    contract, _, _, _, group = setup(tmp_path)
    group['targets'][:] = [3., .5]
    partial = train_deferral(contract, [group], seed=1, output_dir=tmp_path / 'partial', stop_after=1)
    assert partial['cost_clip_fraction_by_action'] == [1., 0.]
    artifact = {'id': 'deferral', 'kind': 'policy', 'family': 'cost_sensitive_deferral',
                'path': 'partial/latest.pt', 'sha256': partial['checkpoint_sha256'],
                'protocol_sha256': contract.digest, 'lineage_complete': True,
                'parents': partial['parents'], 'fit_recordings': partial['fit_recordings'],
                'selection_recordings': [], 'calibration_recordings': []}
    verified = ExperimentContract(contract.protocol, tmp_path, [*contract.artifacts.values(), artifact])
    with pytest.raises(ValueError, match='incomplete'):
        load_verified_deferral(verified, 'deferral')


def test_nonfinite_scores_and_changed_protocol_refused(tmp_path):
    model = DeferralHead([0.], [1.], width=0)
    with torch.no_grad():
        model.network.bias[0] = float('nan')
    with pytest.raises(ValueError, match='Nonfinite'):
        deferral_decision(model, torch.zeros(1, 1), support=torch.ones(1, dtype=torch.bool))
    contract, _, _, _, _ = setup(tmp_path)
    contract.protocol['comparators']['cost_sensitive_deferral']['cost_bound'] = 3
    with pytest.raises(ContractError, match='identity changed'):
        comparator_spec(contract)


def test_real_draft_refuses_training():
    script = Path(__file__).resolve().parents[1] / 'scripts/train_m3w_deferral_control.py'
    result = subprocess.run([sys.executable, str(script), '--preflight-only'], capture_output=True, text=True, timeout=20)
    assert result.returncode == 2
    assert not json.loads(result.stdout)['deferral_training_started']


def test_cross_process_fold_and_optimizer_resume_and_load(tmp_path):
    contract, a, b, _, _ = setup(tmp_path)
    architecture, settings = training_config()
    second = train_forecaster(b, architecture=architecture, settings=settings, output_dir=tmp_path / 'second')
    artifact_b = register(contract, second).artifacts['fold_a']
    artifact_b['id'] = 'fold_b'
    records = [contract.artifacts['fold_a'], artifact_b]
    manifest_paths = []
    for record in records:
        path = tmp_path / f'{record["id"]}.json'
        path.write_text(json.dumps(record))
        manifest_paths.append(str(path))
    protocol, mapping = tmp_path / 'protocol.json', tmp_path / 'mapping.json'
    protocol.write_text(json.dumps(contract.protocol))
    mapping.write_text(json.dumps({'0': 'fold_b', '1': 'fold_a'}))
    script = Path(__file__).resolve().parents[1] / 'scripts/train_m3w_deferral_control.py'
    output = tmp_path / 'deferral'
    command = [sys.executable, str(script), '--protocol', str(protocol), '--workspace-root', str(tmp_path),
               '--artifacts', *manifest_paths, '--fold-models', str(mapping), '--baseline', a.baseline_name,
               '--seed', '1', '--threads', '2', '--output-dir', str(output)]
    subprocess.run(command + ['--stop-after-folds', '1'], check=True, capture_output=True, timeout=30)
    subprocess.run(command + ['--resume', '--stop-after', '5'], check=True, capture_output=True, timeout=30)
    assert not (output / 'artifact.json').exists()
    cache = output / 'fold_0.npz'
    original = cache.read_bytes()
    cache.write_bytes(original + b'drift')
    result = subprocess.run(command + ['--resume'], capture_output=True, text=True, timeout=30)
    assert result.returncode != 0 and 'cache changed' in result.stderr
    cache.write_bytes(original)
    subprocess.run(command + ['--resume'], check=True, capture_output=True, timeout=30)
    report = json.loads((output / 'fit_report.json').read_text())
    assert report['training_rows'] == len(a) + len(b) and report['training_complete']
    assert report['oof_folds_reused_cached_verified'] == 2 and not report['test_evaluated']
    policy = json.loads((output / 'artifact.json').read_text())
    verified = ExperimentContract(contract.protocol, tmp_path, [*records, policy])
    head = load_verified_deferral(verified, policy['id'])
    assert not head.training and policy['fit_recordings'] == ['a', 'b']
    before = file_digest(output / 'head/latest.pt')
    subprocess.run(command + ['--resume'], check=True, capture_output=True, timeout=30)
    assert file_digest(output / 'head/latest.pt') == before
