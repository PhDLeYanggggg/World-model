from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import types

import numpy as np
import pytest
import torch

from test_m3w_supervised_intervention import datasets, register, training_config
from src.world_model.m3w_eqmotion_adapter import EqMotionFixedHead, ROOT, SPEC, verified_source
from src.world_model.m3w_supervised_intervention import (
    build_forecaster, collate_forecasts, forecast_mse, load_verified_forecaster,
    make_oof_cost_rows, train_forecaster,
)


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def source():
    destination = ROOT / json.loads(SPEC.read_text())['destination']
    if not (destination / 'eth_ucy/model_t.py').is_file():
        pytest.skip('Optional pinned official source missing; fetch explicitly, never network from pytest')
    verified_source(destination)
    return destination


def architecture(source):
    return {'family': 'eqmotion_fixed_head', 'history_steps': 8, 'prediction_steps': 12,
            'hidden_nf': 8, 'channels': 8, 'layers': 2, 'fixed_head': 0,
            'source_directory': str(source)}


def batch():
    t = torch.arange(-7, 1, dtype=torch.float32) / 12
    xy = torch.stack((t, t.square() * .2), -1)
    history = torch.cat((xy, t[:, None]), -1)[None].repeat(2, 1, 1)
    neighbors = history[:, None].repeat(1, 3, 1, 1)
    neighbors[:, 0, :, 1] += .5
    neighbors[:, 1, :, 1] -= .3
    mask = torch.ones((2, 3, 8), dtype=torch.bool)
    mask[0, 1, 0] = False
    mask[:, 2] = False
    return {'history': history, 'history_mask': torch.ones((2, 8), dtype=torch.bool),
            'neighbors': neighbors, 'neighbor_mask': mask,
            'baseline': torch.zeros((2, 12, 2)),
            'prediction_time': torch.arange(1, 13)[None].repeat(2, 1).float() / 12,
            'request_mask': torch.ones((2, 12), dtype=torch.bool)}


def test_missing_source_refused_and_unknown_model_refused(tmp_path):
    with pytest.raises(ValueError, match='source missing'):
        build_forecaster(architecture(tmp_path))
    with pytest.raises(ValueError, match='family'):
        build_forecaster({'family': 'unknown'})


def test_all_pinned_bytes_checked_even_after_import(source, tmp_path):
    build_forecaster(architecture(source))
    copy = tmp_path / 'source'
    shutil.copytree(source, copy)
    path = copy / 'eth_ucy/gcl_t.py'
    path.write_bytes(path.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='identity mismatch'):
        build_forecaster(architecture(copy))


def test_fixed_head_and_backward_are_finite(source):
    model = build_forecaster(architecture(source))
    inputs = batch()
    prediction = model(inputs)
    assert prediction.shape == (2, 12, 2) and torch.isfinite(prediction).all()
    loss = forecast_mse(prediction, torch.ones_like(prediction), inputs['request_mask'])
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert model.core.predict_head[0].weight.grad is not None
    assert model.core.predict_head[1].weight.grad is None
    assert model.evaluation_semantics['emitted_candidates'] == 1


def test_original_math_matches_direct_author_import(source, monkeypatch):
    model = build_forecaster(architecture(source)).eval()
    code, _ = verified_source(source)
    package, gcl, original = (types.ModuleType(x) for x in ('eth_ucy', 'eth_ucy.gcl_t', 'author_direct'))
    package.__path__ = []
    monkeypatch.setitem(sys.modules, 'eth_ucy', package)
    monkeypatch.setitem(sys.modules, 'eth_ucy.gcl_t', gcl)
    exec(compile(code['eth_ucy/gcl_t.py'], 'gcl_t.py', 'exec'), gcl.__dict__)
    exec(compile(code['eth_ucy/model_t.py'], 'model_t.py', 'exec'), original.__dict__)
    direct = original.EqMotion(8, 0, 8, 8, 8, 12, n_layers=2, recurrent=True).eval()
    direct.load_state_dict(model.core.state_dict())
    inputs = batch()
    context = model.prepare_context(inputs)
    with torch.no_grad():
        expected, _ = direct(context['speed'], context['x'], context['velocity'], context['num_valid'])
        actual = model(inputs)
    torch.testing.assert_close(actual, expected[:, 0, 0], atol=0, rtol=0)


def test_neighbor_compaction_excludes_only_past_incomplete_or_misaligned(source):
    model = build_forecaster(architecture(source)).eval()
    inputs = batch()
    context = model.prepare_context(inputs)
    assert context['num_valid'].tolist() == [2, 3]
    torch.testing.assert_close(context['velocity'][:, :, 1:], context['x'][:, :, 1:] - context['x'][:, :, :-1])
    with torch.no_grad():
        before = model(inputs)
        inputs['neighbors'][~inputs['neighbor_mask']] = float('nan')
        # Even its observed values are irrelevant if the full past is unavailable.
        inputs['neighbors'][0, 1, 1:, :2] = 999
        after = model(inputs)
    torch.testing.assert_close(before, after, atol=0, rtol=0)
    inputs = batch()
    inputs['neighbors'][1, 1, 0, 2] -= .01
    assert model.prepare_context(inputs)['num_valid'].tolist() == [2, 2]


@pytest.mark.parametrize('violation', ['future_field', 'future_time', 'partial_ego', 'request_padding', 'irregular_grid', 'nonfinite'])
def test_context_boundary_refuses_illegal_inputs(source, violation):
    model, inputs = build_forecaster(architecture(source)), batch()
    if violation == 'future_field':
        inputs['future_endpoint'] = torch.zeros(2, 2)
    elif violation == 'future_time':
        inputs['neighbors'][0, 0, -1, 2] = .1
    elif violation == 'partial_ego':
        inputs['history_mask'][0, 0] = False
    elif violation == 'request_padding':
        inputs['request_mask'][0, -1] = False
    elif violation == 'irregular_grid':
        inputs['prediction_time'][0, 0] += .01
    else:
        inputs['history'][0, 0, 0] = float('nan')
    with pytest.raises(ValueError):
        model(inputs)


def test_neighbor_order_and_rigid_transform_equivariance(source):
    model, inputs = build_forecaster(architecture(source)).eval(), batch()
    with torch.no_grad():
        expected = model(inputs)
        permuted = deepcopy(inputs)
        for key in ('neighbors', 'neighbor_mask'):
            permuted[key] = permuted[key][:, [2, 1, 0]]
        torch.testing.assert_close(expected, model(permuted), atol=2e-6, rtol=2e-6)
        rotation = torch.tensor([[0., -1.], [1., 0.]])
        shift = torch.tensor([1.3, -.8])
        moved = deepcopy(inputs)
        for key in ('history', 'neighbors'):
            moved[key][..., :2] = moved[key][..., :2] @ rotation + shift
        moved['baseline'] = moved['baseline'] @ rotation + shift
        torch.testing.assert_close(model(moved), expected @ rotation + shift, atol=3e-5, rtol=3e-5)


def test_future_mutation_cannot_change_predictions(source, tmp_path):
    _, train, _ = datasets(tmp_path)
    before = train[0]
    reader = train.readers[0]
    changed = np.asarray(reader.points).copy()
    changed[changed[:, 0] > before['identity']['frame_id'], 2:] += 1000
    reader.points = changed
    after = train[0]
    assert not np.array_equal(before['target'], after['target'])
    model = build_forecaster(architecture(source)).eval()
    with torch.no_grad():
        torch.testing.assert_close(model(collate_forecasts([before])['inputs']),
                                   model(collate_forecasts([after])['inputs']), atol=0, rtol=0)


def test_resumable_fit_verified_load_and_oof_cost_path(source, tmp_path):
    contract, train, other = datasets(tmp_path)
    _, settings = training_config()
    arch = architecture(source)
    complete = train_forecaster(train, architecture=arch, settings=settings, output_dir=tmp_path / 'complete')
    train_forecaster(train, architecture=arch, settings=settings, output_dir=tmp_path / 'resume', stop_after=3)
    resumed = train_forecaster(train, architecture=arch, settings=settings, output_dir=tmp_path / 'resume', resume=True)
    a, b = (torch.load(r['checkpoint'], weights_only=True) for r in (complete, resumed))
    assert a['losses'] == b['losses'] and a['cursor'] == b['cursor']
    for key in a['model']:
        torch.testing.assert_close(a['model'][key], b['model'][key], atol=0, rtol=0)
    for key, values in a['optimizer']['state'].items():
        for name, value in values.items():
            torch.testing.assert_close(value, b['optimizer']['state'][key][name], atol=0, rtol=0)
    assert a['identity']['model_dependencies']['commit'] == json.loads(SPEC.read_text())['commit']
    contract = register(contract, complete)
    loaded = load_verified_forecaster(contract, 'fold_a', device='cpu')
    rows = make_oof_cost_rows(contract, 'fold_a', other, loaded, batch_size=4, device='cpu')
    assert len(rows['targets']) == len(other) and np.isfinite(rows['features']).all()
    with pytest.raises(ValueError, match='identity changed'):
        train_forecaster(train, architecture={**arch, 'fixed_head': 1}, settings=settings,
                         output_dir=tmp_path / 'resume', resume=True)


def test_checkpoint_dependency_manifest_tampering_refused(source, tmp_path):
    contract, train, _ = datasets(tmp_path)
    _, settings = training_config()
    result = train_forecaster(train, architecture=architecture(source), settings=settings, output_dir=tmp_path / 'fit')
    path = Path(result['checkpoint'])
    state = torch.load(path, weights_only=True)
    state['identity']['model_dependencies']['adapter_sha256'] = 'bad'
    torch.save(state, path)
    contract = register(contract, result)
    with pytest.raises(ValueError, match='dependency identity'):
        load_verified_forecaster(contract, 'fold_a', device='cpu')


def test_checkpoint_head_cannot_differ_from_frozen_training_identity(source, tmp_path):
    contract, train, _ = datasets(tmp_path)
    _, settings = training_config()
    result = train_forecaster(train, architecture=architecture(source), settings=settings, output_dir=tmp_path / 'fit')
    path = Path(result['checkpoint'])
    state = torch.load(path, weights_only=True)
    state['architecture'] = {**state['architecture'], 'fixed_head': 1}
    torch.save(state, path)
    contract = register(contract, result)
    with pytest.raises(ValueError, match='frozen training identity'):
        load_verified_forecaster(contract, 'fold_a', device='cpu')


def test_cross_process_checkpoint_resume(source, tmp_path):
    contract, _, _ = datasets(tmp_path)
    arch = architecture(source)
    _, settings = training_config()
    (tmp_path / 'protocol.json').write_text(json.dumps(contract.protocol))
    (tmp_path / 'config.json').write_text(json.dumps({'architecture': arch,
                                'fit_settings': {k: v for k, v in settings.items() if k != 'seed'}}))
    device = os.environ.get('M3W_EQMOTION_TEST_DEVICE', 'cpu')
    common = [sys.executable, str(ROOT / 'scripts/train_m3w_causal_forecaster.py'), '--protocol', str(tmp_path / 'protocol.json'),
              '--workspace-root', str(tmp_path), '--config', str(tmp_path / 'config.json'), '--fit-recordings', 'a',
              '--baseline', 'constant_velocity_causal_fd', '--seed', '1', '--threads', '2', '--device', device]
    for name, extra in (('resumed', ['--stop-after', '3']), ('resumed', ['--resume']), ('full', [])):
        subprocess.run(common + ['--output-dir', str(tmp_path / name)] + extra,
                       check=True, capture_output=True, timeout=120)
    a, b = (torch.load(tmp_path / name / 'latest.pt', weights_only=True) for name in ('resumed', 'full'))
    assert a['runtime']['device'] == device and a['runtime']['dataloader_workers'] == 0
    assert a['losses'] == b['losses'] and a['cursor'] == b['cursor']
    assert torch.equal(a['order'], b['order']) and torch.equal(a['torch_rng'], b['torch_rng'])
    for name in a['model']:
        torch.testing.assert_close(a['model'][name], b['model'][name], atol=0, rtol=0)
    for key, values in a['optimizer']['state'].items():
        assert values['step'].device.type == 'cpu'
        for name, value in values.items():
            torch.testing.assert_close(value, b['optimizer']['state'][key][name], atol=0, rtol=0)
