import copy
import hashlib
import inspect
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from src.world_model import m3w_completion_cost as api
from tests.test_m3w_scene_context_cost import scene_sample, SETTINGS


def sample(unknown=True):
    x, scene, envelope, data = scene_sample()
    labels = data['truth'].numpy().copy()
    if unknown:
        labels[::2] = np.nan
    bound, known = api.completion_endpoints(labels, envelope)
    data['known'] = torch.from_numpy(known)
    data['truth'] = torch.from_numpy(np.where(known[:, None, None], labels, 0.).astype(np.float32))
    data['bound_truth'] = torch.from_numpy(bound.astype(np.float32))
    data['groups'] = [np.arange(len(x))]
    data['input_hashes'] = dict(data['input_hashes'], fixture_completion=hashlib.sha256(bound.tobytes()).hexdigest())
    data['draw_pool'] = dict(rows=len(x), known_rows=int(known.sum()), unknown_rows=int((~known).sum()))
    return x, scene, envelope, data, labels


def test_endpoints_do_not_impute_or_modify_original_unknown_costs():
    _, _, envelope, _, labels = sample()
    original = labels.copy()
    bound, known = api.completion_endpoints(labels, envelope)
    np.testing.assert_array_equal(labels, original)
    assert np.isnan(labels[~known]).all()
    np.testing.assert_array_equal(bound[known], labels[known])
    np.testing.assert_array_equal(bound[~known][..., (0, 2, 3)], 0.)
    np.testing.assert_array_equal(bound[~known, :, 1], envelope[~known])
    np.testing.assert_array_equal(bound[~known, :, 4], envelope[~known])


def test_malformed_support_or_invalid_envelope_rejected():
    _, _, env, _, labels = sample()
    partial = labels.copy(); partial[0, 0, 0] = 0.
    with pytest.raises(ValueError):
        api.completion_endpoints(partial, env)
    with pytest.raises(ValueError):
        api.completion_endpoints(labels, -env)


def test_unknown_endpoints_bound_random_masked_future_completions():
    rng = np.random.default_rng(29)
    floor = rng.normal(size=(11, 12, 2))
    candidate = rng.normal(size=(11, 7, 12, 2))
    env = np.linalg.norm(candidate-floor[:, None], axis=-1).max(2)
    bound, known = api.completion_endpoints(np.full((11, 7, 5), np.nan), env)
    assert not known.any()
    for _ in range(20):
        target = 20*rng.normal(size=floor.shape)
        mask = rng.random((11, 12)) > .4; mask[:, 0] = True
        r = (np.linalg.norm(floor-target, axis=-1)*mask).sum(1)/mask.sum(1)
        n = (np.linalg.norm(candidate-target[:, None], axis=-1)*mask[:, None]).sum(2)/mask.sum(1)[:, None]
        benefit, harm = np.maximum(r[:, None]-n, 0), np.maximum(n-r[:, None], 0)
        assert np.all(bound[..., 0] <= benefit)
        assert np.all(bound[..., 1]+1e-12 >= harm)
        assert np.all(bound[..., 2] <= r[:, None])
        assert np.all(bound[..., 4]+1e-12 >= harm)


def test_common_draw_pool_does_not_filter_by_future_label_support():
    _, _, env, data, labels = sample()
    names = np.array(['a']*16+['b']*16)
    packets = {name: dict(y=labels[names == name], envelope=env[names == name],
        scales=np.ones(16), sites=names[names == name]) for name in ('a', 'b')}
    fake = SimpleNamespace(training_data=lambda _: copy.deepcopy(data),
        fingerprint=lambda a: hashlib.sha256(np.asarray(a).tobytes()).hexdigest())
    built = api.training_data(packets, fake)
    assert built['draw_pool']['unknown_rows'] == 16
    assert not built['draw_pool']['future_support_used_for_group_membership']
    for p in packets.values():
        p['y'] = np.nan_to_num(p['y'], nan=0.)
    changed = api.training_data(packets, fake)
    for left, right in zip(built['groups'], changed['groups']):
        np.testing.assert_array_equal(left, right)
    assert changed['draw_pool']['unknown_rows'] == 0


def test_real_resume_and_paired_draws_all_four_cells(tmp_path):
    torch.set_num_threads(1)
    _, _, _, data, _ = sample(); states = []
    for bank in api.BANKS:
        for mode in api.MODES:
            kw = dict(data=data, mode=mode, bank=bank, settings=SETTINGS,
                      seed=17, identity=dict(fixture=True))
            direct = api.fit(**kw, path=tmp_path/f'{bank}_{mode}_direct.gz')
            path = tmp_path/f'{bank}_{mode}_resume.gz'
            api.fit(**kw, path=path, stop_at=2)
            resumed = api.fit(**kw, path=path, resume=True)
            api.exact_training(direct, resumed); states.append(direct)
            assert direct['unknown_rows_drawn'] > 0
            assert direct['unknown_rows_with_loss'] > 0 if mode == 'completion_bounds' else direct['unknown_rows_with_loss'] == 0
    for state in states[1:]:
        api.exact_paired_design(states[0], state)
    assert any(not torch.equal(states[0]['model'][k], states[1]['model'][k]) for k in states[0]['model'])


def test_all_known_is_exact_neutral_and_inference_rejects_future_fields(tmp_path):
    torch.set_num_threads(1)
    x, scene, env, data, _ = sample(unknown=False)
    kw = dict(data=data, bank='physical_neural', settings=SETTINGS, seed=17, identity={})
    a = api.fit(**kw, mode='known_only', path=tmp_path/'known.gz')
    b = api.fit(**kw, mode='completion_bounds', path=tmp_path/'bound.gz')
    for key in ('model', 'optimizer', 'draw_hash', 'sampler_rng', 'torch_rng'):
        api.base.exact(a[key], b[key])
    assert np.isfinite(api.predict(b, x, scene, env, np.ones(len(x)))).all()
    assert set(inspect.signature(api.predict).parameters) == {'state', 'x', 'static_scene', 'envelope', 'scales'}
    with pytest.raises(TypeError):
        api.predict(b, x, scene, env, np.ones(len(x)), future_mask=np.ones((len(x), 12)))


def test_resume_rejects_changed_target_hash_and_inactive_neural_cannot_change_physical(tmp_path):
    torch.set_num_threads(1)
    _, _, _, data, _ = sample(); kw = dict(mode='completion_bounds', bank='physical_only', settings=SETTINGS, seed=17, identity={})
    state = api.fit(data, **kw, path=tmp_path/'original.gz')
    changed = copy.deepcopy(data)
    changed['z'].motion[:, 6] += 1e4; changed['z'].scene[:, 6] += 1e4
    changed['bound_truth'][:, 6] += 1e4
    other = api.fit(changed, **kw, path=tmp_path/'inactive.gz')
    api.base.exact(state['model'], other['model'])
    changed['input_hashes']['fixture_completion'] = 'changed'
    with pytest.raises(ValueError):
        api.fit(changed, **kw, path=tmp_path/'original.gz', resume=True)
