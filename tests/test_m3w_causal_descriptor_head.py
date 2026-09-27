import numpy as np
import pytest
import torch
from src.world_model import m3w_causal_descriptor_head as api
from src.world_model import m3w_fixed_floor_excess as control
from scripts.replay_m3w_dimensionless_training import exact


def sample():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(40, 5)).astype(np.float32)
    u = rng.normal(size=(40, 6)).astype(np.float32)
    y = np.abs(rng.normal(size=(40, 4))); y[:, [1, 3]] *= .01
    sites = np.array(['a']*20+['b']*20); env = np.ones(40)*3
    known = np.ones(40, bool); known[-1] = False; y[-1] = np.nan
    w = np.zeros(40); w[:20] = .5/20; w[20:39] = .5/19
    pr = dict(mean=x.mean(0), std=x.std(0), cost_scale=1., clip=10., known=known,
        weights=w, constant=(y[known]*w[known,None]).sum(0), training_sites=['a', 'b'])
    return x, u, y, sites, env, pr


def test_zero_descriptor_branch_preserves_initial_function_and_rng():
    *_, pr = sample()
    base = control.initialize(pr, 8, 17, 3.)
    expected_rng = torch.get_rng_state()
    new = api.initialize(pr, 8, 17, 3.)
    exact(expected_rng, torch.get_rng_state())
    for k, v in base.state_dict().items(): exact(v, new.state_dict()[k])
    assert torch.count_nonzero(new.descriptor.weight) == 0
    x = torch.randn(10, 5); u = torch.randn(10, 6); env = torch.ones(10)
    exact(base(x, env), new(x, env, u))


def test_descriptor_preprocessing_never_fits_unknown_targets():
    _, u, _, _, _, pr = sample()
    before = api.descriptor_preprocess(u, pr)
    u[-1] = 100000
    after = api.descriptor_preprocess(u, pr)
    exact(before, after)
    assert np.all(after['std'] > 0)


def test_lossless_atomic_checkpoint_and_unknown_draw_exclusion(tmp_path):
    x, u, y, sites, env, pr = sample()
    settings = dict(width=8, steps=5, batch_size=8, learning_rate=.0003,
        gradient_clip=5., checkpoint_every=2, heartbeat_every=2)
    args = dict(seed=17, settings=settings, identity={'test': True}, heartbeat=lambda **kw: None)
    model, info = api.fit(x, u, y, sites, env, pr, directory=tmp_path/'a', **args)
    assert info['unknown_rows_sampled'] == 0 and info['complete']
    api.fit(x, u, y, sites, env, pr, directory=tmp_path/'b', stop_at=2, **args)
    other, resumed = api.fit(x, u, y, sites, env, pr, directory=tmp_path/'b', resume=True, **args)
    exact(model.state_dict(), other.state_dict())
    a, b = [api.read_checkpoint(tmp_path/v/'checkpoint.pt.gz') for v in ('a', 'b')]
    for k in ('model', 'optimizer', 'sampler_rng', 'torch_rng', 'draws', 'trace'): exact(a[k], b[k])
    assert resumed['new_updates'] == 3
    with pytest.raises(ValueError): api.fit(x, u, y, sites, env, pr, directory=tmp_path/'a', **args)


def test_descriptor_prediction_accepts_no_targets():
    import inspect
    assert not set(('target', 'future', 'valid', 'future_mask')) & set(inspect.signature(api.predict).parameters)
