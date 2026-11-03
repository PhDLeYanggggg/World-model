import numpy as np
import pytest
import torch

from src.world_model import m3w_interaction_feature_cost as api
from tests.test_m3w_bound_aware_cost import example


def data():
    old = example()
    c = api.attach(old['causal'], np.random.default_rng(23).normal(size=(12, 64)).astype(np.float32))
    pr = dict(old['preprocess'])
    pr['mean'] = np.r_[pr['mean'], np.zeros(64)]
    pr['std'] = np.r_[pr['std'], np.ones(64)]
    z = np.clip((c['x']-pr['mean'])/pr['std'], -pr['clip'], pr['clip']).astype(np.float32)
    return dict(old, causal=c, preprocess=pr, z=torch.from_numpy(z))


def test_zero_ablation_preserves_all_original_data_and_targets():
    d = data()
    a = api.for_arm(d, 'zero64')
    assert torch.count_nonzero(a['z'][..., -64:]) == 0
    torch.testing.assert_close(a['z'][..., :-64], d['z'][..., :-64], rtol=0, atol=0)
    assert a['truth_by_arm'] is d['truth_by_arm'] and a['preprocess'] is d['preprocess']
    assert torch.count_nonzero(d['z'][..., -64:]) > 0
    with pytest.raises(ValueError):
        api.attach(dict(d['causal'], future_endpoint=np.ones((12, 2))), np.ones((12, 64), np.float32))


def test_real_optimizer_resume_and_capacity_draw_pairing(tmp_path):
    torch.set_num_threads(1)
    d = data()
    settings = dict(width=8, learning_rate=.001, query_batch_size=4, minimum_updates=8,
                    largest_source_cycles=2, gradient_clip=5., heartbeat_every=2, checkpoint_every=2)
    states = []
    for arm in api.ARMS:
        kwargs = dict(arm=arm, seed=17, settings=settings, identity={'synthetic': True})
        path = tmp_path/(arm+'.pt.gz')
        api.fit(d, path=path, stop_at=4, **kwargs)
        a = api.fit(d, path=path, resume=True, **kwargs)
        b = api.fit(d, path=tmp_path/(arm+'_direct.pt.gz'), **kwargs)
        for key in a:
            if key != 'seconds':
                api.base.base.exact(a[key], b[key])
        pred = api.predict(a, d['causal'])
        assert pred.shape == (12, 7, 5)
        if arm == 'interaction64':
            np.testing.assert_array_equal(pred, api.base.base.predict(a, d['causal']))
        else:
            changed = dict(d['causal'], x=d['causal']['x'].copy())
            changed['x'][..., -64:] += 1000
            np.testing.assert_array_equal(pred, api.predict(a, changed))
        with pytest.raises(ValueError):
            api.predict(a, dict(d['causal'], future_valid=np.ones((12, 12), bool)))
        states.append(a)
    for key in ('initial_model', 'parameters', 'budget', 'sampler', 'draw_hash', 'fit_ids', 'preprocess', 'torch_rng'):
        api.base.base.exact(states[0][key], states[1][key])
