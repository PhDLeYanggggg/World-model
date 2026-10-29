import copy
import numpy as np
import pytest
import torch
from src.world_model import m3w_multiaction_cost as base
from src.world_model import m3w_shared_reference_cost as api
from tests.test_m3w_multiaction_cost import sample


def paired_data():
    x, envelope, y, scales, sites, data = sample()
    floor = x[:, 0].copy()
    data['z'] = api.PairedCausalFeatures(data['z'], torch.from_numpy(base.standardized(floor, data['preprocess'])))
    return x, floor, envelope, data


def test_same_parameters_and_initial_weights_but_reference_is_invariant():
    torch.set_num_threads(1); x, floor, envelope, data = paired_data()
    new = api.SharedReferenceCostHead(8, 8, 17, data['preprocess']['bias'])
    old = api.OriginalCostHead(4, 8, 17, data['preprocess']['bias'])
    base.exact(new.state_dict(), old.state_dict())
    with torch.no_grad():
        new.network[-1].weight.fill_(.1)
    features = data['z'][np.arange(len(x))]
    p = new(features, data['envelope'], torch.arange(7))
    changed = features.clone(); changed[:, 6, :4] += 10
    q = new(changed, data['envelope'], torch.arange(7))
    for k in (2, 3):
        torch.testing.assert_close(p[..., k], p[:, :1, k].expand_as(p[..., k]), rtol=0, atol=0)
        torch.testing.assert_close(p[..., k], q[..., k], rtol=0, atol=0)
    assert (p[..., :2].sum(-1) <= data['envelope']+1e-6).all()
    assert (p[..., 0] <= p[..., 2]).all()
    assert (p[..., 3] <= p[..., 2]).all() and (p[..., 4] <= p[..., 1]).all()


def test_exact_resume_all_banks_and_no_inactive_neural_input_effect(tmp_path):
    torch.set_num_threads(1); x, floor, envelope, data = paired_data()
    settings = dict(width=8, steps=4, batch_size=8, learning_rate=.001,
        gradient_clip=5., heartbeat_every=2, checkpoint_every=2, underharm_weight=4., ranking_weight=.5)
    states = []
    with api.installed():
        for arm in base.ARMS:
            kw = dict(data=data, arm=arm, settings=settings, seed=17, identity={'fixture':True})
            direct = base.fit(**kw, path=tmp_path/(arm+'.gz'))
            base.fit(**kw, path=tmp_path/(arm+'_resume.gz'), stop_at=2)
            resumed = base.fit(**kw, path=tmp_path/(arm+'_resume.gz'), resume=True)
            for key in direct:
                if key != 'seconds':
                    base.exact(direct[key], resumed[key])
            states.append(direct)
        changed = copy.deepcopy(data); changed['z'].candidate[:, 6] += 1e5
        changed['truth'][:, 6] += 1e5
        other = base.fit(changed, arm='physical_only', settings=settings, seed=17,
            identity={'fixture':True}, path=tmp_path/'inactive.gz')
    base.exact(states[0]['model'], other['model'])
    for state in states[1:]:
        for key in ('initial_model', 'draw_hash', 'torch_rng', 'sampler_rng', 'parameters'):
            base.exact(states[0][key], state[key])
    p = api.predict(states[1], x, floor, envelope, np.ones(len(x)))
    assert np.isfinite(p).all()
    with pytest.raises(TypeError):
        api.predict(states[1], x, floor, envelope, np.ones(len(x)), future_endpoint=np.zeros((len(x), 2)))


def test_disabled_override_preserves_original_fit_exactly(tmp_path):
    torch.set_num_threads(1); *_, data = sample()
    settings = dict(width=8, steps=2, batch_size=8, learning_rate=.001,
        gradient_clip=5., heartbeat_every=1, checkpoint_every=1, underharm_weight=4., ranking_weight=.5)
    kw = dict(data=data, arm='physical_neural', settings=settings, seed=17, identity={'neutral':True})
    original = base.fit(**kw, path=tmp_path/'original.gz')
    with api.installed(False):
        neutral = base.fit(**kw, path=tmp_path/'neutral.gz')
    for key in original:
        if key != 'seconds':
            base.exact(original[key], neutral[key])
