import copy
import numpy as np
import pytest
import torch

from src.world_model import m3w_multiaction_cost as api


def sample():
    rng = np.random.default_rng(7); n = 32
    x = rng.normal(size=(n, 7, 4)).astype(np.float32)
    envelope = np.full((n, 7), 4.)
    reference = rng.uniform(.2, 2, n)
    easy = rng.random(n) < .4
    difference = rng.uniform(-.2, .2, (n, 7))
    y = np.stack((np.maximum(difference, 0), np.maximum(-difference, 0),
                  np.broadcast_to(reference[:, None], (n, 7)),
                  np.broadcast_to((reference*easy)[:, None], (n, 7)),
                  np.maximum(-difference, 0)*easy[:, None]), -1)
    y[-1] = np.nan
    sites = np.array(['a']*16+['b']*16); scales = np.ones(n)
    pr = api.preprocessing(x, envelope, y, scales, sites)
    known = np.isfinite(y).all((1, 2))
    data = dict(preprocess=pr, z=torch.from_numpy(api.standardized(x, pr)),
        envelope=torch.tensor(envelope, dtype=torch.float32),
        truth=torch.tensor(np.where(known[:, None, None], y, 0), dtype=torch.float32),
        allowed=torch.ones((n, 7), dtype=torch.bool),
        groups=[np.flatnonzero(known & (sites == s)) for s in ('a', 'b')], input_hashes={'fixture': 'fixed'})
    return x, envelope, y, scales, sites, data


def test_normalization_ignores_neural_and_does_not_modify_unknowns():
    x, e, y, s, names, data = sample()
    old = y.copy(); x[:, 6] += 1e6; y[:-1, 6] *= 50
    changed = api.preprocessing(x, e, y, s, names)
    api.exact(data['preprocess'], changed)
    np.testing.assert_array_equal(np.isnan(old[-1]), np.isnan(y[-1]))


def test_action_costs_obey_causal_bounds():
    *_, data = sample()
    model = api.ActionCostHead(4, 8, 17, data['preprocess']['bias'])
    e = data['envelope'].clone(); e[:, 0] = 0
    p = model(data['z'], e, torch.arange(7))
    assert (p >= 0).all() and (p[..., :2].sum(-1) <= e+1e-6).all()
    assert (p[..., 0] <= p[..., 2]).all()
    assert (p[..., 3] <= p[..., 2]).all() and (p[..., 4] <= p[..., 1]).all()
    assert torch.count_nonzero(p[:, 0, :2]) == 0


def test_causal_selection_enforces_arm_and_risk_not_oracle_labels():
    p = np.tile([1., 0., 2., 1., 0.], (3, 7, 1)); p[:, 6, 0] = 2
    allowed = np.ones((3, 7), bool); allowed[1] = False
    p[2, :, 4] = .1
    full, _ = api.select_actions(p, allowed, arm='physical_neural')
    physical, _ = api.select_actions(p, allowed, arm='physical_only')
    assert full.tolist() == [6, -1, -1] and physical.tolist() == [0, -1, -1]
    with pytest.raises(TypeError):
        api.select_actions(p, allowed, arm='physical_neural', future_endpoint=np.zeros((3, 2)))


def test_three_arms_have_identical_initialization_draws_and_exact_resume(tmp_path):
    torch.set_num_threads(1)
    *_, data = sample()
    settings = dict(width=8, steps=6, batch_size=8, learning_rate=.001,
        gradient_clip=5., heartbeat_every=2, checkpoint_every=2, underharm_weight=4., ranking_weight=.5)
    states = []
    for arm in api.ARMS:
        kw = dict(data=data, arm=arm, settings=settings, seed=17, identity={'same': True})
        direct = api.fit(**kw, path=tmp_path/(arm+'.pt.gz'))
        api.fit(**kw, path=tmp_path/(arm+'_resume.pt.gz'), stop_at=3)
        resumed = api.fit(**kw, path=tmp_path/(arm+'_resume.pt.gz'), resume=True)
        for key in direct:
            if key != 'seconds':
                api.exact(direct[key], resumed[key])
        assert direct['unknown_rows_sampled'] == 0
        assert any(not torch.equal(v, direct['initial_model'][k]) for k, v in direct['model'].items())
        states.append(direct)
    for state in states[1:]:
        for key in ('initial_model', 'draw_hash', 'row_draws', 'sampler_rng', 'torch_rng', 'parameters'):
            api.exact(states[0][key], state[key])


def test_masked_actions_cannot_affect_cost_training_loss():
    prediction = torch.full((4, 6, 5), .5, requires_grad=True)
    target = torch.ones_like(prediction); real = torch.ones((4, 6), dtype=torch.bool)
    real[:, 1] = False
    out = api.continuous_loss(prediction, target, real, torch.ones(5), underharm_weight=4, ranking_weight=.5)
    out['total'].backward()
    assert torch.count_nonzero(prediction.grad[:, 1]) == 0
