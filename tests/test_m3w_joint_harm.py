import numpy as np
import pytest
import torch
from src.world_model import m3w_joint_harm_hurdle as api
from src.evaluation.m3w_joint_harm_scores import scores, fitting_prior


def packet():
    c = dict(x=np.zeros((4, 7, 428), np.float32), envelope=np.ones((4, 7)), scales=np.ones(4, np.float32),
        allowed=np.tile(np.array([False, True, True, True, True, True, True]), (4, 1)),
        groups=[np.arange(2), np.arange(2, 4)], source_groups=[np.array([0, 1])], sites=np.array(['s']*4), ids=np.arange(4))
    truth = torch.zeros(4, 7, 5)
    truth[:, :, 2] = 1; truth[:, :, 3] = .5
    truth[0, :, 0] = .2; truth[1, :, 1] = .3
    truth[2, :, 1] = .4; truth[2, :, 4] = .4
    return dict(causal=c, z=torch.zeros(4, 7, 428), observed_future=np.ones(4, bool),
        defined_by_arm=dict(bound_aware=np.ones(4, bool)), truth_by_arm=dict(bound_aware=truth),
        preprocess=dict(mean=np.zeros(428), std=np.ones(428), rms=np.ones(5, np.float32),
            bias=np.zeros(5, np.float32), clip=8.))


def test_exclusive_moments_and_zero_envelope():
    model = api.JointHarmHead(428, 4, 17, np.zeros(5))
    env = torch.rand(4, 7); env[:, 0] = 0
    parts = model.components(torch.randn(4, 7, 428), env, torch.tensor([0, 0, 1, 1]), 2)
    p = parts['moments']
    assert torch.all(p >= 0) and torch.all(p[..., :2].sum(-1) <= env+1e-6)
    assert torch.all(p[..., 4] <= p[..., 1]) and torch.all(p[..., 3] <= p[..., 2])
    assert torch.count_nonzero(p[:, 0, [0, 1, 4]]) == 0
    assert torch.allclose(parts['probability'].sum(-1), torch.ones(4, 7))


def test_event_labels_and_unknown_event_loss_mask():
    d = packet(); event, frac = api.event_targets(d['truth_by_arm']['bound_aware'], torch.ones(4, 7))
    assert torch.equal(event[:, 1], torch.tensor([1, 2, 3, 0]))
    assert torch.allclose(frac[:, 1], torch.tensor([.2, .3, .4, 0.]))
    d['observed_future'][:] = False
    model = api.JointHarmHead(428, 4, 17, np.zeros(5))
    seg = torch.tensor([0, 0, 1, 1]); parts = model.components(d['z'], torch.ones(4, 7), seg, 2)
    a = api.objective(parts, d, np.arange(4), seg, 2, 'product_moment')
    b = api.objective(parts, d, np.arange(4), seg, 2, 'event_supervised')
    assert b['occurrence'] == b['conditional_fraction'] == b['event_supervised_rows'] == 0
    assert torch.equal(a['total'], b['total'])
    b['total'].backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())


def test_event_auxiliary_changes_gradients_not_base_loss():
    d = packet(); model = api.JointHarmHead(428, 4, 17, np.zeros(5)); seg = torch.tensor([0, 0, 1, 1])
    parts = model.components(d['z'], torch.ones(4, 7), seg, 2)
    a = api.objective(parts, d, np.arange(4), seg, 2, 'product_moment')
    b = api.objective(parts, d, np.arange(4), seg, 2, 'event_supervised')
    assert torch.equal(a['regression'], b['regression'])
    assert torch.allclose(b['total']-a['total'], b['occurrence']+b['conditional_fraction'])


def test_real_torch_resume_and_matched_sampling(tmp_path):
    d = packet(); settings = dict(width=4, learning_rate=.001, query_batch_size=2, minimum_updates=3,
        largest_source_cycles=1, gradient_clip=5., heartbeat_every=1, checkpoint_every=1)
    states = []
    for arm in api.ARMS:
        path = tmp_path / (arm+'.pt.gz')
        api.fit(d, arm=arm, seed=17, settings=settings, identity={}, path=path, stop_at=1)
        a = api.fit(d, arm=arm, seed=17, settings=settings, identity={}, path=path, resume=True)
        b = api.fit(d, arm=arm, seed=17, settings=settings, identity={}, path=tmp_path/(arm+'_direct.pt.gz'))
        api.exact_state(a, b); states.append(a)
        p = api.predict(a, d['causal']); assert p['moments'].shape == (4, 7, 5)
        with pytest.raises(ValueError, match='Only declared causal'):
            api.predict(a, dict(d['causal'], future_endpoint=np.ones((4, 2))))
    for key in ('initial_model', 'sampler', 'draw_hash', 'preprocess'):
        api.base.exact(states[0][key], states[1][key])
    assert any(not torch.equal(states[0]['model'][k], states[1]['model'][k]) for k in states[0]['model'])


def test_fitting_prior_and_empty_diagnostics():
    d = packet(); prior = fitting_prior(d)
    assert np.allclose(prior[1:], .25)
    prediction = dict(moments=d['truth_by_arm']['bound_aware'].numpy(), probability=np.full((4, 7, 4), .25))
    out = scores(prediction, prediction['moments'], d['observed_future'], d['causal'], np.full(4, -1), prior)
    assert out['all_legal']['H_fraction_MSE'] == 0
    assert out['fixed_proposal']['status'] == 'not_estimable_no_known_legal_actions'
