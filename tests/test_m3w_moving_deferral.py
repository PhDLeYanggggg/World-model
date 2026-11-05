import numpy as np
import pytest
import torch

from src.world_model import m3w_moving_deferral as api


def packet():
    c = dict(x=np.zeros((4, 7, 428), np.float32), envelope=np.ones((4, 7)), scales=np.ones(4, np.float32),
        allowed=np.tile(np.array([False, True, True, True, True, True, True]), (4, 1)),
        groups=[np.arange(2), np.arange(2, 4)], source_groups=[np.array([0, 1])], sites=np.array(['s']*4), ids=np.arange(4))
    truth = torch.zeros(4, 7, 5)
    truth[..., 2] = 1
    truth[:, 6, 0] = .8
    return dict(causal=c, z=torch.zeros(4, 7, 428), observed_future=np.ones(4, bool),
        truth_by_arm=dict(bound_aware=truth), preprocess=dict(mean=np.zeros(428), std=np.ones(428), clip=8.))


def test_exact_shape_count_and_floor_tie():
    d = packet(); model = api.DeferralRouter(428, 64, 17)
    assert sum(p.numel() for p in model.parameters()) == 27969
    logits = model(d['z'], torch.tensor([0, 0, 1, 1]), 2)
    choice, margin = api.choose(logits.detach().numpy(), d['causal']['allowed'])
    assert np.all(choice == -1) and np.all(margin == 0)
    assert not d['causal']['allowed'][:, 0].any()


def test_unknown_labels_never_contribute_and_queries_do_not_mix():
    d = packet(); d['observed_future'][:2] = False
    logits = torch.zeros(4, 7, requires_grad=True)
    segments = torch.tensor([0, 0, 1, 1])
    a = api.objective(logits, d, np.arange(4), segments, 2)
    d['truth_by_arm']['bound_aware'][:2] = float('nan')
    b = api.objective(logits, d, np.arange(4), segments, 2)
    assert torch.equal(a['total'], b['total']) and b['supervised_queries'] == 1
    b['total'].backward()
    assert torch.count_nonzero(logits.grad[:2]) == 0
    assert logits.grad[2, 6] < logits.grad[2, 0]
    d['observed_future'][:] = False
    empty = api.objective(logits, d, np.arange(4), segments, 2)
    assert empty['total'] == 0 and empty['unknown_outcomes_used'] == 0


def test_illegal_action_and_context_isolation():
    d = packet(); logits = np.arange(28).reshape(4, 7).astype(float)
    allowed = d['causal']['allowed']; allowed[:, 6] = False
    choice, _ = api.choose(logits, allowed)
    assert np.all(choice == 5)
    model = api.DeferralRouter(428, 4, 17)
    torch.nn.init.normal_(model.action.weight)
    a = model(d['z'], torch.tensor([0, 0, 1, 1]), 2)
    z = d['z'].clone(); z[2:] = 1
    b = model(z, torch.tensor([0, 0, 1, 1]), 2)
    assert torch.equal(a[:2], b[:2])


def test_real_resume_prediction_and_input_boundary(tmp_path):
    d = packet(); settings = dict(width=4, learning_rate=.001, query_batch_size=2, minimum_updates=3,
        largest_source_cycles=1, gradient_clip=5., heartbeat_every=1, checkpoint_every=1)
    path = tmp_path / 'resume.pt.gz'
    api.fit(d, seed=17, settings=settings, identity={}, path=path, stop_at=1)
    a = api.fit(d, seed=17, settings=settings, identity={}, path=path, resume=True)
    b = api.fit(d, seed=17, settings=settings, identity={}, path=tmp_path/'direct.pt.gz')
    api.exact_state(a, b)
    p = api.predict(a, d['causal'])
    assert p.shape == (4, 7)
    np.testing.assert_array_equal(p, api.predict(b, d['causal']))
    c = dict(d['causal'], x=d['causal']['x'].copy()); c['x'][..., 380:] = 999
    np.testing.assert_array_equal(p, api.predict(a, c))
    with pytest.raises(ValueError, match='Only declared causal'):
        api.predict(a, dict(d['causal'], future_endpoint=np.ones((4, 2))))
