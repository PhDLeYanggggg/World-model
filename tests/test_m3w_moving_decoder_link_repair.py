import copy
import numpy as np
import pytest
import torch

from src.world_model.m3w_moving_decoder_link_repair import auxiliary, objective, fit, base


def test_extreme_wrong_logits_receive_finite_corrective_gradient():
    raw = torch.tensor([[[0., -100., -100.]]], requires_grad=True)
    truth = torch.tensor([[[0., .5, 1., 1., .5]]])
    loss = auxiliary(raw, truth, torch.ones(1, 1), torch.ones(1, 1, dtype=torch.bool), torch.zeros(1, dtype=torch.long), 1)
    loss.backward()
    assert torch.isfinite(loss)
    assert raw.grad[0, 0, 1] < -.49
    assert raw.grad[0, 0, 2] < -.49


def test_zero_harm_and_unknown_bound_are_coherent_soft_cost_targets():
    raw = torch.zeros((2, 7, 3), requires_grad=True)
    truth = torch.zeros((2, 7, 5))
    truth[1, :, 1] = 1
    truth[1, :, 4] = 1
    legal = torch.zeros((2, 7), dtype=torch.bool)
    legal[:, 6] = True
    result = auxiliary(raw, truth, torch.ones(2, 7), legal, torch.zeros(2, dtype=torch.long), 1)
    result.backward()
    assert torch.isfinite(raw.grad).all()
    assert not torch.count_nonzero(raw.grad[:, :6])
    assert raw.grad[1, 6, 2] < 0


def test_bad_cost_geometry_is_rejected_not_renormalized_silently():
    raw = torch.zeros((1, 7, 3))
    truth = torch.ones((1, 7, 5))
    with pytest.raises(ValueError, match='causal envelope'):
        auxiliary(raw, truth, torch.ones(1, 7), torch.ones(1, 7, dtype=torch.bool), torch.zeros(1, dtype=torch.long), 1)


def fixture():
    c = dict(x=np.zeros((4, 7, 428), np.float32), envelope=np.ones((4, 7)),
        scales=np.ones(4, np.float32), ids=np.arange(4), groups=[np.arange(2), np.arange(2, 4)],
        source_groups=[np.arange(2)], sites=np.array(['synthetic']*4), allowed=np.zeros((4, 7), bool))
    c['allowed'][:, 6] = True
    pr = dict(mean=np.zeros(428), std=np.ones(428), rms=np.ones(5, np.float32), clip=8., bias=np.zeros(5, np.float32))
    truth = torch.zeros((4, 7, 5))
    truth[..., 1:] = torch.tensor([.3, 1., .5, .2])
    data = dict(causal=c, preprocess=pr, z=torch.from_numpy(c['x']),
        truth_by_arm=dict(bound_aware=truth), defined_by_arm=dict(bound_aware=np.ones(4, bool)),
        observed_future=np.ones(4, bool))
    model = base.base.CostHead(428, 8, 17, pr['bias'], 'context')
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    sampler = base.base.CyclicQueries(c['groups'], c['source_groups'], 2, 17)
    sampler.draw()
    # Synthetic warm identity only; real2k parents are verified by artifact hashes.
    parent = dict(step=2000, identity=dict(feature_arm='local_history'), seed=17, preprocess=pr,
        fit_ids=c['ids'], model=copy.deepcopy(model.state_dict()), optimizer=opt.state_dict(),
        sampler=sampler.state(), torch_rng=torch.get_rng_state(), draw_hash='0'*64)
    settings = dict(width=8, learning_rate=.001, query_batch_size=2, gradient_clip=5.,
        checkpoint_every=1, heartbeat_every=1)
    return data, parent, settings


def test_disabled_auxiliary_preserves_original_loss_and_gradients():
    data, parent, settings = fixture()
    model = base.base.CostHead(428, 8, 17, data['preprocess']['bias'], 'context')
    model.load_state_dict(parent['model'])
    ix, segments = np.arange(4), torch.tensor([0, 0, 1, 1])
    captured = []
    hook = model.action.register_forward_hook(lambda _, __, raw:captured.append(raw))
    p = model(data['z'], torch.ones(4, 7), segments, 2)
    original = base.objective(p, data, ix, segments, 2, 'bound_aware')['total']
    neutral = objective(p, captured[0], data, ix, segments, 2, 0.)['total']
    torch.testing.assert_close(original, neutral, rtol=0, atol=0)
    first = torch.autograd.grad(original, tuple(model.parameters()), retain_graph=True)
    second = torch.autograd.grad(neutral, tuple(model.parameters()))
    for a,b in zip(first, second):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    hook.remove()


@pytest.mark.parametrize('arm', ['continue_original', 'decoder_auxiliary'])
def test_warm_optimizer_is_not_mutated_and_resume_is_exact(tmp_path, arm):
    data, parent, settings = fixture()
    initial = copy.deepcopy(parent)
    common = dict(arm=arm, settings=settings, identity=dict(synthetic=True))
    direct = fit(data, parent, path=tmp_path/'direct.pt.gz', stop_at=4, **common)
    fit(data, parent, path=tmp_path/'resume.pt.gz', stop_at=2, **common)
    resumed = fit(data, parent, path=tmp_path/'resume.pt.gz', resume=True, stop_at=4, **common)
    base.base.exact(parent, initial)
    base.base.exact({k:v for k,v in direct.items() if k != 'seconds'},
                    {k:v for k,v in resumed.items() if k != 'seconds'})
