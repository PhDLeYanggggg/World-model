import copy
import numpy as np
import pytest
import torch
from src.world_model import m3w_selection_harm_deviance as api


def test_stable_logs_reproduce_current_decoder_and_recover_tiny_cost_gradient():
    raw = torch.zeros((2, 7, 3), requires_grad=True)
    with torch.no_grad(): raw[..., 1] = -80; raw[..., 2] = -80
    env = torch.ones((2, 7)); log_h, log_eh = api.log_harms(raw, env)
    fractions = torch.softmax(torch.cat((raw[..., :2], torch.zeros_like(raw[..., :1])), -1), -1)
    torch.testing.assert_close(log_h.exp(), env*fractions[..., 1])
    torch.testing.assert_close(log_eh.exp(), env*fractions[..., 1]*torch.sigmoid(raw[..., 2]))
    mse = (log_eh.exp()-.2).square().mean()
    old = torch.autograd.grad(mse, raw, retain_graph=True)[0]
    dev = api.easy_deviance(log_eh.flatten(), torch.full((14,), .2), env.flatten(), torch.tensor(1.)).mean()
    new = torch.autograd.grad(dev, raw)[0]
    assert torch.isfinite(new).all() and new[..., 2].abs().sum() > .1
    assert old[..., 2].abs().sum() < 1e-20


def fixture():
    n = 4; env = torch.ones((n, 7)); raw = torch.zeros((n, 7, 3), requires_grad=True)
    h, eh = api.log_harms(raw, env)
    p = torch.stack((torch.ones_like(h)*.2, h.exp(), torch.ones_like(h),
                     torch.ones_like(h)*.4, eh.exp()), -1)
    truth = p.detach().clone(); truth[..., 1] = .4; truth[..., 4] = .2
    data = dict(known=np.array([True, True, False, True]),
        causal=dict(allowed=np.ones((n, 7), bool)), truth=truth,
        preprocess=dict(rms=np.ones(5, np.float32)))
    return p, data, np.arange(n), torch.tensor([0, 0, 1, 1]), raw, env


def test_mean_control_is_exact_and_unknown_target_values_do_not_supervise():
    p, data, ix, seg, raw, env = fixture()
    a = api.objective(p, data, ix, seg, 2, 'mean_absolute')
    b = api.previous.objective(p, data, ix, seg, 2, 'mean_absolute')
    assert all(torch.equal(a[k], b[k]) if isinstance(a[k], torch.Tensor) else a[k] == b[k] for k in a)
    first = api.objective(p, data, ix, seg, 2, 'dual_deviance', raw, env)['total']
    other = copy.deepcopy({k:v for k,v in data.items() if k != 'truth'}); other['truth'] = data['truth'].clone()
    other['truth'][2] *= .5
    second = api.objective(p, other, ix, seg, 2, 'dual_deviance', raw, env)['total']
    assert torch.equal(first, second)


def test_zero_envelope_is_masked_without_numerical_gradient_artifacts():
    raw = torch.zeros((1, 7, 3), requires_grad=True); env = torch.zeros((1, 7))
    h, eh = api.log_harms(raw, env)
    loss = api.easy_deviance(eh.flatten(), torch.zeros(7), env.flatten(), torch.tensor(1.)).sum()
    assert loss == 0 and torch.equal(torch.autograd.grad(loss, raw)[0], torch.zeros_like(raw))
    with pytest.raises(ValueError): api.log_harms(raw, -torch.ones_like(env))
