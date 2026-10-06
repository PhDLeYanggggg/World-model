import pytest
import torch

from src.world_model import m3w_easy_harm_deviance as api
from src.world_model import m3w_inner_separability as old


def decode(raw, env):
    f = torch.softmax(torch.cat((raw[:, :2], torch.zeros_like(raw[:, :1])), 1), 1)
    ref = torch.nn.functional.softplus(raw[:, 2])
    harm = env*f[:, 1]
    return torch.stack((env*f[:, 0], harm, ref, ref*raw[:, 3].sigmoid(), harm*raw[:, 4].sigmoid()), 1)


def test_log_matches_existing_decoder_without_probability_clipping():
    raw = torch.tensor([[1., -2., .3, .4, -3.], [0., 2., 1., 0., 4.]], dtype=torch.float64)
    env = torch.tensor([1., 10.], dtype=torch.float64)
    torch.testing.assert_close(api.log_easy_harm(raw, env).exp(), decode(raw, env)[:, 4])


def test_tiny_predicted_harm_has_finite_nonvanishing_deviance_gradient():
    raw = torch.tensor([[0., -30., 0., 0., -30.]], dtype=torch.float64, requires_grad=True)
    env = torch.ones(1, dtype=torch.float64); target = torch.tensor([.1], dtype=torch.float64)
    log_p = api.log_easy_harm(raw, env)
    mse = (log_p.exp()-target).square().sum()
    mg = torch.autograd.grad(mse, raw, retain_graph=True)[0]
    loss = api.easy_deviance(log_p, target, env, env[0]).sum()
    dg = torch.autograd.grad(loss, raw)[0]
    assert torch.isfinite(dg).all()
    assert abs(mg[0, 4]) < 1e-20
    assert dg[0, 4] == pytest.approx(-.2)


def test_zero_envelope_is_exact_zero_not_missing_label():
    raw = torch.zeros((2, 5), requires_grad=True); env = torch.tensor([0., 1.])
    target = torch.zeros(2)
    d = api.easy_deviance(api.log_easy_harm(raw, env), target, env, torch.tensor(1.))
    assert d[0] == 0 and d[1] > 0
    with pytest.raises(ValueError):
        api.easy_deviance(api.log_easy_harm(raw, env), torch.tensor([.1, 0.]), env, torch.tensor(1.))


def test_only_one_moment_changes_with_equal_query_weighting():
    raw = torch.zeros((3, 5), dtype=torch.float64, requires_grad=True)
    env = torch.ones(3, dtype=torch.float64); p = decode(raw, env)
    y = p.detach()*.5; rms = torch.ones(8, dtype=torch.float64); seg = torch.tensor([0, 0, 1])
    before = old.losses(p, y, seg, 2, rms)
    after = api.objective(p, y, api.log_easy_harm(raw, env), env, seg, 2, rms)
    assert after['decision_scores'] == before['decision_scores']
    dev = api.easy_deviance(api.log_easy_harm(raw, env), y[:, 4], env, rms[4])
    change = dev-(p[:, 4]-y[:, 4]).square()
    expected = (change[:2].mean()+change[2])/20
    torch.testing.assert_close(after['total']-before['total'], expected)
    after['total'].backward(); assert torch.isfinite(raw.grad).all()


def test_nonfinite_or_partial_target_rejected():
    raw = torch.zeros((1, 5)); env = torch.ones(1); p = decode(raw, env); y = p.clone(); y[0, 0] = float('nan')
    with pytest.raises(ValueError):
        api.objective(p, y, api.log_easy_harm(raw, env), env, torch.tensor([0]), 1, torch.ones(8))
