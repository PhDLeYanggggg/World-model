import numpy as np
import torch

from src.world_model.m3w_av2_steered_reference import MAX_ANGLE, Steer, label, objective, rotate


def test_steering_starts_at_incumbent_and_preserves_each_radius():
    net = Steer(4)
    x = torch.ones(2, 2, 4)
    delta = torch.randn(2, 2, 12, 2)
    torch.testing.assert_close(rotate(delta, net(x), torch), delta)
    with torch.no_grad():
        net.layers[-1].bias[:] = torch.tensor([-9., 2., 7.])
    a = net(x)
    assert float(a.abs().max().detach()) <= MAX_ANGLE + 1e-7
    torch.testing.assert_close(rotate(delta, a, torch).norm(dim=-1), delta.norm(dim=-1))


def test_guarded_objective_debits_missing_forecasts_and_backpropagates():
    c = torch.zeros(1, 2, 12, 2)
    d = torch.ones_like(c)
    a = torch.zeros(1, 2, 12, requires_grad=True)
    valid = torch.ones_like(a, dtype=torch.bool)
    valid[:, 1] = False
    args = (c, c, d, torch.ones(1), c+.1, valid, torch.tensor([[True, False]]), torch.tensor([[0., .1]]))
    ade = objective(a, *args, guarded=False)
    strict = objective(a, *args, guarded=True)
    assert strict > ade
    strict.backward()
    assert torch.isfinite(a.grad).all()


def test_unedited_agents_contribute_to_ols_floor_not_candidate_inputs():
    d = dict(cv=np.zeros((3, 12, 2)), ols4=np.ones((3, 12, 2)))
    y = np.zeros_like(d["cv"])
    v = np.ones((3, 12), dtype=bool)
    v[2] = False
    _, _, _, context = label(d, [0, 1], y, v, .2)
    np.testing.assert_allclose(context, [-12*np.sqrt(2.), 0.])
