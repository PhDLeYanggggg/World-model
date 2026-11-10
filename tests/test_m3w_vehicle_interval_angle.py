import numpy as np
import torch

from src.evaluation.m3w_interaction_frontier import pair_reduction
from src.world_model import m3w_vehicle_pair_angle as prior
from src.world_model.m3w_vehicle_interval_angle import loss, relative_bounds


def test_interval_lower_reproduces_the_old_target_and_bounds_complete_labels():
    rng = np.random.default_rng(2917)
    a, b, c = rng.normal(size=(3, 7, 12, 2))
    valid = rng.random(c.shape[:2]) > .2
    incomplete = np.where(valid[..., None], c, np.nan)
    interval = relative_bounds(a[None], b, incomplete, valid)[0]
    np.testing.assert_allclose(interval[0], pair_reduction(a[None], b, incomplete, valid)[0, 1])
    actual = pair_reduction(a[None], b, c, np.ones_like(valid))[0, 1]
    assert interval[0] <= actual <= interval[1]


def test_interval_loss_removes_interior_gradient_and_preserves_point_loss():
    net = torch.nn.Linear(1, 3, bias=False)
    net.weight.data.fill_(0.)
    x = torch.ones((1, 1))
    y = torch.tensor([[.2, -.3, 1., .4]])
    loss(net, (x, y)).backward()
    assert net.weight.grad[1] == 0.
    assert net.weight.grad[0] < 0. and net.weight.grad[2] < 0.
    net.zero_grad()
    y[0, 1] = y[0, 3] = .4
    loss(net, (x, y)).backward()
    assert net.weight.grad[1] < 0.


def test_point_intervals_are_the_archived_direct_objective():
    rng = torch.Generator().manual_seed(17)
    x = torch.randn((11, 3), generator=rng)
    target = torch.rand((11, 2), generator=rng)*2.-1.
    dual = (target[:, :1] > 0.).float()
    net = torch.nn.Linear(3, 3)
    old = torch.cat((target, target, dual), dim=1)
    new = torch.cat((target, dual, target[:, 1:2]), dim=1)
    torch.testing.assert_close(loss(net, (x, new)), prior.loss(net, (x, old), direct=True))
