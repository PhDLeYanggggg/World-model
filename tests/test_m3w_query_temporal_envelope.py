import numpy as np
import torch

from scripts.audit_m3w_query_temporal_capacity import minimize_observed_error
from src.world_model.m3w_query_temporal_envelope import TemporalEnvelope, fixed_radius, objective


def test_peak_preserved_with_nonmonotone_motion_and_identity():
    d = torch.tensor([[[0., 0.], [3., 4.], [0., -1.]],
                      [[0., 0.], [0., 0.], [0., 0.]],
                      [[1., 0.], [-4., 0.], [3., 0.]]])
    weight = torch.tensor([.1, .2, 1.])
    a = fixed_radius(d, weight)
    torch.testing.assert_close(a.norm(dim=-1).amax(1), .25*d.norm(dim=-1).amax(1))
    assert torch.all(a*d >= 0.)
    assert torch.count_nonzero(a[1]) == 0


def test_flat_start_and_gradient_through_masked_dual_reference_objective():
    net = TemporalEnvelope(3)
    x = torch.ones(2, 3)
    d = torch.arange(48, dtype=torch.float32).reshape(2, 12, 2)/48
    edit, shape = net(x, d)
    torch.testing.assert_close(edit, .25*d)
    truth = d.clone()
    mask = torch.tensor([[True]*8 + [False]*4, [False]*12])
    easy = torch.tensor([True, False])
    loss = objective(edit, shape, truth, d*.4, mask, easy)
    truth[~mask] = 1e8
    torch.testing.assert_close(objective(edit, shape, truth, d*.4, mask, easy), loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in net.parameters())
    assert net.head[-1].weight.grad.abs().sum() > 0.


def test_empty_query_or_all_unknown_stays_finite():
    net = TemporalEnvelope(3)
    assert net(torch.zeros(0, 3), torch.zeros(0, 12, 2))[0].shape == (0, 12, 2)
    edit, shape = net(torch.zeros(1, 3), torch.zeros(1, 12, 2))
    loss = objective(edit, shape, edit, edit, torch.zeros(1, 12, dtype=torch.bool),
                     torch.zeros(1, dtype=torch.bool))
    assert loss.item() == 0.


def test_radius_constraint_can_use_unobserved_step_but_cannot_use_zero_ray():
    d = np.array([[[4., 0.], [0., 1.], [0., 0.]]]*2)
    y = np.array([[[.2, 0.], [0., .3], [0., 0.]]]*2)
    valid = np.array([[True, False, True], [True, True, False]])
    best = minimize_observed_error(d, y, valid)
    np.testing.assert_allclose(best[0], [[.2, 0.], [0., 1.], [0., 0.]])
    np.testing.assert_allclose(best[1], [[.2, 0.], [0., 1.], [0., 0.]])
