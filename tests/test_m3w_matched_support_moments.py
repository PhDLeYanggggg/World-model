import numpy as np
import torch

from src.world_model.m3w_matched_support_moments import SupportMoments, allocate, objective
from scripts.audit_m3w_matched_support_recipients import optimize_paths


def test_moments_are_nested_and_bounded_for_both_losses():
    net = SupportMoments(4)
    x, radius = torch.ones((4, 3, 4)), torch.ones((4, 3))
    radius[:, -1] = 0.
    pred = net(x, radius)
    ref, costs = pred[:2]
    assert torch.all(ref[:, 0] >= ref[:, 1])
    assert torch.all(costs[..., 1] >= costs[..., 2])
    assert torch.all(costs[..., 1] <= radius)
    assert torch.all(costs[:, -1] == 0.)
    known = torch.tensor([True, True, False, False])
    easy = torch.tensor([True, False, False, False])
    b = torch.tensor([.1, 1., 0., 0.])
    gain = torch.tensor([[.1, -.2, 0.], [-.5, .2, 0.], [0., 0., 0.], [0., 0., 0.]])
    for arm in (False, True):
        loss = objective(net(x, radius), b, gain, known, easy, radius, arm)
        loss.backward()
        assert torch.isfinite(loss)
        assert all(p.grad is not None and p.grad.isfinite().all() for p in net.parameters())


def test_risk_rank_preserves_recording_total_and_query_control():
    ref = np.ones((4, 2))
    moment = np.zeros((4, 1, 3))
    moment[:, 0, 0] = [3., 2., 1., .5]
    moment[:2, 0, 1:] = .7
    inv, legal = np.array([1, 0, 1, 0]), np.ones((4, 1), bool)
    w = np.ones(4)/2
    (gain, _), (safe, ratios) = allocate(ref, moment, legal, inv, 2, w)
    np.testing.assert_array_equal(np.flatnonzero(gain >= 0), [0, 1])
    np.testing.assert_array_equal(np.flatnonzero(safe >= 0), [2, 3])
    assert ratios == [0., 0.]
    _, (q, _) = allocate(ref, moment, legal, inv, 2, w, query_counts=np.array([2, 0]))
    np.testing.assert_array_equal(np.flatnonzero(q >= 0), [1, 3])
    _, (_, undef) = allocate(ref, moment, legal, inv, 0, w)
    assert undef == [None, None]


def test_hindsight_path_swap_excludes_identity_and_uses_missing_debit():
    cv = np.zeros((2, 12, 2))
    bank = np.stack((cv, cv+1., cv+3.), axis=1)
    y = cv.copy()
    y[0] = 3.
    valid = np.ones((2, 12), bool)
    valid[1] = False
    np.testing.assert_array_equal(optimize_paths(cv, bank, y, valid, np.arange(2)), [2, 1])
