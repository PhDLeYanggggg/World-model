import numpy as np
import torch

from scripts.run_m3w_vehicle_anchor_rank import base, trajectory_lower
from scripts.audit_m3w_vehicle_anchor_rank_paired import intervals
from src.world_model import m3w_vehicle_anchor_rank as rank


def test_keeps_a_recipient_but_not_the_exact_anchor():
    rows = np.array([0, 0, 1, 2, 3])
    first = np.array([0, 1, 0, 2, 3])
    second = np.array([2, 2, 4, 3, 4])
    np.testing.assert_array_equal(rank.alternatives(rows, first, second, np.array([0, 2])),
                                  [False, True, True, True, False])


def test_conservative_trajectory_lower_matches_direct_predictions_with_missing_labels():
    rng = np.random.default_rng(7)
    cv = rng.normal(size=(4, 12, 2))
    rows = np.array([0, 0, 1, 2, 3])
    p = cv[rows]+rng.normal(size=(5, 12, 2))
    y = rng.normal(size=cv.shape)
    valid = rng.random(cv.shape[:2]) > .3
    y[~valid] = np.nan
    first, second, anchor = np.array([0, 1, 2]), np.array([2, 4, 3]), np.array([0, 2])
    ref, result = trajectory_lower(cv, p, rows, anchor, first, second, y, valid)
    for k, ij in enumerate(zip(first, second, strict=True)):
        pred = cv.copy()
        pred[rows[list(ij)]] = p[list(ij)]
        np.testing.assert_allclose(result[k], base.parent.paired(ref, pred, y, valid)[1].sum(), atol=1e-10)
    assert result[0] == 0.


def test_head_is_actor_order_invariant_and_only_relative_loss_uses_extra_targets():
    torch.manual_seed(3)
    net = rank.AnchorValue()
    a, b, u, v = [torch.randn(5, rank.WIDTH) for _ in range(4)]
    r, t, c = torch.randn(5, 6), torch.randn(5, 6), torch.ones(5, 1)
    inputs = (a, b, r, u, v, t, c)
    torch.testing.assert_close(net(*inputs), net(b, a, r, v, u, t, c))
    target = torch.rand(5, 6)
    changed = target.clone()
    changed[:, 4:] += 5.
    torch.testing.assert_close(rank.loss(net, (*inputs, target), False),
                               rank.loss(net, (*inputs, changed), False))
    assert rank.loss(net, (*inputs, changed), True) > rank.loss(net, (*inputs, target), True)


def test_bootstrap_weights_locations_not_repeated_directions():
    result = intervals([('a', 1.), ('a', 3.), ('b', 8.)], ['a', 'b'], np.array([[0, 0], [1, 1]]))
    assert result['site_means'] == [2., 8.]
    assert result['mean'] == 5.
    np.testing.assert_allclose(result['ci95'], [2.15, 7.85])
