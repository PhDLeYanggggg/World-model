import numpy as np
import torch

from scripts.run_m3w_vehicle_pair_learning import choices
from src.world_model.m3w_vehicle_pair_learning import PairValue, labels, merit


def test_distinct_actor_sampling_is_reproducible_and_bounded():
    peak = np.ones((50, 3))
    a = choices(peak, 2, 49, np.random.default_rng(2))
    b = choices(peak, 2, 49, np.random.default_rng(2))
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x, y)
    rows, _, i, j = a
    assert len(i) == 256
    assert (rows[i] != rows[j]).all()


def test_joint_and_additive_predictions_are_order_invariant():
    x, y, r = torch.randn(12, 45), torch.randn(12, 45), torch.randn(12, 6)
    for joint in (False, True):
        net = PairValue(joint)
        torch.testing.assert_close(net(x, y, r), net(y, x, r))
    additive = PairValue(False)
    torch.testing.assert_close(additive(x, y, r), additive(x, y, r+8))


def test_dual_label_needs_support_risk_and_full_record_ols_gain():
    i, j = np.array([0, 0, 1]), np.array([1, 2, 2])
    risk = np.zeros((3, 3))
    support = np.array([1., 0., 0.])
    gain = np.array([[2., 2.], [1., 2.], [1., 0.]])
    y = labels(i, j, risk, support, gain, 0., -3., 4.)
    np.testing.assert_array_equal(y[:, :2], [[1, 1], [1, 0], [0, 0]])
    np.testing.assert_array_equal(y[:, 2:], [[.75, 1.], [.75, .5], [.5, .5]])
    assert merit(np.array([[1., 1., 1., 1.], [-1., 1., 1., 1.]]))[0] > merit(
        np.array([[-1., 1., 1., 1.]]))[0]
