import numpy as np
import pytest
import torch

from src.world_model import m3w_vehicle_pair_completion as m


def test_same_ranking_at_exact_targets_including_signed_ols_debit():
    old = np.array([[1., 1., .5, 1.25], [0., 0., -.5, -.25]])
    g = np.array([1., 0.])
    direct = m.targets(old, g, np.ones((2, 1)), 'direct_aux')
    factored = m.targets(old, g, np.ones((2, 1)), 'factorized')
    np.testing.assert_array_equal(factored[:, 3], [.5, .5])
    np.testing.assert_array_equal(factored[:, 4], [-.25, .25])
    np.testing.assert_allclose(m.merit(direct, 'direct_aux'), m.merit(factored, 'factorized'))


def test_invalid_cv_debit_label_is_rejected():
    with pytest.raises(ValueError, match='cannot increase'):
        m.targets(np.array([[1., 1., 2., .4]]), [.3], np.ones((1, 1)), 'factorized')


def test_same_capacity_network_and_loss_for_both_supervisions():
    rng = np.random.default_rng(12)
    a, b = [torch.from_numpy(rng.normal(size=(3, m.base.WIDTH)).astype(np.float32)) for _ in range(2)]
    rel = torch.zeros((3, 6))
    net = m.PairValue()
    torch.testing.assert_close(net(a, b, rel), net(b, a, rel))
    for arm in m.ARMS:
        y = torch.from_numpy(m.targets(np.zeros((3, 4)), np.zeros(3), np.ones((3, 1)), arm))
        value = m.loss(net, (a, b, rel, y))
        assert torch.isfinite(value)
        value.backward()
