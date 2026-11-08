import numpy as np
import torch

from src.world_model import m3w_direct_count_value as direct
from src.world_model import m3w_factorized_count_risk as m


def test_factorized_risk_reconstructs_signed_targets_with_missing_labels():
    cv = np.zeros((3, 2, 2))
    ols = np.ones_like(cv)
    bank = ols[:, None]
    y = np.full_like(cv, .1)
    valid = np.array([[True, True], [False, False], [True, False]])
    row, arm = np.array([[0, 1], [2, -1]]), np.array([[0, 0], [0, -1]])
    args = cv, ols, bank, y, valid, 1., row, arm, np.ones(2)
    truth = direct.targets(*args)
    z = m.targets(*args)
    np.testing.assert_allclose(z[..., :2], truth[..., :2], atol=1e-6)
    np.testing.assert_allclose(z[..., 2]+z[..., 4]-.02*z[..., 5], truth[..., 2], atol=1e-6)
    np.testing.assert_allclose(z[..., 3]+z[..., 4]-.02*z[..., 6], truth[..., 3], atol=1e-6)
    np.testing.assert_allclose(z[..., 3]-z[..., 7], truth[..., 5], atol=1e-6)


def test_factorized_fit_is_finite_even_without_positive_easy_events():
    net = m.Factorized(4)
    p = net(torch.zeros(3, 4))
    t = torch.zeros((3, 3, 8))
    legal = torch.tensor([[True, True, False]]*3)
    l = m.objective(p, t, legal)
    l.backward()
    assert torch.isfinite(l)
    assert not m.moments_for_score(p.detach())[:, 0].any()
