import numpy as np
import torch

from scripts.run_m3w_expiry_pricing import withdrawal_geometry
from src.evaluation.m3w_opendd_location import paired
from src.world_model.m3w_expiry_pricing import Value, loss, score


def test_withdrawal_bound_holds_with_missing_future_points():
    rng = np.random.default_rng(3)
    path, cv, y = rng.normal(size=(3, 5, 12, 2))
    valid = rng.random((5, 12)) > .45
    y[~valid] = np.nan
    scale = np.arange(1., 6.)
    idx = np.arange(5)
    bound = withdrawal_geometry(cv, path, idx, idx, scale)
    exact = paired(path, cv, y, valid)[1].mean(1)/scale
    assert (np.abs(exact) <= bound+1e-12).all()


def test_asymmetric_price_never_credits_predicted_expiry_gain():
    z = np.zeros((2, 7))
    z[:, 4] = 1.
    z[:, 5] = [2., -1.]
    z[:, 6] = [-2., 2.]
    v = score(z, np.array([1., 3.]))
    np.testing.assert_allclose(v["mixture"], [1.5, 0.])
    np.testing.assert_allclose(v["priced"], [-.5, 0.])
    np.testing.assert_allclose(v["geometry"], [0., -1.])


def test_empty_outcome_branch_keeps_loss_finite():
    net = Value(4)
    y = torch.ones(3, 5)
    y[:, 3:] = .2
    v = loss(net, torch.zeros(3, 4), y)
    v.backward()
    assert torch.isfinite(v)
