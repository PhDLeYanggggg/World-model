import numpy as np
import torch

from src.world_model import m3w_causal_velocity_spline as model
from scripts.audit_m3w_spline_ols_tail import matched


def predict(net, x, peers=True):
    return net(*(torch.from_numpy(a) for a in x), peers)


def test_initial_forecast_is_ols4_and_history_only():
    rng = np.random.default_rng(12)
    h = rng.normal(size=(7, 8, 2)).cumsum(1)
    x = model.causal_inputs(h, (np.arange(5), np.array([5]), np.array([6])))
    net = model.Spline().eval()
    p = predict(net, x).detach().numpy()
    np.testing.assert_allclose(p, x[-1][:, None]*np.arange(1., 13.)[None, :, None], rtol=1e-6)
    assert x[2].sum(1).tolist() == [4., 4., 4., 4., 4., 0., 0.]


def test_nonzero_model_is_rotation_and_translation_equivariant():
    rng = np.random.default_rng(10)
    h = rng.normal(size=(5, 8, 2)).cumsum(1)
    g = (np.arange(len(h)),)
    rotation = np.array([[0., -1.], [1., 0.]])
    net = model.Spline()
    with torch.no_grad():
        net.head[-1].bias[:] = torch.tensor([.1, -.2, .3, .1, -.2, -.1])
        a = predict(net, model.causal_inputs(h, g)).numpy()
        b = predict(net, model.causal_inputs(h@rotation+17., g)).numpy()
    np.testing.assert_allclose(a@rotation, b, atol=1e-6)


def test_missing_step_penalty_is_zero_for_ols_and_positive_for_divergence():
    v = torch.zeros(2, 2)
    y = torch.full((2, 12, 2), torch.nan)
    p = torch.ones_like(y)
    valid = torch.zeros(2, 12, dtype=torch.bool)
    s = torch.ones(2)
    assert model.loss(p, v, y, valid, s, False) == 0.
    assert model.loss(torch.zeros_like(p), v, y, valid, s, True) == 0.
    assert torch.isclose(model.loss(p, v, y, valid, s, True), torch.tensor(2.).sqrt())


def test_taper_matches_rowwise_displacement_and_returns_to_ols():
    rng = np.random.default_rng(41)
    ols, forecast = rng.normal(size=(2, 4, 12, 2))
    forecast[0] = ols[0]
    p = matched(ols, forecast)
    np.testing.assert_allclose(np.linalg.norm(p['taper'], axis=2).sum(1),
                               np.linalg.norm(p['uniform'], axis=2).sum(1))
    np.testing.assert_array_equal(p['taper'][:, -1], 0.)
    np.testing.assert_array_equal(p['uniform'][0], 0.)
