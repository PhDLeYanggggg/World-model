import numpy as np
import torch

from src.world_model import m3w_av2_observed_harm as model


def test_observed_harm_has_no_missing_step_debit_and_needs_support():
    cv = np.zeros((3, 12, 2))
    y = cv.copy()
    y[0, :, 0], y[1, :, 0] = 1., 3.
    v = np.ones(cv.shape[:-1], dtype=bool)
    v[0, -2:] = False
    v[2] = False
    paths = np.repeat(cv[:, None], 2, axis=1)
    paths[0, 0, :, 0] = -.1
    paths[0, 1, -2:, 0] = -200.
    opts = np.array([[0, 1], [1, 0], [1, 2]])
    support, fail, severity = model.targets(cv, paths, opts, y, v, 2.)
    np.testing.assert_array_equal(support, [True, True, False])
    np.testing.assert_array_equal(fail, [True, False, False])
    np.testing.assert_allclose(severity[0], np.log1p(np.array([.025, .1])/.02), rtol=1e-6)


def test_harm_classifier_and_magnitude_have_finite_gradients():
    z = torch.zeros(2, 3, requires_grad=True)
    model.loss(z, torch.tensor([True, False]), torch.tensor([[2., 1.], [0., .5]]))[0].backward()
    assert torch.isfinite(z.grad).all()
    assert z.grad[0, 0] < 0 < z.grad[1, 0]
