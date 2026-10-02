import numpy as np
from scipy.optimize._numdiff import approx_derivative
from src.world_model import m3w_cost_harm_newton as api
from test_m3w_positive_harm import sample


def test_exact_residual_curvature_matches_independent_finite_differences():
    _, q, y, w = sample(); w = w/w.sum(); z = q-w@q
    beta = np.arange(7)[None, :]*.025; inv = np.zeros(len(w), int)
    mu = np.array([w@y[:, 0]]); target = y[:, 0]-.4*q[:, 3]
    def evaluate(b): return api.quantities(b.reshape(1, 7), z, inv, w, np.array([1.]), mu, target, .7, 1.)
    loss, gradient, hessian, _ = evaluate(beta)
    numeric_g = approx_derivative(lambda b: evaluate(b)[0], beta.ravel()).ravel()
    numeric_h = approx_derivative(lambda b: evaluate(b)[1].ravel(), beta.ravel())
    np.testing.assert_allclose(gradient[0], numeric_g, rtol=2e-6, atol=2e-8)
    np.testing.assert_allclose(hessian[0], numeric_h, rtol=2e-6, atol=2e-8)


def test_newton_fit_reduces_same_loss_preserves_means_and_zero_harm():
    leaves, q, y, w = sample()
    a = api.cost_leaf(leaves, q, y, w, np.ones(2))
    b = api.v1.cost_leaf(leaves, q, y, w, np.ones(2))
    np.testing.assert_allclose(a['coef'], b['coef'], atol=1e-6)
    assert a['maximum_gradient'] <= 1e-7 and a['train_relative_mean_error'] < 1e-9
    zero = api.cost_leaf(leaves, q, np.zeros_like(y), w, np.ones(2))
    np.testing.assert_array_equal(zero['coef'], 0)
    np.testing.assert_array_equal(zero['log_normalizer'], 0)
