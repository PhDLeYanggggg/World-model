"""Independent boundary checks; do not modify the registered training solver."""
import numpy as np
import pytest
from scipy.optimize import minimize
from scipy.optimize._numdiff import approx_derivative

from src.world_model import m3w_cost_harm_newton as api


def fixture():
    rng = np.random.default_rng(20261002)
    leaves = np.repeat([3, 19, 80], [27, 35, 22])
    q = rng.normal(size=(len(leaves), 7))
    w = rng.uniform(.05, 2., len(leaves))
    y = np.column_stack((np.exp(.2*q[:, 0]), np.maximum(q[:, 1], 0)))
    target = y + .15*q[:, [3, 4]]
    return leaves, q, y, w, target


def test_weighted_multi_leaf_hessian_matches_separate_scalar_objectives():
    leaves, q, y, w, target = fixture()
    _, inv = np.unique(leaves, return_inverse=True)
    mass = np.bincount(inv, weights=w)
    wn = w/mass[inv]
    wm = np.bincount(inv, weights=wn)
    mean = np.stack([wn[inv == k]@q[inv == k] for k in range(3)])
    z = q-mean[inv]
    mu = np.array([wn[inv == k]@y[inv == k, 0] for k in range(3)])
    beta = np.arange(21).reshape(3, 7)*.007
    values, gradients, hessians, _ = api.quantities(
        beta, z, inv, wn, wm, mu, target[:, 0], .8, 1.)
    for k in range(3):
        use = inv == k
        weight, features, truth = wn[use], z[use], target[use, 0]

        def independent(b, derivative=False):
            exp = np.exp(features@b)
            pred = mu[k]*exp/(weight@exp)
            residual = pred-truth
            if not derivative:
                return .5*(weight@(residual/.8)**2) + .5*b@b
            center = ((weight*exp)@features)/(weight@exp)
            jacobian = pred[:, None]*(features-center)
            return (weight*residual/.8**2)@jacobian+b

        np.testing.assert_allclose(values[k], independent(beta[k]), atol=1e-12)
        numeric_g = approx_derivative(independent, beta[k]).ravel()
        numeric_h = approx_derivative(lambda b: independent(b, True), beta[k])
        np.testing.assert_allclose(gradients[k], numeric_g, rtol=2e-6, atol=2e-8)
        np.testing.assert_allclose(hessians[k], numeric_h, rtol=2e-6, atol=2e-8)


def test_batched_fits_match_independent_scipy_leaf_optimizers():
    leaves, q, y, w, target = fixture()
    scales = np.array([.8, 1.3])
    fit = api.cost_leaf(leaves, q, y, w, scales, squared_target=target)
    for i, leaf in enumerate(fit['nodes']):
        use = leaves == leaf
        weight = w[use]/w[use].sum()
        features = q[use]-weight@q[use]
        for channel in range(2):
            mu = weight@y[use, channel]

            def independent(b):
                exp = np.exp(features@b)
                pred = mu*exp/(weight@exp)
                return .5*weight@((pred-target[use, channel])/scales[channel])**2 + .5*b@b

            solved = minimize(independent, np.zeros(7), method='BFGS', options={'gtol': 1e-7})
            assert np.linalg.norm(solved.jac) < 1e-5
            np.testing.assert_allclose(fit['coef'][i, channel], solved.x, atol=2e-6)


def test_leaf_weight_rescaling_does_not_change_conditional_fit():
    leaves, q, y, w, target = fixture()
    altered = w*np.where(leaves == 3, .02, np.where(leaves == 19, 90., 3.))
    a = api.cost_leaf(leaves, q, y, w, np.ones(2), squared_target=target)
    b = api.cost_leaf(leaves, q, y, altered, np.ones(2), squared_target=target)
    for key in ('means', 'coef', 'log_normalizer', 'target_means'):
        np.testing.assert_allclose(a[key], b[key], rtol=1e-8, atol=1e-8)


def test_zero_actual_harm_stays_zero_despite_negative_equivalent_target():
    leaves, q, y, w, target = fixture()
    y[leaves == 19, 1] = 0
    target[leaves == 19, 1] = -2
    fit = api.cost_leaf(leaves, q, y, w, np.ones(2), squared_target=target)
    at = np.flatnonzero(fit['nodes'] == 19)[0]
    assert fit['zero_target'][at, 1]
    np.testing.assert_array_equal(fit['coef'][at, 1], 0.)
    assert fit['log_normalizer'][at, 1] == 0.


def test_iteration_exhaustion_is_a_failure_not_a_partial_model():
    leaves, q, y, w, target = fixture()
    with pytest.raises(RuntimeError, match='not converged'):
        api.cost_leaf(leaves, q, y, w, np.ones(2), squared_target=target,
                      max_iter=1, tolerance=1e-12)
