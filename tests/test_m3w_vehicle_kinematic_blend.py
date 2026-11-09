import numpy as np
import pytest

from src.world_model import m3w_vehicle_kinematic_blend as m


def test_kinematic_paths_are_origin_invariant_and_obey_the_bound():
    rng = np.random.default_rng(19)
    h = rng.normal(size=(12, 8, 2))
    cv = (h[:, -1:]-h[:, -2:-1])*np.arange(1, 13)[None, :, None]
    scale = np.linspace(.1, 3., len(h))
    basis = m.paths(h, cv, scale)
    np.testing.assert_allclose(basis, m.paths(h+[47., 12.], cv, scale), atol=1e-12)
    assert np.linalg.norm(basis, axis=2).max() <= 2.+1e-12
    path = m.predict(basis, cv, scale, [.25]*4)
    assert np.all(np.linalg.norm(path-cv, axis=-1).max(1) <= 2*scale+1e-12)
    np.testing.assert_array_equal(m.predict(basis, cv, scale, [0.]*4), cv)


def test_gradient_and_fitting_never_use_missing_coordinates():
    rng = np.random.default_rng(8)
    b = rng.normal(size=(7, 12, 2, 4))
    cv = np.zeros((7, 12, 2))
    s = np.ones(7)
    mask = np.ones((7, 12), bool)
    mask[::2, -1] = False
    y = .4*b[..., 0]
    y[~mask] = np.nan
    t = np.where(mask[..., None], y, 0.)
    x = np.array([.1, .2, .1, .1])
    _, analytic = m.completion_terms(x, b, t)
    jac = [(m.completion_terms(x+z, b, t)[0]-m.completion_terms(x-z, b, t)[0])/2e-6
           for z in 1e-6*np.eye(4)]
    np.testing.assert_allclose(analytic, jac, atol=1e-7)
    fit = m.fit(b, cv, y, mask, s)
    y[~mask] = 300.
    assert fit == m.fit(b, cv, y, mask, s)
    assert fit['donor_loss'] < fit['cv_donor_loss']
    with pytest.raises(ValueError, match='feasible'):
        m.predict(b, cv, s, [1.]*4)
