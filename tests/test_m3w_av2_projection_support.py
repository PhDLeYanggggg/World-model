import numpy as np

from src.world_model import m3w_av2_projection_support as model


def test_easy_support_needs_positive_observed_reference():
    cv = np.zeros((4, 12, 2))
    y = cv.copy()
    y[:, :, 0] = np.array([.1, .1, .0, .5])[:, None]
    v = np.ones((4, 12), dtype=bool)
    v[1] = False
    np.testing.assert_array_equal(model.labels(cv, y, v, .2), [True, False, False, False])
    assert model.either(np.array([0., 1.])) == 1.
    assert model.either(np.array([0., 0.])) == 0.


def test_support_features_are_causal_and_rotation_invariant():
    h = np.arange(8, dtype=float)[None, :, None]*np.array([[1., .3], [.1, .7]])[:, None]
    cv, ols, _, _ = model.history.motion.rollout(h, 12)
    r = np.array([[0., 1.], [-1., 0.]])
    x = model.inputs(dict(h=h, cv=cv, ols4=ols), .4)
    q = model.inputs(dict(h=h@r+[5., -7.], cv=cv@r, ols4=ols@r, future=np.full_like(cv, np.nan)), .4)
    np.testing.assert_allclose(x, q, atol=1e-7)
