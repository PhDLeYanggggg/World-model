import numpy as np

from src.world_model import m3w_opendd_component_guard as model


def test_component_reconstruction_and_matched_magnitude():
    history = np.zeros((3, 8, 2))
    history[:2, :, 0] = np.arange(8)
    cv = np.zeros((3, 12, 2))
    ols = cv + [2., -1.]
    neural = cv + [1., 2.]
    parts, axis, l_ok, t_ok = model.split(history, cv, ols, neural)
    np.testing.assert_array_equal(axis, [True, True, False])
    np.testing.assert_array_equal(l_ok, [True, True, False])
    np.testing.assert_array_equal(t_ok, [False, False, False])
    np.testing.assert_allclose(parts["longitudinal"]+parts["lateral"], parts["full"])
    np.testing.assert_allclose(parts["guard"][:2, 0], [[1., 0.], [1., 0.]])
    np.testing.assert_allclose(model.peak(parts["guard"]), model.peak(parts["equal_peak"]))
    np.testing.assert_array_equal(parts["guard"][2], 0.)


def test_rotation_does_not_change_guard_decisions():
    rng = np.random.default_rng(13)
    h = rng.normal(size=(13, 8, 2)).cumsum(1)
    p = [rng.normal(size=(13, 12, 2)) for _ in range(3)]
    r = np.array([[0., 1.], [-1., 0.]])
    a, *x = model.split(h, *p)
    b, *y = model.split(h@r+17., *(v@r for v in p))
    np.testing.assert_array_equal(x, y)
    for k in a:
        np.testing.assert_allclose(a[k]@r, b[k], atol=1e-12)
