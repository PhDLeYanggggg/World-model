import numpy as np

from src.world_model import m3w_opendd_reserve_shape as m


def test_common_radius_never_amplifies_and_preserves_degenerate_slots():
    cv = np.zeros((2, 12, 2))
    neural = cv.copy()
    ols = cv.copy()
    chosen = cv.copy()
    neural[..., 0] = 2.
    ols[0, :, 1] = 3.
    chosen[..., 1] = -1.
    paths, legal, common = m.match(cv, neural, ols, chosen)
    np.testing.assert_array_equal(legal, [True, False])
    np.testing.assert_allclose(common, [1., 0.])
    for p in paths:
        np.testing.assert_allclose(m.peak(p-cv), [1., 2.])
        np.testing.assert_array_equal(p[1], neural[1])
    assert paths[0][0, 0, 0] == 1.
    assert paths[1][0, 0, 1] == 1.
    assert paths[2][0, 0, 1] == -1.


def test_hindcast_uses_only_eight_observed_points_and_is_translation_invariant():
    rng = np.random.default_rng(7)
    h = rng.normal(size=(5, 8, 2)).cumsum(1)
    a = m.analytic(h)
    b = m.analytic(h+[30., -12.])
    for x, y in zip(a, b, strict=True):
        np.testing.assert_allclose(x, y, rtol=0., atol=1e-12)
    assert a[2].shape == (5, 12, 2)
