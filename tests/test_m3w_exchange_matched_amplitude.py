import numpy as np

from src.world_model.m3w_exchange_matched_amplitude import matched_amplitude_paths


def test_transfer_preserves_direction_endpoint_peak_and_norm_mass():
    rng = np.random.default_rng(87)
    cv = rng.normal(size=(10, 12, 2))
    d = rng.normal(size=cv.shape)
    p = cv+d
    shapes, moved = matched_amplitude_paths(cv, p)
    orig = np.linalg.norm(d, axis=-1)
    for k, sign in (('early', 1.), ('late', -1.)):
        q = shapes[k]-cv
        a = np.linalg.norm(q, axis=-1)
        np.testing.assert_array_equal(shapes[k][:, -1], p[:, -1])
        np.testing.assert_allclose(a.max(1), orig.max(1))
        np.testing.assert_allclose(a.sum(1), orig.sum(1))
        np.testing.assert_allclose(a[:, :6].sum(1)-orig[:, :6].sum(1), sign*moved[k])
        np.testing.assert_allclose(d[..., 0]*q[..., 1]-d[..., 1]*q[..., 0], 0., atol=1e-12)
        assert (moved[k] > 0.).all()


def test_no_transfer_without_headroom_or_donor():
    cv = np.zeros((2, 12, 2))
    p = cv.copy()
    p[1, :, 0] = 1.
    q, mass = matched_amplitude_paths(cv, p)
    for k in mass:
        np.testing.assert_array_equal(mass[k], 0.)
        np.testing.assert_array_equal(q[k], p)
