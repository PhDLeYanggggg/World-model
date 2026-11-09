import numpy as np

from src.world_model.m3w_sensor_noise_control import normalized, perturb


def test_sensor_noise_keeps_absolute_targets_and_missing_mask():
    rng = np.random.default_rng(97)
    h = rng.normal(size=(35, 8, 2))
    y = rng.normal(size=(35, 12, 2))
    y[0, :3] = np.nan
    moved, target = perturb(h, y, rng)
    assert (moved != h).any()
    np.testing.assert_allclose(target+moved[:, -1:], y+h[:, -1:], atol=1e-12)
    np.testing.assert_array_equal(np.isfinite(target), np.isfinite(y))


def test_normalization_uses_only_past_and_restores_coordinates():
    rng = np.random.default_rng(9)
    h = rng.normal(size=(4, 8, 2))
    h[-1] = 2.
    y = rng.normal(size=(4, 12, 2))
    z, scale, target = normalized(h, y)
    a, b, missing = normalized(h)
    np.testing.assert_array_equal(a, z)
    np.testing.assert_array_equal(b, scale)
    assert missing is None and np.isfinite(z).all()
    np.testing.assert_allclose(z*scale[:, None, None]+h[:, -1:], h)
    np.testing.assert_allclose(target*scale[:, None, None], y)
