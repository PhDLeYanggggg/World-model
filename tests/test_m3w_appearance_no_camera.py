import numpy as np
import pytest

from scripts.run_m3w_appearance_no_camera import no_camera_geometry


def test_only_camera_input_removed_without_refitting_normalization():
    x = np.arange(96, dtype=float).reshape(3, 32)
    mean, std = np.ones(32), np.ones(32) * 2
    original = x.copy()
    result = no_camera_geometry(x, mean, std)
    np.testing.assert_array_equal(result[:, :28], np.clip((x - mean) / std, -10, 10).astype(np.float32)[:, :28])
    np.testing.assert_array_equal(result[:, 28:], np.zeros((3, 4)))
    np.testing.assert_array_equal(x, original)
    x[:, 28:] += 1e8
    np.testing.assert_array_equal(no_camera_geometry(x, mean, std), result)


def test_bad_schema_and_scale_rejected():
    with pytest.raises(ValueError):
        no_camera_geometry(np.zeros((1, 28)), np.zeros(32), np.ones(32))
    with pytest.raises(ValueError):
        no_camera_geometry(np.zeros((1, 32)), np.zeros(32), np.zeros(32))
