import numpy as np

from src.world_model.m3w_map_path_value import MAP_WIDTH, map_features, wrong_map


class Grid:
    def sample(self, x):
        return np.where(x[..., 0] < 10, x[..., 1], np.nan)


def test_only_simultaneous_supported_targets_are_exchanged():
    x = np.arange(6*2*MAP_WIDTH).reshape(6, 2, MAP_WIDTH)
    real, wrong, active = wrong_map(x, np.array([1, 1, 1, 2, 2, 3]), np.array([1, 1, 0, 1, 1, 1], dtype=bool))
    assert active.tolist() == [True, True, False, True, True, False]
    np.testing.assert_array_equal(real[active], x[active])
    np.testing.assert_array_equal(wrong[active], x[[1, 0, 4, 3]])
    assert not wrong[~active].any()


def test_map_descriptors_are_finite_and_mask_unavailable_candidate():
    h = np.zeros((2, 8, 2))
    h[..., 1] = np.arange(8)[None]
    cv = np.zeros((2, 12, 2))
    bank = np.zeros((2, 2, 12, 2))
    bank[1, 1, :, 0] = 20.
    x, supported = map_features(h, cv, bank, np.ones((2, 2)), Grid())
    assert x.shape == (2, 2, MAP_WIDTH)
    assert supported.tolist() == [True, False]
    assert not x[1].any()
