import numpy as np
import pytest

from src.world_model import m3w_observation_augmentation as aug


def test_transform_keeps_labels_scaled_and_rebases_to_noisy_observation():
    h = np.ones((2, 8, 2))
    y = np.full((2, 12, 2), 3.)
    v = np.ones((2, 12), bool)
    v[1, -1] = False
    y[1, -1] = np.nan
    noise = np.zeros_like(h)
    noise[:, -1] = .2
    x, z = aug.transform(h, y, v, .5, noise)
    np.testing.assert_allclose(z[v]+np.broadcast_to(x[:, -1:], z.shape)[v], 1.)
    np.testing.assert_array_equal(z[~v], 0.)


def test_count_match_keeps_legal_candidates_even_with_negative_gains():
    g = np.array([[4., 5.], [-2., -1.], [0., 1.]])
    legal = np.array([[True, False], [True, True], [False, False]])
    np.testing.assert_array_equal(aug.match_gain(g, legal, 2), [1, 2, 0])
    np.testing.assert_array_equal(aug.match_gain(g, legal, 0), 0)
    with pytest.raises(ValueError, match="legal recipients"):
        aug.match_gain(g, legal, 3)
