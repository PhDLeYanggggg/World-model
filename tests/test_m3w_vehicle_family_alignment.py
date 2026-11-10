import numpy as np

from src.world_model.m3w_vehicle_family_alignment import align_anchor


def test_keep_recipients_and_use_only_legal_learned_family():
    paths = np.zeros((2, 3, 12, 2))
    paths[:, 0, :, 0] = 2.
    paths[:, 1, :, 0] = -2.
    paths[:, 2, :, 1] = 2.
    anchor = np.zeros((4, 12, 2))
    anchor[[1, 3]] = paths[:, 0]
    legal = np.array([[True, True, True], [True, True, False]])
    out, before, after = align_anchor(anchor, [1, 3], paths, legal)
    np.testing.assert_array_equal(out[[0, 2, 3]], anchor[[0, 2, 3]])
    np.testing.assert_array_equal(out[1], paths[0, 2])
    np.testing.assert_array_equal(before, [[True, False, False], [True, False, False]])
    np.testing.assert_array_equal(after, [[False, False, True], [True, False, False]])
