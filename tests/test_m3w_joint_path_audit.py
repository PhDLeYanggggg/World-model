import numpy as np

from src.evaluation.m3w_joint_path_audit import bootstrap, query


def test_common_translation_changes_scene_error_but_not_pair_geometry():
    h = np.zeros((2, 8, 2))
    h[1, :, 0] = 3.
    target = np.zeros((2, 12, 2))
    first = target + 2.
    arms, pairs, changed, counts = query(h, np.stack((first, target)), target,
                                        np.ones((2, 12), bool), [(0, 1)])
    np.testing.assert_array_equal(pairs, np.zeros_like(pairs))
    assert counts == (12, 12, 1, 1)
    assert not changed[0]
    assert arms[0, 3] > arms[1, 3]


def test_unknown_step_bounds_contain_true_paired_difference():
    h = np.zeros((2, 8, 2))
    y = np.zeros((2, 12, 2))
    a = y.copy()
    a[0, :, 0] = 1.
    valid = np.ones((2, 12), bool)
    valid[1, -1] = False
    paths = np.stack((a, y))
    _, b, changed, counts = query(h, paths, y, valid, [(0, 1)])
    assert changed[0] and counts == (12, 11, 1, 0)
    np.testing.assert_allclose(b[0, :3], [1., 10./12, 1.])
    assert np.isnan(b[0, 3])
    np.testing.assert_array_equal(b[0, 4:], [-1., 1.])
    for end in (-6., 3.):
        y[0, -1, 0] = end
        _, actual, _, _ = query(h, paths, y, np.ones_like(valid), [(0, 1)])
        assert b[0, 1] <= actual[0, 0] <= b[0, 2]


def test_bootstrap_does_not_drop_sources_or_seed_cells():
    draw = np.array([[0, 1], [1, 0]])
    assert bootstrap([[1., 2.], [3., 4.]], draw)['point'] == 2.5
    assert bootstrap([[1., 2.], [3., np.nan]], draw) is None
