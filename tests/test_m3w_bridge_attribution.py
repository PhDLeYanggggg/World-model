import numpy as np
import pytest
from src.world_model.m3w_bridge_attribution import motion_pair, query_groups, decisions, independent_replay, POLICIES


def test_motion_pair_has_no_neural_input_and_preserves_floor_geometry():
    g = np.zeros((3, 476), np.float32); g[:, 16:24] = np.arange(-7, 1)
    b = np.zeros((3, 12, 2)); d = b+2
    eb, ab = np.array([0, 1, 1], bool), np.array([1, 0, 1], bool)
    x, env, r, p = motion_pair(g, b, d, eb, ab)
    np.testing.assert_array_equal(x[:, -4:-2], np.zeros((3, 2)))
    np.testing.assert_array_equal(r[0], b[0]); np.testing.assert_array_equal(p[0], d[0])
    np.testing.assert_array_equal(env, [np.sqrt(8), np.sqrt(8), 0])
    with pytest.raises(TypeError): motion_pair(g, b, d, eb, ab, neural=d)


def test_matched_count_is_per_query_and_not_per_scene_or_label():
    u = np.tile([2., 1.], (6, 1)); nr = np.column_stack((np.ones(6)*100, [1, 3, 1, 1, 3, 3]))
    rr = np.column_stack((np.ones(6)*100, [3, 1, 1, 3, 1, 3]))
    groups = query_groups(np.array(['s']*6), np.ones(6), np.array([1, 1, 1, 2, 2, 2]))
    out, _, counts = decisions(u, nr, u, rr, np.ones(6, bool), np.ones(6), groups, np.arange(6))
    np.testing.assert_array_equal(counts, [2, 1])
    assert set(out) == set(POLICIES)
    for ids, count in zip(groups, counts):
        assert all(out[k][ids].sum() == count for k in ('neural_matched', 'ridge_matched', 'hash_matched'))
    replay = independent_replay(u, nr, u, rr, np.ones(6, bool), np.ones(6), groups, np.arange(6))
    for k in POLICIES: np.testing.assert_array_equal(out[k], replay[k])
    assert not np.array_equal(out['neural_matched'], out['ridge_matched'])


def test_stopped_identical_and_zero_denominator_rows_never_enter_common_pool():
    u = np.tile([2., 1.], (4, 1)); r = np.array([[100., 1.], [100., 1.], [0., 0.], [100., 1.]])
    moving, env = np.array([0, 1, 1, 1], bool), np.array([1., 0., 1., 1.])
    out, common, counts = decisions(u, r, u, r, moving, env, [np.arange(4)], np.arange(4))
    np.testing.assert_array_equal(common, [0, 0, 0, 1]); assert counts.tolist() == [1]
    for name in ('neural_matched', 'ridge_matched', 'hash_matched'):
        np.testing.assert_array_equal(out[name], [0, 0, 0, 1])


def test_query_and_row_identifiers_must_be_unique_and_complete():
    u = np.ones((2, 2)); r = np.ones((2, 2)); move = np.ones(2, bool); env = np.ones(2)
    with pytest.raises(ValueError): decisions(u, r, u, r, move, env, [np.array([0])], np.arange(2))
    with pytest.raises(ValueError): decisions(u, r, u, r, move, env, [np.arange(2)], np.zeros(2, int))


def test_equal_scores_use_stable_row_identifier_not_input_order():
    u = np.tile([2., 1.], (3, 1)); nr = np.array([[100., 1.], [100., 1.], [100., 1.]])
    rr = np.array([[100., 3.], [100., 1.], [100., 3.]])
    out, _, counts = decisions(u, nr, u, rr, np.ones(3, bool), np.ones(3), [np.arange(3)], np.array([9, 8, 7]))
    assert counts.tolist() == [1]; np.testing.assert_array_equal(out['neural_matched'], [0, 0, 1])
