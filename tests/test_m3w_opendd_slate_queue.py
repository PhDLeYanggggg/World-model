import numpy as np

from src.world_model import m3w_opendd_slate_queue as m


def packet(start, release):
    rows, fixed = m.menu(np.arange(2), ([3., 2.], [1., 2.]), 2)
    x = np.ones((len(rows), 11), dtype=np.float32)
    x[:, 0] = [0, 1, 2, 3]
    return m.Packet(tuple(r+start for r in rows), x, fixed, np.array([len(r) for r in rows]), release, 1.)


def test_learned_slate_changes_recipients_without_changing_clock():
    seq = [packet(0, 1), packet(2, 1)]
    predict = lambda x: np.column_stack((x[:, 0], x[:, 0]))
    logs, expired = m.stream(iter(seq), predict, "adaptive_queue")
    assert expired == 0
    assert [(a.tolist(), b.tolist(), c) for a, b, c in logs] == [
        ([1], [0], [(0, 0)]), ([3], [2], [(1, 2)])]
    before, _ = m.stream(iter(seq[:1]), predict, "adaptive_queue")
    np.testing.assert_array_equal(before[0][0], logs[0][0])


def test_same_size_actions_share_continuation_and_pair_ablation_matches():
    p = packet(0, 1)
    _, x = m.actions(p, (0, 0))
    _, z = m.actions(p, (0, 0), False)
    np.testing.assert_array_equal(z, m.remove_geometry(x.copy()))
    assert (z[:, 1:11] == 0).all()
    np.testing.assert_array_equal(z[:, 11:], x[:, 11:])
    _, y = next(m.training([p, packet(2, 0)], np.array([.1, .3, .6, .7])))
    np.testing.assert_allclose(y[1]-y[0], [.2, .2])


def test_geometry_invariant_to_origin_and_consistent_permutation():
    history = np.array([[[1., 2.]], [[2., 1.]], [[3., 4.]]])
    cv = np.zeros((3, 2, 2))
    ego = cv + np.arange(3)[:, None, None]
    a = m.geometry(history, cv, ego, [np.array([0, 2])], 1.)
    perm = [1, 0, 2]
    b = m.geometry(history[perm]+[7., 8.], cv[perm], ego[perm], [np.array([1, 2])], 1.)
    np.testing.assert_allclose(a, b)
