import numpy as np
import torch

from src.world_model import m3w_atomic_swap as m


def test_roster_preserves_each_query_count_and_excludes_unchanged_forecasts():
    direct = np.array([0, -1, -1, -1])
    groups = [np.array([0, 1]), np.array([2, 3])]
    legal = np.array([[True, True], [False, True], [True, True], [True, True]])
    a, b = m.roster(groups, direct, legal)
    np.testing.assert_array_equal(a, [[1, 1, 0]])
    assert b.shape == (0, 3)
    assert m.first(list(m.ranked([a, b], np.array([.6]))), .5) == (1, 1, 0)


def test_removing_unknown_still_pays_original_disagreement():
    atoms = np.zeros((3, 1, 6))
    atoms[:, 0, -1] = [-4., 1., -2.]
    direct = np.array([0, -1, 0])
    known = np.array([False, True, True])
    delta, lower = m.edits(atoms, direct, known, np.array([[1, 0, 0], [1, 0, 2]]))
    np.testing.assert_array_equal(delta[:, -1], [5., 3.])
    np.testing.assert_array_equal(lower, [-3., 3.])


def test_nonadditive_term_can_be_ablated_without_changing_features():
    torch.manual_seed(2)
    net = m.SwapHead(3)
    x = torch.randn(4, 2, 3)
    before = net(x).detach()
    torch.testing.assert_close(before[:, 0], before[:, 1])
    with torch.no_grad():
        net.cross.weight.fill_(1.)
    after = net(x).detach()
    torch.testing.assert_close(before[:, 0], after[:, 0])
    assert not torch.allclose(after[:, 0], after[:, 1])
