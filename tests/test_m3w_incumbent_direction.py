import numpy as np
import torch

from scripts.audit_m3w_moving_observability_cost import decision_partition, paired_terms
from src.world_model import m3w_incumbent_direction as m


def test_direction_head_is_antisymmetric_and_identical_paths_zero():
    net = m.PairedGain(3)
    a, b = torch.randn(2, 4, 3), torch.randn(2, 4, 3)
    torch.testing.assert_close(net(a, b), -net(b, a), rtol=0, atol=0)
    torch.testing.assert_close(net(a, a), torch.zeros(2, 4), rtol=0, atol=0)


def test_one_swap_per_query_and_exact_matched_control():
    legal = np.array([[False, True], [False, True], [False, True]])
    initial = np.zeros(3, dtype=int)
    groups = [np.array([0, 1]), np.array([2])]
    scores = np.array([[0., 1.], [0., 2.], [0., -3.]])
    a, used = m.single_swaps(scores, legal, groups, initial)
    np.testing.assert_array_equal(a, [0, 1, 0])
    b, used2 = m.single_swaps(-scores, legal, groups, initial, used)
    np.testing.assert_array_equal(b, [1, 0, 0])
    np.testing.assert_array_equal(used, used2)
    np.testing.assert_array_equal(m.lower_gain([2, 2], np.array([0, 1]), [5, 5]), [2, -5])


def test_missing_radius_is_counted_once_in_row_partition():
    a = np.array([0, 0, -1, -1, 0, 0])
    b = np.array([1, -1, 1, 1, 1, 0])
    core = np.array([1, -1, 1, -1, -1, -1])
    p = decision_partition(a, b, core)
    np.testing.assert_array_equal(np.stack(list(p.values())).sum(0), [1, 1, 1, 1, 1, 0])
    x = np.zeros((6, 2, 2))
    y = x.copy()
    y[:5, :, 0] = 1.
    target = np.ones_like(x)
    mask = np.ones((6, 2), dtype=bool)
    mask[[0, 2, 4]] = False
    z, terms = paired_terms(x, y, x, target, mask, np.ones(6), p)
    assert z["unknown_radius"] == 3.
    assert terms["direction_fill"]["unknown_radius"] == 1.
