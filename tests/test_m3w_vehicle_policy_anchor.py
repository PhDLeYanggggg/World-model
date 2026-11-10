import numpy as np

from scripts.audit_m3w_vehicle_nominee_topk import hit
from src.world_model.m3w_vehicle_policy_anchor import best_pair


def test_ranked_policy_pair_preserves_ordered_families_and_distinct_recipients():
    head = np.zeros((3, 3, 4))
    legal = np.zeros((3, 3), dtype=bool)
    legal[[0, 1], 2] = True
    legal[[1, 2], 1] = True
    head[1, 2, 2:] = 4.
    head[1, 1, 2:] = 7.
    pick, n = best_pair(head, legal, [2, 1])
    # Actor 1 is strongest in each family but cannot be used twice.
    assert n == 3
    np.testing.assert_array_equal(pick, [0, 1])
    np.testing.assert_array_equal(best_pair(head, legal, [2, 2])[0], [0, 1])


def test_shortlist_capacity_is_one_success_per_recording():
    win = np.zeros((2, 12), dtype=bool)
    win[0, 2] = win[1, 1] = True
    assert hit(win, np.arange(24)) == [0, 1, 1, 1]
