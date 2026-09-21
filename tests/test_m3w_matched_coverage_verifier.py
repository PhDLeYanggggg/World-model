import numpy as np
from scripts.verify_m3w_native_matched_coverage import partition_choice
from src.evaluation.m3w_native_matched_coverage import top_count


def test_independent_partition_agrees_on_ties_and_all_counts():
    ids = np.array([8, 1, 7, 3, 6, 2])
    scores = np.array([1., -1., 0., 1., 1., 0.])
    eligible = np.array([1, 1, 0, 1, 1, 1], bool)
    for k in range(eligible.sum()+1):
        np.testing.assert_array_equal(partition_choice(scores, eligible, ids, k), top_count(scores, eligible, ids, k))


def test_partition_can_leave_all_ineligible_unknowns_untouched():
    scores = np.zeros(3)
    np.testing.assert_array_equal(partition_choice(scores, np.zeros(3, bool), np.arange(3), 0), np.zeros(3, bool))
