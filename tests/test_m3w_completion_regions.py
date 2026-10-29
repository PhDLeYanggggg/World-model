import copy
import numpy as np
import pytest

from scripts.diagnose_m3w_completion_regions import check_pair
from src.evaluation.m3w_policy_region_shift import compare_regions


def test_same_region_check_preserves_unknowns_and_observed_mass():
    p = np.tile([1., .001, 10., 3., .001], (3, 7, 1)); q = 2*p
    y = np.tile([1., .1, 5., 2., .1], (3, 7, 1)); y[-1] = np.nan
    allowed = np.ones((3, 7), bool); a, b = np.array([0, -1, -1]), np.array([-1, 1, 2])
    left = compare_regions(p, y, allowed, a, b, np.ones(3), tuple(range(7)))
    right = compare_regions(q, y, allowed, b, a, np.ones(3), tuple(range(7)))
    check_pair(left, right)
    assert left['frozen_teacher_region']['unknown_action_views'] == 1
    changed = copy.deepcopy(right); changed['selected']['unknown_action_views'] = 0
    with pytest.raises(ValueError):
        check_pair(left, changed)
    changed = copy.deepcopy(right); changed['all_legal']['observed_known_mass']['harm'] += 1.
    with pytest.raises(ValueError):
        check_pair(left, changed)
