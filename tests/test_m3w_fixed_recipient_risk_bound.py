from itertools import combinations

import numpy as np
import pytest

from scripts.audit_m3w_fixed_recipient_risk_bound import query_minimum


def test_each_component_matches_exhaustive_search():
    anchor = np.array([[1., 5., 1.], [3., 2., 0.], [2., 1., 3.]])
    neural = np.array([[2., 0., 2.], [0., 2., 1.], [2., 4., 0.]])
    all_assignments = []
    for k in range(4):
        brute = []
        for take in combinations(range(3), k):
            rows = anchor.copy()
            rows[list(take)] = neural[list(take)]
            brute.append(rows.sum(0))
        all_assignments.extend(brute)
        fixed, free = query_minimum(anchor, neural, k)
        np.testing.assert_array_equal(fixed, np.min(brute, axis=0))
    np.testing.assert_array_equal(free, np.min(all_assignments, axis=0))


def test_ineligible_neural_rows_still_allow_a_free_motion_choice():
    a, b = np.ones((2, 3)), np.full((2, 3), np.inf)
    np.testing.assert_array_equal(query_minimum(a, b, 0)[1], [2, 2, 2])
    with pytest.raises(ValueError, match='Cannot realize'):
        query_minimum(a, b, 1)
