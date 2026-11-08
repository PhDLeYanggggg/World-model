from itertools import combinations

import numpy as np
import pytest

from src.evaluation.m3w_perfect_switch_ceiling import capped_masses, effective_caps


def test_ceiling_matches_exhaustive_query_weighted_subsets():
    a = np.array([3., 9., 0., 7., 2.])
    offsets = np.array([0, 3, 5])
    caps = np.array([[0, 1, 2, 10], [0, 2, 1, 10]])
    expected = []
    for column in caps.T:
        best = 0
        for (lo, hi), k in zip(zip(offsets[:-1], offsets[1:]), column):
            best += max(sum(a[j] for j in subset)
                        for n in range(min(k, hi-lo)+1)
                        for subset in combinations(range(lo, hi), n))/(hi-lo)
        expected.append(best)
    np.testing.assert_allclose(capped_masses(a, offsets, caps), expected)


def test_identity_is_not_a_switch_and_cross_query_slots_are_refused():
    base = np.zeros((4, 12, 2))
    offsets = np.array([0, 2, 4])
    slots = np.array([[0, 1], [2, -1]])
    paths = np.zeros((2, 2, 12, 2))
    paths[0, 1, 4, 0] = 1
    np.testing.assert_array_equal(effective_caps(base, offsets, slots, paths), [1, 0])
    slots[0, 0] = 2
    with pytest.raises(ValueError, match="own query"):
        effective_caps(base, offsets, slots, paths)
