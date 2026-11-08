import numpy as np

from src.world_model import m3w_nested_velocity_selection as m


def test_nested_candidates_exclude_both_sources():
    rows = np.repeat(["a", "b", "c", "d"], 3)
    pairs = m.donor_folds(rows)
    assert len(pairs) == 6
    for outer, inner in m.folds(rows).items():
        for row_source, pair in inner.items():
            assert outer != row_source and pair in pairs
            use = rows[~np.isin(rows, pair)]
            assert outer not in use and row_source not in use


def test_matched_selection_skips_non_actions_and_supports_abstention():
    q = [np.array([0, 1, 2]), np.array([3, 4]), np.array([5])]
    values = np.array([10, 2, 1, -1, -2, 5])
    legal = np.array([False, True, True, True, True, True])
    np.testing.assert_array_equal(m.choose(values, legal, q), [0, 1, 0, 1, 0, 0])
    np.testing.assert_array_equal(m.choose(values, legal, q, optional=True), [0, 1, 0, 0, 0, 0])


def test_missing_target_never_inferred_as_zero_error():
    ref = np.zeros((2, 12, 2))
    delta = np.zeros_like(ref)
    delta[..., 0] = 12.
    y = ref + delta
    valid = np.ones((2, 12), dtype=bool)
    valid[1] = False
    y[~valid] = np.nan
    np.testing.assert_array_equal(m.targets(ref, delta, y, valid, np.ones(2)), [[1., 0.], [-1., 1.]])
