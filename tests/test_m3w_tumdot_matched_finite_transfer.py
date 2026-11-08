import numpy as np

from scripts.audit_m3w_tumdot_matched_transfer_capacity import best_utility, split_debit
from scripts.run_m3w_tumdot_matched_finite_transfer import matched_paths


def test_matching_preserves_shared_recipients_and_realized_peak():
    cv = np.zeros((2, 2, 2))
    anchor = cv.copy()
    anchor[0, 1, 0] = 4
    first = np.ones_like(cv)
    second = np.full_like(cv, -3)
    paths, edited = matched_paths(cv, anchor, {"first": first, "second": second}, .5)
    np.testing.assert_array_equal(edited, [True, False])
    for path in paths.values():
        np.testing.assert_array_equal(path[1], cv[1])
        np.testing.assert_allclose(np.linalg.norm(path - cv, axis=-1).max(1), [2, 0])


def test_utility_optimum_counts_missing_steps_and_partitions_debit():
    cv = np.zeros((2, 3, 2))
    y = np.full_like(cv, np.nan)
    y[0, 0] = [2, 0]
    valid = np.isfinite(y).all(2)
    a, b = cv.copy(), cv.copy()
    a[..., 0] = [[2, 2, 2], [2, 2, 2]]
    b[..., 0] = [[-1, 0, 0], [2, 0, 0]]
    np.testing.assert_array_equal(best_utility(cv, [a, b], y, valid), b)
    assert split_debit(cv, a, y, valid, 2) == {
        "observed_pp": 100, "partial_unknown_pp": 200,
        "wholly_unknown_pp": 300, "conservative_pp": -400,
    }
