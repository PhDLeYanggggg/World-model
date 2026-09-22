import numpy as np
import pytest

from scripts.audit_m3w_log_cost_turnover import gain_contributions, turnover_masks


def test_turnover_is_disjoint_and_exhaustive():
    groups = turnover_masks(np.array([1, 1, 0, 0], bool), np.array([1, 0, 1, 0], bool))
    np.testing.assert_array_equal(np.stack(list(groups.values())), np.eye(4, dtype=bool))


def test_exact_difference_is_added_minus_dropped_not_retained():
    groups = turnover_masks(np.array([1, 1, 0, 0], bool), np.array([1, 0, 1, 0], bool))
    out = gain_contributions([10, 10, 10, 10], [5, 12, 8, 1], groups, np.ones(4, bool))
    assert out["old_gain_percent"] == 7.5
    assert out["new_gain_percent"] == 17.5
    assert out["difference_pp"] == 10
    assert out["group_gain_contribution_pp"]["retained"] == 12.5
    assert out["group_gain_contribution_pp"]["dropped"] == -5
    assert out["group_gain_contribution_pp"]["added"] == 5


def test_unknown_outcomes_not_filled_with_zero():
    groups = turnover_masks(np.array([0, 0], bool), np.array([1, 1], bool))
    out = gain_contributions([10, np.nan], [5, np.nan], groups, np.ones(2, bool))
    assert out["supported_rows"] == 1
    assert out["new_gain_percent"] == 50
    assert out["reference_error_sum"] == 10


def test_subset_uses_shared_denominator_and_zero_reference_has_no_percentage():
    groups = turnover_masks(np.array([1, 0], bool), np.array([0, 1], bool))
    out = gain_contributions([0, 0], [1, 2], groups, np.array([1, 0], bool))
    assert out["old_gain_percent"] is None and out["difference_pp"] is None
    assert out["group_net_error_reduction_sum"]["dropped"] == -1
    assert out["group_net_error_reduction_sum"]["added"] == 0


@pytest.mark.parametrize("old,new", [([0, 1], [1, 0]),
                                     (np.ones(2, bool), np.ones(3, bool)),
                                     (np.ones((2, 1), bool), np.ones((2, 1), bool))])
def test_invalid_decision_masks_rejected(old, new):
    with pytest.raises(ValueError):
        turnover_masks(old, new)


def test_mismatched_support_and_overlapping_partition_rejected():
    groups = turnover_masks(np.array([1, 0], bool), np.array([0, 1], bool))
    with pytest.raises(ValueError, match="support"):
        gain_contributions([1, np.nan], [1, 2], groups, np.ones(2, bool))
    groups["retained"][0] = True
    with pytest.raises(ValueError, match="partition"):
        gain_contributions([1, 2], [1, 2], groups, np.ones(2, bool))


def test_negative_error_rejected():
    groups = turnover_masks(np.array([1], bool), np.array([0], bool))
    with pytest.raises(ValueError, match="negative"):
        gain_contributions([-1], [1], groups, np.ones(1, bool))
