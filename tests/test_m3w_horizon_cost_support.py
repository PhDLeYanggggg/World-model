import numpy as np
import pytest
from scripts.audit_m3w_horizon_cost_support import step_errors, prefix_summary, observed_support_summary


def test_safe_full_horizon_can_hide_prefix_harm():
    out = prefix_summary([[1, 9], [2, 2]], [[3, 1], [1, 1]], np.ones(2, bool))
    assert out["full_horizon_nonharmful"] == 2
    assert out["full_nonharmful_but_any_prefix_harmful"] == 1
    assert out["full_horizon_harm_mean"] == 0
    assert out["max_prefix_harm_mean"] == 1
    assert out["prefix"][0]["harm_mean"] == 1
    assert out["prefix"][1]["harm_mean"] == 0


def test_partial_costs_not_used_as_full_horizon_targets():
    cv = np.array([[1, np.nan], [2, 2], [np.nan, np.nan]])
    neural = np.array([[3, np.nan], [1, 1], [np.nan, np.nan]])
    out = prefix_summary(cv, neural, np.ones(3, bool))
    assert out["selected"] == 3 and out["complete"] == 1
    assert out["full_nonharmful_but_any_prefix_harmful"] == 0
    s = observed_support_summary(cv, neural, np.ones(3, bool))
    assert s[0]["rows"] == 1 and s[0]["gross_harm_sum"] is None
    assert s[1]["gross_harm_sum"] == 2 and s[1]["prefix_mask_rows"] == 1


def test_gapped_masks_not_treated_as_prefixes():
    s = observed_support_summary([[1, np.nan, 3]], [[2, np.nan, 2]], np.ones(1, bool))
    assert s[2]["rows"] == 1 and s[2]["prefix_mask_rows"] == 0
    assert s[2]["gross_harm_sum"] == 0


def test_native_step_errors_ignore_only_explicit_missing_labels():
    b=np.zeros((1, 2, 2)); p=np.ones_like(b); y=np.array([[[0., 0.], [np.nan, np.nan]]])
    be, pe=step_errors(b, p, y, np.array([[True, False]]), np.array([3.]))
    assert be[0,0] == 0 and np.isclose(pe[0,0],3*np.sqrt(2))
    assert np.isnan(be[0,1]) and np.isnan(pe[0,1])


def test_empty_complete_support_reported_not_imputed():
    out=prefix_summary([[np.nan, np.nan]], [[np.nan, np.nan]], np.ones(1, bool))
    assert out["complete"] == 0 and out["max_prefix_harm_mean"] is None


@pytest.mark.parametrize("other", [[[1, np.inf]], [[1, -1]], [[1, np.nan]]])
def test_invalid_step_error_input_rejected(other):
    with pytest.raises(ValueError): prefix_summary([[1,2]],other,np.ones(1,bool))


def test_valid_target_nan_is_an_error():
    with pytest.raises(ValueError):
        step_errors(np.zeros((1,1,2)),np.zeros((1,1,2)),np.full((1,1,2),np.nan),np.ones((1,1),bool),np.ones(1))
