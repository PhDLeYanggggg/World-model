import numpy as np
import pytest

from src.evaluation.m3w_dut_matched_motion import disjoint_queries, matched, ols4, paired_summary, peak


def test_queries_do_not_overlap_even_on_a_ragged_clock():
    raw = [7, 8, 23, 28, 29, 47, 48, 100]
    np.testing.assert_array_equal(disjoint_queries(raw), [7, 28, 48, 100])
    with pytest.raises(ValueError):
        disjoint_queries([8, 8])


def test_ols_slope_is_anchored_to_the_last_past_observation():
    h = np.zeros((1, 8, 2))
    h[0, -4:, 0] = [2., 4., 6., 12.]
    p = ols4(h)
    expected = (np.array([-1.5, -.5, .5, 1.5]) @ h[0, -4:, 0]) / 5.
    assert p[0, 0, 0] == pytest.approx(12. + expected)
    assert p[0, -1, 0] == pytest.approx(12. + 12. * expected)


def test_zero_direction_cannot_make_a_budget_matched_edit():
    cv = np.zeros((3, 12, 2))
    neural = cv + [0., 2.]
    a = cv + [1., 0.]
    b = a.copy()
    b[1] = cv[1]
    p, on = matched(cv, neural, [True, True, False], {"a": a, "b": b})
    np.testing.assert_array_equal(on, [True, False, False])
    for result in p.values():
        np.testing.assert_allclose(peak(result - cv), [2., 0., 0.])


def test_paired_gain_charges_missing_disagreement_without_inventing_target():
    cv = np.zeros((1, 12, 2))
    p = cv + [1., 0.]
    y = np.full_like(cv, np.nan)
    y[0, :3] = [1., 0.]
    ok = np.isfinite(y).all(2)
    result = paired_summary(cv, p, y, ok, [1.], .1, .2)
    assert result["lower"] == pytest.approx(-.5)
    assert result["upper"] == pytest.approx(1.)
    assert result["complete_ref"] == 0.


def test_full_recount_keeps_past_eligible_rows_without_future():
    from scripts.audit_m3w_dut_full_ols4 import complete_motion

    time = np.arange(20)
    xy = np.column_stack((time, np.zeros(20)))
    a = np.column_stack((time, np.ones(20), xy))
    b = a[time != 9].copy()
    b[:, 1] = 2
    score = complete_motion(np.concatenate((a, b)))
    assert score["targets"] == 13 + 2 + 3
    assert score["complete"] == 1
    assert score["ols4_ade_sum"] == pytest.approx(0.)
    assert score["cv_ade_sum"] == 0.
