import numpy as np
import pytest

from scripts.audit_m3w_motion_bound_headroom import oracle_errors, summarize


def test_ball_oracle_preserves_stationary_future_error():
    baseline = np.zeros((2, 2, 2))
    target = np.array([[[3., 4.], [0., 2.]], [[0., 1.], [0., 2.]]])
    before, after = oracle_errors(baseline, target, np.array([[2., 3.], [0., 0.]]))
    np.testing.assert_allclose(before, [3.5, 1.5])
    np.testing.assert_allclose(after, [1.5, 1.5])
    report = summarize(before, after, [False, True])
    assert report['zero_budget_rows'] == 1
    assert report['zero_budget_share_of_cv_error'] == pytest.approx(.3)
    assert report['optimistic_improvement_upper_bound_pct'] == pytest.approx(40.)


def test_invalid_radius_refused():
    with pytest.raises(ValueError, match='nonnegative'):
        oracle_errors(np.zeros((1, 2, 2)), np.zeros((1, 2, 2)), np.array([[0., -1.]]))
