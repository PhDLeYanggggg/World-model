import numpy as np
from scripts.verify_m3w_native_protected_risk import labels, check_event
from src.evaluation.m3w_native_protected_risk_eval import event_quality


def test_independent_targets_do_not_label_partial_future_as_safe():
    cv = np.array([0., 1e-12, 0., 2.])
    harm = np.array([3., 4., 5., 6.])
    complete = np.array([True, True, False, True])
    np.testing.assert_equal(labels(cv, harm, complete, True),
        [[1., 3.], [0., 0.], [np.nan, np.nan], [0., 0.]])


def test_independent_event_check_covers_edges_and_unknowns():
    p, y = np.array([0., .5, 1., .01]), np.array([0., 1., 1., np.nan])
    complete = np.array([True, True, True, False])
    check_event(p, y, complete, dict(rows=3, positive_rows=2, brier=1/12,
        score_le_0p01_rows=1, score_le_0p01_positive_rows=0, ece=1/6))


def test_float32_mean_and_sum_rounding_is_not_an_ece_mismatch():
    rng = np.random.default_rng(83)
    p = rng.random(15097).astype(np.float32)
    y = (rng.random(len(p)) < p).astype(float)
    full = np.ones(len(p), bool)
    error = check_event(p, y, full, event_quality(p, y, full))
    assert error < 4*np.finfo(np.float32).eps
