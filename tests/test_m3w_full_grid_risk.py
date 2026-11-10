import numpy as np
import pytest

from src.evaluation.m3w_full_grid_risk import assess, assess_easy


def test_cv_completion_attains_upper_risk_even_when_observed_risk_passes():
    cv = np.zeros((2, 12, 2))
    p, y = cv.copy(), cv.copy()
    p[..., 0], y[..., 0] = .1, 1.
    v = np.ones((2, 12), bool)
    v[0, 1:] = False
    r = assess(cv, p, y, v)
    assert r['partial_tracks'] == 1
    assert r['original_all'] == 0.
    assert r['full_grid_upper'] == pytest.approx(1/13)
    witness = np.where(v[..., None], y, cv)
    assert assess(cv, p, witness, np.ones_like(v))['original_all'] == pytest.approx(r['full_grid_upper'])
    alternate = np.where(v[..., None], y, 3.)
    assert assess(cv, p, alternate, np.ones_like(v))['original_all'] <= r['full_grid_upper']


def test_zero_denominator_is_not_a_pass():
    cv = np.zeros((2, 12, 2))
    out = assess(cv, cv+1., cv, np.zeros((2, 12), bool))
    assert out['original_all'] is None and out['full_grid_upper'] is None
    assert out['unknown_tracks'] == 2


def test_optional_easy_track_can_fail_even_though_observed_easy_is_safe():
    cv = np.zeros((2, 12, 2))
    p, y = cv.copy(), cv.copy()
    p[..., 0], y[..., 0] = .1, 1.
    y[1, :, 0] = .1
    v = np.ones((2, 12), bool)
    v[0, 1:] = False
    r = assess_easy(cv, p, y, v, easy_cut=.2)
    assert r['original_easy'] == 0.
    assert r['easy_support_guaranteed'] is True
    assert r['optional_easy_tracks'] == 1
    assert r['easy_excess_supremum'] == pytest.approx(.956)
    witness = np.where(v[..., None], y, cv)
    assert assess_easy(cv, p, witness, np.ones_like(v), easy_cut=.2)['original_easy'] == pytest.approx(1/2.2)


def test_incomplete_observed_easy_track_does_not_guarantee_support():
    cv = np.zeros((2, 12, 2))
    p, y = cv+.01, cv+.1
    v = np.ones((2, 12), bool)
    v[:, -1] = False
    r = assess_easy(cv, p, y, v, easy_cut=.2)
    assert r['original_easy'] == 0. and r['easy_excess_supremum'] == 0.
    assert not r['easy_support_guaranteed'] and not r['selected_easy_pass_for_all']


def test_optional_track_can_be_excluded_to_maximize_excess():
    cv = np.zeros((2, 12, 2))
    p, y = cv.copy(), cv.copy()
    p[..., 0], y[..., 0] = .01, .1
    v = np.ones((2, 12), bool)
    v[1, -1] = False
    r = assess_easy(cv, p, y, v, easy_cut=.2)
    assert r['easy_support_guaranteed'] and r['selected_easy_pass_for_all']
    assert r['easy_excess_supremum'] == pytest.approx(-.024)
    witness = np.where(v[..., None], y, 5.)
    assert assess_easy(cv, p, witness, np.ones_like(v), easy_cut=.2)['easy_excess_supremum'] == pytest.approx(-.024)


def test_zero_cv_floor_incomplete_track_can_approach_positive_excess():
    cv = np.zeros((2, 12, 2))
    p, y = cv.copy(), cv.copy()
    p[..., 0] = .1
    y[0, :, 0] = .1
    v = np.ones((2, 12), bool)
    v[1, -1] = False
    r = assess_easy(cv, p, y, v, easy_cut=.2)
    assert r['easy_support_guaranteed'] and not r['selected_easy_pass_for_all']
    assert r['easy_excess_supremum'] == pytest.approx(1.176)
    y[1, -1, 0] = 1e-6
    assert assess_easy(cv, p, y, np.ones_like(v), easy_cut=.2)['easy_excess_supremum'] == pytest.approx(1.176, abs=3e-6)


def test_optional_easy_threshold_is_inclusive():
    cv = np.zeros((2, 12, 2))
    p, y = cv.copy(), cv.copy()
    p[..., 0] = 1.
    y[1, :, 0] = 1.
    v = np.ones((2, 12), bool)
    v[0, -1] = False
    y[0, -2, 0] = 2.
    y[0, :-2, 0] = 1.
    assert assess_easy(cv, p, y, v, easy_cut=1.)['optional_easy_tracks'] == 1
    assert assess_easy(cv, p, y, v, easy_cut=.999)['optional_easy_tracks'] == 0
