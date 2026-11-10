import numpy as np
import pytest

from src.evaluation.m3w_full_grid_risk import assess


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
