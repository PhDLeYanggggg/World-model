import numpy as np
import pytest

from scripts import run_m3w_opendd_easy_reference_release as study
from src.evaluation.m3w_selection_forecast_readout import policy


def test_reference_matches_original_risk_denominator():
    cv = np.zeros((5, 12, 2))
    y = cv.copy()
    y[0, 0, 0] = .1
    y[1, :, 0] = .5
    y[2, :, 0] = .01
    valid = np.ones((5, 12), dtype=bool)
    valid[0, 1:] = False
    valid[2] = False
    result = study.easy_events(cv, y, valid, .2)
    np.testing.assert_array_equal(result, [True, False, False, False, False])
    for rows in ([0, 1], [1, 2], [3, 4]):
        pred = cv.copy()
        pred[rows, :, 0] = .02
        r = policy(pred, cv, cv, y, valid, ['r']*5, easy_cut=.2, hard_cut=1.)['recordings']['r']
        assert (r['completion_easy_risk_upper'] is not None) == result[rows].any()


def test_reference_features_ignore_absolute_origin_and_recipient_order():
    rng = np.random.default_rng(17)
    history = rng.normal(size=(4, 8, 2))
    offsets = rng.normal(size=(4, 2))
    cv = rng.normal(size=(4, 12, 2))
    ols = rng.normal(size=cv.shape)
    a = study.actor_features(history, offsets, cv, ols)
    b = study.actor_features(history, offsets+1000., cv, ols)
    np.testing.assert_allclose(a, b, atol=1e-10)
    i, j = [0, 1], [2, 3]
    np.testing.assert_array_equal(study.pair_features(a, i, j), study.pair_features(a, j, i))


def test_reference_changes_only_score_and_rejects_invalid_predictions():
    sign = np.array([.8, .4])
    p = np.array([.1, .9])
    result = study.policies(sign, p)
    assert result['sign'] is sign
    assert result['reference_only'] is p
    assert result['sign_times_reference'].argmax() == 1
    with pytest.raises(ValueError):
        study.policies(sign, np.array([1., np.nan]))
