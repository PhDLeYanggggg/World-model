import numpy as np

from scripts.diagnose_m3w_tumdot_ing_frozen_transfer import decompose
from scripts.run_m3w_tumdot_ing_frozen_transfer import blend, score, windows


def test_disappearing_target_does_not_change_roster_or_history():
    data = np.tile(np.arange(229, dtype=float)[None, :, None], (2, 1, 2))
    h, y, starts = windows(data)
    data[0, 96:] = np.nan
    h_new, y_new, starts_new = windows(data)
    assert starts == starts_new == [0]
    np.testing.assert_array_equal(h, h_new)
    assert np.isfinite(y[0]).all() and np.isnan(y_new[0]).all()


def test_blends_match_recipient_and_amplitude():
    cv = np.zeros((3, 12, 2))
    forecast = cv.copy()
    forecast[1:] = 2.
    old = np.ones_like(cv)
    neural, matched = blend(cv, forecast, old, np.array([True, True, False]), .05)
    np.testing.assert_array_equal((neural != cv).any(axis=(1, 2)), [False, True, False])
    np.testing.assert_array_equal((matched != cv).any(axis=(1, 2)), [False, True, False])
    np.testing.assert_allclose(neural[1], .1)
    np.testing.assert_allclose(matched[1], .05)


def test_missing_steps_pay_disagreement_and_identity_does_not_pass_undefined_risk():
    cv = np.zeros((2, 12, 2))
    valid = np.array([[True]*12, [False]*12])
    y = np.where(valid[..., None], np.ones_like(cv), np.nan)
    d = decompose(cv, np.ones_like(cv), cv, y, valid)
    assert d['observed_pp'] == 100
    assert d['debit_pp'] == 100
    assert d['lower_pp'] == 0
    risk = score(cv, cv, cv, y, valid, 'test', [2., 3.])['risk']
    assert risk['completion_easy_risk_upper'] is None
    assert risk['supported'] is False
