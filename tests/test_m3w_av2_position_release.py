import numpy as np

from scripts.run_m3w_av2_position_release import calibrate
from src.world_model import m3w_av2_position_release as model


def test_calibration_cannot_skip_a_failed_high_score_or_split_a_tie():
    rows = [dict(scores={k: s for k in model.ARMS}, result={"eligible": ok}) for s, ok in ((1., True), (1., False), (.5, True))]
    assert all(q["threshold"] is None for q in calibrate(rows).values())
    assert model.choose_calibrated(calibrate(rows)) is None
    rows[0]["scores"]["risk_replay"] = 2.
    q = calibrate(rows)
    assert q["risk_replay"]["threshold"] == 2.
    assert q["risk_replay"]["accepted"] == 1
    assert model.choose_calibrated(q) == "risk_replay"


def test_history_scores_are_positive_causal_and_translation_invariant():
    h = np.arange(8, dtype=float)[None, :, None]*np.array([[1., .3], [.1, .7], [-.3, .4]])[:, None]
    cv, _, _, _ = model.history.motion.rollout(h, 12)
    x = cv.copy()
    x[:2, :, 0] += .1
    a = model.history_scores(dict(h=h, cv=cv), x, np.arange(2), .5)
    b = model.history_scores(dict(h=h+[5., 3.], cv=cv, future=np.full_like(cv, np.nan)), x, np.arange(2), .5)
    np.testing.assert_allclose(a, b, atol=1e-12)
    assert 0. < a[0] <= a[1]
