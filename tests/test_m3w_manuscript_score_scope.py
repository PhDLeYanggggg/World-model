import numpy as np
import pytest

from src.evaluation.m3w_opendd_location import paired as step_score
from src.evaluation.m3w_selection_forecast_readout import paired as row_score


def test_partial_row_has_different_debits_under_the_two_scoring_rules():
    cv = np.zeros((2, 12, 2))
    pred = cv.copy()
    pred[0, :2, 0] = 2.
    pred[1, :2, 0] = 3.
    valid = np.zeros((2, 12), dtype=bool)
    valid[0, 0] = True
    target = np.full_like(cv, np.nan)
    target[0, 0] = (2., 0.)
    rows = row_score(cv, pred, cv, target, valid)
    known, lower = step_score(cv, pred, target, valid)
    assert rows['unknown_path_disagreement'] == 3.
    assert rows['known_pp'] == 100.
    assert rows['completion_lower_pp'] == -50.
    assert known.sum() == 2.
    assert 100.*lower.sum()/2. == pytest.approx(-300.)
