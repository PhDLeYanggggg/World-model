import numpy as np
import pytest

from src.evaluation.m3w_opendd_location import paired
from src.evaluation.m3w_recipient_effect_split import decompose


def test_components_sum_to_direct_gain_and_missing_bounds():
    n = 5
    cv = np.zeros((n, 12, 2))
    joint, control = cv.copy(), cv.copy()
    joint[[0, 1], :, 0] = .3
    control[[1, 2], :, 0] = -.4
    y = cv+.2
    valid = np.ones(y.shape[:2], dtype=bool)
    valid[1, :3] = False
    valid[2, 7:] = False
    y[~valid] = np.nan
    r = decompose(joint, control, y, valid, [0, 1], [1, 2], cv_mass=9.)
    observed, lower = paired(control, joint, y, valid)
    _, reverse = paired(joint, control, y, valid)
    assert r['joint_only']['missing_debit_pp'] == 0.
    assert r['shared']['missing_debit_pp'] > 0.
    for k, v in [('observed_pp', observed), ('lower_pp', lower), ('upper_pp', -reverse)]:
        assert sum(x[k] for x in r.values()) == pytest.approx(v.sum()*100./(n*9.))


def test_unselected_difference_is_not_absorbed_into_a_component():
    a = np.zeros((3, 12, 2))
    b = a.copy()
    b[2, :, 0] = 1.
    with pytest.raises(ValueError, match='untouched'):
        decompose(a, b, a, np.ones(a.shape[:2], dtype=bool), [0, 1], [0, 1], 1.)
