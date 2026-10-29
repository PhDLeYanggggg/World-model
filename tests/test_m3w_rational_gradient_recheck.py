import numpy as np
import pytest

from scripts.check_m3w_rational_gradients import compact, aggregate


def test_account_only_common_derivative_fields_and_unknowns():
    v = dict(prediction=np.zeros((3, 5)), jacobian=np.zeros((3, 2, 2)),
        cap=np.ones(3), anchor_h=np.full(3, .1), anchor_q=np.full(3, .1), raw=np.full((3, 2), -.5))
    y = np.ones((3, 5)); y[1] = np.nan
    r = compact(v, y, np.ones(3, bool), 1.)
    assert 'causes' not in r and 'underprediction_hard_q_lower' not in r['easy']
    assert r['known'] == 2 and r['unknown'] == 1 and r['easy']['positive_underprediction'] == 2


def test_incomplete_census_rejected():
    with pytest.raises(ValueError):
        aggregate([])
