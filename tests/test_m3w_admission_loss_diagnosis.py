import numpy as np
import pytest
from scripts.diagnose_m3w_moving_admission_loss import decompose


def test_decomposition_preserves_loss_weights_and_unknown_separation():
    p = np.array([[-1., -2.], [3., 4.], [0., -1.], [8., 8.]])
    y = np.array([[2., -1.], [-1., 3.], [1., 2.], [9., 9.]])
    w = np.array([.2, .3, .5, 0.]); known = np.array([True, True, False, True])
    rows = decompose(p, y, w, known)
    assert sum(r['weighted_MSE'] for r in rows) == pytest.approx((w*((p-y)**2).mean(1)).sum())
    for channel in ('all', 'easy'):
        subset = [r for r in rows if r['channel'] == channel]
        assert sum(r['query_weight'] for r in subset) == pytest.approx(1.)
        assert sum(r['channel_MSE_share'] for r in subset) == pytest.approx(1.)
        assert sum(r['channel_gradient_share'] for r in subset) == pytest.approx(1.)
        assert sum(r['rows'] for r in subset if r['supervision'] == 'bound_only') == 1
    assert sum(r['both_risk_outputs_admit_rows'] for r in rows) == 4
