import numpy as np

from src import stage41_learned_shape_gain_gate as gate


def test_ridge_gain_gate_recovers_simple_signal():
    x = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=np.float64)
    y = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    w = gate._ridge_fit(x, y, lam=1e-6)
    pred = gate._ridge_predict(x, w)
    assert np.corrcoef(pred, y)[0, 1] > 0.99


def test_candidate_modes_include_tiny_and_medium_residuals():
    modes = gate._candidate_modes()
    assert ("intermediate_only", 0.25, 0.04) in modes
    assert ("all_waypoints", 0.5, 0.08) in modes
