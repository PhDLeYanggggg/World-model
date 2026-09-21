import numpy as np

from scripts.verify_m3w_native_forecast import independent_errors
from scripts.summarize_m3w_native_forecast_training import write_once


def test_independent_error_reducer_preserves_partial_and_absent_support():
    prediction = np.zeros((3, 3, 2), np.float32)
    target = np.array([[[3, 4], [0, 2], [0, 1]],
                       [[float('nan'), float('nan')], [0, 2], [float('nan'), float('nan')]],
                       [[float('nan'), float('nan')]]*3], np.float32)
    valid = np.array([[True, True, True], [False, True, False], [False]*3])
    ade, fde = independent_errors(prediction, target, valid, np.array([2., 3., 4.]))
    np.testing.assert_allclose(ade, [16/3, 6, np.nan], equal_nan=True)
    np.testing.assert_allclose(fde, [2, np.nan, np.nan], equal_nan=True)


def test_independent_error_reducer_uses_each_native_scale():
    prediction = np.ones((2, 12, 2))
    target = np.zeros_like(prediction)
    valid = np.ones((2, 12), bool)
    ade, fde = independent_errors(prediction, target, valid, np.array([1., 10.]))
    np.testing.assert_allclose(ade, np.sqrt(2)*np.array([1., 10.]))
    np.testing.assert_allclose(ade, fde)


def test_training_summary_replay_is_immutable(tmp_path):
    path = tmp_path/'summary.csv'
    text = 'trial,step,loss\na,50,0.7\n'
    write_once(path, text)
    write_once(path, text)
    import pytest
    with pytest.raises(ValueError, match='differs'):
        write_once(path, 'trial,step,loss\na,50,0.6\n')
    assert path.read_text() == text
