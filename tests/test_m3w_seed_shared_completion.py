import numpy as np
import pytest

from src.evaluation.m3w_seed_shared_completion import seed_shared_completion


def test_swapped_seed_predictions_cancel_at_one_shared_label():
    ref = np.array([[[[0., 0.]]], [[[2., 0.]]], [[[4., 0.]]]])
    pred = ref[[1, 2, 0]]
    y = np.full((1, 1, 2), np.nan)
    result = seed_shared_completion(ref, pred, y, np.array([[False]]), np.zeros_like(y))
    assert result["diagonal_upper"] == pytest.approx(8./3.)
    assert result["shared_lower"] == result["shared_upper"] == 0.
    assert result["template_low"] == result["template_high"] == 0.
    assert result["tighter_missing_steps"] == 1


def test_known_steps_are_preserved_and_unknown_witnesses_are_attainable():
    ref = np.zeros((3, 1, 2, 2))
    pred = ref.copy()
    pred[..., 0] = 2.
    y = np.array([[[2., 0.], [np.nan, np.nan]]])
    result = seed_shared_completion(ref, pred, y, np.array([[True, False]]), ref[0])
    assert result["observed"] == 2.
    assert result["diagonal_lower"] == result["shared_lower"] == result["template_low"] == 0.
    assert result["diagonal_upper"] == result["shared_upper"] == result["template_high"] == 4.


def test_no_missing_steps_needs_no_completion():
    ref = np.zeros((3, 1, 1, 2))
    pred = ref+1
    y = pred[0]
    result = seed_shared_completion(ref, pred, y, np.array([[True]]), ref[0])
    assert result["observed"] == pytest.approx(np.sqrt(2))
    assert result["shared_lower"] == result["template_high"] == result["observed"]
    assert result["missing_steps"] == 0
