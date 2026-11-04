import numpy as np
import pytest
from scripts.evaluate_m3w_moving_adaptive_record_cost import check_choice_hashes, check_prediction_binding, source
from src.evaluation.m3w_verified_choice_cache import POLICIES, binding


def test_every_control_array_must_match_verified_parent():
    choices = {k: np.array([-1, 6], np.int64) for k in POLICIES}
    expected = {k: source.array_hash(v) for k, v in choices.items()}
    check_choice_hashes(choices, expected)
    choices['joint_free'][0] = 6
    with pytest.raises(ValueError):
        check_choice_hashes(choices, expected)


def test_missing_empty_control_cannot_be_dropped():
    choices = {k: np.array([-1, -1], np.int64) for k in POLICIES}
    expected = {k: source.array_hash(v) for k, v in choices.items()}
    del choices['joint_matched']
    with pytest.raises(ValueError):
        check_choice_hashes(choices, expected)


def test_original_solver_prediction_binding_uses_its_own_shape_dtype_order():
    prediction = np.arange(70, dtype=np.float64).reshape(2, 7, 5)
    meta = dict(original_prediction_binding=binding(prediction))
    assert meta['original_prediction_binding'] != source.array_hash(prediction)
    check_prediction_binding(meta, prediction)
    changed = prediction.copy()
    changed[0, 0, 0] = .01
    for value in (changed, prediction.astype(np.float32), prediction.reshape(7, 2, 5)):
        with pytest.raises(ValueError):
            check_prediction_binding(meta, value)
