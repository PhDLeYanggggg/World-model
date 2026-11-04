import numpy as np
import pytest
from scripts.evaluate_m3w_moving_adaptive_record_cost import check_choice_hashes, source
from src.evaluation.m3w_verified_choice_cache import POLICIES


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
