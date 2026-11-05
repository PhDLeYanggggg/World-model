import numpy as np
import pytest
from src.evaluation.m3w_harm_admission import analyze


def test_attribution_separates_new_harm_from_recovered_benefit_and_ignores_unknown():
    y = np.array([[[2., 0.]], [[-4., 0.]], [[6., 0.]], [[8., 0.]]])
    before = np.array([[1.], [1.], [-1.], [1.]])
    after = -before
    a = analyze(before, after, y, np.ones((4, 1), bool), np.array([1, 1, 1, 0], bool), np.ones(4))
    assert a['slices']['new_harmful']['count'] == 1
    assert a['slices']['new_harmful']['weighted_cost'] == 2
    assert a['slices']['new_useful']['weighted_cost'] == 4
    assert a['slices']['repaired_harmful']['weighted_cost'] == 6
    assert a['positive_target_mass'] == 8
    assert a['negative_target_mass'] == 4
    assert a['cost_change'] == -8


def test_attribution_rejects_multi_action_or_nonfinite_scores():
    y = np.ones((1, 2, 2)); scores = np.ones((1, 2))
    with pytest.raises(ValueError):
        analyze(scores, scores, y, np.ones((1, 2), bool), np.ones(1, bool), np.ones(1))
    scores[0, 0] = np.nan
    with pytest.raises(ValueError):
        analyze(scores, scores, y, np.array([[1, 0]], bool), np.ones(1, bool), np.ones(1))
