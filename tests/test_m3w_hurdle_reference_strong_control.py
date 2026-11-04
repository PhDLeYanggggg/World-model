import numpy as np
import pytest

from scripts.audit_m3w_hurdle_strong_reference import scores


def fixture():
    causal = dict(ids=np.arange(3), sites=np.array(['a'] * 3),
                  groups=[np.arange(3)], allowed=np.ones((3, 7), dtype=bool))
    return causal, np.array([2., 999., 4.]), np.array([True, False, True])


def test_conditional_score_excludes_unknown_supervision_without_imputing_zero():
    causal, target, known = fixture()
    prediction = dict(head=np.array([1., 5., 4.]))
    assert scores(target, known, causal, prediction)['head']['MSE'] == .5
    target[1] = 10000.
    assert scores(target, known, causal, prediction)['head']['MSE'] == .5


def test_original_bound_keeps_unknown_reference_zero_contribution_separate():
    causal, target, known = fixture()
    prediction = dict(head=np.array([1., 5., 4.]))
    result = scores(target, known, causal, prediction, bounded=True)['head']
    assert result['MSE'] == pytest.approx(26 / 3)
    assert result['MAE'] == pytest.approx(2.)
