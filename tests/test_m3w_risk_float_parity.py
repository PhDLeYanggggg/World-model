import numpy as np
import pytest

from src.evaluation.m3w_risk_float_parity import compare


def test_normalized_roundoff_does_not_move_decisions():
    before=np.array([[1.74798953848823e-5,-1e-4]])
    after=before+np.array([[1.06280145e-9,0.]])
    r=compare(after,before,np.array([.02,.01]))
    assert r['original_scalar_tolerance_violations']==1
    assert r['all_channel_decisions_exact']


def test_even_tiny_sign_change_fails():
    with pytest.raises(AssertionError):
        compare(np.array([[1e-12,-1.]]),np.array([[-1e-12,-1.]]),np.ones(2))


def test_training_scale_matters_and_large_normalized_error_fails():
    with pytest.raises(AssertionError):
        compare(np.array([[1.01e-6,-1.]]),np.array([[1e-6,-1.]]),np.array([1e-6,1.]))
