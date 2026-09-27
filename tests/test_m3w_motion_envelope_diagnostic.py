import numpy as np
import pytest
from src.evaluation.m3w_motion_envelope_diagnostic import envelope_statistics


def test_projected_circle_lower_bound_and_zero_budget():
    b=np.zeros((1,2,2)); p=np.array([[[2.,0.],[0.,0.]]]); y=np.array([[[5.,0.],[3.,0.]]])
    d=envelope_statistics(b,p,y,np.ones((1,2),bool),np.array([[2.,0.]]))
    assert d['oracle_envelope_lower_ADE']==3 and d['prediction_ADE']==3
    assert d['lower_fraction_of_prediction_error']==1 and d['zero_budget_positive_error_steps']==1
    assert d['fraction_above_95pct_budget']==1


def test_invalid_labels_do_not_enter_diagnostic():
    b=np.zeros((1,2,2)); y=np.array([[[np.nan,np.nan],[1.,0.]]])
    d=envelope_statistics(b,b,y,np.array([[False,True]]),np.ones((1,2)))
    assert d['rows']==1 and d['labeled_steps']==1 and d['oracle_envelope_lower_ADE']==0


def test_envelope_violation_is_not_a_pass():
    b=np.zeros((1,1,2)); p=np.ones_like(b)*5
    with pytest.raises(ValueError,match='outside'):
        envelope_statistics(b,p,b,np.ones((1,1),bool),np.ones((1,1)))
