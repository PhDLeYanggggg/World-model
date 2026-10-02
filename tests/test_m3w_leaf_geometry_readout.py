import copy

import numpy as np
import pytest

from scripts.verify_m3w_leaf_geometry import verify_policy, equal
from src.world_model.m3w_unknown_outcome_bounds import completion_bounds


def test_known_unknown_arithmetic():
    y=np.array([[1.,0.,2.,2.,0.],[np.nan]*5])
    p=completion_bounds(y,np.ones(2,bool),np.array([1.,2.]))
    assert verify_policy(p)==6
    assert not p['finite_completion_supported']


def test_zero_denominator_cannot_be_promoted():
    y=np.array([[1.,0.,2.,2.,0.]])
    p=completion_bounds(y,np.zeros(1,bool),np.ones(1))
    assert verify_policy(p)==6
    bad=copy.deepcopy(p);bad['finite_completion_supported']=True
    with pytest.raises(AssertionError):verify_policy(bad)


def test_changed_harm_and_none_zero_rejected():
    p=completion_bounds(np.array([[0.,1.,2.,2.,1.]]),np.ones(1,bool),np.ones(1))
    p['selected_known_harm_mass']=0
    with pytest.raises(AssertionError):verify_policy(p)
    with pytest.raises(AssertionError):equal(None,0)
