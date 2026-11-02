import numpy as np
import pytest
from src.world_model import m3w_frozen_utility_guard as api


def fixture():
    p=np.zeros((2,7,5));p[...,2]=1;p[...,3]=.5;p[...,1]=.001;p[...,4]=.0001;p[...,0]=.5
    r=p.copy();r[...,0]=.0001
    return p,r,np.ones((2,7),bool)


def test_separate_utility_can_admit_new_negative_gain_without_changing_risk():
    p,r,a=fixture();chosen=api.select(p,r,a)
    np.testing.assert_array_equal(chosen,[0,0])
    assert (r[np.arange(2),chosen,1]<=.02*r[np.arange(2),chosen,2]).all()
    r[...,0]=100
    np.testing.assert_array_equal(chosen,api.select(p,r,a))


def test_original_gain_and_new_two_percent_guards_remain_required():
    p,r,a=fixture();p[0,:,0]=0;r[1,:,4]=.1
    np.testing.assert_array_equal(api.select(p,r,a),[-1,-1])
    with pytest.raises(TypeError):api.select(p,r,a,future_endpoint=np.zeros((2,2)))


def test_coupled_choice_replay_and_diagnosis_are_label_free():
    p,r,a=fixture();predictions=dict(disabled=p,real=r,noise=p.copy())
    saved={k:api.select_actions(v,a,arm="physical_neural")[0] for k,v in predictions.items()}
    choices=api.policies(predictions,a,saved);d=api.causal_diagnosis(predictions,a,choices)
    assert d["original_positive_real_guard_but_new_gain_nonpositive_action_views"]==14
    assert not d["future_labels_used"] and set(choices)==set(api.POLICIES)
