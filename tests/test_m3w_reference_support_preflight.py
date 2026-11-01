import numpy as np
import pytest
from src.world_model import m3w_reference_action_support as support
from src.evaluation import m3w_reference_support_preflight as api


def packet():
    y=np.zeros((3,7,5));y[:,:,2:4]=10.;y[0,0,0]=4.;y[1,1,0]=3.;y[2]=np.nan
    p=np.ones_like(y);p[:,:,0]=0;p[:,:,1]=0;p[:,:,2:4]=10;p[:,:,4]=0
    pack=dict(y=y,envelope=np.full((3,7),5.),allowed=np.ones((3,7),bool),
        recordings=np.repeat('rec',3),frames=np.zeros(3,int),ids=np.arange(3))
    return pack,p,{a:np.full(3,-1,int) for a in api.CONTROLS}


def test_old_gain_mask_can_block_a_safe_reference_witness():
    pack,p,c=packet();assert not support.original.positive_actions(p,pack['allowed']).any()
    result=api.diagnose(pack,p,c)
    assert result['supports']['reference_pointwise']['positive_record_views']==1
    assert result['supports']['reference_pointwise']['witness_gain_pp']==35.
    assert result['supports']['reference_pointwise']['witness']['selected_unknown']==0


def test_removing_sign_gate_does_not_relax_either_cost_budget():
    pack,p,_=packet();chosen=np.array([0,-1,-1]);assert support.feasible(p,pack['allowed'],chosen)
    p[0,0,1]=.21;assert not support.feasible(p,pack['allowed'],chosen)
    p[0,0,1]=0;p[0,0,4]=.21;assert not support.feasible(p,pack['allowed'],chosen)


def test_reference_availability_and_real_causal_mask_are_still_required():
    pack,p,_=packet();chosen=np.array([0,-1,-1]);pack['allowed'][0,0]=False
    assert not support.feasible(p,pack['allowed'],chosen)
    pack['allowed'][0,0]=True;p[0,0,3]=0
    assert not support.feasible(p,pack['allowed'],chosen)
    assert not support.feasible(p,pack['allowed'],np.array([7,-1,-1]))


def test_zero_reference_or_empty_risk_not_credited_as_safe():
    pack,p,c=packet();pack['allowed'][:]=False
    result=api.diagnose(pack,p,c)
    assert result['supports']['reference_pointwise']['supported_record_views']==0
    assert result['supports']['reference_pointwise']['witness']['easy_selected_risk_upper'] is None
    pack['y'][:]=np.nan
    with pytest.raises(ValueError):api.diagnose(pack,p,c)


def test_pointwise_failure_cannot_prove_joint_failure():
    pack,p,_=packet();p[0,0,1]=.3;p[1,1,2]=100.
    assert not support.pointwise_actions(p,pack['allowed'])[0,0]
    assert support.feasible(p,pack['allowed'],np.array([0,1,-1]))


def test_original_positive_family_is_contained_and_inputs_unchanged():
    pack,p,_=packet();rng=np.random.default_rng(42);p[:]=rng.uniform(.01,3,p.shape)
    before=p.copy();old=support.original.positive_actions(p,pack['allowed']);new=support.reference_actions(p,pack['allowed'])
    assert (new|~old).all();np.testing.assert_array_equal(before,p)
