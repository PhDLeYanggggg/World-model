import numpy as np
import pytest
from src.world_model.m3w_coherent_cost_projection import project
from scripts.verify_m3w_reference_refit_coherence import coherence


def inputs():
    rng=np.random.default_rng(20261101);p=rng.uniform(.1,8,(13,7,5))
    p[:,:,4]=np.minimum(p[:,:,4],p[:,:,1]);p[:,:,3]=np.minimum(p[:,:,3],p[:,:,2])
    mask=rng.random((13,7))>.25;mask[0]=False;mask[1]=False;mask[1,2]=True
    return p,mask


def test_projection_removes_reference_and_benefit_contradictions_without_changing_harm():
    p,ok=inputs();before=p.copy();q=project(p,ok);v=coherence(q,ok,np.ones(len(q)))
    np.testing.assert_array_equal(p,before)
    np.testing.assert_array_equal(q[:,:,[1,4]],p[:,:,[1,4]])
    np.testing.assert_array_equal(q[~ok],p[~ok])
    assert (q[:,:,0][ok]<=p[:,:,0][ok]).all()
    assert (q[:,:,0]+q[:,:,1]<=p[:,:,0]+p[:,:,1]).all()
    for d in v['violations'].values():assert d['exact_positive_action_views']==0
    for d in v['reference_spread'].values():assert d['rows_with_spread_above1e_12']==0
    np.testing.assert_array_equal(project(q,ok),q)


def test_pool_uses_only_available_candidates_and_never_creates_an_action():
    p,ok=inputs();q=project(p,ok);other=p.copy();other[~ok]=1e12
    np.testing.assert_array_equal(project(other,ok)[ok],q[ok])
    np.testing.assert_array_equal(q[0],p[0])
    np.testing.assert_array_equal(q[1,2,[2,3]],p[1,2,[2,3]])
    with pytest.raises(TypeError):project(p,ok,future_endpoint=np.zeros((13,2)))


def test_reference_MSE_nonincrease_is_a_projection_property_not_decision_evidence():
    p,ok=inputs();q=project(p,ok);truth=np.random.default_rng(17).uniform(0,10,(len(p),1,2))
    for k,j in ((2,0),(3,1)):
        old=np.square(p[:,:,k]-truth[:,:,j])[ok].sum()
        new=np.square(q[:,:,k]-truth[:,:,j])[ok].sum()
        assert new<=old+1e-10


def test_invalid_input_or_a_different_unrepaired_harm_contract_is_not_silently_changed():
    p,ok=inputs();p[2,0,4]=100;ok[2,0]=True
    with pytest.raises(ValueError):project(p,ok)
    with pytest.raises(ValueError):project(np.zeros((4,2,5)),np.ones((4,2),bool))


def test_projection_cannot_invent_an_easy_reference_or_positive_gain_when_all_references_are_zero():
    from src.world_model.m3w_frame_action_allocation import positive_actions
    p,ok=inputs();p[:,:,2:4]=0.;q=project(p,ok)
    assert not q[:,:,2:4][ok].any()
    assert not q[:,:,0][ok].any()
    assert not positive_actions(q,ok).any()
