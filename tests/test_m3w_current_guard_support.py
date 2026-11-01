import numpy as np
from src.evaluation.m3w_current_guard_support import profile
from src.world_model.m3w_current_bank_deferral import choices


def example():
    p=np.broadcast_to([.2,.1,1.,.5,.1],(2,7,5)).copy()
    y=np.broadcast_to([.2,0.,1.,1.,0.],(2,7,5)).copy();y[1]=np.nan
    return p,y,np.ones((2,7),bool),np.ones(2),np.zeros(2),np.arange(2)


def test_false_rejection_can_block_an_oracle_despite_safe_legal_bank():
    args=example();rows,mask=profile(*args);r=rows['0.0']
    assert not mask.any() and r['true_beneficial_safe_known_easy_rejected']==7
    assert r['original_legal_bank_has_known_easy_reference']
    assert r['frozen_pointwise_guard_cannot_supply_known_easy_reference']
    assert r['frozen_native_joint_cannot_supply_known_easy_reference_singleton_proof']
    assert r['unknown_rows']==1 and r['guard_kept_known_easy_rows']==0


def test_pointwise_empty_support_is_not_a_multiagent_joint_impossibility_proof():
    p,y,ok,s,r,f=example();f[:]=0
    result,_=profile(p,y,ok,s,r,f)
    assert result['0.0']['frozen_pointwise_guard_cannot_supply_known_easy_reference']
    assert not result['0.0']['frozen_native_joint_cannot_supply_known_easy_reference_singleton_proof']


def test_profile_mask_exactly_matches_real_deployment_guard():
    rng=np.random.default_rng(8);p=rng.uniform(.001,1,(24,7,5));p[:12,:,1]=0;p[:12,:,4]=0
    ok=rng.random((24,7))>.2;y=np.broadcast_to([.1,0.,1.,1.,0.],p.shape).copy()
    _,mask=profile(p,y,ok,np.ones(24),np.zeros(24),np.arange(24))
    for k in range(7):
        scores=np.zeros((24,8));scores[:,k+1]=1
        actual=choices(scores,ok,p)['guarded'][0]
        np.testing.assert_array_equal(actual==k,mask[:,k])
