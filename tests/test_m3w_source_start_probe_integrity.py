import numpy as np
import pytest

from src.world_model.m3w_source_start_probe import fit_mlp,fit_normalizer,fit_trees,normalize,start_supervision
from src.evaluation.m3w_motion_start_probe import paired_agent_interval


def test_unseen_constant_training_feature_is_not_reestimated_from_held_inputs():
    train=np.array([[1.,4.],[3.,4.]])
    state=fit_normalizer(train,np.array([.5,.5]))
    held=np.array([[1000.,999.]])
    z,clip=normalize(held,state)
    np.testing.assert_array_equal(z,[[10.,0.]])
    np.testing.assert_array_equal(state['mean'],[2.,4.])
    assert clip==1


def test_resume_rejects_training_identity_changes(tmp_path):
    x=np.arange(64,dtype=np.float32).reshape(16,4)/64; y=np.arange(16)%2; w=np.full(16,1/16)
    mlp=dict(updates=2,batch_size=8,learning_rate=.001,weight_decay=0.,checkpoint_every=1)
    tree=dict(n_estimators=2,min_samples_leaf=2,max_features=1.)
    for method,cfg,extension in ((fit_mlp,mlp,'pt'),(fit_trees,tree,'joblib')):
        kwargs=dict(seed=17,config=cfg,checkpoint=tmp_path/('model.'+extension),heartbeat=lambda **_:None)
        method(x,y,w,identity={'split':'fit'},**kwargs)
        with pytest.raises(ValueError,match='identity'):
            method(x,y,w,identity={'split':'changed'},**kwargs)


def test_invalid_future_support_cannot_become_a_negative_supervised_example():
    target=np.zeros((2,12,2)); target[0,-1]=np.nan
    valid=np.ones((2,12),bool)
    with pytest.raises(ValueError,match='finite'):
        start_supervision(target,valid)
    valid[0,-1]=False
    _,complete=start_supervision(target,valid)
    assert complete.tolist()==[False,True]


def test_bootstrap_averages_seed_losses_not_ensemble_probabilities():
    y=np.array([0,1]); p=np.array([[0.,0.],[1.,1.]]); reference=np.full((2,2),.5)
    result=paired_agent_interval(y,p,reference,np.array(['a','b']),resamples=50)
    assert result['agent_balanced_lift']==-.25
    assert result['independent_confirmation'] is False


def test_prior_shift_is_not_attributed_to_within_site_predictive_information():
    from scripts.analyze_m3w_source_start_probe import brier_decomposition
    y=np.array([0,1,0,1]); p=np.full(4,.5)
    r=brier_decomposition(y,p,.8)
    assert r['within_site_variation_contribution']==0
    assert np.isclose(r['mean_shift_contribution'],.09)
    r=brier_decomposition(y,y.astype(float),.5)
    assert r['mean_shift_contribution']==0 and r['within_site_variation_contribution']==.25
