import numpy as np
import torch

from src.world_model.m3w_source_start_probe import stationary_membership,start_supervision,training_rows,fit_normalizer,normalize,fit_mlp,fit_trees,probabilities,metrics


def test_past_membership_separate_from_incomplete_future_labels():
    x=np.zeros((3,476)); x[1,0]=1
    assert stationary_membership(x).tolist()==[True,False,True]
    y=np.zeros((3,12,2)); mask=np.ones((3,12),bool)
    y[0,-1]=1; mask[2,-1]=False; y[2,-1]=np.nan
    label,complete=start_supervision(y,mask)
    assert label.tolist()==[1,0,0] and complete.tolist()==[True,True,False]
    assert stationary_membership(x).tolist()==[True,False,True]


def test_source_only_ignores_target_domain_inputs_labels_and_mixed_mass():
    x=np.arange(12).reshape(6,2).astype(float); y=np.array([0,1,0,1,0,1])
    a=training_rows(x[:2],y[:2],x,y,'source_only')
    b=training_rows(x[:2]+999,1-y[:2],x,y,'source_only')
    for p,q in zip(a,b):
        np.testing.assert_array_equal(p,q)
    _,_,w=training_rows(x[:2],y[:2],x,y,'mixed')
    np.testing.assert_allclose([w[:2].sum(),w[2:].sum()],[.5,.5],rtol=0,atol=1e-15)
    norm=fit_normalizer(a[0],a[2]); z,_=normalize(a[0],norm)
    assert np.isfinite(z).all()


def test_exact_mlp_resume_and_probability_replay(tmp_path):
    rng=np.random.default_rng(1); x=rng.normal(size=(30,8)).astype(np.float32); y=np.arange(30)%2; w=np.full(30,1/30)
    cfg=dict(updates=7,batch_size=8,learning_rate=.0003,weight_decay=.0001,checkpoint_every=2)
    def fit(name,stop=None):
        return fit_mlp(x,y,w,seed=17,config=cfg,identity={'data':'fixed'},checkpoint=tmp_path/name,heartbeat=lambda **_:None,stop_at=stop)
    a,_=fit('a.pt'); fit('b.pt',3); b,result=fit('b.pt')
    assert result['new_updates']==4
    np.testing.assert_array_equal(probabilities(a,x),probabilities(b,x))
    c,done=fit('b.pt'); assert done['new_updates']==0
    np.testing.assert_array_equal(probabilities(c,x),probabilities(b,x))


def test_brier_lift_not_accuracy_or_trajectory_improvement():
    y=np.array([0,0,1,1]); p=np.array([.1,.2,.8,.9]); m=metrics(y,p,.5)
    assert np.isclose(m['brier_lift'],.225)
    assert m['probability_not_trajectory_utility']


def test_trees_exact_checkpoint_extension_and_completed_resume(tmp_path):
    rng=np.random.default_rng(9); x=rng.normal(size=(50,8)); y=np.arange(50)%2; w=np.full(50,1/50)
    cfg=dict(n_estimators=16,min_samples_leaf=2,max_features=1.)
    def fit(name,stop=None):
        return fit_trees(x,y,w,seed=17,config=cfg,identity={'fixed':1},checkpoint=tmp_path/name,heartbeat=lambda **_:None,stop_at=stop)
    a,_=fit('a.joblib'); fit('b.joblib',5); b,r=fit('b.joblib')
    assert r['new_trees']==11 and r['complete']
    np.testing.assert_array_equal(probabilities(a,x),probabilities(b,x))
    c,r=fit('b.joblib'); assert r['new_trees']==0
    np.testing.assert_array_equal(probabilities(c,x),probabilities(b,x))
