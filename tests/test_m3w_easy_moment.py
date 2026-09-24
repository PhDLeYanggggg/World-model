import joblib
import numpy as np
import pytest

from src.evaluation.m3w_easy_moment import targets, decisions
from src.training.m3w_easy_moment import fit, predict


def test_joint_target_is_not_product_of_marginals():
    y=np.array([[0,.01],[0,0.]])
    q=targets(y,np.array([.01,1]),np.ones(2),np.ones(2,bool),.1).mean(0)
    assert q[0]==pytest.approx(.005)
    assert q[2]*q[3]==pytest.approx(.0025)


def test_unknown_future_labels_have_no_training_target():
    q=targets(np.array([[np.nan,np.nan],[0,0]]),np.array([np.nan,0]),
              np.ones(2),np.array([False,True]),.1)
    np.testing.assert_array_equal(q,[[0,0,0,0],[0,0,1,0]])


def test_gate_cannot_accept_zero_expected_easy_denominator():
    past=np.zeros((2,8,2));past[:,-1]=1
    f=np.array([[0,0,0,.1],[0,.1,.5,.1]])
    r=decisions(f,np.ones(2),.1,np.array([[.5,.1],[.5,.1]]),past)
    np.testing.assert_array_equal(r['joint_easy_moment'],[False,True])
    np.testing.assert_array_equal(r['product_easy_marginals'],[False,False])


def test_stopped_history_and_identical_forecast_always_fallback():
    past=np.zeros((2,8,2));past[1,-1]=1
    r=decisions(np.tile([0,.1,.5,0],(2,1)),np.array([1,0]),.1,
                np.array([[.5,0],[0,0]]),past)
    assert not any(v.any() for v in r.values())


@pytest.mark.parametrize('kind',['negative','out_of_bound','bad_harm','bad_denominator','nan'])
def test_reject_invalid_predictions(kind):
    f=np.array([[.1,.2,.5,.3]])
    if kind=='negative': f[0,0]=-.1
    if kind=='out_of_bound':f[0,3]=2
    if kind=='bad_harm':f[0,0]=.4
    if kind=='bad_denominator':f[0,1]=.6
    if kind=='nan':f[0,1]=np.nan
    with pytest.raises(ValueError):
        decisions(f,np.ones(1),.1,np.array([[.3,.1]]),np.ones((1,8,2)))


def test_atomic_resume_matches_uninterrupted(tmp_path):
    rng=np.random.default_rng(2);x=rng.normal(size=(80,4))
    b=rng.random(80);d=np.ones(80);y=np.c_[np.zeros(80),rng.random(80)*.5]
    q=targets(y,b,d,np.ones(80,bool),.3)
    pr=dict(mean=x.mean(0),std=x.std(0),known=np.ones(80,bool))
    draws=np.ones(80,dtype=int)
    settings=dict(trees=8,checkpoint_every=4,max_depth=4,min_samples_leaf=2,max_features=1.,fit_threads=1)
    kwargs=dict(settings=settings,seed=7,identity={'source':'fixture'},heartbeat=lambda **k:None)
    fit(x,q,d,pr,draws,directory=tmp_path/'full',**kwargs)
    fit(x,q,d,pr,draws,directory=tmp_path/'resume',stop_at=4,**kwargs)
    result=fit(x,q,d,pr,draws,directory=tmp_path/'resume',resume=True,**kwargs)
    assert result['complete'] and result['trees']==8
    a=joblib.load(tmp_path/'full/checkpoint.joblib');b=joblib.load(tmp_path/'resume/checkpoint.joblib')
    np.testing.assert_array_equal(predict(a['model'],x,pr),predict(b['model'],x,pr))
    with pytest.raises(ValueError,match='identity'):
        fit(x,q,d,pr,draws,directory=tmp_path/'resume',resume=True,
            **{**kwargs,'identity':{'source':'changed'}})
