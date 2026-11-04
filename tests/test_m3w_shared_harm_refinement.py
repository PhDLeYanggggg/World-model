import numpy as np
import pytest
from src.world_model import m3w_shared_harm_refinement as api
from tests.test_m3w_harm_refinement import fixture


def test_same435width_and_original387prefix_with_shared_semantics():
    data,p=fixture();c,pr=data['causal'],data['preprocess']
    old=api.previous.features(c,p,pr)
    a,b=(api.features(c,p,pr,arm)for arm in api.ARMS)
    assert a.shape==b.shape==(len(old),435)
    np.testing.assert_array_equal(a[:,:387],old);np.testing.assert_array_equal(b[:,:387],old)
    assert not a[:,387:].any()
    rows,actions,_=api.previous.layout(c)
    np.testing.assert_allclose(b[:,387:],np.clip((c['x'][rows,actions,380:]-pr['mean'][380:])/
        pr['std'][380:],-pr['clip'],pr['clip']),rtol=0,atol=1e-6)
    assert b[:,387:].any()
    with pytest.raises(ValueError):api.features(dict(c,future_endpoint=np.zeros((len(p),2))),p,pr,'shared_motion')


@pytest.mark.parametrize('arm',api.ARMS)
def test_actual435trees_exact_resume_and_frozen_reference_invariants(tmp_path,arm):
    data,p=fixture();c=data['causal']
    settings=dict(rounds=4,learning_rate=.5,max_depth=3,max_features=64,min_samples_leaf=1,region_multiplier=4.)
    kw=dict(arm=arm,seed=17,settings=settings,identity=dict(synthetic=True))
    a=api.fit(data,p,path=tmp_path/'direct.pt.gz',**kw)
    api.fit(data,p,path=tmp_path/'resumed.pt.gz',stop_at=2,**kw)
    b=api.fit(data,p,path=tmp_path/'resumed.pt.gz',resume=True,**kw);api.exact_refit(a,b)
    out=api.predict(a,c,p)
    np.testing.assert_array_equal(out,api.predict(b,c,p));np.testing.assert_array_equal(out[:,:,[2,3]],p[:,:,[2,3]])
    assert a['feature_dimension']==435 and all(t.n_features_in_==435 for t in a['trees'])
    assert (out[:,:,1]>=p[:,:,1]-1e-6).all()and (out[:,:,4]>=p[:,:,4]-1e-6).all()
    if arm=='local_zero':
        x=c['x'].copy();x[:,:,380:]+=100
        np.testing.assert_array_equal(out,api.predict(a,dict(c,x=x),p))
    with pytest.raises(ValueError):api.predict(a,dict(c,central_velocity=np.zeros((len(p),2))),p)


def test_same_frozen_costs_targets_and_different_effective_context(tmp_path):
    data,p=fixture();settings=dict(rounds=1,learning_rate=.5,max_depth=3,max_features=64,min_samples_leaf=1,region_multiplier=4.)
    states=[api.fit(data,p,arm=arm,seed=17,settings=settings,identity={},path=tmp_path/(arm+'.pt.gz'))for arm in api.ARMS]
    for key in ('frozen_prediction_sha256','truth_sha256','fit_ids_sha256','preprocess','raw_feature_sha256'):
        api.base.exact(states[0][key],states[1][key])
    assert states[0]['initial_effective_feature_sha256']!=states[1]['initial_effective_feature_sha256']
    with pytest.raises(ValueError):api.predict(dict(states[0],feature_dimension=387),data['causal'],p)
