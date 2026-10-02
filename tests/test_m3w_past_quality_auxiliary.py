import inspect
import io
import numpy as np
import pytest
from sklearn.ensemble import ExtraTreesRegressor
from src.world_model import m3w_past_quality_auxiliary as api


def fixture():
    rng=np.random.default_rng(71); n=160
    x=rng.normal(size=(n,3)).astype(np.float32); q=rng.uniform(size=(n,7)); env=np.full(n,4.)
    harm=np.where(q[:,0]>.5,q[:,0],0.); benefit=np.where(q[:,0]<=.5,1-q[:,0],0.)
    ref=5+q[:,1]; easy=q[:,2]>.5
    y=np.column_stack((benefit,harm,ref,ref*easy,harm*easy));y[-1]=np.nan
    site=np.full(n,'source'); rec=np.repeat(np.arange(4),40); frame=np.arange(n)
    pr=api.forest.core.preprocess(x,env,y,site,rec,frame,training_site='source')
    z,_=api.forest.causal_inputs(x,env,pr);known=np.isfinite(y).all(1)
    w,_=api.forest.core.weights(site,rec,frame,known);t=api.forest.transformed_targets(y[known],pr)
    model=ExtraTreesRegressor(n_estimators=3,min_samples_leaf=12,random_state=17,n_jobs=1).fit(z[known],t,sample_weight=w[known])
    inputs=dict(features=api.forest.fingerprint(z[known]),targets=api.forest.fingerprint(t),weights=api.forest.fingerprint(w[known]),known_mask=api.forest.fingerprint(known))
    return dict(model=model,preprocess=pr,input_hashes=inputs),x,env,y,q,site,rec,frame


def test_future_proxy_rejected():
    fields={k:np.ones(5) for k in api.FEATURES}; fields['future_mean_detector_confidence']=np.ones(5)
    with pytest.raises(ValueError): api.quality_matrix(fields)


def test_within_recording_permutation_is_reproducible_and_local():
    r=np.repeat(np.arange(4),40); ix=api.within_recording_permutation(r,17)
    np.testing.assert_array_equal(r[ix],r)
    np.testing.assert_array_equal(ix,api.within_recording_permutation(r,17))
    np.testing.assert_array_equal(np.sort(ix),np.arange(len(r)))
    assert not np.array_equal(ix,np.arange(len(r)))


def test_weighted_ridge_preserves_leaf_training_mean():
    q=np.zeros((6,7));q[:,0]=[0,1,2,3,4,5];y=np.tile(q[:,0,None],(1,5))
    w=np.array([1.,2.,4.,1.,2.,4.]);leaves=np.repeat([5,9],3)
    f=api.leaf_regression(leaves,q,y,w,1.)
    assert f['loss_after']<f['loss_before']
    assert f['max_centering_error']<1e-10
    for j,node in enumerate(f['nodes']):
        at=leaves==node
        z=q[at]-f['means'][j];t=y[at]-f['original'][j];wt=w[at]/w[at].sum()
        scalar=np.linalg.solve(z.T@(z*wt[:,None])+np.eye(7),z.T@(t*wt[:,None]))
        np.testing.assert_allclose(scalar,f['coef'][j],atol=1e-12)


def test_fit_and_prediction_replay_exact_with_known_train_only():
    state,x,e,y,q,s,r,f=fixture()
    a=api.fit(state,x,e,y,q,s,r,f,arm='quality',seed=17)
    b=api.fit(state,x,e,y,q,s,r,f,arm='quality',seed=17)
    for k in a:
        if isinstance(a[k],np.ndarray): np.testing.assert_array_equal(a[k],b[k])
        else: assert a[k]==b[k]
    old,new,support,diag=api.predict(state,a,x,e,q)
    np.testing.assert_array_equal(old,api.forest.predict(state,x,e)[0])
    assert a['unknown_train_rows']==1 and diag['correction_RMS']>0
    assert (new>=0).all() and (new[:,:2].sum(1)<=e+1e-10).all()
    assert (new[:,3]<=new[:,2]).all() and (new[:,4]<=new[:,1]).all()


def test_zero_quality_is_exact_unchanged_control():
    state,x,e,y,q,s,r,f=fixture();q[:]=0
    a=api.fit(state,x,e,y,q,s,r,f,arm='quality',seed=17)
    np.testing.assert_array_equal(a['coef'],0)
    old,new,_,_=api.predict(state,a,x,e,q)
    np.testing.assert_array_equal(old,new)


def test_modified_training_target_is_rejected():
    state,x,e,y,q,s,r,f=fixture();y[0,2]+=1
    with pytest.raises(AssertionError): api.fit(state,x,e,y,q,s,r,f,arm='quality',seed=17)


def test_predict_has_no_label_mask_or_future_input():
    assert list(inspect.signature(api.predict).parameters)==['state','fit','x','env','q']


def test_unknown_train_features_do_not_fit_standardization():
    state,x,e,y,q,s,r,f=fixture()
    a=api.fit(state,x,e,y,q,s,r,f,arm='quality',seed=17); q[-1]=1e8
    b=api.fit(state,x,e,y,q,s,r,f,arm='quality',seed=17)
    for key in ('quality_mean','quality_std','coef','means'):np.testing.assert_array_equal(a[key],b[key])


@pytest.mark.parametrize('penalty',[0.,-1.])
def test_no_unregularized_singular_leaf_fit(penalty):
    with pytest.raises(ValueError): api.leaf_regression(np.zeros(5,int),np.zeros((5,7)),np.zeros((5,5)),np.ones(5),penalty)


def test_serialized_arrays_reproduce_inference():
    state,x,e,y,q,s,r,f=fixture();a=api.fit(state,x,e,y,q,s,r,f,arm='placebo',seed=17)
    stream=io.BytesIO();np.savez_compressed(stream,**{k:v for k,v in a.items() if isinstance(v,np.ndarray)})
    with np.load(io.BytesIO(stream.getvalue()),allow_pickle=False) as z: loaded={k:z[k] for k in z.files}
    for v,w in zip(api.predict(state,a,x,e,q)[:3],api.predict(state,loaded,x,e,q)[:3]):np.testing.assert_array_equal(v,w)


def test_matched_comparisons_preserve_query_counts():
    state,x,e,y,q,s,r,f=fixture();pred={}
    for arm in api.ARMS:
        a=api.fit(state,x,e,y,q,s,r,f,arm=arm,seed=17)
        old,p,support,_=api.predict(state,a,x,e,q);pred[arm]=p
    pred['original']=old
    result,actions=api.evaluate(state,pred,y,e,np.ones(len(x),bool),support,s,r,f//4,np.arange(len(x)))
    for other in ('original','placebo'):
        for frame in np.unique(f//4):
            assert actions[other+'_matched_quality'][f//4==frame].sum()==actions['quality_matched_'+other][f//4==frame].sum()
    assert result['policies']['quality']['unknown_rows']==1
