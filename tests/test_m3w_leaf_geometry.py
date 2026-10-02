import numpy as np
import pytest
from sklearn.ensemble import ExtraTreesRegressor

from src.world_model import m3w_leaf_geometry as api


def test_scale_mixing_changes_fraction_not_reference():
    y = np.array([[0., 1., 5., 5., 1.], [0., 5., 8., 8., 5.]])
    a = api.aggregate_leaves(np.zeros(2,int), y, np.array([1., 10.]), np.array([.5,.5]), 1)
    assert a['original'][0, 1] == 3.
    assert a['refit'][0, 1] == .75
    assert a['env_mean'][0] == 5.5
    assert a['env_std'][0] == 4.5


def test_fraction_bounds_and_zero_envelope():
    f, n = api.fractions(np.array([[0.,0.,5.,0.,0.], [3.,0.,4.,4.,0.]]), np.array([0.,4.]))
    np.testing.assert_array_equal(f, [[0.,0.,0.],[.75,0.,0.]])
    assert n == 0


@pytest.mark.parametrize('y,e', [([[0.,1.,2.,2.,1.]], [0.]), ([[0.,2.,3.,3.,2.]], [1.]),
                                ([[np.nan]*5], [1.]), ([[0.,1.,2.,2.,2.]], [1.])])
def test_invalid_supervision_rejected(y,e):
    with pytest.raises(ValueError):
        api.fractions(np.array(y), np.array(e))


def fixture():
    rng = np.random.default_rng(17)
    x = rng.normal(size=(100,3)).astype(np.float32)
    e = rng.uniform(1,10,len(x))
    r = rng.uniform(10,20,len(x))
    h = np.where(x[:,0] > 0, e*.3, 0.)
    b = np.where(x[:,0] <= 0, e*.2, 0.)
    y = np.column_stack((b,h,r,r,h))
    y[-1] = np.nan
    sites,rec,frames = np.full(len(x),'source'),np.full(len(x),'r'),np.arange(len(x))
    pr = api.forest.core.preprocess(x,e,y,sites,rec,frames,training_site='source')
    z,_ = api.forest.causal_inputs(x,e,pr)
    known = np.isfinite(y).all(1)
    w,_ = api.forest.core.weights(sites,rec,frames,known)
    t = api.forest.transformed_targets(y[known],pr)
    model = ExtraTreesRegressor(n_estimators=3,min_samples_leaf=5,random_state=17,n_jobs=1).fit(z[known],t,sample_weight=w[known])
    inputs = dict(features=api.forest.fingerprint(z[known]),targets=api.forest.fingerprint(t),
        weights=api.forest.fingerprint(w[known]),known_mask=api.forest.fingerprint(known))
    state = dict(model=model,preprocess=pr,input_hashes=inputs)
    return state,x,e,y,sites,rec,frames


def test_refit_reconstructs_original_and_freezes_reference():
    state,x,e,y,s,r,f = fixture()
    fit = api.fit(state,x,e,y,s,r,f)
    old,new,mask,diag = api.predict(state,fit,x,e)
    np.testing.assert_array_equal(old[:,2:4],new[:,2:4])
    assert (new[:,:2].sum(1) <= e+1e-10).all()
    assert (new[:,4] <= new[:,1]+1e-10).all()
    assert fit['unknown_training_rows'] == 1
    replay = api.fit(state,x,e,y,s,r,f)
    np.testing.assert_array_equal(fit['values'],replay['values'])
    assert (diag['mean_leaf_envelope_std'] >= 0).all()


def test_modified_training_membership_rejected():
    state,x,e,y,s,r,f = fixture()
    y = y.copy(); y[0,2] += 1
    with pytest.raises(AssertionError):
        api.fit(state,x,e,y,s,r,f)


def test_inference_does_not_accept_targets_or_availability():
    import inspect
    assert list(inspect.signature(api.predict).parameters) == ['state','fitted','x','env']


def test_duplicate_leaf_weighting_not_row_count():
    y = np.array([[0.,1.,5.,5.,1.],[0.,5.,8.,8.,5.]])
    a = api.aggregate_leaves(np.zeros(2,int),y,np.array([1.,10.]),np.array([.9,.1]),1)
    np.testing.assert_allclose(a['refit'][0], [0.,.95,.95])


def test_checkpoint_roundtrip_predicts_identically():
    import io
    from scripts.run_m3w_leaf_geometry import serialize
    state,x,e,y,s,r,f = fixture()
    fitted = api.fit(state,x,e,y,s,r,f)
    payload = serialize(fitted,dict(group='synthetic'))
    assert payload == serialize(fitted,dict(group='synthetic'))
    with np.load(io.BytesIO(payload),allow_pickle=False) as z:
        loaded = {k:z[k] for k in ('values','offsets','stats')}
    for a,b in zip(api.predict(state,fitted,x,e)[:3],api.predict(state,loaded,x,e)[:3]):
        np.testing.assert_array_equal(a,b)


def test_exact_matched_query_counts_in_evaluation():
    state,x,e,y,s,r,f = fixture()
    fit = api.fit(state,x,e,y,s,r,f)
    old,new,support,diag = api.predict(state,fit,x,e)
    result,act = api.evaluate(state,old,new,y,e,np.ones(len(x),bool),support,s,r,f,np.arange(len(x)),diag)
    np.testing.assert_array_equal(act['original_matched'],act['refit_matched'])
    assert result['policies']['original']['unknown_rows']==1
