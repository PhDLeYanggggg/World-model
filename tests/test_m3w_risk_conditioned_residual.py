import numpy as np
import pytest
from src.world_model import m3w_risk_conditioned_residual as m
from src.world_model import m3w_context_residual as old


def data():
    rng=np.random.default_rng(23); n=180
    x=rng.normal(size=(n,7)); sites=np.repeat(['a','b','c'],60)
    p=np.column_stack((np.ones(n),rng.uniform(.01,1,n),np.full(n,.2),np.zeros(n)))
    p[:,3]=p[:,1]*rng.uniform(.01,1,n)
    y=p.copy(); y[:,3]=p[:,1]*(p[:,3]/p[:,1]>.6)
    w=np.ones(n)/n; env=np.ones(n)
    return x,p,y,w,sites,env


def test_causal_ratios_scale_invariant_and_zero():
    x,p,y,w,s,e=data()
    np.testing.assert_allclose(m.risk_features(p,e),m.risk_features(p*9,e*9))
    p[0]=0; e[0]=0
    np.testing.assert_array_equal(m.risk_features(p,e)[0],[0,0])
    with pytest.raises(ValueError): m.risk_features(p,-e)
    p[1,3]=p[1,1]+1
    with pytest.raises(ValueError): m.risk_features(p,e)


@pytest.mark.parametrize('arm',m.ARMS)
def test_prediction_only_changes_easy_harm_and_replays(arm):
    x,p,y,w,s,e=data()
    a=m.fit(x,p,e,y,w,s,'d',arm=arm,variant='oof')
    b=m.fit(x,p,e,y,w,s,'d',arm=arm,variant='oof')
    assert a==b and not a['fitted_with_in_sample_base_predictions']
    out=m.predict(a,x,p,e)
    np.testing.assert_array_equal(out[:,:3],p[:,:3])
    assert (out[:,3]>=0).all() and (out[:,3]<=out[:,1]).all()
    assert (out[:,3]!=p[:,3]).any()
    # No target or future-label argument exists in inference.
    import inspect
    assert list(inspect.signature(m.predict).parameters)==['model','causal_context','prediction','envelope']


def test_unknown_exclusion_and_outer_guard():
    x,p,y,w,s,e=data(); y[0]=np.nan; w[0]=0
    m.fit(x,p,e,y,w,s,'d',arm='risk_context',variant='oof')
    with pytest.raises(ValueError): m.fit(x,p,e,y,w,s,'a',arm='risk_context',variant='oof')
    w[0]=1
    with pytest.raises(ValueError): m.fit(x,p,e,y,w,s,'d',arm='risk_context',variant='oof')


def test_old_columns_equal_and_risk_conditioning_not_label_input():
    x,p,y,w,s,e=data(); x[0,4]=np.nan
    full=m.features(x,p,e,'risk_context')
    np.testing.assert_array_equal(full[:,:7],x)
    out=m.fit(x,p,e,y,w,s,'d',arm='risk_context',variant='oof')
    control=old.fit(x,p,y,w,s,'d')['context_bias']
    assert out['cuts'][:7]==control['cuts'] and out['rms']==control['rms']
    changed=m.features(x,p*.8,e,'risk_context')
    assert not np.array_equal(full[:,7],changed[:,7])


def test_producer_projection_and_algebraic_noop():
    x,p,y,w,s,e=data(); q=p.copy(); q[:,3]*=.5
    a=old.fit(x,p,y,w,s,'d')['context_bias']; b=old.fit(x,q,y,w,s,'d')['context_bias']
    diag=m.producer_diagnostic(x,p,q,e,y,w,a,b)
    assert diag['coefficient_identity_max_error']<1e-8
    # An explicit prediction-difference offset only returns the old in-sample residual.
    np.testing.assert_allclose((y[:,3]-p[:,3])+(p[:,3]-q[:,3]),y[:,3]-q[:,3])


def test_inference_feature_rows_do_not_modify_fit_bins():
    x,p,y,w,s,e=data()
    a=m.fit(x,p,e,y,w,s,'d',arm='risk_context',variant='oof')
    original=repr(a)
    m.predict(a,np.full_like(x,1e6),p,e)
    assert repr(a)==original
