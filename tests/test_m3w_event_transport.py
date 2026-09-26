import copy
import numpy as np
import pytest
from src.world_model import m3w_event_transport as m
from src.world_model import m3w_context_residual as residual


def fixture():
    rng = np.random.default_rng(17); n = 120
    x = rng.normal(size=(n, 7)); sites = np.repeat(['a', 'b', 'c'], 40)
    cv = np.linspace(.1, 3, n); raw = np.column_stack((cv, cv*.2))
    p = np.tile([3., 2., 1., .2], (n, 1)); w = np.ones(n)/n
    inner = m.common_event_target(raw, cv, sites, 'd', .5)
    common = m.common_event_target(raw, cv, sites, 'd', 1.)
    return x, sites, raw, cv, p, w, inner, common


def test_common_labels_support_and_no_predictor_mutation():
    x,s,raw,cv,p,w,a,b = fixture(); before = p.copy()
    assert (a[:,3] != b[:,3]).any()
    np.testing.assert_array_equal(a[:,:2], b[:,:2])
    old = residual.fit(x,p,a,w,s,'d'); new = residual.fit(x,p,b,w,s,'d')
    for arm in residual.ARMS:
        q = m.transport_accounting(old[arm],new[arm],x,a,b,w,x,p)
        assert q['coefficient_identity_max_error'] < 1e-8
        out = residual.predict(new[arm],x,p)
        np.testing.assert_array_equal(out[:,:3],p[:,:3])
        assert (out[:,3] >= 0).all() and (out[:,3] <= out[:,1]).all()
    np.testing.assert_array_equal(p,before)


def test_outer_rejected_and_bad_cut():
    _,s,raw,cv,*_ = fixture()
    with pytest.raises(ValueError): m.common_event_target(raw,cv,s,'a',1.)
    for cut in (0,-1,np.nan,np.inf):
        with pytest.raises(ValueError): m.common_event_target(raw,cv,s,'d',cut)


def test_unknown_labels_zero_weight_and_identity():
    x,s,raw,cv,p,w,a,b = fixture(); raw[0] = np.nan; cv[0] = np.nan; w[0] = 0
    a = m.common_event_target(raw,cv,s,'d',.5); b = m.common_event_target(raw,cv,s,'d',1.)
    assert np.isnan(a[0]).all() and np.isnan(b[0]).all()
    old = residual.fit(x,p,a,w,s,'d'); new = residual.fit(x,p,b,w,s,'d')
    m.transport_accounting(old['context_bias'],new['context_bias'],x,a,b,w,x,p)
    w[0] = 1
    with pytest.raises(ValueError): residual.fit(x,p,b,w,s,'d')


def test_clipping_is_not_linear_and_no_held_label_argument():
    x,s,raw,cv,p,w,a,b = fixture()
    old = residual.fit(x,p,a,w,s,'d')['global_bias']; new = copy.deepcopy(old)
    old['coefficients'] = [10/old['rms']]; new['coefficients'] = [12/new['rms']]
    assert np.allclose(residual.predict(old,x,p),residual.predict(new,x,p))
    assert not np.allclose(m.raw_shift(old,x),m.raw_shift(new,x))
    import inspect
    assert 'held_labels' not in inspect.signature(m.transport_accounting).parameters


def test_same_cut_zero_transport_and_permutation():
    x,s,raw,cv,p,w,a,b = fixture()
    model = residual.fit(x,p,b,w,s,'d')
    q = m.transport_accounting(model['context_bias'],model['context_bias'],x,b,b,w,x,p)
    assert q['held_preclip_event_RMS'] == q['held_postclip_change_RMS'] == 0
    order = np.random.default_rng(2).permutation(len(s))
    perm = m.common_event_target(raw[order],cv[order],s[order],'d',1.)
    np.testing.assert_array_equal(perm,b[order])
