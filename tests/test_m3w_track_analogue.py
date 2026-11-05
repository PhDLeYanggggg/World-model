import numpy as np
import pytest
from src.evaluation import m3w_track_analogue as api


def data():
    rng=np.random.default_rng(93); n=200
    x=rng.normal(size=(n,4)); y=np.column_stack((x[:,0]-.5,x[:,1]*.2))
    sites=np.array(['fit']*160+['held']*40)
    agents=np.r_[np.repeat(np.arange(40),4),np.arange(40)]
    return x,y,np.ones(n),sites,agents,np.arange(n)


def test_distinct_identities_and_source_exclusion():
    x,y,w,s,a,ids=data(); bank=api.fit_bank(x,y,w,s,a,ids,'held')
    r=api.neighbors(bank,x[s=='held'],s[s=='held'])
    assert (s[r['donor_positions']]!='held').all()
    assert all(len(set(v.tolist()))==32 for v in r['identity_indices'])
    assert all(len(set(a[v].tolist()))==32 for v in r['donor_positions'])


def test_future_held_labels_cannot_change_model_or_predictions():
    x,y,w,s,a,ids=data(); b=api.fit_bank(x,y,w,s,a,ids,'held')
    changed=y.copy(); changed[s=='held']=1e9
    c=api.fit_bank(x,changed,w,s,a,ids,'held')
    for k in b: np.testing.assert_array_equal(b[k],c[k])
    r=api.neighbors(b,x[s=='held'],s[s=='held']); q=api.neighbors(c,x[s=='held'],s[s=='held'])
    for k in r: np.testing.assert_array_equal(r[k],q[k])


def test_chunk_boundaries_reproduce_exactly():
    x,y,w,s,a,ids=data(); b=api.fit_bank(x,y,w,s,a,ids,'held'); q=x[s=='held']; ss=s[s=='held']
    r=api.neighbors(b,q,ss)
    chunks=[api.neighbors(b,q[i:i+7],ss[i:i+7]) for i in range(0,len(q),7)]
    for k in r: np.testing.assert_array_equal(r[k],np.concatenate([c[k] for c in chunks]))


def test_unsupported_bank_does_not_claim_fallback_safety():
    x,y,w,s,a,ids=data(); a[:160]=0
    b=api.fit_bank(x,y,w,s,a,ids,'held')
    with pytest.raises(ValueError,match='not estimable'): api.neighbors(b,x[s=='held'],s[s=='held'])


def test_wrong_source_and_nan_queries_fail():
    x,y,w,s,a,ids=data(); b=api.fit_bank(x,y,w,s,a,ids,'held')
    with pytest.raises(ValueError): api.neighbors(b,x[:2],s[:2])
    q=x[-2:].copy(); q[0,0]=np.nan
    with pytest.raises(ValueError): api.neighbors(b,q,s[-2:])


def test_permutation_is_deterministic_and_label_free():
    ids=np.arange(40)
    p=api.permutation(ids,'s'); np.testing.assert_array_equal(np.sort(p),ids)
    np.testing.assert_array_equal(api.permutation(ids,'s'),p)
    assert not np.array_equal(p,ids)
