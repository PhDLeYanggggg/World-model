import numpy as np
import pytest
import torch

from scripts.audit_m3w_cv_exchange_peak_envelope import query_upper
from scripts.audit_m3w_cv_exchange_record_pool import pool
from scripts.train_m3w_cv_matched_exchange import upper
from src.world_model import m3w_cv_matched_exchange as model


def x(values):
    out = np.zeros((len(values),12,2))
    out[:,:,0] = np.asarray(values)[:,None]
    return out


def test_one_exchange_keeps_count_and_peak_in_every_query():
    c,o,u = x([0,0,0,0]),x([0,3,0,2]),x([2,0,1,0])
    groups = [np.array([0,1]),np.array([2,3])]
    i,j,q = model.shortlist(c,o,u,np.array([1,0,1,0],bool),groups)
    p = model.incoming_path(c[j],o[j],model.peak(u[i]-c[i]))
    result = model.apply(c,u,i,j,p,model.decode(np.array([1.,-1.]),q,0.))
    np.testing.assert_allclose(result,x([0,2,1,0]))
    for g in groups:
        a,b = (model.peak(z[g]-c[g]) for z in (u,result))
        assert (a>0).sum()==(b>0).sum()
        assert a.sum()==pytest.approx(b.sum())
    with pytest.raises(ValueError,match='amplify'):
        model.incoming_path(c[j],o[j],np.array([4.,3.]))


def test_one_exchange_envelope_matches_brute_force():
    c,o,u,y = x([0,0]),x([0,3]),x([2,0]),x([0,3])
    q,v = np.zeros(2,int),np.ones((2,12),bool)
    i,j,p = np.array([0]),np.array([1]),x([2])
    ans = upper(c,o,u,y,v,q,i,j,p)
    assert ans[0]==pytest.approx(100.*(3-1)/3)
    assert ans[1]==pytest.approx(100.*(0-1)/3)
    np.testing.assert_allclose(upper(c,o,u,y,v,q,i[:0],j[:0],p[:0]),[-200/3,-500/3])


def test_swap_targets_stay_finite_when_future_is_missing():
    c,o,p,y = x([0]),x([3]),x([1]),x([np.nan])
    z = model.targets(c,o,p,y,np.zeros((1,12),bool),1.)
    assert np.isfinite(z).all()
    np.testing.assert_allclose(z[0],[-1.,1.,1.,1.,0.])
    assert torch.isfinite(model.objective(torch.nn.Linear(2,5),torch.ones(1,2),torch.from_numpy(z).float()))


def test_peak_envelope_is_valid_for_every_feasible_allocation():
    dist = np.array([[1.,4.],[2.,2.],[3.,3.]])
    bound = query_upper(dist,np.array([2.,0.,0.]),1,(1.,None))
    for i in range(len(dist)):
        for r in (0.,.5,2.):
            assert np.minimum(dist[i],r).sum()/3<=bound[0,0]+1e-10
    assert bound[0,0]==pytest.approx(4/3)
    assert bound[1,0]==pytest.approx(2.)
    inactive = query_upper(dist,np.zeros(3),0,(1.,None))
    np.testing.assert_allclose(inactive,[[0.,0.],[0.,5.]])
    pooled = pool(dist,np.full(3,1/3),np.array([2.,0.,0.]),(1.,None))
    assert pooled[0]>=4/3
    np.testing.assert_allclose(pooled[1],2.)
