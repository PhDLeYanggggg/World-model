import numpy as np

from scripts.run_m3w_wepdtof_external_readout import fixed,augment_metrics
from src.world_model import m3w_wepdtof_matched_policy as m


def test_both_policy_families_use_identical_query_count_and_peak():
    h = np.zeros((3,8,2))
    h[:,:,0] = np.arange(8)[None,:]
    h[:, :,1] = np.arange(3)[:,None]
    cv,ols = m.analytic(h)
    p = cv+2.
    paths = m.bank(cv,ols+1.,p,np.zeros(3,dtype=int))
    legal = (paths!=cv[:,None]).any((2,3))
    est = np.zeros((3,2,6))
    est[:,:,:2] = 2.
    est[:,:,4:6] = 1.
    q = np.zeros(3,dtype=int)
    a = m.apply(cv,paths,m.choose(est,legal,q,h,cv,paths),q)
    b = fixed(est,legal,q,h,cv,paths)
    all_peaks = [np.linalg.norm(x-cv,axis=-1).max(1) for x in (*a.values(),*b.values())]
    assert all((v>0).sum()==2 for v in all_peaks)
    np.testing.assert_allclose([v.sum() for v in all_peaks],all_peaks[0].sum())


def test_empty_matched_controls_are_not_estimates():
    cv = np.zeros((2,12,2))
    q = np.zeros(2,dtype=int)
    y = np.ones_like(cv)
    v = np.ones((2,12),dtype=bool)
    names = ('cv','ols4','joint','independent','uniform','no_pair',
             'fixed_joint','fixed_independent','fixed_uniform','fixed_no_pair')
    p = {k:cv for k in names}
    z = augment_metrics(p,y,v,q,'clip',(2.,4.))
    assert all(x==[None,None] for x in z['comparisons'].values())
    assert not z['arms']['joint']['risk']['supported']
    assert z['arms']['joint']['risk']['completion_easy_risk_upper'] is None
