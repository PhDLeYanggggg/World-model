import numpy as np

from scripts.audit_m3w_opendd_amplitude_spread import spread
from src.evaluation.m3w_causal_amplitude import (
    LEVELS, choose, geometry, labels, matched, prefix_features, reconstruct, rollout,
)
from src.evaluation.m3w_opendd_prefix_risk import compression


def test_geometry_reproduces_joint_pair_compression():
    rng = np.random.default_rng(21)
    origin = rng.normal(size=(4,2))
    cv = rng.normal(size=(4,12,2))
    bank = rng.normal(size=(4,3,12,2))
    one,two = geometry(origin,cv,bank)
    actual = cv.copy()
    actual[0],actual[3] = bank[0,1],bank[3,2]
    np.testing.assert_allclose(one[0,1]+one[3,2]+two[0,3,1,2],
                               compression(origin,cv,actual,(0,3)))


def test_factorized_labels_recover_exact_direct_values_including_missing():
    rng = np.random.default_rng(10)
    cv = rng.normal(size=(4,12,2))
    inc = cv+rng.normal(size=cv.shape)
    y = cv+.2
    y[0] = np.nan
    y[1,2:9] = np.nan
    valid = np.isfinite(y).all(2)
    scale = np.array([.1,1.,2.,5.])
    direct,factor = labels(cv,y,valid,inc,.5,scale)
    a = reconstruct(direct,cv,inc,scale,False)
    b = reconstruct(factor,cv,inc,scale,True)
    np.testing.assert_allclose(a,b,atol=1e-13)
    assert (b[0,:,0]<0).all() and (b[0,:,1]>0).all()


def test_only_low_amplitude_fits_budget_and_matching_can_reassign():
    moments = np.zeros((3,3,5))
    moments[:,:,:] = [1.,.04,.04,1.,1.]
    moments[0,0] = [2.,.01,.01,1.,1.]
    edits,_ = choose(moments,np.ones(3,dtype=bool),np.zeros(4))
    assert edits==((0,0),)
    cv = np.zeros((3,12,2))
    inc = cv+np.array([.1,.6,.8])[:,None,None]
    r = np.linalg.norm(inc-cv,axis=-1).max(1)
    utility = np.array([[1.,1.,1.],[2.,4.,8.],[3.,6.,12.]])
    independent,additive = matched(np.zeros((3,2)),cv,inc,r,[r[0]*.5,r[1]],utility)
    for arm in (independent,additive):
        assert len(arm)==2
        radii = [np.linalg.norm(p-cv[i],axis=-1).max() for i,p in arm.items()]
        np.testing.assert_allclose(sorted(radii),sorted([r[0]*.5,r[1]]))


def test_prefix_and_rollout_are_invariant_to_future_query_changes():
    rng = np.random.default_rng(9)
    h = rng.normal(size=(6,8,2))
    inc = rng.normal(size=(6,12,2))
    cv = np.zeros_like(inc)
    rec = np.array(['a']*6)
    offsets = [0,3,6]
    z = prefix_features(h,offsets,rec,inc)
    h[3:] += 10.
    inc[3:] -= 9.
    np.testing.assert_array_equal(z[:3],prefix_features(h,offsets,rec,inc)[:3])
    m = np.broadcast_to([2.,.01,.01,1.,1.],(6,len(LEVELS),5)).copy()
    x = {k:m.copy() for k in ('direct','factor_local','factor_prefix')}
    p = rollout(cv,inc,h[:,-1],offsets,rec,x)
    for a in x:
        x[a][3:] *= 0.
    q = rollout(cv,inc,h[:,-1],offsets,rec,x)
    for a in p:
        np.testing.assert_array_equal(p[a][:3],q[a][:3])


def test_spread_matches_query_movement_but_can_use_more_recipients():
    cv = np.zeros((4,12,2))
    inc = cv+np.array([1.,3.,2.,4.])[:,None,None]
    joint = cv.copy()
    joint[1] = inc[1]
    p = spread(cv,inc,joint,np.array([0,2,4]))
    np.testing.assert_allclose(p[:2],inc[:2]*.75)
    np.testing.assert_array_equal(p[2:],cv[2:])
