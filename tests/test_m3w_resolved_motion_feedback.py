import numpy as np

from src.world_model import m3w_resolved_motion_feedback as m


def test_endpoint_feedback_is_mature_and_matches_track_id():
    times = np.array([0,0,11,12,12,24]); people = np.array([1,2,1,2,1,2])
    origin = np.array([[0,0],[2,0],[11,0],[14,0],[12,0],[26,0]],float)
    floor = np.zeros((6,12,2)); p = floor[:,None].copy(); p[0,0,-1,0] = 12.
    g = [np.array([0,1]),np.array([2]),np.array([3,4]),np.array([5])]
    z, receipt = m.features(g,times,people,origin,np.ones(6),floor,p)
    assert receipt['exact_maturity_matches'] == 3
    assert z[2,0,0] == 0. and z[4,0,0] == 1. and z[4,0,3] > 0.
    assert z[3,0,3] == 0. and z[3,0,m.OWN] > 0.


def test_future_suffix_changes_do_not_change_earlier_features():
    rng = np.random.default_rng(17)
    t = np.repeat([0,12,24],2); agents = np.tile([1,2],3); g = [np.arange(k,k+2) for k in (0,2,4)]
    origin = rng.normal(size=(6,2)); floor = rng.normal(size=(6,12,2)); p = rng.normal(size=(6,2,12,2))
    z,_ = m.features(g,t,agents,origin,np.ones(6),floor,p)
    origin[4:] += 1000; p[4:] *= -7
    new,_ = m.features(g,t,agents,origin,np.ones(6),floor,p)
    np.testing.assert_array_equal(z[:4],new[:4])


def test_matched_ablations_mask_only_feedback_columns():
    x = np.ones((3,2,4)); f = np.ones((3,2,m.OWN+m.SCENE))
    a,b,c = [m.augment(x,f,k) for k in ('static','own','own_scene')]
    assert a.shape == b.shape == c.shape
    np.testing.assert_array_equal(a[...,:4],x)
    assert a[...,4:].sum()==0 and b[...,4+m.OWN:].sum()==0 and c.min()==1
