import numpy as np

from src.evaluation.m3w_exchange_context import feature_arms, relative


def test_relative_geometry_is_translation_invariant_and_masks_missing_velocity():
    h = np.zeros((1,8,2))
    n = np.full((1,2,8,2),np.nan)
    n[0,0,-2:] = [[3.,0.],[2.,0.]]
    n[0,1,-1] = [3.,4.]
    R = np.eye(2)[None]
    f = relative(h,n,R,np.ones(1)).reshape(2,11)
    np.testing.assert_array_equal(f[:,:2],[[1.,1.],[1.,0.]])
    np.testing.assert_allclose(f[:,9],[2./12.,0.])
    np.testing.assert_allclose(relative(h+42.,n+42.,R,np.ones(1)).reshape(2,11),f)
    missing = relative(h,np.full_like(n,np.nan),R,np.ones(1))
    np.testing.assert_array_equal(missing,0.)


def test_same_width_and_same_query_negative_control():
    x = np.arange(6).reshape(6,1)
    f = feature_arms(x,x+10,x+100,[0,2,6])
    assert {v.shape for v in f.values()}=={(6,3)}
    np.testing.assert_array_equal(f['own'][:,1:],0.)
    np.testing.assert_array_equal(f['pedestrian'][:,2],0.)
    np.testing.assert_array_equal(f['shuffled_traffic'][:,:2],f['traffic'][:,:2])
    np.testing.assert_array_equal(f['shuffled_traffic'][:,2],[101,100,105,102,103,104])
