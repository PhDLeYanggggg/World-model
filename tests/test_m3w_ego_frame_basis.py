import numpy as np
import pytest
from src.world_model import m3w_ego_frame_basis as m


def packet():
    x=np.zeros((2,7,428),np.float32); x[...,24:32]=1
    h=x[...,:24].reshape(2,7,8,3); h[...,0]=np.arange(-7,1)/100; h[...,2]=np.arange(8)
    n=x[...,32:224].reshape(2,7,8,8,3); n[...,0]=.9; n[...,2]=np.arange(8)
    x[...,224:288]=1
    x[...,306:330]=np.tile(np.column_stack((np.arange(1,13)/100,np.zeros(12))).ravel(),(2,7,1))
    x[...,330:354]=x[...,306:330]*1.1
    return x,np.array([100.,100.])


def test_change_in_context_scale_cancels_in_ego_coordinates():
    x,s=packet(); a,r=m.coordinates(x,s)
    shifted=x.copy(); shifted[...,m.XY]/=2
    b,rr=m.coordinates(shifted,s*2)
    np.testing.assert_array_equal(a,b)
    np.testing.assert_allclose(rr,r/2)
    assert np.array_equal(a[...,2:24:3],x[...,2:24:3])
    assert np.array_equal(a[...,224:288],x[...,224:288])


def test_stationary_history_has_declared_one_image_unit_floor():
    x,s=packet(); x[...,:24].reshape(2,7,8,3)[...,:2]=0
    z,r=m.coordinates(x,s); np.testing.assert_allclose(r,.01)
    assert np.isfinite(z).all()


def test_chunking_keeps_all_features_exact_and_context_explicit():
    x,s=packet(); a,r=m.chunked(x,s,chunk=1); b,rr=m.chunked(x,s,chunk=2)
    assert a['history120'].shape==(2,7,120) and a['history121'].shape==(2,7,121)
    for k in m.ARMS:
        np.testing.assert_array_equal(a[k],b[k])
    np.testing.assert_allclose(a['history121'][...,120],np.log(r)[:,None]+np.zeros((2,7)))


def test_metadata_does_not_enter_features():
    x,s=packet(); a,_=m.chunked(x,s); x[...,380:]=2000
    b,_=m.chunked(x,s)
    for k in m.ARMS:
        np.testing.assert_array_equal(a[k],b[k])


def test_invalid_scales_or_action_dependent_history_rejected():
    x,s=packet()
    with pytest.raises(ValueError): m.coordinates(x,s*0)
    x[:,2,0]+=.1
    with pytest.raises(ValueError): m.coordinates(x,s)
