import numpy as np
import pytest
from src.evaluation import m3w_fitting_prefix_audit as m


def arrays():
    h=np.zeros((3,8,2));h[:,:,0]=np.arange(8)
    b=np.zeros((3,12,2));b[:,:,0]=np.arange(1,13)
    p=b.copy();p[1,:,0]*=-1
    return h,p,b


def test_features_cannot_take_future_and_are_translation_invariant():
    h,p,b=arrays();x=m.features(h,p,b)
    np.testing.assert_array_equal(x,m.features(h+1000,p,b))
    assert x.shape==(3,len(m.NAMES)) and np.isfinite(x).all()
    assert m.direction_bin(x[:,5],x[:,6].astype(bool)).tolist()==[4,1,4]
    with pytest.raises(TypeError):m.features(h,p,b,target=b)


def test_prefix_labels_match_direct_available_label_reductions():
    _,p,b=arrays();y=b.copy();v=np.ones((3,12),bool);v[0,2:]=False;v[2]=False;y[~v]=np.nan
    t=m.labels(p,b,y,v)
    for j,k in enumerate(m.PREFIXES):
        for i in range(3):
            take=v[i,:k]
            if take.any():
                assert t['model'][i,j]==pytest.approx(np.linalg.norm(p[i,:k][take]-y[i,:k][take],axis=1).mean())
            else:assert np.isnan(t['model'][i,j]) and not t['known'][i,j]
    assert t['count'][0,-1]==2 and not t['complete'][0,-1]


def test_existing_basis_recovers_first_direction():
    x=np.zeros((3,120));x[:,20]=2;x[:,74]=[0,-4,-2];x[:,75]=[0,0,2]
    c,v=m.first_cos_from_existing120(x)
    np.testing.assert_array_equal(c,[1,-1,0]);assert v.all()


def test_unknowns_and_empty_mass_are_not_zero_error_labels():
    _,p,b=arrays();y=b.copy();v=np.zeros((3,12),bool);y[:]=np.nan
    t=m.labels(p,b,y,v);s=m.summarize(t,np.ones(3),np.zeros(3,bool),np.array([4,1,4]))
    assert all(r['rows']==3 and r['wholly_unknown_rows']==3 for r in s)
    assert all(r['known_gain_percent'] is None and r['completion_lower_percent'] is None for r in s)
    assert all(z['easy_harm_mass_share'] is None for r in s for z in r['direction_bins'])


def test_fixed_bins_preserve_invalid_direction_separately():
    c=np.array([-1.,0,.5,.9,1.,0.]);v=np.array([1,1,1,1,1,0],bool)
    assert m.direction_bin(c,v).tolist()==[1,2,3,4,4,0]
