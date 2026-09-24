import numpy as np
import pytest
from src.evaluation.m3w_moving_zero_support import fixed_controls,neighbors,describe_neighbors


def test_controls_are_metadata_only_and_order_invariant():
    ids=np.arange(8); r=np.array(['r']*8); f=ids*12; t=ids.astype(str)
    a=fixed_controls(ids,r,f,t,3)
    np.testing.assert_array_equal(a,fixed_controls(ids[::-1],r,f,t,3))


def test_direct_distance_preserves_tiny_changes_on_large_offsets():
    x=np.array([[1e8,0],[1e8+1,0],[1e8,2]],float)
    r=neighbors(x,x[0],np.array([3,2,1]),block=1)
    np.testing.assert_array_equal(r['squared_distance'],[0,.5,2])
    np.testing.assert_array_equal(r['exact'],[True,False,False])


def test_ties_are_stable_by_source_id():
    r=neighbors(np.array([[1.],[-1.],[0.]]),np.array([0.]),np.array([20,10,30]))
    np.testing.assert_array_equal(r['order'],[2,1,0])


def test_labels_cannot_change_neighbors():
    x=np.arange(8.)[:,None]; r=neighbors(x,np.array([1.]),np.arange(8))
    before=r['order'].copy()
    for event in (np.ones(8,bool),np.zeros(8,bool)):
        describe_neighbors(r,event,np.arange(8.),np.zeros(8),np.arange(8))
    np.testing.assert_array_equal(r['order'],before)


def test_exact_alias_and_repeated_tracks_are_explicit():
    r=neighbors(np.array([[0.],[0.],[1.]]),np.array([0.]),np.arange(3))
    d=describe_neighbors(r,np.array([True,False,False]),np.array([0.,1.,0.]),np.array([1.,0.,1.]),np.array(['a','a','b']),ks=(2,))
    assert d['exact_zero_rows']==d['exact_nonzero_rows']==1
    assert d['nearest_zero_rank']==1 and d['neighborhoods']['2']['tracks']==1


def test_absent_event_is_not_zero_distance():
    r=neighbors(np.zeros((2,1)),np.zeros(1),np.arange(2))
    d=describe_neighbors(r,np.zeros(2,bool),np.zeros(2),np.zeros(2),np.arange(2))
    assert d['nearest_zero_rank'] is None and d['nearest_zero_distance'] is None


@pytest.mark.parametrize('x,q,ids',[(np.array([[np.nan]]),np.zeros(1),np.arange(1)),
    (np.zeros((2,1)),np.zeros(1),np.zeros(2)),(np.zeros((2,1)),np.zeros(2),np.arange(2))])
def test_bad_features_rejected(x,q,ids):
    with pytest.raises(ValueError): neighbors(x,q,ids)
