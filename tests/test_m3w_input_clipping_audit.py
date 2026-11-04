import numpy as np
import pytest
from src.evaluation.m3w_input_clipping_audit import profile


def test_clipping_uses_original_scale_and_does_not_drop_unknown_rows():
    x = np.zeros((4,20),np.float32)
    x[1,0] = 9
    x[2,:6] = -10
    x[3] = 20
    observed = np.array([True,True,False,True])
    moments = np.ones((4,5))
    out = profile(x,np.zeros(20),np.ones(20),8.,dict(all=np.ones(4,bool)),observed,moments,chunk=2)['all']
    assert [r['rows'] for r in out['bins']] == [1,1,1,1]
    assert sum(r['known_rows'] for r in out['bins']) == 3
    assert out['maximum_absolute_standardized_value'] == 20
    assert out['clipped_feature_counts'][0] == 3
    assert out['bins'][2]['known_moments'] == [0.]*5
    assert out['bins'][2]['all_bound_moments'] == [1.]*5


def test_fixed_region_is_not_replaced_by_label_support():
    x = np.zeros((3,20),np.float32)
    mask = np.array([False,True,False])
    observed = np.array([True,False,True])
    a = profile(x,np.zeros(20),np.ones(20),8.,dict(selected=mask),observed,np.ones((3,5)))
    b = profile(x,np.zeros(20),np.ones(20),8.,dict(selected=mask),~observed,np.full((3,5),99.))
    assert a['selected']['rows'] == b['selected']['rows'] == 1
    assert a['selected']['clipped_feature_counts'] == b['selected']['clipped_feature_counts']
    assert sum(c['known_rows'] for c in a['selected']['bins']) == 0


def test_zero_normalization_and_non_boolean_regions_are_rejected():
    args = (np.zeros((2,20)),np.zeros(20),np.ones(20),8.,dict(all=np.ones(2,bool)),np.ones(2,bool),np.zeros((2,5)))
    bad = list(args)
    bad[2] = np.zeros(20)
    with pytest.raises(ValueError):
        profile(*bad)
    bad = list(args)
    bad[4] = dict(all=np.ones(2))
    with pytest.raises(ValueError):
        profile(*bad)
