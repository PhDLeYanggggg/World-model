import numpy as np
import pytest
from src.world_model.m3w_producer_detector_data import coverage,normalize


def population():
    sites=np.array([f's{i}' for i in range(12) for _ in range(3)])
    return sites,np.arange(36),{f's{i}':i%3 for i in range(12)}


def test_full_producer_coverage_never_silently_becomes_controller_subset():
    sites,records,folds=population();out=coverage(sites,records,np.arange(0,36,3),folds)
    assert out['rows']==36 and out['cached_rows']==12 and out['missing_rows']==24
    assert len(out['jobs'])==6 and not out['future_labels_used']
    assert out['jobs']['single0']['train_rows']==12
    assert out['jobs']['single0']['missing_train_inputs']==8
    assert out['jobs']['single0']['producer_excluded_fitting_readout_rows']==8
    with pytest.raises(ValueError):coverage(sites,records,np.array([0,0]),folds)
    with pytest.raises(ValueError):coverage(sites,records,np.array([36]),folds)


def test_detector_normalizer_does_not_use_other_sources_or_future_support():
    sites,_,_=population();x=np.arange(36*64,dtype=float).reshape(36,64)
    fit=['s0','s1','s2','s3'];a=normalize(x,sites,fit)
    x[~np.isin(sites,fit)]=1e10;b=normalize(x,sites,fit)
    np.testing.assert_array_equal(a['mean'],b['mean']);np.testing.assert_array_equal(a['std'],b['std'])
    assert not a['future_support_used']
    with pytest.raises(ValueError):normalize(x,sites,list(set(sites)))
