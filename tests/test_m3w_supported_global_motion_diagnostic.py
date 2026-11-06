import numpy as np
import pytest
from scripts.diagnose_m3w_supported_global_motion import probability_metrics


def test_probability_endpoints_and_empty_bins():
    a=probability_metrics(np.array([0.,1.]),np.array([False,True]))
    assert a['Brier']==a['ECE']==0
    assert sum(b['count'] for b in a['bins'])==2
    assert a['bins'][-1]['count']==1
    assert a['bins'][4]['prediction'] is None


def test_future_observation_role_is_required_only_for_diagnostic():
    with pytest.raises(ValueError): probability_metrics(np.array([np.nan]),np.array([True]))
    with pytest.raises(ValueError): probability_metrics(np.array([.2]),np.array([1]))
    a=probability_metrics(np.array([0.,1.]),np.array([True,False]))
    assert a['Brier']==a['ECE']==1
