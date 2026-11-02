import numpy as np
import pytest
from src.evaluation import m3w_selection_forecast_readout as api


def test_unknown_row_kept_and_missing_endpoint_not_called_full_fde():
    y=np.zeros((3,12,2));valid=np.ones((3,12),bool);valid[1]=False;valid[2,-1]=False
    floor=np.ones_like(y);good=np.zeros_like(y);cv=floor*2
    result=api.policy(good,floor,cv,y,valid,np.array([0,0,1]),easy_cut=10,hard_cut=20)
    assert result['rows']==3 and result['known_rows']==2 and result['wholly_unknown_rows']==1
    assert result['full_roster_FDE_pixel'] is None and result['available_endpoint_FDE_diagnostic_pixel']==0
    assert result['recordings']['0']['unknown_selected']==1
    assert result['completion_lower_gain_floor_percent']==pytest.approx(50.)


def test_easy_net_gain_does_not_cancel_positive_harm_or_rescue_zero_denominator():
    y=np.zeros((2,12,2));valid=np.ones((2,12),bool);floor=np.zeros_like(y);floor[:,:,0]=1
    p=np.zeros_like(y);p[0,:,0]=1.1
    result=api.policy(p,floor,floor,y,valid,np.zeros(2,int),easy_cut=2,hard_cut=3)
    rec=result['recordings']['0']
    assert rec['known_easy_net_degradation']<0 and rec['known_easy_risk']>.02
    assert not rec['supported']
    empty=api.policy(floor,floor,floor,y,valid,np.zeros(2,int),easy_cut=2,hard_cut=3)['recordings']['0']
    assert empty['selected']==0 and empty['completion_easy_risk_upper'] is None and not empty['supported']


def test_paired_bound_cancels_identical_unknown_choices():
    y=np.zeros((2,12,2));valid=np.ones((2,12),bool);valid[1]=False
    floor=np.ones_like(y);a=floor.copy();b=np.zeros_like(y);b[1]=a[1]
    result=api.paired(a,b,floor,y,valid)
    assert result['known_pp']==pytest.approx(100.)
    assert result['completion_lower_pp']==result['known_pp']==result['completion_upper_pp']


def test_locality_bootstrap_rejects_missing_scene_and_does_not_use_rows_as_units():
    rows=[dict(site=f's{i}',value=float(i)) for i in range(6) for _ in range(9)]
    result=api.interval(rows,lambda r:r['value'],expected_sites=[f's{i}' for i in range(6)],resamples=32,seed=1)
    assert result['point']==2.5 and len(result['by_locality'])==6
    with pytest.raises(ValueError):api.interval(rows[:-9],lambda r:r['value'],expected_sites=[f's{i}' for i in range(6)],resamples=32,seed=1)
