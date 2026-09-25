import json
from pathlib import Path
import numpy as np
import pytest
from src.evaluation.m3w_symmetric_risk import validate_config, risk_diagnostic

ROOT=Path(__file__).resolve().parents[1]


def test_only_risk_loss_changes():
    old=json.loads((ROOT/'configs/m3w_european_symmetric_utility_v1.json').read_text())
    new=json.loads((ROOT/'configs/m3w_european_symmetric_risk_v1.json').read_text())
    validate_config(new,old)
    for key,v in [('predicted_risk_budget',.03),('utility_head','different'),('independent_reserved_readout',True)]:
        with pytest.raises(ValueError,match='simultaneous'):
            validate_config(dict(new,**{key:v}),old)


def test_selected_moments_are_not_population_calibration():
    p=np.array([[10.,.1],[1.,.01],[1.,.01]])
    y=np.array([[10.,0.],[1.,2.],[np.nan,np.nan]])
    r=risk_diagnostic(p,y,np.array([False,True,True]))
    assert r['selected']['predicted_ratio']==.01
    assert r['selected']['realized_ratio']==2.
    assert r['all']['realized_ratio']==pytest.approx(2/11)
    assert r['unknown_rows']==r['selected_unknown_rows']==1
    assert r['calibrated_safety'] is False


def test_empty_event_support_is_not_zero_risk():
    r=risk_diagnostic(np.ones((2,2)),np.zeros((2,2)),np.array([True,False]))
    assert r['selected']['realized_ratio'] is None
    r=risk_diagnostic(np.ones((2,2)),np.full((2,2),np.nan),np.ones(2,bool))
    assert r['selected']['rows']==0 and r['selected']['mae'] is None


@pytest.mark.parametrize('target',[np.array([[np.nan,0.]]),np.array([[1.,-1.]])])
def test_invalid_support_is_rejected(target):
    with pytest.raises(ValueError):
        risk_diagnostic(np.ones((1,2)),target,np.ones(1,bool))
