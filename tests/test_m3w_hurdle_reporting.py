import numpy as np
from scripts.run_m3w_european_hurdle_risk import calibration_diagnostic


def test_factor_reliability_unknown_empty_selected_and_positive_only():
    labels=np.array([[1.,0.],[1.,.5],[np.nan,np.nan]])
    factors=np.array([[.2,.1],[.8,.5],[1.,1.]])
    result=calibration_diagnostic(factors,labels,np.ones(3),np.array(['a','a','b']),np.array([False,True,True]))
    a=result['a']['population']
    assert a['rows']==2 and a['positive_rows']==1
    assert abs(a['brier']-.04)<1e-8 and abs(a['ece']-.2)<1e-8
    assert a['positive_fraction_mse']==0
    assert result['a']['selected']['positive_rate']==1
    assert result['b']['selected']['rows']==0
    assert result['b']['selected']['brier'] is None
    assert result['b']['selected']['positive_fraction_mse'] is None
