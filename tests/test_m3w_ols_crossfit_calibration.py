import numpy as np

from src.evaluation.m3w_ols_crossfit_calibration import choose_threshold,intervals,score_bins,targets


def test_empty_future_charges_debit_to_gain_and_harm():
    floor = np.zeros((2,2,2)); pred = floor[:,None].copy(); pred[...,0] = 1.
    target = floor.copy(); target[1] = np.nan
    cv,g,h,k = targets(floor,floor,pred,target,np.array([[True,False],[False,False]]))
    np.testing.assert_array_equal(g[:,0],[-1,-1])
    np.testing.assert_array_equal(h[:,0],[1,1])
    np.testing.assert_array_equal(k,[True,False])
    assert cv.sum() == 0


def test_selection_requires_actual_intervention_source_support_and_positive_ci():
    def entry(t,n,p,c,g):
        return dict(threshold=t,switches=n,lower_gain_pp=dict(point=p,interval95=[c,1.],positive_sources=g))
    rows = [entry(0,20,.3,0,9),entry(.01,0,.4,.02,9),entry(.02,8,.2,.01,6),entry(.04,5,.1,.01,7)]
    assert choose_threshold(rows,min_sources=7) == .04
    assert choose_threshold(rows[:-1],min_sources=7) is None


def test_weighted_score_bins_and_source_averaging():
    b = score_bins(np.array([.001,.002,.5]),np.array([-.1,.1,.4]),np.array([3.,1.,1.]))
    assert b[1]['rows'] == 2 and b[1]['realized'] < 0.
    a = intervals(np.array([[1.,3.],[0.,0.]]),np.array([[0,0],[1,1]]))
    assert a['point'] == 1. and a['positive_sources'] == 1
