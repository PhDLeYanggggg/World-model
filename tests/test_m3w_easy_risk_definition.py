import numpy as np
import pytest
from src.evaluation.m3w_easy_risk_definition import sums


def test_positive_harm_and_net_degradation_are_distinct():
    r=sums(np.array([1.,1.]),np.array([1.2,.6]),np.ones(2,bool),np.ones(2,bool),
           np.array([.2,0]),np.ones(2),2.)
    assert r['observed_positive_selected_ratio']==pytest.approx(10.)
    assert r['observed_net_selected_ratio']==pytest.approx(-10.)
    assert r['benefit_cancels_positive_harm_percent']==pytest.approx(200.)


def test_missing_outcomes_retained_as_unknown_not_zero_harm():
    r=sums(np.array([1.,np.nan]),np.array([1.2,np.nan]),np.array([True,False]),
           np.ones(2,bool),np.array([.2,.1]),np.ones(2),2.)
    assert r['selected']==2 and r['complete_selected']==1 and r['incomplete_selected']==1
    assert r['observed_positive_harm']==pytest.approx(.2)


def test_zero_CV_harm_is_not_a_percent_ratio():
    r=sums(np.zeros(1),np.ones(1),np.ones(1,bool),np.ones(1,bool),np.ones(1),np.ones(1),2.)
    assert r['zero_CV_harmed']==1 and r['observed_net_selected_ratio'] is None


def test_population_denominator_includes_unswitched_complete_easy_rows():
    r=sums(np.ones(2),np.array([1.2,1.]),np.ones(2,bool),np.array([True,False]),np.ones(2),np.ones(2),2.)
    assert r['observed_positive_selected_ratio']==pytest.approx(20.)
    assert r['observed_positive_population_ratio']==pytest.approx(10.)


def test_observed_arrays_cannot_be_nonfinite_on_complete_rows():
    with pytest.raises(ValueError):
        sums(np.ones(1),np.array([np.nan]),np.ones(1,bool),np.ones(1,bool),np.ones(1),np.ones(1),2.)
