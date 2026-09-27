import numpy as np
import pytest
from scripts.verify_m3w_query_excess_refit import check_selected_costs


def test_zero_coverage_is_undefined_not_zero_risk():
    cv=np.array([1.,2.,np.nan]);floor=np.array([1.,1.,np.nan]);neural=np.array([2.,.5,np.nan])
    m=dict(known_rows=2,unknown_interventions=0,error_sum=2.,floor_error_sum=2.,CV_error_sum=3.,
           intervention_rate=0.,selected_positive_harm_ratio=None,zero_CV_harmed=0)
    check_selected_costs(m,cv,floor,neural,np.zeros(3,bool))
    with pytest.raises(AssertionError):
        check_selected_costs(dict(m,selected_positive_harm_ratio=0.),cv,floor,neural,np.zeros(3,bool))


def test_harm_cannot_be_cancelled_by_benefit_or_unknown_labels():
    cv=np.array([1.,2.,np.nan]);floor=np.array([1.,1.,np.nan]);neural=np.array([2.,0.,np.nan])
    m=dict(known_rows=2,unknown_interventions=1,error_sum=2.,floor_error_sum=2.,CV_error_sum=3.,
           intervention_rate=1.,selected_positive_harm_ratio=.5,zero_CV_harmed=0)
    check_selected_costs(m,cv,floor,neural,np.ones(3,bool))
    with pytest.raises(AssertionError):
        check_selected_costs(dict(m,selected_positive_harm_ratio=0.),cv,floor,neural,np.ones(3,bool))
