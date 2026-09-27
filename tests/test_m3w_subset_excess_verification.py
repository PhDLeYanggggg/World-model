import numpy as np
import pytest
from scripts.verify_m3w_subset_excess import check_fixed_denominator,reconstruct_bank


def test_no_intervention_has_zero_total_harm_but_no_selected_denominator():
    f=np.array([1.,2.,np.nan]);n=np.array([3.,0.,np.nan]);known=np.isfinite(f)
    m=dict(positive_harm_sum=0.,benefit_sum=0.,selected_reference_error=0.,selected_known_count=0,
           positive_harm_over_all_floor=0.,floor_error_sum=3.,error_sum=3.)
    check_fixed_denominator(m,f,n,known,np.zeros(3,bool))
    with pytest.raises(AssertionError):check_fixed_denominator(dict(m,error_sum=2.),f,n,known,np.zeros(3,bool))


def test_positive_harm_not_cancelled_by_equal_benefit():
    f=np.array([1.,2.,np.nan]);n=np.array([3.,0.,np.nan]);known=np.isfinite(f)
    m=dict(positive_harm_sum=2.,benefit_sum=2.,selected_reference_error=3.,selected_known_count=2,
           positive_harm_over_all_floor=2/3,floor_error_sum=3.,error_sum=3.)
    check_fixed_denominator(m,f,n,known,np.ones(3,bool))
    with pytest.raises(AssertionError):check_fixed_denominator(dict(m,positive_harm_sum=0.),f,n,known,np.ones(3,bool))


def test_independent_bank_singleton_odd_and_source_boundaries():
    m=reconstruct_bank(np.array(['a','a','a','b']),np.repeat('r',4),np.ones(4),
        np.array([2,1,0,3]),np.ones(4,bool),np.array([0,1,1,0],bool),np.ones(4))
    np.testing.assert_array_equal(m[:,1],[False,True,True,True])
    np.testing.assert_array_equal(m[:,2],[True,False,False,False])
