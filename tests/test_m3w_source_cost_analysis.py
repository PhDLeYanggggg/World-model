import numpy as np
import pytest
from scripts.analyze_m3w_source_cost_dynamics import site_ratio_interval, blocked_error_interval


def test_site_primary_is_ratio_of_equal_site_errors_not_mean_percentage():
    v=site_ratio_interval([1.,8.],[2.,8.])
    assert v['gain_percent']==pytest.approx(10.)
    assert v['gain_percent']!=pytest.approx(25.)
    assert not v['independent_confirmation']


def test_blocked_errors_average_seed_losses_and_keep_video_rows_together():
    p=np.array([[0,0,4],[2,2,0]],float);r=np.full((2,3),2.)
    v=blocked_error_interval(p,r,np.array(['a','a','b']))
    assert v['model_error']==pytest.approx(4/3)
    assert v['gain_percent']==pytest.approx(100/3)
    assert v['relative_conditional_ci95']==pytest.approx([0,50])
    assert v['blocks']==2


def test_zero_reference_blocks_are_disclosed_not_divided_by_epsilon():
    v=blocked_error_interval(np.ones((1,3)),np.array([[0,0,2.]]),np.array(['a','a','b']))
    assert v['bootstrap_zero_reference_draws']>0
    assert v['relative_conditional_ci95'] is not None
    z=blocked_error_interval(np.ones((1,2)),np.zeros((1,2)),np.array(['a','b']))
    assert z['gain_percent'] is None and z['relative_conditional_ci95'] is None
    with pytest.raises(ValueError):site_ratio_interval([1.,2.],[0.,1.])
