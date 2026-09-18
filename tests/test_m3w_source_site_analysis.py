import numpy as np
import pytest
from scripts.analyze_m3w_source_site_probe import paired_block_interval, equal_site_interval


def test_bootstrap_keeps_rows_together_and_averages_seed_losses():
    y=np.array([0,0,1]);p=np.array([[0,0,0],[1,1,1]],float);r=np.full((2,3),.5)
    result=paired_block_interval(y,p,r,np.array(['a','a','b']))
    assert result['row_lift']==pytest.approx(-.25)
    assert result['conditional_ci95']==pytest.approx([-.25,-.25])
    assert result['blocks']==2
    assert not result['independent_confirmation']


def test_video_block_row_weighting_is_not_equal_agent_weighting():
    y=np.zeros(4);p=np.array([[0,0,0,1]]);r=np.full((1,4),.5)
    result=paired_block_interval(y,p,r,np.array(['a','a','a','b']))
    assert result['row_lift']==pytest.approx(0)
    assert result['conditional_ci95']==pytest.approx([-.75,.25])


def test_equal_site_estimate_not_pooled_window_estimate():
    r=equal_site_interval(np.array([.1,-.2,.3,-.1,0]))
    assert r['equal_site_lift']==pytest.approx(.02)
    assert r['physical_sites']==5 and r['resamples']==2000
    assert r==equal_site_interval(np.array([.1,-.2,.3,-.1,0]))


def test_invalid_predictions_and_single_block_are_explicit():
    with pytest.raises(ValueError):
        paired_block_interval(np.ones(2),np.ones(2),np.ones(2),np.ones(2))
    with pytest.raises(ValueError):
        equal_site_interval([np.nan,0])
    assert not paired_block_interval(np.zeros(2),np.zeros((1,2)),np.zeros((1,2)),np.array(['a','a']))['available']
