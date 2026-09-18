import numpy as np
import pytest
from scripts.analyze_m3w_source_cost_dynamics import site_ratio_interval, blocked_error_interval, loss_trace_summary, sampled_cv_trace


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


def test_loss_trace_does_not_claim_all_batch_gradients_or_convergence():
    trace=[dict(step=s,objective_loss=1.,normalized_batch_ade=2.,gradient_norm=g)
           for s,g in [(1,5.),(100,6.),(200,4.)]]
    result=loss_trace_summary(trace)
    assert result['logged_gradient_clipping_fraction']==pytest.approx(1/3)
    assert result['last_step']==200 and not result['convergence_established']
    assert not result['all_batch_gradients_observed']
    with pytest.raises(ValueError):loss_trace_summary(trace[::-1])
    with pytest.raises(ValueError):loss_trace_summary([])


def test_cv_loss_control_replays_training_sampler_not_held_rows():
    import torch
    cv=np.array([0.,1.,3.],np.float32);w=np.array([.2,.3,.5]);c=2.
    values,counts=sampled_cv_trace(cv,w,c,17,dict(updates=4,batch_size=5),[1,4])
    rng=torch.Generator().manual_seed(17+7919);expected=np.zeros(3,int)
    for step in range(1,5):
        idx=torch.multinomial(torch.as_tensor(w,dtype=torch.float64),5,replacement=True,generator=rng)
        np.add.at(expected,idx.numpy(),1)
        if step in (1,4):
            batch=torch.from_numpy(cv)[idx]/c
            assert values[step]['ade']==float(batch.mean())
            assert values[step]['log_ade']==float(torch.log1p(batch).mean())
    np.testing.assert_array_equal(counts,expected)
    assert counts.sum()==20
