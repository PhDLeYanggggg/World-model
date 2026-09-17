import numpy as np
import pytest

from src.evaluation.m3w_quantized_motion_feasibility import quantized_cv_feasibility


def test_hidden_subpixel_cv_can_fit_stationary_then_small_grid_steps():
    times=np.arange(-7,13)
    xy=np.column_stack([np.round(times*.1),np.zeros(len(times))])
    assert quantized_cv_feasibility(xy,times,np.eye(3))['feasible'] is True


def test_large_accelerated_start_after_eight_stationary_points_is_infeasible():
    times=np.arange(-7,13)
    x=np.maximum(times,0)
    r=quantized_cv_feasibility(np.column_stack([x,np.zeros(len(times))]),times,np.eye(3))
    assert r['feasible'] is False and not r['physical_motion_proven']


def test_projective_mapping_is_not_replaced_by_linear_pixel_velocity():
    times=np.arange(-7,13)
    h=np.array([[.1,0,1.],[0,.2,0.],[.001,0,1.]])
    native=np.column_stack([2+times*.03,np.ones(len(times))])
    hi=np.linalg.inv(h)
    pixel_h=np.c_[native,np.ones(len(native))]@hi.T
    pixel=np.round(pixel_h[:,:2]/pixel_h[:,2:])
    label_h=np.c_[pixel,np.ones(len(pixel))]@h.T
    labels=label_h[:,:2]/label_h[:,2:]
    assert quantized_cv_feasibility(labels,times,h)['feasible'] is True


def test_noninteger_lineage_or_invalid_times_are_not_declared_failed_physics():
    xy=np.array([[.2,.2],[.3,.3]])
    r=quantized_cv_feasibility(xy,[0,1],np.eye(3))
    assert r['feasible'] is None and r['status'].startswith('not_run')
    with pytest.raises(ValueError):
        quantized_cv_feasibility(xy,[0,0],np.eye(3))


def test_aggregate_retains_inconclusive_and_undefined_zero_error_share():
    from scripts.audit_m3w_quantized_cv_feasibility import summarize
    rows = [{'recording_id': 'a', 'agent_id': 1, 'first_row': 0}]*2
    results = [{'feasible': True, 'status': 'feasible'}, {'feasible': None, 'status': 'not_run'}]
    report = summarize(rows, results, np.ones(2, bool), np.zeros(2))
    assert report['categories']['not_run']['rows']==1
    assert report['categories']['feasible']['native_cv_error_share'] is None
    assert report['categories']['infeasible']['rows']==0


def test_agent_run_counts_are_source_scoped_not_window_counts():
    from scripts.audit_m3w_quantized_cv_feasibility import summarize
    rows = [{'recording_id': rid, 'agent_id': 1, 'first_row': 0} for rid in ('a','a','b')]
    results = [{'feasible': False, 'status': 'infeasible'}]*3
    report = summarize(rows, results, np.ones(3, bool), np.array([1.,2.,3.]))
    category = report['categories']['infeasible']
    assert (category['rows'], category['agents'], category['runs'])==(3,2,2)
    assert category['native_cv_error_share']==1
