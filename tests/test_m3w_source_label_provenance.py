import numpy as np
import pytest
from src.evaluation.m3w_european_squares_intake import DTYPE
from src.evaluation import m3w_source_label_provenance as api


def example(frames,other):
    raw=np.zeros(len(frames)+len(other),dtype=DTYPE)
    raw['agent']=np.r_[np.ones(len(frames),int),np.full(len(other),2)]
    raw['frame']=np.r_[frames,other];raw['x_min']=raw['frame'];raw['x_max']=raw['frame']+2
    raw['y_max']=2;raw.sort(order=['agent','frame'])
    grid=84+12*np.arange(1,13);valid=np.isin(grid,frames)[None]
    target=np.zeros((1,12,2));target[0,valid[0],0]=grid[valid[0]]+1;target[0,valid[0],1]=1
    return raw,np.array([1]),np.array([84]),np.array([[85.,1.]]),target,valid


def test_missing_steps_partition_without_inventing_a_video_end():
    frames=np.array([f for f in range(0,193,12) if f not in (120,144)])
    args=example(frames,np.array([0,120,300]));out=api.audit_rows(*args)
    assert out['all_stored_masks_and_known_coordinates_match']
    assert out['missing_steps_by_reason']==dict(zip(api.REASONS,[0,3,1,1]))
    assert out['observed_future_steps']==7 and out['final_step_observed_rows']==0
    assert out['outputs_not_inference_features']


def test_after_supplied_export_end_has_priority_but_is_not_a_physical_claim():
    out=api.audit_rows(*example(np.arange(0,169,12),np.array([0])))
    assert out['missing_steps_by_reason']==dict(zip(api.REASONS,[5,0,0,0]))
    assert out['missing_reason_is_export_evidence_not_physical_exit_or_video_end']


def test_raw_mask_and_known_xy_errors_are_reported_not_repaired():
    args=list(example(np.arange(0,229,12),np.array([0])))
    args[-1][0,0]=False;args[-2][0,1,0]+=1
    out=api.audit_rows(*args)
    assert not out['all_stored_masks_and_known_coordinates_match']
    assert out['raw_present_but_stored_missing_steps']==1 and out['known_coordinate_mismatch_steps']==1
    assert not args[-1][0,0]


def test_unknown_placeholder_values_do_not_become_labels_or_safe_zeros():
    args=list(example(np.arange(0,85,12),np.array([300])))
    args[-2][:]=np.nan;out=api.audit_rows(*args)
    assert out['missing_future_rows']==1 and out['observed_future_steps']==0
    assert sum(out['missing_steps_by_reason'].values())==12
    assert api.combine([out,out])['missing_future_rows']==2


def test_alignment_duplicates_and_empty_cohorts_fail():
    args=list(example(np.arange(0,229,12),np.array([0])))
    args[3]=args[3]+1
    with pytest.raises(ValueError):api.audit_rows(*args)
    args=list(example(np.arange(0,229,12),np.array([0])))
    args[0]=np.concatenate([args[0][:1],args[0]])
    with pytest.raises(ValueError):api.audit_rows(*args)
    with pytest.raises(ValueError):api.combine([])
