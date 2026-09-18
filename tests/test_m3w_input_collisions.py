import json
import numpy as np
from src.evaluation.m3w_input_collisions import input_keys,collision_summary


def test_restoration_frame_and_coverage_are_part_of_key():
    x=np.zeros((3,4),np.float32);c=np.ones((3,8,1,2,2),np.float32)
    r=np.ones(3,np.float32);q=np.repeat(np.eye(2)[None],3,axis=0);s=np.ones(3,bool)
    base=input_keys(x,c,r,q,s)
    assert len(set(base))==1
    r[1]=2;c[2,0,0,0,0]=0
    assert len(set(input_keys(x,c,r,q,s)))==3


def test_signed_zero_canonicalization():
    x=np.array([[0.],[-0.]],np.float32)
    keys=input_keys(x,np.ones((2,1)),np.ones(2),np.zeros((2,2,2)),np.ones(2,bool))
    assert keys[0]==keys[1]


def test_zero_majority_conflicting_targets_and_distinct_inputs():
    y=np.zeros((5,12,2));y[2,:,0]=3;y[3,:,0]=4;y[4,:,0]=2
    result=collision_summary(np.array(['a','a','a','b','c']),y)
    assert result['unique_observed_inputs']==3
    assert result['conflicting_groups']==1
    assert result['rows_in_zero_optimal_conflicting_groups']==3
    assert result['baseline_cost_fraction_in_zero_optimal_conflicts']==1/3
    assert json.loads(json.dumps(result,allow_nan=False))==result


def test_no_claim_when_zero_is_not_majority():
    y=np.zeros((3,12,2));y[1:,:,0]=2
    r=collision_summary(np.array(['a']*3),y)
    assert r['conflicting_groups']==1
    assert r['zero_optimal_conflicting_groups']==0
