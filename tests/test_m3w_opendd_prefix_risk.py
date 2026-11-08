import numpy as np

from scripts.audit_m3w_opendd_prefix_transitions import transitions
from src.evaluation.m3w_opendd_prefix_risk import choose, compression, walk


def test_streaming_prefix_uses_only_past_predicted_credit():
    cv = np.zeros((4,12,2))
    candidate = cv+1.
    d = dict(cv=cv,origin=np.array([[0.,0.],[1.,0.],[0.,0.],[1.,0.]]),offsets=[0,2,4],recordings=np.array(['a']*4))
    moments = np.array([[1.,0.,0.,1.,1.],[-1.,0.,0.,1.,1.],
                        [1.,.03,.03,1.,1.],[-1.,0.,0.,1.,1.]])
    switches,_ = walk(d,candidate,moments)
    assert switches['local'].tolist()==[True,False,False,False]
    assert switches['prefix'].tolist()==[True,False,True,False]
    assert switches['joint'].tolist()==switches['matched_independent'].tolist()


def test_no_predicted_easy_reference_means_no_edit():
    cv = np.zeros((2,12,2))
    costs = np.array([[1.,0.,0.,1.,0.],[1.,0.,0.,1.,0.]])
    choices,_ = choose(costs,np.ones(2,dtype=bool),np.zeros((2,2)),cv,cv+1.,np.zeros(4))
    assert choices==()


def test_joint_geometry_differs_from_sum_of_single_edits():
    origin = np.array([[0.,0.],[2.,0.],[50.,0.]])
    cv = np.zeros((3,12,2))
    new = cv.copy()
    new[:2,:,0] = 1.
    assert compression(origin,cv,new,(0,1))==0.
    assert compression(origin,cv,new,(0,))+compression(origin,cv,new,(1,))==1.


def test_paired_flight_audit_keeps_support_loss_distinct_from_rescue():
    cv = np.zeros((4,12,2))
    y = cv.copy()
    y[:,:,0] = np.array([.1,1.,.1,.1])[:,None]
    candidate = cv.copy()
    candidate[:,:,0] = np.array([.01,.01,.01,-.1])[:,None]
    d = dict(cv=cv,y=y,valid=np.ones((4,12),dtype=bool),recordings=np.array(['a','a','b','b']))
    masks = {a:np.array([False,True,True,False]) for a in ('local','prefix','joint')}
    r = transitions(d,candidate,masks,.2,.8)['joint']
    assert r['lost_pass']==r['lost_pass_undefined_easy']==r['rescued_pass']==1
