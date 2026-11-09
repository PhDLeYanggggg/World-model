import numpy as np

from scripts import run_m3w_cv_record_allocation as study
from scripts import audit_m3w_cv_record_allocation_capacity as capacity


def test_offline_count_matches_each_recording_and_respects_query_cap():
    q = np.repeat(np.arange(3),10)
    rec = np.repeat([0,1],[20,10])
    groups = study.blocks(q,rec)
    paths = np.zeros(30,dtype=np.int8)
    score = np.arange(30)
    quota = np.repeat([8,1,2],10)
    out = study.offline(paths,score,groups,quota)
    assert (out[:20]>=0).sum()==9 and (out[20:]>=0).sum()==2
    assert (out[10:20]>=0).sum()==8
    np.testing.assert_array_equal(np.flatnonzero(out>=0),[9,*range(12,20),28,29])


def test_online_no_future_credit_and_prefix_is_invariant_to_future():
    q = np.arange(6)
    rec = np.array([0,0,0,0,1,1])
    quota = np.array([0,2,1,0,0,1])
    score = np.array([3.,0.,2.,2.,3.,0.])
    path = np.zeros(6,dtype=np.int8)
    groups = study.blocks(q,rec)
    action,late = study.online(path,score,groups,quota,1.,carry=True)
    np.testing.assert_array_equal(action,[-1,-1,0,0,-1,-1])
    assert late==2
    local,_ = study.online(path,score,groups,quota,1.,carry=False)
    np.testing.assert_array_equal(local,[-1,-1,0,-1,-1,-1])
    for stop in range(1,len(q)+1):
        prefix = study.blocks(q[:stop],rec[:stop])
        truncated,_ = study.online(path[:stop],score[:stop],prefix,quota[:stop],1.,carry=True)
        np.testing.assert_array_equal(truncated,action[:stop])


def test_large_recording_budget_does_not_overflow_byte_quota():
    q = np.repeat(np.arange(30),8)
    rec = np.zeros(len(q))
    out = study.offline(np.zeros(len(q),dtype=np.int8),q,study.blocks(q,rec),
                        np.full(len(q),8,dtype=np.int8))
    assert (out>=0).sum()==240


def test_expiry_prevents_old_or_cross_record_spend():
    n = study.TTL+4
    q = np.arange(n)
    quota = np.zeros(n,dtype=int)
    quota[[0,n-2]] = 1
    score = np.zeros(n)
    score[[study.TTL+1,n-1]] = 10.
    rec = np.r_[np.zeros(n-1),1]
    out,late = study.online(np.zeros(n,dtype=int),score,study.blocks(q,rec),quota,1.,carry=True)
    assert (out<0).all() and late==0


def test_record_budget_can_find_support_unavailable_at_exact_query_counts():
    data = dict(row=np.arange(2),group=np.arange(2),counts=np.ones(2),ref=1.,
                easy=np.array([False,True]),risk=np.zeros((2,3)),easy_cap=0.,
                offset=np.zeros(2),gain=np.array([[2.,2.],[1.,1.]]))
    counts = np.array([1,0])
    choice,status = capacity.solve(data,counts,'query_exact')
    assert choice is None and status=='risk_infeasible'
    choice,_ = capacity.solve(data,counts,'record_exact')
    np.testing.assert_array_equal(choice,[False,True])
