import numpy as np

from src.evaluation import m3w_prefix_capacity as old, m3w_timing_capacity as new


def test_same_recording_total_can_recover_an_eligible_easy_row():
    c = np.zeros((4,1,2))
    y = c.copy()
    y[...,0] = 1.
    b = (c + [.6,0.])[:,None]
    valid = np.array([[False],[False],[True],[True]])
    ch = np.array([0,-1,-1,-1])
    direct = c.copy()
    direct[0] = b[0,0]
    g = [np.array([0,1]),np.array([2,3])]
    est = np.zeros((2,4,1))
    legal = np.ones((4,1),bool)
    p = old.prepare(c,b,direct,ch,y,valid,g,2.,est,legal)
    assert old.solve(p,c,b,direct,y,valid,'r',2.,4.,'fixed',2.)['status']=='infeasible'
    result = new.solve(p,c,b,direct,y,valid,'r',2.,4.,'total_prefix',2.)
    assert result['status']=='optimal'
    assert result['switches']==result['retimed_switches']==1
    assert result['risk']['supported']


def test_online_credit_can_only_move_to_later_queries():
    g = [np.array([i,i+1]) for i in (0,2,4)]
    mint = np.array([-1,-1,0,-1,-1,-1])
    u = np.array([[9.],[0.],[-1.],[0.],[5.],[4.]])
    a = np.zeros_like(u)
    legal = np.ones_like(u,dtype=bool)
    local = new.retime(g,u,a,a,legal,mint,carry=False)
    banked = new.retime(g,u,a,a,legal,mint,carry=True)
    np.testing.assert_array_equal(local,[-1,-1,-1,-1,-1,-1])
    np.testing.assert_array_equal(banked,[-1,-1,-1,-1,0,-1])
    truncated = new.retime(g[:2],u[:4],a[:4],a[:4],legal[:4],mint[:4],carry=True)
    np.testing.assert_array_equal(truncated,banked[:4])
