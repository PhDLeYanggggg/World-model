import numpy as np

from src.evaluation.m3w_opendd_easy_support import event, revise


def test_event_is_about_positive_harm_and_nonidentity_easy_support():
    v = np.ones(4,dtype=bool)
    got = event(v,np.array([True,True,False,True]),np.ones(4),np.array([.9,1.03,.9,1.01]))
    assert got.tolist()==[True,False,False,True]


def test_rerank_never_changes_matched_count_or_selects_predicted_unsafe_edit():
    m = np.array([[1.,.01,.01,1.,1.],[.5,.015,.015,1.,1.],[2.,.03,.03,1.,1.]])
    legal = np.ones(3,dtype=bool)
    p = np.array([.1,.45,.9])
    assert revise((0,),m,legal,p,0.,False)==(1,)
    assert revise((),m,legal,p,0.,False)==()
    assert revise((),m,legal,p,0.,True)==(1,)
    assert revise((0,),m,legal,p,1.,False)==(0,)
