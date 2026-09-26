import numpy as np
import pytest
from src.evaluation.m3w_easy_membership_diagnosis import decompose,summary


def fixture():
    cv=np.array([5.,5.,1.,1.,np.nan,0.]); env=np.array([2.,2.,2.,2.,2.,0.])
    y=np.array([[1,0,0,0],[1,1,0,0],[1,0,1,0],[1,1,1,1],[np.nan]*4,[0]*4],float)
    a=np.ones((6,4)); b=a.copy(); b[:,3]=2
    return a,b,y,cv,env


def test_four_partitions_preserve_total_and_ignore_unknown_structural_zero():
    d=decompose(*fixture(),2.)
    assert d['rows']==4 and d['easy']==2
    assert all(v['rows']==1 for v in d['parts'].values())
    assert sum(v['excess_MSE_contribution'] for v in d['parts'].values())==pytest.approx(d['excess_MSE'])
    assert d['parts']['outside_easy_positive_harm']['excess_MSE_contribution']==pytest.approx(.75)


def test_bad_easy_label_is_rejected():
    args=list(fixture()); args[2][0,3]=1
    with pytest.raises(AssertionError): decompose(*args,2.)


def test_unknown_support_mismatch_is_rejected():
    args=list(fixture()); args[3][4]=1
    with pytest.raises(ValueError): decompose(*args,2.)


def test_summary_is_dependent_views_not_independent_rows():
    d=decompose(*fixture(),2.); s=summary([dict(pair='full',folds=[dict(diagnosis=d)])])['full']
    assert s['views']==1 and s['worse_views']==1 and s['outside_easy_dominant']==1
