import numpy as np
import pytest
from src.world_model.m3w_centered_risk_policy import decisions, check_queries


def test_fixed_offset_changes_allocation_not_count_in_matched_control():
    risk=np.array([[-2.,-2.],[-2.,-2.],[1.,1.]])
    eligible=np.ones(3,bool);keys=np.array(['r']*3);frames=np.ones(3,int);ids=np.arange(3)
    # Raw admits the high-utility third row with one negative companion.
    # Centering makes that pair infeasible while retaining a count of two.
    out,_=decisions(np.array([1.,2.,9.]),risk,[.75,.75],eligible,keys,frames,ids)
    assert out['centered_joint'].tolist()==[True,True,False]
    assert out['matched_raw_joint'].tolist()==[False,True,True]
    c=check_queries(out,risk,[.75,.75],eligible,keys,frames,ids)
    assert c['changed_equal_count_queries']==1


def test_zero_offsets_recover_identical_control():
    risk=np.array([[-2.,-2.],[-2.,-2.],[1.,1.]])
    out,_=decisions(np.array([1.,2.,9.]),risk,[0.,0.],np.ones(3,bool),np.array(['r']*3),np.ones(3,int),np.arange(3))
    np.testing.assert_array_equal(out['centered_joint'],out['matched_raw_joint'])


def test_no_new_independent_admissions_and_no_cross_query_borrowing():
    risk=np.array([[-.5,-.5],[-.5,-.5],[1.,1.],[1.,1.]])
    e=np.ones(4,bool);rec=np.array(['r']*4);frames=np.array([1,2,1,2]);ids=np.arange(4)
    out,_=decisions(np.arange(1.,5),risk,[.6,0],e,rec,frames,ids)
    assert not out['centered_independent'].any() and not out['centered_joint'].any()
    assert check_queries(out,risk,[.6,0],e,rec,frames,ids)['zero_action_queries']==2


def test_unknown_outcome_cannot_be_an_argument():
    import inspect
    assert set(inspect.signature(decisions).parameters)=={'utility','signed_risk','offset','eligible','recordings','frames','ids','node_limit'}


@pytest.mark.parametrize('offset',[[-1,0],[np.nan,0],[0,np.inf],[1]])
def test_bad_offset_rejected(offset):
    with pytest.raises(ValueError):decisions(np.ones(2),-np.ones((2,2)),offset,np.ones(2,bool),np.array(['r']*2),np.ones(2,int),np.arange(2))


def test_ineligible_rows_never_selected():
    e=np.array([True,False]);q=-np.ones((2,2));r=np.array(['r']*2);f=np.ones(2,int);ids=np.arange(2)
    out,_=decisions(np.array([1.,0]),q,[.1,.1],e,r,f,ids)
    assert all(not v[1] for v in out.values());check_queries(out,q,[.1,.1],e,r,f,ids)


def test_row_permutation_preserves_non_tied_solution():
    q=np.array([[-2.,-2.],[-2.,-2.],[1.,1.]])
    u=np.array([1.,2.,9.]);e=np.ones(3,bool);r=np.array(['r']*3);f=np.ones(3,int);ids=np.array([8,1,5]);ix=np.array([2,0,1])
    a,_=decisions(u,q,[.75,.75],e,r,f,ids)
    b,_=decisions(u[ix],q[ix],[.75,.75],e[ix],r[ix],f[ix],ids[ix])
    for key in a:np.testing.assert_array_equal(a[key][ix],b[key])
