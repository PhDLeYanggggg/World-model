import numpy as np
import pytest

from src.evaluation.m3w_amplitude_shape_audit import diagnose


def fixture():
    alpha=np.array([0.,1/32,1/16,1/8,1/4,1/2,1.])
    p=np.zeros((4,7,5));env=np.broadcast_to(alpha,(4,7)).copy()
    p[:,:,1]=alpha**2
    p[:,:,4]=.5*alpha**2
    p[:,:,2]=2.;p[:,:,3]=1.
    return p,env


def test_convex_harm_and_shared_references_have_no_shape_violations():
    p,env=fixture()
    result=diagnose(p,p.copy(),np.array([True,True,True,False]),np.array([6,3,-1,6]),env)
    assert result['active_rows']==4 and result['selected']==3
    for component in ('H','EH'):
        assert result['predicted'][component]['nonmonotone_rows']==0
        assert result['predicted'][component]['nonconvex_rows']==0
        assert result['predicted'][component]['ratio_order_violating_rows']==0
    assert result['reference_max_relative_span']==[0.,0.]


def test_selected_harm_mass_distinguishes_count_from_harm_and_unknown_labels():
    p,env=fixture();truth=p.copy()
    p[0,5,1]=.4;p[0,6,1]=.1
    p[0,5,4]=.2;p[0,6,4]=.05
    # This third row is unknown and must not add invented realized easy harm.
    p[2]=p[0]
    truth[0,:,4]=0.;truth[0,6,4]=10.
    truth[1,:,4]=0.;truth[1,6,4]=1.
    result=diagnose(p,truth,np.array([True,True,False,True]),np.array([6,6,6,-1]),env)
    assert result['predicted']['H']['nonmonotone_rows']==2
    assert result['selected_ratio_inversion']['H']['selected_rows']==2
    assert result['selected_ratio_inversion']['H']['known_EH_mass']==10.
    assert result['selected_known_EH_mass']==11.
    assert result['selected_ratio_inversion']['H']['known_EH_mass_share']==pytest.approx(10/11)


def test_full_convexity_is_stronger_than_monotone_harm_over_amplitude():
    p,env=fixture()
    p[:,:,1]=np.array([0.,.01,.03,.09,.2,.49,.99])
    result=diagnose(p,p.copy(),np.ones(4,bool),np.full(4,6),env)
    assert result['predicted']['H']['nonmonotone_rows']==0
    assert result['predicted']['H']['ratio_order_violating_rows']==0
    assert result['predicted']['H']['nonconvex_rows']==4


def test_alias_rows_and_absent_selected_mass_remain_undefined_not_safe_zero():
    p,env=fixture();p[:,:,:]=0.;env[:]=0.
    result=diagnose(p,p.copy(),np.ones(4,bool),np.full(4,-1),env)
    assert result['active_rows']==0
    assert result['selected_ratio_inversion']['EH']['known_EH_mass_share']is None
    with pytest.raises(ValueError):
        diagnose(p,p,np.ones(4),np.full(4,-1),env)
