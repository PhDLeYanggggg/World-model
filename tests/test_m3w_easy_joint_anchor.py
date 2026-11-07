import numpy as np

from src.world_model import m3w_easy_joint_anchor as model


def test_opening_never_looks_after_a_direct_switch_or_uses_singleton():
    groups = [np.array([0]),np.array([1,2]),np.array([3,4]),np.array([5,6])]
    direct = np.array([-1,-1,-1,-1,0,-1,-1])
    v = np.arange(7)[:,None].astype(float)
    q = model.prefix_options(groups,direct,v,np.ones_like(v,bool))
    assert [(x[1],x[2]) for x in q]==[(2,0)]
    assert model.first(q,None) is None


def test_calibration_repairs_zero_support_without_losing_existing_passes():
    groups = np.array([0,1])
    atoms = np.array([[[1.,1.,0.,0.,-.2,.2]],[[1.,1.,.3,.3,.3,-.3]]])
    records = [dict(mass=np.zeros(6),easy=3.,options=[(.5,0,0,groups)],atoms=atoms,crowd=[2,2]),
               dict(mass=np.array([1.,1.,0.,0.,0.,0.]),easy=1.,
                    options=[(.2,1,0,groups)],atoms=atoms,crowd=[2,2])]
    cut,table = model.calibrate(records)
    assert cut==.5
    assert table[0]['lost']==1


def test_unknown_and_easy_support_are_distinct_events():
    c = np.zeros((3,1,2))
    y = np.array([[[1.,0.]],[[3.,0.]],[[np.nan,np.nan]]])
    bank = np.array([[[[.5,0.]]],[[[.5,0.]]],[[[.5,0.]]]])
    event,atoms,easy = model.labels(c,bank,y,np.array([[True],[True],[False]]),2.)
    np.testing.assert_array_equal(event[:,0],[[True,True,True,False],
                                              [False,False,True,False],
                                              [False,False,False,True]])
    assert easy==1. and atoms[2,0,3]==.5
