import numpy as np

from src.evaluation.m3w_opendd_location import paired
from src.evaluation.m3w_single_exchange_envelope import bound


def test_exchange_envelope_covers_known_and_missing_step_utility():
    cv = np.zeros((2,3,2))
    ols = np.array([[[1.,0.]]*3, [[.4,0.]]*3])
    old = cv.copy()
    old[0,:,0] = .5
    y = np.array([[[1.,0.]]*3, [[1.,0.]]*3])
    valid = np.array([[True,True,False],[True,False,True]])
    refs, w = np.stack((cv,ols)), np.array([.5,.5])
    _,swap,query = bound(cv,refs,old,y,valid,[np.arange(2)],w)
    rng = np.random.default_rng(9)
    for _ in range(100):
        other = cv.copy()
        theta = rng.uniform(-np.pi,np.pi,3)
        other[1] = .5*np.column_stack((np.cos(theta),np.sin(theta)))
        gain = (paired(refs,other,y,valid)[1].sum(-1)*w).sum(1)
        assert (gain <= swap+1e-12).all()
    assert (swap <= query+1e-12).all()
