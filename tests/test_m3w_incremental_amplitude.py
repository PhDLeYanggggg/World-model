import numpy as np

from src.world_model import m3w_incremental_amplitude as m


def test_no_upgrade_and_safe_tie_at_smallest():
    d = np.array([[[3,1,1],[2,0,0]],[[1,2,0],[2,2,0]],[[5,0,0],[5,0,0]]])
    np.testing.assert_array_equal(m.choose(np.array([1,2,0]),m.scores(d)),[1,0,0])


def test_amplitude_multiset_control_keeps_row_caps():
    upper = np.array([2,1,2,0,2])
    chosen = np.array([1,0,2,0,1])
    v = np.array([[0,1,2],[0,3,0],[0,1,2],[0,9,9],[0,3,0]])
    z, n = m.match([np.array([0,1]),np.array([2,3]),np.array([4])],upper,chosen,v)
    np.testing.assert_array_equal(z,[0,1,2,0,1])
    assert n==1


def test_absolute_harm_increment_has_unchanged_completion_radius():
    cv = np.zeros((2,12,2))
    path = np.ones_like(cv)
    y = cv.copy()
    valid = np.array([[True]*12,[False]*12])
    z = m.outcomes(path[:,None],cv,y,valid,1.)
    a = m.outcomes(.25*path[:,None],cv,y,valid,1.)
    d = m.increment(a,z)
    np.testing.assert_allclose(d[:,0,0],-.75*np.sqrt(2))
    np.testing.assert_allclose(d[:,0,1],.75*np.sqrt(2))
    np.testing.assert_allclose(d[:,0,2],[0.,.75*np.sqrt(2)])
    np.testing.assert_array_equal(m.project(np.array([[5.,4.,8.]]),np.array([2.])),[[2.,2.,2.]])
