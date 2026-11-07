import numpy as np
import pytest

from src.world_model import m3w_bounded_path_bank as bank


def test_level_major_choices_keep_floor_exact():
    floor = np.arange(4*12*2,dtype=float).reshape(4,12,2)
    paths = floor[:,None]+np.array([2.,4.])[None,:,None,None]
    choice = np.array([-1,1,3,4])
    y = bank.selected(paths,floor,choice)
    np.testing.assert_array_equal(y-floor,np.broadcast_to(np.array([0.,4.,2.,.5])[:,None,None],floor.shape))
    with pytest.raises(ValueError):
        bank.selected(paths,floor,np.array([-1,0,1,6]))


def test_attenuation_bounds_positive_harm_with_same_observed_mask():
    g = np.random.default_rng(51)
    b, p, y = g.normal(size=(3,32,12,2))
    mask = g.random((32,12))>.5
    choice = np.where(np.arange(32)%3,0,-1)
    baseline = bank.selected(p[:,None],b,choice)
    error = lambda x: (np.linalg.norm(x-y,axis=-1)*mask).sum(1)
    for alpha in (.5,.25):
        a = bank.selected(p[:,None],b,bank.attenuate(choice,1,alpha))
        assert np.all(np.maximum(error(a)-error(b),0.) <= alpha*np.maximum(error(baseline)-error(b),0.)+1e-12)
        np.testing.assert_allclose(np.linalg.norm(a-b,axis=-1).max(1),
                                   alpha*np.linalg.norm(baseline-b,axis=-1).max(1))
