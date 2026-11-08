import numpy as np

from src.evaluation.m3w_ols_replacement_amplitude import components, maximize, value_slope


def test_linear_unknown_debit_and_exact_zero_residual_right_derivative():
    floor = np.zeros((2,2,2)); p = np.ones_like(floor)
    valid = np.array([[True,False],[False,False]])
    target = floor.copy(); target[~valid] = np.nan
    c = components(floor,p,target,valid)
    v,s = value_slope(c,0.)
    np.testing.assert_array_equal(v,[0.,0.])
    np.testing.assert_allclose(s,[-np.sqrt(2.)]*2)
    np.testing.assert_allclose(value_slope(c,.5)[0],[-np.sqrt(2.)/2.]*2)


def test_amplitude_optimum_at_observed_target_and_declines_unknown_switch():
    floor = np.zeros((2,1,2)); p = np.ones_like(floor)
    y = p*.25; valid = np.array([[True],[False]])
    c = components(floor,p,y,valid)
    a,v,_ = maximize(lambda z: value_slope(c,z))
    np.testing.assert_allclose(a,[.25,0.],atol=1e-9)
    np.testing.assert_allclose(v,[np.sqrt(2.)/4.,0.],atol=1e-9)
