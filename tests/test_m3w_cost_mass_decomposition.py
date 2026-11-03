import numpy as np
import pytest

from src.evaluation.m3w_cost_mass_decomposition import cost_masses, risks


def test_amplitude_scales_harm_but_not_selected_reference_and_keeps_unknown_bound():
    y = np.array([[2., 4., 8., 3., 1.], [0., 6., 0., 0., 6.], [0., 100., 1., 1., 100.]])
    take = np.array([True, True, False])
    m = cost_masses(y, y, take, np.array([.25, .5, 0.]))
    assert m['H']['target_mass'] == 4.
    assert m['EH']['target_mass'] == 3.25
    assert m['R']['target_mass'] == 8.
    assert m['ER']['target_mass'] == 3.
    assert risks(m)['target_easy'] == 3.25/3
    assert risks(m)['predicted_all'] == .5


def test_empty_set_and_zero_reference_are_undefined_not_safe_zero():
    x = np.zeros((2, 5)); x[:, 1] = 3
    assert risks(cost_masses(x, x, np.array([False, False])))['target_all'] is None
    assert cost_masses(x, x, np.array([True, True]))['R']['predicted_over_target'] is None
    with pytest.raises(ValueError): cost_masses(x, x, np.array([1, 1]))
