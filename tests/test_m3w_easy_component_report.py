import copy
import numpy as np
import pytest
from src.world_model.m3w_easy_component_diagnostic import diagnose
from scripts.report_m3w_easy_component_diagnostic import check_node


def node():
    a = np.array([[.4, 2., .3], [.6, 3., .2], [.7, 2., .1]])
    b = a + np.array([.02, .2, .01])
    y = np.array([[1., 1., .1], [1., 2., 0.], [0., 1., .1]])
    return diagnose(a, b, y, np.array(['a', 'a', 'b']), np.array(['r', 'r', 's']), np.array([1, 2, 1]))['source_balanced']


def test_independent_formula_verifies_diagnostic():
    assert check_node(node()) > 0


@pytest.mark.parametrize('field', ['attribution', 'objective', 'partition', 'empty'])
def test_rejects_changed_arithmetic(field):
    d = copy.deepcopy(node())
    if field == 'attribution': d['attribution']['marginal'][0] += .01
    if field == 'objective': d['transplants']['3']['marginal'] += .01
    if field == 'partition': d['arms']['uncapped']['partitions']['easy_zero_harm']['MSE_contribution'] += .01
    if field == 'empty': d['arms']['uncapped']['partitions']['all']['weight_mass'] = 0
    with pytest.raises(AssertionError): check_node(d)
