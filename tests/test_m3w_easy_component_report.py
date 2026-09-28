import copy
import numpy as np
import pytest
from src.world_model.m3w_easy_component_diagnostic import diagnose, compose, weights, objective
from scripts.report_m3w_easy_component_diagnostic import check_node


def node():
    a = np.array([[.4, 2., .3], [.6, 3., .2], [.7, 2., .1]])
    b = a + np.array([.02, .2, .01])
    y = np.array([[1., 1., .1], [1., 2., 0.], [0., 1., .1]])
    return diagnose(a, b, y, np.array(['a', 'a', 'b']), np.array(['r', 'r', 's']), np.array([1, 2, 1]))['source_balanced']


def test_independent_formula_verifies_diagnostic():
    assert check_node(node()) > 0


def test_expected_weighted_objective_matches_actual_torch_loss():
    from src.world_model.m3w_easy_hurdle import losses, torch
    a = np.array([[.4, 2., .3], [.6, 3., .2], [.7, 2., .1], [.2, 3., .2]])
    y = np.array([[1., 1., .1], [1., 2., 0.], [0., 1., .1], [1., 3., .2]])
    w, q = weights(np.array(['a', 'a', 'a', 'b']), np.array(['r', 'r', 'r', 's']),
                   np.array([1, 1, 2, 1]), np.ones(4, bool))
    error = compose(a)-y[:, 0]*(y[:, 2]-.02*y[:, 1])
    expected = objective(error, w, q)['marginal']
    # Two queries from source a and two copies of source b's only query.
    ix = np.array([0, 1, 2, 3, 3]); segments = torch.tensor([0, 0, 1, 2, 3])
    pred = np.column_stack((a, compose(a), np.log(a[:, 0]/(1-a[:, 0]))))
    actual = losses(torch.from_numpy(pred[ix]), torch.from_numpy(y[ix]), segments, 4)['marginal']
    assert float(actual) == pytest.approx(expected, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize('field', ['attribution', 'objective', 'partition', 'empty'])
def test_rejects_changed_arithmetic(field):
    d = copy.deepcopy(node())
    if field == 'attribution': d['attribution']['marginal'][0] += .01
    if field == 'objective': d['transplants']['3']['marginal'] += .01
    if field == 'partition': d['arms']['uncapped']['partitions']['easy_zero_harm']['MSE_contribution'] += .01
    if field == 'empty': d['arms']['uncapped']['partitions']['all']['weight_mass'] = 0
    with pytest.raises(AssertionError): check_node(d)
