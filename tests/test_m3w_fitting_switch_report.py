import copy
import numpy as np
import pytest

from scripts.report_m3w_fitting_switches import aggregate, verify_node
from src.world_model.m3w_fitting_switch_diagnostic import diagnose


def node():
    return diagnose(ids=np.arange(3), sites=np.array(['A']*3), recordings=np.array(['r']*3),
        frames=np.zeros(3, int), utility=np.array([1., 2., -1.]), moving=np.ones(3, bool),
        supported=np.ones(3, bool), risks={'raw': np.array([[-1., -1.], [1., 1.], [1., 1.]])},
        known=np.ones(3, bool), easy=np.zeros(3), reference=np.ones(3),
        benefit=np.array([0., 2., 3.]), harm=np.array([1., 0., 0.]))


def test_independent_accounting_checks_valid_node():
    assert verify_node(node()) > 10


@pytest.mark.parametrize('kind', ['partition', 'query', 'oracle'])
def test_corrupt_accounting_is_rejected(kind):
    n = node(); x = n['arms']['raw']
    if kind == 'partition': x['partition']['admitted']['benefit'] += 1
    elif kind == 'query': x['queries']['all']['undefined'] += 1
    else: x['ranking']['oracle_minus_utility_sum'] += 1
    with pytest.raises(AssertionError): verify_node(n)


def test_aggregate_does_not_convert_undefined_risk_to_zero():
    n = node(); z = aggregate([n, copy.deepcopy(n)])
    assert z['arms']['raw']['row_screen']['easy_selected_positive_harm_ratio'] == dict(mean=None, defined=0, total=2)
    assert z['rows'] == 6
    assert z['arms']['raw']['query_statuses']['easy']['undefined'] == 2
