import numpy as np
import pytest

from src.evaluation.m3w_dual_reference_capacity import action_masses
from src.evaluation.m3w_missing_debit_capacity import ARMS, evidence, variant_masses


@pytest.mark.parametrize('arm', tuple(ARMS))
def test_diagnostic_masses_match_outcome_and_only_risk_switch_affects_support(arm):
    cv = np.zeros((3, 2, 2))
    ols = cv.copy()
    ols[:, :, 0] = 1.
    y = cv.copy()
    y[:, :, 0] = 4.
    valid = np.array([[True, True], [False, False], [True, False]])
    bank = (cv+np.array([2., 0.]))[:, None]
    w = np.full(3, 1./3)
    cut = dict(easy=5., hard=8.)
    base = action_masses(cv, ols, bank, y, valid, w, cut['easy'])
    m = variant_masses(base, cv, ols, bank, y, valid, w, arm)
    # Take a labeled easy row and an unknown row; do not change the partial row.
    take = m['row'] < 2
    pred = cv.copy()
    pred[:2] = bank[:2, 0]
    got = evidence(cv, ols, pred, y, valid, 'r', cut, w, arm)
    np.testing.assert_allclose(got['gains'], m['offset']+m['utility'][:, take].sum(1), atol=1e-10)
    assert got['supported'] is (not ARMS[arm][0])
    assert m['easy_cap'] == base['easy_cap']
    np.testing.assert_array_equal(m['easy'], base['easy'])


def test_omit_both_still_requires_observed_selected_easy_support():
    cv = np.zeros((2, 2, 2))
    pred = cv.copy()
    pred[1] += 1.
    valid = np.array([[True, False], [False, False]])
    out = evidence(cv, cv, pred, cv+2., valid, 'r', dict(easy=5., hard=8.),
                   np.full(2, .5), 'omit_both_debits')
    assert not out['supported']
