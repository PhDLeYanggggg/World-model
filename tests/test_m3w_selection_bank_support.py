import numpy as np
from src.evaluation.m3w_selection_bank_support import diagnose


def labels():
    y = np.zeros((3,7,5)); y[:,:,2:4] = 1.; y[:,:,1] = y[:,:,4] = .03
    return y


def test_weighted_unsafe_mixture_cannot_supply_original_easy_support():
    out = diagnose(labels(),np.ones((3,7),bool),np.array(['r','r','r']))
    assert out['bank_infeasible_record_views'] == 1
    assert out['recordings']['r']['reason'] == 'all_legal_known_easy_actions_exceed_original_budget'


def test_one_legal_known_easy_safe_action_supplies_support_but_is_not_causal_success():
    y = labels(); y[1,2,1] = y[1,2,4] = .01
    out = diagnose(y,np.ones((3,7),bool),np.array(['r','r','r']))
    assert not out['bank_infeasible_record_views']
    assert out['future_informed_diagnostic_not_causal_model'] and not out['oracle_decisions_exported']
    assert out['recordings']['r']['safe_known_easy_action_occurrences'] == 1


def test_alias_only_easy_rows_and_wholly_unknown_rows_do_not_supply_known_support():
    y = labels(); y[2] = np.nan; legal = np.ones((3,7),bool); legal[0] = False
    out = diagnose(y,legal,np.array(['alias','other','unknown']))
    assert out['recordings']['alias']['reason'] == 'no_genuine_legal_easy_action'
    assert out['recordings']['unknown']['reason'] == 'no_positive_known_easy_reference'
    assert out['bank_infeasible_record_views'] == 3
