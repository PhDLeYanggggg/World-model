import pytest
from scripts.report_m3w_european_dual_event_bridge import gates, summarize


def test_engineering_and_predicted_constraints_cannot_unlock_deployment():
    a = {'dual_risk': {'easy_pass': 18, 'zero_CV_harm_views': 0}}
    s = {'a': {'dual_risk': {'CI': [1., 2.]}}}
    ab = {'a': {'easy_risk_only': {'all': {'CI': [1., 2.]}}}}
    g = gates(a, s, ab)
    assert g['dual_easy_preserved']
    for k in ('independent_calibration', 'independent_confirmation', 'deployment_changed',
              'submission_ready', 'stage5c_executed', 'smc_enabled'):
        assert not g[k]


def test_easy_failure_not_hidden_by_positive_average():
    a = {'dual_risk': {'easy_pass': 17, 'zero_CV_harm_views': 0}}
    s = {'a': {'dual_risk': {'CI': [-1., 2.]}}}
    ab = {'a': {'easy_risk_only': {'all': {'CI': [-1., 2.]}}}}
    g = gates(a, s, ab)
    assert not g['dual_easy_preserved']
    assert not g['dual_better_than_old_easy_add_all_six_seed_means']
    assert not g['dual_better_than_easy_risk_all_six_seed_means']


def test_all_groups_required():
    with pytest.raises(ValueError): summarize([])


def test_missing_seed_groups_cannot_pass_gain_gate():
    a = {'dual_risk': {'easy_pass': 18, 'zero_CV_harm_views': 0}}
    g = gates(a, {}, {})
    assert not g['dual_better_than_old_easy_add_all_six_seed_means']
    assert not g['dual_better_than_easy_risk_all_six_seed_means']
