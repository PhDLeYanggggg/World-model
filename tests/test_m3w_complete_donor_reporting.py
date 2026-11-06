import pytest

from scripts.report_m3w_moving_complete_donor_policy import category,validate


def view():
    return dict(known_all_risk=None,known_easy_risk=None,known_easy_net_degradation=0.,
                completion_all_risk_upper=None,completion_easy_risk_upper=None,supported=False)


def test_empty_reference_is_not_an_observed_violation_or_a_pass():
    assert category(view())=='missing_reference_support'


def test_observed_and_completion_failures_are_distinct():
    v=view();v['completion_all_risk_upper']=.1
    assert category(v)=='unknown_completion_bound_violation'
    v['known_easy_risk']=.03
    assert category(v)=='observed_harm_violation'


def test_no_partial_population_or_deployment_overclaim():
    a=dict(complete_own_B_rows=62466,rows=[{}]*8,identity_certification=False,new_outer_C_H_readout=False,
        independent_confirmation=False,deployment_changed=False,submission_ready=False,stage5c_executed=False,smc_enabled=False)
    validate(a)
    a['deployment_changed']=True
    with pytest.raises(ValueError):validate(a)
