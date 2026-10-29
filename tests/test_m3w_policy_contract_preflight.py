import numpy as np
import pytest

from src.evaluation import m3w_policy_contract_preflight as api
from src.world_model import m3w_scaled_risk_readout as readout


def rows(count):
    return [dict(group='g',source='s',seed=17,anchor='original',selected_count=count)]


def test_actual_matching_forces_empty_risk_even_for_a_safe_full_candidate():
    target = np.tile([.1,0.,1.,1.,0.],(2,1))
    predictions = {arm:target.copy() for arm in readout.ARMS}
    predictions['original'][:,0] = 0
    action = readout.decisions(predictions,np.ones(2,bool),np.ones(2,bool),
        np.ones(2,int),np.zeros(2,int),np.arange(2))
    assert action['budget_scaled'].all()
    assert not action['budget_scaled_matched_original'].any()
    assert readout.policy_score(target,action['budget_scaled'],np.ones(2))['finite_completion_supported']
    matched = readout.policy_score(target,action['budget_scaled_matched_original'],np.ones(2))
    assert matched['easy_selected_risk_upper'] is None and not matched['finite_completion_supported']
    with pytest.raises(ValueError): api.require_no_structural_obstruction(rows(0),require_matched_finite_risk=True)


def test_nonempty_does_not_prove_success_or_change_budget():
    result = api.require_no_structural_obstruction(rows(2),require_matched_finite_risk=True)
    assert not result['scientific_success'] and result['full_policy_safety_unchanged']


def test_separating_comparison_from_deployment_does_not_call_empty_safe():
    result = api.require_no_structural_obstruction(rows(0),require_matched_finite_risk=False)
    assert result['forced_empty_comparisons'] and not result['empty_risk_is_zero']
    assert not result['scientific_success']


@pytest.mark.parametrize('count',[-1,True,1.2])
def test_invalid_counts_fail_closed(count):
    with pytest.raises(ValueError): api.assess_count_contract(rows(count),require_matched_finite_risk=True)


def test_duplicate_roster_cannot_inflate_evidence():
    with pytest.raises(ValueError): api.assess_count_contract(rows(2)*2,require_matched_finite_risk=True)


def prospective(policy=None, interval=None):
    safe = dict(selected_count=3,all_selected_risk_upper=.01,easy_selected_risk_upper=.01,
                easy_degradation_upper=0.,finite_completion_supported=True)
    if policy: safe.update(policy)
    return api.prospective_advancement({'g':safe},{'matched':[.1,.2] if interval is None else interval},
                                       expected_views=['g'],expected_contrasts=['matched'])


def test_prospective_pass_is_not_independent_scientific_success():
    result = prospective()
    assert result['development_advancement']
    assert not result['scientific_success'] and not result['deployment_changed']


@pytest.mark.parametrize('policy',[
    {'selected_count':0}, {'easy_selected_risk_upper':None},
    {'easy_selected_risk_upper':.020001}, {'all_selected_risk_upper':.03},
    {'easy_degradation_upper':.03}, {'finite_completion_supported':False}])
def test_matched_utility_cannot_rescue_unsafe_full_policy(policy):
    assert not prospective(policy)['development_advancement']


def test_zero_utility_does_not_become_positive_evidence():
    assert not prospective(interval=[0.,0.])['development_advancement']


def test_no_silent_removal_of_failed_comparators():
    with pytest.raises(ValueError):
        api.prospective_advancement({}, {}, expected_views=['missing'],expected_contrasts=['all'])


def test_nonfinite_risk_cannot_pass():
    with pytest.raises(ValueError): prospective({'easy_selected_risk_upper':float('nan')})
