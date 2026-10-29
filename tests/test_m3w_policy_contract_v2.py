import copy

import pytest

from src.evaluation import m3w_policy_contract_preflight as legacy
from src.evaluation import m3w_policy_contract_v2 as api


def policy():
    return dict(selected_count=3, all_selected_risk_upper=.01,
                easy_selected_risk_upper=.01, easy_degradation_upper=-.03,
                finite_completion_supported=True)


def run(p):
    return api.prospective_advancement({'g': p}, {'matched': [.1, .2]},
                                      expected_views=['g'], expected_contrasts=['matched'])


def test_signed_improvement_reproduces_v1_bug_and_v2_fix():
    p = policy()
    before = copy.deepcopy(p)
    with pytest.raises(ValueError):
        legacy.prospective_advancement({'g': p}, {'matched': [.1, .2]},
                                      expected_views=['g'], expected_contrasts=['matched'])
    result = run(p)
    assert result['development_advancement']
    assert result['original_easy_degradation_upper']['g'] == -.03
    assert p == before
    assert not result['scientific_success'] and not result['legacy_results_rescored']


@pytest.mark.parametrize('value', [None, .020001])
def test_undefined_or_excess_degradation_still_fails(value):
    assert not run({**policy(), 'easy_degradation_upper': value})['development_advancement']


@pytest.mark.parametrize('key', ['all_selected_risk_upper', 'easy_selected_risk_upper'])
def test_negative_positive_harm_ratio_still_invalid(key):
    with pytest.raises(ValueError):
        run({**policy(), key: -.01})


@pytest.mark.parametrize('value', [float('nan'), float('inf'), True])
def test_malformed_degradation_fails_closed(value):
    with pytest.raises(ValueError):
        run({**policy(), 'easy_degradation_upper': value})
