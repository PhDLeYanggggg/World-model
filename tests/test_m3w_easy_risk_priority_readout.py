import numpy as np
import pytest

from scripts.verify_m3w_easy_risk_priority import check_actions, check_contrasts
from scripts.report_m3w_easy_risk_priority import interpret
from src.world_model import m3w_easy_risk_priority as repair


def actions():
    ids = np.arange(4)
    scores = np.tile([.5, 1., 0., -.01], (4, 1))
    risk = np.tile([-1., -.01], (4, 1))
    a, _ = repair.decisions(ids+1., {arm: risk for arm in ('raw', *repair.ARMS)},
        np.ones(4, bool), ['r']*4, np.ones(4), ids, 256)
    a.update(ids=ids, eligible=np.ones(4, bool), raw_scores=risk,
             uncapped_scores=scores.copy(), risk_priority_scores=scores.copy())
    return a


def test_query_constraints_detect_count_and_factorization_tampering():
    a = actions()
    assert check_actions(a, ['s']*4, ['r']*4, np.ones(4), np.arange(4)+1.) == (6, 0)
    a['risk_priority_matched'][0] = False
    with pytest.raises(AssertionError):
        check_actions(a, ['s']*4, ['r']*4, np.ones(4), np.arange(4)+1.)
    a = actions(); a['risk_priority_scores'][0, 3] = .1
    with pytest.raises(AssertionError):
        check_actions(a, ['s']*4, ['r']*4, np.ones(4), np.arange(4)+1.)


def test_empty_eligibility_stays_abstention_not_artificial_intervention():
    a = actions(); a['eligible'][:] = False
    for key, value in a.items():
        if value.dtype == bool:
            value[:] = False
    assert check_actions(a, ['s']*4, ['r']*4, np.ones(4), np.arange(4)+1.) == (6, 0)


def test_contrast_sign_and_undefined_denominator_checked_independently():
    left = dict(error_sum=9., easy_gain_floor=1., hard_gain_floor=3.,
        selected_positive_harm_ratio=None, positive_harm_over_all_floor=.01,
        intervention_rate=0.)
    right = dict(error_sum=10., easy_gain_floor=0., hard_gain_floor=1.,
        selected_positive_harm_ratio=.02, positive_harm_over_all_floor=.02,
        intervention_rate=0.)
    reported = dict(ADE_gain_percent=10., easy_gain_floor_difference_pp=1.,
        hard_gain_floor_difference_pp=2., selected_harm_reduction_pp=None,
        all_reference_harm_reduction_pp=1., intervention_difference_pp=0.)
    check_contrasts(reported, left, right)
    reported['selected_harm_reduction_pp'] = 0.
    with pytest.raises(AssertionError):
        check_contrasts(reported, left, right)
    reported['selected_harm_reduction_pp'] = None
    reported['ADE_gain_percent'] = -10.
    with pytest.raises(AssertionError):
        check_contrasts(reported, left, right)


@pytest.mark.parametrize('ade,screen,expected', [
    ([.01, .02], False, 'advantage_but_screen_failed'),
    ([.01, .02], True, 'exploratory_screen_pass_only'),
    ([-.02, -.01], False, 'negative_primary'),
    ([-.01, .01], False, 'no_resolved_primary_advantage'),
    (None, False, 'undefined_primary')])
def test_report_does_not_promote_partial_or_undefined_results(ade, screen, expected):
    summary = dict(paired={'risk_priority_matched_vs_uncapped_matched': {
        'ADE_gain_percent': dict(ci95=ade)}}, gates=dict(exploratory_screen_pass=screen))
    result = interpret(summary)
    assert result['verdict'] == expected
    assert not result['deployment_changed'] and not result['independent_confirmation']
    assert not result['formal_primary_replaced'] and not result['submission_ready']
