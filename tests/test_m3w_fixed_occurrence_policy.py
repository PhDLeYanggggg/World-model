import numpy as np
import pytest
from scripts.run_m3w_fixed_occurrence_policy import causal_view, decisions, gates_for, parent


class Poison:
    def __array__(self, *args, **kwargs): raise AssertionError('Future was accessed')


def test_causal_view_never_touches_future():
    data = {k: np.arange(3) for k in parent.CAUSAL_KEYS}
    data.update(target_eval=Poison(), future_endpoint=Poison(), valid=Poison())
    assert set(causal_view(data)) == set(parent.CAUSAL_KEYS)


def test_count_matching_and_eligibility():
    q = np.array([[-.1, -.1], [-.1, .05], [-.1, .1], [-.1, -.1]])
    risks = dict(raw=q, trainable=q.copy(), fixed=q.copy())
    result, _ = decisions(np.array([1., 2., 3., 4.]), risks, np.array([True, True, True, False]),
                          np.array(['a']*4), np.array([1]*4), np.arange(4))
    assert result['common_anchor'].sum() == 1
    for arm in risks:
        assert result[arm+'_matched'].sum() == 1
        assert not result[arm+'_matched'][-1]


def example():
    paired = {'fixed_matched_vs_trainable_matched': dict(ADE_gain_percent={'ci95': [1., 2.]},
                                                       all_reference_harm_reduction_pp={'ci95': [.1, .2]})}
    rows = [dict(policy='fixed_matched', metric=dict(selected_positive_harm_ratio=.01,
                                                    easy_gain_CV=0., zero_CV_harmed=0))]
    contrasts = [dict(policy='fixed_matched_vs_trainable_matched', metric=dict(intervention_difference_pp=0.))]
    return paired, rows, contrasts


@pytest.mark.parametrize('field,value', [('selected_positive_harm_ratio', None),
    ('selected_positive_harm_ratio', .021), ('easy_gain_CV', -2.01), ('zero_CV_harmed', 1)])
def test_positive_accuracy_does_not_hide_risk_failure(field, value):
    a, b, c = example(); assert gates_for(a, b, c)['exploratory_screen_pass']
    b[0]['metric'][field] = value
    out = gates_for(a, b, c); assert not out['exploratory_screen_pass']
    assert not out['deployment_changed'] and not out['independent_confirmation']


def test_empty_and_unmatched_readout_do_not_pass():
    a, b, c = example()
    with pytest.raises(AssertionError): gates_for(a, [], c)
    c[0]['metric']['intervention_difference_pp'] = 1
    assert not gates_for(a, b, c)['matched_count']
