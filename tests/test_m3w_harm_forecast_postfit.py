import pytest
from src.evaluation.m3w_harm_forecast_postfit import recording_budget_counts, paired_fit_losses


def row(risk, flag, net, unknown=0):
    return dict(arm='new', subset='all', reference='CV', known_positive_harm_percent=risk,
        known_budget_pass=flag, known_net_degradation_percent=net, unknown_rows=str(unknown),
        complete_population_budget_claim='False',
        support_status='not_estimable' if risk == '' else 'supported_labels_only')


def test_empty_risk_not_a_pass_and_net_benefit_does_not_cancel_harm():
    summary = recording_budget_counts([row('', '', ''), row('3', 'False', '-10'),
                                      row('2', 'True', '2', 1)])['new/all/CV']
    assert summary['required_views'] == 3
    assert summary['known_budget_passes'] == 1
    assert summary['known_budget_failures'] == 1
    assert summary['not_estimable'] == 1
    assert summary['views_with_unknown_labels'] == 1
    assert not summary['complete_policy_safety_certified']
    with pytest.raises(ValueError):
        recording_budget_counts([row('', 'True', '')])
    with pytest.raises(ValueError):
        recording_budget_counts([row('3', 'True', '-10')])


def test_compare_native_loss_only_and_retain_fixed_training_budget():
    shared = dict(step=4000, total_draws=256000, unique_training_rows=10, held_rows_sampled=0)
    old = dict(shared, losses=[dict(step=4000, loss=1.)])
    new = dict(shared, losses=[dict(step=4000, loss=20., native_loss=2.,
                selected_positive_harm=.2, cv_positive_harm=.1)])
    freeze = dict(fits={'a':dict(fit=new)}, identity=dict(controls={'a':dict(fit=old)}))
    result = paired_fit_losses(freeze)[0]
    assert result['new_native_ADE'] == 2.
    assert result['old_native_ADE'] == 1.
    assert not result['whole_training_loss_claim']
    new['held_rows_sampled'] = 1
    with pytest.raises(ValueError):
        paired_fit_losses(freeze)
