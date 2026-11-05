import pytest
from scripts.report_m3w_joint_harm import outcome, loss_arrays, require_complete, risk_diagnostics, run


def test_missing_primary_and_risk_failure_cannot_be_rescued_by_secondary():
    analysis = dict(nonempty_matched=dict(event_supervised=dict(completion_lower_pp=dict(CI95=None))),
        safety=dict(event_supervised_veto=dict(full_original_risk_pass=False)),
        secondary_full_strategy=dict(large_gain=999))
    assert 'not estimable, not passed' in outcome(analysis, dict(original_rules_pass=False))
    with pytest.raises(ValueError, match='override'):
        outcome(analysis, dict(original_rules_pass=True))


def test_positive_primary_is_not_enough_without_full_original_risk():
    analysis = dict(nonempty_matched=dict(event_supervised=dict(completion_lower_pp=dict(CI95=[1., 2.]))),
        safety=dict(event_supervised_veto=dict(full_original_risk_pass=False)))
    assert 'requirements fail' in outcome(analysis, dict(original_rules_pass=False))


def test_loss_plot_compares_common_loss_not_different_total_objectives():
    trace = [dict(step=1, regression=2., absolute_gain=4., total=100., occurrence=1.)]
    fits = [dict(arm='event_supervised', trace=trace)]*54
    _, common, occurrence = loss_arrays(fits, 'event_supervised')
    assert (common == 4.).all() and (occurrence == 1.).all()
    with pytest.raises(ValueError, match='all 54'):
        loss_arrays(fits[:53], 'event_supervised')


def test_reporting_rejects_incomplete_replay_and_changed_analysis():
    freeze = dict(complete_heads=108, all_before_validation=True)
    audit = dict(full108_complete=True, complete_matched_pairs=54)
    verification = dict(all108_policy_blocks_and_prediction_hashes_exact=True, analysis_sha256='a')
    fits = [dict(steps=2000, parameters=28489)]*108
    groups = [dict(rows=[1, 2])]*108
    with pytest.raises(ValueError, match='drop'):
        require_complete(freeze, audit, verification, 'a', fits, groups)
    with pytest.raises(ValueError, match='separate replay'):
        require_complete(freeze, audit, verification, 'changed', fits, groups)


def test_risk_diagnostic_keeps_undefined_views_and_signed_worst_not_absolute():
    records = dict(a=dict(known_easy_net_degradation=-.5, selected=1),
        b=dict(known_easy_net_degradation=.01, selected=2),
        c=dict(known_easy_net_degradation=None, selected=0))
    groups = [dict(rows=[dict(policies={a: dict(recordings=records) for a in run.POLICIES})])]
    result = risk_diagnostics(groups)['event_supervised_veto']
    assert result['required_record_views'] == 3
    assert result['undefined_full_known_easy_net_views'] == 1
    assert result['no_intervention_views'] == 1
    assert result['worst_defined_signed_easy_net_percent'] == 1.
