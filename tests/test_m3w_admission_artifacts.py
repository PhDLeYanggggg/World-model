from scripts import report_m3w_admission_cost as report
from scripts import run_m3w_admission_cost as run
from src.evaluation.m3w_metric_bundle import verify


def test_all_real_heads_complete_and_reports_do_not_promote_failed_risk():
    value = report.build()
    assert len(value['traces']) == 54 and len(value['groups']) == 72
    assert value['training']['primary_optimizer_updates'] == 108000
    assert value['training']['exact_parent_controls'] == 36
    assert not value['summary']['fitting_gates']['complete_recording_risk']
    assert not value['summary']['independent_confirmation']
    assert not value['summary']['deployment_changed'] and not value['summary']['submission_ready']
    assert all(trace[-1]['step'] == 2000 for trace in value['traces'].values())


def test_report_is_plain_markdown_and_matches_live_verified_values():
    value = report.build()
    path = run.PUBLIC / 'readout.md'
    assert path.read_text().startswith('# Online selected-action loss')
    assert path.read_text() == report.render(value)
    assert '-0.075404' in path.read_text() and '308' in path.read_text()


def test_aggregate_bundle_contains_exact72groups_not_weights_or_source_rows():
    result = verify(run.ROOT, run.PUBLIC)
    assert result['groups'] == 72 and result['bytes_verified']
    assert not result['raw_data_or_weights_included']


def test_selected_attribution_has_no_imputed_unknown_targets_or_changed_risk_rule():
    value = report.build()
    upper = value['attribution']['upper_score']
    current = value['attribution']['admitted_score']
    assert upper['selected_unknown_rows'] == 3658 and current['selected_unknown_rows'] == 3636
    assert current['channels']['easy']['false_admitted_known_rows'] == 33143
    assert current['channels']['easy']['observed_harm_mass'] > 70 * current['channels']['easy']['predicted_harm_mass']
    for group in value['groups']:
        for part in group['result']['attribution'].values():
            assert not part['unknown_outcomes_imputed'] and not part['new_policy_constructed']
            assert part['risk_budget'] == .02
            for channel in part['channels'].values():
                assert sum(channel['witness_counts'].values()) == channel['false_admitted_known_rows']
                assert abs(channel['decomposition_residual']) < 1e-8
