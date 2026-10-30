import json
from scripts import report_m3w_current_policy_critic as report
from src.evaluation.m3w_metric_bundle import verify


def test_training_36_current_heads_match_fixed_controls_and_preserve_reference_units():
    freeze, fits, pairs = report.training(); p = report.run.PUBLIC
    assert len(fits) == len(pairs) == 36
    assert freeze['fresh_optimizer_updates'] == 72000 and freeze['exact_refit_updates'] == 4000
    assert freeze['cached_verified_fixed_query_heads'] == 36
    published = json.loads((p/'training_trace_metrics.json').read_text())
    assert published['pairs'] == pairs and not published['raw_data_or_weights_included']
    assert published['training_freeze_sha256'] == report.run.sha(p/'training_freeze.json')
    for pair in pairs:
        a, b = report.comparable_curve(pair['fixed'], 'fixed'), report.comparable_curve(pair['current'], 'current')
        assert a['steps'] == b['steps'] and a['steps'][-1] == 2000
        assert a['frozen_probe'][0] == b['frozen_probe'][0] and a['pointwise'][0] == b['pointwise'][0]
        assert a['moving_query_term'] is None and b['moving_query_term'] is not None
        assert pair['unknown_rows_drawn'] > 0 and pair['unknown_rows_with_loss'] == 0


def test_training_report_does_not_equate_moving_loss_with_common_population_error():
    freeze, _, pairs = report.training(); stats = report.training_stats(pairs); p = report.run.PUBLIC
    assert (p/'training_report.md').read_text() == report.training_text(freeze, stats)
    assert json.loads((p/'training_diagnostics.json').read_text()) == stats
    assert stats['moving_loss_not_common_population_error'] and stats['original_set_probe_comparable']
    for arm in stats['arms'].values():
        assert arm['unknown_rows_with_loss'] == 0
        assert arm['roles']['fixed']['median_final_moving_masked_query_term'] is None
        assert arm['roles']['current']['gradient_snapshots'] == 738
    assert not stats['independent_confirmation'] and not stats['deployment_changed']


def test_complete_current_query_primary_and_original_risk_remain_distinct():
    value = report.build(); s = value['summary']; risk = s['safety']['set_joint']
    assert s['groups'] == 72 and s['cached_verified_fixed_query_heads'] == 36
    assert s['fitting_gates']['complete_recording_risk'] == (
        risk['supported_record_views'] == risk['required_record_views'] and risk['supported_views'] == 72)
    assert s['safety']['row_matched']['selected'] == s['safety']['set_matched']['selected']
    assert not s['independent_confirmation'] and not s['deployment_changed'] and not s['submission_ready']
    assert s['empty_comparison_not_estimable'] and s['matched_controls_utility_only']


def test_complete_current_query_report_and_aggregate_bundle_are_exact():
    value = report.build(); p = report.run.PUBLIC
    assert (p/'readout.md').read_text() == report.render(value)
    bundle = verify(report.ROOT, p)
    assert bundle['groups'] == 72 and bundle['bytes_verified'] and not bundle['raw_data_or_weights_included']
    evidence = json.loads((p/'evidence.json').read_text())
    assert evidence['report_code_sha256'] == report.run.sha(report.ROOT/'scripts/report_m3w_current_policy_critic.py')
    assert evidence['report_sha256'] == report.run.sha(p/'readout.md')
