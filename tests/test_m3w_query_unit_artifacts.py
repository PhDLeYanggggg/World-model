import json
from scripts import report_m3w_query_unit_critic as report
from src.evaluation.m3w_metric_bundle import verify


def test_training_all36_pairs_preserve_original_design_and_common_units():
    freeze, fits, traces = report.training(); p = report.run.PUBLIC
    assert len(fits) == len(traces) == 36
    assert freeze['fresh_optimizer_updates'] == 72000 and freeze['exact_refit_updates'] == 4000
    assert freeze['cached_verified_control_heads'] == 36
    saved = json.loads((p/'training_trace_metrics.json').read_text())
    assert saved['training_freeze_sha256'] == report.run.sha(p/'training_freeze.json')
    assert saved['traces'] == traces and not saved['raw_data_or_weights_included']
    for pair in traces:
        old, new = report.common_curve(pair['old'], 'old'), report.common_curve(pair['new'], 'new')
        assert old['steps'] == new['steps'] and old['steps'][-1] == 2000
        for key in ('pointwise', 'query', 'original_objective'): assert old[key][0] == new[key][0]
        assert pair['unknown_rows_drawn'] > 0 and pair['unknown_rows_with_loss'] == 0
        assert pair['query_normalizer']['fitting_labels_only']
        assert not pair['query_normalizer']['inference_feature']


def test_training_report_reconstructs_all_real_checkpoint_telemetry():
    freeze, _, traces = report.training(); stats = report.training_stats(traces); p = report.run.PUBLIC
    assert json.loads((p/'training_diagnostics.json').read_text()) == stats
    assert (p/'training_report.md').read_text() == report.training_text(freeze, stats)
    for pair in stats['arms'].values():
        assert pair['unknown_rows_with_loss'] == 0
        assert {v['gradient_snapshots'] for v in pair['modes'].values()} == {738}
        assert {v['heads'] for v in pair['modes'].values()} == {18}
    assert not stats['independent_confirmation'] and not stats['deployment_changed']


def test_complete_query_unit_results_preserve_original_controls_and_risk():
    value = report.build(); s = value['summary']; risk = s['safety']['set_joint']
    assert len(value['rows']) == s['groups'] == 72
    assert s['cached_verified_critic_controls'] == 36
    assert all(r['old_critic_controls_exact'] and r['paired_training_verified'] for r in value['rows'])
    assert s['fitting_gates']['complete_recording_risk'] == (
        risk['supported_record_views'] == risk['required_record_views'] and risk['supported_views'] == 72)
    assert s['safety']['row_matched']['selected'] == s['safety']['set_matched']['selected']
    assert s['matched_controls_utility_only'] and s['empty_comparison_not_estimable']
    assert not s['independent_confirmation'] and not s['deployment_changed'] and not s['submission_ready']


def test_complete_report_and_aggregate_bytes_match_frozen_readout():
    value = report.build(); p = report.run.PUBLIC
    assert (p/'readout.md').read_text() == report.render(value)
    bundle = verify(report.ROOT, p)
    assert bundle['groups'] == 72 and bundle['bytes_verified']
    assert not bundle['raw_data_or_weights_included']
    receipt = json.loads((p/'evidence.json').read_text())
    assert receipt['report_sha256'] == report.run.sha(p/'readout.md')
    assert receipt['report_code_sha256'] == report.run.sha(report.ROOT/'scripts/report_m3w_query_unit_critic.py')
