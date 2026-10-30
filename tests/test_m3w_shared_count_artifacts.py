import json
import pytest
from scripts import report_m3w_shared_count_bank as report
from src.evaluation.m3w_metric_bundle import verify


def test_complete_comparison_preserves_counts_and_legacy_controls():
    value = report.build(); s = value['summary']
    assert len(value['rows']) == 72 and s['cached_verified_cost_heads'] == 18
    assert s['new_training_updates'] == 0
    assert s['query_views'] == 112596
    assert value['cohort']['rows'] == 149510
    assert value['cohort']['current_frame_queries'] == 18766
    assert value['cohort']['recordings'] == 74
    assert value['cohort']['repeated_row_views'] == 897060
    assert all(r['legacy_controls_exact'] for r in value['rows'])
    assert all(r['result']['same_count_every_query'] for r in value['rows'])
    counts = [s['safety'][a]['selected'] for a in report.run.api.ARMS]
    assert len(set(counts)) == 1
    for row in value['rows']:
        for a in ('physical_independent', 'physical_joint'):
            assert row['result']['policies'][a]['neural_interventions'] == 0
        assert row['result']['allocation']['predicted_bank_dominance_every_query']
        assert not row['result']['allocation']['labels_used']


def test_own_unknown_bound_and_original_risk_are_not_replaced():
    value = report.build(); s = value['summary']
    for row in value['rows']:
        contrast = row['result']['interaction_contrast']
        assert contrast['own_four_policy_bound'] and contrast['not_difference_of_lower_bounds']
        assert not contrast['causal_identification']
    assert s['common_count_comparison_not_main_deployment_repair']
    assert s['matched_controls_utility_only'] and s['empty_risk_not_estimable']
    safe = s['safety']['neural_joint']
    complete = (safe['supported_views'] == 72 and safe['supported_record_views'] == safe['required_record_views'])
    assert s['fitting_gates']['complete_primary_recording_risk'] == complete
    assert not s['independent_confirmation'] and not s['deployment_changed']
    assert not s['submission_ready'] and not s['stage5c_executed'] and not s['smc_enabled']


def test_complete_reports_and_exact_aggregate_archive():
    value = report.build(); p = report.run.PUBLIC
    assert (p/'readout.md').read_text() == report.render(value)
    receipt = json.loads((p/'evidence.json').read_text())
    assert receipt['summary_sha256'] == report.run.sha(p/'summary.json')
    assert receipt['report_sha256'] == report.run.sha(p/'readout.md')
    assert receipt['replay_runtime_sha256'] == report.run.sha(p/'replay_runtime.json')
    assert receipt['figure_sha256'] == report.run.sha(p/'figures/shared_count_summary.svg')
    assert receipt['locality_figure_sha256'] == report.run.sha(p/'figures/shared_count_by_locality.svg')
    result = verify(report.ROOT, p)
    assert result['groups'] == 72 and result['bytes_verified']
    assert not result['raw_data_or_weights_included']


def test_report_refuses_missing_full_replay(tmp_path, monkeypatch):
    p = report.run.PUBLIC
    for name in ('complete.json', 'summary.json'):
        (tmp_path/name).write_bytes((p/name).read_bytes())
    monkeypatch.setattr(report.run, 'PUBLIC', tmp_path)
    with pytest.raises(FileNotFoundError):
        report.build()
