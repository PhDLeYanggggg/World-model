import json
import numpy as np
from scripts import report_m3w_selected_query_critic as report
from src.evaluation.m3w_metric_bundle import verify


def test_training_all36_real_heads_and_exact_trace_reduction():
    freeze, fits = report.training(); p = report.run.PUBLIC
    traces = json.loads((p/'training_trace_metrics.json').read_text())
    assert traces['training_freeze_sha256'] == report.run.sha(p/'training_freeze.json')
    assert len(traces['fits']) == len(fits) == 36
    assert not traces['raw_data_or_weights_included']
    assert {f['step'] for f in fits} == {2000}
    text, stats = report.training_text(freeze, fits, traces['fits'])
    assert (p/'training_report.md').read_text() == text
    assert json.loads((p/'training_diagnostics.json').read_text()) == stats
    assert stats['primary_optimizer_updates'] == 72000 and stats['new_trajectory_forecaster_fits'] == 0
    for arm in report.run.api.ARMS:
        s = stats['arms'][arm]
        assert s['heads'] == 18 and s['gradient_snapshots'] == 738
        assert s['unknown_rows_with_loss'] == 0 and s['unknown_rows_drawn'] > 0
        assert not stats['independent_confirmation'] and not stats['submission_ready']


def test_training_pair_initialization_draws_and_hashes_are_identical():
    _, fits = report.training()
    for context, seed in {(r['context'], r['seed']) for r in fits}:
        pair = [r for r in fits if (r['context'], r['seed']) == (context, seed)]
        assert len(pair) == 2 and pair[0]['draw_hash'] == pair[1]['draw_hash']
        assert pair[0]['input_hashes'] == pair[1]['input_hashes']
        np.testing.assert_array_equal(pair[0]['rms'], pair[1]['rms'])
    for arm in report.run.api.ARMS:
        assert json.loads((report.run.PUBLIC/('first_full_refit_'+arm+'.json')).read_text())['scientific_state_exact']


def test_complete_policy_evidence_and_legacy_controls():
    value = report.build(); s = value['summary']
    assert len(value['rows']) == 72 and s['fresh_critic_fits'] == 36
    assert s['critic_training_updates'] == 72000 and s['new_trajectory_forecaster_fits'] == 0
    assert all(r['parent_controls_exact'] and r['paired_training_verified'] for r in value['rows'])
    assert all(r['result']['original_benefit_references_unchanged'] for r in value['rows'])
    assert s['safety']['row_matched']['selected'] == s['safety']['set_matched']['selected']
    assert s['matched_controls_utility_only'] and s['empty_comparison_not_estimable']
    risk = s['safety']['set_joint']
    assert s['fitting_gates']['complete_context_recording_risk'] == (
        risk['supported_record_views'] == risk['required_record_views'] and risk['supported_views'] == 72)
    assert not s['independent_confirmation'] and not s['deployment_changed'] and not s['submission_ready']


def test_complete_plain_report_and_byte_verified_aggregate_bundle():
    value = report.build(); p = report.run.PUBLIC
    assert (p/'readout.md').read_text() == report.render(value)
    result = verify(report.ROOT, p)
    assert result['groups'] == 72 and result['bytes_verified'] and not result['raw_data_or_weights_included']
    receipt = json.loads((p/'evidence.json').read_text())
    assert receipt['report_code_sha256'] == report.run.sha(report.ROOT/'scripts/report_m3w_selected_query_critic.py')
    assert receipt['report_sha256'] == report.run.sha(p/'readout.md')
