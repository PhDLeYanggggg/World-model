import pytest
from scripts import report_m3w_frame_action_allocation as main
from scripts import report_m3w_frame_allocation_attribution as diagnosis
from src.evaluation.m3w_metric_bundle import verify


def test_complete_new_decoding_is_not_relabelled_as_new_neural_training():
    value = main.build(); s = value['summary']
    assert len(value['rows']) == 72 and s['cached_verified_cost_heads'] == 18
    assert s['new_training_updates'] == 0 and s['trained_models'] == 0
    assert s['query_views'] == 112596 and s['changed_primary_query_views'] == 36305
    assert s['safety']['independent']['selected'] == s['safety']['joint_matched']['selected'] == 265671
    assert all(r['result']['exact_count_per_query'] for r in value['rows'])


def test_positive_paired_utility_cannot_replace_failed_risk_or_hide_solver_fallback():
    value = main.build(); s = value['summary']
    assert s['comparisons']['joint_matched_minus_independent']['full_lower_pp']['CI95'][0] > 0
    assert s['primary_solver_nonoptimal_query_views'] == 8
    assert s['safety']['joint_matched']['known_easy_risk_violations'] == 324
    assert not s['fitting_advancement_gate'] and not s['independent_confirmation']
    assert not s['deployment_changed'] and not s['submission_ready']
    assert value['oracle_counts']['algebraic_budget_failures'] == 0
    assert value['oracle_counts']['defined_easy_support'] < value['oracle_counts']['queries']


def test_neural_reallocation_has_own_bound_but_is_not_reoptimized_physical_only():
    s = diagnosis.build()['summary']; m = main.build()['summary']
    assert s['all72legacy_independent_attributions_exact'] and s['all288frozen_action_hashes_exact']
    inc = s['joint_minus_independent_incremental_neural_use']
    assert inc['lower_pp']['CI95'][0] > 0
    assert inc['known_pp']['point']+s['neural_deleted_joint_minus_independent']['known_pp']['point'] == pytest.approx(
        m['comparisons']['joint_matched_minus_independent']['full_known_pp']['point'])
    assert all(not r['result']['arms']['joint_matched']['optimized_physical_only_comparator'] for r in diagnosis.build()['rows'])
    assert not s['causal_identification'] and not s['deployment_changed']


def test_plain_reports_and_both_exact_aggregate_bundles():
    for report in (main, diagnosis):
        value = report.build()
        assert (report.run.PUBLIC/'readout.md').read_text() == report.render(value)
        result = verify(report.ROOT, report.run.PUBLIC)
        assert result['groups'] == 72 and result['bytes_verified']
        assert not result['raw_data_or_weights_included']
