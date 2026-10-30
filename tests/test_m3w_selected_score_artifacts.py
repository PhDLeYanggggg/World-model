import json
import numpy as np
from scripts import report_m3w_selected_scores as report
from src.evaluation.m3w_metric_bundle import verify


def test_complete_score_roster_preserves_original_and_final_query_populations():
    value = report.build(); variants = value['summary']['variants']
    assert len(variants) == 8 and value['summary']['groups'] == 72
    for name, row in variants.items():
        counts = row['counts']
        assert counts['rows'] == 897060 and counts['queries'] == 112596
        assert sum(counts[k] for k in ('complete_selected_queries', 'unknown_selected_queries', 'empty_selected_queries')) == counts['queries']
        if name.endswith('_original'):
            assert counts['selected_rows'] == 265671 and counts['selected_unknown_rows'] == 4098
            assert counts['complete_selected_queries'] == 58970
            assert counts['unknown_selected_queries'] == 3294 and counts['empty_selected_queries'] == 50332
            assert counts['changed_rows'] == 0
            m = row['moments']['mean_score_squared_error']
            assert m['supported_views'] == 66 and m['supported_localities'] == 11
        for m in row['moments'].values(): assert m['conditional_support_only_not_full_cohort_effect']


def test_complete_arithmetic_is_not_an_inference_feature_or_risk_certificate():
    value = report.build(); s = value['summary']
    assert s['all288_original_selected_MSE_controls_close'] and s['no_new_model_or_policy']
    assert not s['independent_confirmation'] and not s['deployment_changed'] and not s['submission_ready']
    for row in s['variants'].values():
        m = row['moments']; error = m['mean_signed_score_underprediction']['mean']
        np.testing.assert_allclose(error, np.add(m['mean_harm_underprediction']['mean'],
            m['mean_reference_bias_contribution']['mean']), rtol=1e-12, atol=1e-12)
        assert row['max_abs_decomposition_error'] < 1e-12


def test_complete_report_and_metric_bundle_reconstruct_exactly():
    value = report.build(); p = report.run.PUBLIC
    assert (p/'readout.md').read_text() == report.render(value)
    bundle = verify(report.ROOT, p)
    assert bundle['groups'] == 72 and bundle['bytes_verified'] and not bundle['raw_data_or_weights_included']
    evidence = json.loads((p/'evidence.json').read_text())
    assert evidence['report_code_sha256'] == report.run.sha(report.ROOT/'scripts/report_m3w_selected_scores.py')
    assert evidence['report_sha256'] == report.run.sha(p/'readout.md')
