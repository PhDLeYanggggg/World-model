import pytest
from scripts.report_m3w_european_selected_risk_learning import summarize, fmt, fitting_and_transport_diagnostics


def test_incomplete_source_study_cannot_be_summarized_as_complete():
    with pytest.raises(ValueError): summarize([], {})


def test_missing_interval_is_not_zero_gain():
    assert fmt(None) == 'undefined'
    assert fmt([-1., 2.]) == '-1.00000 to +2.00000'


def test_descriptive_transport_sums_bins_and_excludes_zero_mass():
    training = [dict(identity={'path': str(i)}, arm=arm, fit={'trace': [dict(
        moment_mse=2. if arm == 'mean' else 1., selected_group_mse=1.)]})
        for i in range(36) for arm in ('mean', 'selected')]
    bins = {'0': dict(true_selected_harm=1., predicted_selected_harm=0.),
            '1': dict(true_selected_harm=3., predicted_selected_harm=3.)}
    zero = {'0': dict(true_selected_harm=0., predicted_selected_harm=1.)}
    rows = {'full': [{'diagnostics': {arm: {'a': bins, 'b': zero}
        for arm in ('raw', 'mean', 'selected')}}]}
    result = fitting_and_transport_diagnostics(training, rows)
    assert result['fixed_training_batch']['moment_mse']['selected_lower_count'] == 36
    d = result['source_C_selected_harm']['full']['selected']
    assert d['supported_locality_views'] == 1
    assert d['median_predicted_over_actual_selected_harm'] == .75
    assert d['underestimated_locality_views'] == 1
    assert not result['changed_decisions']


def test_unknown_future_budget_accounting_never_changes_action():
    import numpy as np
    from scripts.audit_m3w_selected_query_budget import account
    m = np.array([[1., .03, 1., .03], [1., 0., 1., 0.]])
    y = np.array([[1., .05, 1., .05], [np.nan]*4])
    bits = np.array([True, False]); saved = bits.copy()
    row = account(m, y, bits, [np.array([0, 1])], np.array(['a', 'a']))['a']['events']['easy']
    assert row['unknown_predicted_mass_fraction'] == .5
    assert row['supported_spend_exceeds_supported_predicted_budget'] == 1
    assert row['actual_harm_ratio'] == .05
    np.testing.assert_array_equal(bits, saved)
