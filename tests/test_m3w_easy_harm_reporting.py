import pytest
from scripts.report_m3w_european_easy_harm_sampling import span, summarize, diagnostics


def test_missing_or_invalid_span_not_zero_gain():
    assert span([]) is None
    assert span([None, 1.]) is None
    assert span([float('nan'), 1.]) is None
    assert span([3., -1.]) == [-1., 3.]


def test_incomplete_sampling_readout_cannot_be_complete():
    with pytest.raises(ValueError): summarize([], {})
    with pytest.raises(ValueError): diagnostics([], {})


def test_exposure_and_selected_mass_have_distinct_denominators():
    fit = dict(trace=[dict(moment_mse=2.)], base_positive_probability=.01,
        sampled_positive_probability=.505, positive_easy_harm_draws=50, total_draws=100)
    training = [dict(pair=p, fit=fit, control_fit=dict(trace=[dict(moment_mse=4.)]))
        for p in ('full','motion_only') for _ in range(18)]
    event = dict(actual_selected_harm=10., predicted_selected_harm_supported=2.,
        actual_harm_ratio=.1, predicted_harm_ratio=.02, unknown_predicted_mass_fraction=.05)
    rows = {p:[dict(group='a',query_budget={'x':dict(events={'easy':event})})]
        for p in ('full','motion_only')}
    result = diagnostics(training, rows)
    assert result['pairs']['full']['fixed_batch_mse_ratio_median'] == .5
    assert result['pairs']['full']['empirical_positive_draw_fraction'] == [.5,.5]
    assert result['pairs']['full']['selected_easy_harm_supported_ratio_median'] == .2
    assert not result['decisions_changed']


def test_population_audit_keeps_unknown_out_and_fixed_selection():
    import numpy as np
    from scripts.audit_m3w_easy_harm_fitting import summarize_fit
    y = np.array([[2.,1.,1.,.5],[4.,0.,0.,0.],[np.nan]*4])
    pred = np.array([[2.,1.,1.,.25],[4.,0.,0.,.25],[100.]*4])
    p = np.array([.5,.5,0.]); selected = np.array([True,False,True])
    out = summarize_fit(pred,y,p,1.,np.ones(4),selected)
    assert out['population']['easy_harm_fitted_over_actual'] == 1.
    assert out['old_raw_selected']['easy_harm_fitted_over_actual'] == .5
    assert out['old_raw_selected']['rows'] == 1
    assert out['population']['component_mse'][-1] == .0625


def test_matched_transport_uses_equal_locality_evaluation_not_fit_weights():
    import numpy as np
    from scripts.audit_m3w_easy_harm_transport import evaluation_weights
    y=np.array([[1.]*4,[1.]*4,[1.]*4,[np.nan]*4])
    p=evaluation_weights(y,np.array(['a','a','b','b']))
    np.testing.assert_array_equal(p,[.25,.25,.5,0.])
    with pytest.raises(ValueError): evaluation_weights(y,np.array(['a','a','a','b']))
