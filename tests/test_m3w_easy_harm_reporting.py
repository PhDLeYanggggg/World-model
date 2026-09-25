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
