from scripts.report_m3w_duration_support_diagnosis import median_text


def test_undefined_mass_is_not_a_zero_share_or_zero_risk():
    value = dict(quantiles_0_25_50_75_100=None)
    assert median_text(value) == 'not estimable'
    assert median_text(value, percent=True) == 'not estimable'


def test_mass_share_and_ratio_have_distinct_display_units():
    value = dict(quantiles_0_25_50_75_100=[0, .25, .5, .75, 1])
    assert median_text(value) == '0.500000'
    assert median_text(value, percent=True) == '50.00%'


def test_tiny_nonzero_ratio_is_not_rounded_to_zero():
    value = dict(quantiles_0_25_50_75_100=[0, 0, 1.42e-8, 1, 2])
    assert median_text(value) == '1.42e-08'
