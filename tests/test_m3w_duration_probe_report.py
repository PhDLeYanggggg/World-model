from scripts.report_m3w_feature_duration_probe import shown_median


def test_absent_selected_target_mass_is_reported_not_estimable_not_zero():
    data = {'selected_actual_interpolation': {'EH': {'quantiles_0_25_50_75_100': None}}}
    assert shown_median(data, 'EH') == 'not estimable'


def test_defined_median_is_preserved_without_support_fabrication():
    data = {'selected_actual_interpolation': {'EH': {'quantiles_0_25_50_75_100': [0, .1, .2, .4, 1]}}}
    assert shown_median(data, 'EH') == '0.200000'
