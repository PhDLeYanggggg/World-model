from scripts import report_m3w_european_nested_calibration as report


def test_summary_keeps_unsafe_undefined_and_zero_intervention():
    m = dict(indexed_rows=4, supported_rows=0, unknown_rows=4,
        equal_scene_gain_percent=None, worst_scene_gain_percent=None,
        scene_bootstrap_ci95=None, by_scene={'a': {'rows': 0}})
    a = dict(result_source='fixture', source_rows=4,
        policies={'fixture': dict(ADE_vs_CV=m, safety_observed_pass=False, switch_rate=0.)},
        neural_vs_damping={'fixture': {'all': m}}, calibration_vs_none={})
    got = report.summarize(a)
    assert got['policies']['fixture']['ADE_vs_CV']['equal_scene_gain_percent'] is None
    assert not got['policies']['fixture']['safety_observed_pass']
    assert got['policies']['fixture']['switch_rate'] == 0
    assert not got['calibrated_safety'] and not got['deployment_changed']
