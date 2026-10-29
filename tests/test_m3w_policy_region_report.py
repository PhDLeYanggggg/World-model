from scripts.report_m3w_policy_region_cost import interval_text, render, NAME


def test_undefined_interval_is_not_a_zero_or_safety_pass():
    assert interval_text(dict(point=None, CI95=None)) == 'not_estimable'
    assert interval_text(dict(point=0., CI95=None)) == 'not_estimable'
    assert interval_text(dict(point=-1., CI95=[-2., 0.])) == '-1.000000 [-2.000000,0.000000]'


def test_report_keeps_training_calibration_independence_and_gate_failures_separate():
    values = {f'{mode}_{kind}_pp': dict(point=0., CI95=[-1., 1.])
              for mode in ('full', 'matched') for kind in ('known', 'lower', 'upper')}
    value = dict(policies=[], region_minus_parent={'inner_held': values},
        inner_held_mixed_minus_physical={NAME: values},
        training={NAME: dict(fits=54, updates=108000, seconds=2., peak_RSS_bytes=100)},
        unchanged_primary_full_roster_budget_supported=False,
        unchanged_neural_increment_conditional_CI_positive=False)
    text = render(value)
    assert 'gate: False' in text
    assert 'neural-increment CI: False' in text
    assert 'not safe zero' in text
    assert 'not independent confirmation' in text
    assert 'fitting-region mechanism check is separate' in text
    assert 'three seeds refer to controller heads' in text
