from scripts.report_m3w_score_support_calibration import interval


def test_undefined_risk_never_printed_as_zero():
    assert interval(dict(point=None,ci95=None)) == 'undefined'
    assert interval(dict(point=0,ci95=[0,0])) == '0.0000 [0.0000, 0.0000]'


def test_probability_and_percentage_are_explicit():
    assert interval(dict(point=.02,ci95=[.01,.03]),100) == '2.0000 [1.0000, 3.0000]'
