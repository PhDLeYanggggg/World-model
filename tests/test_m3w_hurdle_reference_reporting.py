from scripts.report_m3w_moving_hurdle_reference import loss_interval


def test_small_negative_loss_and_interval_keep_sign_and_magnitude():
    result = loss_interval(dict(point=-2.210944e-5, CI95=[-4.308389e-5, -3.675668e-6]))
    assert result == '-2.21094e-05 [-4.30839e-05,-3.67567e-06]'
    assert '-0.0000' not in result


def test_missing_interval_is_not_reported_as_safe_zero():
    assert loss_interval(dict(point=0., CI95=None)) == 'not_estimable'
    assert loss_interval(dict(point=None, CI95=[0., 0.])) == 'not_estimable'
