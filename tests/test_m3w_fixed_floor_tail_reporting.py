"""Reporting must not hide empty coverage or confuse risk with net gains."""
from scripts.report_m3w_fixed_floor_tail import ci


def test_undefined_risk_is_not_zero():
    assert ci({'point':None,'ci95':None})=='undefined'


def test_percent_scale_preserves_negative_interval():
    assert ci({'point':-.01,'ci95':[-.02,.0]},100)=='-1.0000 [-2.0000, 0.0000]'


def test_missing_interval_not_claimed_as_point_evidence():
    assert ci({'point':.1,'ci95':None})=='undefined'
