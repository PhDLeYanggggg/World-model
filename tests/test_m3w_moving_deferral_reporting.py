import pytest

from scripts.report_m3w_moving_deferral import outcome


def evidence(ci, risk, decision):
    a = dict(nonempty_matched=dict(public=dict(completion_lower_pp=dict(CI95=ci))),
        safety=dict(public_veto=dict(full_original_risk_pass=risk)))
    return a, dict(development_criteria_pass=decision)


def test_missing_primary_cannot_pass_even_with_floor_risk():
    a, d = evidence(None, True, False)
    assert 'not estimable, not passed' in outcome(a, d)
    with pytest.raises(ValueError):
        outcome(a, dict(development_criteria_pass=True))


def test_positive_utility_does_not_erase_risk_failure():
    a, d = evidence([.1, 1.], False, False)
    assert 'requirements fail' in outcome(a, d)


def test_negative_ci_and_positive_result_are_distinct():
    a, d = evidence([-.1, 1.], True, False)
    assert 'no positive' in outcome(a, d)
    a, d = evidence([.1, 1.], True, True)
    assert 'requirements pass' in outcome(a, d)
