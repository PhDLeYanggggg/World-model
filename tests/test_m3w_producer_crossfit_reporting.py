import copy

import pytest

from scripts.report_m3w_moving_producer_crossfit import validate


def objects():
    a = dict(models=6, optimizer_updates=24000, donor_bank_complete=True,
             downstream_risk_lift='not_run', deployment_changed=False)
    fits = [dict(result=dict(step=4000, complete=True), held_sample_draws=0) for _ in range(6)]
    return a, fits, dict(six_predictions_exact=True)


def test_complete_bank_is_not_risk_lift():
    validate(*objects())


@pytest.mark.parametrize('key,value', [('models', 5), ('downstream_risk_lift', 'pass'),
                                      ('deployment_changed', True), ('donor_bank_complete', False)])
def test_reject_partial_or_overclaim(key, value):
    a, f, r = objects()
    a[key] = value
    with pytest.raises(ValueError):
        validate(a, f, r)


def test_reject_teacher_exposure():
    a, f, r = objects()
    f[0]['held_sample_draws'] = 1
    with pytest.raises(ValueError):
        validate(a, f, r)
