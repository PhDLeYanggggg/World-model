import pytest
from scripts.report_m3w_calibration_cost_attribution import reconcile


def test_native_observed_risks_reconcile_without_changing_zero_support():
    reconcile(dict(observed_all_risk_on_known=.03, observed_easy_risk_on_known=None),
              dict(known_all_risk=.03, known_easy_risk=None))
    with pytest.raises(ValueError, match='Undefined'):
        reconcile(dict(observed_all_risk_on_known=0., observed_easy_risk_on_known=0.),
                  dict(known_all_risk=0., known_easy_risk=None))


def test_mismatched_original_risk_rejected():
    with pytest.raises(ValueError, match='reconcile'):
        reconcile(dict(observed_all_risk_on_known=.03, observed_easy_risk_on_known=.01),
                  dict(known_all_risk=.01, known_easy_risk=.01))
