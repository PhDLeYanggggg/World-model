import numpy as np
import pytest
from scripts.diagnose_m3w_role_separated_selection import moment_summary,amplitudes


def test_actual_empty_reference_stays_na_despite_positive_prediction():
    s=moment_summary(np.array([0.,.001,1.,1.,.001]),np.array([0.,1.,0.,0.,1.]))
    assert s['predicted_easy_risk']==.001
    assert s['actual_completion_easy_risk'] is None
    assert s['easy_reference_overprediction_factor'] is None


def test_risk_factorization_does_not_confuse_harm_with_denominator_error():
    p=np.array([1.,.01,1.,.5,.005]);a=np.array([1.,.03,.5,.1,.02])
    s=moment_summary(p,a)
    assert np.isclose(s['actual_completion_easy_risk']/s['predicted_easy_risk'],
                      s['easy_harm_underprediction_factor']*s['easy_reference_overprediction_factor'])
    assert amplitudes(np.array([-1,1,6])).tolist()[0]==0
    with pytest.raises(ValueError):amplitudes(np.array([0]))
