import copy
import pytest
from scripts import report_m3w_whole_record_policy as api
from tests.test_m3w_whole_record_readout import example


def test_complete_arithmetic_and_empty_matched_utility_only():
    d = example()
    assert api.validate_result(d) == 23
    bad = copy.deepcopy(d)
    bad['comparisons']['exact_gradient_minus_origin']['matched_risk_role'] = 'safe'
    with pytest.raises(ValueError): api.validate_result(bad)
    del d['recordings']['a']
    with pytest.raises(ValueError): api.validate_result(d)


def test_corrupt_full_policy_is_detected():
    d = example(); d['policies']['exact_gradient']['known_easy_positive_risk'] = 1.
    with pytest.raises(ValueError): api.validate_result(d)


def test_diagnostic_mass_units_and_unknown_completion():
    p = dict(selected_unknown_envelope_mass=2.,selected_known_harm_mass=4.,
        selected_known_reference_mass=100.,selected_known_easy_harm_mass=1.,
        selected_known_easy_reference_mass=50.)
    assert api.risk_from_masses(p,2.,10) == [.2,.1]
    with pytest.raises(ValueError): api.risk_from_masses(p,0.,10)
    with pytest.raises(ValueError): api.risk_from_masses(p,2.,0)
