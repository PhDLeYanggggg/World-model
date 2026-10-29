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
