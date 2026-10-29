import copy
import pytest
from tests.test_m3w_cost_sensitive_readout import fixture
from scripts.report_m3w_cost_sensitive_policy import validate_result


def test_mass_validation_and_empty_utility_only_comparison():
    assert validate_result(fixture())==13


def test_empty_numeric_recording_slice_is_rejected():
    d=copy.deepcopy(fixture())
    d['recordings']['1']['origin']['known_rows']=0
    with pytest.raises((ValueError,AssertionError)):validate_result(d)


def test_changed_utility_denominator_is_rejected():
    d=copy.deepcopy(fixture());d['full_known_reference_mass']*=2
    with pytest.raises((ValueError,AssertionError)):validate_result(d)
