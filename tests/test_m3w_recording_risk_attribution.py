import pytest

from src.evaluation.m3w_recording_risk_attribution import attribute, summarize


def record(observed, upper, easy, easy_upper, signed, *, supported=False):
    return dict(known_all_risk=observed, completion_all_risk_upper=upper,
                known_easy_risk=easy, completion_easy_risk_upper=easy_upper,
                known_easy_net_degradation=signed, supported=supported,
                selected=2, unknown_selected=1)


def test_recording_patterns_do_not_turn_undefined_into_a_pass():
    unsupported = record(.005, .03, None, None, .0)
    observed = record(.03, .04, .01, .03, .03)
    passes = record(.02, .02, .0, .0, .0, supported=True)
    assert attribute(unsupported) == ('debit_exceeds', 'undefined', 'within')
    assert attribute(observed) == ('observed_exceeds', 'debit_exceeds', 'observed_exceeds')
    summary = summarize([unsupported, observed, passes])
    assert summary['disjoint_recording_patterns'] == {
        'debit_exceeds+undefined': 1, 'observed_exceeds+debit_exceeds': 1, 'pass': 1,
    }
    assert summary['checks']['all_positive_harm']['observed_exceeds'] == 1
    assert summary['switches'] == 6 and summary['unknown_switches'] == 3


def test_original_verdict_and_nondecreasing_upper_bound_are_required():
    with pytest.raises(ValueError, match='original recording verdict'):
        attribute(record(.01, .01, .01, .01, .01))
    with pytest.raises(ValueError, match='Inconsistent'):
        attribute(record(.02, .01, .01, .01, .01))
