import pytest
from scripts.report_m3w_european_selected_risk_learning import summarize, fmt


def test_incomplete_source_study_cannot_be_summarized_as_complete():
    with pytest.raises(ValueError): summarize([], {})


def test_missing_interval_is_not_zero_gain():
    assert fmt(None) == 'undefined'
    assert fmt([-1., 2.]) == '-1.00000 to +2.00000'
