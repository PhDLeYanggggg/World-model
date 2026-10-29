import copy

import pytest

from tests.test_m3w_direct_policy_readout import result
from scripts import report_m3w_direct_policy as report


def test_receipt_arithmetic_checks_all_policies_and_empty_controls():
    data = result()
    assert report.validate_result(data) == 25
    data['comparisons']['hard_record_minus_origin']['matched_risk_role'] = 'safety_gate'
    with pytest.raises(ValueError, match='utility only'):
        report.validate_result(data)


def test_changed_mass_and_omitted_recording_fail_verification():
    data = result()
    changed = copy.deepcopy(data)
    changed['policies']['hard_record']['selected_known_harm_mass'] = 10
    with pytest.raises(ValueError):
        report.validate_result(changed)
    del data['recordings']['a']
    with pytest.raises(ValueError):
        report.validate_result(data)


def test_empty_risk_kept_undefined_and_seed_slices_are_not_independent():
    rows = [dict(result=result(), source='one', seed=s, context='ctx', scope='fit')
            for s in (17, 29, 43)]
    out = report.aggregate(rows)
    assert out['views'] == 3 and out['row_occurrences'] == 12
    assert out['safety']['floor']['undefined_easy_views'] == 3
    assert out['safety']['floor']['budget_views'] == 0
    pair = out['comparisons']['hard_record_minus_origin']
    assert pair['empty_matched_views'] == 3
    assert pair['full_lower_pp'] > 0 and pair['matched_lower_pp'] == 0
    assert out['independent_observations'] is False
