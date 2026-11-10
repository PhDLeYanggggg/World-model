import pytest

from scripts.audit_m3w_opendd_exchange_attrition import summarize, trace


PASS = dict(edits=2, undefined=0, risk_pass=1, dual_positive=1)
UNDEFINED = dict(edits=2, undefined=1, risk_pass=0, dual_positive=0)
RISK_FAIL = dict(edits=2, undefined=0, risk_pass=0, dual_positive=0)
UTILITY_FAIL = dict(edits=2, undefined=0, risk_pass=1, dual_positive=0)
PICK = dict(aligned=0, gain_only=1, least_motion=1)


def test_separates_gate_miss_from_rank_miss_and_unavoidable_loss():
    cases = [
        trace(UNDEFINED, [RISK_FAIL, PASS], {}, 'no_predicted_gain'),
        trace(RISK_FAIL, [UTILITY_FAIL, PASS], PICK, None),
        trace(PASS, [RISK_FAIL, UNDEFINED], PICK, None),
        trace(PASS, [], {}, None),
    ]
    rows = [{'counts': c, 'arms': a} for c, a in cases]
    s = summarize({'held': rows})
    assert s['counts']['menu_or_keep_pass'] == 4
    assert s['counts']['same_mask_oracle_pass'] == 2
    assert s['counts']['rescue_outside_mask'] == s['counts']['rescue_no_predicted_gain'] == 1
    assert s['counts']['rescue_inside_mask'] == s['counts']['missed_in_mask_no_dual_utility'] == 1
    assert s['counts']['forced_unavoidable_loss'] == s['counts']['destroy_to_violates_risk'] == 1
    assert s['arms']['aligned']['dual_pass'] == 1
    assert s['arms']['gain_only']['dual_pass'] == 2
    assert s['arms']['random']['dual_pass'] == 1.5


def test_rejects_control_on_an_unmatched_view():
    with pytest.raises(ValueError, match='Inconsistent'):
        trace(UNDEFINED, [PASS], {'aligned': 0}, 'no_predicted_gain')
