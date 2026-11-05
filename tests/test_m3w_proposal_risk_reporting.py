import pytest

from scripts.report_m3w_moving_proposal_risk import verdict


def test_empty_primary_and_fallback_never_pass():
    a = dict(nonempty_matched={'conditioned': {'completion_lower_pp': {'CI95': None}}},
        safety={'conditioned_veto': {'full_original_risk_pass': False}})
    assert 'not estimable, not passed' in verdict(a, {'development_criteria_pass': False})
    with pytest.raises(ValueError, match='override'):
        verdict(a, {'development_criteria_pass': True})


def test_positive_known_effect_does_not_replace_full_safety():
    a = dict(nonempty_matched={'conditioned': {'completion_lower_pp': {'CI95': [.1, 1]}}},
        safety={'conditioned_veto': {'full_original_risk_pass': False}})
    assert 'contract fails' in verdict(a, {'development_criteria_pass': False})
    a['safety']['conditioned_veto']['full_original_risk_pass'] = True
    assert 'contract passes' in verdict(a, {'development_criteria_pass': True})
