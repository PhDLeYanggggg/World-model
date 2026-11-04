import pytest
from scripts.audit_m3w_moving_support_ceiling import support_ceiling


def row(job='a', mass=0., count=0):
    return dict(job=job, recording='r', full_known_easy_reference_mass=mass,
        positive_easy_reference_rows=count, known_easy_rows=count, unknown_rows=20)


def test_unknown_outcomes_cannot_create_observed_reference_support():
    result = support_ceiling([row(), row('b', mass=1., count=1)])
    assert result['absolute_upper_bound_on_supported_views'] == 1
    assert not result['whole_roster_positive_reference_support_possible']
    assert result['zero_mass_views'][0]['unknown_rows'] == 20
    assert result['upper_bound_not_achievable_support_estimate']


def test_positive_support_is_not_claimed_to_prove_safety():
    result = support_ceiling([row(mass=1., count=1)])
    assert result['whole_roster_positive_reference_support_possible']
    assert 'safety_pass' not in result


def test_inconsistent_or_filtered_identity_evidence_is_rejected():
    for rows in ([], [row(), row()], [row(mass=float('nan'))], [row(mass=1.)]):
        with pytest.raises(ValueError):
            support_ceiling(rows)
