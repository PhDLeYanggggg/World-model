import pytest

from scripts.audit_m3w_neural_bank_capacity import combine, ensure_nested, merge_expanded, positive_state


def result(*, witness=None, upper=None, optimal=False, infeasible=False):
    return dict(feasible=witness is not None, positive=witness is not None and witness > 0.,
                witness_mass=witness, upper_mass=upper,
                optimum_proved=optimal, proved_infeasible=infeasible)


def test_timeout_does_not_become_negative_result():
    assert positive_state(result(witness=-1., upper=1.)) is None
    assert positive_state(result(witness=-1., upper=0.)) is False
    assert positive_state(result(witness=.1, upper=1.)) is True
    assert positive_state(result(infeasible=True)) is False


def test_refinement_keeps_checked_witness_and_tightest_upper():
    before, after = result(witness=1., upper=3.), result(witness=.5, upper=2.)
    merged = combine(after, before)
    assert merged['witness_mass'] == 1.
    assert merged['upper_mass'] == 2.
    with pytest.raises(ValueError, match='contradicts'):
        combine(result(infeasible=True), before)


def test_causal_subset_cannot_outperform_upper_bound_of_superset():
    ensure_nested(result(witness=1.), result(witness=2., upper=2.))
    with pytest.raises(ValueError, match='Nested slate'):
        ensure_nested(result(witness=1.), result(upper=.1))


def test_restricted_slate_contributes_only_a_witness():
    refined = result(witness=2., upper=2., optimal=True)
    initial = result(witness=1., upper=3.)
    restricted = result(witness=.5, upper=.5, optimal=True)
    assert merge_expanded(refined, initial, restricted)['upper_mass'] == 2.
