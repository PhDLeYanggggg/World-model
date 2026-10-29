import pytest

from scripts.diagnose_m3w_query_context import components, locality_mean


def test_benefit_harm_and_unknown_decomposition():
    policies = dict(self=dict(selected_known_benefit_mass=20., selected_known_harm_mass=3.),
                    peer=dict(selected_known_benefit_mass=29., selected_known_harm_mass=7.))
    result = dict(full_known_reference_mass=100., policies=policies,
                  matched={'self': policies}, contrasts={'self_full': 3., 'self_matched': 3.},
                  pairs={k: dict(unknown_exchanged_envelope_mass=2.) for k in ('self_full', 'self_matched')})
    for mode in ('full', 'matched'):
        assert components(result, 'self', mode) == dict(known_benefit_delta_pp=9.,
            known_harm_delta_pp=4., unknown_penalty_pp=2., paired_lower_utility_pp=3.)
    result['full_known_reference_mass'] = 0
    with pytest.raises(ValueError):
        components(result, 'self', 'full')


def test_localities_not_repeated_views_get_equal_weight():
    out = locality_mean([('a', {'x': 1.}), ('a', {'x': 3.}), ('b', {'x': 10.})])
    assert out == dict(locality_mean={'x': 6.}, per_locality={'a': {'x': 2.}, 'b': {'x': 10.}})
    with pytest.raises(ValueError):
        locality_mean([])
