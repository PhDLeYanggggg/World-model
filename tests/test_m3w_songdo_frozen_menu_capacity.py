import pytest

from scripts import audit_m3w_songdo_frozen_menu_capacity as audit


def test_hindsight_menu_keeps_both_utility_checks_and_all_risk_checks():
    def proposal(cv, ols, risk=(.02, .02, .02)):
        return dict(risk=risk, gains=(cv, ols))

    view = dict(arms=dict(independent=proposal(.003, .2), additive=proposal(.01, 0.),
                          same_family=proposal(.05, .3, (.01, None, .01)),
                          joint=proposal(.09, .6, (.03, .01, .01))))
    assert audit.best(view, audit.MENUS['all']) == .003
    assert audit.best(view, ('additive', 'joint')) == 0.
    view['arms']['joint']['gains'] = (float('nan'), .5)
    with pytest.raises(ValueError, match='finite or undefined'):
        audit.best(view, ('joint',))


def test_views_flights_and_sources_remain_distinct():
    groups = [[[1, 0, 1], [1, 1, 1]], [[0, 0, 0]]]
    assert audit.select_counts(groups, bool) == dict(
        views=5, flights_any_seed=2, flights_all_seeds=1, source_components=1)
