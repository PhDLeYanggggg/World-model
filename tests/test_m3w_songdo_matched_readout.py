import numpy as np
import pytest

from src.evaluation import m3w_songdo_matched_readout as m


def fixture():
    cv = np.zeros((4, 12, 2))
    y = cv.copy()
    y[..., 0] = .25
    valid = np.ones((4, 12), bool)
    valid[3] = False
    y[~valid] = np.nan
    replacement = np.zeros((2, 12, 2))
    replacement[..., 0] = .1
    props = dict(bounds=(0, 4), radius=.1,
                 arms={a: (np.array([0, 1]), replacement.copy()) for a in m.ARMS})
    return cv, y, valid, np.array([0, 4]), props


def test_pair_coverage_is_not_full_roster_coverage():
    cv, y, valid, offsets, p = fixture()
    s = m.score_flight(cv, cv, y, valid, offsets, p, easy_cut=.3, hard_cut=.5)
    assert s['coverage']['complete_pair_windows'] == 1
    assert s['coverage']['complete_roster_windows'] == 0
    assert s['arms']['cv']['complete_roster_fde_native'] is None
    assert s['arms']['cv']['available_fde_native'] == .25
    assert s['arms']['joint']['dual_pass']
    assert s['matched_contrasts']['independent']['lower_pp'] == 0.


def test_missing_step_bound_uses_direct_paired_difference_and_old_risk():
    cv, y, valid, offsets, p = fixture()
    p['arms']['joint'] = (np.array([0, 3]), p['arms']['joint'][1])
    s = m.score_flight(cv, cv, y, valid, offsets, p, easy_cut=.3, hard_cut=.5)
    joint = s['arms']['joint']
    assert joint['risk']['unknown_selected'] == 1
    assert joint['risk']['completion_easy_risk_upper'] == pytest.approx(.4)
    assert not joint['risk']['supported'] and not joint['dual_pass']
    d = s['matched_contrasts']['independent']
    assert d['known_pp'] == pytest.approx(-100*.1/.75)
    assert d['lower_pp'] == pytest.approx(-100*.2/.75)
    assert d['relative_lower_native'] <= d['relative_available_native']


def test_roster_and_radius_cannot_be_changed_by_an_arm():
    cv, y, valid, offsets, p = fixture()
    p['arms']['joint'][1][1, :, 0] = .2
    with pytest.raises(ValueError, match='radius differs'):
        m.score_flight(cv, cv, y, valid, offsets, p, easy_cut=.3, hard_cut=.5)


def test_empty_matched_view_is_not_a_risk_pass():
    cv, y, valid, offsets, _ = fixture()
    s = m.score_flight(cv, cv, y, valid, offsets, None, easy_cut=.3, hard_cut=.5)
    assert not s['matched_eligible']
    assert not s['matched_contrasts']
    assert 'risk' not in s['arms']['cv']


def test_bootstrap_averages_seeds_and_flights_before_groups():
    def row(v):
        return {s: dict(matched_contrasts={'independent': {'lower_pp': x}})
                for s, x in zip(m.SEEDS, v, strict=True)}
    views = {'f1': row([0, 3, 6]), 'f2': row([5, 6, 7]), 'f3': row([12, 12, 12])}
    g = {'f1': 'A+B', 'f2': 'A+B', 'f3': 'C'}
    s = m.component_interval(views, g, 'independent', 'lower_pp')
    assert s['by_component'] == {'A+B': 4.5, 'C': 12}
    assert s['point'] == 8.25
    views['f3'] = row([None, None, None])
    assert m.component_interval(views, g, 'independent', 'lower_pp')['ci95'] is None
