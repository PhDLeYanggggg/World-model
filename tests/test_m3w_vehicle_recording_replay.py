import numpy as np
import pytest

from reproducibility.vehicle_recording_replay import verify
from scripts.audit_m3w_vehicle_recording_ranking import compare
from scripts import package_m3w_vehicle_recording_replay as pkg


def test_zero_support_is_not_zero_risk_and_missing_debit_counts_against_both():
    parts = dict(risk=[100., 0., 10., 0., 0., 0., 0.], cv=[1., 0.], ols=[1., 0.],
                 add=[0., 0.], pair=[1., 0., 4])
    assert verify.qualification(parts)[:4] == (False, False, False, True)
    parts['risk'] = [100., 10., 10., 0., 0., .2, 0.]
    assert verify.qualification(parts)[:4] == (True, True, True, False)
    parts['risk'][5] += .01
    assert verify.qualification(parts)[:4] == (False, False, False, False)


def test_calibration_cannot_split_ties_and_top_five_is_tie_averaged():
    scores = [(4., True), (3., True), (3., False), (2., True), (None, True)]
    assert verify.prefix(scores) == (4., 1)
    assert verify.prefix([(1., False), (1., True)]) == (None, 0)
    assert verify.top_mass(scores, 2) == (1.5, 2)


def test_ranking_control_matches_the_number_and_direction_of_proposals():
    def row(score, harm):
        return dict(arms=dict(relative=dict(score=score, risk=[100., 10., 10., harm, 0., 0., 0.],
                    cv=[1., 0.], ols=[1., 0.], add=[0., 0.], pair=[1., 0., 4])))

    stats = compare([row(i, 0. if i < 5 else 3.) for i in range(10)], 'relative')
    assert stats['easy_reference'] == (1., 1.)
    assert stats['risk'] == stats['dual'] == stats['full'] == (0., .5)


def test_pair_components_rebuild_lower_bound_with_missing_labels():
    rng = np.random.default_rng(3)
    cv, p, y = (rng.normal(size=(5, 12, 2)) for _ in range(3))
    mask = rng.random(cv.shape[:2]) > .4
    y[~mask] = np.nan
    expected = pkg.rank.audit.pair_reduction(p[None], cv, y, mask)[0, 1]
    assert verify.relative(pkg.pair_terms(cv, p, y, mask)) == pytest.approx(expected)


def test_risk_components_match_three_original_ratios_with_debit():
    cv = np.zeros((4, 2, 2))
    p, y = cv.copy(), cv.copy()
    p[:2, :, 0] = 1.
    y[0, :, 0] = .9
    y[2:, :, 0] = .5
    mask = np.ones((4, 2), bool)
    mask[1] = False
    rec = np.array(['a']*4)
    m, = pkg.risk_masses(p, cv, y, mask, 1., np.zeros(4, int), np.array([0]))
    orig = pkg.base.policy(p, cv, cv, y, mask, rec, easy_cut=1., hard_cut=2.)['recordings']['a']
    assert (m[3]+m[5])/m[0] == pytest.approx(orig['completion_all_risk_upper'])
    assert (m[4]+m[5])/m[1] == pytest.approx(orig['completion_easy_risk_upper'])
    assert m[6]/m[2] == pytest.approx(orig['known_easy_net_degradation'])
