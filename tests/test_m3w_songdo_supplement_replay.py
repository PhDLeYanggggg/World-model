import pytest

from reproducibility.songdo_review_supplement import replay_songdo as replay


def view(score, risk, seed=17):
    return dict(seed=seed, arms={a: dict(score=score, risk=risk, gains=[1., 1.]) for a in replay.ARMS})


def test_calibration_excludes_tie_with_failed_view():
    cases = [[view(1., [.01, .01, -.01])], [view(1., [0., None, 0.])], [view(2., [.02, .02, .02])]]
    thresholds, counts = replay.calibration([cases])
    assert thresholds['17']['joint'] == 2.
    assert counts['17']['joint'] == dict(eligible=3, released=1)
    assert thresholds['29']['joint'] is None


def test_no_release_does_not_count_as_a_risk_pass():
    cases = [view(1., [.02, .02, -.01]), view(2., [0., None, .03])]
    actual = replay.risk_counts(cases, {'17': {'joint': None}}, 'joint')
    assert actual['edited'] == actual['original_risk_pass'] == actual['dual_pass'] == 0
    assert actual['no_edit_not_certified'] == 2
    matched = replay.risk_counts(cases, None, 'joint')
    assert matched['edited'] == 2 and matched['dual_pass'] == 1
    assert matched['undefined_risk'] == matched['measured_risk_violation'] == 1


def test_bootstrap_averages_seeds_then_flights_then_sources():
    groups = [[[1., 2., 3.], [7., 8., 9.]], [[10., 11., 12.]]]
    result = replay.interval(groups, float, [[0, 1], [0, 0], [1, 1]])
    assert result['point'] == 8.
    assert result['ci95'] == pytest.approx([5.15, 10.85])
    assert result['defined_seed_views'] == 9
    groups[1][0][2] = None
    result = replay.interval(groups, lambda x: x, [[0, 1]])
    assert result['point'] is None and result['estimable_components'] == 1
