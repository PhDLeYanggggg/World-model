import pytest

from reproducibility.songdo_review_supplement import replay_songdo
from src.evaluation.m3w_frozen_score_ranking import (
    RISK_NAMES, calibration_prefix, equal_tie_weights, failure_reason, metrics, stratified_rank,
)

CHECKS = metrics(replay_songdo)


def cell(score, risk=(0., 0., 0.), gains=(.1, .1)):
    return dict(score=score, risk=risk, gains=gains)


def test_ties_share_the_exact_quota_and_keep_the_first_failure_block():
    assert equal_tie_weights([8, 9, 9, 12, 9, 10], 3) == [0., 1/3, 1/3, 1., 1/3, 1.]
    cells = [cell(9.), cell(9., (0., None, 0.)), cell(8.), cell(8., (.04, None, 0.))]
    row = calibration_prefix(cells, CHECKS['risk_pass'], lambda c: failure_reason(c, CHECKS), RISK_NAMES)
    assert row['released'] == 0
    assert row['risk_supported'] == 2
    assert row['first_block'] == {'supported': 1, 'undefined_only': 1}
    assert row['first_block_undefined']['completion_easy_risk_upper'] == 1
    assert failure_reason(cells[-1], CHECKS) == 'undefined_and_violation'


def test_ranked_and_random_select_same_proposal_count_but_not_same_pass_rate():
    cells = [cell(9., (0., None, 0.)), cell(8.), cell(7., gains=(.1, -.1)), cell(6.)]
    x = stratified_rank(cells, dict(dual=CHECKS['dual']))
    assert x == dict(ranked=dict(dual=0.), uniform=dict(dual=.5), eligible=4, quota=1)
    assert stratified_rank([cell(2.)], dict(dual=CHECKS['dual']))['uniform']['dual'] == 1.
    with pytest.raises(ValueError, match='Invalid ranking'):
        equal_tie_weights([1., float('nan')], 1)
