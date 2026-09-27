import numpy as np
import pytest
from scripts.verify_m3w_easy_hurdle import check_actions, check_quality
from src.world_model import m3w_easy_hurdle as api


def fixture():
    ids = np.arange(4); eligible = np.ones(4, bool)
    p = np.tile([.5, 1., 0., -.01], (4, 1))
    q = np.tile([-1., -.01], (4, 1)); risks = {a: q for a in ('raw', *api.ARMS)}
    a, _ = api.decisions(ids+1., risks, eligible, ['r']*4, np.ones(4), ids)
    a.update(ids=ids, eligible=eligible, raw_scores=q, marginal_scores=p, supervised_scores=p)
    return a


def test_independent_checker_accepts_and_rejects_tampered_counts():
    a = fixture()
    assert check_actions(a, ['s']*4, ['r']*4, np.ones(4), np.arange(4)+1.) == (6, 0)
    a['supervised_matched'][0] = False
    with pytest.raises(AssertionError):
        check_actions(a, ['s']*4, ['r']*4, np.ones(4), np.arange(4)+1.)


def test_independent_probability_arithmetic():
    y = np.array([[1., 1., .1], [0., 2., .2], [np.nan]*3])
    p = np.array([[.8, 1., .1, .04], [.2, 2., .2, .03], [0., 0., 0., 0.]])
    q = api.quality(p, y, ['r']*3, [0, 0, 1])
    check_quality(q, p, y, ['r']*3, [0, 0, 1])
    q['Brier'] += .01
    with pytest.raises(AssertionError):
        check_quality(q, p, y, ['r']*3, [0, 0, 1])
