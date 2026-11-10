from io import StringIO

import numpy as np
import pytest

from src.data_unification.m3w_swiss_roundabout_prefix import pack, plan, project
from src.data_unification.m3w_swiss_roundabout_support import scan
from src.evaluation.m3w_swiss_roundabout_future_join import join
from src.evaluation.m3w_swiss_roundabout_readout import ARMS, score


def csv_fixture(missing=False):
    out = ['track_id,type,lon,lat,time']
    for i, t in enumerate(np.rint(np.arange(20)*100/30).astype(int)):
        for agent in (1, 2):
            lon = '' if missing and (i, agent) == (9, 2) else 8.1+i*.00001
            out.append(f'{agent},Car,{lon},{47.5+agent*.00001},00:00:00.{t:02}')
    return '\n'.join(out)


def test_target_join_keeps_absent_steps_and_never_bridges_a_segment():
    txt = csv_fixture()
    segments, _ = scan(StringIO(txt))
    pre = pack(StringIO(txt), segments, *plan(segments, 1))
    y, v = join(StringIO(csv_fixture(True)), pre, segments, 1)
    assert v.sum() == 23 and not v[1, 1]
    np.testing.assert_allclose(y[0, 0], project([47.50001, 8.10008])-project([47.50001, 8.10007]),
                               rtol=0., atol=1e-8)
    assert not y[~v].any()
    with pytest.raises(ValueError, match='crosses'):
        join(StringIO(txt), pre, [dict(segments[0], ticks=19)], 1)


def proposals(q, row, n):
    p = dict(query_index=np.array([q]), offsets=np.array([0, n]), radius=np.array([.25]))
    for arm in ARMS:
        p[f'{arm}_rows'] = np.asarray([row])
        p[f'{arm}_paths'] = np.zeros((1, 2, 12, 2))
        p[f'{arm}_paths'][..., 0] = -.25 if arm == 'joint' else .25
    return p


def test_same_budget_preserves_two_agent_fallback_and_uses_original_indices():
    cv = np.zeros((5, 12, 2))
    y = cv+.5
    v = np.ones(y.shape[:2], dtype=bool)
    off = np.array([0, 2, 5])
    p = proposals(1, [2, 4], 3)
    result = score(cv, cv, cv, y, v, off, p, easy_cut=1., hard_cut=2.)
    assert result['matched_target_rows'] == 3 and result['unmatched_queries'] == 1
    assert result['full_roster']['joint']['original_risk']['rows'] == 5
    assert result['full_roster']['joint']['original_risk']['selected'] == 2
    assert result['matched_joint_minus']['independent']['lower_pp'] < 0.
    p['joint_rows'][0, 0] = 0
    with pytest.raises(ValueError, match='row indices'):
        score(cv, cv, cv, y, v, off, p)


def test_match_never_relaxes_the_minimum_three_actor_rule():
    cv = np.zeros((5, 12, 2))
    with pytest.raises(ValueError, match='three actors'):
        score(cv, cv, cv, cv+.5, np.ones(cv.shape[:2], dtype=bool), np.array([0, 2, 5]),
              proposals(0, [0, 1], 2))


def test_empty_matched_control_is_not_zero_effect_or_certified():
    cv = np.zeros((2, 12, 2))
    off = np.array([0, 2])
    r = score(cv, cv, cv, cv+.1, np.ones(cv.shape[:2], dtype=bool), off,
              {'query_index': np.empty(0, dtype=int)}, easy_cut=.2, hard_cut=.5)
    assert r['matched_joint_minus'] is None and not r['deployment_certified']
    assert r['full_roster']['joint']['intervention_rate'] == 0.
    assert not r['full_roster']['joint']['original_risk']['supported']
