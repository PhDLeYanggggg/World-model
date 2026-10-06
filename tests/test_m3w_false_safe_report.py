import numpy as np
import pytest

from scripts.report_m3w_false_safe_diagnostic import summarize
from src.world_model.m3w_false_safe_diagnostic import diagnose


def rows(empty=False):
    p = np.tile([1, .01, 1, .5, .001], (2, 1))
    if empty: p[:] = 0
    y = np.array([[1, .05, 1, .1, .05], [1, 0, 1, .5, 0]])
    r = diagnose(p, y, np.ones(2, bool), np.ones(2, bool), ['a', 'a'], [1, 1], 1)
    return [dict(arm=arm, identity=dict(source='site0'), result=r) for arm in ('none','rowmean','temporal')]


def test_signed_locality_decomposition_preserved():
    s = summarize(rows())['temporal']
    assert s['easy_risk_violations'] == 1
    v = [x['equal_locality_mean_pp'] for x in s['components'].values()]
    assert sum(v[:3]) == pytest.approx(v[3])
    assert s['positive_recording_excess_occurrences'] == 1


def test_undefined_view_never_dropped_from_aggregate():
    s = summarize(rows()+rows(True))['temporal']
    assert s['undefined_easy_risk_views'] == 1
    assert s['defined_easy_risk_views'] == 1
    assert all(x['equal_locality_mean_pp'] is None for x in s['components'].values())
