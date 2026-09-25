import numpy as np
import pytest
from scripts.evaluate_m3w_european_bridge_attribution import seed_summary
from scripts.report_m3w_european_bridge_attribution import summarize


def test_seed_summary_averages_within_locality_not_windows():
    rows = []
    for a in range(3):
        for b in range(3):
            if a == b: continue
            for i, seed in enumerate([17, 29, 43]):
                rows.append(dict(producer=a, controller=b, seed=seed, contrasts={'contrast': {'all': {
                    'by_scene': {'s1': {'gain_percent': 1.+i}, 's2': {'gain_percent': 7.+i}}}}}))
    values = seed_summary(rows, dict(bootstrap_seed=12, bootstrap_resamples=3000))
    assert len(values) == 6
    for v in values.values():
        r = v['contrast__all']
        assert r['gain_percent'] == 5. and r['by_locality'] == {'s1': 2., 's2': 8.}
        assert r['seed_points'] == [4., 5., 6.] and r['independent_localities'] == 2
        assert r['groups_not_independent']


def test_missing_seed_cannot_masquerade_as_three_seed_evidence():
    with pytest.raises(AssertionError): seed_summary([], dict(bootstrap_seed=12, bootstrap_resamples=3000))


def test_aggregate_requires_complete_family():
    with pytest.raises(ValueError): summarize([], {})
