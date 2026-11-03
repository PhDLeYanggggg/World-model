import numpy as np
import pytest

from scripts import run_m3w_moving_relational_diagnostic as runner


def test_group_statistics_keep_seed_producer_averaging_and_missing_groups():
    rows = [dict(site=site, value=float(i)) for site in ('a', 'b') for i in range(6)]
    cfg = dict(bootstrap_resamples=100, bootstrap_seed=17)
    metric = runner.interval(rows, lambda r:r['value'], ['a', 'b'], cfg)
    assert metric['point'] == 2.5
    assert metric['CI95'] == [2.5, 2.5]
    assert 'secondary' in metric['status']
    rows[0]['value'] = None
    missing = runner.interval(rows, lambda r:r['value'], ['a', 'b'], cfg)
    assert missing['point'] is None
    assert missing['CI95'] is None
    assert missing['undefined_groups'] == ['a']


def test_cached_forecasts_must_cover_exact_producer_excluded_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    path = tmp_path/'prediction.npz'
    identity = dict(fit_groups=['fit'])
    data = dict(sites=np.array(['fit', 'held', 'held']))
    prediction = np.zeros((2, 12, 2), np.float32)
    np.savez(path, ids=np.array([1, 2]), relative_prediction=prediction)
    ref = dict(path=path.name, sha256=runner.sha(path), identity=identity)
    ids, values = runner.cached(ref, data, identity)
    np.testing.assert_array_equal(ids, [1, 2])
    np.testing.assert_array_equal(values, prediction)
    np.savez(path, ids=np.array([1, 1]), relative_prediction=prediction)
    with pytest.raises(ValueError, match='Complete aligned'):
        runner.cached(dict(ref, sha256=runner.sha(path)), data, identity)
