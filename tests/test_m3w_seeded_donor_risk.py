import numpy as np
import pytest

from src.world_model import m3w_seeded_donor_risk as m


def data():
    rng = np.random.default_rng(81)
    return dict(x=rng.normal(size=(30, 120)), target=rng.normal(size=(30, 2)),
                sites=np.repeat(['a', 'b', 'c'], 10), ids=np.arange(30), weights=np.ones(30)/30)


def test_seed17_matches_frozen_engine(tmp_path):
    d = data()
    a = m.original.fit(d, 'a', {}, tmp_path/'old.gz', stop_at=4)
    b = m.fit(d, 'a', {}, tmp_path/'new.gz', seed=17, stop_at=4)
    m.original.exact(a, b)


def test_seed29_resume_exact_and_not_seed17(tmp_path):
    d = data()
    m.fit(d, 'a', {}, tmp_path/'resume.gz', seed=29, stop_at=2)
    a = m.fit(d, 'a', {}, tmp_path/'resume.gz', seed=29, stop_at=4, resume=True)
    b = m.fit(d, 'a', {}, tmp_path/'direct.gz', seed=29, stop_at=4)
    m.original.exact(a, b)
    c = m.fit(d, 'a', {}, tmp_path/'other.gz', seed=17, stop_at=4)
    assert not m.torch.equal(a['model']['net.0.weight'], c['model']['net.0.weight'])


def test_held_labels_do_not_change_fit(tmp_path):
    d = data()
    a = m.fit(d, 'a', {}, tmp_path/'a.gz', seed=43, stop_at=2)
    d['target'][:10] = 1e9
    b = m.fit(d, 'a', {}, tmp_path/'b.gz', seed=43, stop_at=2)
    m.original.exact(a, b)


def test_paired_cluster_interval_not_window_bootstrap():
    b = [np.ones((3, n)) for n in (2, 3, 3)]
    a = [v*.8 for v in b]
    result = m.bootstrap_costs(a, b)
    np.testing.assert_allclose(result['interval95'], [20, 20])
    assert result['valid'] == 3000
    assert not result['independent_confirmation']


def test_reject_flat_window_or_missing_seed_bootstrap():
    with pytest.raises(ValueError):
        m.bootstrap_costs([np.ones((2, 8))]*3, [np.ones((2, 8))]*3)
