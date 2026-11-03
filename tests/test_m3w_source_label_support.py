import numpy as np
import pytest

from src.evaluation.m3w_source_label_support import features, fit, predict, preprocessing, score


def test_schema_only_accepts_past_fields():
    g = np.zeros((8, 476)); d = np.zeros((8, 64))
    assert features(geometry=g, detector_history=d, arm='geometry_detector').shape == (8, 540)
    with pytest.raises(TypeError): features(geometry=g, detector_history=d, arm='geometry', future_valid=np.ones((8, 12)))
    with pytest.raises(ValueError): features(geometry=g[:, :-1], detector_history=d, arm='geometry')


def test_forest_resume_matches_direct_and_replay_is_exact(tmp_path):
    rng = np.random.default_rng(8); x = rng.normal(size=(120, 8)).astype(np.float32)
    y = x[:, 0] > 0; sites = np.array(['a']*60+['b']*60)
    settings = dict(n_estimators=16, max_depth=4, min_samples_leaf=3, max_features=1/3,
                    n_jobs=2, class_weight=None, warm_start=True)
    kw = dict(settings=settings, seed=17, identity={'test':'synthetic'}, clip=8, chunk=4)
    fit(x, y, sites, path=tmp_path/'resume.joblib', stop_at=8, **kw)
    a = fit(x, y, sites, path=tmp_path/'resume.joblib', resume=True, **kw)
    b = fit(x, y, sites, path=tmp_path/'direct.joblib', **kw)
    assert a['tree_fingerprint'] == b['tree_fingerprint']
    np.testing.assert_array_equal(predict(a, x), predict(b, x))
    np.testing.assert_array_equal(predict(a, x), predict(a, x))
    with pytest.raises(ValueError): fit(x, y, sites, path=tmp_path/'resume.joblib', **kw)


def test_one_class_is_not_an_artificial_perfect_score():
    s = score(np.zeros(8, bool), np.zeros(8))
    assert s['AUROC'] is None and s['AUPRC'] is None and s['Brier'] == 0


def test_equal_site_statistics_do_not_use_a_held_site():
    x = np.array([[0., 1.], [2., 3.], [10., 11.]])
    pre, w = preprocessing(x, np.array(['a', 'a', 'b']))
    np.testing.assert_array_equal(pre['mean'], [5.5, 6.5])
    assert w[:2].sum() == w[2]
