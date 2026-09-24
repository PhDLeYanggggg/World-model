import joblib
import numpy as np
import pytest
from src.world_model.m3w_dimensionless_risk import features, targets, preprocess, scores
from src.training.m3w_fraction_forest import fit, predict


def toy():
    rng = np.random.default_rng(91)
    x = rng.normal(size=(96, 356)).astype(np.float32)
    d = np.linspace(.1, 2, len(x))
    s = np.linspace(.5, 7, len(x))
    x[:, 354], x[:, 355] = np.log(s), np.log1p(d)
    known = np.arange(len(x)) < 80
    parent = dict(weights=known/known.sum(), known=known, training_sites=["source_a", "source_b"])
    costs = np.column_stack((d*.2, np.zeros(len(x))))
    costs[1::2] = np.column_stack((np.zeros(len(x)//2), d[1::2]*.3))
    cv = np.linspace(0, 1, len(x))
    costs[~known], cv[~known] = np.nan, np.nan
    return x, d, s, known, parent, targets(costs, cv, d, known, .5)


@pytest.mark.parametrize("factor", [.01, 100.])
def test_unit_features_and_fraction_targets_covary_consistently(factor):
    x, d, s, k, parent, q = toy()
    changed = x.copy()
    changed[:, 354], changed[:, 355] = np.log(s*factor), np.log1p(d*factor)
    a = features(x, d, s, "dimensionless")
    b = features(changed, d*factor, s*factor, "dimensionless")
    np.testing.assert_array_equal(a, b)
    assert a.shape == (96, 355)
    assert not np.array_equal(x, features(changed, d*factor, s*factor, "native"))
    cost = np.column_stack((d*.2, np.zeros(len(d))))
    cv = d*.5
    np.testing.assert_allclose(targets(cost, cv, d, k, .5), targets(cost*factor, cv*factor, d*factor, k, .5*factor))


def test_source_only_preprocessing_unknown_mutation():
    x, d, s, k, parent, q = toy()
    z = features(x, d, s, "dimensionless")
    pr = preprocess(z, parent)
    z[~k] = 1e8
    other = preprocess(z, parent)
    np.testing.assert_array_equal(pr["mean"], other["mean"])
    np.testing.assert_array_equal(pr["std"], other["std"])
    assert not q[~k].any()


def test_overall_gain_and_signed_easy_risk_are_distinct():
    f = np.array([[.6, .1, .1, 0., .5, .8]])
    h = np.zeros((1, 8, 2)); h[0, -1, 0] = 1
    out = scores(f, np.array([2.]), 1., h)
    assert out["gain"][0] == 1. and out["risk"][0] == .2
    assert not out["point"][0]
    h[:] = 0
    assert not scores(f, np.array([2.]), 1., h)["eligible"].any()


def test_exact_resume_and_unknown_labels_never_sampled(tmp_path):
    x, d, s, k, parent, q = toy()
    z = features(x, d, s, "dimensionless")
    pr = preprocess(z, parent)
    settings = dict(trees=8, max_depth=4, min_samples_leaf=2, max_features=1/3,
                    fit_threads=2, checkpoint_every=2)
    draws = k.astype(np.int64)*3
    args = dict(settings=settings, seed=17, identity={"contract": "synthetic"}, heartbeat=lambda **v: None)
    part = fit(z, q, d, pr, draws, directory=tmp_path/"resumed", stop_at=2, **args)
    assert not part["complete"]
    result = fit(z, q, d, pr, draws, directory=tmp_path/"resumed", resume=True, **args)
    fit(z, q, d, pr, draws, directory=tmp_path/"full", **args)
    a, b = [joblib.load(tmp_path/n/"checkpoint.joblib") for n in ("resumed", "full")]
    np.testing.assert_array_equal(predict(a["model"], z, pr), predict(b["model"], z, pr))
    assert result["unknown_sampled"] == 0 and result["complete"]
    bad = draws.copy(); bad[-1] = 1
    with pytest.raises(ValueError, match="known-source"):
        fit(z, q, d, pr, bad, directory=tmp_path/"bad", **args)
    with pytest.raises(ValueError, match="explicit resume"):
        fit(z, q, d, pr, draws, directory=tmp_path/"full", **args)


@pytest.mark.parametrize("bad", ["width", "negative_d", "zero_scale", "nan"])
def test_invalid_features_rejected(bad):
    x, d, s, *_ = toy()
    if bad == "width": x = x[:, :-1]
    elif bad == "negative_d": d[0] = -1
    elif bad == "zero_scale": s[0] = 0
    else: x[0, 0] = np.nan
    with pytest.raises(ValueError): features(x, d, s, "dimensionless")
