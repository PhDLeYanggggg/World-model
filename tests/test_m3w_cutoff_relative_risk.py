import numpy as np
import pytest
from src.world_model.m3w_cutoff_relative_risk import features, information_loss_witness
from src.world_model.m3w_dimensionless_risk import scores, preprocess


def fixture():
    rng = np.random.default_rng(52)
    raw = rng.normal(size=(12, 356)).astype(np.float32)
    return raw, np.geomspace(.001, 1000, 12), np.geomspace(.01, 200, 12), 6.25


def test_same_width_and_normalized_context_preserved():
    raw, d, s, c = fixture()
    x = features(raw, d, s, c)
    assert x.shape == raw.shape and x.dtype == np.float32
    np.testing.assert_array_equal(x[:, :354], raw[:, :354])
    np.testing.assert_allclose(x[:, 354], np.log(s/c), rtol=1e-7)
    np.testing.assert_allclose(x[:, 355], np.log1p(d/c), rtol=1e-7)


@pytest.mark.parametrize("factor", [.01, 100.])
def test_unit_conversion_requires_cutoff_conversion(factor):
    raw, d, s, c = fixture()
    x = features(raw, d, s, c)
    changed = raw.copy(); changed[:, -2:] = 999
    np.testing.assert_array_equal(features(changed, d*factor, s*factor, c*factor), x)
    assert not np.array_equal(features(raw, d*factor, s*factor, c), x)


def test_scale_only_context_is_not_enough_for_native_easy_definition():
    w = information_loss_witness()
    assert w["dimensionless_shape_features_identical"]
    assert w["fixed_cutoff_easy_labels"] == [True, False]
    assert w["cutoff_relative_features_distinguish"] and w["unit_relabel_preserves_labels"]


@pytest.mark.parametrize("cutoff", [0., -1., np.nan, np.inf, np.ones(12)])
def test_invalid_cutoff_rejected(cutoff):
    raw, d, s, _ = fixture()
    with pytest.raises(ValueError): features(raw, d, s, cutoff)


def test_preprocess_ignores_unsupported_rows():
    raw, d, s, c = fixture()
    x = features(raw, d, s, c)
    known = np.arange(12) < 8
    parent = dict(known=known, weights=known.astype(float)/8)
    a = preprocess(x, parent)
    x[~known] = 1e20
    b = preprocess(x, parent)
    np.testing.assert_array_equal(a["mean"], b["mean"])
    np.testing.assert_array_equal(a["std"], b["std"])


def test_scores_rescale_but_point_decision_does_not():
    f = np.tile([.4, .1, .02, .03, .2, .5], (12, 1))
    _, d, _, c = fixture()
    h = np.zeros((12, 8, 2)); h[:, -1, 0] = 1
    a = scores(f, d, c, h)
    b = scores(f, d*100, c*100, h)
    for k in ("gain", "risk", "denominator"):
        np.testing.assert_allclose(b[k], a[k]*100)
    np.testing.assert_array_equal(a["point"], b["point"])
