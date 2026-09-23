import numpy as np
import pytest

from scripts.verify_m3w_external_cost_bank import direct_labels, replay_draws, serialized_costs
from src.evaluation.m3w_forecast_cost_bounds import disagreement
from src.evaluation.m3w_native_metrics import native_errors


def test_independent_native_cost_sign_scale_and_missing_support():
    b = np.zeros((3, 12, 2))
    p = np.ones_like(b)
    target = np.stack([p[0], b[1], p[2]])
    valid = np.ones((3, 12), bool)
    valid[-1, -1] = False
    target[-1, -1] = np.nan
    y, d, full = direct_labels(b, p, target, valid, np.array([1., 3., 2.]))
    np.testing.assert_allclose(y[:2], [[2**.5, 0.], [0., 3*2**.5]])
    np.testing.assert_allclose(d, np.array([1, 3, 2])*2**.5)
    assert np.isnan(y[-1]).all() and full.tolist() == [True, True, False]
    y2, _, _ = direct_labels(b, b, target, valid, np.ones(3))
    assert not y2[:2].any()


def test_independent_sampler_only_draws_supported_source_rows():
    sites = np.array(["a", "a", "b", "b", "c", "c"])
    full = np.array([True, False, True, True, False, True])
    counts = replay_draws(sites, full, 17, updates=10, batch_size=32)
    assert counts.sum() == 320 and not counts[~full].any()
    np.testing.assert_array_equal(counts, replay_draws(sites, full, 17, updates=10, batch_size=32))
    with pytest.raises(ValueError):
        replay_draws(sites, np.array([False, False, True, True, True, True]), 17)


def test_serialized_arithmetic_matches_registered_costs_with_noninteger_scale():
    rng = np.random.default_rng(128)
    b, p, target = [rng.normal(size=(64, 12, 2)).astype(np.float32) for _ in range(3)]
    scale = np.exp(rng.normal(size=64))
    valid = rng.random((64, 12)) > .03
    target[~valid] = np.nan
    y, d, full = serialized_costs(b, p, target, valid, scale)
    cv, _ = native_errors(b, target, valid, scale)
    error, _ = native_errors(p, target, valid, scale)
    expected = np.column_stack((np.maximum(cv-error, 0), np.maximum(error-cv, 0)))
    expected[~full] = np.nan
    np.testing.assert_array_equal(y, expected)
    np.testing.assert_array_equal(d, disagreement(p, b, scale).mean(1))
    assert full.any() and (~full).any()
