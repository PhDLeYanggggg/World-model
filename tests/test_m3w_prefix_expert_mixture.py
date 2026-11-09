import numpy as np
import pytest
import torch

from src.world_model import m3w_prefix_expert_mixture as m


def test_anchored_experts_follow_linear_motion_and_translate_invariantly():
    h = np.arange(8)[None, :, None] * np.array([2., -3.])[None, None]
    p = m.bank(h)
    np.testing.assert_allclose(p, m.bank(h + 1234.))
    np.testing.assert_allclose(p[:, 0], p[:, 1])
    np.testing.assert_allclose(p[:, 0], p[:, 2])
    np.testing.assert_allclose(p[:, 0], p[:, 4])
    np.testing.assert_allclose(p[0, 1, 0], [2., -3.])


def test_ablation_features_and_prediction_are_past_only():
    h = np.random.default_rng(17).normal(size=(5, 8, 2))
    s, _ = m.features(h, "static")
    a, _ = m.features(h, "kinematic")
    b, _ = m.features(h, "prefix")
    assert not s.any() and not a[:, 15:].any()
    np.testing.assert_array_equal(a[:, :15], b[:, :15])
    assert b[:, 15:].any()
    p, w = m.Mixture(b.shape[1])(torch.from_numpy(b), torch.from_numpy(m.bank(h).astype(np.float32)))
    torch.testing.assert_close(w.sum(1), torch.ones(5))
    assert p.shape == (5, 12, 2)


def test_full_grid_missing_steps_charge_disagreement_even_on_partial_row():
    cv = np.zeros((1, 12, 2))
    p = cv + [1., 0.]
    y = p.copy()
    valid = np.zeros((1, 12), bool)
    valid[:, :5] = True
    y[~valid] = np.nan
    m0 = m.full_masses(cv, p, cv, y, valid, np.array([1.]))
    assert m0 == pytest.approx(dict(reference=5., known_delta=5., unknown_radius=7.))
