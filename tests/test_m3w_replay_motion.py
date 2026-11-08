import numpy as np
import torch

from src.world_model.m3w_replay_motion import (
    ReplayMotion, arm_features, design, mix, objective,
)


def test_straight_and_still_paths_agree():
    h = np.zeros((2, 8, 2))
    h[1, :, 0] = 2. * np.arange(8)
    ego, replay, delta, scale, paths = design(h)
    np.testing.assert_allclose(replay, 0., atol=1e-6)
    np.testing.assert_allclose(delta, 0., atol=1e-6)
    np.testing.assert_allclose(paths["ols4"], paths["cv"])
    assert np.isfinite(scale).all()
    assert arm_features(ego, replay, "global_mix").shape == (2, 31)


def test_mix_is_translation_invariant_and_bounded():
    h = np.random.default_rng(2).normal(size=(9, 8, 2)).cumsum(1)
    ego, replay, deltas, scale, paths = design(h)
    alt = design(h + [100., -35.])
    for original, shifted in zip((ego, replay, deltas, scale), alt[:4], strict=True):
        np.testing.assert_allclose(original, shifted, atol=1e-6)
    np.testing.assert_allclose(paths["ols4"], alt[-1]["ols4"])
    x = torch.from_numpy(arm_features(ego, replay, "replay"))
    weights = ReplayMotion()(x)
    assert torch.allclose(weights.sum(1), torch.ones(len(h)))
    delta = mix(weights, torch.from_numpy(deltas)).detach().numpy()
    assert (np.linalg.norm(delta, axis=-1) <= .35 * scale[:, None] * np.arange(1, 13) + 1e-5).all()
    np.testing.assert_allclose(arm_features(ego, replay, "history")[:, 16:], 0.)


def test_invalid_targets_are_not_used_in_loss():
    delta = torch.randn(2, 12, 2, requires_grad=True)
    floor = torch.zeros_like(delta)
    valid = torch.zeros(2, 12, dtype=torch.bool)
    target = torch.full_like(delta, torch.nan)
    loss = objective(delta, floor, target, valid, torch.ones(2))
    expected = torch.linalg.vector_norm(delta, dim=-1).mean() / 12.
    assert torch.allclose(loss, expected)
    loss.backward()
    assert torch.isfinite(delta.grad).all()
