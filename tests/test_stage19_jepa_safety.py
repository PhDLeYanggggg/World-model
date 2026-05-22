from __future__ import annotations

from src.stage19_pipeline import train_wam_jepa


def test_stage19_wam_jepa_is_non_generative_and_no_smc():
    metrics = train_wam_jepa(quick=True)
    assert metrics["uses_next_token_transformer"] is False
    assert metrics["uses_pixel_reconstruction"] is False
    assert metrics["uses_diffusion"] is False
    assert metrics["uses_latent_generative_rollout"] is False
    assert metrics["smc"] is False

