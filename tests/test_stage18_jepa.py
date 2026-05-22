from __future__ import annotations

from src.stage14_pipeline import read_json
from src.stage18_pipeline import build_jepa_dataset, train_jepa


def test_stage18_jepa_dataset_has_no_test_endpoint_goal_leakage():
    build_jepa_dataset(quick=True)
    samples = read_json("data/stage18_jepa_dataset/samples.json", [])
    assert samples
    assert all(sample["test_endpoints_used_for_goal_construction"] is False for sample in samples[:50])
    assert all(sample["future_labels_for_evaluation_only"] is True for sample in samples[:50])


def test_stage18_jepa_training_is_non_generative():
    metrics = train_jepa("configs/stage18_jepa_quick.yaml")
    assert metrics["uses_autoregressive_next_token_transformer"] is False
    assert metrics["uses_pixel_reconstruction"] is False
    assert metrics["uses_latent_generative_rollout"] is False
    assert metrics["smc"] is False

