import numpy as np

from src.final_model_pipeline import _demo_episode
from src.models.final_world_model import BPSGMAWorldModel


def test_final_world_model_predicts_all_agents():
    episode = _demo_episode()
    model = BPSGMAWorldModel(force_dataset_baseline=True)
    out = model.predict(
        all_agents_past_states=episode["past_states"],
        valid_mask=episode["valid_mask"],
        strongest_causal_baseline_rollout=episode["baseline_rollout"],
        horizons=[10, 25, 50],
        scene_features=episode["scene_features"],
        goal_features=episode["goal_features"],
        metadata=episode["metadata"],
    )
    assert set(out["predictions"].keys()) == {"10", "25", "50"}
    assert out["predictions"]["50"].shape == (3, 2)
    assert np.allclose(out["predictions"]["50"], out["baseline_predictions"]["50"])

